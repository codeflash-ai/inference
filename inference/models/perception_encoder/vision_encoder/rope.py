from math import log, pi
from typing import Literal, Optional, Union

import torch
from einops import rearrange, repeat
from torch import Tensor, broadcast_tensors, einsum, nn
from torch.amp import autocast
from torch.nn import Module, ModuleList

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# broadcat, as tortoise-tts was using it


def broadcat(tensors, dim=-1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


# rotary embedding helper functions


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    # optimize split with memory efficient slicing (avoid triple temporary tensors if possible)
    if start_index == 0 and end_index == t.shape[-1]:
        t_rot = t
        t_left = t_right = None
    else:
        t_left = t[..., :start_index] if start_index > 0 else None
        t_rot = t[..., start_index:end_index]
        t_right = t[..., end_index:] if end_index < t.shape[-1] else None

    t_rot = (t_rot * freqs.cos() * scale) + (rotate_half(t_rot) * freqs.sin() * scale)

    if t_left is None and t_right is None:
        out = t_rot
    elif t_left is None:
        out = torch.cat((t_rot, t_right), dim=-1)
    elif t_right is None:
        out = torch.cat((t_left, t_rot), dim=-1)
    else:
        out = torch.cat((t_left, t_rot, t_right), dim=-1)

    return out.type(dtype)


# learned rotation helpers


def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = einsum("..., f -> ... f", rotations, freq_ranges)
        rotations = rearrange(rotations, "... r f -> ... (r f)")

    rotations = repeat(rotations, "... n -> ... (n r)", r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


# classes


class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Optional[Tensor] = None,
        freqs_for: Union[
            Literal["lang"], Literal["pixel"], Literal["constant"]
        ] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=True,
    ):
        super().__init__()

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            arange = torch.arange(0, dim, 2)[: (dim // 2)]
            freqs = 1.0 / (theta ** (arange.float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            # torch.ones allocates contiguous, use float directly
            freqs = torch.ones(num_freqs, dtype=torch.float32)

        self.cache_if_possible = cache_if_possible

        self.tmp_store("cached_freqs", None)
        self.tmp_store("cached_scales", None)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.learned_freq = learned_freq

        # dummy for device
        self.tmp_store("dummy", torch.tensor(0))

        # default sequence dimension
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors
        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # xpos
        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = xpos_scale_base
        self.tmp_store("scale", scale)

        # add apply_rotary_emb as static method
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        # All compute in float32 from the start for precision, .arange with dtype
        return (
            torch.arange(seq_len, device=device, dtype=dtype) + offset
        ) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert (
            not self.use_xpos
        ), "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        freqs = self.forward(
            self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset),
            seq_len=seq_len,
            offset=offset,
        )

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")

        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        rotated_q = self.rotate_queries_or_keys(
            q, seq_dim=seq_dim, offset=k_len - q_len + offset
        )
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim, offset=offset)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)

        if rotated_q.dtype != q.dtype:
            rotated_q = rotated_q.type(q.dtype)
        if rotated_k.dtype != k.dtype:
            rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(self, t: Tensor, seq_len: Optional[int] = None, offset=0):
        assert self.use_xpos

        should_cache = self.cache_if_possible and exists(seq_len)

        if (
            should_cache
            and exists(self.cached_scales)
            and (seq_len + offset) <= self.cached_scales.shape[0]
        ):
            return self.cached_scales[offset : (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - t.shape[0] // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = torch.cat((scale, scale), dim=-1)

        if should_cache:
            self.tmp_store("cached_scales", scale)

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        should_cache = (
            self.cache_if_possible
            and not self.learned_freq
            and exists(seq_len)
            and self.freqs_for != "pixel"
        )

        if (
            should_cache
            and exists(self.cached_freqs)
            and (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            # .detach() is needed for safety, but not if already detached
            cached = self.cached_freqs[offset : (offset + seq_len)]
            if cached.requires_grad:
                return cached.detach()
            return cached

        freqs = self.freqs

        # Avoid excess casting: directly use input dtype if matching freqs.dtype
        t_cast = t if t.dtype == freqs.dtype else t.type(freqs.dtype)
        freqs_e = einsum("..., f -> ... f", t_cast, freqs)
        freqs_e = repeat(freqs_e, "... n -> ... (n r)", r=2)

        if should_cache:
            self.tmp_store("cached_freqs", freqs_e.detach())

        return freqs_e


class Rope2D:
    """Helper class to apply RoPE2D as well as interpolate on the fly."""

    def __init__(self, dim, use_cls_token=False):
        self.dim = dim
        self.use_cls_token = use_cls_token
        self.grid_size = None
        self.freq = None

    def init_tensors(self):
        self.rope = RotaryEmbedding(self.dim // 2)

    def update_grid(self, device, grid_h, grid_w):
        if self.grid_size != (grid_h, grid_w):
            self.grid_size = (grid_h, grid_w)

            self.rope = self.rope.to(device)

            if self.use_cls_token:
                # +1 to leave space for the cls token to be (0, 0)
                grid_y_range = torch.arange(grid_h, device=device) + 1
                grid_x_range = torch.arange(grid_w, device=device) + 1
            else:
                grid_y_range = torch.arange(grid_h, device=device)
                grid_x_range = torch.arange(grid_w, device=device)

            freqs_y = self.rope(grid_y_range)[:, None].expand(grid_h, grid_w, -1)
            freqs_x = self.rope(grid_x_range)[None, :].expand(grid_h, grid_w, -1)
            freq = torch.cat([freqs_x, freqs_y], dim=-1).reshape(grid_h * grid_w, -1)

            if self.use_cls_token:
                freq = torch.cat(
                    [torch.zeros(1, freq.shape[-1], device=device), freq], dim=0
                )

            self.freq = freq[None, ...]

        self.freq = self.freq.to(device)

    def __call__(self, q, k):
        # batch, heads, seq, dim = q.shape
        q = apply_rotary_emb(self.freq[:, None, :, :], q)
        k = apply_rotary_emb(self.freq[:, None, :, :], k)

        return q, k
