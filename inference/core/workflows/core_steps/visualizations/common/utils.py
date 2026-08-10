import supervision as sv
import re
from functools import lru_cache

_rgb_bgr_pattern = re.compile(r"(rgb|bgr)\((\d+),\s*(\d+),\s*(\d+)\)")

_COLOR_ATTRS = {attr for attr in dir(sv.Color) if not attr.startswith("_")}


@lru_cache(maxsize=128)
def str_to_color(color: str) -> sv.Color:
    # Fast path for hex string
    if color.startswith("#"):
        return sv.Color.from_hex(color)
    # Fast regex match for 'rgb(...)' or 'bgr(...)'
    match = _rgb_bgr_pattern.fullmatch(color)
    if match:
        mode, first, second, third = match.group(1, 2, 3, 4)
        a, b, c = int(first), int(second), int(third)
        if mode == "rgb":
            return sv.Color.from_rgb_tuple((a, b, c))
        else:  # bgr
            return sv.Color.from_bgr_tuple((a, b, c))
    # Cached direct color name lookup
    color_upper = color.upper()
    if color_upper in _COLOR_ATTRS:
        return getattr(sv.Color, color_upper)
    # If not recognized, raise ValueError
    raise ValueError(
        f"Invalid text color: {color}; valid formats are #RRGGBB, rgb(R, G, B), bgr(B, G, R), or a valid color name (like WHITE, BLACK, or BLUE)."
    )
