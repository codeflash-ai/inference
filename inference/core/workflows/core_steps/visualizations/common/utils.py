import supervision as sv

_color_names = set(name for name in dir(sv.Color) if name.isupper())

_named_color_cache = {}


def str_to_color(color: str) -> sv.Color:
    if color.startswith("#"):
        return sv.Color.from_hex(color)
    elif color.startswith("rgb"):
        r, g, b = map(int, color[4:-1].split(","))
        return sv.Color.from_rgb_tuple((r, g, b))
    elif color.startswith("bgr"):
        b, g, r = map(int, color[4:-1].split(","))
        return sv.Color.from_bgr_tuple((b, g, r))
    else:
        color_upper = color.upper()
        if color_upper in _color_names:
            if color_upper not in _named_color_cache:
                _named_color_cache[color_upper] = getattr(sv.Color, color_upper)
            return _named_color_cache[color_upper]
        else:
            raise ValueError(
                f"Invalid text color: {color}; valid formats are #RRGGBB, rgb(R, G, B), bgr(B, G, R), or a valid color name (like WHITE, BLACK, or BLUE)."
            )
