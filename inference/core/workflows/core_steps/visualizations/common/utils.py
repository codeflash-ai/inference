import supervision as sv

_COLOR_NAME_MAP = {
    attr: getattr(sv.Color, attr)
    for attr in dir(sv.Color)
    if not attr.startswith("_") and isinstance(getattr(sv.Color, attr), sv.Color)
}

_COLOR_NAME_KEYS = frozenset(_COLOR_NAME_MAP.keys())


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
        upper_color = color.upper()
        if upper_color in _COLOR_NAME_KEYS:
            return _COLOR_NAME_MAP[upper_color]
        else:
            raise ValueError(
                f"Invalid text color: {color}; valid formats are #RRGGBB, rgb(R, G, B), bgr(B, G, R), or a valid color name (like WHITE, BLACK, or BLUE)."
            )
