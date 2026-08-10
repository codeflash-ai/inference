import supervision as sv

_COLOR_ATTRS = {
    attr
    for attr in dir(sv.Color)
    if not attr.startswith("_") and isinstance(getattr(sv.Color, attr), sv.Color)
}

_COLOR_NAME_MAP = {attr.lower(): getattr(sv.Color, attr) for attr in _COLOR_ATTRS}


def str_to_color(color: str) -> sv.Color:
    if color.startswith("#"):
        return sv.Color.from_hex(color)
    elif color.startswith("rgb"):
        # Avoid map(int, ...) for short tuples; use a generator expression with tuple unpack for small benefit
        # Avoid parsing overhead by splitting once and unpacking directly
        values = color[4:-1].split(",")
        if len(values) == 3:
            # Avoid the slight overhead of map for 3 ints
            r = int(values[0])
            g = int(values[1])
            b = int(values[2])
            return sv.Color.from_rgb_tuple((r, g, b))
        # Error case - let it raise naturally as before (ValueError from int conversion or tuple unpack)
        r, g, b = map(int, values)
        return sv.Color.from_rgb_tuple((r, g, b))
    elif color.startswith("bgr"):
        values = color[4:-1].split(",")
        if len(values) == 3:
            b = int(values[0])
            g = int(values[1])
            r = int(values[2])
            return sv.Color.from_bgr_tuple((b, g, r))
        b, g, r = map(int, values)
        return sv.Color.from_bgr_tuple((b, g, r))
    else:
        # Only compute .lower() ONCE and use precomputed mapping for attribute access
        color_key = color.lower()
        if color_key in _COLOR_NAME_MAP:
            return _COLOR_NAME_MAP[color_key]
        raise ValueError(
            f"Invalid text color: {color}; valid formats are #RRGGBB, rgb(R, G, B), bgr(B, G, R), or a valid color name (like WHITE, BLACK, or BLUE)."
        )
