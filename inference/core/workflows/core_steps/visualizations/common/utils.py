import supervision as sv


def str_to_color(color: str) -> sv.Color:
    if color.startswith("#"):
        return sv.Color.from_hex(color)
    elif color.startswith("rgb"):
        # Slice efficiently, avoid intermediate string manipulation.
        values = color[4:-1].split(",")
        r, g, b = map(int, values)
        return sv.Color.from_rgb_tuple((r, g, b))
    elif color.startswith("bgr"):
        values = color[4:-1].split(",")
        b, g, r = map(int, values)
        return sv.Color.from_bgr_tuple((b, g, r))
    else:
        color_upper = color.upper()
        color_attr = getattr(sv.Color, color_upper, None)
        if color_attr is not None:
            return color_attr
        raise ValueError(
            f"Invalid text color: {color}; valid formats are #RRGGBB, rgb(R, G, B), bgr(B, G, R), or a valid color name (like WHITE, BLACK, or BLUE)."
        )
