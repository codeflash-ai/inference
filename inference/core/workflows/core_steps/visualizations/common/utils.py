import supervision as sv


def str_to_color(color: str) -> sv.Color:
    if color.startswith("#"):
        return sv.Color.from_hex(color)
    elif color.startswith("rgb"):
        r, g, b = (int(x) for x in color[4:-1].split(","))
        return sv.Color.from_rgb_tuple((r, g, b))
    elif color.startswith("bgr"):
        b, g, r = (int(x) for x in color[4:-1].split(","))
        return sv.Color.from_bgr_tuple((b, g, r))
    else:
        color_upper = color.upper()
        if hasattr(sv.Color, color_upper):
            return getattr(sv.Color, color_upper)
        else:
            raise ValueError(
                f"Invalid text color: {color}; valid formats are #RRGGBB, rgb(R, G, B), bgr(B, G, R), or a valid color name (like WHITE, BLACK, or BLUE)."
            )
