import supervision as sv


def str_to_color(color: str) -> sv.Color:
    if color[:1] == "#":
        return sv.Color.from_hex(color)
    elif color[:3] == "rgb":
        vals = color[4:-1].split(",")
        return sv.Color.from_rgb_tuple((int(vals[0]), int(vals[1]), int(vals[2])))
    elif color[:3] == "bgr":
        vals = color[4:-1].split(",")
        return sv.Color.from_bgr_tuple((int(vals[0]), int(vals[1]), int(vals[2])))
    else:
        attr_name = color.upper()
        if hasattr(sv.Color, attr_name):
            return getattr(sv.Color, attr_name)
        else:
            raise ValueError(
                f"Invalid text color: {color}; valid formats are #RRGGBB, rgb(R, G, B), bgr(B, G, R), or a valid color name (like WHITE, BLACK, or BLUE)."
            )
