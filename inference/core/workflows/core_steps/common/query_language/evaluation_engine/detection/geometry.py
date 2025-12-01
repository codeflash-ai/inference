from typing import List, Tuple

import shapely


def is_point_in_zone(
    point: Tuple[int, int],
    zone: List[Tuple[float, float]],
) -> bool:
    # Avoid unnecessary list comprehension by passing zone directly
    polygon = shapely.geometry.Polygon(zone)
    # Use shapely.Point directly with unpacking
    point_obj = shapely.geometry.Point(*point)
    return point_obj.within(polygon)
