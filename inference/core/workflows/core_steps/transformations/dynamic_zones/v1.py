from typing import List, Literal, Optional, Tuple, Type, Union

import cv2 as cv
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.constants import (
    POLYGON_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "zones"
OUTPUT_KEY_DETECTIONS: str = "predictions"
OUTPUT_KEY_SIMPLIFICATION_CONVERGED: str = "simplification_converged"
TYPE: str = "roboflow_core/dynamic_zone@v1"
SHORT_DESCRIPTION = (
    "Simplify polygons so they are geometrically convex "
    "and contain only the requested amount of vertices."
)
LONG_DESCRIPTION = """
The `DynamicZoneBlock` is a transformer block designed to simplify polygon
so it's geometrically convex and then reduce number of vertices to requested amount.
This block is best suited when Zone needs to be created based on shape of detected object
(i.e. basketball field, road segment, zebra crossing etc.)
Input detections should be filtered and contain only desired classes of interest.
"""


class DynamicZonesManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Dynamic Zone",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-square-dashed",
                "blockPriority": 3,
                "opencv": True,
            },
        }
    )
    type: Literal[f"{TYPE}", "DynamicZone"]
    predictions: Selector(
        kind=[
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="",
        examples=["$segmentation.predictions"],
    )
    required_number_of_vertices: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Keep simplifying polygon until number of vertices matches this number",
        examples=[4, "$inputs.vertices"],
    )
    scale_ratio: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=1,
        description="Expand resulting polygon along imaginary line from centroid to edge by this ratio",
        examples=[1.05, "$inputs.scale_ratio"],
    )
    apply_least_squares: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        default=False,
        description="Apply least squares algorithm to fit resulting polygon edges to base contour",
        examples=[True, "$inputs.apply_least_squares"],
    )
    midpoint_fraction: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=1,
        description="Fraction of vertices to keep in the middle of each edge before fitting least squares line. "
        "This parameter is useful when vertices of convex polygon are not aligned with edge that would be otherwise fitted to points closer to the center of each edge.",
        examples=[0.9, "$inputs.midpoint_fraction"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["predictions"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(
                name=OUTPUT_KEY_DETECTIONS, kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]
            ),
            OutputDefinition(
                name=OUTPUT_KEY_SIMPLIFICATION_CONVERGED, kind=[BOOLEAN_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def calculate_simplified_polygon(
    contours: List[np.ndarray], required_number_of_vertices: int, max_steps: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    # Early exit for clearly insufficient input
    if not contours:
        raise ValueError("No contours provided.")
    # Find the largest contour by length (avoid Python loop, use argmax if input is large enough)
    largest_contour = max(contours, key=len)

    convex_contour = cv.convexHull(
        points=largest_contour,
        returnPoints=True,
        clockwise=True,
    )
    perimeter = cv.arcLength(curve=convex_contour, closed=True)
    lower_epsilon = 1e-7
    upper_epsilon = perimeter

    # Initial binary search setup: preallocation outside the loop
    epsilon = (lower_epsilon + upper_epsilon) / 2
    simplified_polygon = cv.approxPolyDP(
        curve=convex_contour, epsilon=epsilon, closed=True
    )
    n_vertices = len(simplified_polygon)

    last_polygon = None
    last_diff = np.inf
    did_converge = False

    for _ in range(max_steps):
        n_vertices = len(simplified_polygon)
        if n_vertices == required_number_of_vertices:
            did_converge = True
            break
        # Binary search: adjust epsilon up or down
        if n_vertices > required_number_of_vertices:
            lower_epsilon = epsilon
        else:
            upper_epsilon = epsilon
        prev_epsilon = epsilon
        epsilon = (lower_epsilon + upper_epsilon) / 2
        # If epsilon stops changing, break to avoid infinite loop on weird contours
        if epsilon == prev_epsilon:
            break
        simplified_polygon = cv.approxPolyDP(
            curve=convex_contour, epsilon=epsilon, closed=True
        )

    simplified_polygon = _flatten_polygon(simplified_polygon)
    return simplified_polygon, largest_contour


def calculate_least_squares_polygon(
    contour: np.ndarray, polygon: np.ndarray, midpoint_fraction: float = 1
) -> np.ndarray:
    def find_closest_index(point: np.ndarray, contour: np.ndarray) -> int:
        # Use broadcasting for fast vectorized distance computation
        dists = np.linalg.norm(contour - point, axis=1)
        return np.argmin(dists)

    def pick_contour_points_between_vertices(
        point_1: np.ndarray, point_2: np.ndarray, contour: np.ndarray
    ) -> np.ndarray:
        i1 = find_closest_index(point_1, contour)
        i2 = find_closest_index(point_2, contour)
        if i1 <= i2:
            return contour[i1 : i2 + 1]
        else:
            # Avoid np.concatenate overhead if not needed
            return np.vstack((contour[i1:], contour[: i2 + 1]))

    def least_squares_line(points: np.ndarray) -> Optional[Tuple[float, float]]:
        if len(points) < 2:
            return None
        x = points[:, 0]
        y = points[:, 1]
        A = np.empty((len(x), 2), dtype=x.dtype)
        A[:, 0] = x
        A[:, 1] = 1.0
        # lstsq is not the performance bottleneck so keep as is for numerical stability
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return (a, b)

    def intersect_lines(
        line_1: Optional[Tuple[float, float]], line_2: Optional[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        if line_1 is None or line_2 is None:
            return None
        a_1, b_1 = line_1
        a_2, b_2 = line_2
        if np.isclose(a_1, a_2):
            return None
        x = (b_2 - b_1) / (a_1 - a_2)
        y = a_1 * x + b_1
        return np.array([x, y])

    n = len(polygon)
    # Avoid repeated operations in main loop—prepare pairs directly
    pairs = np.empty((n, 2, polygon.shape[1]), dtype=polygon.dtype)
    pairs[0, 0] = polygon[-1]
    pairs[0, 1] = polygon[0]
    if n > 1:
        pairs[1:, 0] = polygon[:-1]
        pairs[1:, 1] = polygon[1:]

    lines = []
    for point_1, point_2 in pairs:
        segment_points = pick_contour_points_between_vertices(point_1, point_2, contour)
        if midpoint_fraction < 1:
            number_of_points = int(round(len(segment_points) * midpoint_fraction))
            if number_of_points > 2:
                number_of_points_to_discard = (
                    len(segment_points) - number_of_points
                ) // 2
                segment_points = segment_points[
                    number_of_points_to_discard : (
                        len(segment_points) - number_of_points_to_discard
                    )
                ]
        line_params = least_squares_line(segment_points)
        lines.append(line_params)

    # Use list comprehensions for looping and intersection, which is a slight overhead savings
    intersections = [
        intersect_lines(lines[i], lines[(i + 1) % len(lines)])
        for i in range(len(lines))
    ]

    return np.array(intersections, dtype=float).round().astype(int)


def scale_polygon(polygon: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1:
        return polygon

    M = cv.moments(polygon)
    if M["m00"] == 0:
        return polygon

    centroid_x = M["m10"] / M["m00"]
    centroid_y = M["m01"] / M["m00"]

    # Use in-place operations for numpy arrays to reduce temporary allocations
    shifted = polygon - np.array([centroid_x, centroid_y])
    scaled = shifted * scale
    result = scaled + np.array([centroid_x, centroid_y])
    return result.round().astype(np.int32)


def _flatten_polygon(polygon: np.ndarray) -> np.ndarray:
    """
    Flattens a polygon from potentially nested arrays to a 2D array.
    """
    while len(polygon.shape) > 2:
        polygon = np.concatenate(polygon)
    return polygon


class DynamicZonesBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DynamicZonesManifest

    def run(
        self,
        predictions: Batch[sv.Detections],
        required_number_of_vertices: int,
        scale_ratio: float = 1,
        apply_least_squares: bool = False,
        midpoint_fraction: float = 1,
    ) -> BlockResult:
        result = []
        append_result = result.append  # minor micro-optimization
        for detections in predictions:
            if detections is None:
                append_result(
                    {
                        OUTPUT_KEY: None,
                        OUTPUT_KEY_DETECTIONS: None,
                        OUTPUT_KEY_SIMPLIFICATION_CONVERGED: False,
                    }
                )
                continue
            if detections.mask is None:
                append_result(
                    {
                        OUTPUT_KEY: [],
                        OUTPUT_KEY_DETECTIONS: None,
                        OUTPUT_KEY_SIMPLIFICATION_CONVERGED: False,
                    }
                )
                continue

            simplified_polygons = []
            updated_detections = []
            all_converged = True
            masks = detections.mask
            det_get = detections.__getitem__
            for i in range(len(masks)):
                mask = masks[i]
                updated_detection = det_get(i)

                contours = sv.mask_to_polygons(mask)
                simplified_polygon, largest_contour = calculate_simplified_polygon(
                    contours=contours,
                    required_number_of_vertices=required_number_of_vertices,
                )
                if apply_least_squares:
                    simplified_polygon = calculate_least_squares_polygon(
                        contour=largest_contour,
                        polygon=simplified_polygon,
                        midpoint_fraction=midpoint_fraction,
                    )
                vertices_count = simplified_polygon.shape[0]
                if vertices_count < required_number_of_vertices:
                    all_converged = False
                    if vertices_count == 0:
                        # If simplified_polygon is empty, pad with zeros
                        simplified_polygon = np.zeros(
                            (required_number_of_vertices, 2), dtype=np.int32
                        )
                    else:
                        npad = required_number_of_vertices - vertices_count
                        # Do this in one step for efficiency, not a per-loop append
                        pad_vals = np.repeat([simplified_polygon[-1]], npad, axis=0)
                        simplified_polygon = np.vstack((simplified_polygon, pad_vals))
                elif vertices_count > required_number_of_vertices:
                    all_converged = False
                    simplified_polygon = simplified_polygon[
                        :required_number_of_vertices
                    ]
                updated_detection[POLYGON_KEY_IN_SV_DETECTIONS] = np.array(
                    [simplified_polygon]
                )
                scaled_polygon = scale_polygon(
                    polygon=simplified_polygon,
                    scale=scale_ratio,
                )
                simplified_polygons.append(scaled_polygon.tolist())
                updated_detection.mask = np.array(
                    [
                        sv.polygon_to_mask(
                            polygon=scaled_polygon,
                            resolution_wh=mask.shape[::-1],
                        )
                    ]
                )
                updated_detections.append(updated_detection)
            append_result(
                {
                    OUTPUT_KEY: simplified_polygons,
                    OUTPUT_KEY_DETECTIONS: sv.Detections.merge(updated_detections),
                    OUTPUT_KEY_SIMPLIFICATION_CONVERGED: all_converged,
                }
            )
        if not result:
            result.append(
                {
                    OUTPUT_KEY: [],
                    OUTPUT_KEY_DETECTIONS: None,
                    OUTPUT_KEY_SIMPLIFICATION_CONVERGED: False,
                }
            )
        return result
