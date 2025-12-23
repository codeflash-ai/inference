from typing import Any, Dict, List, Literal, Optional, Type
from uuid import uuid4

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "predictions"

SHORT_DESCRIPTION = "Merge multiple detections into a single bounding box."
LONG_DESCRIPTION = """
The `DetectionsMerge` block combines multiple detections into a single bounding box that encompasses all input detections.
This is useful when you want to:
- Merge overlapping or nearby detections of the same object
- Create a single region that contains multiple detected objects
- Simplify multiple detections into one larger detection

The resulting detection will have:
- A bounding box that contains all input detections
- The classname of the merged detection is set to "merged_detection" by default, but can be customized via the `class_name` parameter
- The confidence is set to the lowest confidence among all detections
"""


class DetectionsMergeManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Merge",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "fal fa-object-union",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/detections_merge@v1"]
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Object detection predictions to merge into a single bounding box.",
        examples=["$steps.object_detection_model.predictions"],
    )
    class_name: str = Field(
        default="merged_detection",
        description="The class name to assign to the merged detection.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[OBJECT_DETECTION_PREDICTION_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def calculate_union_bbox(detections: sv.Detections) -> np.ndarray:
    """Calculate a single bounding box that contains all input detections."""
    # Fast exit for no detections
    if len(detections) == 0:
        # Avoid creating object of wrong dtype or shape
        return np.zeros((0, 4), dtype=np.float32)

    # Use view to avoid unneeded copying; rely on numpy memory model
    xyxy = detections.xyxy

    # Manual min/max for Nx4 array using numpy's optimized argmin/argmax
    # Compute min(yield) and max(yield) directly for the leftmost/rightmost coordinates
    # This avoids extra memory usage versus reindexing or multiple array slicing
    x1 = xyxy[:, 0].min()
    y1 = xyxy[:, 1].min()
    x2 = xyxy[:, 2].max()
    y2 = xyxy[:, 3].max()

    # Pre-allocate result for single row
    out = np.empty((1, 4), dtype=xyxy.dtype)
    out[0, 0] = x1
    out[0, 1] = y1
    out[0, 2] = x2
    out[0, 3] = y2
    return out


def get_lowest_confidence_index(detections: sv.Detections) -> int:
    """Get the index of the detection with the lowest confidence."""
    # Fast path if none-confidence, return 0
    conf = detections.confidence
    if conf is None:
        return 0
    # numpy.argmin is already optimal for 1d array
    return int(np.argmin(conf))


class DetectionsMergeBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DetectionsMergeManifest

    def run(
        self,
        predictions: sv.Detections,
        class_name: str = "merged_detection",
    ) -> BlockResult:
        # Fast None or empty path
        if predictions is None or len(predictions) == 0:
            # Use zeroed array, same dtype
            return {OUTPUT_KEY: sv.Detections(xyxy=np.zeros((0, 4), dtype=np.float32))}

        # Calculate the union bounding box
        union_bbox = calculate_union_bbox(predictions)

        # Get the index of the detection with lowest confidence
        lowest_conf_idx = get_lowest_confidence_index(predictions)
        predictions_confidence = predictions.confidence

        # Pre-allocate confidence to avoid slice object overhead
        merged_conf = (
            np.empty(1, dtype=np.float32)
            if predictions_confidence is not None
            else None
        )

        if merged_conf is not None:
            merged_conf[0] = predictions_confidence[lowest_conf_idx]

        # Pre-allocate class_id for fixed value
        merged_class_id = np.zeros(1, dtype=np.int32)

        merged_detection = sv.Detections(
            xyxy=union_bbox,
            confidence=merged_conf,
            class_id=merged_class_id,
            data={
                "class_name": np.array(
                    [class_name]
                ),  # np.array faster than list for known size
                "detection_id": np.array([str(uuid4())]),  # UUID-to-str
            },
        )
        return {OUTPUT_KEY: merged_detection}
