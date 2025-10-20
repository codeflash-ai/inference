import uuid
from copy import deepcopy
from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field, PositiveInt

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Apply a fixed offset to the width and height of a detection.

You can use this block to add padding around the result of a detection. This is useful 
to ensure that you can analyze bounding boxes that may be within the region of an 
object instead of being around an object.
"""

SHORT_DESCRIPTION = "Apply a padding around the width and height of detections."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detection Offset",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "fal fa-distribute-spacing-horizontal",
                "blockPriority": 3,
            },
        }
    )
    type: Literal["roboflow_core/detection_offset@v1", "DetectionOffset"]
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Model predictions to offset dimensions for.",
        examples=["$steps.object_detection_model.predictions"],
    )
    offset_width: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="Offset for box width.",
        examples=[10, "$inputs.offset_x"],
        validation_alias=AliasChoices("offset_width", "offset_x"),
    )
    offset_height: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="Offset for box height.",
        examples=[10, "$inputs.offset_y"],
        validation_alias=AliasChoices("offset_height", "offset_y"),
    )
    units: Literal["Percent (%)", "Pixels"] = Field(
        default="Pixels",
        description="Units for offset dimensions.",
        examples=["Pixels", "Percent (%)"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["predictions"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionOffsetBlockV1(WorkflowBlock):
    # TODO: This block breaks parent coordinates.
    # Issue report: https://github.com/roboflow/inference/issues/380

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions: Batch[sv.Detections],
        offset_width: int,
        offset_height: int,
        units: str = "Pixels",
    ) -> BlockResult:
        use_percentage = units == "Percent (%) - of bounding box width / height"
        return [
            {
                "predictions": offset_detections(
                    detections=detections,
                    offset_width=offset_width,
                    offset_height=offset_height,
                    use_percentage=use_percentage,
                )
            }
            for detections in predictions
        ]


def offset_detections(
    detections: sv.Detections,
    offset_width: int,
    offset_height: int,
    parent_id_key: str = PARENT_ID_KEY,
    detection_id_key: str = DETECTION_ID_KEY,
    use_percentage: bool = False,
) -> sv.Detections:
    if len(detections) == 0:
        return detections
    # Avoid deepcopy, instead reconstruct Detections like original but only copy/modify as needed for speed
    _detections = sv.Detections(
        **detections.data
    )  # Shallow copy of data dict (including xyxy and image_dimensions)
    image_dimensions = _detections.data["image_dimensions"]
    xyxy = _detections.xyxy
    num_boxes = xyxy.shape[0]
    # Vectorized modification of bounding boxes
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]

    if use_percentage:
        # Vectorized percent-based offset
        box_widths = x2 - x1
        box_heights = y2 - y1
        offset_w = (box_widths * offset_width / 200).astype(int)
        offset_h = (box_heights * offset_height / 200).astype(int)
        img_heights = np.array([dim[0] for dim in image_dimensions])
        img_widths = np.array([dim[1] for dim in image_dimensions])

        new_x1 = np.maximum(0, x1 - offset_w)
        new_y1 = np.maximum(0, y1 - offset_h)
        new_x2 = np.minimum(img_widths, x2 + offset_w)
        new_y2 = np.minimum(img_heights, y2 + offset_h)

        _detections.xyxy = np.stack((new_x1, new_y1, new_x2, new_y2), axis=-1)
    else:
        # Vectorized pixel-based offset
        half_w = offset_width // 2
        half_h = offset_height // 2
        img_heights = np.array([dim[0] for dim in image_dimensions])
        img_widths = np.array([dim[1] for dim in image_dimensions])

        new_x1 = np.maximum(0, x1 - half_w)
        new_y1 = np.maximum(0, y1 - half_h)
        new_x2 = np.minimum(img_widths, x2 + half_w)
        new_y2 = np.minimum(img_heights, y2 + half_h)

        _detections.xyxy = np.stack((new_x1, new_y1, new_x2, new_y2), axis=-1)

    # Copy parent id and assign new unique uuid4 ids (faster than a list comprehension where possible)
    _detections[parent_id_key] = detections[detection_id_key].copy()
    # List comprehension for uuid4 is still fastest due to Python overhead
    _detections[detection_id_key] = [str(uuid.uuid4()) for _ in range(len(detections))]
    return _detections
