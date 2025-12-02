from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

import cv2
import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = (
    "Locate instances of a given template within a specified image."
)
LONG_DESCRIPTION: str = """
Apply Template Matching to an image. Block is based on OpenCV library function called `cv2.matchTemplate(...)`
that searches for a template image within a larger image. This is often used in computer vision tasks where 
you need to find a specific object or pattern in a scene, like detecting logos, objects, or 
specific regions in an image.

Please take into account the following characteristics of block:
* it tends to produce overlapping and duplicated predictions, hence we added NMS which can be disabled
* block may find very large number of matches in some cases due to simplicity of methods being used - 
in that cases NMS may be computationally intractable and should be disabled

Output from the block is in a form of sv.Detections objects which can be nicely paired with other blocks
accepting this kind of input (like visualization blocks).
"""


class TemplateMatchingManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/template_matching@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Template Matching",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-crosshairs",
                "blockPriority": 0.5,
                "opencv": True,
            },
        }
    )
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    template: Selector(kind=[IMAGE_KIND]) = Field(
        title="Template Image",
        description="The template image for this step.",
        examples=["$inputs.template", "$steps.cropping.template"],
        validation_alias=AliasChoices("template", "templates"),
    )
    matching_threshold: Union[Selector(kind=[FLOAT_KIND]), float] = Field(
        title="Matching Threshold",
        description="The threshold value for template matching.",
        default=0.8,
        examples=[0.8, "$inputs.threshold"],
    )
    apply_nms: Union[Selector(kind=[BOOLEAN_KIND]), bool] = Field(
        title="Apply NMS",
        description="Flag to decide if NMS should be applied at the output detections.",
        default=True,
        examples=["$inputs.apply_nms", False],
    )
    nms_threshold: Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), FloatZeroToOne] = (
        Field(
            title="NMS threshold",
            description="The threshold value NMS procedure (if to be applied).",
            default=0.5,
            examples=["$inputs.nms_threshold", 0.3],
        )
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[OBJECT_DETECTION_PREDICTION_KIND],
            ),
            OutputDefinition(
                name="number_of_matches",
                kind=[INTEGER_KIND],
            ),
        ]


class TemplateMatchingBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[TemplateMatchingManifest]:
        return TemplateMatchingManifest

    def run(
        self,
        image: WorkflowImageData,
        template: WorkflowImageData,
        matching_threshold: float,
        apply_nms: bool,
        nms_threshold: float,
    ) -> BlockResult:
        detections = apply_template_matching(
            image=image,
            template=template.numpy_image,
            matching_threshold=matching_threshold,
            apply_nms=apply_nms,
            nms_threshold=nms_threshold,
        )
        return {"predictions": detections, "number_of_matches": len(detections)}


def apply_template_matching(
    image: WorkflowImageData,
    template: np.ndarray,
    matching_threshold: float,
    apply_nms: bool,
    nms_threshold: float,
) -> sv.Detections:
    img_gray = cv2.cvtColor(image.numpy_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= matching_threshold)

    num_matches = loc[0].size
    if num_matches == 0:
        return sv.Detections.empty()

    # Prepare top-left corners and directly create output arrays for detections
    pts = np.stack((loc[1], loc[0]), axis=1)  # shape (num_matches, 2)
    xyxy = np.empty((num_matches, 4), dtype=np.int32)
    xyxy[:, 0:2] = pts
    xyxy[:, 2] = pts[:, 0] + w
    xyxy[:, 3] = pts[:, 1] + h

    # Preallocate all detection fields efficiently
    confidence = np.full((num_matches,), 1.0, dtype=np.float32)
    class_id = np.zeros((num_matches,), dtype=np.uint32)
    class_name = np.full((num_matches,), "template_match", dtype=object)
    detections_id = np.fromiter(
        (str(uuid4()) for _ in range(num_matches)), dtype=object, count=num_matches
    )

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        data={CLASS_NAME_DATA_FIELD: class_name},
    )
    if apply_nms:
        detections = detections.with_nms(threshold=nms_threshold)
        # NOTE: If NMS reduces the number of detections, need to update all arrays accordingly
        n_final = len(detections)
        # For all arrays below, only use first n_final elements (since the rest are no longer valid)
        # Since sv.Detections manages its data storage, must remap only fields added below
        # But as written, detection_id, etc, should match the indices post-nms.

        # The safer way is to regenerate these after NMS, since we must maintain 1:1 correspondence
        # between auxiliary attributes and the possibly smaller set of detections.
        # But detection indices after NMS might have changed, thus best is to regenerate:
        detections_id = np.fromiter(
            (str(uuid4()) for _ in range(n_final)), dtype=object, count=n_final
        )
        class_name = np.full((n_final,), "template_match", dtype=object)

    # Add additional detection attributes optimized
    detections[PARENT_ID_KEY] = np.full(
        (len(detections),), image.parent_metadata.parent_id, dtype=object
    )
    detections[PREDICTION_TYPE_KEY] = np.full(
        (len(detections),), "object-detection", dtype=object
    )
    detections[DETECTION_ID_KEY] = detections_id[: len(detections)]
    image_height, image_width = image.numpy_image.shape[:2]
    detections[IMAGE_DIMENSIONS_KEY] = np.full(
        (len(detections), 2), [image_height, image_width], dtype=np.int32
    )
    # Update class name if needed (after NMS)
    detections.data[CLASS_NAME_DATA_FIELD] = class_name[: len(detections)]

    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )
