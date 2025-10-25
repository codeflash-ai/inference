from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

from pydantic import ConfigDict, Field, PositiveInt

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INTEGER_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Crop a Region of Interest (RoI) from an image, using absolute coordinates.

This is useful when placed after an ObjectDetection block as part of a multi-stage 
workflow. For example, you could use an ObjectDetection block to detect objects, then 
the AbsoluteStaticCrop block to crop objects, then an OCR block to run character 
recognition on each of the individual cropped regions.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Absolute Static Crop",
            "version": "v1",
            "short_description": "Crop an image using fixed pixel coordinates.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "far fa-crop-alt",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/absolute_static_crop@v1", "AbsoluteStaticCrop"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    x_center: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="Center X of static crop (absolute coordinate)",
        examples=[40, "$inputs.center_x"],
    )
    y_center: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="Center Y of static crop (absolute coordinate)",
        examples=[40, "$inputs.center_y"],
    )
    width: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="Width of static crop (absolute value)",
        examples=[40, "$inputs.width"],
    )
    height: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="Height of static crop (absolute value)",
        examples=[40, "$inputs.height"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="crops", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class AbsoluteStaticCropBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        x_center: int,
        y_center: int,
        width: int,
        height: int,
    ) -> BlockResult:
        # Precompute crop bounds once for the batch if x/y/width/height are the same for all images
        x_min = round(x_center - width / 2)
        y_min = round(y_center - height / 2)
        x_max = (
            x_min + width
        )  # round is not needed here, as width is int, and x_min is already rounded
        y_max = (
            y_min + height
        )  # round is not needed here, as height is int, and y_min is already rounded

        uuid_prefix = f"absolute_static_crop.{uuid4()}"

        result = []
        for idx, image in enumerate(images):
            crop = _take_static_crop_with_bounds(
                image=image,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                crop_identifier=f"{uuid_prefix}.{idx}",
            )
            result.append({"crops": crop})
        return result


def take_static_crop(
    image: WorkflowImageData,
    x_center: int,
    y_center: int,
    width: int,
    height: int,
) -> Optional[WorkflowImageData]:
    x_min = round(x_center - width / 2)
    y_min = round(y_center - height / 2)
    x_max = round(x_min + width)
    y_max = round(y_min + height)
    cropped_image = image.numpy_image[y_min:y_max, x_min:x_max]
    if not cropped_image.size:
        return None
    return WorkflowImageData.create_crop(
        origin_image_data=image,
        crop_identifier=f"absolute_static_crop.{uuid4()}",
        cropped_image=cropped_image,
        offset_x=x_min,
        offset_y=y_min,
        preserve_video_metadata=True,
    )


def _take_static_crop_with_bounds(
    image: WorkflowImageData,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    crop_identifier: str,
) -> Optional[WorkflowImageData]:
    # Fast check for bounds without unnecessary slicing
    img_array = image.numpy_image
    h, w = img_array.shape[:2]
    # Ensure crop bounds are within image boundaries (otherwise returns None)
    # Only crop if the crop region is non-empty and inside the image bounds
    if (
        x_min < 0
        or y_min < 0
        or x_max > w
        or y_max > h
        or x_max <= x_min
        or y_max <= y_min
    ):
        return None

    cropped_image = img_array[y_min:y_max, x_min:x_max]
    if cropped_image.size == 0:
        return None

    return WorkflowImageData.create_crop(
        origin_image_data=image,
        crop_identifier=crop_identifier,
        cropped_image=cropped_image,
        offset_x=x_min,
        offset_y=y_min,
        preserve_video_metadata=True,
    )
