from typing import List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION: str = "Calculate a score to indicate how well-focused a camera is."
LONG_DESCRIPTION: str = """
This block calculate the Brenner function score which is a measure of the texture in the image. 
An in-focus image has a high Brenner function score, and contains texture at a smaller scale than
 an out-of-focus image. Conversely, an out-of-focus image has a low Brenner function score, and 
 does not contain small-scale texture.
"""


class CameraFocusManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/camera_focus@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Camera Focus",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-aperture",
                "blockPriority": 8,
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

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[IMAGE_KIND],
            ),
            OutputDefinition(
                name="focus_measure",
                kind=[
                    FLOAT_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class CameraFocusBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_manifest(cls) -> Type[CameraFocusManifest]:
        return CameraFocusManifest

    def run(self, image: WorkflowImageData, *args, **kwargs) -> BlockResult:
        # Calculate the Brenner measure
        brenner_image, brenner_value = calculate_brenner_measure(image.numpy_image)
        output = WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=brenner_image,
        )
        return {
            OUTPUT_IMAGE_KEY: output,
            "focus_measure": brenner_value,
        }


def calculate_brenner_measure(
    input_image: np.ndarray,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    text_thickness: int = 2,
) -> Tuple[np.ndarray, float]:
    """
    Brenner's focus measure.

    Parameters
    ----------
    input_image : np.ndarray
        The input image in grayscale.
    text_color : Tuple[int, int, int], optional
        The color of the text displaying the Brenner value, in BGR format. Default is white (255, 255, 255).
    text_thickness : int, optional
        The thickness of the text displaying the Brenner value. Default is 2.

    Returns
    -------
    Tuple[np.ndarray, float]
        The Brenner image and the Brenner value.
    """
    # Convert image to grayscale if it has 3 channels
    if len(input_image.shape) == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Convert image to 16-bit integer format
    converted_image = input_image.astype(np.int16)

    # Directly calculate horizontal and vertical focus measures using slicing for max efficiency
    h_diff = np.clip(converted_image[:, 2:] - converted_image[:, :-2], 0, None)
    v_diff = np.clip(converted_image[2:, :] - converted_image[:-2, :], 0, None)

    # Prepare an output focus_measure of correct shape; fill using maximums from diffs
    focus_measure = np.zeros_like(converted_image, dtype=np.int32)
    # For the overlapping regions, use the max of h_diff and v_diff
    h_area = (slice(None), slice(0, converted_image.shape[1] - 2))
    v_area = (slice(0, converted_image.shape[0] - 2), slice(None))

    # Horizontal diff into proper location (saves memory and copying)
    focus_measure[h_area] = h_diff
    # Vertical diff, element-wise max
    focus_measure[v_area] = np.maximum(focus_measure[v_area], v_diff)

    # Square in-place for final focus measure
    np.square(focus_measure, out=focus_measure)

    # Convert focus measure matrix to 8-bit for visualization, avoid division by zero
    max_val = focus_measure.max()
    if max_val == 0:
        focus_measure_image = np.zeros_like(focus_measure, dtype=np.uint8)
    else:
        focus_measure_image = (focus_measure / max_val * 255).astype(np.uint8)

    # Display the Brenner value on the top left of the image
    cv2.putText(
        focus_measure_image,
        f"Focus value: {focus_measure.mean():.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        text_color,
        text_thickness,
    )

    return focus_measure_image, focus_measure.mean()
