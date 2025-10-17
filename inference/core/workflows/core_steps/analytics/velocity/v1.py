from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from typing_extensions import Literal, Type

from inference.core.workflows.execution_engine.constants import (
    SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS,
    SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS,
    SPEED_KEY_IN_SV_DETECTIONS,
    VELOCITY_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
    StepOutputSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "velocity_detections"
SHORT_DESCRIPTION = "Calculate the velocity and speed of tracked objects with smoothing and unit conversion."
LONG_DESCRIPTION = """
The `VelocityBlock` computes the velocity and speed of objects tracked across video frames.
It includes options to smooth the velocity and speed measurements over time and to convert units from pixels per second to meters per second.
It requires detections from Byte Track with unique `tracker_id` assigned to each object, which persists between frames.
The velocities are calculated based on the displacement of object centers over time.

Note: due to perspective and camera distortions calculated velocity will be different depending on object position in relation to the camera.

"""


class VelocityManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Velocity",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "analytics",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-gauge",
                "blockPriority": 2.5,
            },
        }
    )
    type: Literal["roboflow_core/velocity@v1"]
    image: WorkflowImageSelector
    detections: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Model predictions to calculate the velocity for.",
        examples=["$steps.object_detection_model.predictions"],
    )
    smoothing_alpha: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=0.5,
        description="Smoothing factor (alpha) for exponential moving average (0 < alpha <= 1). Lower alpha means more smoothing.",
        examples=[0.5],
    )
    pixels_per_meter: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        default=1.0,
        description="Conversion from pixels to meters. Velocity will be converted to meters per second using this value.",
        examples=[0.01],  # Example: 1 pixel = 0.01 meters
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class VelocityBlockV1(WorkflowBlock):
    def __init__(self):
        # Store previous positions and timestamps for each tracker_id
        self._previous_positions: Dict[
            str, Dict[Union[int, str], Tuple[np.ndarray, float]]
        ] = {}
        # Store smoothed velocities for each tracker_id
        self._smoothed_velocities: Dict[str, Dict[Union[int, str], np.ndarray]] = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return VelocityManifest

    def run(
        self,
        image: WorkflowImageData,
        detections: sv.Detections,
        smoothing_alpha: float,
        pixels_per_meter: float,
    ) -> BlockResult:
        if detections.tracker_id is None:
            raise ValueError(
                "tracker_id not initialized, VelocityBlock requires detections to be tracked"
            )
        if not (0 < smoothing_alpha <= 1):
            raise ValueError(
                "smoothing_alpha must be between 0 (exclusive) and 1 (inclusive)"
            )
        if not (pixels_per_meter > 0):
            raise ValueError("pixels_per_meter must be greater than 0")

        if image.video_metadata.comes_from_video_file and image.video_metadata.fps != 0:
            ts_current = image.video_metadata.frame_number / image.video_metadata.fps
        else:
            ts_current = image.video_metadata.frame_timestamp.timestamp()

        video_id = image.video_metadata.video_identifier
        previous_positions = self._previous_positions.setdefault(video_id, {})
        smoothed_velocities = self._smoothed_velocities.setdefault(video_id, {})

        num_detections = len(detections)

        # Compute current positions (center of bounding boxes) in a single step
        bbox_xyxy = detections.xyxy  # Shape (num_detections, 4)
        x_centers = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) * 0.5
        y_centers = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) * 0.5
        current_positions = np.stack([x_centers, y_centers], axis=1)

        # Preallocate output arrays
        velocities = np.empty_like(current_positions)
        speeds = np.empty(num_detections)
        smoothed_velocities_arr = np.empty_like(current_positions)
        smoothed_speeds = np.empty(num_detections)
        velocity_zeros = np.zeros(2, dtype=current_positions.dtype)

        tracker_ids = detections.tracker_id
        # Convert tracker_ids using numpy for performance if possible
        try:
            tracker_ids_int = tracker_ids.astype(int, copy=False)
        except Exception:
            # fallback if tracker_ids is not a numpy array
            tracker_ids_int = [int(tracker_id) for tracker_id in tracker_ids]

        for i in range(num_detections):
            tracker_id = tracker_ids_int[i]
            current_position = current_positions[i]

            prev = previous_positions.get(tracker_id)
            if prev is not None:
                prev_position, prev_timestamp = prev
                delta_time = ts_current - prev_timestamp
                if delta_time > 0:
                    displacement = current_position - prev_position
                    velocity = displacement / delta_time
                    speed = np.sqrt(np.dot(velocity, velocity))
                else:
                    velocity = velocity_zeros
                    speed = 0.0
            else:
                velocity = velocity_zeros
                speed = 0.0

            prev_smoothed_velocity = smoothed_velocities.get(tracker_id)
            if prev_smoothed_velocity is not None:
                smoothed_velocity = (
                    smoothing_alpha * velocity
                    + (1 - smoothing_alpha) * prev_smoothed_velocity
                )
            else:
                smoothed_velocity = velocity

            smoothed_speed = np.sqrt(np.dot(smoothed_velocity, smoothed_velocity))

            # Store for next frame
            previous_positions[tracker_id] = (current_position, ts_current)
            smoothed_velocities[tracker_id] = smoothed_velocity

            # Convert to meters per second
            velocities[i] = velocity / pixels_per_meter
            speeds[i] = speed / pixels_per_meter
            smoothed_velocities_arr[i] = smoothed_velocity / pixels_per_meter
            smoothed_speeds[i] = smoothed_speed / pixels_per_meter

        detections.data[VELOCITY_KEY_IN_SV_DETECTIONS] = velocities
        detections.data[SPEED_KEY_IN_SV_DETECTIONS] = speeds
        detections.data[SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS] = (
            smoothed_velocities_arr
        )
        detections.data[SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS] = smoothed_speeds

        return {OUTPUT_KEY: detections}
