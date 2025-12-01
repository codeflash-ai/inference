from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.env import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    load_core_model,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    FloatZeroToOne,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.http.utils.iterables import make_batches

LONG_DESCRIPTION = """
Run YOLO-World, a zero-shot object detection model, on an image.

YOLO-World accepts one or more text classes you want to identify in an image. The model 
returns the location of objects that meet the specified class, if YOLO-World is able to 
identify objects of that class.

We recommend experimenting with YOLO-World to evaluate the model on your use case 
before using this block in production. For example on how to effectively prompt 
YOLO-World, refer to the [Roboflow YOLO-World prompting 
guide](https://blog.roboflow.com/yolo-world-prompting-tips/).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "YOLO-World Model",
            "version": "v1",
            "short_description": "Run a zero-shot object detection model.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 8,
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/yolo_world_model@v1", "YoloWorldModel", "YoloWorld"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    class_names: Union[Selector(kind=[LIST_OF_VALUES_KIND]), List[str]] = Field(
        description="One or more classes that you want YOLO-World to detect. The model accepts any string as an input, though does best with short descriptions of common objects.",
        examples=[["person", "car", "license plate"], "$inputs.class_names"],
    )
    version: Union[
        Literal[
            "v2-s",
            "v2-m",
            "v2-l",
            "v2-x",
            "s",
            "m",
            "l",
            "x",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="v2-s",
        description="Variant of YoloWorld model",
        examples=["v2-s", "$inputs.variant"],
    )
    confidence: Union[
        Optional[FloatZeroToOne],
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.005,
        description="Confidence threshold for detections",
        examples=[0.005, "$inputs.confidence"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class YoloWorldModelBlockV1(WorkflowBlock):

    def __init__(
        self,
        model_manager: ModelManager,
        api_key: Optional[str],
        step_execution_mode: StepExecutionMode,
    ):
        self._model_manager = model_manager
        self._api_key = api_key
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        class_names: List[str],
        version: str,
        confidence: Optional[float],
    ) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                class_names=class_names,
                version=version,
                confidence=confidence,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                class_names=class_names,
                version=version,
                confidence=confidence,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        class_names: List[str],
        version: str,
        confidence: Optional[float],
    ) -> BlockResult:
        # Optimization: Precompute model version ID and cache ModelManager lookup
        core_model = "yolo_world"
        yolo_world_version_id = version

        # Single model instance loading if all images use the same version id (common case)
        model_manager = self._model_manager

        predictions = []
        # Avoid attribute lookup in loop
        api_key = self._api_key
        for single_image in images:
            image_format = single_image.to_inference_format(numpy_preferred=True)
            inference_request = YOLOWorldInferenceRequest(
                image=image_format,
                yolo_world_version_id=yolo_world_version_id,
                confidence=confidence,
                text=class_names,
                api_key=api_key,
            )
            # call load_core_model on each (if versions ever differ)
            yolo_world_model_id = load_core_model(
                model_manager=model_manager,
                inference_request=inference_request,
                core_model=core_model,
            )
            result = model_manager.infer_from_request_sync(
                yolo_world_model_id, inference_request
            )
            # model_dump is likely a pydantic/v2 data model serialization. It's fast to inline.
            predictions.append(result.model_dump(by_alias=True, exclude_none=True))

        return self._post_process_result(
            images=images,
            predictions=predictions,
        )

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        class_names: List[str],
        version: str,
        confidence: Optional[float],
    ) -> BlockResult:
        # Optimization: Avoid duplicate calls and redundant attribute lookups
        remote_target = WORKFLOWS_REMOTE_API_TARGET
        max_concurrent = WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
        api_url = (
            LOCAL_INFERENCE_API_URL
            if remote_target != "hosted"
            else HOSTED_CORE_MODEL_URL
        )
        api_key = self._api_key

        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=api_key,
        )

        # Only call select_api_v0 if needed, avoiding duplicate call
        if remote_target == "hosted":
            client.select_api_v0()

        configuration = InferenceConfiguration(
            max_concurrent_requests=max_concurrent,
        )
        client.configure(inference_configuration=configuration)

        # Convert images to inference format up-front, minimizing attribute lookups
        inference_images = [i.to_inference_format() for i in images]

        # Optimization: Make batches using a generator to lower memory pressure
        # and avoid list() unless absolutely needed
        # However, we ultimately accumulate all predictions for _post_process, so keep as list for compatibility
        predictions = []
        for sub_batch in make_batches(
            iterable=inference_images,
            batch_size=max_concurrent,
        ):
            batch_input = [img["value"] for img in sub_batch]
            sub_batch_predictions = client.infer_from_yolo_world(
                inference_input=batch_input,
                class_names=class_names,
                model_version=version,
                confidence=confidence,
            )
            predictions.extend(sub_batch_predictions)

        return self._post_process_result(images=images, predictions=predictions)

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
    ) -> BlockResult:
        predictions = convert_inference_detections_batch_to_sv_detections(predictions)
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions,
            prediction_type="object-detection",
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=predictions,
        )
        return [{"predictions": prediction} for prediction in predictions]
