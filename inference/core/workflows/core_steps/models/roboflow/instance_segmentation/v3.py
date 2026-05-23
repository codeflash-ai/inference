import uuid
from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field, PositiveInt, model_validator
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.entities.requests.inference import (
    InstanceSegmentationInferenceRequest,
)
from inference.core.models.inference_models_adapters import (
    InferenceModelsInstanceSegmentationAdapter,
)
from inference_models.models.auto_loaders.entities import PreProcessingOverrides
from inference.core.env import (
    HOSTED_INSTANCE_SEGMENTATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    filter_out_unwanted_classes_from_sv_detections_batch,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INFERENCE_ID_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    FloatZeroToOne,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

LONG_DESCRIPTION = """
Run inference on an instance segmentation model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Instance Segmentation Model",
            "version": "v3",
            "short_description": "Predict the shape, size, and location of objects.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["yolo", "rfdetr", "rf-detr"],
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-chart-network",
                "blockPriority": 1,
                "inference": True,
                "popular": True,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/roboflow_instance_segmentation_model@v3"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    confidence_mode: Union[
        Literal["best", "default", "custom"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="best",
        description="How confidence thresholds are determined.",
        json_schema_extra={
            "always_visible": True,
            "values_metadata": {
                "best": {
                    "name": "Best (Recommended)",
                    "description": "Use F1-optimal thresholds from model evaluation.",
                },
                "default": {
                    "name": "Default",
                    "description": "Use the model's built-in default threshold.",
                },
                "custom": {
                    "name": "Custom",
                    "description": "Specify a custom confidence threshold.",
                },
            },
        },
    )
    custom_confidence: Union[
        Optional[FloatZeroToOne],
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Custom confidence threshold for predictions.",
        examples=[0.3, "$inputs.confidence_threshold"],
        json_schema_extra={
            "relevant_for": {
                "confidence_mode": {"values": ["custom"], "required": True},
            },
        },
    )
    class_filter: Union[Optional[List[str]], Selector(kind=[LIST_OF_VALUES_KIND])] = (
        Field(
            default=None,
            description="List of accepted classes. Classes must exist in the model's training set.",
            examples=[["a", "b", "c"], "$inputs.class_filter"],
        )
    )
    iou_threshold: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.3,
        description="Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).",
        examples=[0.4, "$inputs.iou_threshold"],
    )
    max_detections: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=300,
        description="Maximum number of detections to return.",
        examples=[300, "$inputs.max_detections"],
    )
    class_agnostic_nms: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Boolean flag to specify if NMS is to be used in class-agnostic mode.",
        examples=[True, "$inputs.class_agnostic_nms"],
    )
    max_candidates: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        default=3000,
        description="Maximum number of candidates as NMS input to be taken into account.",
        examples=[3000, "$inputs.max_candidates"],
    )
    mask_decode_mode: Union[
        Literal["accurate", "tradeoff", "fast"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="accurate",
        description="Parameter of mask decoding in prediction post-processing.",
        examples=["accurate", "$inputs.mask_decode_mode"],
    )
    tradeoff_factor: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.0,
        description="Post-processing parameter to dictate tradeoff between fast and accurate.",
        examples=[0.3, "$inputs.tradeoff_factor"],
    )
    disable_active_learning: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Boolean flag to disable project-level active learning for this block.",
        examples=[True, "$inputs.disable_active_learning"],
    )
    active_learning_target_dataset: Union[
        Selector(kind=[ROBOFLOW_PROJECT_KIND]), Optional[str]
    ] = Field(
        default=None,
        description="Target dataset for active learning, if enabled.",
        examples=["my_project", "$inputs.al_target_project"],
    )

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        if self.confidence_mode == "custom" and self.custom_confidence is None:
            raise ValueError(
                "`custom_confidence` is required when `confidence_mode` is 'custom'"
            )
        return self

    @classmethod
    def get_compatible_task_types(cls) -> Optional[List[str]]:
        return ["instance-segmentation"]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=INFERENCE_ID_KEY, kind=[INFERENCE_ID_KIND]),
            OutputDefinition(
                name="predictions",
                kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND],
            ),
            OutputDefinition(name="model_id", kind=[ROBOFLOW_MODEL_ID_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowInstanceSegmentationModelBlockV3(WorkflowBlock):

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
        model_id: str,
        confidence_mode: str,
        custom_confidence: Optional[float],
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        mask_decode_mode: Literal["accurate", "tradeoff", "fast"],
        tradeoff_factor: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        confidence = (
            custom_confidence if confidence_mode == "custom" else confidence_mode
        )
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            return self.run_locally(
                images=images,
                model_id=model_id,
                class_agnostic_nms=class_agnostic_nms,
                class_filter=class_filter,
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                max_candidates=max_candidates,
                mask_decode_mode=mask_decode_mode,
                tradeoff_factor=tradeoff_factor,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        elif self._step_execution_mode is StepExecutionMode.REMOTE:
            return self.run_remotely(
                images=images,
                model_id=model_id,
                class_agnostic_nms=class_agnostic_nms,
                class_filter=class_filter,
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                max_candidates=max_candidates,
                mask_decode_mode=mask_decode_mode,
                tradeoff_factor=tradeoff_factor,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )

    def run_locally(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Union[None, float, Literal["best", "default"]],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        mask_decode_mode: Literal["accurate", "tradeoff", "fast"],
        tradeoff_factor: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        self._model_manager.add_model(
            model_id=model_id,
            api_key=self._api_key,
        )
        if disable_active_learning is True and active_learning_target_dataset is None:
            direct_result = self._try_run_rfdetr_trt_fast_path(
                images=images,
                class_filter=class_filter,
                model_id=model_id,
                confidence=confidence,
                class_agnostic_nms=class_agnostic_nms,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                max_candidates=max_candidates,
                mask_decode_mode=mask_decode_mode,
                tradeoff_factor=tradeoff_factor,
                disable_active_learning=disable_active_learning,
                active_learning_target_dataset=active_learning_target_dataset,
            )
            if direct_result is not None:
                return direct_result
        inference_images = [i.to_inference_format(numpy_preferred=True) for i in images]
        request = InstanceSegmentationInferenceRequest(
            api_key=self._api_key,
            model_id=model_id,
            image=inference_images,
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter,
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            mask_decode_mode=mask_decode_mode,
            tradeoff_factor=tradeoff_factor,
            source="workflow-execution",
        )
        if disable_active_learning is True and active_learning_target_dataset is None:
            direct_result = self._try_run_inference_models_fast_path(
                images=images,
                inference_images=inference_images,
                request=request,
                class_filter=class_filter,
                model_id=model_id,
            )
            if direct_result is not None:
                return direct_result
        predictions = self._model_manager.infer_from_request_sync(
            model_id=model_id, request=request
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        predictions = [
            e.model_dump(by_alias=True, exclude_none=True) for e in predictions
        ]
        return self._post_process_result(
            images=images,
            predictions=predictions,
            class_filter=class_filter,
            model_id=model_id,
        )

    def _try_run_inference_models_fast_path(
        self,
        images: Batch[WorkflowImageData],
        inference_images: List[dict],
        request: InstanceSegmentationInferenceRequest,
        class_filter: Optional[List[str]],
        model_id: str,
    ) -> Optional[BlockResult]:
        model = self._model_manager[model_id]
        if not isinstance(model, InferenceModelsInstanceSegmentationAdapter):
            return None
        inference_kwargs = request.model_dump()
        inference_kwargs.pop("image", None)
        is_rfdetr_trt = (
            model._model.__class__.__name__ == "RFDetrForInstanceSegmentationTRT"
        )
        if is_rfdetr_trt and inference_kwargs.get("response_mask_format") != "rle":
            inference_kwargs["defer_cuda_stream_sync"] = True
        pre_processed_images, preprocessing_metadata = model.preprocess(
            image=inference_images,
            **inference_kwargs,
        )
        predictions = model.predict(pre_processed_images, **inference_kwargs)
        post_process_kwargs = model.map_inference_kwargs(inference_kwargs)
        if is_rfdetr_trt:
            post_process_kwargs["defer_fused_postprocess_count"] = True
        detections = model._model.post_process(
            predictions,
            preprocessing_metadata,
            **post_process_kwargs,
        )
        predictions = self._convert_inference_models_detections_to_sv_detections(
            model=model,
            detections=detections,
            preprocessing_metadata=preprocessing_metadata,
            inference_id=request.id,
        )
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions,
            prediction_type="instance-segmentation",
        )
        predictions = filter_out_unwanted_classes_from_sv_detections_batch(
            predictions=predictions,
            classes_to_accept=class_filter,
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=predictions,
        )
        return [
            {
                "inference_id": request.id,
                "predictions": prediction,
                "model_id": model_id,
            }
            for prediction in predictions
        ]

    def _try_run_rfdetr_trt_fast_path(
        self,
        images: Batch[WorkflowImageData],
        class_filter: Optional[List[str]],
        model_id: str,
        confidence: Union[None, float, Literal["best", "default"]],
        class_agnostic_nms: Optional[bool],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        mask_decode_mode: Literal["accurate", "tradeoff", "fast"],
        tradeoff_factor: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> Optional[BlockResult]:
        model = self._model_manager[model_id]
        if not isinstance(model, InferenceModelsInstanceSegmentationAdapter):
            return None
        if model._model.__class__.__name__ != "RFDetrForInstanceSegmentationTRT":
            return None
        pre_processing_overrides = PreProcessingOverrides(
            disable_contrast_enhancement=False,
            disable_grayscale=False,
            disable_static_crop=False,
        )
        pre_processed_images, preprocessing_metadata = model._model.pre_process(
            images=[image.numpy_image for image in images],
            input_color_format="bgr",
            pre_processing_overrides=pre_processing_overrides,
            defer_cuda_stream_sync=True,
        )
        predictions = model._model.forward(
            pre_processed_images,
            defer_cuda_stream_sync=True,
        )
        detections = model._model.post_process(
            predictions,
            preprocessing_metadata,
            confidence=confidence,
            mask_format="dense",
            defer_cuda_stream_sync=True,
            defer_fused_postprocess_count=True,
        )
        inference_id = str(uuid.uuid4())
        predictions = self._convert_inference_models_detections_to_sv_detections(
            model=model,
            detections=detections,
            preprocessing_metadata=preprocessing_metadata,
            inference_id=inference_id,
        )
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions,
            prediction_type="instance-segmentation",
        )
        predictions = filter_out_unwanted_classes_from_sv_detections_batch(
            predictions=predictions,
            classes_to_accept=class_filter,
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=predictions,
        )
        return [
            {
                "inference_id": inference_id,
                "predictions": prediction,
                "model_id": model_id,
            }
            for prediction in predictions
        ]

    @staticmethod
    def _convert_inference_models_detections_to_sv_detections(
        model: InferenceModelsInstanceSegmentationAdapter,
        detections,
        preprocessing_metadata,
        inference_id: Optional[str],
    ) -> List[sv.Detections]:
        result = []
        for detections_element, metadata in zip(detections, preprocessing_metadata):
            valid_count = None
            if detections_element.image_metadata is not None:
                valid_count = detections_element.image_metadata.get("valid_count")
            if valid_count is not None:
                valid_count = int(valid_count.detach().cpu().item())
                xyxy_tensor = detections_element.xyxy[:valid_count]
                confidence_tensor = detections_element.confidence[:valid_count]
                class_id_tensor = detections_element.class_id[:valid_count]
                mask_tensor = detections_element.mask[:valid_count]
            else:
                xyxy_tensor = detections_element.xyxy
                confidence_tensor = detections_element.confidence
                class_id_tensor = detections_element.class_id
                mask_tensor = detections_element.mask
            xyxy = xyxy_tensor.detach().cpu().numpy()
            confidence = confidence_tensor.detach().cpu().numpy()
            class_id = class_id_tensor.detach().cpu().numpy()
            masks = mask_tensor.detach().cpu().numpy()
            class_names = np.array(
                [
                    (
                        model.class_names[int(class_id_element)]
                        if 0 <= int(class_id_element) < len(model.class_names)
                        else str(int(class_id_element))
                    )
                    for class_id_element in class_id
                ]
            )
            sv_detections = sv.Detections(
                xyxy=xyxy,
                mask=masks,
                confidence=confidence,
                class_id=class_id,
                data={CLASS_NAME_DATA_FIELD: class_names},
            )
            sv_detections[DETECTION_ID_KEY] = np.array(
                [str(uuid.uuid4()) for _ in range(len(sv_detections))]
            )
            sv_detections[PARENT_ID_KEY] = np.array([""] * len(sv_detections))
            sv_detections[IMAGE_DIMENSIONS_KEY] = np.array(
                [[metadata.original_size.height, metadata.original_size.width]]
                * len(sv_detections)
            )
            if inference_id is not None:
                sv_detections[INFERENCE_ID_KEY] = np.array(
                    [inference_id] * len(sv_detections)
                )
            result.append(sv_detections)
        return result

    def run_remotely(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
        confidence: Union[None, float, Literal["best", "default"]],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        max_candidates: Optional[int],
        mask_decode_mode: Literal["accurate", "tradeoff", "fast"],
        tradeoff_factor: Optional[float],
        disable_active_learning: Optional[bool],
        active_learning_target_dataset: Optional[str],
    ) -> BlockResult:
        api_url = (
            LOCAL_INFERENCE_API_URL
            if WORKFLOWS_REMOTE_API_TARGET != "hosted"
            else HOSTED_INSTANCE_SEGMENTATION_URL
        )
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self._api_key,
        )
        if WORKFLOWS_REMOTE_API_TARGET == "hosted":
            client.select_api_v0()
        client_config = InferenceConfiguration(
            disable_active_learning=disable_active_learning,
            active_learning_target_dataset=active_learning_target_dataset,
            class_agnostic_nms=class_agnostic_nms,
            class_filter=class_filter,
            confidence_threshold=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_candidates=max_candidates,
            mask_decode_mode=mask_decode_mode,
            tradeoff_factor=tradeoff_factor,
            max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
            source="workflow-execution",
        )
        client.configure(inference_configuration=client_config)
        inference_images = [i.base64_image for i in images]
        predictions = client.infer(
            inference_input=inference_images,
            model_id=model_id,
        )
        if not isinstance(predictions, list):
            predictions = [predictions]
        return self._post_process_result(
            images=images,
            predictions=predictions,
            class_filter=class_filter,
            model_id=model_id,
        )

    def _post_process_result(
        self,
        images: Batch[WorkflowImageData],
        predictions: List[dict],
        class_filter: Optional[List[str]],
        model_id: str,
    ) -> BlockResult:
        inference_ids = [p.get(INFERENCE_ID_KEY, None) for p in predictions]
        predictions = convert_inference_detections_batch_to_sv_detections(predictions)
        predictions = attach_prediction_type_info_to_sv_detections_batch(
            predictions=predictions,
            prediction_type="instance-segmentation",
        )
        predictions = filter_out_unwanted_classes_from_sv_detections_batch(
            predictions=predictions,
            classes_to_accept=class_filter,
        )
        predictions = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images,
            predictions=predictions,
        )
        return [
            {
                "inference_id": inference_id,
                "predictions": prediction,
                "model_id": model_id,
            }
            for inference_id, prediction in zip(inference_ids, predictions)
        ]
