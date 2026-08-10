from typing import List, Union

from fastapi.encoders import jsonable_encoder

from inference.core.devices.utils import GLOBAL_INFERENCE_SERVER_ID
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InferenceResponse,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
)
from inference.core.env import TINY_CACHE
from inference.core.logger import logger
from inference.core.version import __version__

RESPONSE_HANDLERS = {
    ClassificationInferenceResponse: lambda r: [
        {"class": pred.class_name, "confidence": pred.confidence}
        for pred in r.predictions
    ],
    MultiLabelClassificationInferenceResponse: lambda r: [
        {"class": cls, "confidence": pred.confidence}
        for cls, pred in r.predictions.items()
    ],
    ObjectDetectionInferenceResponse: lambda r: [
        {"class": pred.class_name, "confidence": pred.confidence}
        for pred in r.predictions
    ],
    InstanceSegmentationInferenceResponse: lambda r: [
        {"class": pred.class_name, "confidence": pred.confidence}
        for pred in r.predictions
    ],
    KeypointsDetectionInferenceResponse: lambda r: [
        {"class": keypoint.class_name, "confidence": keypoint.confidence}
        for pred in r.predictions
        for keypoint in pred.keypoints
    ],
}


def to_cachable_inference_item(
    infer_request: InferenceRequest,
    infer_response: Union[InferenceResponse, List[InferenceResponse]],
) -> dict:
    if not TINY_CACHE:
        return {
            "inference_id": infer_request.id,
            "inference_server_version": __version__,
            "inference_server_id": GLOBAL_INFERENCE_SERVER_ID,
            "request": jsonable_encoder(infer_request),
            "response": jsonable_encoder(infer_response),
        }

    included_request_fields = {
        "api_key",
        "confidence",
        "model_id",
        "model_type",
        "source",
        "source_info",
    }
    request = infer_request.dict(include=included_request_fields)
    response = build_condensed_response(infer_response)
    return {
        "inference_id": infer_request.id,
        "inference_server_version": __version__,
        "inference_server_id": GLOBAL_INFERENCE_SERVER_ID,
        "request": jsonable_encoder(request),
        "response": jsonable_encoder(response),
    }


def build_condensed_response(responses):
    if not isinstance(responses, list):
        responses = [responses]

    # Precompute classes tuple for maximal isinstance efficiency
    handler_classes = tuple(RESPONSE_HANDLERS.keys())

    formatted_responses = []
    for response in responses:
        # Short-circuit for None and empty values in one check
        predictions = getattr(response, "predictions", None)
        if not predictions:
            continue
        try:
            # Fast path: use type as dict key instead of slow isinstance() check
            resp_type = type(response)
            handler = RESPONSE_HANDLERS.get(resp_type)
            if handler is None:
                # Slow path: fallback to isinstance for subclassing edge cases (rare)
                for cls in handler_classes:
                    if isinstance(response, cls):  # covers subclasses
                        handler = RESPONSE_HANDLERS[cls]
                        break
            if handler is not None:
                formatted_responses.append(
                    {
                        "predictions": handler(response),
                        "time": response.time,
                    }
                )
        except Exception as e:
            logger.warning(f"Error formatting response, skipping caching: {e}")

    return formatted_responses


def from_classification_response(response: ClassificationInferenceResponse):
    return [
        {"class": pred.class_name, "confidence": pred.confidence}
        for pred in response.predictions
    ]


def from_multilabel_classification_response(
    response: MultiLabelClassificationInferenceResponse,
):
    return [
        {"class": cls, "confidence": pred.confidence}
        for cls, pred in response.predictions.items()
    ]


def from_object_detection_response(response: ObjectDetectionInferenceResponse):
    return [
        {"class": pred.class_name, "confidence": pred.confidence}
        for pred in response.predictions
    ]


def from_instance_segmentation_response(
    response: InstanceSegmentationInferenceResponse,
):
    return [
        {"class": pred.class_name, "confidence": pred.confidence}
        for pred in response.predictions
    ]


def from_keypoints_detection_response(response: KeypointsDetectionInferenceResponse):
    return [
        {"class": keypoint.class_name, "confidence": keypoint.confidence}
        for pred in response.predictions
        for keypoint in pred.keypoints
    ]
