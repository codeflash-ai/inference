import sys
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Combine results of detection model with classification results performed separately for 
each and every bounding box. 

Bounding boxes without top class predicted by classification model are discarded, 
for multi-label classification results, most confident label is taken as bounding box
class.  
"""

SHORT_DESCRIPTION = "Replace classes of detections with classes predicted by a chained classification model."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Classes Replacement",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-arrow-right-arrow-left",
                "blockPriority": 5,
            },
        }
    )
    type: Literal[
        "roboflow_core/detections_classes_replacement@v1",
        "DetectionsClassesReplacement",
    ]
    object_detection_predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Regions of Interest",
        description="The output of a detection model describing the bounding boxes that will have classes replaced.",
        examples=["$steps.my_object_detection_model.predictions"],
    )
    classification_predictions: Selector(kind=[CLASSIFICATION_PREDICTION_KIND]) = Field(
        title="Classification results for crops",
        description="The output of classification model for crops taken based on RoIs pointed as the other parameter",
        examples=["$steps.my_classification_model.predictions"],
    )
    fallback_class_name: Union[Optional[str], Selector(kind=[STRING_KIND])] = Field(
        default=None,
        title="Fallback class name",
        description="The class name to be used as a fallback if no class is predicted for a bounding box",
        examples=["unknown"],
    )
    fallback_class_id: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(
        default=None,
        title="Fallback class id",
        description="The class id to be used as a fallback if no class is predicted for a bounding box;"
        f"if not specified or negative, the class id will be set to {sys.maxsize}",
        examples=[77],
    )

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return True

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {
            "object_detection_predictions": 0,
            "classification_predictions": 1,
        }

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "object_detection_predictions"

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
            )
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionsClassesReplacementBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        object_detection_predictions: Optional[sv.Detections],
        classification_predictions: Optional[Batch[Optional[dict]]],
        fallback_class_name: Optional[str],
        fallback_class_id: Optional[int],
    ) -> BlockResult:
        if object_detection_predictions is None:
            return {"predictions": None}
        if not classification_predictions:
            return {"predictions": sv.Detections.empty()}
        # Check directly for useful predictions, short-circuit if all unsuitable
        cls_preds = classification_predictions
        all_invalid = True
        for p in cls_preds:
            if (
                p is not None
                and ("predictions" in p)
                and (("top" not in p) or p["top"])
            ):
                all_invalid = False
                break
        if all_invalid:
            return {"predictions": sv.Detections.empty()}

        # Use zip-comprehension and batch-extract for improved performance
        # Build detection_id_by_class dict and filter mask in one pass
        detection_id_by_class: Dict[str, Optional[Tuple[str, int, float]]] = {}
        for prediction in cls_preds:
            if prediction is not None:
                pid = prediction[PARENT_ID_KEY]
                cls_tpl = extract_leading_class_from_prediction(
                    prediction=prediction,
                    fallback_class_name=fallback_class_name,
                    fallback_class_id=fallback_class_id,
                )
                detection_id_by_class[pid] = cls_tpl

        od_ids = object_detection_predictions.data[DETECTION_ID_KEY]
        detections_to_remain_mask = np.fromiter(
            (detection_id_by_class.get(did) is not None for did in od_ids),
            dtype=bool,
            count=len(od_ids),
        )
        selected_object_detection_predictions = object_detection_predictions[
            detections_to_remain_mask
        ]

        sel_ids = selected_object_detection_predictions.data[DETECTION_ID_KEY]
        # Collect replacement values as numpy arrays efficiently
        class_name_list = []
        class_id_list = []
        confidence_list = []
        for detection_id in sel_ids:
            name, cid, conf = detection_id_by_class[detection_id]
            class_name_list.append(name)
            class_id_list.append(cid)
            confidence_list.append(conf)
        replaced_class_names = np.array(class_name_list)
        replaced_class_ids = np.array(class_id_list)
        replaced_confidences = np.array(confidence_list)

        # Assign results in-place
        selected_object_detection_predictions.class_id = replaced_class_ids
        selected_object_detection_predictions.confidence = replaced_confidences
        selected_object_detection_predictions.data[CLASS_NAME_DATA_FIELD] = (
            replaced_class_names
        )
        # Efficiently generate new uuids as strings
        selected_object_detection_predictions.data[DETECTION_ID_KEY] = np.array(
            [str(uuid4()) for _ in range(len(selected_object_detection_predictions))]
        )
        return {"predictions": selected_object_detection_predictions}


def extract_leading_class_from_prediction(
    prediction: dict,
    fallback_class_name: Optional[str] = None,
    fallback_class_id: Optional[int] = None,
) -> Optional[Tuple[str, int, float]]:
    if "top" in prediction:
        if not prediction.get("predictions") and not fallback_class_name:
            return None
        elif not prediction.get("predictions") and fallback_class_name:
            try:
                fallback_class_id = int(fallback_class_id)
            except ValueError:
                fallback_class_id = None
            if fallback_class_id is None or fallback_class_id < 0:
                fallback_class_id = sys.maxsize
            return fallback_class_name, fallback_class_id, 0
        class_name = prediction["top"]
        # Convert to dict lookup for speed if input is large,
        # but for a short list the following list-comprehension is still efficient.
        preds = prediction["predictions"]
        found_class_id = None
        found_conf = None
        for p in preds:
            if p["class"] == class_name:
                if found_class_id is not None:
                    raise ValueError(
                        f"Could not resolve class id for prediction: {prediction}"
                    )
                found_class_id = p["class_id"]
                found_conf = p["confidence"]
        if found_class_id is None:
            raise ValueError(f"Could not resolve class id for prediction: {prediction}")
        return class_name, found_class_id, found_conf
    predicted_classes = prediction.get("predicted_classes", [])
    if not predicted_classes:
        return None
    # Find max confidence using a single loop
    max_confidence = -1
    max_confidence_class_name = None
    max_confidence_class_id = None
    for class_name, prediction_details in prediction["predictions"].items():
        current_class_confidence = prediction_details["confidence"]
        current_class_id = prediction_details["class_id"]
        if current_class_confidence > max_confidence:
            max_confidence = current_class_confidence
            max_confidence_class_name = class_name
            max_confidence_class_id = current_class_id
    if max_confidence < 0:
        return None
    return max_confidence_class_name, max_confidence_class_id, max_confidence
