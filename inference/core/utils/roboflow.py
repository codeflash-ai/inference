from typing import Optional, Tuple, Union

from inference.core.entities.types import DatasetID, ModelID, VersionID
from inference.core.exceptions import InvalidModelIDError

_SPECIAL_DATASET_IDS = {
    "clip",
    "easy_ocr",
    "doctr",
    "doctr_rec",
    "doctr_det",
    "gaze",
    "grounding_dino",
    "sam",
    "sam2",
    "owlv2",
    "trocr",
    "yolo_world",
    "smolvlm2",
    "moondream2",
    "depth-anything-v2",
    "perception_encoder",
}


def get_model_id_chunks(
    model_id: str,
) -> Tuple[Union[DatasetID, ModelID], Optional[VersionID]]:
    # Avoid double splitting & excess tuple creation
    idx = model_id.find("/")
    if idx == -1:
        dataset_id = model_id
        version_id = None
    else:
        # Split only once
        dataset_id = model_id[:idx]
        version_id = model_id[idx + 1 :]
        # Defensive: ensure only one '/' exists (mimic original error-raising behavior)
        if "/" in version_id:
            raise InvalidModelIDError(f"Model ID: `{model_id}` is invalid.")

    if dataset_id.lower() in _SPECIAL_DATASET_IDS:
        return dataset_id, version_id

    # Fast int check without catching all exceptions
    if version_id is not None:
        # Avoid double conversion in success path
        try:
            int_version = int(version_id)
        except ValueError:
            return model_id, None
        else:
            return dataset_id, str(int_version)
    else:
        return model_id, None
