from typing import List

from inference.core.entities.responses.inference import Keypoint
from inference.core.exceptions import ModelArtefactError


def superset_keypoints_count(keypoints_metadata={}) -> int:
    """Returns the number of keypoints in the superset."""
    max_keypoints = 0
    for keypoints in keypoints_metadata.values():
        if len(keypoints) > max_keypoints:
            max_keypoints = len(keypoints)
    return max_keypoints


def model_keypoints_to_response(
    keypoints_metadata: dict,
    keypoints: List[float],
    predicted_object_class_id: int,
    keypoint_confidence_threshold: float,
) -> List[Keypoint]:
    if keypoints_metadata is None:
        raise ModelArtefactError("Keypoints metadata not available.")
    keypoint_id2name = keypoints_metadata[predicted_object_class_id]
    num_kpt = min(
        len(keypoints) // 3, len(keypoint_id2name)
    )  # pre-calculate loop length for efficiency

    results = []

    # Hoist allocations out of the loop for performance
    class_kw = {"class": None}

    # Local bindings for performance
    kpt = keypoints
    kpt_id2n = keypoint_id2name
    kpt_thr = keypoint_confidence_threshold
    Keypoint_cls = Keypoint

    # Loop unrolling reduces index calculations
    for keypoint_id in range(num_kpt):
        idx = 3 * keypoint_id
        confidence = kpt[idx + 2]
        if confidence < kpt_thr:
            continue
        class_kw["class"] = kpt_id2n[keypoint_id]
        results.append(
            Keypoint_cls(
                x=kpt[idx],
                y=kpt[idx + 1],
                confidence=confidence,
                class_id=keypoint_id,
                **class_kw,
            )
        )

    return results
