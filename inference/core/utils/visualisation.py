from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from inference.core.entities.requests.inference import (
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.responses.inference import (
    InstanceSegmentationPrediction,
    Keypoint,
    KeypointsPrediction,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    Point,
)
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image_rgb


def draw_detection_predictions(
    inference_request: Union[
        ObjectDetectionInferenceRequest,
        InstanceSegmentationInferenceRequest,
        KeypointsDetectionInferenceRequest,
    ],
    inference_response: Union[
        ObjectDetectionInferenceResponse,
        InstanceSegmentationPrediction,
        KeypointsPrediction,
    ],
    colors: Dict[str, str],
) -> bytes:
    image = load_image_rgb(inference_request.image)
    for box in inference_response.predictions:
        color = tuple(
            int(colors.get(box.class_name, "#4892EA")[i : i + 2], 16) for i in (1, 3, 5)
        )
        image = draw_bbox(
            image=image,
            box=box,
            color=color,
            thickness=inference_request.visualization_stroke_width,
        )
        if hasattr(box, "points"):
            image = draw_instance_segmentation_points(
                image=image,
                points=box.points,
                color=color,
                thickness=inference_request.visualization_stroke_width,
            )
        if hasattr(box, "keypoints"):
            draw_keypoints(
                image=image,
                keypoints=box.keypoints,
                color=color,
                thickness=inference_request.visualization_stroke_width,
            )
        if inference_request.visualization_labels:
            image = draw_labels(
                image=image,
                box=box,
                color=color,
            )
    image_bgr = image[:, :, ::-1]
    return encode_image_to_jpeg_bytes(image=image_bgr)


def draw_bbox(
    image: np.ndarray,
    box: ObjectDetectionPrediction,
    color: Tuple[int, ...],
    thickness: int,
) -> np.ndarray:
    left_top, right_bottom = bbox_to_points(box=box)
    return cv2.rectangle(
        image,
        left_top,
        right_bottom,
        color=color,
        thickness=thickness,
    )


def draw_instance_segmentation_points(
    image: np.ndarray,
    points: List[Point],
    color: Tuple[int, ...],
    thickness: int,
) -> np.ndarray:
    points_array = np.array([(int(p.x), int(p.y)) for p in points], np.int32)
    if len(points) > 2:
        image = cv2.polylines(
            image,
            [points_array],
            isClosed=True,
            color=color,
            thickness=thickness,
        )
    return image


def draw_keypoints(
    image: np.ndarray,
    keypoints: List[Keypoint],
    color: Tuple[int, ...],
    thickness: int,
) -> None:
    for keypoint in keypoints:
        center_coordinates = (round(keypoint.x), round(keypoint.y))
        image = cv2.circle(
            image,
            center_coordinates,
            thickness,
            color,
            -1,
        )


def draw_labels(
    image: np.ndarray,
    box: Union[ObjectDetectionPrediction, InstanceSegmentationPrediction],
    color: Tuple[int, ...],
) -> np.ndarray:
    (x1, y1), _ = bbox_to_points(box=box)
    text = f"{box.class_name} {box.confidence:.2f}"
    # Cache frequently used params
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # Compute button size and its maximum possible overlays using min() only once
    button_w = text_width + 20
    button_h = text_height + 20
    img_h, img_w = image.shape[:2]
    max_w = img_w - x1 if x1 < img_w else 0
    max_h = img_h - y1 if y1 < img_h else 0
    draw_w = min(button_w, max_w)
    draw_h = min(button_h, max_h)
    if draw_w <= 0 or draw_h <= 0:
        return image

    # Reversed color as before but create the array only as big as needed to overlay
    button_img = np.empty((draw_h, draw_w, 3), dtype=np.uint8)
    button_img[...] = color[::-1]
    text_x = 10
    text_y = 10 + text_height
    if text_y < draw_h and text_x < draw_w:
        cv2.putText(
            button_img,
            text,
            (text_x, min(text_y, draw_h - 1)),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )
    # Overwrite image region in-place only for the intersection
    image[y1 : y1 + draw_h, x1 : x1 + draw_w] = button_img
    return image


def bbox_to_points(
    box: Union[ObjectDetectionPrediction, InstanceSegmentationPrediction],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # Avoid repeated box property access
    bx = box.x
    by = box.y
    bw = box.width
    bh = box.height
    x1 = int(bx - bw * 0.5)
    x2 = int(bx + bw * 0.5)
    y1 = int(by - bh * 0.5)
    y2 = int(by + bh * 0.5)
    return (x1, y1), (x2, y2)
