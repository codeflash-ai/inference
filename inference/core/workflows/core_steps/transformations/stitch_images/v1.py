from typing import List, Literal, Optional, Type, Union

import cv2 as cv
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.logger import logger
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "stitched_image"
LONG_DESCRIPTION = """
This block combines two related scenes both containing fair amount of details.
Block is utilizing Scale Invariant Feature Transform (SIFT)
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Stitch Images",
            "version": "v1",
            "short_description": "Stitch two images by common parts.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "far fa-object-union",
                "opencv": True,
                "blockPriority": 6,
            },
        }
    )
    type: Literal["roboflow_core/stitch_images@v1"]
    image1: Selector(kind=[IMAGE_KIND]) = Field(
        title="First image to stitch",
        description="First input image for this step.",
        examples=["$inputs.image1"],
        validation_alias=AliasChoices("image1"),
    )
    image2: Selector(kind=[IMAGE_KIND]) = Field(
        title="Second image to stitch",
        description="Second input image for this step.",
        examples=["$inputs.image2"],
        validation_alias=AliasChoices("image2"),
    )
    max_allowed_reprojection_error: Union[Optional[float], Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=3,
        description="Advanced parameter overwriting cv.findHomography ransacReprojThreshold parameter."
        " Maximum allowed reprojection error to treat a point pair as an inlier."
        " Increasing value of this parameter for low details photo may yield better results.",
        examples=[3, "$inputs.min_overlap_ratio_w"],
    )
    count_of_best_matches_per_query_descriptor: Union[Optional[int], Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        default=2,
        description="Advanced parameter overwriting cv.BFMatcher.knnMatch `k` parameter."
        " Count of best matches found per each query descriptor or less if a query descriptor has less than k possible matches in total.",
        examples=[2, "$inputs.k"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=OUTPUT_KEY, kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class StitchImagesBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image1: WorkflowImageData,
        image2: WorkflowImageData,
        count_of_best_matches_per_query_descriptor: int,
        max_allowed_reprojection_error: float,
    ) -> BlockResult:
        if count_of_best_matches_per_query_descriptor == 0:
            raise ValueError(
                "count_of_best_matches_per_query_descriptor must be greater than 0"
            )
        try:
            merged_image = stitch_images(
                image1=image1.numpy_image,
                image2=image2.numpy_image,
                count_of_best_matches_per_query_descriptor=abs(
                    int(round(count_of_best_matches_per_query_descriptor))
                ),
                max_allowed_reprojection_error=abs(max_allowed_reprojection_error),
            )
        except Exception as exc:
            logger.info("Stitching failed, %s", exc)
            return {OUTPUT_KEY: None}
        parent_metadata = ImageParentMetadata(
            parent_id=f"{image1.parent_metadata.parent_id} + {image2.parent_metadata.parent_id}"
        )
        return {
            OUTPUT_KEY: WorkflowImageData(
                parent_metadata=parent_metadata,
                numpy_image=merged_image,
            )
        }


def stitch_images(
    image1: np.ndarray,
    image2: np.ndarray,
    count_of_best_matches_per_query_descriptor: int,
    max_allowed_reprojection_error: float,
) -> np.ndarray:
    # Create SIFT detector once for this function
    sift = cv.SIFT_create()

    # Compute keypoints and descriptors for each image
    keypoints_1, descriptors_1 = sift.detectAndCompute(image=image1, mask=None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image=image2, mask=None)

    # If not enough descriptors, raise immediately (to avoid running downstream code)
    if (
        descriptors_1 is None
        or descriptors_2 is None
        or len(descriptors_1) < 2
        or len(descriptors_2) < 2
    ):
        raise RuntimeError(
            "Not enough descriptors in one or both images for stitching."
        )

    bf = cv.BFMatcher_create()
    matches = bf.knnMatch(
        queryDescriptors=descriptors_1,
        trainDescriptors=descriptors_2,
        k=count_of_best_matches_per_query_descriptor,
    )

    # Use NumPy array for filtering distances to reduce Python-level iteration
    # Only retain pairs where both 2 or more matches are found (OpenCV can return less)
    filtered_matches = [m for m in matches if len(m) >= 2]
    if len(filtered_matches) == 0:
        raise RuntimeError("No matches found between given images for stitching.")

    distances = np.array(
        [[pair[0].distance, pair[1].distance] for pair in filtered_matches]
    )
    good_indices = np.where(distances[:, 0] < 0.75 * distances[:, 1])[0]
    if good_indices.size == 0:
        raise RuntimeError("No good matches found between images for stitching.")
    # Only extract matches that passed the ratio test
    good_matches = [filtered_matches[idx][0] for idx in good_indices]

    # Compose correspondence points using list comprehensions with preallocation
    # This is about as fast as possible in Python for nontrivial lists of custom objects
    image1_pts = np.empty((len(good_matches), 1, 2), dtype=np.float32)
    image2_pts = np.empty((len(good_matches), 1, 2), dtype=np.float32)
    for i, m in enumerate(good_matches):
        image1_pts[i, 0, :] = keypoints_1[m.queryIdx].pt
        image2_pts[i, 0, :] = keypoints_2[m.trainIdx].pt

    # Compute which image should be 'first' by mean of x-coordinates, in a vectorized way
    keypoints_1_x = np.empty(len(keypoints_1), dtype=np.float32)
    keypoints_2_x = np.empty(len(keypoints_2), dtype=np.float32)
    for i, kp in enumerate(keypoints_1):
        keypoints_1_x[i] = kp.pt[0]
    for i, kp in enumerate(keypoints_2):
        keypoints_2_x[i] = kp.pt[0]
    image1_first = keypoints_1_x.mean() < keypoints_2_x.mean()
    if image1_first:
        first_image_pts = image1_pts
        second_image_pts = image2_pts
        first_image = image2
        second_image = image1
    else:
        first_image_pts = image2_pts
        second_image_pts = image1_pts
        first_image = image1
        second_image = image2

    # Find homography
    transformation_matrix, mask = cv.findHomography(
        srcPoints=first_image_pts,
        dstPoints=second_image_pts,
        method=cv.RANSAC,
        ransacReprojThreshold=max_allowed_reprojection_error,
    )
    if transformation_matrix is None:
        raise RuntimeError("Homography computation failed.")

    h1, w1 = first_image.shape[:2]
    h2, w2 = second_image.shape[:2]

    # Warp all 4 corners of the second image and compute bounds
    second_image_corners = np.array(
        [[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32
    ).reshape(-1, 1, 2)
    warped_corners = cv.perspectiveTransform(
        src=second_image_corners,
        m=transformation_matrix,
    )
    min_xy = warped_corners.min(axis=0).ravel()
    max_xy = warped_corners.max(axis=0).ravel()
    xmin, ymin = np.int32(min_xy)
    xmax, ymax = np.int32(max_xy)

    xmax = max(xmax, w1)
    ymax = max(ymax, h1)

    translation_dist = [-xmin, -ymin]
    # Clamp translation if negative (kept from original logic)
    if translation_dist[0] < 0 or translation_dist[1] < 0:
        translation_dist = [max(0, translation_dist[0]), max(0, translation_dist[1])]

    H_translation = np.eye(3, dtype=np.float64)
    H_translation[0, 2] = translation_dist[0]
    H_translation[1, 2] = translation_dist[1]

    output_size = (xmax - xmin, ymax - ymin)

    second_image_warped = cv.warpPerspective(
        src=second_image,
        M=H_translation @ transformation_matrix,
        dsize=output_size,
    )

    # Only copy first image if within boundaries
    t_x, t_y = translation_dist
    if (
        t_x + w1 <= second_image_warped.shape[1]
        and t_y + h1 <= second_image_warped.shape[0]
    ):
        second_image_warped[t_y : t_y + h1, t_x : t_x + w1] = first_image

    return second_image_warped
