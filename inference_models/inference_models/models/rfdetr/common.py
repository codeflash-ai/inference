from typing import List, Optional, Tuple, Union

import torch
from torchvision.transforms import functional

from inference_models import Detections, InstanceDetections, InstancesRLEMasks
from inference_models.entities import ImageDimensions
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
    rescale_image_detections,
)
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle
from inference_models.models.rfdetr.class_remapping import ClassesReMapping
from inference_models.models.rfdetr.fused_postprocess import (
    fused_resize_selected_masks,
    fused_select_topk_boxes,
)
from inference_models.models.rfdetr.post_processor import select_topk_predictions
from inference_models.utils.file_system import read_json


def parse_model_type(config_path: str) -> str:
    try:
        parsed_config = read_json(path=config_path)
        if not isinstance(parsed_config, dict):
            raise ValueError(
                f"decoded value is {type(parsed_config)}, but dictionary expected"
            )
        if "model_type" not in parsed_config or not isinstance(
            parsed_config["model_type"], str
        ):
            raise ValueError(
                "could not find required entries in config - either "
                "'model_type' field is missing or not a string"
            )
        return parsed_config["model_type"]
    except (IOError, OSError, ValueError) as error:
        raise CorruptedModelPackageError(
            message=f"Model type config file is malformed: "
            f"{error}. In case that the package is "
            f"hosted on the Roboflow platform - contact support. If you created model package manually, please "
            f"verify its consistency in docs.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        ) from error


def post_process_object_detection_results(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
    device: torch.device,
) -> List[Detections]:
    logits_sigmoid = torch.nn.functional.sigmoid(logits)
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(device=device, dtype=logits_sigmoid.dtype)
    results = []
    for image_bboxes, image_logits, image_meta in zip(
        bboxes, logits_sigmoid, pre_processing_meta
    ):
        predicted_confidence, top_classes, image_bboxes, _ = select_topk_predictions(
            logits_sigmoid=image_logits,
            bboxes_cxcywh=image_bboxes,
        )
        if classes_re_mapping is not None:
            remapping_mask = torch.isin(
                top_classes, classes_re_mapping.remaining_class_ids
            )
            top_classes = classes_re_mapping.class_mapping[top_classes[remapping_mask]]
            predicted_confidence = predicted_confidence[remapping_mask]
            image_bboxes = image_bboxes[remapping_mask]
        else:
            # drop DETR no-object rows
            named = top_classes < num_classes
            predicted_confidence = predicted_confidence[named]
            top_classes = top_classes[named]
            image_bboxes = image_bboxes[named]
        confidence_mask = predicted_confidence > (
            threshold[top_classes.long()]
            if isinstance(threshold, torch.Tensor)
            else threshold
        )
        predicted_confidence = predicted_confidence[confidence_mask]
        top_classes = top_classes[confidence_mask]
        selected_boxes = image_bboxes[confidence_mask]
        # select_topk_predictions returns scores sorted descending; the boolean
        # filters above preserve that order.
        cxcy = selected_boxes[:, :2]
        wh = selected_boxes[:, 2:]
        xy_min = cxcy - 0.5 * wh
        xy_max = cxcy + 0.5 * wh
        selected_boxes_xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
        denorm_size = (
            image_meta.nonsquare_intermediate_size or image_meta.inference_size
        )
        inference_size_whwh = torch.tensor(
            [
                denorm_size.width,
                denorm_size.height,
                denorm_size.width,
                denorm_size.height,
            ],
            device=device,
        )
        selected_boxes_xyxy = selected_boxes_xyxy_pct * inference_size_whwh
        selected_boxes_xyxy = rescale_image_detections(
            image_detections=selected_boxes_xyxy,
            image_metadata=image_meta,
        )
        results.append(
            Detections(
                xyxy=selected_boxes_xyxy.round().int(),
                confidence=predicted_confidence,
                class_id=top_classes.int(),
            )
        )
    return results


def post_process_instance_segmentation_results(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
    defer_fused_postprocess_count: bool = False,
) -> List[InstanceDetections]:
    logits_sigmoid = torch.nn.functional.sigmoid(logits)
    results = []
    device = bboxes.device
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(device=device, dtype=logits_sigmoid.dtype)
    for image_bboxes, image_logits, image_masks, image_meta in zip(
        bboxes, logits_sigmoid, masks, pre_processing_meta
    ):
        fused_result = _try_fused_instance_segmentation_post_process(
            image_bboxes=image_bboxes,
            image_logits=image_logits,
            image_masks=image_masks,
            image_meta=image_meta,
            threshold=threshold,
            classes_re_mapping=classes_re_mapping,
            defer_count=defer_fused_postprocess_count,
        )
        if fused_result is not None:
            results.append(fused_result)
            continue
        confidence, top_classes, image_bboxes, query_indices = select_topk_predictions(
            logits_sigmoid=image_logits,
            bboxes_cxcywh=image_bboxes,
        )
        image_masks = image_masks[query_indices]
        if classes_re_mapping is not None:
            remapping_mask = torch.isin(
                top_classes, classes_re_mapping.remaining_class_ids
            )
            top_classes = classes_re_mapping.class_mapping[top_classes[remapping_mask]]
            confidence = confidence[remapping_mask]
            image_bboxes = image_bboxes[remapping_mask]
            image_masks = image_masks[remapping_mask]
        else:
            # drop DETR no-object rows
            named = top_classes < num_classes
            confidence = confidence[named]
            top_classes = top_classes[named]
            image_bboxes = image_bboxes[named]
            image_masks = image_masks[named]
        confidence_mask = confidence > (
            threshold[top_classes.long()]
            if isinstance(threshold, torch.Tensor)
            else threshold
        )
        confidence = confidence[confidence_mask]
        top_classes = top_classes[confidence_mask]
        selected_boxes = image_bboxes[confidence_mask]
        selected_masks = image_masks[confidence_mask]
        # select_topk_predictions returns scores sorted descending; the boolean
        # filters above preserve that order.
        cxcy = selected_boxes[:, :2]
        wh = selected_boxes[:, 2:]
        xy_min = cxcy - 0.5 * wh
        xy_max = cxcy + 0.5 * wh
        selected_boxes_xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
        denorm_size = (
            image_meta.nonsquare_intermediate_size or image_meta.inference_size
        )
        denorm_size_whwh = torch.tensor(
            [
                denorm_size.width,
                denorm_size.height,
                denorm_size.width,
                denorm_size.height,
            ],
            device=device,
        )
        padding = (
            image_meta.pad_left,
            image_meta.pad_top,
            image_meta.pad_right,
            image_meta.pad_bottom,
        )
        selected_boxes_xyxy = selected_boxes_xyxy_pct * denorm_size_whwh
        aligned_boxes, aligned_masks = align_instance_segmentation_results(
            image_bboxes=selected_boxes_xyxy,
            masks=selected_masks,
            padding=padding,
            scale_height=image_meta.scale_height,
            scale_width=image_meta.scale_width,
            original_size=image_meta.original_size,
            size_after_pre_processing=image_meta.size_after_pre_processing,
            inference_size=denorm_size,
            static_crop_offset=image_meta.static_crop_offset,
        )
        detections = InstanceDetections(
            xyxy=aligned_boxes.round().int(),
            confidence=confidence,
            class_id=top_classes.int(),
            mask=aligned_masks,
        )
        results.append(detections)
    return results


def _try_fused_instance_segmentation_post_process(
    image_bboxes: torch.Tensor,
    image_logits: torch.Tensor,
    image_masks: torch.Tensor,
    image_meta: PreProcessingMetadata,
    threshold: Union[float, torch.Tensor],
    classes_re_mapping: Optional[ClassesReMapping],
    defer_count: bool = False,
) -> Optional[InstanceDetections]:
    if isinstance(threshold, torch.Tensor) or classes_re_mapping is None:
        return None
    if (
        image_meta.pad_left != 0
        or image_meta.pad_top != 0
        or image_meta.pad_right != 0
        or image_meta.pad_bottom != 0
        or image_meta.static_crop_offset.offset_x != 0
        or image_meta.static_crop_offset.offset_y != 0
        or image_meta.nonsquare_intermediate_size is not None
        or image_meta.original_size.width != image_meta.size_after_pre_processing.width
        or image_meta.original_size.height
        != image_meta.size_after_pre_processing.height
    ):
        return None
    fused = fused_select_topk_boxes(
        image_bboxes=image_bboxes,
        image_logits=image_logits,
        threshold=threshold,
        inference_width=image_meta.inference_size.width,
        inference_height=image_meta.inference_size.height,
        scale_width=image_meta.scale_width,
        scale_height=image_meta.scale_height,
        original_width=image_meta.original_size.width,
        original_height=image_meta.original_size.height,
        class_mapping=classes_re_mapping.class_mapping,
        return_cpu_count=not defer_count,
    )
    if fused is None:
        return None
    confidence, top_classes, selected_boxes, query_indices, selected_count = fused
    if defer_count:
        aligned_masks = fused_resize_selected_masks(
            image_masks=image_masks,
            query_indices=query_indices,
            count=selected_count,
            output_height=image_meta.original_size.height,
            output_width=image_meta.original_size.width,
        )
        if aligned_masks is None:
            return None
        return InstanceDetections(
            xyxy=selected_boxes.round().int(),
            confidence=confidence,
            class_id=top_classes.int(),
            mask=aligned_masks,
            image_metadata={"valid_count": selected_count},
        )
    if selected_count == 0:
        aligned_masks = torch.empty(
            size=(0, image_meta.original_size.height, image_meta.original_size.width),
            dtype=torch.bool,
            device=image_bboxes.device,
        )
    else:
        selected_masks = image_masks[query_indices]
        aligned_masks = (
            functional.resize(
                selected_masks,
                [image_meta.original_size.height, image_meta.original_size.width],
                interpolation=functional.InterpolationMode.BILINEAR,
            )
            .gt_(0.0)
            .to(dtype=torch.bool)
        )
    return InstanceDetections(
        xyxy=selected_boxes.round().int(),
        confidence=confidence,
        class_id=top_classes.int(),
        mask=aligned_masks,
    )


def post_process_instance_segmentation_results_to_rle_masks(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    pre_processing_meta: List[PreProcessingMetadata],
    threshold: Union[float, torch.Tensor],
    num_classes: int,
    classes_re_mapping: Optional[ClassesReMapping],
) -> List[InstanceDetections]:
    logits_sigmoid = torch.nn.functional.sigmoid(logits)
    final_results = []
    device = bboxes.device
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(device=device, dtype=logits_sigmoid.dtype)
    for image_bboxes, image_logits, image_masks, image_meta in zip(
        bboxes, logits_sigmoid, masks, pre_processing_meta
    ):
        confidence, top_classes, image_bboxes, query_indices = select_topk_predictions(
            logits_sigmoid=image_logits,
            bboxes_cxcywh=image_bboxes,
        )
        image_masks = image_masks[query_indices]
        if classes_re_mapping is not None:
            remapping_mask = torch.isin(
                top_classes, classes_re_mapping.remaining_class_ids
            )
            top_classes = classes_re_mapping.class_mapping[top_classes[remapping_mask]]
            confidence = confidence[remapping_mask]
            image_bboxes = image_bboxes[remapping_mask]
            image_masks = image_masks[remapping_mask]
        else:
            # drop DETR no-object rows
            named = top_classes < num_classes
            confidence = confidence[named]
            top_classes = top_classes[named]
            image_bboxes = image_bboxes[named]
            image_masks = image_masks[named]
        confidence_mask = confidence > (
            threshold[top_classes.long()]
            if isinstance(threshold, torch.Tensor)
            else threshold
        )
        confidence = confidence[confidence_mask]
        top_classes = top_classes[confidence_mask]
        selected_boxes = image_bboxes[confidence_mask]
        selected_masks = image_masks[confidence_mask]
        # select_topk_predictions returns scores sorted descending; the boolean
        # filters above preserve that order.
        cxcy = selected_boxes[:, :2]
        wh = selected_boxes[:, 2:]
        xy_min = cxcy - 0.5 * wh
        xy_max = cxcy + 0.5 * wh
        selected_boxes_xyxy_pct = torch.cat([xy_min, xy_max], dim=-1)
        denorm_size = (
            image_meta.nonsquare_intermediate_size or image_meta.inference_size
        )
        denorm_size_whwh = torch.tensor(
            [
                denorm_size.width,
                denorm_size.height,
                denorm_size.width,
                denorm_size.height,
            ],
            device=device,
        )
        padding = (
            image_meta.pad_left,
            image_meta.pad_top,
            image_meta.pad_right,
            image_meta.pad_bottom,
        )
        selected_boxes_xyxy = selected_boxes_xyxy_pct * denorm_size_whwh
        aligned_boxes, aligned_masks = align_instance_segmentation_results(
            image_bboxes=selected_boxes_xyxy,
            masks=selected_masks,
            padding=padding,
            scale_height=image_meta.scale_height,
            scale_width=image_meta.scale_width,
            original_size=image_meta.original_size,
            size_after_pre_processing=image_meta.size_after_pre_processing,
            inference_size=denorm_size,
            static_crop_offset=image_meta.static_crop_offset,
        )
        rle_masks = [torch_mask_to_coco_rle(mask) for mask in aligned_masks]
        instances_masks = InstancesRLEMasks.from_coco_rle_masks(
            image_size=(
                image_meta.original_size.height,
                image_meta.original_size.width,
            ),
            masks=rle_masks,
        )
        if aligned_boxes.shape[0] > 0:
            final_results.append(
                InstanceDetections(
                    xyxy=aligned_boxes.round().int(),
                    confidence=confidence,
                    class_id=top_classes.int(),
                    mask=instances_masks,
                )
            )
        else:
            final_results.append(
                InstanceDetections(
                    xyxy=torch.empty(
                        (0, 4), dtype=torch.int32, device=image_bboxes.device
                    ),
                    class_id=top_classes.int(),
                    confidence=confidence,
                    mask=instances_masks,
                )
            )
    return final_results
