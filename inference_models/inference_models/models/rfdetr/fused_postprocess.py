from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


MAX_RFDETR_DETECTIONS = 100


if triton is not None:

    @triton.jit
    def _select_topk_boxes_kernel(
        logits_ptr,
        bboxes_ptr,
        class_mapping_ptr,
        scores_out_ptr,
        classes_out_ptr,
        boxes_out_ptr,
        queries_out_ptr,
        count_out_ptr,
        threshold: tl.constexpr,
        inference_width: tl.constexpr,
        inference_height: tl.constexpr,
        scale_width: tl.constexpr,
        scale_height: tl.constexpr,
        original_width: tl.constexpr,
        original_height: tl.constexpr,
        num_queries: tl.constexpr,
        num_logits_classes: tl.constexpr,
        has_class_mapping: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.arange(0, block_size)
        valid_offsets = offsets < (num_queries * num_logits_classes)
        scores = tl.load(logits_ptr + offsets, mask=valid_offsets, other=-float("inf"))
        scores = tl.where(valid_offsets, scores, -float("inf"))
        selected_count = tl.full((), 0, tl.int32)
        iteration = tl.full((), 0, tl.int32)
        top_score = threshold + 1.0

        while (iteration < 100) & (top_score > threshold):
            top_score = tl.max(scores, axis=0)
            is_top = scores == top_score
            top_offset = tl.min(tl.where(is_top, offsets, block_size), axis=0)
            query_index = top_offset // num_logits_classes
            raw_class_id = top_offset - query_index * num_logits_classes

            if has_class_mapping:
                class_id = tl.load(class_mapping_ptr + raw_class_id)
                keep = (class_id >= 0) & (top_score > threshold)
            else:
                class_id = raw_class_id
                keep = (raw_class_id < (num_logits_classes - 1)) & (
                    top_score > threshold
                )

            out_index = selected_count
            cx = tl.load(bboxes_ptr + query_index * 4 + 0)
            cy = tl.load(bboxes_ptr + query_index * 4 + 1)
            w = tl.load(bboxes_ptr + query_index * 4 + 2)
            h = tl.load(bboxes_ptr + query_index * 4 + 3)
            x1 = (cx - 0.5 * w) * inference_width / scale_width
            y1 = (cy - 0.5 * h) * inference_height / scale_height
            x2 = (cx + 0.5 * w) * inference_width / scale_width
            y2 = (cy + 0.5 * h) * inference_height / scale_height
            x1 = tl.minimum(tl.maximum(x1, 0.0), original_width)
            y1 = tl.minimum(tl.maximum(y1, 0.0), original_height)
            x2 = tl.minimum(tl.maximum(x2, 0.0), original_width)
            y2 = tl.minimum(tl.maximum(y2, 0.0), original_height)

            tl.store(scores_out_ptr + out_index, top_score, mask=keep)
            tl.store(classes_out_ptr + out_index, class_id, mask=keep)
            tl.store(queries_out_ptr + out_index, query_index, mask=keep)
            tl.store(boxes_out_ptr + out_index * 4 + 0, x1, mask=keep)
            tl.store(boxes_out_ptr + out_index * 4 + 1, y1, mask=keep)
            tl.store(boxes_out_ptr + out_index * 4 + 2, x2, mask=keep)
            tl.store(boxes_out_ptr + out_index * 4 + 3, y2, mask=keep)
            selected_count += keep.to(tl.int32)
            scores = tl.where(offsets == top_offset, -float("inf"), scores)
            iteration += 1

        tl.store(count_out_ptr, selected_count)

    @triton.jit
    def _resize_selected_masks_kernel(
        masks_ptr,
        query_indices_ptr,
        count_ptr,
        output_ptr,
        in_height: tl.constexpr,
        in_width: tl.constexpr,
        out_height: tl.constexpr,
        out_width: tl.constexpr,
        block_size: tl.constexpr,
    ):
        det_index = tl.program_id(0)
        pixel_block = tl.program_id(1)
        count = tl.load(count_ptr)
        offsets = pixel_block * block_size + tl.arange(0, block_size)
        total_pixels = out_height * out_width
        valid = (det_index < count) & (offsets < total_pixels)
        query_index = tl.load(query_indices_ptr + det_index, mask=det_index < count)
        out_y = offsets // out_width
        out_x = offsets - out_y * out_width

        in_y = (out_y.to(tl.float32) + 0.5) * in_height / out_height - 0.5
        in_x = (out_x.to(tl.float32) + 0.5) * in_width / out_width - 0.5
        y0f = tl.floor(in_y)
        x0f = tl.floor(in_x)
        y0 = y0f.to(tl.int32)
        x0 = x0f.to(tl.int32)
        y1 = y0 + 1
        x1 = x0 + 1
        wy = in_y - y0f
        wx = in_x - x0f
        y0 = tl.minimum(tl.maximum(y0, 0), in_height - 1)
        y1 = tl.minimum(tl.maximum(y1, 0), in_height - 1)
        x0 = tl.minimum(tl.maximum(x0, 0), in_width - 1)
        x1 = tl.minimum(tl.maximum(x1, 0), in_width - 1)

        mask_base = masks_ptr + query_index * in_height * in_width
        v00 = tl.load(mask_base + y0 * in_width + x0, mask=valid, other=0.0)
        v01 = tl.load(mask_base + y0 * in_width + x1, mask=valid, other=0.0)
        v10 = tl.load(mask_base + y1 * in_width + x0, mask=valid, other=0.0)
        v11 = tl.load(mask_base + y1 * in_width + x1, mask=valid, other=0.0)
        top = v00 * (1.0 - wx) + v01 * wx
        bottom = v10 * (1.0 - wx) + v11 * wx
        resized = top * (1.0 - wy) + bottom * wy
        tl.store(
            output_ptr + det_index * total_pixels + offsets,
            resized > 0.0,
            mask=valid,
        )


def fused_select_topk_boxes(
    image_bboxes: torch.Tensor,
    image_logits: torch.Tensor,
    threshold: float,
    inference_width: int,
    inference_height: int,
    scale_width: float,
    scale_height: float,
    original_width: int,
    original_height: int,
    class_mapping: Optional[torch.Tensor],
    return_cpu_count: bool = True,
) -> Optional[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    if triton is None or not image_logits.is_cuda or image_logits.ndim != 2:
        return None
    num_queries, num_logits_classes = image_logits.shape
    if num_queries != MAX_RFDETR_DETECTIONS:
        return None
    block_size = triton.next_power_of_2(num_queries * num_logits_classes)
    scores = torch.empty((MAX_RFDETR_DETECTIONS,), device=image_logits.device)
    classes = torch.empty(
        (MAX_RFDETR_DETECTIONS,), dtype=torch.int32, device=image_logits.device
    )
    boxes = torch.empty(
        (MAX_RFDETR_DETECTIONS, 4),
        dtype=image_bboxes.dtype,
        device=image_bboxes.device,
    )
    query_indices = torch.zeros(
        (MAX_RFDETR_DETECTIONS,), dtype=torch.int32, device=image_logits.device
    )
    count = torch.empty((1,), dtype=torch.int32, device=image_logits.device)
    if class_mapping is None:
        class_mapping = torch.empty((1,), dtype=torch.int32, device=image_logits.device)
        has_class_mapping = False
    else:
        has_class_mapping = True
    _select_topk_boxes_kernel[(1,)](
        image_logits,
        image_bboxes,
        class_mapping,
        scores,
        classes,
        boxes,
        query_indices,
        count,
        float(threshold),
        int(inference_width),
        int(inference_height),
        float(scale_width),
        float(scale_height),
        int(original_width),
        int(original_height),
        num_queries,
        num_logits_classes,
        has_class_mapping,
        block_size,
        num_warps=8,
    )
    if not return_cpu_count:
        return scores, classes, boxes, query_indices.to(dtype=torch.long), count
    selected_count = int(count.cpu().item())
    return (
        scores[:selected_count],
        classes[:selected_count],
        boxes[:selected_count],
        query_indices[:selected_count].to(dtype=torch.long),
        selected_count,
    )


def fused_resize_selected_masks(
    image_masks: torch.Tensor,
    query_indices: torch.Tensor,
    count: torch.Tensor,
    output_height: int,
    output_width: int,
    detection_limit: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if triton is None or not image_masks.is_cuda:
        return None
    output = torch.empty(
        (MAX_RFDETR_DETECTIONS, output_height, output_width),
        dtype=torch.bool,
        device=image_masks.device,
    )
    _, input_height, input_width = image_masks.shape
    block_size = 256
    if detection_limit is None:
        detection_limit = MAX_RFDETR_DETECTIONS
    detection_limit = min(max(int(detection_limit), 1), MAX_RFDETR_DETECTIONS)
    grid = (
        detection_limit,
        triton.cdiv(output_height * output_width, block_size),
    )
    _resize_selected_masks_kernel[grid](
        image_masks,
        query_indices,
        count,
        output,
        input_height,
        input_width,
        output_height,
        output_width,
        block_size,
        num_warps=4,
    )
    return output
