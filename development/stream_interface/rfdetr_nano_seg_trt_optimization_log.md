# RFDETR Nano Seg TRT Optimization Log

Benchmark command:

```bash
PYTHONPATH=/app/inference_models python development/stream_interface/rfdetr_nano_seg_trt_workflow.py --video_reference vehicles_312px.mp4
```

Hardware observed: Tesla T4, CUDA driver 580.159.04, PyTorch 2.6.0+cu124.

## 2026-05-22

### Baseline

- Hypothesis: Establish the current end-to-end workflow FPS and identify CPU/GPU bottlenecks before changing code.
- Command: benchmark command above.
- Result: `frames=538 elapsed=7.45s fps=72.23`.
- Profiling:
  - `nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --sample=none --cpuctxsw=none`.
  - CUDA API summary showed `cudaStreamSynchronize` as the largest CUDA API cost: 17,552 calls, 2.141s total API time.
  - Kernel summary showed TRT kernels plus PyTorch postprocess kernels; visible postprocess costs included `topk`, sort/indexing, and mask resize.
- Learning: RFDETR TRT inference already has a CUDA graph cache implementation, but the benchmark path was paying explicit CPU waits around preprocessing, TRT execution, and postprocessing.

### Existing CUDA Graph Cache Enabled By Env

- Hypothesis: The existing TensorRT CUDA graph cache should reduce per-frame launch overhead for the static RFDETR-nano input shape.
- Change: No code change. Ran with `ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=True`.
- Result: `frames=538 elapsed=7.28s fps=73.95`.
- Learning: Graph replay helps, but the requested benchmark command does not set this env var, and stage-level synchronizations still leave performance on the table.

### Async RFDETR Stage Scheduling

- Hypothesis: Replace RFDETR instance-segmentation CPU synchronizations with CUDA stream dependencies so CPU work can continue while GPU work is queued.
- Change:
  - Added `synchronize=True` parameter to `infer_from_trt_engine(...)` to preserve existing default behavior.
  - RFDETR instance segmentation calls TRT with `synchronize=False`.
  - Replaced RFDETR pre/forward/post `stream.synchronize()` calls with `wait_stream(...)` dependencies.
  - Made CUDA graph replay stream wait on the caller stream and the caller stream wait on graph replay, so asynchronous graph replay remains ordered without a CPU sync.
- Result without graph env: `frames=538 elapsed=7.30s fps=73.66`.
- Result with graph env: `frames=538 elapsed=7.00s fps=76.84`.
- Correctness: Compared optimized graph path against non-graph TRT path on 32 frames from `vehicles_312px.mp4`; class IDs matched exactly and max box delta was `0` px.
- Learning: Async scheduling helps modestly by itself and unlocks the graph cache benefit when graph replay is available.

### RFDETR Instance TRT Graph Cache By Default

- Hypothesis: Since RFDETR-nano-seg has static batch/input shape in this benchmark, enabling the graph cache by default for the RFDETR instance TRT model should make the requested command use the faster path without requiring env setup.
- Change: If no caller/env graph cache is supplied, `RFDetrForInstanceSegmentationTRT.from_pretrained(...)` now creates a `TRTCudaGraphCache` with the model's default cache capacity.
- Result on requested command: best observed `frames=538 elapsed=7.07s fps=76.12`; final verification repeat `frames=538 elapsed=7.14s fps=75.39`.
- Learning: End-to-end FPS improved from `72.23` to `75.39-76.12` (+4.4% to +5.4%) on the exact command.

### Rejected: Shared cv2 Preprocess Path

- Hypothesis: The shared cv2 stretch preprocessing path may be faster than RFDETR's PIL resize/to-tensor path.
- Change tested: No committed code change. Manually ran shared `pre_process_numpy_image(...)` into RFDETR TRT forward/postprocess.
- Result: Class order changed by frame 2 (`[7, 2, 2, 2]` vs `[2, 7, 2, 2]`).
- Learning: Even if boxes might remain close, this is too risky for the explicit "classes don't change" invariant. Keep RFDETR's PIL-compatible preprocessing.

### Rejected: Best-Class-Per-Query Postprocess

- Hypothesis: Replace flat top-k over `(queries * classes)` with one best valid class per query.
- Change tested: No committed code change.
- Result: Not semantics-preserving. Frame 81 had two above-threshold classes for the same query in the legacy path.
- Learning: RFDETR can emit multiple valid classes for one query; a max-per-query shortcut drops detections.

### Rejected: Threshold-First Exact Postprocess

- Hypothesis: Select all above-threshold valid query/class pairs first, then sort/top-k only those candidates to preserve flat-top-k semantics while reducing work.
- Change tested: Temporary code only; reverted.
- Correctness: Matched legacy postprocess on 128 video frames with exact class IDs and `0` px max box delta.
- Result: Requested workflow was `frames=538 elapsed=7.08s fps=75.97`, effectively flat/slightly worse than the graph-cache default result.
- Learning: The remaining end-to-end bottleneck is not improved enough by this postprocess rewrite; do not keep the added complexity.

### Batched RFDETR RLE Mask Alignment

- Hypothesis: The workflow v3 adapter asks instance segmentation models that support RLE for `mask_format="rle"`, then converts masks to polygons for the v3 response. RFDETR's RLE branch resized each selected mask one at a time; using the same batched mask alignment as the dense path should reduce GPU launch and resize overhead while preserving RLE output.
- Change: `post_process_instance_segmentation_results_to_rle_masks(...)` now calls `align_instance_segmentation_results(...)` once for the selected masks, then encodes each aligned boolean mask to COCO RLE.
- Correctness: Compared dense postprocess masks against decoded RLE masks on 64 frames from `vehicles_312px.mp4`; masks matched exactly, class IDs matched exactly, and max box delta was `0` px.
- Micro-result: RFDETR RLE postprocess improved from `3.19 ms/frame` to `2.58 ms/frame` over 120 video frames.
- Result on requested command: `frames=538 elapsed=6.86s fps=78.43`.
- Learning: After TRT graph replay, mask alignment/encoding is a meaningful share of the model-side cost. Batched GPU resize is preferable for this small-frame RFDETR workflow despite temporarily materializing dense aligned masks.

### Rejected: GPU/Tensor Preprocessing For Numpy Frames

- Hypothesis: Convert cv2 numpy frames to CUDA tensors immediately and use RFDETR's tensor preprocessing path to avoid PIL resize/to-tensor overhead.
- Change tested: No committed code change. Manually converted BGR `uint8` frames to CUDA `float32` CHW `[0, 1]` tensors and called the existing tensor preprocessing branch.
- Result: Not output-order preserving. By frame 4, the tensor path returned the same boxes/classes in a different confidence order; same-index box comparison had a max delta of `123` px.
- Learning: The PIL-compatible numpy preprocessing path remains necessary for the benchmark invariant. Revisit only with an order-insensitive downstream contract or a more exact PIL-equivalent GPU resize.

### Avoid RLE Round-Trip For Polygon Responses

- Hypothesis: Workflow v3 polygon responses should not ask `inference_models` instance segmentation backends for RLE masks. The previous adapter path requested RLE whenever the model supported it, then decoded RLE back to polygons, adding avoidable RLE encode/decode work.
- Change: `InferenceModelsInstanceSegmentationAdapter.map_inference_kwargs(...)` now requests `mask_format="rle"` only when `response_mask_format == "rle"`. Polygon responses use the model's dense default and convert dense masks directly to polygons.
- Correctness: Compared dense-to-polygon against RLE-decode-to-polygon on 64 frames from `vehicles_312px.mp4`; polygons matched exactly, class IDs matched exactly, and max box delta was `0` px.
- Micro-result: Adapter-like mask postprocess for polygon output improved from `3.40 ms/frame` with RLE round-trip to `2.24 ms/frame` with dense masks over 160 video frames.
- Result on requested command: `frames=538 elapsed=6.08s fps=88.54`.
- Learning: After model-side optimizations, response format selection became a large workflow-level bottleneck. Avoiding unnecessary RLE round-trips is safe for polygon outputs and preserves explicit RLE responses.

### RFDETR Preprocess Channel Swap After PIL Resize

- Hypothesis: In the common BGR numpy input path, RFDETR can resize the original contiguous BGR image with PIL and swap channels after `to_tensor()`. PIL resize is channel independent, so this should be equivalent to making an RGB numpy copy before PIL resize while avoiding that pre-resize copy.
- Change: RFDETR numpy preprocessing now tracks whether color channels need swapping and applies `[2, 1, 0]` on the tensor after PIL resize/to-tensor for 3-channel images.
- Correctness: Compared against the previous pre-PIL BGR-to-RGB path on 64 frames from `vehicles_312px.mp4`; preprocessed tensors matched exactly (`max_tensor_diff=0`), class IDs matched exactly, and max box delta was `0` px.
- Micro-result: isolated RFDETR preprocessing measured `2.16 ms/frame` over 240 frames after the change.
- Result on requested command: `frames=538 elapsed=6.04s fps=89.04`.
- Learning: Small host-side image copies are visible at this frame size. Exact channel-order transformations can be moved across per-channel PIL resize safely.
