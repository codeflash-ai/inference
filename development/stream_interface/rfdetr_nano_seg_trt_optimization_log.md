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

### Avoid Redundant Empty-Mask Scan Before Polygon Contours

- Hypothesis: `masks2poly()` and `masks2multipoly()` do an `np.any()` scan before `cv2.findContours()`, but OpenCV already returns no contours for empty masks. Removing the pre-scan should avoid a second full-mask traversal for non-empty masks.
- Change: Removed the explicit `np.any()` empty-mask fast path and let `mask2poly()` / `mask2multipoly()` handle empty contours.
- Correctness: Compared old and new polygon conversion on RFDETR dense masks from 40 frames; polygon arrays matched exactly. Empty-mask smoke check still returns `(0, 2)` `float32` polygon output.
- Micro-result: RFDETR dense mask polygon conversion improved from `0.18 ms/frame` to `0.11 ms/frame` over 240 frames.
- Result on requested command: `frames=538 elapsed=5.99s fps=89.84`.
- Test note: Attempted `PYTHONPATH=/app/inference_models pytest -q inference_models/tests/unit_tests/models/common/test_rle_utils.py tests/inference/unit_tests/models/test_rfdetr.py`, but collection failed before running tests with `ModuleNotFoundError: No module named 'tests.conftest'`.
- Learning: At high FPS, even small CPU mask scans are measurable; rely on the contour operation's empty output instead of scanning twice.

### Rejected: Pydantic model_construct For Polygon Responses

- Hypothesis: Bypassing Pydantic validation with `model_construct()` for instance segmentation response objects or polygon `Point` objects could reduce workflow response construction overhead.
- Change tested: No committed code change. Built RFDETR polygon response objects with `model_construct()` for all response models, and separately with only `Point.model_construct()`.
- Result: Both variants were slower in the local response-construction harness. Full construct measured `0.76 ms/frame` vs `0.57 ms/frame` for normal constructors; point-only construct measured `0.64 ms/frame` vs `0.54 ms/frame` and emitted Pydantic serializer warnings for NumPy scalar values.
- Learning: Pydantic construction is not the next profitable target in this form. Keep validated constructors and look elsewhere for workflow overhead.

### Rejected: Workflow v3 RLE Response For Local Conversion

- Hypothesis: Workflow v3 converts polygon predictions back into `sv.Detections` masks via `supervision.from_inference`; asking the local model response for RLE could avoid polygon rasterization in that conversion.
- Change tested: Temporary workflow v3 local request change only; passed `response_mask_format="rle"` to `InstanceSegmentationInferenceRequest`.
- Result on requested command: `frames=538 elapsed=6.25s fps=86.12`, slower than the dense polygon-response path.
- Learning: For this RFDETR-nano segmentation workflow, model-side RLE encoding plus local conversion is still more expensive than the dense-to-polygon path. Keep polygon responses as the local workflow default unless the caller explicitly asks for RLE.

### Current Exact Command Checkpoint

- Command: benchmark command above.
- Result on requested command: `frames=538 elapsed=5.85s fps=91.94`.
- Learning: The accumulated committed changes now improve the original `72.23 fps` baseline by about `27%` on the exact benchmark command.

### Avoid Eager Workflow Detection UUID Generation

- Hypothesis: `convert_inference_detections_batch_to_sv_detections(...)` calls `str(uuid.uuid4())` as a `dict.get(...)` default, so UUIDs are generated for every detection even when local model responses already include `detection_id`.
- Change: Generate a UUID only when `detection_id` is missing from a raw prediction, preserving present values including `None`.
- Correctness: Smoke test patched `uuid.uuid4` to raise and converted a response containing `detection_id="known-id"`; conversion preserved the ID without calling UUID generation.
- Result on requested command: `frames=538 elapsed=5.84s fps=92.12`.
- Learning: This is a small but safe workflow-level CPU reduction; RFDETR local response objects already pay the necessary `detection_id` creation cost.

### Pipeline Workflow CPU And GPU Work Across Frames

- Hypothesis: The optimized path was still serialized at the pipeline level: workflow CPU response construction for frame `N` had to finish before GPU preprocessing/inference/postprocess for frame `N+1` could start. Allowing multiple workflow frame batches in flight should overlap CPU-side workflow work with GPU work while preserving sink dispatch order.
- Change: `InferencePipeline.init_with_workflow(...)` now allows ordered in-flight workflow batches via `max_inflight_workflow_batches`, defaulting to `3`. Generic `init_with_custom_logic(...)` keeps the old single-worker default unless explicitly configured.
- Correctness: Compared sequential workflow execution (`max_inflight_workflow_batches=1`) against the new default on all 538 frames from `vehicles_312px.mp4`; frame order matched, class IDs matched exactly, and max box delta was `0` px.
- Tuning: `max_inflight=2` measured `121.24 fps`; `max_inflight=3` measured best at `141.01 fps`; `max_inflight=4` regressed to `135.86 fps`.
- Result on requested command: `frames=538 elapsed=3.82s fps=141.01`.
- Learning: Once per-frame CPU paths were reduced enough, cross-frame pipeline concurrency became the largest remaining gain. The exact benchmark improved from the original `72.23 fps` baseline to `141.01 fps` (+95%).

### Direct Local Workflow Detections And Remove Redundant RFDETR GPU Work

- Hypothesis: Nsight Systems on the pipelined path showed the largest visible GPU costs were RFDETR PyTorch postprocess kernels (`topk`, radix sort/indexing, mask resize), while workflow v3 still built polygon API responses and converted them back into `sv.Detections`. Avoiding redundant postprocess sorting, avoiding a one-image CUDA `stack`, and directly constructing local workflow `sv.Detections` from `inference_models` detections should reduce both GPU and CPU work.
- Change:
  - Removed redundant confidence re-sorts after `select_topk_predictions(...)`; `torch.topk(..., sorted=True)` already returns descending scores and the later boolean filters preserve order.
  - Added a single-image RFDETR preprocessing fast path that uses `unsqueeze(0)` instead of `torch.stack([tensor])` after the host-to-device copy.
  - Added a local workflow v3 fast path for `InferenceModelsInstanceSegmentationAdapter` when active learning is disabled. It runs adapter preprocess/predict, converts dense `InstanceDetections` directly to `sv.Detections`, attaches the same workflow metadata fields, and falls back to the existing response path otherwise.
- Correctness:
  - Old sorted dense postprocess vs new unsorted postprocess on all 538 frames: class IDs exact, boxes exact, confidences exact, dense masks exact.
  - Single-image `unsqueeze(0)` batch vs previous one-image `torch.stack(...)` equivalent on 128 frames: `max_tensor_diff=0`.
  - Existing workflow fallback vs new local fast path on all 538 frames: frame order matched, class IDs exact, and max box delta was `0` px.
- Micro-result: Dense RFDETR postprocess improved from `2.585 ms/frame` to `1.548 ms/frame` over 240 frames when synchronized around postprocess.
- Tuning: After the postprocess/preprocess changes, `max_inflight=3` remained best in the clean sink harness (`143.64 fps`), with `max_inflight=4` at `142.30 fps`, `5` at `137.45 fps`, and `6` at `130.28 fps`.
- Result on requested command: `frames=538 elapsed=3.46s fps=155.28`. In-memory prototype of the same direct workflow fast path measured `157.61 fps`.
- Learning: After cross-frame pipelining, the biggest remaining workflow overhead was the local API response round trip, not the sink conversion itself. Direct local `sv.Detections` construction preserves the benchmark's class/box contract and lets the pipeline spend more time feeding the serialized GPU path.

### Rejected: Thread-Local TRT CUDA Graph Replay

- Hypothesis: The three workflow workers still serialize RFDETR TRT graph replay on a model-level lock and shared inference stream. Giving each worker a thread-local inference stream and thread-local CUDA graph cache could let independent TensorRT execution contexts replay concurrently and increase GPU utilization.
- Change tested: Temporary code only; created thread-local inference streams and one-entry `TRTCudaGraphCache` instances, using the model lock only when a thread-local graph cache was empty.
- Result: Severe regression. The requested command only reached `[progress] frames=50 fps=2.09` during graph capture/warmup, so the run was stopped and the patch reverted.
- Learning: Per-worker CUDA graph capture/context setup is far too expensive for this path, and concurrent graph replay needs a pre-warmed context/graph pool rather than lazy per-thread capture in the hot pipeline.

### Rejected: Guarded Threshold-First RFDETR Instance Postprocess

- Hypothesis: The benchmark confidence threshold is high enough that only a few RFDETR query/class scores survive, so selecting scores above threshold first could avoid the global top-k/radix-sort work visible in Nsight Systems.
- Change tested: Temporary `common.py` helper for dense/RLE instance segmentation that selected candidates above the minimum threshold, used that path only when the candidate count was no larger than `num_queries`, sorted survivors by confidence, and otherwise fell back to the existing `select_topk_predictions(...)` semantics.
- Correctness: Compared the temporary selector against the previous top-k selector on raw TRT outputs for all 538 frames from `vehicles_312px.mp4`: detection counts matched, class IDs matched exactly, max box delta was `0` px, confidences matched exactly, and dense masks matched exactly.
- Result on requested command: Two exact passes measured `frames=538 elapsed=3.49s fps=154.33` and `frames=538 elapsed=3.49s fps=153.98`, below the current best committed path.
- Learning: Even with exact outputs, the extra CUDA `nonzero`/candidate-selection work and synchronization risk do not beat PyTorch's current global top-k path end-to-end. A profitable version likely needs a fused CUDA/Triton kernel that thresholds, remaps, compacts, and gathers boxes/masks in one pass without host-visible candidate counting.

### Fused RFDETR Dense Postprocess And Pipeline Rebalance

- Hypothesis: Nsight Systems still showed RFDETR PyTorch postprocess kernels (`topk`, radix sort/indexing, mask resize) after the direct workflow fast path. Fusing top-score selection, class remapping, box decode, and workflow mask resize should reduce postprocess kernel launch overhead and shift the best workflow pipeline depth.
- Change:
  - Added a gated Triton RFDETR dense postprocess path for the benchmark-shape case: scalar threshold, one image, no padding, no static crop, dense masks. The general PyTorch path remains the fallback.
  - The fused selector walks global scores in descending order only until the next score falls below threshold, preserving the old top-k/filter semantics without always materializing 100 selections.
  - For the local workflow RFDETR TRT fast path, the selected count stays on GPU through a Triton mask resize and is copied only at the existing NumPy conversion boundary; public model postprocess still returns exact-sized tensors by default.
  - Changed the workflow default `max_inflight_workflow_batches` from `3` to `2` after retuning.
- Correctness:
  - Default postprocess vs previous PyTorch selector on all 538 frames: detection counts matched, class IDs exact, max box delta `0` px, confidences exact, dense masks exact.
  - Deferred workflow postprocess mode vs default exact-sized postprocess on all 538 frames: detection counts matched, class IDs exact, max box delta `0` px, dense masks exact.
- Micro-result: Synchronized dense postprocess over 240 cached raw TRT outputs improved from `2.055 ms/frame` to `0.408 ms/frame`.
- Pipeline tuning: With the deferred fused path, depth `3` measured `frames=538 elapsed=3.44s fps=156.61`; depth `2` measured `frames=538 elapsed=3.10s fps=173.39`.
- Result on requested command: `frames=538 elapsed=3.07s fps=175.12`.
- Learning: Once postprocess count synchronization and mask resize were moved to the workflow conversion boundary, the optimal pipeline depth dropped from `3` to `2`; the third in-flight worker became extra contention instead of useful overlap.
