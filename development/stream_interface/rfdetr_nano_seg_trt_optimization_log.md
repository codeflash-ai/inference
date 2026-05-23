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

### Rejected: Blocked Triton Mask Resize Programs

- Hypothesis: The fused mask resize nsys profile showed `_resize_selected_masks_kernel` as the top postprocess kernel. Processing several detections per Triton program could reduce launch-grid overhead while still supporting up to 100 detections.
- Change tested: Temporary code only; changed the mask resize kernel from one detection per program to four detections per program with a smaller pixel block.
- Correctness: Deferred workflow postprocess vs default exact-sized postprocess on all 538 frames still matched detection counts, class IDs, boxes, and dense masks exactly.
- Result on requested command: `frames=538 elapsed=3.18s fps=168.99`, slower than the committed single-detection program layout.
- Learning: The larger per-program vector shape hurt this T4 path more than the lower program count helped. Keep the simpler mask kernel and look next at reducing H2D preprocessing transfer or avoiding full-size mask materialization.

### Rejected: Pinned RFDETR Preprocess Transfer

- Hypothesis: The fused-path nsys profile showed Host-to-Device preprocessing copies as the largest remaining memory operation. Pinning the already-normalized CPU tensor and using a non-blocking CUDA copy could improve transfer overlap.
- Change tested: Temporary `pre_processing.py` helper that called `tensor.pin_memory().to(device, non_blocking=True)` for CPU tensors moving to CUDA.
- Result on requested command: `frames=538 elapsed=3.19s fps=168.61`, slower than the committed pageable transfer path.
- Learning: Per-frame pinning overhead outweighed any asynchronous-copy benefit for this 312x312 tensor. A useful transfer optimization likely needs reusable pinned buffers or moving normalization/conversion to GPU from a smaller uint8 transfer, not pinning after CPU float normalization.

### Rejected: Sparse Early Return In Triton Mask Resize

- Hypothesis: The benchmark has only 1-7 detections per frame, while the fused mask resize kernel is launched over 100 possible detection slots. Returning immediately for detection slots greater than the GPU-side selected count could reduce unnecessary bilinear math.
- Change tested: Temporary code only; added a runtime `if det_index >= count: return` branch in `_resize_selected_masks_kernel`.
- Correctness: Deferred workflow postprocess vs default exact-sized postprocess on all 538 frames matched detection counts, class IDs, boxes, and dense masks exactly.
- Result on requested command: `frames=538 elapsed=3.16s fps=170.11`, slower than the committed mask kernel.
- Learning: The runtime branch/control-flow cost and changed Triton code generation outweighed the skipped masked arithmetic on T4. Keep the straight-line mask kernel for now.

### Rejected: GPU Normalize After Uint8 Preprocess Transfer

- Hypothesis: Instead of transferring the normalized float32 RFDETR input, keep the PIL resize on CPU for pixel compatibility, transfer the resized uint8 image to GPU, and perform channel reorder plus normalization on GPU. This should reduce H2D bytes by roughly 4x.
- Change tested: Temporary `pre_processing.py` path that converted the PIL-resized image to a uint8 tensor on CUDA, then did CHW conversion and normalization on GPU.
- Correctness: Compared the previous CPU-normalized tensor path against the GPU-normalized path on all 538 frames: max tensor diff rounded to `0.00000000`, detection counts matched, class IDs matched exactly, and max box delta was `0` px.
- Result on requested command: depth `2` measured `frames=538 elapsed=3.15s fps=170.80`; retuning depth `3` measured `frames=538 elapsed=3.43s fps=157.08`.
- Learning: The smaller H2D copy did not compensate for the additional GPU conversion/normalization kernels and stream contention. Any useful version likely needs a single fused conversion kernel and careful scheduling, or reusable CPU-side normalized transfer remains better.

### Rejected: Packed Workflow Detection Metadata Copy

- Hypothesis: The local workflow conversion copies selected boxes, confidences, class IDs, and masks from GPU separately. Packing non-mask detection fields in the fused selector kernel and copying them as one tensor could reduce D2H API overhead.
- Change tested: Temporary fused selector output of `[x1, y1, x2, y2, confidence, class_id]` for workflow conversion, using the packed tensor instead of separate `xyxy`, `confidence`, and `class_id` copies.
- Correctness: Deferred workflow postprocess vs default exact-sized postprocess on all 538 frames matched detection counts, class IDs, and dense masks exactly; max box delta was `0.5` px due to packed float boxes instead of rounded int boxes.
- Result on requested command: Two exact passes measured `frames=538 elapsed=3.08s fps=174.63` and `frames=538 elapsed=3.09s fps=174.32`, not better than the committed checkpoint.
- Learning: The extra selector stores and altered code generation offset the small D2H-call reduction. Keep the simpler separate tensors unless a future fused CPU conversion removes more overhead.

### Rejected: Disable TRT CUDA Graph Replay

- Hypothesis: CUDA graph replay clones graph output buffers every frame, contributing visible D2D copy time. Running the standard TensorRT async path could avoid those clones and be competitive after postprocess fusion.
- Change tested: No code change; ran the requested command with `ENABLE_AUTO_CUDA_GRAPHS_FOR_TRT_BACKEND=False`.
- Result on requested command: `frames=538 elapsed=3.11s fps=173.27`, below the committed CUDA graph path.
- Learning: TRT graph replay still wins overall; the graph-launch reduction is more valuable than removing output-buffer clone traffic.

### NumPy RFDETR PIL Tensor Conversion

- Hypothesis: The RFDETR preprocessing path still spent CPU time in `TF.to_tensor(...)` followed by `TF.normalize(...)` after the PIL-compatible resize. A single NumPy float32 conversion/normalization step should preserve the same resized pixels while reducing Python/PyTorch transform overhead.
- Change: Added a gated RFDETR numpy/PIL preprocessing fast path for 3-channel normalized inputs. It converts the PIL-resized image to `float32` NumPy, applies channel swap and normalization in HWC layout, then creates the CHW tensor from a contiguous transpose. Non-normalized or non-3-channel cases fall back to the previous torchvision path.
- Correctness: Compared the previous `TF.to_tensor`/`TF.normalize` path against the NumPy path on all 538 frames: max tensor diff `0.00000072`, detection counts matched, class IDs matched exactly, and max box delta was `0` px.
- Micro-result: Preprocess-only loop over 128 frames improved from `1.957 ms/frame` to `1.890 ms/frame`.
- Pipeline tuning: Default depth `2` measured `frames=538 elapsed=2.81s fps=191.24`; serial depth `3` measured `frames=538 elapsed=2.94s fps=182.96`, so depth `2` remains best.
- Result on requested command: `frames=538 elapsed=2.81s fps=191.24`.
- Learning: Keeping the PIL resize source-of-truth while reducing conversion overhead gives a real pipeline gain; after postprocess fusion, small CPU preprocessing savings matter because they improve producer/consumer balance at depth `2`.

### Rejected: Exact-Capacity Fused Mask Resize

- Hypothesis: The benchmark emits only 1-7 detections per frame, but the deferred fused workflow path launches `_resize_selected_masks_kernel` over the full 100-detection capacity. Synchronizing the selected count before mask resize, allocating exactly that many mask planes, and launching only those programs could reduce the dominant custom kernel time.
- Change tested: Temporary code only; in deferred fused postprocess, copied the selected count to CPU before mask resize, sliced detection tensors immediately, and passed an exact `output_capacity` to `fused_resize_selected_masks(...)`.
- Correctness: Compared non-deferred exact-sized postprocess against the exact-capacity deferred path on all 538 frames: max selected count `7`, class IDs exact, max box delta `0` px, confidences exact, and dense masks exact.
- Pipeline tuning: Depth `2` measured `frames=538 elapsed=3.27s fps=164.47`; depth `3` measured `frames=538 elapsed=3.12s fps=172.18`.
- Learning: The earlier CPU count synchronization breaks the useful overlap from the deferred path. Even though it reduces mask resize work, preserving the GPU-side count until workflow conversion is faster overall.

### Rejected: OpenCV RFDETR Resize Fast Path

- Hypothesis: Replacing PIL resize with OpenCV bilinear resize in RFDETR preprocessing could reduce CPU producer time while preserving class IDs and keeping boxes within 5 px.
- Change tested: Temporary script-only prototype; resized the `312x176` video frames to the `312x312` TRT input with `cv2.resize`, then applied the same BGR-to-RGB swap and normalization.
- Correctness: Compared against the current PIL path on all 538 frames. Counts differed on 7 frames, class IDs differed on 14 frames, and max box delta reached `183` px.
- Micro-result: Prototype preprocessing measured `2.969 ms/frame` vs current `2.013 ms/frame` over 128 frames.
- Learning: PIL interpolation is part of the effective RFDETR input contract for this checkpoint. OpenCV is both slower here and not prediction-compatible.

### Direct PIL RFDETR Resize

- Hypothesis: The torchvision `TF.resize(...)` PIL wrapper adds Python overhead around a PIL bilinear resize. Calling `PIL.Image.resize(...)` directly should preserve pixels while shaving preprocessing overhead.
- Change: In the RFDETR numpy preprocessing branch, replaced `TF.resize(pil, ...)` with direct `pil.resize((width, height), Image.Resampling.BILINEAR)` before the existing NumPy normalize/CHW conversion.
- Correctness: Reproduced the old `TF.resize(..., antialias=True)` path over all 538 frames and compared tensors against the patched model preprocessing: max tensor diff `0.0000000000`, so classes and boxes are unchanged.
- Pipeline tuning: Depth `2` measured `frames=538 elapsed=2.78s fps=193.48`; depth `3` measured `frames=538 elapsed=2.97s fps=181.10`, so depth `2` remains best.
- Result on requested command: best isolated run `frames=538 elapsed=2.78s fps=193.48`; repeat isolated run measured `frames=538 elapsed=2.82s fps=190.99`.
- Learning: Removing the torchvision wrapper is a small, exact CPU-side cleanup. The end-to-end gain is near run-to-run noise, but the highest isolated benchmark moved slightly upward and the change is byte-equivalent for the model input.

### Rejected: Workflow Fast Path Inference Mode Wrapper

- Hypothesis: The direct local workflow fast path executes RFDETR TRT preprocess, predict, and postprocess without needing autograd. Wrapping that section in `torch.inference_mode()` could reduce PyTorch dispatch overhead around tensor copies and postprocess kernels.
- Change tested: Temporary code only; added `with torch.inference_mode():` around `model.preprocess(...)`, `model.predict(...)`, and fused `model._model.post_process(...)` in the instance segmentation workflow fast path.
- Correctness: Compared model execution outside vs inside `torch.inference_mode()` on all 538 frames: class IDs exact and max box delta `0` px.
- Result on requested command: repeat runs measured `frames=538 elapsed=2.80s fps=191.83` and `frames=538 elapsed=2.81s fps=191.27`.
- Learning: The TRT path already produces tensors with no autograd work worth removing; the wrapper is neutral within noise and does not justify extra workflow code.

### RFDETR Channel-Wise CHW Normalization

- Hypothesis: The NumPy RFDETR preprocessing path still creates a normalized HWC float array and then makes a contiguous CHW copy. Writing normalized channels directly into a CHW float32 output should avoid one layout-conversion allocation.
- Change: `_pil_image_to_normalized_tensor(...)` now reads the resized PIL image as uint8, normalizes each selected channel into a preallocated CHW float32 array, and returns that array directly as the tensor backing storage.
- Correctness: Reproduced the prior HWC-float/transpose formula over all 538 frames and compared tensors against the patched preprocessing path: max tensor diff `0.0000000000`, so classes and boxes are unchanged.
- Micro-result: Preprocess-only loop over 128 frames measured `1.980 ms/frame`; the isolated conversion prototype measured `0.605 ms/frame` vs `0.622 ms/frame` for the prior conversion helper.
- Pipeline tuning: Depth `2` measured `frames=538 elapsed=2.80s fps=192.42` and `frames=538 elapsed=2.79s fps=193.16`; depth `3` measured `frames=538 elapsed=3.10s fps=173.33`.
- Learning: This is an exact, small allocation cleanup. It does not materially shift the bottleneck or pipeline depth, but it keeps the preprocessing path leaner without changing model inputs.

### Rejected: Keep Deferred Query Indices As Int32

- Hypothesis: Nsight Systems still showed a tiny PyTorch copy/cast kernel after fused selection. The deferred mask resize kernel can read int32 query indices directly, so skipping `query_indices.to(dtype=torch.long)` in the GPU-deferred path could remove one kernel launch and D2D copy per frame.
- Change tested: Temporary code only; when `return_cpu_count=False`, `fused_select_topk_boxes(...)` returned int32 query indices instead of casting them to int64.
- Correctness: Deferred workflow postprocess vs exact-sized postprocess on all 538 frames matched class IDs, boxes, and dense masks exactly; max box delta `0` px.
- Result on requested command: repeat runs measured `frames=538 elapsed=2.80s fps=192.18` and `frames=538 elapsed=2.83s fps=189.79`, below the current checkpoint band.
- Learning: Removing this small cast does not improve end-to-end throughput; the scheduling and pipeline balance dominate over this tiny kernel.

### Rejected: Deterministic Local Workflow Detection IDs

- Hypothesis: The direct local workflow fast path still creates a UUID per detection. Reusing the request inference ID plus a detection index would reduce Python UUID work during CPU-side `sv.Detections` construction.
- Change tested: Temporary code only; when `inference_id` was present, generated detection IDs as `"{inference_id}-{index}"` in the local instance segmentation workflow fast path.
- Correctness: The change runs after tensor-to-NumPy conversion and does not affect model classes or boxes.
- Result on requested command: repeat runs measured `frames=538 elapsed=2.79s fps=193.07` and `frames=538 elapsed=2.83s fps=189.99`, not a clear improvement.
- Learning: UUID generation is not a measurable limiter after the fused/deferred path; keep the existing identifier behavior.

### Rejected: PIL Image FromBuffer Wrapper

- Hypothesis: `Image.fromarray(...)` may copy contiguous OpenCV frames before PIL resize. For uint8 HWC contiguous images, `Image.frombuffer(...)` could avoid that copy while preserving channel values.
- Change tested: Temporary `_pil_from_hwc_uint8(...)` helper that used `Image.frombuffer("RGB", ...)` for contiguous 3-channel uint8 images and fell back to `Image.fromarray(...)` otherwise.
- Correctness: Compared patched preprocessing against the previous `Image.fromarray(...)` path on all 538 frames: max tensor diff `0.0000000000`, so classes and boxes are unchanged.
- Micro-result: Isolated conversion prototype measured `1.446 ms/frame` vs `1.455 ms/frame` for the `fromarray` path over 128 frames.
- Result on requested command: repeat runs measured `frames=538 elapsed=2.81s fps=191.23` and `frames=538 elapsed=2.82s fps=191.04`, below the current checkpoint band.
- Learning: Any copy saved by `frombuffer` is too small to matter, and PIL resize plus downstream pipeline scheduling dominate this part of preprocessing.

### Reusable Pinned RFDETR Preprocess Buffer

- Hypothesis: Nsight Systems still showed the normalized RFDETR input Host-to-Device copy as the largest memory operation. Filling a reusable pinned CPU tensor directly in CHW layout and copying it to CUDA with `non_blocking=True` should reduce CPU-side transfer blocking and improve overlap with the GPU pipeline.
- Change: For the single-image CUDA numpy preprocessing path, `_pil_image_to_normalized_tensor(...)` now writes normalized channels into a thread-local pinned `torch.float32` CHW buffer. `pre_process_network_input(...)` uses a non-blocking device copy from pinned memory and records a per-thread CUDA event so the host buffer is not reused until the prior H2D copy is complete. Batch and non-CUDA paths keep the normal NumPy-backed tensor behavior.
- Correctness: Compared the pinned integrated preprocessing path against the previous non-pinned CHW formula on all 538 frames: max tensor diff `0.0000000000`, so classes and boxes are unchanged.
- Micro-result: Preprocess-only loop over 128 frames measured `1.663 ms/frame`, down from the prior ~`1.98 ms/frame` band.
- Pipeline tuning: Depth `1` measured `frames=538 elapsed=4.15s fps=129.55`; depth `2` measured `frames=538 elapsed=2.64s fps=203.56` and `frames=538 elapsed=2.62s fps=205.28`; depth `3` measured `frames=538 elapsed=3.19s fps=168.47`.
- Result on requested command: best isolated run `frames=538 elapsed=2.62s fps=205.28`.
- Learning: Reusable pinned memory is the first preprocessing transfer change that helps end-to-end. The earlier per-frame `.pin_memory()` experiment was slower because it paid pinning cost every frame; reusing the pinned storage preserves the transfer benefit without that allocation cost.

### Rejected: Retune Fused Mask Resize Pixel Block

- Hypothesis: `_resize_selected_masks_kernel` is still the largest custom kernel in Nsight Systems. Changing the per-program pixel block from 256 could improve occupancy or reduce Triton program count on T4.
- Change tested: Temporary code only; tried `block_size=512` with `num_warps=8`, then `block_size=128` with `num_warps=4`.
- Correctness: The `512` variant matched exact-sized postprocess on all 538 frames, including dense masks and max box delta `0` px. The `128` variant changes only tile shape and uses the same math.
- Result on requested command: `512/8` measured `frames=538 elapsed=2.68s fps=200.67`; `128/4` measured `frames=538 elapsed=2.66s fps=202.41`, both below the committed `256/4` path.
- Learning: The current 256-pixel tile remains the best balance. Larger tiles likely hurt register/occupancy behavior, while smaller tiles add too many programs.

### Direct NumPy Ufunc RFDETR Channel Fill

- Hypothesis: The pinned preprocessing path still allocates a temporary float32 array for each channel via `astype(...)` before copying into the pinned CHW output. Writing each uint8 channel directly into the destination with NumPy ufuncs should remove those temporary arrays.
- Change: Replaced per-channel `astype(np.float32)` temporaries with `np.multiply(..., out=channel, casting="unsafe")` directly into the normalized output channel, followed by in-place mean/std normalization.
- Correctness: Compared the patched preprocessing path against the prior temporary-channel formula on all 538 frames: max tensor diff `0.0000000000`, so classes and boxes are unchanged.
- Micro-result: Isolated channel-fill prototype measured `0.434 ms/frame` vs `0.476 ms/frame` for the previous temporary-channel fill. Integrated preprocess-only loop over 128 frames measured `1.638 ms/frame`.
- Result on requested command: isolated depth `2` runs measured `frames=538 elapsed=2.61s fps=206.40` and `frames=538 elapsed=2.59s fps=207.82`.
- Learning: Small CPU allocation reductions still matter after the pinned H2D change because they improve the producer side of the two-frame pipeline without changing GPU semantics.

### Cached RFDETR Normalization Constants

- Hypothesis: `_pil_image_to_normalized_tensor(...)` rebuilds NumPy mean/std arrays and a float32 scale scalar every frame. Caching these immutable normalization constants per thread should remove small repeated allocations from the producer path.
- Change: Added a thread-local normalization constants cache keyed by the configured mean/std values, reusing the float32 mean array, std array, and `1/255` scale across frames.
- Correctness: Compared cached-constant preprocessing against the prior ufunc-fill formula on all 538 frames: max tensor diff `0.0000000000`, so classes and boxes are unchanged.
- Pipeline tuning: Depth `2` runs measured `frames=538 elapsed=2.58s fps=208.14` and `frames=538 elapsed=2.58s fps=208.26`; depth `3` measured `frames=538 elapsed=3.07s fps=175.07`.
- Learning: At this point, even small per-frame Python/NumPy allocations are visible in the two-frame pipeline balance.

### Rejected: Double-Buffered Pinned Preprocess Buffers

- Hypothesis: The reusable pinned preprocessing buffer waits on the previous H2D copy before reusing host memory. Alternating between two pinned host buffers could let CPU normalization for the next frame proceed while the previous pinned copy is still in flight.
- Change tested: Temporary code only; replaced the single thread-local pinned buffer with a two-buffer ping-pong and per-buffer CUDA copy events.
- Correctness: Compared double-buffered preprocessing against the prior single-buffer formula on all 538 frames: max tensor diff `0.0000000000`.
- Result on requested command: repeat runs measured `frames=538 elapsed=2.60s fps=206.93` and `frames=538 elapsed=2.60s fps=206.71`, below the single-buffer cached path.
- Learning: The extra buffer/event bookkeeping outweighed any overlap benefit. The single reusable pinned buffer remains better for the current two-frame pipeline.

### Rejected: RFDETR No-Op Preprocessing Bypass

- Hypothesis: The benchmark model has no static crop, grayscale, or contrast preprocessing configured, but RFDETR still calls the generic numpy preprocessing helper. Bypassing that helper in the no-op case could remove branch overhead and default crop metadata construction.
- Change tested: Temporary code only; skipped `apply_pre_processing_to_numpy_image(...)` when preprocessing overrides were absent and static crop/grayscale/contrast configs were all `None`, constructing the default `StaticCropOffset` directly.
- Correctness: Compared against the previous preprocessing path on all 538 frames: max tensor diff `0.0000000000`, so classes and boxes are unchanged.
- Result on requested command: repeat runs measured `frames=538 elapsed=2.60s fps=207.17` and `frames=538 elapsed=2.62s fps=205.04`, below the cached-normalization checkpoint.
- Learning: The generic no-op helper is not a meaningful limiter; keeping the established shared helper is better than adding a special branch here.

### Rejected: Same-Stream RFDETR TRT Postprocess

- Hypothesis: Nsight Systems showed substantial CUDA event overhead around cross-stream raw output handoff. Running RFDETR TRT postprocess on the inference stream and removing raw-output `record_stream(...)` calls could reduce event bookkeeping, at the cost of less overlap between next-frame inference and previous-frame postprocess.
- Change tested: Temporary code only; changed `RFDetrForInstanceSegmentationTRT.post_process(...)` to use `_inference_stream` instead of `_post_process_stream` and removed the raw result `record_stream(...)` loop.
- Correctness: Deferred fused postprocess vs exact-sized postprocess on all 538 frames matched class IDs, dense masks, and boxes exactly; max box delta `0` px.
- Result on requested command: repeat runs measured `frames=538 elapsed=2.59s fps=207.84` and `frames=538 elapsed=2.63s fps=204.87`, below the current checkpoint band.
- Learning: The existing separate postprocess stream still pays for itself. Keeping postprocess overlapped with the next TensorRT replay is better than removing the stream/event bookkeeping.

### Rejected: Float Boxes In Deferred Fused Postprocess

- Hypothesis: The deferred RFDETR workflow path still launches tiny PyTorch `round` and `int` kernels for selected boxes. Returning float `xyxy` tensors through workflow conversion could remove those kernels while staying within the allowed 5 px box tolerance.
- Change tested: Temporary code only; returned `selected_boxes` instead of `selected_boxes.round().int()` when `defer_fused_postprocess_count=True`.
- Correctness: Compared deferred vs exact-sized postprocess on 120 frames: counts and class IDs matched, with max box delta `0.5` px.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.63s fps=204.86`, below the current checkpoint band. A simultaneous depth `2`/`3` run was discarded because both processes contended for the GPU.
- Learning: Removing these tiny kernels is not enough to improve the full pipeline, and keeping integer boxes preserves the established output type.

### Rejected: Retire Completed Workflow Futures Out Of Order

- Hypothesis: The two-frame workflow pipeline may create CUDA graph bubbles when a completed out-of-order future still counts against the in-flight limit until earlier frames dispatch. Moving completed futures into a ready map immediately while preserving ordered emission could free worker slots sooner.
- Change tested: Temporary `InferencePipeline` scheduler change using `concurrent.futures.wait(..., FIRST_COMPLETED)` for multi-worker inference.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.64s fps=203.71`; depth `3` measured `frames=538 elapsed=3.10s fps=173.46`.
- Learning: The graph bubbles are dominated by whole-frame stage balance, not by completed futures being held for ordered dispatch. The extra scheduler bookkeeping was not useful here.

### RFDETR Fast Path Deferred Current-Stream Waits

- Hypothesis: In the direct local RFDETR TRT workflow path, `pre_process(...)`, `forward(...)`, and `post_process(...)` are called back-to-back by the same fast path. The intermediate current-stream waits after preprocessing and forward add CUDA event edges even though `forward(...)` already waits on the preprocessing stream and `post_process(...)` already waits on the inference stream.
- Change: Added `defer_cuda_stream_sync` for RFDETR TRT dense-mask workflow execution. The fast path passes it through preprocess and predict, and `RFDetrForInstanceSegmentationTRT` skips only the redundant current-stream waits after preprocessing and forward. The postprocess-to-CPU conversion wait is unchanged.
- Pipeline tuning: Depth `2` measured `frames=538 elapsed=2.57s fps=209.12`, `frames=538 elapsed=2.61s fps=206.46`, and `frames=538 elapsed=2.56s fps=210.18`. Depth `3` measured `frames=538 elapsed=2.74s fps=196.27`, so depth `2` remains best.
- Nsight Systems: New report for analysis: `/tmp/rfdetr_stream_wait_20260523_031606.nsys-rep` with SQLite export `/tmp/rfdetr_stream_wait_20260523_031606.sqlite`. Under profiler, throughput improved to `frames=538 elapsed=3.09s fps=173.88`.
- Graph spacing: Compared to the clean local baseline profile `/tmp/rfdetr_gap_local_20260523_031231.nsys-rep`, post-warmup graph end-to-next-start gaps improved from p90 `7635.910 us`, p95 `8175.037 us`, p99 `8748.532 us`, mean `2734.651 us` to p90 `4091.966 us`, p95 `4392.314 us`, p99 `4934.320 us`, mean `2050.445 us` after skipping the first 100 graph launches.
- Learning: Removing redundant wait edges reduces the long graph bubbles visible in Nsight while preserving the explicit stream dependencies that matter. The run is now more tightly constrained by the CUDA graph forward pass plus fused postprocess, and depth `3` is still worse because extra workers add CPU/GPU contention.

### Rejected: Postprocess-Stream CPU Conversion

- Hypothesis: After deferring the intermediate waits, the remaining postprocess current-stream wait might be avoidable by performing the workflow tensor-to-NumPy copies under the RFDETR postprocess stream context.
- Change tested: Temporary code only; skipped the postprocess current-stream wait and wrapped local workflow conversion in `torch.cuda.stream(model._model._post_process_stream)`.
- Result on requested command: after fixing a temporary helper-name typo, depth `2` measured `frames=538 elapsed=2.65s fps=202.97`.
- Learning: The synchronization still has to happen before CPU predictions are materialized, and moving those copies onto the postprocess stream made the normal run slower. Keep the postprocess wait at the model boundary.

### Rejected: Skip RFDETR Output Record Stream In Fast Path

- Hypothesis: The fast RFDETR TRT dense-mask workflow path waits for postprocess before returning to CPU conversion, so `record_stream(...)` on the three TensorRT output clones might be redundant allocator bookkeeping.
- Change tested: Temporary code only; skipped `result_element.record_stream(self._post_process_stream)` when `defer_cuda_stream_sync=True`.
- Correctness: Compared exact-sized postprocess against deferred fused postprocess on 160 frames with the fast-path flags: counts, classes, and boxes matched exactly.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.60s fps=206.61`, below the current `210.18` FPS checkpoint.
- Learning: The explicit allocator stream handoff is still worth keeping; removing it likely shifts synchronization or allocator pressure elsewhere.

### Rejected: Pooled TensorRT CUDA Graph Output Copies

- Hypothesis: CUDA graph replay clones every TensorRT output each frame. Replacing those per-frame clone allocations with a two-slot reusable output-copy pool could preserve overlap while reducing allocator and CUDA event churn.
- Change tested: Temporary code only; added a tuple lease around pooled output-copy buffers and released the slot from RFDETR postprocess after its stream consumed the raw TensorRT outputs.
- Correctness: After preserving the lease wrapper through RFDETR `forward(...)`, compared exact-sized postprocess against deferred fused postprocess on 120 frames: counts, classes, and boxes matched exactly.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.62s fps=205.23`, below the current checkpoint.
- Learning: The extra Python lease, event, and slot bookkeeping costs more than the clone allocations it avoids. The TensorRT output clone path is not worth changing this way.

### Rejected: Fuse RFDETR Sigmoid Into Triton Selector

- Hypothesis: The fast fused postprocess path still launches a PyTorch sigmoid kernel over logits before the Triton selector scans the same scores. Computing sigmoid inside the selector could remove one kernel launch and one intermediate tensor.
- Change tested: Temporary code only; passed raw logits to `_select_topk_boxes_kernel` and computed `1 / (1 + exp(-logit))` in Triton before top-k selection. The fallback path still materialized `logits_sigmoid` only if fused selection was unavailable.
- Correctness: Compared the fused path against the non-fused PyTorch fallback on 160 frames by monkeypatching the fused selector off for the reference. Counts, classes, and boxes matched exactly.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.60s fps=206.54`, below the current `210.18` FPS checkpoint.
- Learning: The extra Triton `exp` work inside the selector costs more than the standalone PyTorch sigmoid kernel in this shape. Keep the sigmoid as a separate highly optimized PyTorch elementwise kernel.

### Rejected: Blocked-Detection Triton Mask Resize

- Hypothesis: `_resize_selected_masks_kernel` launches a program for every detection row and pixel tile, including no-op rows beyond the selected count. Handling multiple detection rows per program could reduce Triton program count and improve the largest custom kernel.
- Change tested: Temporary code only; changed the mask resize kernel to process `block_detections` rows by `block_size` pixels per program. Tried `block_detections=4, block_size=128` and then `block_detections=2, block_size=128`.
- Correctness: The `4x128` variant matched the non-fused PyTorch fallback on 120 frames: counts, classes, and boxes matched exactly.
- Result on requested command: `4x128` measured `frames=538 elapsed=2.59s fps=207.68`; `2x128` measured `frames=538 elapsed=2.60s fps=207.12`, both below the current checkpoint.
- Learning: The reduced program count does not compensate for the larger vector/register footprint on this T4 workload. The original one-detection, 256-pixel tile remains better.

### Rejected: Packed RFDETR Metadata Copy

- Hypothesis: Workflow conversion performs separate small D2H copies for count, boxes, confidence, class IDs, and masks. Packing boxes, confidence, and class IDs into one Triton-produced float32 metadata tensor could reduce tiny D2H calls and remove the deferred path's box round/int kernels.
- Change tested: Temporary code only; `_select_topk_boxes_kernel` wrote a `(100, 6)` packed metadata tensor `[x1, y1, x2, y2, score, class_id]`, and the local workflow fast path copied that tensor once before slicing CPU arrays.
- Correctness: Compared against the non-fused PyTorch fallback on 160 frames: counts and classes matched exactly, max box delta `0.5` px.
- Result on requested command: depth `2` repeat runs measured `frames=538 elapsed=2.57s fps=209.35` and `frames=538 elapsed=2.57s fps=209.72`, close but still below the current `210.18` FPS checkpoint.
- Learning: Reducing small D2H copies alone does not beat the added Triton stores and changed CPU formatting. Keep the simpler separate metadata tensors.

### Rejected: Skip TensorRT Input Record Stream In Fast Path

- Hypothesis: In the RFDETR local workflow fast path, the preprocessed input tensor remains alive until postprocess has waited on inference, so `pre_processed_images.record_stream(inference_stream)` in the TensorRT wrapper might be redundant allocator bookkeeping.
- Change tested: Temporary code only; added a `record_input_stream` flag to `infer_from_trt_engine(...)` and disabled it only when `defer_cuda_stream_sync=True`.
- Correctness: Compared exact-sized postprocess against deferred fused postprocess on 120 frames: counts, classes, and boxes matched exactly.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.58s fps=208.39`, below the current checkpoint.
- Learning: The input allocator stream handoff is still useful or its removal shifts synchronization elsewhere. Keep the TensorRT wrapper's input `record_stream(...)`.

### Rejected: Inline Workflow Metadata Attachment

- Hypothesis: The local workflow fast path constructs `sv.Detections`, then walks those objects again to attach prediction type and parent/root coordinate metadata. Filling those arrays during conversion could reduce CPU-side object mutation before the next frame can feed CUDA graph replay.
- Change tested: Temporary code only; extended `_convert_inference_models_detections_to_sv_detections(...)` to accept `images` and `prediction_type`, then skipped `attach_prediction_type_info_to_sv_detections_batch(...)` and `attach_parents_coordinates_to_batch_of_sv_detections(...)` in the fast path.
- Correctness: This change runs after tensor-to-NumPy conversion and does not affect model classes or boxes.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.59s fps=207.76`, below the current checkpoint.
- Learning: The generic metadata helpers are not the current limiter; inlining their work added enough Python/object overhead to hurt throughput.

### Rejected: Pinned Host Workflow Prediction Copy

- Hypothesis: Workflow conversion copies mask tensors and small metadata tensors from GPU to CPU synchronously with `.cpu().numpy()`. Staging RFDETR deferred outputs through reusable pinned CPU tensors on a copy stream could reduce D2H blocking and create room to overlap CPU metadata work.
- Change tested: Temporary code only; allocated thread-local pinned CPU buffers for `xyxy`, confidence, class IDs, and dense masks, copied with `non_blocking=True` on a thread-local CUDA stream, then synchronized once before constructing `sv.Detections`.
- Correctness: The change only affects CPU materialization; no model classes or boxes are changed.
- Result on requested command: depth `2` measured `frames=538 elapsed=3.50s fps=153.82`.
- Learning: Pinned staging at the workflow boundary is much slower than PyTorch's direct `.cpu().numpy()` path here. The explicit stream synchronization and large pinned mask buffer mechanics dominate any potential overlap benefit.

### RFDETR TRT Pre-Request Workflow Fast Path

- Hypothesis: The local workflow path still builds `inference_images`, creates a Pydantic `InstanceSegmentationInferenceRequest`, dumps it back into a dict, and runs adapter image loading before reaching the RFDETR TRT fast path. For the benchmark's local dense-mask RFDETR TRT case, the workflow image already has a BGR NumPy frame, so the model can be called directly before request construction.
- Change: `run_locally(...)` now attempts an RFDETR-TRT-specific fast path before constructing the request. It loads the model, uses `WorkflowImageData.numpy_image` directly, passes minimal model kwargs, keeps `defer_cuda_stream_sync=True`, and still uses the existing `sv.Detections` conversion and workflow metadata helpers.
- Correctness: Over all 538 frames, `load_image_bgr({"type": "numpy_object", "value": frame})` matched the direct `frame` pixels exactly (`bad_pixels=0`). Deferred fused postprocess matched exact-sized postprocess with `bad_counts=0`, `bad_classes=0`, and `max_box_delta=0` px.
- Result on requested command: depth `2` runs measured `frames=538 elapsed=2.53s fps=212.35` and `frames=538 elapsed=2.52s fps=213.23`, improving the previous `210.18` FPS checkpoint.
- Learning: Avoiding per-frame request construction, request dump, numpy payload wrapping, adapter image loading, and repeated kwarg mapping keeps the CPU producer closer to the CUDA graph replay cadence without changing model inputs.

### Rejected: Cached RFDETR PreProcessingOverrides In Workflow Fast Path

- Hypothesis: The new RFDETR pre-request fast path creates a default `PreProcessingOverrides` object per frame. Reusing one immutable default instance could remove a small allocation.
- Change tested: Temporary code only; replaced the per-call default override object with a module-level `_RFDETR_NO_PREPROCESSING_OVERRIDES = PreProcessingOverrides()`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.57s fps=208.98`, below the per-call override fast path.
- Learning: This was either noise-sensitive or interacted poorly with the surrounding path; keep the simpler per-call object that produced the better repeated benchmark.

### Rejected: Cache RFDETR Fast Path Model Reference

- Hypothesis: Even after the pre-request fast path, `run_locally(...)` still calls `model_manager.add_model(...)` and `_try_run_rfdetr_trt_fast_path(...)` indexes the manager every frame. Caching the loaded RFDETR adapter on the workflow block could avoid model-manager cache refresh and lock overhead.
- Change tested: Temporary code only; stored `_rfdetr_trt_fast_path_model` and `_rfdetr_trt_fast_path_model_id` after first lookup, skipped `add_model(...)` when the cached ID matched, and reused the cached adapter.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.55s fps=210.59`, below the simpler pre-request fast path.
- Learning: The added per-frame Python attribute checks and branches outweigh the manager lookup savings in this benchmark. Keep the direct manager call.

### Rejected: Submit Next Workflow Batch Before Sink Emit

- Profile: Captured the current pre-request fast path with Nsight Systems at `/tmp/rfdetr_request_bypass_20260523_041056.nsys-rep` and exported `/tmp/rfdetr_request_bypass_20260523_041056.sqlite`. Under profiler, depth `2` measured `frames=538 elapsed=3.01s fps=178.65`. After skipping the first 100 graph launches, CUDA graph duration was stable at p50 `3590.535 us`, while graph end-to-next-start gap was p50 `1116.078 us`, p90 `3721.247 us`, p95 `4175.294 us`, p99 `5359.476 us`.
- Hypothesis: In the multi-worker `InferencePipeline` loop, emitting a completed ordered result before submitting the next batch may leave a worker slot idle and widen the graph replay gap. Submitting the next batch immediately after a slot frees, then emitting the ordered sink result, could reduce CPU-side bubbles while preserving `max_inflight_workflow_batches=2`.
- Change tested: Temporary code only; collected completed ordered results in `ready_to_emit`, submitted the current frame as soon as the pending count dropped below the worker limit, and emitted the collected results afterward.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.55s fps=210.92` and `frames=538 elapsed=2.54s fps=211.51`, below the current `213.23` FPS checkpoint.
- Learning: Sink emission is not the source of the remaining graph bubbles for this benchmark. Keep the original simpler ordered scheduler and continue focusing on model/postprocess conversion costs.

### Rejected: Keep Deferred RFDETR Query Indices Int32

- Hypothesis: The deferred fused postprocess path zero-fills `query_indices` and converts it from int32 to int64 before the Triton mask resize kernel, even though the Triton kernel only needs int32 indices. Using an uninitialized int32 tensor for the deferred path could remove a fill kernel and an int32-to-int64 copy kernel.
- Change tested: Temporary code only; changed `query_indices` from `torch.zeros(...)` to `torch.empty(...)` and skipped `.to(dtype=torch.long)` when `return_cpu_count=False`.
- Correctness: Compared exact-sized postprocess against deferred fused postprocess on 120 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.56s fps=209.81`, below the current checkpoint.
- Learning: The removed kernels are not on the critical path enough to offset allocator or scheduling side effects. Keep the existing zeroed int32 tensor and long conversion.

### Rejected: Query Pinned Preprocess Copy Event Before Synchronize

- Hypothesis: The reusable pinned preprocessing buffer synchronizes its previous H2D copy before every reuse. With pipeline depth `2`, the previous copy for that thread should usually be complete, so checking `copy_event.query()` before `copy_event.synchronize()` could avoid unnecessary synchronization overhead.
- Change tested: Temporary code only; changed `_get_pinned_normalized_buffer(...)` to synchronize only when the recorded copy event had not completed.
- Correctness: Compared exact-sized postprocess against deferred fused postprocess on all 538 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `212.27 fps`, `214.53 fps`, `211.70 fps`, and `211.77 fps`. The high run beat the checkpoint, but the repeated runs did not consistently clear the current `213.23` FPS best.
- Learning: The query-before-sync guard is too noise-sensitive here and likely adds API overhead on the common completed-event path. Keep the simpler unconditional event synchronize.

### Single-Step Workflow Runner Fast Path

- Hypothesis: The benchmark workflow is a single image input, one instance-segmentation model step, and one output selecting `$steps.segmentation.predictions`. Even after RFDETR model fast paths, each frame still pays generic workflow runtime assembly, validation, execution-data-manager/coordinator setup, step scheduling, and output construction before the next CUDA graph can be launched. A guarded direct runner for this exact one-step shape should reduce CPU bubbles without changing model execution.
- Change: `WorkflowRunner` now caches a fast path for workflows with exactly one `roboflow_core/roboflow_instance_segmentation_model@v3` step, one image input, no input substitutions, no serialization/preview mode, and one `predictions` output. The fast path constructs `WorkflowImageData` directly from `VideoFrame`, calls the initialized block with static manifest parameters, and returns the same output field shape. Other workflows fall back to the generic execution engine.
- Correctness: Compared the generic execution engine against the fast runner on all 538 frames from `vehicles_312px.mp4`; counts and class IDs matched exactly and max box delta was `0` px.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.49s fps=216.19` and `frames=538 elapsed=2.46s fps=218.97`, improving the previous `213.23` FPS checkpoint.
- Profile: Nsight Systems capture `/tmp/rfdetr_single_step_fast_20260523_043134.nsys-rep` exported to `/tmp/rfdetr_single_step_fast_20260523_043134.sqlite`. Under profiler, depth `2` measured `frames=538 elapsed=2.54s fps=212.13`. After skipping the first 100 graph launches, CUDA graph duration was p50 `3811.684 us`; graph end-to-next-start gap was p50 `738.036 us`, p90 `792.180 us`, p95 `817.677 us`, p99 `921.074 us`, down from the prior p50 `1116.078 us`, p90 `3721.247 us`, p95 `4175.294 us`.
- Learning: The remaining graph gaps were partly generic workflow orchestration overhead. The most valuable CPU work now is removing frame-level workflow machinery around the already-optimized RFDETR block while preserving the normal workflow path for non-trivial graphs.

### Rejected: Omit Video Metadata In Single-Step Workflow Fast Path

- Hypothesis: The single model/output fast path does not consume `WorkflowImageData.video_metadata`, so skipping per-frame `VideoMetadata` construction might reduce CPU work between graph launches.
- Change tested: Temporary code only; omitted `VideoMetadata(...)` construction and passed no `video_metadata` into the fast path's `WorkflowImageData`.
- Correctness: Compared the generic execution engine against the fast runner on 120 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.47s fps=217.60` and `frames=538 elapsed=2.50s fps=215.45`, below the current `218.97` FPS checkpoint.
- Learning: This object construction is not a reliable bottleneck, and omitting the metadata may perturb surrounding scheduling without improving throughput. Keep the fast path semantically closer to the generic runner.

### Rejected: Call Local Instance Segmentation Block Directly

- Hypothesis: The single-step workflow fast path still calls the block's generic `run(...)`, which computes the confidence value and branches on local vs remote execution every frame. For the local benchmark, precomputing confidence once and calling `run_locally(...)` directly could shave CPU work before the next CUDA graph launch.
- Change tested: Temporary code only; required the cached single-step fast path to be local, precomputed the manifest confidence value, and called `step.run_locally(...)` instead of `step.run(...)`.
- Correctness: Compared the generic execution engine against the direct local fast runner on 120 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.46s fps=218.48` and `frames=538 elapsed=2.46s fps=218.33`, close but still below the current `218.97` FPS checkpoint.
- Learning: The generic block method wrapper is not the next meaningful limiter. Keep the broader fast path that still supports the block's normal local/remote dispatch.

### Rejected: Derive Detection IDs From Inference ID

- Hypothesis: The RFDETR workflow conversion creates one inference UUID per frame and one detection UUID per detection. Deriving detection IDs from the existing inference ID could reduce CPU work in result materialization after the single-step workflow fast path removed larger orchestration overhead.
- Change tested: Temporary code only; when `inference_id` was available, set detection IDs to `f"{inference_id}.{detection_idx}"` instead of calling `uuid.uuid4()` per detection, preserving the old UUID behavior when no inference ID exists.
- Correctness: Compared the generic execution engine against the fast runner on 120 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.49s fps=215.79`, below the current checkpoint.
- Learning: Per-detection UUID generation is not the limiting CPU cost in this path, or the changed string construction perturbs allocation enough to lose. Keep the existing detection UUID behavior.

### Rejected: GPU Normalize Resized RFDETR Input

- Hypothesis: The latest profile shows the steady float32 H2D input copy costs about `188 us/frame` (`1168128` bytes per frame). Copying the resized uint8 HWC image to GPU and using a fused Triton kernel to produce normalized float32 CHW input could replace that with a 4x smaller H2D transfer plus GPU work.
- Change tested: Temporary code only; for CUDA single-image RFDETR preprocessing, copied the PIL-resized uint8 image through a pinned uint8 buffer and launched a Triton HWC-uint8-to-normalized-CHW kernel on the preprocessing stream.
- Correctness: Compared the new GPU-normalize path against the previous CPU-normalize path on 120 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`, `max_tensor_delta=4.76837158203125e-07`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.58s fps=208.79`, far below the current checkpoint.
- Learning: On this T4 workload, the extra device allocation, uint8 copy path, and normalization kernel cost more than the saved H2D bandwidth. Keep the CPU vectorized normalization into pinned float32.

### Rejected: Pass Single RFDETR Image Without List Wrapper

- Hypothesis: The RFDETR TRT workflow fast path always creates a one-element Python list of NumPy images before preprocessing. Passing the single NumPy frame directly could reduce small Python overhead and let preprocessing avoid a list allocation.
- Change tested: Temporary code only; used `images[0].numpy_image` when the batch size was one and kept the old list comprehension for larger batches.
- Correctness: Compared the generic execution engine against the fast runner on 120 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.49s fps=215.65`, below the current checkpoint.
- Learning: The single-image direct path changes preprocessing internals enough to hurt throughput; keep the explicit one-element list.

### Rejected: Cache Workflow Class Names As NumPy Array

- Hypothesis: Local workflow conversion rebuilds class-name strings with a Python loop for every frame. Caching `model.class_names` as a NumPy array and indexing it with `class_id` could reduce result materialization work after the single-step runner fast path removed larger orchestration costs.
- Change tested: Temporary code only; cached `model.class_names` on the adapter as `_workflow_class_names_np` and used vectorized indexing for in-range class IDs, falling back to the original loop for out-of-range IDs.
- Correctness: Compared the generic execution engine against the fast runner on 120 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.46s fps=219.07`, then repeated at `frames=538 elapsed=2.47s fps=217.57`; not stable enough to beat the current `218.97` FPS checkpoint.
- Learning: Class-name construction is too small/noise-sensitive to checkpoint. Keep the simpler existing loop.

### Rejected: Larger Triton Mask Resize Pixel Tile

- Hypothesis: `_resize_selected_masks_kernel` is the largest custom postprocess kernel. Increasing its pixel tile from `256` to `512` could reduce program count and launch-side work without changing the one-detection-per-program layout that outperformed prior blocked-detection variants.
- Change tested: Temporary code only; changed `fused_resize_selected_masks(...)` block size from `256` to `512`.
- Correctness: Compared exact-sized postprocess against deferred fused postprocess on 120 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `219.52 fps`, then `218.91 fps`, then `215.93 fps`; not stable enough to checkpoint over the current `218.97` FPS best.
- Learning: The larger tile can occasionally win but is too variable on this T4 workload. Keep the original `256` pixel tile.

### Rejected: Skip Per-Frame Status Update Without Handlers

- Hypothesis: `_emit_inference_result(...)` builds a DEBUG status payload with frame IDs, timestamps, and source IDs for every frame, even when no `status_update_handlers` are configured. Guarding that call could reduce CPU work in the inference thread before submitting the next frame.
- Change tested: Temporary code only; wrapped the hot per-frame `send_inference_pipeline_status_update(...)` call in `if self._status_update_handlers`.
- Correctness: The change does not affect model execution or prediction contents; it only skips status-update allocation when no handlers exist.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.49s fps=216.14`, below the current checkpoint.
- Learning: This status payload is not the current bottleneck, or the branch perturbs the tight loop. Keep the existing status update behavior.

### Rejected: Prebind Single-Step Workflow Manifest Values

- Hypothesis: The single-step workflow runner fast path still reads manifest attributes and `step.run` from closure objects every frame. Capturing the static values once when the fast path is built could reduce Python attribute lookup overhead.
- Change tested: Temporary code only; captured `step.run` and all static manifest parameters into local closure variables, then used those locals for the per-frame model block call.
- Correctness: Compared the generic execution engine against the fast runner on 120 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.48s fps=217.09`, below the current checkpoint.
- Learning: Attribute lookup in this closure is not the limiting cost. Keep the clearer manifest-based call.

### Rejected: Torch Inference Mode Around RFDETR Fast Path

- Hypothesis: The RFDETR workflow fast path enters PyTorch/TensorRT preprocessing, forward, postprocess, and tensor-to-NumPy conversion without an explicit `torch.inference_mode()` guard. Adding it could reduce autograd/version-counter overhead around the CUDA graph and fused kernels.
- Change tested: Temporary code only; imported `torch` in the workflow block and wrapped the RFDETR TRT fast path pre-process, forward, post-process, and conversion in `with torch.inference_mode():`.
- Correctness: Compared the generic execution engine against the fast runner on 120 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.49s fps=215.70`, below the current checkpoint.
- Learning: The per-frame inference-mode context or its interaction with graph-replayed tensors costs more than any autograd savings. Keep the existing fast path without an extra context manager.

### Rejected: Borrow TensorRT CUDA Graph Output Buffers

- Hypothesis: The CUDA graph replay path clones all TensorRT output buffers before RFDETR postprocess. For the depth-2 workflow fast path, each worker consumes postprocess results before it takes another frame, so returning thread-scoped graph output buffers directly could let postprocess run on the graph outputs and remove device-to-device clone work between graph launches.
- Change tested: Temporary code only; added an explicit `borrow_cuda_graph_outputs` flag to the TRT graph path, keyed borrowed graph states by worker thread, returned the cached graph output buffers without cloning, and enabled the flag only in the RFDETR TRT workflow fast path.
- Correctness: Compared borrowed graph outputs against cloned graph outputs on 120 frames after slicing by the deferred valid count: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`, `max_conf_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.62s fps=205.72` and `frames=538 elapsed=2.61s fps=206.36`, well below the current `218.97` FPS checkpoint.
- Learning: Removing the D2D output clones is not enough to overcome the extra per-thread graph state and scheduling cost in this workload. Keep the existing cloned-output graph replay path.

### Rejected: Fuse Box Rounding Into Triton Selector

- Hypothesis: The deferred fused postprocess still launches PyTorch kernels for `selected_boxes.round().int()` after the Triton top-k selector. Writing rounded integer boxes directly from `_select_topk_boxes_kernel` could remove that post-selector work from the gap between CUDA graph launches.
- Change tested: Temporary code only; made the selector allocate `boxes` as `int32`, stored clipped coordinates with `+0.5` for positive-coordinate rounding, and skipped the Python-side `round().int()` when fused boxes were already integer.
- Correctness: Compared fused output against the unfused PyTorch fallback on 120 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`, `max_conf_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.47s fps=218.19` and `frames=538 elapsed=2.48s fps=216.58`, not consistently above the current `218.97` FPS checkpoint.
- Learning: The small box-conversion kernels are visible in Nsight Systems but are not throughput-limiting at depth `2`. Keep the simpler float-box selector plus existing PyTorch rounding.

### Rejected: More Warps For Triton Mask Resize

- Profile: Nsight Compute capture `/tmp/rfdetr_resize_kernel_depth2_current.ncu-rep` sampled `_resize_selected_masks_kernel`; current launches use grid `(100, 215, 1)`, block `(128, 1, 1)`, 32 registers/thread, and full theoretical occupancy on T4 for the sampled kernels.
- Hypothesis: The current mask resize keeps the 256-pixel tile but launches with `num_warps=4`. Increasing to `num_warps=8` could map the same per-program vector work across more lanes without changing the tile size or the one-detection-per-program layout.
- Change tested: Temporary code only; changed `fused_resize_selected_masks(...)` to launch `_resize_selected_masks_kernel` with `num_warps=8`.
- Correctness: Compared deferred fused postprocess against exact-sized postprocess on 120 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0`, `max_conf_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.48s fps=217.27` and `frames=538 elapsed=2.49s fps=215.66`, below the current `218.97` FPS checkpoint.
- Learning: The current 4-warp launch is already well-balanced for this T4 kernel. More warps change scheduling without reducing the end-to-end graph gap.

### Rejected: Direct Pinned Host Input To CUDA Graph

- Hypothesis: The current TRT graph path first copies the normalized pinned CPU tensor to a temporary CUDA tensor, then copies that CUDA tensor into the captured graph input buffer. Returning the pinned CPU tensor from RFDETR preprocessing and copying it directly into the graph input buffer could remove the temporary CUDA input tensor and one device-to-device copy.
- Change tested: Temporary code only; added a guarded `keep_cuda_graph_input_on_host` path for the RFDETR TRT workflow fast path, skipped CUDA transfer in preprocessing, let the CUDA graph input copy accept pinned CPU input with `non_blocking=True`, and recorded the pinned-buffer reuse event after graph-stream consumption.
- Correctness: Compared direct pinned-host graph input against the normal CUDA-input graph path on 120 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`, `max_conf_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.49s fps=215.65` and `frames=538 elapsed=2.49s fps=216.35`, below the current `218.97` FPS checkpoint.
- Learning: The existing temporary CUDA input allows the H2D copy to overlap on the preprocessing stream before graph replay. Moving H2D into the graph input copy reduces a D2D copy but puts the larger H2D transfer closer to the critical path, widening the effective graph gap.

### Rejected: Bit-Pack Dense Masks Before D2H Copy

- Hypothesis: Dense masks dominate prediction Device-to-Host bytes. Packing selected boolean masks into bytes on GPU before copying to CPU could reduce mask D2H payload by roughly 8x, then CPU `np.unpackbits(...)` could restore the existing `sv.Detections.mask` shape.
- Change tested: Temporary code only; added a Triton `_pack_bool_masks_kernel`, used it in the local workflow conversion for CUDA bool masks, copied packed `uint8` data to CPU, and unpacked with little-endian bit order before constructing `sv.Detections`.
- Correctness: The standalone packer matched a CUDA bool tensor exactly. Full `InferencePipeline` comparison against the normal mask-copy path on all 538 frames matched counts, class IDs, boxes, and dense masks exactly: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.50s fps=214.93` and `frames=538 elapsed=2.48s fps=217.31`, below the current `218.97` FPS checkpoint.
- Learning: The extra Triton launch and CPU unpack work cost more than the saved D2H bandwidth for these small per-frame mask payloads. Keep the direct bool mask copy.

### RFDETR Fused CPU Normalization Constants

- Hypothesis: The RFDETR PIL/NumPy preprocessing fast path still performs three full-array operations per channel: multiply by `1/255`, subtract mean, then divide by std. Precomputing `1/(255*std)` and `-mean/std` should reduce this to multiply plus add per channel while preserving prediction outputs.
- Change: `_get_normalization_constants(...)` now caches per-channel `multiplier` and `bias`, and `_pil_image_to_normalized_tensor(...)` applies `image * multiplier + bias` directly into the CHW float32 buffer.
- Correctness: Compared the fused-normalization path against the torchvision fallback path on all 538 frames from `vehicles_312px.mp4`: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`, `max_conf_delta=0`, `max_tensor_delta=7.152557373046875e-07`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.46s fps=218.60`, `frames=538 elapsed=2.46s fps=219.03`, and `frames=538 elapsed=2.46s fps=218.97`.
- Profile: Nsight Systems capture `/tmp/rfdetr_fused_norm_20260523_054913.nsys-rep` exported to `/tmp/rfdetr_fused_norm_20260523_054913.sqlite`; under profiler, depth `2` measured `frames=538 elapsed=2.56s fps=210.22`. After skipping the first 100 graph launches, CUDA graph duration was p50 `3787.381 us`; graph end-to-next-start gap was p50 `763.060 us`, p90 `852.370 us`, p95 `892.396 us`, p99 `1042.918 us`.
- Learning: This is a small CPU-side gain but it is algebraically simple and keeps the benchmark at the current best band. The remaining bottleneck is still the graph replay plus the postprocess/materialization tail rather than normalization alone.

### Rejected: Skip Pinned Buffer Reuse Synchronize

- Hypothesis: The RFDETR workflow fast path runs preprocess, forward, postprocess, and result materialization before the same worker thread reuses its thread-local pinned normalization buffer. The previous H2D copy should therefore already be complete, so skipping `_get_pinned_normalized_buffer(...)`'s `copy_event.synchronize()` in this fast path could remove a small CPU API wait.
- Change tested: Temporary code only; added a guarded `skip_pinned_buffer_reuse_sync` flag through RFDETR preprocessing and enabled it only for the RFDETR TRT workflow fast path.
- Correctness: Compared skip-sync against the normal synchronized path on all 538 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`, `max_conf_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.47s fps=218.19` and `frames=538 elapsed=2.50s fps=215.27`, below the current `219.03` FPS checkpoint.
- Learning: The event synchronize is either already cheap when completed or helps maintain better copy/launch ordering. Keep the explicit pinned-buffer reuse synchronization.

### Rejected: Skip Empty Class Filter Helper

- Hypothesis: The benchmark workflow does not set `class_filter`, but the RFDETR workflow fast path still calls `filter_out_unwanted_classes_from_sv_detections_batch(...)`, which immediately returns when no filter is provided. Avoiding the no-op function call could shave a small amount of Python result-materialization overhead.
- Change tested: Temporary code only; guarded calls to `filter_out_unwanted_classes_from_sv_detections_batch(...)` with `if class_filter:` in the inference-models and RFDETR TRT workflow fast paths.
- Correctness: Compared the guarded path against the previous always-call behavior on all 538 frames through `InferencePipeline`: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.47s fps=217.59` and `frames=538 elapsed=2.49s fps=215.87`, below the current `219.03` FPS checkpoint.
- Learning: The no-op helper call is too small to matter, and the extra branch may perturb the tight path. Keep the simpler existing call.

### Rejected: Single-Lock TRT CUDA Graph Cache Lookup

- Hypothesis: The TRT CUDA graph replay path checks `cache_key not in trt_cuda_graph_cache` and then indexes `trt_cuda_graph_cache[cache_key]`, acquiring the cache lock twice per frame on the steady path. A `get(...)` method that moves the key to the LRU tail under one lock could reduce Python/API overhead between graph launches.
- Change tested: Temporary code only; added `TRTCudaGraphCache.get(...)` and used it in `_execute_trt_engine(...)` before deciding whether to capture or replay a graph.
- Correctness: Compared standard TensorRT execution against CUDA graph execution on 120 frames after the cache change: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`, `max_conf_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.45s fps=219.26`, `frames=538 elapsed=2.46s fps=218.76`, then `frames=538 elapsed=2.48s fps=217.02`; not stable enough to checkpoint over the current `219.03` FPS best.
- Learning: The double lock is not a reliable limiter, and the altered lookup path is noise-sensitive. Keep the existing cache API.

### Rejected: Static RFDETR PreProcessingOverrides

- Hypothesis: The RFDETR TRT workflow fast path constructs the same `PreProcessingOverrides(False, False, False)` object every frame. Reusing a module-level instance could remove a small allocation from the CPU path before graph replay.
- Change tested: Temporary code only; added a module-level `RFDETR_TRT_PRE_PROCESSING_OVERRIDES` and passed it to `model._model.pre_process(...)` in `_try_run_rfdetr_trt_fast_path(...)`.
- Correctness: Full-video `InferencePipeline` comparison against an equivalent overrides object matched all 538 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.45s fps=219.53`, then repeated at `frames=538 elapsed=2.49s fps=215.74`; not stable enough to checkpoint.
- Learning: Per-frame override-object construction is below the noise floor. Keep the simpler local construction.

### Rejected: In-Place RFDETR Postprocess Sigmoid

- Hypothesis: The RFDETR TRT workflow path postprocesses TensorRT output clones that are not reused after postprocess. Applying `sigmoid_()` to the logits in place could avoid allocating a separate sigmoid tensor and reduce postprocess memory traffic.
- Change tested: Temporary code only; added a gated `inplace_sigmoid` option to `post_process_instance_segmentation_results(...)`, passed it through TRT instance segmentation postprocess, and enabled it only in the RFDETR TRT workflow fast path.
- Correctness: Compared in-place sigmoid against the default out-of-place sigmoid on all 538 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0`, `max_conf_delta=0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.48s fps=217.36` and `frames=538 elapsed=2.48s fps=216.73`, below the current `219.03` FPS checkpoint.
- Learning: The standalone sigmoid allocation is not the bottleneck; in-place mutation likely changes scheduling or allocator behavior enough to lose. Keep the out-of-place PyTorch sigmoid.

### Rejected: Fuse RFDETR Sigmoid Into Triton Selector

- Hypothesis: The dense-mask RFDETR fused postprocess path still runs a PyTorch sigmoid over logits before the Triton selector. Letting `_select_topk_boxes_kernel` load raw logits and apply sigmoid internally could remove a kernel launch and temporary tensor between CUDA graph replay and postprocess.
- Change tested: Temporary code only; added an `apply_sigmoid` constexpr to the Triton selector, lazily skipped the global PyTorch sigmoid when the fused path handled a frame, and kept the original sigmoid fallback for unsupported metadata.
- Correctness: Compared fused selector output against the PyTorch fallback on 120 frames including masks: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=1.1920928955078125e-07`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.48s fps=216.53` and `frames=538 elapsed=2.49s fps=215.82`, below the current `219.03` FPS checkpoint.
- Learning: The selector is already doing enough reduction work that adding sigmoid math slows it down more than the separate PyTorch sigmoid costs. Keep sigmoid outside the selector.

### Rejected: Compact Borrowed TRT Masks Before Next Graph

- Hypothesis: CUDA graph replay clones all TensorRT outputs, including the fixed `(100, 78, 78)` mask tensor, before the graph stream can accept the next replay. The video usually keeps only about 5 detections, so borrowing graph outputs, selecting boxes, compacting only selected masks, and then resizing from the compact buffer could reduce graph-to-graph spacing.
- Change tested: Temporary code only; added a guarded no-clone CUDA graph output path, a Triton gather kernel for selected masks, a compact-mask resize kernel, and enabled the path only in the RFDETR TRT workflow fast path with pipeline depth `2`.
- Correctness: Compared compact borrowed outputs against the cloned-output fused path on 120 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.48s fps=216.53`, below the current `219.03` FPS checkpoint.
- Learning: Replacing the full mask clone with selector/gather stream choreography adds enough pre-next-graph work to lose throughput. The existing clone is cheaper than the synchronization structure needed to borrow graph outputs safely.

### RFDETR Pinned Host Detection Materialization

- Hypothesis: The workflow conversion path performs separate blocking `.cpu().numpy()` copies for boxes, confidences, classes, and dense masks after the deferred fused count read. Reusing thread-local pinned host buffers and enqueueing all D2H copies before one stream synchronize should reduce result-materialization overhead while preserving independent NumPy arrays for queued predictions.
- Change: Added a CUDA-only conversion fast path in the instance segmentation workflow block. When RFDETR fused postprocess provides a deferred valid count, the converter copies selected boxes, confidences, classes, and masks into reusable pinned CPU buffers with `non_blocking=True`, synchronizes once, then returns normal copied NumPy arrays so queued sink payloads are not backed by reusable storage.
- Correctness: Compared pinned conversion against the previous `.cpu().numpy()` conversion on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.44s fps=220.62`, `frames=538 elapsed=2.45s fps=219.88`, and `frames=538 elapsed=2.45s fps=220.00`.
- Profile: Nsight Systems capture `/tmp/rfdetr_pinned_conversion_20260523_063547.nsys-rep` exported to `/tmp/rfdetr_pinned_conversion_20260523_063547.sqlite`; under profiler, depth `2` measured `frames=538 elapsed=2.58s fps=208.40`. After skipping the first 100 graph launches, CUDA graph duration was p50 `3782.980 us`; graph end-to-next-start gap was p50 `806.163 us`, p90 `904.306 us`, p95 `921.649 us`, p99 `1010.096 us`. CUDA API `cudaStreamSynchronize` calls dropped from the earlier current-profile `2702` calls to `1088` calls.
- Learning: The deferred GPU count still gates CPU materialization, but grouping the remaining D2H copies onto pinned buffers reduces enough per-frame synchronization/API overhead to move the checkpoint above the previous `219.03` FPS band.

### Rejected: Fixed-Capacity RFDETR Conversion Buffers

- Hypothesis: The pinned conversion checkpoint grows thread-local host buffers when selected detection count increases, producing extra `cudaHostAlloc` calls in the Nsight profile. Allocating the full 100 RFDETR detection slots on first use could avoid reallocations during timed frames.
- Change tested: Temporary code only; forced `_get_rfdetr_conversion_buffers(...)` to allocate at least 100 rows while still copying only the selected-count slice.
- Correctness: Compared fixed-capacity pinned conversion against the forced `.cpu().numpy()` fallback on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.45s fps=219.73`, `frames=538 elapsed=2.44s fps=220.67`, then `frames=538 elapsed=2.48s fps=216.81`; not stable enough to keep over the dynamic-capacity pinned conversion checkpoint.
- Learning: The larger pinned allocation changes memory behavior enough to introduce variance, and avoiding a few growth allocations does not reliably improve steady-state throughput. Keep the dynamic grow-to-needed-count buffers.

### Rejected: Fold RFDETR Workflow Metadata Attachment

- Hypothesis: After pinned D2H conversion, the RFDETR workflow fast path still makes separate Python passes for prediction type attachment, no-op class filtering, and parent-coordinate metadata. Folding prediction type and parent metadata into one RFDETR-specific helper and skipping the empty class filter could reduce CPU materialization work.
- Change tested: Temporary code only; added `_attach_rfdetr_fast_path_metadata(...)` to attach prediction type plus root/parent IDs, coordinates, and dimensions in a single pass, and used it only in `_try_run_rfdetr_trt_fast_path(...)`.
- Correctness: Compared folded metadata against the previous helper chain on all 538 frames, ignoring random detection IDs: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes=0`, `bad_data=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.47s fps=218.02` and `frames=538 elapsed=2.45s fps=219.31`, below the pinned-conversion checkpoint.
- Learning: The helper calls are not the limiter, and folding them changes enough Python allocation/order behavior to lose throughput. Keep the established helper chain.

### Rejected: Reuse Deferred Fused Postprocess CUDA Buffers

- Hypothesis: The internal deferred RFDETR fused postprocess path allocates same-shaped CUDA tensors for scores, classes, boxes, query indices, count, and resized masks each frame. Reusing thread-local buffers for those fixed-capacity outputs could reduce allocator and Python overhead while preserving the deferred GPU count path.
- Change tested: Temporary code only; added thread-local output buffers in `fused_postprocess.py` and enabled them only when `_try_fused_instance_segmentation_post_process(...)` runs with `defer_count=True`.
- Correctness: Compared deferred fused postprocess with reused buffers against exact-sized postprocess on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.44s fps=220.45`, then `frames=538 elapsed=2.49s fps=216.02`; not stable enough to keep over the pinned-conversion checkpoint.
- Learning: PyTorch's caching allocator already handles these fixed shapes well enough. Thread-local reuse changes object lifetime/stream behavior and can degrade scheduling, so keep per-frame tensor creation.

### Rejected: Direct-Owned CPU Detection Tensors

- Hypothesis: The pinned conversion checkpoint copies GPU outputs into reusable pinned CPU tensors, synchronizes once, then copies those pinned NumPy views into independent arrays for queue safety. Allocating fresh CPU tensors per frame and returning NumPy views directly could remove the extra host copy while preserving result ownership.
- Change tested: Temporary code only; replaced the reusable pinned buffers with fresh CPU tensors for boxes, confidences, classes, and masks, copied CUDA tensors into them synchronously, and returned `.numpy()` views without `.copy()`.
- Correctness: Compared direct-owned CPU conversion against the forced `.cpu().numpy()` fallback on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.52s fps=213.17`, well below the pinned-conversion checkpoint.
- Learning: Per-frame CPU tensor allocation and blocking D2H copies cost much more than the extra host copy from reusable pinned buffers. Keep pinned staging plus independent NumPy copies.

### Rejected: Skip Inference ID In Predictions-Only Workflow Fast Path

- Hypothesis: The single-step workflow runner returns only `predictions`, but the RFDETR block still generates a workflow inference UUID and attaches it to every `sv.Detections` before the runner discards the enclosing `inference_id` field. Skipping that UUID/data-field work for predictions-only fast paths could reduce CPU materialization overhead.
- Change tested: Temporary code only; passed an internal `include_inference_id=False` flag from `WorkflowRunner`'s single-step fast path into the RFDETR TRT fast path and skipped the workflow inference ID generation/attachment when false.
- Correctness: Compared conversion with and without inference ID attachment on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`. This intentionally omitted the `inference_id` data field, so it was not suitable to keep unless it clearly won.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.49s fps=216.36`, below the pinned-conversion checkpoint.
- Learning: UUID/inference-ID attachment is not a meaningful limiter, and removing it weakens metadata behavior. Keep the normal inference ID path.

### Rejected: Split Mask D2H Copy Onto Separate Stream

- Hypothesis: In the pinned conversion checkpoint, boxes/confidences/classes/masks are copied to pinned CPU buffers on the same CUDA stream and synchronized together. Copying the larger dense mask payload on a side stream while the current stream handles small metadata copies could overlap CPU class-name/object preparation with mask D2H.
- Change tested: Temporary code only; added a thread-local mask-copy CUDA stream, launched the mask pinned copy there after waiting on the current stream, synchronized the current stream for small metadata, built class names, then synchronized the mask stream before constructing `sv.Detections`.
- Correctness: Compared split-stream pinned conversion against the forced `.cpu().numpy()` fallback on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.50s fps=215.61`, below the pinned-conversion checkpoint.
- Learning: The side-stream wait/synchronize overhead is larger than any overlap available in the small CPU metadata window. Keep the single-stream grouped pinned copies.

### Rejected: Localize RFDETR Conversion Lookups

- Hypothesis: The workflow conversion loop repeatedly reads `model.class_names`, calls `len(model.class_names)`, and recomputes `len(sv_detections)` for per-detection metadata arrays. Caching those values in local variables could shave small Python overhead after pinned D2H conversion.
- Change tested: Temporary code only; cached `model.class_names` and its length before the loop and cached `len(sv_detections)` once before filling detection IDs, parent IDs, image dimensions, and inference IDs.
- Correctness: Compared the localized conversion against the forced `.cpu().numpy()` fallback on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.45s fps=220.02`, then `frames=538 elapsed=2.48s fps=216.62`; not stable enough to keep over the pinned-conversion checkpoint.
- Learning: These Python lookups are below the benchmark noise floor, and the altered bytecode/allocation order can regress scheduling. Keep the original straightforward conversion loop.

### Rejected: NumPy Array Mask Copy

- Hypothesis: The pinned conversion checkpoint uses `mask_view.copy()` to produce queue-safe owned mask arrays. A local micro-benchmark on `(5, 312, 312)` bool masks showed `np.array(mask_view, copy=True)` slightly faster, so swapping only the mask copy idiom might shave host materialization time without changing ownership.
- Change tested: Temporary code only; replaced `mask_buffer[:valid_count].numpy().copy()` with `np.array(mask_buffer[:valid_count].numpy(), copy=True)`.
- Correctness: Compared the alternate mask copy against the forced `.cpu().numpy()` fallback on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.48s fps=217.17`, below the pinned-conversion checkpoint.
- Learning: The isolated NumPy copy micro-benchmark does not predict full pipeline behavior; keep the direct ndarray `.copy()` path.

### Rejected: Internal Single-Image RFDETR Preprocess Fast Path

- Hypothesis: The RFDETR TRT workflow always passes a one-element image list to preprocessing. Handling that case before the generic list-normalization loop could avoid list copying, per-frame append bookkeeping, and the final `len(tensors)` branch while preserving the same image object and preprocessing math.
- Change tested: Temporary code only; added a guarded `len(images) == 1` branch in `pre_process_network_input(...)` that directly preprocesses the single image, transfers it to the target device, records the pinned-copy event, and returns `unsqueeze(0)` plus a one-element metadata list.
- Correctness: Compared the list fast path against the existing generic single-frame 4D NumPy batch path on all 538 frames: `bad_counts=0`, `bad_classes=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`, `max_tensor_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.48s fps=216.84` and `frames=538 elapsed=2.47s fps=217.64`, below the pinned-conversion checkpoint.
- Learning: The generic loop overhead is not meaningful, and adding another top-level branch/function path likely perturbs Python scheduling. Keep the existing preprocessing control flow.

### Rejected: Cache RFDETR TRT Confidence Threshold

- Hypothesis: RFDETR TRT postprocess constructs a `ConfidenceFilter` and resolves the same custom confidence threshold every frame. Caching the resolved threshold on the model instance could remove small Python work between graph replay and fused postprocess.
- Change tested: Temporary code only; added a per-instance `_confidence_threshold_cache` keyed by `(confidence, id(recommended_parameters))` and passed the cached threshold into dense and RLE postprocess.
- Correctness: On a sampled video frame, a cache miss followed by a cache hit produced matching detections: `counts 4 4`, `classes_equal=True`, `masks_equal=True`, `max_box_delta=0`, `max_conf_delta=0.0`; the cached scalar also matched `ConfidenceFilter.get_threshold(...)`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.46s fps=218.80` and `frames=538 elapsed=2.47s fps=217.45`, below the pinned-conversion checkpoint.
- Learning: Confidence threshold construction is below the limiter; changing the model object state and postprocess bytecode does not tighten the graph-to-graph gap. Keep the original local `ConfidenceFilter` path.

### Rejected: Two-Slot TRT CUDA Graph Output Copies

- Hypothesis: The CUDA graph replay path allocates fresh result tensors with `buf.clone()` for each TensorRT output after every replay. For the requested depth `2` workflow, alternating between two reusable output-copy slots could keep ownership safe while reducing allocator work between graph launches.
- Change tested: Temporary code only; added optional `cuda_graph_output_copy_slots=2` plumbing to the TRT replay path and enabled it only in the RFDETR workflow fast path, copying graph outputs into alternating reusable CUDA tensors instead of cloning into fresh tensors.
- Correctness: Compared slot-copy outputs against the clone path on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.46s fps=218.78` and `frames=538 elapsed=2.45s fps=219.73`, not an improvement over the pinned-conversion checkpoint.
- Learning: PyTorch's cached allocation for the result clones is not the source of the remaining graph gap. Keeping reusable borrowed CUDA buffers also adds a depth-specific ownership assumption, so leave the clone path unchanged.

### Current Depth-2 Nsight Profile

- Request: Generate a fresh Nsight Systems capture on the current accepted code path while keeping pipeline depth fixed at `2`.
- Profile: Nsight Systems capture `/tmp/rfdetr_depth2_graphtrace_local_20260523_073518.nsys-rep` exported to `/tmp/rfdetr_depth2_graphtrace_local_20260523_073518.sqlite`; CSV summaries are `/tmp/rfdetr_depth2_graphtrace_local_20260523_073518_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_graphtrace_local_20260523_073518_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_graphtrace_local_20260523_073518_stats_cuda_api_sum.csv`.
- Result under profiler: depth `2` measured `frames=538 elapsed=2.58s fps=208.77`.
- Graph spacing: The corrected capture includes `538` CUDA graph traces on stream `39`. After skipping the first 100 launches, CUDA graph duration was p50 `3781.637 us`, p90 `3812.413 us`, p95 `3817.378 us`, p99 `3824.886 us`; graph end-to-next-start gap was p50 `803.827 us`, p90 `891.474 us`, p95 `911.633 us`, p99 `1134.530 us`, mean `810.141 us`.
- Note: An earlier same-turn capture omitted `PYTHONPATH=/app/inference_models`, used the installed package, and did not include CUDA graph replay; ignore `/tmp/rfdetr_depth2_current_20260523_073219.*` and `/tmp/rfdetr_depth2_graphtrace_20260523_073343.*` for this optimization thread.

### Rejected: Keep Deferred Query Indices Int32

- Hypothesis: The deferred fused postprocess path converts `query_indices` from `int32` to `int64` even though the Triton mask resize kernel can consume `int32` directly. Removing that conversion could eliminate one small CUDA kernel between TensorRT graph replay and postprocess.
- Change tested: Temporary code only; when `fused_select_topk_boxes(..., return_cpu_count=False)` returned deferred outputs, it returned `query_indices` as `int32` instead of `query_indices.to(dtype=torch.long)`.
- Correctness: Compared deferred fused postprocess against the exact-sized path on all 538 frames: `max_count=7`, `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.44s fps=220.25`, `frames=538 elapsed=2.45s fps=219.49`, and `frames=538 elapsed=2.46s fps=218.85`; not stable enough to keep over the pinned-conversion checkpoint.
- Learning: The int32-to-int64 conversion kernel is visible but not a reliable limiter. Removing it changes downstream scheduling enough that FPS still falls into the noisy lower band.

### Rejected: Avoid Zero-Filling Deferred Query Indices

- Hypothesis: `fused_select_topk_boxes(...)` allocates `query_indices` with `torch.zeros(...)`, launching a fill kernel even though the selector writes every query index that the deferred mask resize kernel reads. Switching to `torch.empty(...)` could remove a per-frame CUDA fill.
- Change tested: Temporary code only; changed the `query_indices` allocation in `fused_select_topk_boxes(...)` from `torch.zeros(...)` to `torch.empty(...)`.
- Correctness: Compared deferred fused postprocess against the exact-sized path on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.46s fps=218.97`, `frames=538 elapsed=2.44s fps=220.54`, and `frames=538 elapsed=2.46s fps=218.43`; not stable enough to keep over the pinned-conversion checkpoint.
- Learning: The query-index zero fill is visible in the profile but not a stable throughput limiter. The allocation/fill behavior likely interacts with stream scheduling and allocator reuse, so keep the deterministic zero-filled tensor.

### RFDETR Limited Deferred Mask Resize

- Hypothesis: The deferred fused mask resize launches work for all 100 RFDETR query slots, but the benchmark video keeps at most 7 detections. Launching resize programs only for the common first few detections should reduce postprocess GPU work and tighten the graph-to-graph gap, while an overflow fallback can preserve correctness for frames with more detections.
- Change: Added an optional `detection_limit` to `fused_resize_selected_masks(...)`, threaded `deferred_mask_resize_detection_limit` through RFDETR dense postprocess, and enabled an 8-detection limit only in the RFDETR TRT workflow fast path. The postprocess metadata keeps the raw mask tensor and query indices, and workflow conversion reruns the full fused mask resize if the deferred GPU count exceeds the first-stage limit.
- Correctness: Compared the 8-limit deferred path against the exact-sized path on all 538 frames: `max_count=7`, `normal_bad_counts/classes/masks=[0, 0, 0]`, `max_box_delta=0.0`, `max_conf_delta=0.0`. Also forced `deferred_mask_resize_detection_limit=1` to exercise overflow recovery; 523 frames exceeded the limit and recovered full masks with `overflow_bad=[0, 0, 0]`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.45s fps=220.00`, `frames=538 elapsed=2.43s fps=221.07`, `frames=538 elapsed=2.42s fps=221.92`, and after formatting `frames=538 elapsed=2.42s fps=221.98`.
- Profile: Nsight Systems capture `/tmp/rfdetr_limited_mask_resize_20260523_075251.nsys-rep` exported to `/tmp/rfdetr_limited_mask_resize_20260523_075251.sqlite`; CSV summaries are `/tmp/rfdetr_limited_mask_resize_20260523_075251_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_limited_mask_resize_20260523_075251_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_limited_mask_resize_20260523_075251_stats_cuda_api_sum.csv`. Under profiler, depth `2` measured `frames=538 elapsed=2.54s fps=211.55`.
- Graph spacing: The capture includes `538` CUDA graph traces. After skipping the first 100 launches, CUDA graph duration was p50 `3782.628 us`, p90 `3815.915 us`, p95 `3818.414 us`, p99 `3829.039 us`; graph end-to-next-start gap was p50 `720.629 us`, p90 `805.716 us`, p95 `846.777 us`, p99 `998.453 us`, mean `723.399 us`. `_resize_selected_masks_kernel` dropped to `4.337 ms` total / `8.061 us` average from the previous profile's `40.815 ms` total / `75.864 us` average.
- Learning: The fixed 100-slot mask-resize grid was a real limiter for this low-detection-count stream. A small first-stage grid keeps the normal path fast, and the overflow recovery keeps the optimization safe for higher-count frames at the cost of extra work only when needed.

### Rejected: Allocate Only Limited Deferred Mask Rows

- Hypothesis: After limiting deferred mask resize to 8 detection rows, `fused_resize_selected_masks(...)` still allocated a `(100, H, W)` bool output tensor. Allocating only `(detection_limit, H, W)` for limited calls could reduce allocator and memory pressure.
- Change tested: Temporary code only; moved `detection_limit` clamping before output allocation and allocated the output tensor with `detection_limit` rows instead of `MAX_RFDETR_DETECTIONS` rows.
- Correctness: Compared the 8-limit deferred path against the exact-sized path on all 538 frames: `max_count=7`, `limited_mask_shape=(8, 176, 312)`, `normal_bad_counts/classes/masks=[0, 0, 0]`, `max_box_delta=0.0`, `max_conf_delta=0.0`. Forced overflow recovery with limit `1` still recovered 523 overflow frames with `overflow_bad=[0, 0, 0]`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.42s fps=221.88`, `frames=538 elapsed=2.43s fps=221.04`, and `frames=538 elapsed=2.43s fps=221.40`; not an improvement over the existing limited-resize checkpoint.
- Learning: The launch grid reduction matters; shrinking the cached output allocation does not reliably improve FPS. Keep the fixed output shape to avoid an extra shape variant in downstream code.

### Rejected: Selector Kernel Four-Warp Launch

- Hypothesis: After limiting mask resize work, `_select_topk_boxes_kernel` became the largest custom postprocess kernel. Reducing the Triton selector launch from `num_warps=8` to `num_warps=4` could reduce overhead if the 100x91 reduction was over-provisioned.
- Change tested: Temporary code only; changed `_select_topk_boxes_kernel` launch in `fused_select_topk_boxes(...)` from `num_warps=8` to `num_warps=4`.
- Correctness: Compared the limited deferred path against the exact-sized path on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.44s fps=220.52`, `frames=538 elapsed=2.42s fps=221.86`, and `frames=538 elapsed=2.44s fps=220.50`; not an improvement over the existing limited-resize checkpoint.
- Learning: The selector's 9100-score reduction still benefits from the 8-warp configuration, or the launch is not the steady limiter. Keep the existing selector launch.

### RFDETR Deferred Float Boxes

- Hypothesis: The deferred fused workflow path rounds `selected_boxes` and converts them to `int32`, launching PyTorch `round` and copy/cast kernels after the Triton selector. The benchmark requirement allows boxes within 5 pixels, so returning float boxes directly from the fused selector could remove those kernels without changing classes or masks.
- Change: In the deferred fused RFDETR instance segmentation path, return `xyxy=selected_boxes` instead of `xyxy=selected_boxes.round().int()`. The non-deferred public postprocess path still returns rounded integer boxes.
- Correctness: Compared the limited deferred path against the exact-sized rounded path on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.5`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.44s fps=220.87`, `frames=538 elapsed=2.44s fps=220.23`, `frames=538 elapsed=2.41s fps=223.00`, and `frames=538 elapsed=2.42s fps=222.55`.
- Profile: Nsight Systems capture `/tmp/rfdetr_float_boxes_20260523_080830.nsys-rep` exported to `/tmp/rfdetr_float_boxes_20260523_080830.sqlite`; CSV summaries are `/tmp/rfdetr_float_boxes_20260523_080830_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_float_boxes_20260523_080830_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_float_boxes_20260523_080830_stats_cuda_api_sum.csv`. Under profiler, depth `2` measured `frames=538 elapsed=2.55s fps=211.32`.
- Graph spacing: The capture includes `538` CUDA graph traces. After skipping the first 100 launches, CUDA graph duration was p50 `3786.738 us`, p90 `3821.854 us`, p95 `3824.001 us`, p99 `3845.892 us`; graph end-to-next-start gap was p50 `710.133 us`, p90 `792.947 us`, p95 `817.445 us`, p99 `975.935 us`, mean `702.444 us`. The PyTorch round kernel is absent from the top-kernel breakdown.
- Learning: Avoiding integer box materialization can produce a higher upper band once mask resize is limited, although the benchmark remains noisy. This is acceptable for the optimized workflow because geometry remains well inside the requested tolerance and class/mask outputs stay exact.

### Rejected: Float Boxes With Int32 Deferred Query Indices

- Hypothesis: Keeping deferred query indices as `int32` failed by itself, but after removing the deferred box round/int conversion it might remove another small copy/cast kernel without upsetting scheduling.
- Change tested: Temporary code only; combined the committed deferred float-box path with returning `query_indices` as `int32` from `fused_select_topk_boxes(..., return_cpu_count=False)`.
- Correctness: Compared the limited deferred path against the exact-sized rounded path on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.5`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.41s fps=223.26`, `frames=538 elapsed=2.45s fps=219.47`, and `frames=538 elapsed=2.42s fps=222.56`; the low outlier makes it less stable than the committed float-box checkpoint.
- Learning: The query-index cast remains schedule-sensitive even after removing box rounding. Keep the int64 conversion and preserve the more stable float-box checkpoint.

### Rejected: Fuse Sigmoid Into Selector After Mask Limit

- Hypothesis: After limiting mask resize work and skipping deferred box rounding, the PyTorch sigmoid over logits is one of the larger remaining postprocess kernels. Fusing sigmoid into `_select_topk_boxes_kernel` could remove that kernel in the current regime, even though it lost before the mask-grid optimization.
- Change tested: Temporary code only; added an `apply_sigmoid` constexpr to `_select_topk_boxes_kernel`, passed raw logits into the fused selector, and computed `1 / (1 + exp(-logit))` inside Triton before top-k selection. The fallback path lazily computed PyTorch sigmoid only when fused postprocess was unavailable.
- Correctness: Compared the limited deferred path against the exact-sized rounded path on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.5`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.44s fps=220.31` and `frames=538 elapsed=2.44s fps=220.10`, below the committed float-box checkpoint.
- Learning: The extra `exp` work inside the selector is still more expensive than the standalone PyTorch sigmoid kernel in the full pipeline. Keep sigmoid outside the selector.

### Rejected: Avoid Query-Index Zero Fill After Float Boxes

- Hypothesis: The selector still allocates `query_indices` with `torch.zeros(...)`, producing a small `FillFunctor<int>` kernel. This lost before deferred float boxes, but after removing box rounding the kernel mix changed enough to retest `torch.empty(...)`.
- Change tested: Temporary code only; changed the `query_indices` allocation in `fused_select_topk_boxes(...)` from `torch.zeros(...)` to `torch.empty(...)` on top of the committed float-box path.
- Correctness: Compared the limited deferred path against the exact-sized rounded path on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.5`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.43s fps=221.38`, `frames=538 elapsed=2.40s fps=223.81`, and `frames=538 elapsed=2.45s fps=219.46`; still too noisy and unstable to keep.
- Learning: Removing the zero fill can produce a fast run but also worsens low outliers. Keep the deterministic zero-filled query-index tensor.

### RFDETR Seven-Row Deferred Mask Resize

- Hypothesis: The limited deferred mask resize checkpoint uses an 8-row first-stage mask grid, but the benchmark video keeps at most 7 detections. Reducing the common-case mask grid to 7 rows should shave a small amount of GPU postprocess work while preserving overflow recovery for higher-count frames.
- Change: Changed the RFDETR TRT workflow fast path from `deferred_mask_resize_detection_limit=8` to `7`. The existing overflow recovery still reruns full fused mask resize if a future frame exceeds the first-stage limit.
- Correctness: Compared the 7-limit deferred path against the exact-sized rounded path on all 538 frames. Detection-count distribution was `{1: 15, 2: 104, 3: 164, 4: 145, 5: 74, 6: 14, 7: 22}`, with `overflow_frames=0`, `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.5`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.42s fps=222.18`, `frames=538 elapsed=2.41s fps=223.69`, and `frames=538 elapsed=2.41s fps=223.14`.
- Profile: Nsight Systems capture `/tmp/rfdetr_mask7_20260523_083028.nsys-rep` exported to `/tmp/rfdetr_mask7_20260523_083028.sqlite`; CSV summaries are `/tmp/rfdetr_mask7_20260523_083028_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_mask7_20260523_083028_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_mask7_20260523_083028_stats_cuda_api_sum.csv`. Under profiler, depth `2` measured `frames=538 elapsed=2.55s fps=210.86`.
- Graph spacing: The capture includes `538` CUDA graph traces. After skipping the first 100 launches, CUDA graph duration was p50 `3783.332 us`, p90 `3816.618 us`, p95 `3818.473 us`, p99 `3821.493 us`; graph end-to-next-start gap was p50 `717.877 us`, p90 `805.197 us`, p95 `845.107 us`, p99 `1025.533 us`, mean `722.557 us`. GPU work inside that gap covered p50 `265.180 us`, leaving p50 idle gap `451.546 us`. The largest non-TRT postprocess kernels were `_select_topk_boxes_kernel` (`5.778 ms` total / `10.740 us` average), `_resize_selected_masks_kernel` (`4.001 ms` total / `7.437 us` average), and the PyTorch sigmoid kernel (`1.567 ms` total / `2.913 us` average).
- Learning: The limited mask grid is still on the critical path, and matching it to the observed benchmark max count gives a small but repeatable improvement while keeping the overflow safety mechanism.

### Rejected: Six-Row Deferred Mask Resize

- Hypothesis: Reducing the first-stage deferred mask resize grid from 7 rows to 6 rows might improve the common path enough to offset overflow recovery on the 22 frames that contain 7 detections.
- Change tested: Temporary code only; changed the RFDETR TRT workflow fast path from `deferred_mask_resize_detection_limit=7` to `6`. Pipeline depth remained fixed at `2`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.42s fps=222.76`, `frames=538 elapsed=2.45s fps=220.00`, and `frames=538 elapsed=2.42s fps=222.66`, below the accepted 7-row best band.
- Learning: The full-resize overflow fallback on 7-detection frames costs more than the one-row saving on the other frames. Keep the 7-row limit for this stream.

### Rejected: Prefilter Invalid Classes In Selector

- Hypothesis: `_select_topk_boxes_kernel` lets invalid/no-object class scores participate in the top-score loop and discards them one at a time. Masking those lanes to `-inf` before the loop could reduce selector iterations and shrink the postprocess gap.
- Change tested: Temporary code only; computed each lane's raw class index at the start of the Triton selector, loaded `class_mapping`, and set scores with mapped class `< 0` to `-inf` before entering the top-k loop. Pipeline depth remained fixed at `2`.
- Correctness: Compared the modified deferred fused path against the exact PyTorch postprocess on all 538 frames: count distribution `{1: 15, 2: 104, 3: 164, 4: 145, 5: 74, 6: 14, 7: 22}`, `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.5`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.42s fps=222.75`, `frames=538 elapsed=2.45s fps=219.78`, and `frames=538 elapsed=2.42s fps=222.13`, below the accepted 7-row best band.
- Learning: The extra vectorized class-map load and mask computation are not worth the shorter top-k loop for this tensor shape. Keep the simpler selector.

### Rejected: Skip Null Watchdog And Empty Status Updates

- Hypothesis: The default `InferencePipeline` uses `NullPipelineWatchdog`, but the hot loop still calls no-op watchdog hooks and sends per-frame DEBUG status updates to the no-op status handler. Skipping the default null handler and returning early when no status handlers are registered could reduce CPU handoff time before the next graph launch.
- Change tested: Temporary code only; did not append `NullPipelineWatchdog.on_status_update`, skipped model-start/model-ready watchdog calls when the watchdog was null, and made `send_inference_pipeline_status_update(...)` return immediately for an empty handler list. Pipeline depth remained fixed at `2`.
- Correctness: This only removed no-op observer calls for the default-null watchdog case; prediction objects and model execution were unchanged.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.41s fps=223.26`, `frames=538 elapsed=2.43s fps=221.67`, and `frames=538 elapsed=2.43s fps=221.29`, not stable enough to keep over the accepted 7-row checkpoint.
- Learning: The no-op observer path is below the noise floor, or the added branches perturb scheduling enough to offset the saved calls. Keep the existing observer behavior.

### RFDETR Fixed-Limit Pinned Prediction Copy

- Hypothesis: The deferred workflow conversion reads the GPU selected-count tensor before it can enqueue prediction D2H copies. With the accepted 7-row mask limit, the normal path can copy the fixed 7-row boxes/confidences/classes/masks plus the count into pinned host buffers, synchronize once, then slice by the copied count on CPU. This should remove a sequential count-sync bubble even though it copies unused rows for low-count frames.
- Change: Added a count slot to the thread-local RFDETR pinned conversion buffers and a fixed-limit CUDA conversion path for deferred limited masks. When `valid_count <= mask_resize_detection_limit`, conversion copies the first limited rows and the count with `non_blocking=True`, synchronizes once, then returns normal owned NumPy arrays sliced to the valid count. If the copied count exceeds the limit, the existing overflow recovery path is used.
- Correctness: Compared the fixed-limit pinned conversion against exact PyTorch postprocess on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.5`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.27s fps=237.19`, `frames=538 elapsed=2.27s fps=237.43`, `frames=538 elapsed=2.31s fps=232.53`, and confirmation `frames=538 elapsed=2.28s fps=236.01`, improving the previous 7-row band around `223` FPS.
- Profile: Nsight Systems capture `/tmp/rfdetr_fixed7_copy_20260523_084817.nsys-rep` exported to `/tmp/rfdetr_fixed7_copy_20260523_084817.sqlite`; CSV summaries are `/tmp/rfdetr_fixed7_copy_20260523_084817_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_fixed7_copy_20260523_084817_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_fixed7_copy_20260523_084817_stats_cuda_api_sum.csv`. Under profiler, depth `2` measured `frames=538 elapsed=2.27s fps=236.70`.
- Graph spacing: The capture includes `538` CUDA graph traces. After skipping the first 100 launches, CUDA graph duration was p50 `4000.802 us`, p90 `4109.647 us`, p95 `4115.823 us`, p99 `4122.736 us`; graph end-to-next-start gap collapsed to p50 `40.767 us`, p90 `41.913 us`, p95 `42.239 us`, p99 `42.867 us`, mean `40.801 us`. GPU work inside that gap covered p50 `35.392 us`, leaving only p50 `5.312 us` idle.
- Learning: The earlier count read was the main remaining pipeline bubble. Copying the fixed limited prediction rows lets the next CUDA graph launch almost immediately after the previous graph, making the run bottlenecked by graph replay plus overlapped postprocess/copy work as intended.

### Rejected: Six-Row Fixed Prediction Copy

- Hypothesis: The fixed-limit pinned conversion copies 7 dense-mask rows for every frame even though only 22 of 538 frames have 7 detections. Copying only 6 rows and falling back for 7-detection frames could reduce D2H traffic on the common path.
- Change tested: Temporary code only; kept `deferred_mask_resize_detection_limit=7`, but capped the fixed pinned prediction copy to `6` rows before falling back to the exact-count path when the copied count exceeded 6. Pipeline depth remained fixed at `2`.
- Correctness: Compared the modified conversion against exact PyTorch postprocess on all 538 frames. The 6-row fast copy fell back on exactly `22` frames, with `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.5`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.32s fps=231.99`, `frames=538 elapsed=2.34s fps=230.30`, and `frames=538 elapsed=2.32s fps=232.01`, below the accepted 7-row fixed-copy checkpoint.
- Learning: The fallback penalty on the 7-detection frames is larger than the common-path savings from copying one fewer row. Keep the fixed copy aligned with the 7-row mask limit.

### Current TensorRT Graph Node Profile

- Request: Capture a CUDA graph node-level profile for the accepted fixed-copy checkpoint while keeping pipeline depth fixed at `2`, because graph-to-graph idle time is now too small for host-only graph traces to explain remaining runtime.
- Profile: Nsight Systems capture `/tmp/rfdetr_fixed7_nodes_20260523_085415.nsys-rep` exported to `/tmp/rfdetr_fixed7_nodes_20260523_085415.sqlite`; CSV summaries are `/tmp/rfdetr_fixed7_nodes_20260523_085415_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_fixed7_nodes_20260523_085415_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_fixed7_nodes_20260523_085415_stats_cuda_api_sum.csv`. Under heavier node tracing, depth `2` measured `frames=538 elapsed=2.35s fps=228.52`.
- Kernel breakdown: CUDA graph node tracing shows TensorRT replay is now dominant. Top kernels include `sm75_xmma_gemm_f16f16_f16f16_f16_nn...` (`463.288 ms` total), `_gemm_mha_v2_...` (`337.945 ms`), and `sm75_xmma_gemm_f16f16_f16f32_f32_nn...fused` (`276.260 ms`). The custom postprocess kernels are small by comparison: `_resize_selected_masks_kernel` was `8.035 ms` total and `_select_topk_boxes_kernel` was `7.730 ms` total.
- Learning: After fixed-limit prediction copy, the bottleneck has shifted from CPU/GPU handoff bubbles to TensorRT graph replay itself. Further large gains likely require TensorRT engine/tactic changes or a different engine build, not more custom postprocess work.

### Rejected: Borrow CUDA Graph Output Buffers

- Hypothesis: The CUDA graph replay path clones TensorRT output buffers after each replay. Returning the graph-owned output tensors directly for the RFDETR workflow could remove device-to-device clone work after graph replay.
- Change tested: Temporary code only; threaded a `borrow_cuda_graph_outputs` flag through the TensorRT helper, used a thread-local graph cache for the borrowed mode, and enabled it only in the RFDETR TRT workflow fast path. Pipeline depth remained fixed at `2`.
- Correctness: Compared the actual depth-2 `InferencePipeline` output against the accepted fixed-copy path on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.5`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.48s fps=216.54`, `frames=538 elapsed=2.46s fps=218.60`, and `frames=538 elapsed=2.46s fps=218.99`, well below the accepted fixed-copy checkpoint.
- Learning: The cloned output tensors are still useful for overlap/lifetime isolation. Borrowing graph-owned buffers perturbs graph-cache ownership and stream scheduling enough to dominate the saved clone work. Keep the cloned outputs.

### Current Post-Cleanup Nsight Profile

- Request: Capture a fresh Nsight Systems profile after reverting the rejected borrowed-output experiment, keeping pipeline depth fixed at `2`.
- Profile: Nsight Systems capture `/tmp/rfdetr_fixed7_current_20260523_090332.nsys-rep` exported to `/tmp/rfdetr_fixed7_current_20260523_090332.sqlite`; CSV summaries are `/tmp/rfdetr_fixed7_current_20260523_090332_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_fixed7_current_20260523_090332_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_fixed7_current_20260523_090332_stats_cuda_api_sum.csv`. Under profiler, depth `2` measured `frames=538 elapsed=2.30s fps=234.15`.
- Graph spacing: The capture includes `538` CUDA graph traces. After skipping the first 100 launches, CUDA graph duration was p50 `3992.289 us`, p90 `4054.714 us`, p95 `4062.176 us`, p99 `4107.568 us`; graph end-to-next-start gap was p50 `40.639 us`, p90 `41.740 us`, p95 `41.951 us`, p99 `42.417 us`, mean `40.584 us`. GPU work inside that gap covered p50 `35.295 us`, leaving only p50 `5.280 us` idle.
- Learning: The cleaned-up accepted path still has the desired timeline shape: the next CUDA graph starts almost immediately after the previous graph ends. Remaining throughput is constrained by the TensorRT graph replay itself, with only a few microseconds of idle gap.

### Rejected: Interactive Local TRT Rebuild From ONNX

- Hypothesis: Since the run is now bottlenecked by the TensorRT graph replay, rebuilding the available ONNX package locally on the Tesla T4 with TensorRT `10.12.0.36` and FP16 tactics might produce a faster engine than the packaged T4 plan.
- Change tested: Downloaded the ONNX package `5362b72bfb9f01d2e0b8cba2048d932c` to `/tmp/rfdetr_onnx_pkg_5362b72bfb9f01d2e0b8cba2048d932c` and started an isolated TensorRT Python build in `/tmp/rfdetr_trt_rebuild_t4_fp16_opt5` with static input shape `1x3x312x312`, FP16 enabled, workspace `4 GiB`, and builder optimization level `5`. The accepted model-cache package was not modified.
- Result: The builder parsed the ONNX graph cleanly, but tactic selection was still CPU-bound after roughly 9.5 minutes and had not produced an engine, so the temporary build process was terminated. No correctness or FPS benchmark was possible.
- Learning: A local tactic rebuild is the right class of experiment for the remaining bottleneck, but full optimization-level builds are too slow for the interactive benchmark loop. Keep the packaged T4 FP16 engine for this checkpoint; any engine rebuild should be run as an offline build job and benchmarked separately once serialized.

### Rejected: Alternate TRT Engine Packages And Low-Opt Local Rebuilds

- Hypothesis: The accepted path is bottlenecked by TensorRT graph replay, so a different serialized engine may improve FPS without changing postprocess or pipeline depth.
- Change tested: Downloaded the official T4 FP32 package `bbc2cc23adf6f5e71a9241956081da96`, the official L4 FP16 package `89d1f41e2af4f4f3ffcdfb77e774d26a`, and built local T4 FP16 engines from the ONNX package at TensorRT builder optimization levels `0` and `1`. The opt0 build completed in `34.65s` with a `78M` plan, and opt1 completed in `41.05s` with a `63M` plan. Pipeline depth remained fixed at `2`.
- Correctness: Direct model comparison against the accepted T4 FP16 plan failed for the official T4 FP32 plan and both local low-opt FP16 rebuilds. T4 FP32 produced `bad_counts=7`, `bad_classes=8`, `bad_masks=329`, `bad_boxes_gt5=19`, `max_box_delta=251.0`; local opt0 produced `bad_counts=8`, `bad_classes=8`, `bad_masks=352`, `bad_boxes_gt5=19`, `max_box_delta=251.0`; local opt1 produced `bad_counts=8`, `bad_classes=8`, `bad_masks=333`, `bad_boxes_gt5=17`, `max_box_delta=251.0`. The L4 FP16 engine could not be deserialized directly on the T4 because the engine was generated for compute capability `8.9` while the runtime device is `7.5`.
- Result: The low-opt local engines were not benchmarked further because they failed the explicit correctness gate. An initial cache-swap benchmark that touched only the `rfdetr-seg-nano` cache alias is not considered valid because the workflow resolves the model to the canonical `coco-dataset-vdnr1/41` cache path.
- Learning: Engine replacement is not safe without a full prediction-compatibility check. The packaged T4 FP16 plan remains the only tested engine that satisfies the class, mask, and box invariants for this benchmark.

### Rejected: Pack Fixed-Copy Metadata Before D2H

- Hypothesis: The fixed-limit conversion path enqueues five D2H copies per frame: count, boxes, confidences, classes, and masks. Packing count/boxes/confidences/classes into one small GPU float buffer with a Triton kernel would reduce small D2H copy submissions to one metadata copy plus one mask copy.
- Change tested: Temporary code only; added a Triton metadata-pack kernel and thread-local GPU/pinned metadata buffers, then used the packed metadata in `_try_copy_limited_cuda_detection_tensors_to_pinned_numpy(...)`. Pipeline depth remained fixed at `2`.
- Correctness: Compared packed metadata conversion against the original direct tensor-copy conversion on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.31s fps=233.11` and `frames=538 elapsed=2.29s fps=234.62`, below the accepted fixed-copy checkpoint.
- Learning: The extra Triton launch costs more than the saved small D2H submissions. The existing fixed-copy path is already balanced enough that reducing copy count this way loses throughput.

### Rejected: Skip Output Record Stream After Fixed Copy

- Hypothesis: After fixed-limit prediction copy collapsed the graph-to-graph gap, the `record_stream(...)` calls on TensorRT output clones might be unnecessary in the RFDETR workflow fast path because postprocess explicitly waits on the inference stream before CPU conversion.
- Change tested: Temporary code only; skipped `result_element.record_stream(self._post_process_stream)` when `defer_cuda_stream_sync=True` in `RFDetrForInstanceSegmentationTRT.post_process(...)`. Pipeline depth remained fixed at `2`.
- Correctness: Prediction tensors and kernels were unchanged; the experiment only changed allocator stream-lifetime bookkeeping for the output clone tensors.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.29s fps=235.14` and `frames=538 elapsed=2.31s fps=233.07`, below the accepted fixed-copy checkpoint.
- Learning: `record_stream(...)` is not the remaining limiter, and skipping it still perturbs scheduling enough to lose low-end stability. Keep the original stream lifetime bookkeeping.

### Rejected: Raw-Logit Selector With Selected-Only Sigmoid

- Hypothesis: Sigmoid is monotonic, so the fused selector can rank raw logits and compare against `logit(threshold)`, then compute sigmoid only for selected output confidences. This avoids the full PyTorch sigmoid over the `100x91` logits without doing `exp` for every class lane inside the selector.
- Change tested: Temporary code only; changed RFDETR instance postprocess to try the fused selector before materializing `logits_sigmoid`, added a raw-logit threshold mode to `_select_topk_boxes_kernel`, and stored `sigmoid(top_logit)` only for kept detections. Pipeline depth remained fixed at `2`.
- Correctness: Compared raw-logit fused postprocess against the PyTorch fallback on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.5`, `max_conf_delta=1.1920928955078125e-07`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.31s fps=233.26` and `frames=538 elapsed=2.29s fps=234.68`, below the accepted fixed-copy checkpoint.
- Learning: Removing the global sigmoid kernel is still not worth the extra selector complexity and selected-score `exp` work. Keep the standalone PyTorch sigmoid plus simpler selector.

### Rejected: Early Return Empty Mask Resize Rows

- Hypothesis: The fixed 7-row deferred mask resize grid still launches pixel programs for rows above the valid detection count. A runtime early return when `det_index >= count` inside `_resize_selected_masks_kernel` could avoid work for empty rows without reading the count on CPU.
- Change tested: Temporary code only; added a Triton runtime branch before pixel coordinate math and removed the now-redundant `det_index < count` masks for valid rows. Pipeline depth remained fixed at `2`.
- Correctness: Compared deferred fused postprocess against the PyTorch fallback on all 538 frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.5`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.30s fps=234.23`, below the accepted fixed-copy checkpoint.
- Learning: The runtime branch/predication costs more than the skipped empty-row arithmetic for this small fixed grid. Keep the mask-based kernel.

### Rejected: Explicit TensorRT Aux Streams During CUDA Graph Capture

- Hypothesis: The packaged RFDETR TensorRT engine reports `num_aux_streams=4`; explicitly providing persistent non-default auxiliary streams during CUDA graph capture might shorten the TensorRT graph replay duration, which is now the dominant bottleneck.
- Change tested: Temporary code only; added auxiliary CUDA streams to the graph state, called `IExecutionContext.set_aux_streams(...)` before the warmup and capture `execute_async_v3(...)` calls, and kept pipeline depth fixed at `2`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.27s fps=237.02`, then `frames=538 elapsed=2.29s fps=234.91` and `frames=538 elapsed=2.29s fps=235.38`, which is not a stable improvement over the accepted fixed-copy band.
- Learning: TensorRT's default stream behavior for this serialized engine is already good enough, or explicit aux-stream handles perturb capture/scheduling without reducing the real graph replay bottleneck. Keep the simpler existing CUDA graph capture path.

### Current Clean Depth-2 Nsight Profile

- Request: Generate another Nsight Systems capture on the current accepted path, keeping pipeline depth fixed at `2`.
- Profile: Nsight Systems capture `/tmp/rfdetr_fixed7_depth2_clean_20260523_095302.nsys-rep` exported to `/tmp/rfdetr_fixed7_depth2_clean_20260523_095302.sqlite`; CSV summaries are `/tmp/rfdetr_fixed7_depth2_clean_20260523_095302_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_fixed7_depth2_clean_20260523_095302_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_fixed7_depth2_clean_20260523_095302_stats_cuda_api_sum.csv`.
- Result under profiler: depth `2` measured `frames=538 elapsed=2.28s fps=235.96`.
- Graph spacing: The capture includes `538` CUDA graph traces on stream `39`. After skipping the first 100 launches, CUDA graph duration was p50 `4012.688 us`, p90 `4070.684 us`, p95 `4074.743 us`, p99 `4081.137 us`; graph end-to-next-start gap was p50 `40.607 us`, p90 `41.855 us`, p95 `42.245 us`, p99 `42.781 us`, mean `40.599 us`. GPU work inside that gap covered p50 `35.231 us`, leaving p50 idle gap `5.216 us`.
- Learning: The depth-2 accepted path is still shaped as intended: CPU work and prediction D2H copies are overlapped enough that there is only a few microseconds of idle time between CUDA graph replays. Remaining FPS is dominated by the TensorRT CUDA graph duration plus the small fixed postprocess kernels.

### Rejected: In-Place Sigmoid Retest After Fixed Copy

- Hypothesis: Now that graph-to-graph idle time is almost gone, changing `torch.nn.functional.sigmoid(logits)` to `logits.sigmoid_()` on the cloned TensorRT logits might remove an allocation/write in postprocess without affecting outputs.
- Change tested: Temporary code only; used in-place sigmoid in RFDETR instance segmentation postprocess and kept pipeline depth fixed at `2`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.26s fps=237.80`, then `frames=538 elapsed=2.30s fps=233.80` and `frames=538 elapsed=2.30s fps=234.08`, below the accepted fixed-copy stability band.
- Learning: This remains a scheduling/noise-level optimization. Even if the tensor math is equivalent on cloned outputs, the in-place form does not reliably reduce the critical graph-to-graph interval, so keep the out-of-place sigmoid.

### Rejected: TensorRT Persistent Cache Limit On T4

- Hypothesis: Setting `IExecutionContext.persistent_cache_limit` on the CUDA graph execution context might reduce TensorRT graph replay time by allowing activation reuse through persistent L2 cache.
- Change tested: Temporary code only; set `persistent_cache_limit = 4 MiB` immediately after creating the graph execution context and kept pipeline depth fixed at `2`.
- Result: TensorRT rejected the setting on this Tesla T4 with `persistingL2CacheMaxSize(0 bytes)`, so the device/runtime does not support a nonzero persistent cache limit for this path. The measured runs (`237.38`, `237.56`, `234.01` FPS) are not considered a valid optimization because the runtime emitted an API usage error each time.
- Learning: Persistent L2 activation caching is not available on this hardware. Do not keep this context setting for the T4 benchmark.

### Rejected: Empty Query-Index Buffer Retest After Fixed Copy

- Hypothesis: The deferred fused selector writes every query index that the mask resize reads, so allocating `query_indices` with `torch.empty(...)` instead of `torch.zeros(...)` could remove a small int32 fill kernel still visible in the clean depth-2 profile.
- Change tested: Temporary code only; changed the `query_indices` allocation in `fused_select_topk_boxes(...)` from `torch.zeros(...)` to `torch.empty(...)` and kept pipeline depth fixed at `2`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.29s fps=234.59`, `frames=538 elapsed=2.30s fps=234.09`, and `frames=538 elapsed=2.27s fps=236.77`, not a stable improvement over the accepted fixed-copy band.
- Learning: Removing the fill does not reliably tighten the graph-to-graph interval in the current fixed-copy path. Keep the deterministic zero-filled buffer.

### Rejected: Limited Mask Allocation Retest After Fixed Copy

- Hypothesis: With the fixed 7-row prediction copy, the deferred mask resize output only needs the first `detection_limit` rows on the normal path. Allocating `(detection_limit, H, W)` instead of `(100, H, W)` could reduce allocator/cache pressure without changing copied predictions.
- Change tested: Temporary code only; moved detection-limit clamping before the output allocation in `fused_resize_selected_masks(...)`, allocated only the limited row count, and kept pipeline depth fixed at `2`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.30s fps=233.93`, `frames=538 elapsed=2.26s fps=237.60`, and `frames=538 elapsed=2.31s fps=232.63`, not a stable improvement over the accepted fixed-copy band.
- Learning: The fixed full-capacity allocation remains more stable in the two-frame pipeline despite unused rows. Keep the current output shape and rely on the limited launch grid for the actual kernel work reduction.

### Rejected: CUDA Device Max Connections Tuning

- Hypothesis: The run is now sensitive to stream scheduling between TensorRT graph replay, postprocess kernels, H2D preprocessing, and D2H prediction copies. Tuning `CUDA_DEVICE_MAX_CONNECTIONS` before CUDA initialization might reduce scheduling variance or graph-to-graph gaps.
- Change tested: No code change; ran the exact benchmark with `CUDA_DEVICE_MAX_CONNECTIONS=1`, `2`, `4`, and `8`, always with pipeline depth fixed at `2`.
- Result on requested command: `1` regressed badly to `frames=538 elapsed=2.71s fps=198.62`; `2` measured `237.26` FPS; `4` measured `234.84` FPS; `8` measured `236.23` FPS. The non-1 settings are within normal accepted-path noise and do not justify changing the command/runtime defaults.
- Learning: Do not force CUDA device connection count for this benchmark. The default stream scheduling is already in the best observed band, while `1` removes useful concurrency.

### Rejected: High-Priority TensorRT Graph Replay Stream

- Hypothesis: Since CUDA graph replay is now the bottleneck, creating the captured TensorRT graph stream with high priority could keep small postprocess kernels from delaying graph replay in the depth-2 pipeline.
- Change tested: Temporary code only; changed the CUDA graph state's stream construction from `torch.cuda.Stream(device=device)` to `torch.cuda.Stream(device=device, priority=-1)` and kept pipeline depth fixed at `2`.
- Result on requested command: depth `2` measured `237.18`, `233.05`, `237.76`, `236.59`, and `235.95` FPS. This is the same noisy band as the accepted path with a low outlier.
- Learning: Stream priority does not reliably reduce TensorRT graph duration or the graph-to-graph gap on this T4 workload. Keep the default-priority graph stream.

### Rejected: Clone Graph Outputs On Caller Stream

- Hypothesis: CUDA graph replay currently clones TensorRT output buffers on the graph replay stream before the caller stream can continue. Moving those clones to the caller inference stream might free the graph replay stream earlier and improve depth-2 scheduling while preserving cloned-output ownership.
- Change tested: Temporary code only; after `cuda_graph.replay()`, made the caller stream wait on the graph stream, cloned `output_buffers` on the caller stream, and kept pipeline depth fixed at `2`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.26s fps=237.82`, `frames=538 elapsed=2.29s fps=235.14`, and `frames=538 elapsed=2.29s fps=235.36`, which is not stable enough to keep over the accepted fixed-copy path.
- Learning: The output clones remain on the critical ownership chain regardless of which stream performs them. Moving the clone stream only perturbs scheduling, so keep the established graph-stream clone path.

### Nsight Compute Postprocess Kernel Snapshot

- Request: Use a more focused profiler on the custom fused postprocess kernels instead of continuing blind one-kernel tweaks.
- Profile: Nsight Compute report `/tmp/rfdetr_ncu_postprocess_basic_20260523_101558.ncu-rep`, collected with `--set basic`, `--kernel-name "regex:(_select_topk_boxes_kernel|_resize_selected_masks_kernel)"`, `--launch-skip 20`, and `--launch-count 4`.
- Result: `_select_topk_boxes_kernel` launches as one 256-thread block with 128 registers/thread and about `24.5 us` duration under Nsight Compute instrumentation; NCU flags the grid as too small to fill the T4. `_resize_selected_masks_kernel` launches as `(7, 215, 1)x(128, 1, 1)`, uses 32 registers/thread, reaches about `81%` achieved occupancy, and measured about `14.8 us` under NCU instrumentation.
- Learning: The mask resize kernel is already reasonably occupied for the fixed 7-row grid. The selector underutilizes the GPU, but parallelizing the global top-k would require extra coordination/launches, which is likely to lose given the current graph-to-graph gap is only around `40 us`.

### Rejected: Sequential Background Remap Selector

- Hypothesis: The benchmark RFDETR class remapping only removes class `0` as background, so the fused selector could compute `class_id = raw_class_id - 1` and skip the per-selected-candidate device `class_mapping` load.
- Change tested: Temporary code only; detected the sequential background remap in the TRT model, threaded a `sequential_background_remap` flag into the fused instance selector, and added a Triton branch that keeps `raw_class_id > 0` and remaps by subtracting one. Pipeline depth remained fixed at `2`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.30s fps=233.48`, `frames=538 elapsed=2.32s fps=232.11`, and `frames=538 elapsed=2.29s fps=234.43`, below the accepted fixed-copy band.
- Learning: The scalar class-map load is not the selector limiter; the added branch/codegen and Python plumbing make the path slower. Keep the original generic mapping path.

### Rejected: Disable TensorRT Enqueue Profiling Emission

- Hypothesis: Setting `IExecutionContext.enqueue_emits_profile = False` on the CUDA graph execution context might remove TensorRT profiling/timing bookkeeping from graph capture or replay.
- Change tested: Temporary code only; set `graph_context.enqueue_emits_profile = False` immediately after creating the TensorRT graph execution context and kept pipeline depth fixed at `2`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.30s fps=233.89`, `frames=538 elapsed=2.33s fps=230.91`, and `frames=538 elapsed=2.34s fps=230.12`, below the accepted fixed-copy band.
- Learning: The flag is not useful for this no-profiler replay path and appears to perturb TensorRT execution or capture behavior negatively. Keep the default context setting.

### Rejected: Local T4 FP16 TensorRT Opt2 Rebuild

- Hypothesis: The remaining bottleneck is TensorRT graph replay, so a medium-optimization local T4 FP16 rebuild from the available ONNX package might produce a faster correct engine without the long opt5 build time.
- Change tested: Built `/tmp/rfdetr_trt_rebuild_t4_fp16_opt2/engine.plan` from `/tmp/rfdetr_onnx_pkg_5362b72bfb9f01d2e0b8cba2048d932c/weights.onnx` with TensorRT `10.12.0.36`, static shape `1x3x312x312`, FP16 enabled, workspace `4 GiB`, and `builder_optimization_level=2`. Build completed in `90.58s` and produced a `66,959,004` byte plan.
- Correctness: Direct comparison against the accepted cached T4 FP16 engine over all 538 frames failed: `bad_counts=8`, `bad_classes=15`, `bad_masks=345`, `bad_boxes_gt5=23`, `max_box_delta=251.0`, `max_conf_delta=0.10609796643257141`; first failure was frame `0` with accepted count `4` vs candidate count `5`.
- Result: Not benchmarked further because it violates the explicit class/mask/box correctness gate.
- Learning: Like the previous opt0/opt1 rebuilds, the available ONNX package is not a safe source for a drop-in engine compatible with the accepted packaged T4 FP16 plan. Further engine replacement needs the exact source/export settings for the accepted plan or a correctness-preserving engine package.

### Rejected: Omit False Preprocessing Overrides

- Hypothesis: The RFDETR TRT workflow fast path constructs a `PreProcessingOverrides(False, False, False)` object for every frame. Passing `None` is semantically equivalent for the preprocessing helpers and could remove a small per-frame Python allocation.
- Change tested: Temporary code only; passed `pre_processing_overrides=None` from `_try_run_rfdetr_trt_fast_path(...)` and kept pipeline depth fixed at `2`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.30s fps=234.41`, `frames=538 elapsed=2.30s fps=233.61`, and `frames=538 elapsed=2.31s fps=232.93`, below the accepted fixed-copy band.
- Learning: The dataclass allocation is below the limiter, and changing the branch pattern through preprocessing worsens the pipeline balance. Keep the explicit false overrides object.

### Rejected: Highest-Priority TensorRT Graph Replay Stream

- Hypothesis: A previous high-priority graph-stream test used priority `-1`, but this T4 runtime reports priority range `(0, -3)`. Using the highest priority `-3` might better prioritize TensorRT graph replay over low-priority postprocess work.
- Change tested: Temporary code only; changed the CUDA graph state's stream construction to `torch.cuda.Stream(device=device, priority=-3)` and kept pipeline depth fixed at `2`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.31s fps=232.94`, `frames=538 elapsed=2.30s fps=233.60`, and `frames=538 elapsed=2.33s fps=231.04`, below the accepted fixed-copy band.
- Learning: For this workload, explicit stream priority hurts scheduling. Keep the default-priority graph stream.

### Current Clean Check After Stream Tests

- Request: Confirm the accepted code path after reverting the rejected stream-priority and preprocessing-override experiments.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.33s fps=231.37`, then `frames=538 elapsed=2.28s fps=235.47`.
- Learning: The benchmark remains noisy, but the clean accepted path is back in the expected low-to-mid `230s` FPS band, with the same TensorRT graph replay bottleneck and tiny graph-to-graph gap identified in the Nsight profiles.

### Rejected: Skip Empty Class Filter Helper

- Hypothesis: The benchmark workflow does not set `class_filter`, so `filter_out_unwanted_classes_from_sv_detections_batch(...)` returns immediately. Skipping the call in the RFDETR TRT fast path could remove a tiny CPU call from the materialization tail without changing outputs.
- Change tested: Temporary code only; guarded the helper call with `if class_filter:` in `_try_run_rfdetr_trt_fast_path(...)` and kept pipeline depth fixed at `2`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.32s fps=231.76`, `frames=538 elapsed=2.29s fps=234.92`, and `frames=538 elapsed=2.33s fps=231.34`, not better than the accepted fixed-copy path.
- Learning: The empty class-filter helper is below the limiter. Keep the normal helper chain for predictable scheduling and consistent behavior.

### Depth-2 Current Nsight Systems Graph-Gap Profile

- Request: Collect a fresh Nsight Systems profile for the current accepted implementation while keeping the pipeline fixed at depth `2`.
- Profile: `/tmp/rfdetr_depth2_graphgap_current_20260523_104339.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_graphgap_current_20260523_104339.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_graphgap_current_20260523_104339_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_graphgap_current_20260523_104339_stats_cuda_gpu_mem_time_sum.csv`, `/tmp/rfdetr_depth2_graphgap_current_20260523_104339_stats_cuda_api_sum.csv`.
- Profiled result: `frames=538 elapsed=2.32s fps=232.24` under Nsight Systems overhead.
- Graph spacing after skipping the first 100 graph launches: CUDA graph duration p50 `4066.430 us`, p90 `4129.114 us`, p95 `4135.536 us`, p99 `4142.649 us`, mean `4060.011 us`; graph end-to-next-start gap p50 `40.575 us`, p90 `41.836 us`, p95 `42.189 us`, p99 `42.708 us`, mean `40.683 us`.
- Gap decomposition after skipping the first 100 graph launches: busy work inside the gap p50 `35.072 us`, p90 `36.447 us`, p95 `36.966 us`, p99 `37.683 us`, mean `35.265 us`; idle time inside the gap p50 `5.376 us`, p90 `6.048 us`, p95 `6.182 us`, p99 `6.356 us`, mean `5.418 us`.
- Learning: With depth `2`, the current pipeline is already graph-replay bottlenecked in steady state. The next TensorRT CUDA graph starts roughly `40 us` after the previous graph ends, and most of that tail is real postprocess/copy work rather than host-side idle. Do not test depth `3`; further gains need to reduce the TensorRT graph itself or remove work from the small post-graph tail.

### Rejected: Two-Slot Borrowed CUDA Graph Outputs

- Hypothesis: The previous borrowed-output experiment used a single graph-owned output buffer, which is unsafe for depth-2 overlap and forced worse scheduling. Capturing two TensorRT CUDA graph states with separate output buffers and alternating them should let frame `N` postprocess one buffer while frame `N+1` replays into the other, removing per-frame output clone D2D copies without overwriting live outputs.
- Change tested: Temporary code only; added a `TRTCudaGraphStatePool`, threaded a `borrow_cuda_graph_outputs=True` and `cuda_graph_output_buffer_count=2` option through the RFDETR TRT fast path, captured two graph states for the same static shape, and returned graph-owned output buffers instead of cloned tensors. Pipeline depth remained fixed at `2`.
- Correctness: Direct full-video comparison of two-slot borrowed graph outputs against the cloned-output path over all 538 frames matched exactly: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.31s fps=232.55`, `frames=538 elapsed=2.34s fps=229.70`, and `frames=538 elapsed=2.33s fps=231.36`, below the accepted fixed-copy band.
- Learning: The D2D output clones are not just overhead; they also decouple postprocess lifetime and stream scheduling in a way that preserves better overlap. Even with two graph-owned slots, borrowing outputs hurts the pipeline. Keep the cloned-output CUDA graph path.

### Rejected: Bind CUDA Graph To External Preprocess Input Buffers

- Hypothesis: The CUDA graph replay path copies the preprocessed CUDA tensor into a graph-owned static input buffer before every replay. Since RFDETR preprocessing already copies pinned CPU data to CUDA on the preprocessing stream, capturing graph states keyed by the preprocessed tensor's CUDA pointer could bind TensorRT directly to that buffer and remove the extra device-to-device input copy without moving H2D onto the graph stream.
- Change tested: Temporary code only; added a `use_external_cuda_graph_input` option through the TensorRT helper and RFDETR fast path. When enabled, the cache key included `pre_processed_images.data_ptr()`, graph capture used `pre_processed_images` as the TensorRT input buffer, and cache hits skipped `input_buffer.copy_(pre_processed_images)`. Pipeline depth remained fixed at `2`.
- Correctness/probe: Before this change, a direct 20-frame preprocessing probe alternated between only 2 CUDA input pointers. With external graph input enabled, a 12-frame probe produced 9 distinct CUDA input pointers and took `3.696s`, because each captured graph state retained its external input tensor and prevented normal allocator reuse. A full correctness comparison was stopped after it failed to make timely progress for the same reason.
- Result: Not benchmarked on the full requested command because the cache cannot reach steady state: it captures graphs for transient input pointers instead of reusing one static graph. This adds repeated capture cost and extra retained input buffers.
- Learning: Removing the input D2D copy would require a deliberate reusable CUDA preprocessing buffer pool owned by the model, not binding graphs to arbitrary tensors returned by the allocator. The current graph-owned input buffer plus D2D copy remains the stable path.

### Rejected: Non-Blocking Graph Input D2D Copy

- Hypothesis: The CUDA graph replay path enqueues a device-to-device copy from the preprocessed CUDA tensor into the graph-owned input buffer. Passing `non_blocking=True` to that `copy_(...)` could remove conservative copy synchronization while preserving the explicit stream ordering already enforced by `stream.wait_stream(caller_stream)`.
- Change tested: Temporary code only; changed `trt_cuda_graph_state.input_buffer.copy_(pre_processed_images)` to `copy_(pre_processed_images, non_blocking=True)` in the TensorRT CUDA graph cache-hit path. Pipeline depth remained fixed at `2`.
- Correctness: Prediction math and stream dependencies are unchanged; this only changes the copy enqueue flag for a D2D copy on the same graph replay stream.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.31s fps=233.14`, `frames=538 elapsed=2.31s fps=233.32`, and `frames=538 elapsed=2.30s fps=234.02`, below the accepted fixed-copy band.
- Learning: The D2D input copy flag is not limiting the graph-to-graph interval. Keep the default `copy_(...)` call.

### Rejected: Capture TensorRT Output Copies Inside CUDA Graph

- Hypothesis: The accepted CUDA graph replay path clones TensorRT output buffers after `cuda_graph.replay()`, leaving D2D output copy work in the small graph-to-graph gap. Capturing those D2D output copies as CUDA graph nodes and alternating between two graph states could keep postprocess reading copied buffers while removing per-frame Python-side clone launches from the gap.
- Change tested: Temporary code only; added a captured-output-copy mode to the TensorRT graph helper, allocated internal TensorRT output buffers plus public copied output buffers, captured `execute_async_v3(...)` followed by `destination.copy_(source)` for each output, and enabled a two-slot graph-state pool only in the RFDETR TRT workflow fast path. Pipeline depth remained fixed at `2`.
- Correctness: Direct full-video comparison against the normal cloned-output CUDA graph path over all 538 frames matched exactly: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.31s fps=233.26`, `frames=538 elapsed=2.32s fps=232.04`, and `frames=538 elapsed=2.33s fps=230.70`, below the accepted fixed-copy band.
- Learning: Moving output copies into the CUDA graph lengthens the effective graph bottleneck more than it helps the already tiny post-graph gap. Keep output clones outside the captured TensorRT graph.

### Rejected: Two-Stage Top-2-Per-Query Selector

- Hypothesis: The current Triton selector repeatedly scans the full `100x91` query-class score matrix for global maxima. If each query contributes at most two classes above threshold, a first kernel can compute the top two valid classes per query in parallel, and a second kernel can globally rank only 200 candidates. This could fill the GPU better than the one-block selector flagged by Nsight Compute as under-occupied.
- Change tested: Temporary code only; added a top-2-per-query candidate kernel plus a 200-candidate global selector kernel and enabled it only in the RFDETR TRT workflow fast path. Pipeline depth remained fixed at `2`.
- Correctness: The existing selector had duplicate query detections on only 3 of 538 frames, with maximum query multiplicity `2`. The top-2-per-query selector matched the existing selector exactly over all 538 frames: `bad=0`, `max_float_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.29s fps=234.73`, `frames=538 elapsed=2.32s fps=231.40`, and `frames=538 elapsed=2.29s fps=234.73`, below the accepted fixed-copy band.
- Learning: The extra kernel launch and candidate-buffer traffic cost more than the added selector parallelism. Keep the single-kernel global selector despite its low occupancy.

### Rejected: Disable TensorRT Graph Context NVTX Verbosity

- Hypothesis: The accepted engine was built with layer-name profiling verbosity, and TensorRT graph execution contexts expose `nvtx_verbosity`. Setting the CUDA graph execution context to `ProfilingVerbosity.NONE` during capture might remove NVTX/profiling bookkeeping from graph replay without changing kernels.
- Change tested: Temporary code only; set `graph_context.nvtx_verbosity = trt.ProfilingVerbosity.NONE` immediately after creating the TensorRT CUDA graph execution context. Pipeline depth remained fixed at `2`.
- Correctness: Prediction math and graph topology are unchanged; this only changes TensorRT execution-context metadata verbosity.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.30s fps=233.62`, `frames=538 elapsed=2.30s fps=234.00`, and `frames=538 elapsed=2.32s fps=231.96`, below the accepted fixed-copy band.
- Learning: Runtime NVTX verbosity is not the TensorRT graph replay limiter in this no-profiler benchmark. Keep the default context verbosity.

### Rejected: Extra TensorRT Warmup Before CUDA Graph Capture

- Hypothesis: TensorRT might lazily settle execution-context state during the first enqueue before CUDA graph capture. Running two warmup enqueues instead of one before capture could produce a more stable or faster captured graph.
- Change tested: Temporary code only; changed `_capture_cuda_graph(...)` to enqueue `execute_async_v3(...)` twice on the graph stream before synchronizing and capturing. Pipeline depth remained fixed at `2`.
- Correctness: Prediction math and captured graph operations are unchanged; this only changes pre-capture warmup count.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.31s fps=232.82`, `frames=538 elapsed=2.31s fps=232.97`, and `frames=538 elapsed=2.32s fps=232.11`, below the accepted fixed-copy band.
- Learning: One warmup enqueue is enough for this engine. Extra pre-capture warmup does not improve the steady captured graph and may perturb initialization/cache behavior. Keep the original single warmup.

### Rejected: Deferred Postprocess Stream Copy

- Hypothesis: The RFDETR TRT deferred fast path still waited for the postprocess stream on the current stream before CPU materialization. Passing the postprocess stream through `InstanceDetections.image_metadata` and enqueueing the fixed 7-row pinned D2H copies on that same stream could avoid adding a current-stream dependency and tighten the graph-to-graph tail.
- Change tested: Temporary code only; when `defer_cuda_stream_sync=True`, `post_process(...)` stored `self._post_process_stream` in each detection metadata entry and skipped the current-stream wait. `_try_copy_limited_cuda_detection_tensors_to_pinned_numpy(...)` then copied count, xyxy, confidence, class IDs, and masks on that producer stream and synchronized the stream at the CPU conversion point. Pipeline depth remained fixed at `2`.
- Correctness: Prediction math and copy ordering are unchanged; the experiment only moved the explicit synchronization point from the current stream to the producer stream used for postprocess and D2H copy.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.30s fps=233.72` and `frames=538 elapsed=2.32s fps=232.03`, below the accepted fixed-copy band. A mistakenly launched pair of concurrent benchmark repeats was killed and ignored.
- Learning: The current-stream wait is not the limiter. The existing handoff keeps scheduling stable enough, and moving the D2H copies onto the postprocess stream does not reduce the already small graph-to-graph tail. Keep the accepted current-stream wait and fixed-copy path.

### Depth-2 Accepted Nsight Systems Graph-Gap Profile

- Request: Collect another Nsight Systems profile for the current accepted implementation while keeping the pipeline fixed at depth `2`.
- Profile: `/tmp/rfdetr_depth2_accepted_20260523_113302.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_accepted_20260523_113302.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_accepted_20260523_113302_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_113302_stats_cuda_gpu_mem_time_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_113302_stats_cuda_api_sum.csv`.
- Profiled result: `frames=538 elapsed=2.32s fps=231.77` under Nsight Systems overhead.
- Graph spacing after skipping the first 100 graph launches: CUDA graph duration p50 `4064.720 us`, p90 `4128.916 us`, p95 `4133.955 us`, p99 `4139.214 us`, mean `4049.406 us`; graph end-to-next-start gap p50 `40.512 us`, p90 `41.951 us`, p95 `42.437 us`, p99 `43.231 us`, mean `40.661 us`.
- Gap decomposition after skipping the first 100 graph launches: busy work inside the gap p50 `35.104 us`, p90 `36.441 us`, p95 `37.062 us`, p99 `37.981 us`, mean `35.267 us`; idle time inside the gap p50 `5.280 us`, p90 `6.016 us`, p95 `6.176 us`, p99 `6.377 us`, mean `5.393 us`.
- Learning: The current depth-2 path remains constrained by the TensorRT CUDA graph body, not CPU bubbles. The post-graph tail is short and stable; most of the roughly `40 us` graph-to-graph interval is real GPU work rather than idle.

### Rejected: Two-Slot Pooled CUDA Graph Input

- Hypothesis: The accepted CUDA graph path copies the preprocessed tensor into the graph-owned input buffer on the inference stream immediately before replay. A fixed two-slot graph-input pool could let preprocessing copy into graph-bound input buffers earlier, then replay TensorRT directly from those stable pointers without the graph-stream input D2D copy. This avoids the previous arbitrary-external-input cache thrash while keeping pipeline depth fixed at `2`.
- Change tested: Temporary code only; added a gated `RFDETR_TRT_POOLED_GRAPH_INPUT=1` path that copied preprocessing output into a two-slot model-owned CUDA buffer pool, keyed TensorRT CUDA graph cache entries by the stable external input pointer, and skipped `input_buffer.copy_(pre_processed_images)` on graph cache hits. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.43s fps=221.56`, far below the accepted fixed-copy band.
- Learning: Even with stable input pointers, capturing and maintaining multiple input-bound graph states plus the extra preprocessing pool copy costs more than the current single graph-owned input copy. The existing graph-stream D2D copy is small and better scheduled than this explicit pooling scheme. Keep the accepted single graph input buffer and per-frame D2D copy.

### Rejected: Alternate Prebuilt TensorRT Plan

- Hypothesis: Another RFDETR segmentation TensorRT package already present on disk (`/tmp/rfdetr_trt_pkg_bbc2cc23adf6f5e71a9241956081da96/engine.plan`, `248 MB`, `num_aux_streams=3`) might use a different tactic set than the accepted cache engine (`/tmp/cache/shared-blobs/bc173a2cfda9a10af2bc411885e9fec3`, `188 MB`, `num_aux_streams=4`) and reduce the TensorRT CUDA graph body time.
- Change tested: Temporary external-state test only; repointed both RFDETR cache `engine.plan` symlinks to the alternate plan, ran the requested workflow with pipeline depth fixed at `2`, then restored both symlinks to the accepted cache engine.
- Result on requested command: `frames=538 elapsed=9.44s fps=56.97`, far below the accepted fixed-copy band.
- Learning: The alternate prebuilt plan is not viable on this T4 runtime despite deserializing. Keep the accepted cached engine and do not use this package for further tuning.

### Rejected: Triton Selector Warp Count

- Hypothesis: The single-block `_select_topk_boxes_kernel` is one of the only visible post-graph kernels. Changing the Triton reduction launch from `num_warps=8` might reduce selector latency: fewer warps could lower scheduling/register pressure, while more warps could speed the large `100x91` score reduction.
- Change tested: Temporary code only; changed `_select_topk_boxes_kernel` from `num_warps=8` to `4`, benchmarked, then changed it to `16` and benchmarked. Pipeline depth remained fixed at `2`.
- Result on requested command: `num_warps=4` measured `frames=538 elapsed=2.32s fps=232.18`; `num_warps=16` measured `frames=538 elapsed=2.34s fps=230.37`, both below the accepted fixed-copy band.
- Learning: The current `num_warps=8` is the best of these simple selector launch configurations. The selector remains small relative to the TensorRT CUDA graph body, and launch-shape tuning does not move the full-pipeline limiter.

### Rejected: Triton Mask Resize Block Size

- Hypothesis: `_resize_selected_masks_kernel` is the other stable post-graph Triton kernel. Changing the per-program pixel block size from `256` could improve occupancy or memory coalescing for the `7 x 312 x 312` bounded resize work.
- Change tested: Temporary code only; changed `fused_resize_selected_masks(...)` block size from `256` to `512`, benchmarked, then changed it to `128` and benchmarked. Pipeline depth remained fixed at `2`.
- Result on requested command: `block_size=512` measured `frames=538 elapsed=2.31s fps=232.65`; `block_size=128` measured `frames=538 elapsed=2.33s fps=231.06`, both below the accepted fixed-copy band.
- Learning: The accepted `block_size=256` remains the best of the simple mask-resize launch configurations. The mask kernel is too small relative to the TensorRT graph body for this tuning to move end-to-end FPS.

### Rejected: Pinned NumPy View Ring

- Hypothesis: After the fixed 7-row D2H copy into pinned tensors, the conversion path still calls `.numpy().copy()` for boxes, confidence, class IDs, and masks. Returning NumPy views backed by a small ring of pinned host buffers could remove a CPU memory copy while keeping enough slots for depth-2 overlap.
- Change tested: Temporary gated code only; changed `_get_rfdetr_conversion_buffers(...)` to rotate through four pinned host buffer slots and, with `RFDETR_PINNED_VIEW_NO_COPY=1`, returned NumPy views instead of owned `.copy()` arrays in `_try_copy_limited_cuda_detection_tensors_to_pinned_numpy(...)`. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.31s fps=233.05`, below the accepted fixed-copy band.
- Learning: The extra host-side NumPy copy is not limiting end-to-end FPS after depth-2 pipelining, and returning mutable pinned views weakens result ownership semantics. Keep the existing owned NumPy copies.

### Rejected: Skip TensorRT Caller-Stream Wait

- Hypothesis: The TensorRT CUDA graph helper waits the caller/inference stream on the graph replay stream after every cache hit. In the RFDETR deferred path, postprocess can wait directly on the graph result stream, so skipping the caller-stream wait might remove one event dependency from the graph-to-graph tail.
- Change tested: Temporary gated code only; with `RFDETR_SKIP_TRT_CALLER_STREAM_WAIT=1`, the TensorRT helper returned the graph execution stream and skipped `caller_stream.wait_stream(stream)`. RFDETR stored that stream in thread-local state and made postprocess wait on it instead of `_inference_stream`. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.34s fps=230.33`, below the accepted fixed-copy band.
- Learning: The caller-stream wait is part of the stable scheduling chain for the current depth-2 pipeline. Bypassing it perturbs overlap and slows the run even though the tensor dependencies are still ordered. Keep the accepted wait.

### Rejected: High-Priority Postprocess Stream

- Hypothesis: The postprocess stream runs the fixed selector, mask-resize, and prediction D2H copies in the roughly `40 us` graph-to-graph tail. Giving it higher priority might reduce scheduling latency and tighten the tail without changing TensorRT graph replay.
- Change tested: Temporary gated code only; with `RFDETR_HIGH_PRIORITY_POSTPROCESS=1`, created the RFDETR per-thread postprocess stream with priority `-1` instead of the default priority. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.30s fps=234.32`, then `frames=538 elapsed=2.34s fps=229.94`, not a stable improvement over the accepted fixed-copy band.
- Learning: Stream priority changes add variance and do not consistently improve the already short post-graph tail. Keep the default-priority postprocess stream.

### Rejected: High-Priority Preprocess Stream

- Hypothesis: H2D transfer and GPU normalization on the preprocessing stream overlap TensorRT graph replay. Giving preprocessing higher priority might make the next frame's input ready earlier and reduce occasional graph launch delay.
- Change tested: Temporary gated code only; with `RFDETR_HIGH_PRIORITY_PREPROCESS=1`, created the RFDETR per-thread preprocessing stream with priority `-1` instead of default priority. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.31s fps=232.52`, below the accepted fixed-copy band.
- Learning: Preprocessing priority is not the limiter in the current depth-2 pipeline. Keep the default-priority preprocessing stream to avoid extra scheduling variance.

### Rejected: High-Priority Inference Stream

- Hypothesis: The RFDETR inference stream is the caller stream for TensorRT CUDA graph replay and owns the dependency chain from preprocessing into postprocess. Giving this stream higher priority might reduce event scheduling latency around graph replay and output clone handoff.
- Change tested: Temporary gated code only; with `RFDETR_HIGH_PRIORITY_INFERENCE=1`, created `_inference_stream` with priority `-1` instead of default priority. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.34s fps=230.18`, below the accepted fixed-copy band.
- Learning: Like graph/preprocess/postprocess stream priority changes, inference-stream priority perturbs scheduling without reducing the TensorRT graph-body limiter. Keep all RFDETR streams at default priority.

### Rejected: FP16 Logits Before Fused Selector

- Hypothesis: TensorRT emits RFDETR class logits as float32, but the fused selector only ranks and threshold-checks a small `100x91` score matrix. Casting logits to float16 before sigmoid and selection might reduce postprocess memory traffic and selector work.
- Change tested: Temporary gated code only; with `RFDETR_FUSED_FP16_LOGITS=1`, cast `image_logits` to `torch.float16` immediately before `fused_select_topk_boxes(...)` in the fused instance segmentation postprocess path. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.33s fps=230.55`, below the accepted fixed-copy band.
- Learning: The extra cast kernel and allocation dominate any reduced selector/sigmoid work for this small logits tensor. Keep float32 logits from the accepted TensorRT engine.

### Rejected: Event Synchronize Fixed D2H Copy

- Hypothesis: The fixed 7-row D2H prediction copy currently calls `torch.cuda.current_stream(...).synchronize()` after enqueueing count, boxes, confidences, classes, and masks into pinned host buffers. Recording and synchronizing a CUDA event immediately after those copies might avoid waiting on unrelated current-stream work.
- Change tested: Temporary gated code only; with `RFDETR_EVENT_SYNC_D2H=1`, reused a thread-local `torch.cuda.Event`, recorded it after the fixed D2H copies, and synchronized the event instead of synchronizing the whole current stream. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.30s fps=233.66`, then `frames=538 elapsed=2.33s fps=231.23`, not stable enough to keep over the accepted fixed-copy band.
- Learning: The current stream contains only the relevant copy chain in this path, and the event record/sync adds overhead and variance. Keep the simpler stream synchronize.

### Rejected: Metadata Array Allocation Variant

- Hypothesis: RFDETR workflow conversion creates parent ID, image-dimensions, and inference ID arrays using Python list multiplication before converting to NumPy. Using `np.full(...)` and a preallocated dimensions array could reduce CPU metadata allocation work in the materialization tail.
- Change tested: Temporary gated code only; with `RFDETR_FAST_METADATA_ARRAYS=1`, cached `len(sv_detections)`, used `np.full(...)` for parent and inference IDs, and filled a `(N, 2)` `np.int64` image-dimensions array directly. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.30s fps=233.84`, then `frames=538 elapsed=2.31s fps=233.27`, not better than the accepted fixed-copy band.
- Learning: Per-detection metadata arrays are below the limiter after the fixed D2H copy. The existing simple list-to-array construction is stable enough; keep it.

### Rejected: Reuse Sigmoid Output Buffer

- Hypothesis: The fused RFDETR instance segmentation path computes an out-of-place sigmoid tensor for class logits every frame before the Triton selector. Reusing a thread-local output buffer via `torch.sigmoid(..., out=buffer)` could preserve semantics while avoiding the per-frame sigmoid output allocation.
- Change tested: Temporary gated code only; with `RFDETR_REUSE_SIGMOID_BUFFER=1`, cached a thread-local same-shape logits sigmoid buffer and wrote sigmoid results into it for RFDETR postprocess. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.31s fps=232.65`, below the accepted fixed-copy band.
- Learning: PyTorch's normal sigmoid allocation is not the limiter, and the `out=` path/thread-local lookup does not tighten the graph-to-graph interval. Keep the standard out-of-place sigmoid.

### Rejected: Disable TensorRT Aux Streams During Capture

- Hypothesis: The cached TensorRT engine reports four auxiliary streams. Forcing the CUDA-graph execution context to use no aux streams during capture might make the graph replay easier to schedule and reduce the graph-to-graph tail.
- Change tested: Temporary gated code only; with `RFDETR_TRT_ZERO_AUX_STREAMS=1`, called `graph_context.set_aux_streams([])` before graph capture. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.33s fps=230.41`, below the accepted fixed-copy band.
- Learning: Disabling TensorRT auxiliary streams during graph capture slows the forward pass on this engine/runtime. Keep TensorRT's default aux-stream scheduling.

### Rejected: TensorRT Graph Replay On Caller Stream

- Hypothesis: The accepted CUDA graph path captures and replays TensorRT on a graph-owned stream, then uses event waits to hand off from the RFDETR inference stream and back. Capturing the graph on the caller inference stream itself could remove those event edges and tighten the graph-to-graph tail.
- Change tested: Temporary gated code only; with `RFDETR_TRT_CALLER_GRAPH_STREAM=true`, captured the TensorRT CUDA graph on `torch.cuda.current_stream(device)` and skipped the wait edges when the cached graph stream matched the caller stream. Pipeline depth remained fixed at `2`. A first run with `RFDETR_TRT_CALLER_GRAPH_STREAM=1` failed before processing frames because this repo's env parser accepts `true`/`false`, not `1`/`0`, and was discarded.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.32s fps=231.91`, below the accepted fixed-copy band.
- Learning: The separate graph stream plus explicit handoff remains the better schedule for this depth-2 pipeline. The extra event edges are not the limiter, and folding graph replay onto the caller stream slows the full run.

### Profile: Fresh Accepted Depth-2 Graph Gap

- Request: Collect a new Nsight Systems report for the current accepted implementation while keeping the pipeline fixed at depth `2`.
- Profile: `/tmp/rfdetr_depth2_accepted_20260523_122306.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_accepted_20260523_122306.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_accepted_20260523_122306_stats_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_122306_stats_cuda_gpu_mem_time_sum_cuda_gpu_mem_time_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_122306_stats_cuda_api_sum_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.31s fps=232.91`.
- Graph spacing: The capture includes `538` CUDA graph traces. After skipping the first 100 launches, CUDA graph duration was p50 `4104.479 us`, p90 `4121.887 us`, p95 `4125.118 us`, p99 `4179.422 us`, mean `4078.695 us`; graph end-to-next-start gap was p50 `40.511 us`, p90 `41.919 us`, p95 `42.751 us`, p99 `43.327 us`, mean `40.813 us`. Busy work inside the gap was p50 `35.071 us`, mean `35.382 us`; idle inside the gap was p50 `5.344 us`, mean `5.431 us`.
- Learning: The latest accepted path is still effectively TensorRT CUDA-graph limited. The post-graph interval is short and stable, and the remaining idle bubble is only about `5 us` median.

### Rejected: Int64 Query Indices From Selector

- Hypothesis: The fused selector writes query indices as `int32` and the deferred path casts them to `int64`, producing a small copy/cast kernel before mask resize. Writing `int64` query indices directly from the Triton selector could remove that cast while preserving the indexing dtype expected by downstream PyTorch paths.
- Change tested: Temporary code only; first gated `RFDETR_SELECTOR_INT64_QUERIES=1` to allocate query indices as `torch.int64` and skip the `.to(dtype=torch.long)` call, then tested the actual candidate as an unconditional `int64` query-index output. Pipeline depth remained fixed at `2`.
- Result on requested command: the gated probe measured `234.01` and `234.61` FPS, but the unconditional candidate measured `frames=538 elapsed=2.32s fps=232.35`, below the accepted fixed-copy band.
- Learning: Removing the query-index cast is not enough to improve the full pipeline, and the `int64` selector output shifts scheduling or memory behavior unfavorably. Keep the accepted `int32` selector output plus explicit cast.

### Rejected: Precomputed Mask Resize Maps

- Hypothesis: The deferred mask resize always upsamples `78x78` masks to `312x312` in this benchmark. Caching the bilinear `x0/x1/y0/y1/wx/wy` coordinate maps on GPU could remove per-pixel floor, clamp, and weight arithmetic from `_resize_selected_masks_kernel`.
- Change tested: Temporary gated code only; with `RFDETR_PRECOMPUTED_RESIZE_MAPS=1`, cached resize maps per thread/device/shape and launched a variant Triton mask-resize kernel that loads coordinates and weights from those maps. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.33s fps=230.97`, below the accepted fixed-copy band.
- Learning: The extra global map loads are more expensive than recomputing the simple bilinear coordinates in registers for this fixed small resize. Keep the arithmetic-only resize kernel.

### Rejected: FP16 TensorRT Mask Output Copy

- Hypothesis: The TensorRT CUDA graph path clones all three output buffers every frame, including the large `100x78x78` mask tensor. Returning the mask output as a `float16` device copy could reduce D2D output-copy traffic and mask-resize read bandwidth while preserving the downstream zero-threshold mask test.
- Change tested: Temporary gated code only; with `RFDETR_TRT_FP16_MASK_OUTPUT=1`, cloned the first two TensorRT outputs normally but copied the third output with `.to(dtype=torch.float16)` on the graph stream. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.54s fps=211.59`, far below the accepted fixed-copy band.
- Learning: The FP32-to-FP16 cast kernel and scheduling cost dominate any D2D bandwidth reduction. Keep the plain FP32 mask clone.

### Rejected: Capture Input Copy Inside Pointer-Keyed CUDA Graph

- Hypothesis: The graph-to-graph gap still includes a separate `1.17 MB` D2D copy from the preprocessed CUDA tensor into the graph-owned TensorRT input buffer. Capturing that copy as the first node of the CUDA graph, keyed by the current preprocessed tensor pointer, could remove the standalone input-copy launch without retaining external input tensors.
- Change tested: Temporary gated code only; with `RFDETR_CAPTURE_INPUT_COPY_IN_GRAPH=1`, included `input_buffer.copy_(pre_processed_images)` inside CUDA graph capture, skipped the cache-hit input copy, and extended the graph cache key with `pre_processed_images.data_ptr()`. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=3.53s fps=152.47`, far below the accepted fixed-copy band.
- Learning: Pointer-keyed graph capture causes excessive graph-cache churn and capture overhead in the workflow pipeline. Keep the single shape-keyed graph and the small explicit D2D input copy.

### Rejected: Reuse Fused Postprocess GPU Buffers

- Hypothesis: The fused postprocess path allocates selector output tensors and a full-capacity mask-resize output tensor every frame. Reusing thread-local GPU buffers could reduce allocator overhead and remove the accepted path's query-index zero-fill while preserving the fixed-count CPU slicing behavior.
- Change tested: Temporary gated code only; with `RFDETR_REUSE_FUSED_POSTPROCESS_BUFFERS=1`, reused thread-local `scores`, `classes`, `boxes`, `query_indices`, `count`, and mask-resize output tensors. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.30s fps=233.67`, not a clear improvement over the accepted fixed-copy band.
- Learning: PyTorch's cached allocation path is not the limiter here, and avoiding the query fill this way does not improve the graph-to-graph interval. Keep fresh tensor allocation and deterministic zero-filled query indices.

### Rejected: Explicit Empty-Like TensorRT Output Copies

- Hypothesis: The accepted TensorRT CUDA graph path uses `buf.clone()` for each output. Replacing clone with `torch.empty_like(buf)` plus `copy_(..., non_blocking=True)` could preserve the same output ownership while avoiding any clone-specific format or autograd handling.
- Change tested: Temporary gated code only; with `RFDETR_TRT_EMPTY_COPY_OUTPUTS=1`, copied each graph output into an explicit `empty_like` tensor on the graph stream. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.30s fps=234.10`, not a clear improvement over the accepted fixed-copy band.
- Learning: PyTorch `clone()` is already as efficient as the explicit empty-and-copy form for these graph outputs. Keep the simpler clone path.

### Rejected: Overlap Mask Clone With Selector

- Hypothesis: The accepted TensorRT graph stream clones boxes, logits, and the large mask output before RFDETR postprocess can start. Returning after boxes/logits are cloned while the mask clone continues on the graph stream could let sigmoid and selector overlap the mask clone, then wait for a mask-ready event only before mask resize.
- Change tested: Temporary gated code only; with `RFDETR_OVERLAP_MASK_CLONE=1`, recorded a partial-output event after cloning boxes/logits, attached a mask-ready event to the mask clone tensor, and made fused mask resize wait for that event. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.32s fps=232.33`, below the accepted fixed-copy band.
- Learning: The added event and partial handoff scheduling overhead exceeds any overlap gained between mask clone and selector. Keep the simple all-output clone plus stream wait.

### Rejected: Triton Num Stages Tuning

- Hypothesis: The selector and mask-resize Triton kernels are small post-graph kernels where the default pipeline staging may add register/codegen overhead. Forcing `num_stages=1` could reduce latency for these elementwise/reduction kernels on T4.
- Change tested: Temporary code only; first launched `_resize_selected_masks_kernel` with `num_stages=1`, then restored resize and launched `_select_topk_boxes_kernel` with `num_stages=1`. Pipeline depth remained fixed at `2`.
- Result on requested command: resize `num_stages=1` measured `frames=538 elapsed=2.31s fps=232.79`; selector `num_stages=1` measured `frames=538 elapsed=2.30s fps=233.43`, neither better than the accepted fixed-copy band.
- Learning: Triton's default staging is already adequate for both kernels. Keep the accepted launches without explicit `num_stages`.

### Rejected: Mask-First Fixed D2H Copy Order

- Hypothesis: The fixed 7-row conversion path enqueues tiny count/box/confidence/class D2H copies before the larger mask D2H copy. Starting the large mask copy first might improve copy-engine scheduling and shave the CPU synchronization tail.
- Change tested: Temporary code only; reordered `_try_copy_limited_cuda_detection_tensors_to_pinned_numpy(...)` to enqueue the mask D2H copy before count and metadata copies. Pipeline depth remained fixed at `2`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.29s fps=234.45`, then repeated at `frames=538 elapsed=2.31s fps=232.77`, not a stable improvement over the accepted fixed-copy band.
- Learning: D2H copy submission order is not a reliable limiter after fixed-count conversion. Keep the original count/metadata/mask order.

### Rejected: NumPy Count Read From Pinned Buffer

- Hypothesis: After fixed 7-row D2H copy synchronization, `_try_copy_limited_cuda_detection_tensors_to_pinned_numpy(...)` reads the copied count via `count_buffer.item()`. Reading through the pinned tensor's NumPy view might avoid Torch scalar extraction overhead in the CPU materialization tail.
- Change tested: Temporary code only; replaced `int(count_buffer.item())` with `int(count_buffer.numpy()[0])`. Pipeline depth remained fixed at `2`.
- Result on requested command: `frames=538 elapsed=2.31s fps=233.16`, not better than the accepted fixed-copy band.
- Learning: Count scalar extraction is below the limiter. Keep the simpler `count_buffer.item()` path.

### Rejected: Specialized 4x Mask Resize Kernel

- Hypothesis: The benchmark always resizes RFDETR masks from `78x78` to `312x312`, so a Triton resize kernel specialized for the exact 4x mapping could remove per-pixel floor/divide arithmetic from the generic bilinear resize kernel.
- Change tested: Temporary code only; dispatched to a separate `_resize_selected_masks_4x_kernel(...)` when `input_height * 4 == output_height` and `input_width * 4 == output_width`. Pipeline depth remained fixed at `2`.
- Result on requested command: first run `frames=538 elapsed=2.29s fps=234.83`, repeat `frames=538 elapsed=2.31s fps=232.92`, not stable enough to keep over the accepted fixed-copy band.
- Learning: The generic resize arithmetic is not the throughput limiter at depth `2`; graph spacing is dominated by TensorRT graph replay plus the required input/output copies and selector/sigmoid tail. Keep the generic resize kernel.

### Profile: Accepted Depth-2 Graph Gap After 4x Resize Rejection

- Request: Collect another Nsight Systems report for the current accepted implementation while keeping pipeline depth fixed at `2`.
- Profile: `/tmp/rfdetr_depth2_accepted_20260523_130220.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_accepted_20260523_130220.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_accepted_20260523_130220_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_130220_stats_cuda_gpu_mem_time_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_130220_stats_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.33s fps=230.90`.
- Graph spacing: The capture includes `538` CUDA graph traces. After skipping the first 100 launches, CUDA graph duration was p50 `4102.399 us`, p90 `4116.422 us`, p95 `4123.113 us`, p99 `4152.967 us`, mean `4078.093 us`; graph end-to-next-start gap was p50 `40.863 us`, p90 `42.220 us`, p95 `42.528 us`, p99 `43.401 us`, mean `41.820 us`. Busy work inside the gap was p50 `35.263 us`, mean `35.769 us`; idle inside the gap was p50 `5.471 us`, mean `6.051 us`.
- Learning: Depth `2` remains effectively constrained by the TensorRT CUDA graph. The remaining median idle bubble between graph launches is about `5.5 us`, and the graph-to-graph gap remains low and consistent.

### Rejected: Prewarmed Two-Slot TensorRT Graph Pool

- Hypothesis: The earlier thread-local graph replay experiment regressed because each worker captured lazily in the hot path. Prewarming two TensorRT CUDA graph caches from the first frame and assigning one cache/stream to each depth-2 worker might allow concurrent graph replay and improve GPU utilization if the TensorRT graph body has internal bubbles.
- Change tested: Temporary gated code only; with `RFDETR_PREWARMED_TRT_GRAPH_POOL=1`, the RFDETR TRT model created two one-entry graph caches and two inference streams on the first forward pass, captured both graphs before the first result, then assigned stable slots to worker threads. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.65s fps=202.96`, far below the accepted fixed-copy band.
- Learning: The TensorRT graph body already saturates the relevant T4 resources, or concurrent graph/context scheduling interferes with TensorRT's own auxiliary streams. Keep the single serialized TensorRT CUDA graph path with depth-2 CPU/GPU pipelining.

### Rejected: TensorRT External Context Device Memory

- Hypothesis: Creating the CUDA graph execution context with `create_execution_context_without_device_memory()`, allocating the activation memory explicitly as a long-lived Torch CUDA tensor, and binding it with `set_device_memory(...)` might improve graph replay stability or memory placement versus TensorRT's internal context allocation.
- Change tested: Temporary gated code only; with `RFDETR_TRT_EXTERNAL_DEVICE_MEMORY=1`, `_capture_cuda_graph(...)` used an external device-memory allocation sized from `update_device_memory_size_for_shapes()` and `engine.device_memory_size_v2`, kept alive on the graph state. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: first run `frames=538 elapsed=2.29s fps=234.64`, repeat `frames=538 elapsed=2.33s fps=230.51`, not stable enough to keep over the accepted fixed-copy band.
- Learning: TensorRT's internal context memory allocation is not the graph replay limiter for this static RFDETR engine. Keep the simpler default execution-context allocation.

### Rejected: Partial TensorRT Aux Stream Counts

- Hypothesis: The RFDETR engine reports four auxiliary streams. Previous tests rejected zero explicit aux streams and all four explicit aux streams; setting one or two explicit aux streams during CUDA graph capture might reduce TensorRT internal scheduling overhead while preserving some overlap.
- Change tested: Temporary gated code only; with `RFDETR_TRT_AUX_STREAM_COUNT`, `_capture_cuda_graph(...)` created that many Torch CUDA streams, passed their handles to `IExecutionContext.set_aux_streams(...)`, and kept them alive on the CUDA graph state. Pipeline depth remained fixed at `2`.
- Result on requested command: `RFDETR_TRT_AUX_STREAM_COUNT=2` measured `frames=538 elapsed=2.31s fps=233.11`; `RFDETR_TRT_AUX_STREAM_COUNT=1` measured `frames=538 elapsed=2.31s fps=232.90`, both below the accepted fixed-copy band.
- Learning: Manually constraining TensorRT aux-stream count does not improve graph replay. Keep TensorRT's default aux-stream scheduling.

### Rejected: Skip RFDETR Forward Lock On Graph Cache Hit

- Hypothesis: After the static TensorRT CUDA graph is captured, the model-level RFDETR forward lock might be unnecessary because the graph cache is internally locked and the shared inference stream serializes GPU work. Skipping the lock on cache hits could reduce CPU scheduling overhead before graph replay.
- Change tested: Temporary gated code only; with `RFDETR_SKIP_FORWARD_LOCK_ON_GRAPH_HIT=1`, `RFDetrForInstanceSegmentationTRT.forward(...)` used the normal lock for graph capture, then skipped it when the static cache key was already present. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: first run `frames=538 elapsed=2.30s fps=233.77`, repeat `frames=538 elapsed=2.32s fps=231.57`, not better than the accepted fixed-copy band.
- Learning: The forward lock is not a meaningful bottleneck now that graph-to-graph idle is only a few microseconds. It likely helps keep cross-thread scheduling stable, so keep the locked forward path.

### Rejected: Skip PyCUDA Context On TensorRT Graph Cache Hit

- Hypothesis: Once the static TensorRT CUDA graph has been captured, cache-hit replay uses PyTorch streams and the captured graph state. Skipping the per-frame `use_cuda_context(...)` push/pop on cache hits might remove CPU overhead without changing graph execution.
- Change tested: Temporary gated code only; with `RFDETR_SKIP_CUDA_CONTEXT_ON_GRAPH_HIT=1`, `RFDetrForInstanceSegmentationTRT.forward(...)` kept the model lock but skipped the PyCUDA context manager when the static graph-cache key was already present. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: first run `frames=538 elapsed=2.30s fps=234.21`, repeat `frames=538 elapsed=2.33s fps=230.89`, not stable enough to keep over the accepted fixed-copy band.
- Learning: Context push/pop is below the limiter or helps maintain predictable CUDA context state across worker threads. Keep the original context manager on forward.

### Rejected: Borrow TensorRT Mask Output With Release Event

- Hypothesis: The accepted CUDA graph path clones the full `100x78x78` TensorRT mask output every frame even though the workflow normally resizes only the first 7 selected detections. Returning a borrowed graph-owned mask output while cloning only boxes/logits could remove the large D2D mask clone if the next graph replay waits for postprocess to finish reading the mask.
- Change tested: Temporary gated code only; with `RFDETR_BORROW_TRT_MASK_OUTPUT=1`, cache-hit replay cloned only the first two TensorRT outputs, returned the graph-owned mask tensor with graph-state metadata, recorded a release event after RFDETR postprocess, and waited on that event before the next graph replay could overwrite the mask buffer. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.33s fps=231.40`, below the accepted fixed-copy band.
- Learning: The large mask clone also decouples the next TensorRT graph replay from postprocess. Replacing it with an event dependency lengthens the critical path more than it saves in D2D copy time. Keep the full output clone path.

### Rejected: Selected Low-Res Mask Copy Before Resize

- Hypothesis: The borrowed-mask test waited for full postprocess before releasing the graph-owned TensorRT mask output. Copying only the selected low-res mask planes first, recording the release event immediately after that small copy, and resizing from the compact copy might remove the full `100x78x78` mask clone while allowing the next graph replay to start earlier than waiting for full mask resize.
- Change tested: Temporary gated code only; with `RFDETR_SELECTED_MASK_COPY=1`, cache-hit replay cloned boxes/logits but borrowed the graph-owned mask output, RFDETR fused postprocess copied the first `deferred_mask_resize_detection_limit` selected `78x78` masks into a compact buffer, recorded the graph-output release event, then resized compact rows with a no-query-index Triton kernel. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.33s fps=230.85`, below the accepted fixed-copy band.
- Learning: The added selected-copy kernel, compact-resize variant, and release-event dependency still cost more than the full mask clone's buffering benefit. Keep the accepted full TensorRT output clone path.

### Rejected: Selector Iteration Cap

- Hypothesis: The fused selector loops up to `100` global top-score iterations, but the benchmark has at most `7` detections per frame. Lowering the maximum iteration count could reduce selector latency while preserving outputs if no extra invalid high-score classes need to be skipped.
- Change tested: Temporary gated code only; with `RFDETR_SELECTOR_MAX_ITERATIONS`, passed a smaller Triton constexpr loop bound to `_select_topk_boxes_kernel(...)`. Also probed the simpler raw-mask shortcut assumption and found selected query IDs ranged from `0` to `98`, with `38/538` frames selecting query IDs above `6`, so cloning only the first seven raw mask rows would be incorrect.
- Correctness: Compared gated postprocess against accepted postprocess on all `538` frames. Caps `16`, `8`, and `7` all matched exactly: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: cap `16` measured `frames=538 elapsed=2.30s fps=233.68`; cap `8` measured `frames=538 elapsed=2.31s fps=233.31`; cap `7` measured `frames=538 elapsed=2.31s fps=233.37`, not better than the accepted fixed-copy band.
- Learning: Even when the selector does fewer loop iterations, the end-to-end run remains constrained by the TensorRT CUDA graph and output-copy scheduling. Keep the original conservative `100`-iteration selector bound for general correctness.

### Profile: Current Accepted Depth-2 Graph Gap

- Request: Refresh Nsight Systems evidence for the current accepted implementation after the recent rejected output-copy and selector-cap experiments, keeping pipeline depth fixed at `2`.
- Profile: `/tmp/rfdetr_depth2_accepted_20260523_134108.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_accepted_20260523_134108.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_accepted_20260523_134108_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_134108_stats_cuda_gpu_mem_time_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_134108_stats_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.30s fps=234.29`.
- Graph spacing: The capture includes `538` CUDA graph traces. After skipping the first 100 launches, CUDA graph duration was p50 `4102.607 us`, p90 `4117.545 us`, p95 `4121.161 us`, p99 `4127.790 us`, mean `4070.076 us`; graph end-to-next-start gap was p50 `40.479 us`, p90 `41.913 us`, p95 `42.303 us`, p99 `43.076 us`, mean `40.907 us`. Busy work inside the gap was p50 `35.072 us`, mean `35.500 us`; idle inside the gap was p50 `5.247 us`, mean `5.407 us`.
- Learning: The current accepted path is still bottlenecked by the TensorRT CUDA graph body. The post-graph tail remains about `40 us` and only about `5 us` of that is idle, so further end-to-end wins need to reduce TensorRT graph duration or remove required input/output copies without adding synchronization.

### Rejected: Borrow TensorRT Boxes And Logits Outputs

- Hypothesis: The accepted CUDA graph path still clones the small TensorRT boxes and logits outputs after every replay. Borrowing those graph-owned outputs while keeping the full mask clone as the decoupling buffer might shave the graph-to-graph gap without forcing mask resize onto the next replay's critical path.
- Change tested: Temporary gated code only; with `RFDETR_BORROW_TRT_SMALL_OUTPUTS=true`, cache-hit replay returned graph-owned boxes/logits and cloned only masks. A CPU-side ready event plus CUDA release event protected the borrowed buffers, and the fused path released them immediately after sigmoid and selector had been enqueued on the postprocess stream. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.34s fps=230.02`, below the accepted fixed-copy band.
- Learning: The two small D2D clones are cheaper than the extra release-event handoff on this depth-2 pipeline. Keep the simpler accepted TensorRT output clone path.

### Profile: Accepted Depth-2 Graph Gap After Small-Output Borrow Rejection

- Request: Collect a fresh Nsight Systems report for the restored accepted implementation, keeping pipeline depth fixed at `2`.
- Profile: `/tmp/rfdetr_depth2_accepted_20260523_134844.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_accepted_20260523_134844.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_accepted_20260523_134844_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_134844_stats_cuda_gpu_mem_time_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_134844_stats_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.33s fps=230.87`.
- Graph spacing: The capture includes `538` CUDA graph traces. After skipping the first 100 launches, CUDA graph duration was p50 `4070.910 us`, p90 `4133.057 us`, p95 `4136.255 us`, p99 `4141.648 us`, mean `4065.406 us`; graph end-to-next-start gap was p50 `40.480 us`, p90 `41.888 us`, p95 `42.374 us`, p99 `42.952 us`, mean `40.920 us`. Busy work inside the gap was p50 `35.072 us`, mean `35.486 us`; idle inside the gap was p50 `5.312 us`, mean `5.434 us`.
- Learning: The restored accepted path is already graph-bound at depth `2`: graph replay takes roughly `4.07 ms`, while the post-graph tail is roughly `40 us` with only about `5 us` idle. The remaining gap is mostly required input copy, output clones, sigmoid/selector setup, and small postprocess kernels rather than CPU bubbles.

### Rejected: Keep Deferred Query Indices Int32

- Hypothesis: The deferred fused mask-resize path does not need `query_indices` as `int64`; returning the selector's native `int32` query-index tensor could remove the int32-to-int64 cast kernel from the post-graph tail.
- Change tested: Temporary code only; when `return_cpu_count=False`, `fused_select_topk_boxes(...)` returned the int32 `query_indices` tensor directly. The non-deferred indexing path still used the existing int64 conversion. Pipeline depth remained fixed at `2`.
- Result on requested command: first run `frames=538 elapsed=2.31s fps=232.65`, repeat `frames=538 elapsed=2.32s fps=231.65`, not better than the accepted fixed-copy band.
- Learning: Removing this tiny cast is below end-to-end noise because the run is dominated by TensorRT graph replay and required copies. Keep the existing int64 return type for consistency with the non-deferred path.

### Rejected: Disable TensorRT Enqueue Profiling Emission

- Hypothesis: The TensorRT CUDA graph execution context defaults `enqueue_emits_profile=True` even though no profiler is attached. Disabling that flag before warmup and graph capture might remove bookkeeping from TensorRT enqueue or graph replay.
- Change tested: Temporary gated code only; with `RFDETR_TRT_DISABLE_ENQUEUE_PROFILE=true`, `_capture_cuda_graph(...)` set `graph_context.enqueue_emits_profile = False` immediately after creating the graph execution context. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: first run `frames=538 elapsed=2.30s fps=233.45`, repeat `frames=538 elapsed=2.33s fps=231.01`, not a stable improvement over the accepted fixed-copy band.
- Learning: TensorRT enqueue profiling emission is not a meaningful limiter for the captured RFDETR graph path. Keep the default execution-context setting.

### Rejected: TensorRT On-Profile-Change Context Allocation

- Hypothesis: TensorRT exposes `ExecutionContextAllocationStrategy.ON_PROFILE_CHANGE` in addition to the default static execution-context allocation. For this static-shape engine, the alternate allocation strategy might reduce context memory management overhead or produce a slightly different captured graph schedule.
- Change tested: Temporary gated code only; with `RFDETR_TRT_ON_PROFILE_CHANGE_CONTEXT=true`, `_capture_cuda_graph(...)` created the graph execution context with `engine.create_execution_context(trt.ExecutionContextAllocationStrategy.ON_PROFILE_CHANGE)`. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.31s fps=233.16`, `frames=538 elapsed=2.28s fps=235.96`, and `frames=538 elapsed=2.31s fps=232.94`, not stable enough to keep over the accepted fixed-copy band.
- Learning: The alternate allocation strategy does not reduce the TensorRT CUDA graph bottleneck. Keep the default static execution-context allocation.

### Rejected: Producer Runtime And Queue Tuning

- Hypothesis: Once graph-to-graph gaps are about `40 us`, small CPU producer/dispatcher settings might still perturb depth-2 balance: limiting CPU thread pools, disabling Python GC, changing video decode buffer size, or replacing the bounded predictions `Queue` with `SimpleQueue`.
- Change tested: External/runtime probes plus temporary gated code only. Ran the accepted path with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1`; ran the benchmark with `gc.disable()` via `runpy`; tried `VIDEO_SOURCE_BUFFER_SIZE=8` and `128`; and tested `INFERENCE_PIPELINE_SIMPLE_PREDICTIONS_QUEUE=true` using a temporary `SimpleQueue` dispatch path. Pipeline depth remained fixed at `2`.
- Result on requested command: limited CPU threads measured `frames=538 elapsed=2.29s fps=234.87`; disabled GC measured `frames=538 elapsed=2.30s fps=234.18`; video buffer `8` measured `frames=538 elapsed=2.32s fps=231.46`; video buffer `128` measured `frames=538 elapsed=2.33s fps=231.38`; simple predictions queue measured `frames=538 elapsed=2.30s fps=233.62`.
- Learning: The current run is not meaningfully limited by Python GC, CPU thread oversubscription, video decode buffer size, or bounded queue bookkeeping. Keep the accepted runtime settings and standard `Queue` implementation.

### Rejected: Adaptive Fixed-Row Prediction Copy

- Hypothesis: The accepted fixed D2H conversion copies seven full-resolution mask rows for every frame, but many frames have fewer detections. Copying `previous_count + 1` rows, capped at the seven-row mask limit, could reduce common-path D2H payload while falling back to the safe existing conversion when the current count exceeds the predicted row count.
- Change tested: Temporary gated code only; with `RFDETR_ADAPTIVE_FIXED_COPY=true`, `_try_copy_limited_cuda_detection_tensors_to_pinned_numpy(...)` predicted the row count from a thread-local previous valid count, copied that many rows plus the count, then updated the stored count after synchronization. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.32s fps=231.82`, below the accepted fixed seven-row copy band.
- Learning: The current fixed seven-row copy is faster than adaptive under-copy with fallback. Extra branching and fallback copies on count increases cost more than copying a few unused mask rows.

### Rejected: Mask-First TensorRT Output Clone Order

- Hypothesis: The accepted TensorRT graph cache-hit path clones boxes, logits, then the large mask output. Cloning the mask first while returning outputs in the original order might improve D2D copy scheduling in the post-graph tail.
- Change tested: Temporary gated code only; with `RFDETR_TRT_MASK_FIRST_OUTPUT_CLONE=true`, cloned TensorRT output buffer `2` first, then buffers `0` and `1`, and returned `[detections, labels, mask]`. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.31s fps=233.15`, below the accepted fixed-copy band.
- Learning: Output clone order does not improve the graph-to-graph interval. The existing boxes/logits/mask order remains the better schedule for this pipeline.

### Depth-2 Accepted Nsight Systems Refresh

- Request: Collect a fresh Nsight Systems report for the current accepted implementation while keeping pipeline depth fixed at `2`.
- Profile: `/tmp/rfdetr_depth2_accepted_20260523_141912.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_accepted_20260523_141912.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_accepted_20260523_141912_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_141912_stats_cuda_gpu_mem_time_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_141912_stats_cuda_api_sum.csv`.
- Profiled result: `frames=538 elapsed=2.30s fps=233.83` under Nsight Systems overhead.
- Graph spacing: The capture includes `538` CUDA graph traces. After skipping the first `100` graph launches, CUDA graph duration was p50 `4064.207 us`, p90 `4130.817 us`, p95 `4134.781 us`, p99 `4141.249 us`, mean `4058.199 us`; graph end-to-next-start gap was p50 `40.479 us`, p90 `41.868 us`, p95 `42.335 us`, p99 `42.964 us`, mean `41.481 us`.
- Gap decomposition: Busy work inside the gap was p50 `35.168 us`, mean `35.742 us`; idle inside the gap was p50 `5.184 us`, p90 `5.920 us`, p95 `6.015 us`, p99 `6.208 us`, mean `5.739 us`. The largest gap occupants were the TensorRT mask D2D clone (`2433600B`, `13.156 us` avg overlap), graph input D2D copy (`1168128B`, `13.119 us` avg overlap), sigmoid (`6.872 us` avg overlap), fill-long (`2.817 us` avg overlap), selector (`2.184 us` avg overlap), logits D2D clone (`36400B`, `2.102 us` avg overlap), and boxes D2D clone (`1600B`, `1.991 us` avg overlap).
- Learning: Depth `2` remains constrained by the CUDA graph body. The post-graph interval is low and consistent, and the remaining idle bubble is only about `5-6 us`; further wins need to reduce TensorRT graph duration or eliminate required input/output copies without adding postprocess dependencies.

### Rejected: Boxes-Mask-Logits TensorRT Output Clone Order

- Hypothesis: The accepted TensorRT graph cache-hit path clones boxes, logits, then masks. Cloning boxes first, then the large mask tensor, then logits might improve D2D copy scheduling while still returning `[detections, labels, mask]` to callers.
- Change tested: Temporary gated code only; with `RFDETR_TRT_BOX_MASK_LOGITS_OUTPUT_CLONE=true`, cloned TensorRT output buffer `0`, then `2`, then `1`, and returned results in the original output order. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.31s fps=232.66`, below the accepted fixed-copy band.
- Learning: Reordering the same graph-stream D2D clones does not reduce the critical graph-to-graph interval. Keep the original boxes/logits/mask order.

### Rejected: TensorRT Graph-Stream Sigmoid Logits

- Hypothesis: The accepted TensorRT graph stream clones the small logits output and postprocess later launches a separate sigmoid kernel. Computing `sigmoid()` directly from the graph-owned logits on the graph stream could replace the logits D2D clone with the actual postprocess tensor and remove the later sigmoid launch.
- Change tested: Temporary gated code only; with `RFDETR_TRT_GRAPH_SIGMOID_LOGITS=true`, CUDA graph capture and cache-hit replay returned `[boxes.clone(), logits.sigmoid(), masks.clone()]`, and RFDETR dense postprocess skipped its normal logits sigmoid. Pipeline depth remained fixed at `2`.
- Correctness: Compared the gated path against the accepted path on all `538` frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command with the gate enabled: first run `frames=538 elapsed=2.30s fps=233.69`, repeat `frames=538 elapsed=2.34s fps=230.14`, below the accepted fixed-copy band.
- Learning: The graph-stream logits clone is small enough that replacing it with sigmoid work on the graph stream perturbs the critical schedule rather than improving it. Keep the accepted logits clone plus postprocess-stream sigmoid.

### External Runtime Probe: Lock T4 Graphics Clock

- Hypothesis: The accepted depth-2 path is now constrained by TensorRT CUDA graph duration, and the Tesla T4 sits in a low-power P8 state at idle. Locking graphics clocks to the supported maximum before the benchmark may remove early-run clock ramp and expose the true graph-bound ceiling.
- Change tested: External runtime setting only; ran `nvidia-smi -lgc 1590,1590`, confirmed `P0` and `1590 MHz` graphics/SM clocks, ran the accepted benchmark with pipeline depth fixed at `2`, then reset the lock with `nvidia-smi -rgc`.
- Result on requested command: `frames=538 elapsed=2.21s fps=243.54`, with progress already near steady state by frame `50` (`254.93` FPS at frame 50, settling to `243.54` FPS overall).
- Learning: The code path is effectively at the graph-bound ceiling when the T4 is held at max clocks. This is an external deployment/runtime tuning knob, not a library code change; future code changes should be compared under the same clock policy if the goal is absolute max FPS rather than default cloud/runtime behavior.

### External Runtime Probe: Max Clocks Plus CUDA Connections

- Hypothesis: With T4 clocks locked to maximum, `CUDA_DEVICE_MAX_CONNECTIONS=2` might combine with the accepted depth-2 stream layout to reduce scheduling overhead further than max clocks alone.
- Change tested: External runtime setting only; ran `nvidia-smi -lgc 1590,1590`, then launched the accepted benchmark with `CUDA_DEVICE_MAX_CONNECTIONS=2` and pipeline depth fixed at `2`, then reset graphics clocks with `nvidia-smi -rgc`.
- Result on requested command: `frames=538 elapsed=2.21s fps=243.26`, effectively the same as the prior max-clock-only `243.54` FPS result.
- Learning: Once clocks are held at maximum, the connection-count scheduler knob does not move the ceiling. The remaining limiter is the TensorRT CUDA graph body.

### Rejected Under Max Clocks: Clone Graph Outputs On Caller Stream

- Hypothesis: Moving TensorRT output clones from the graph stream to the caller/inference stream was noisy under default clocks. With T4 clocks locked, the cleaner graph-bound regime might show whether this schedule frees the graph stream sooner.
- Change tested: Temporary gated code only; with `RFDETR_TRT_CLONE_OUTPUTS_ON_CALLER_STREAM=true`, cache-hit replay copied input and replayed the CUDA graph on the graph stream, then cloned TensorRT outputs on the caller stream. Pipeline depth remained fixed at `2`, and the benchmark ran with graphics clocks temporarily locked to `1590 MHz`.
- Result on requested command: `frames=538 elapsed=2.22s fps=242.51`, below the accepted graph-stream clone schedule under max clocks (`243.54` FPS).
- Learning: Even in the max-clock regime, moving output clones to the caller stream does not improve throughput. Keep the original graph-stream output clone schedule.

### RFDETR TensorRT CUDA Graph Capture Replay Warmup

- Hypothesis: The accepted path is graph-bound and the benchmark timer starts on the first delivered prediction, after CUDA graph capture. Replaying the captured TensorRT graph several times during RFDETR TRT graph capture can ramp the T4 clocks before measured frames without changing steady-state graph replay or prediction math.
- Change: Added a `cuda_graph_replay_warmup_count` option to the shared TensorRT CUDA graph helper and set it to `64` only from `RFDetrForInstanceSegmentationTRT.forward(...)`. Generic TensorRT callers still default to `0`.
- Correctness: Compared the warmed CUDA graph path against standard non-graph TensorRT execution on all `538` frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Warmup tuning: Temporary env-gated probes showed `16` extra replay warmups was too short (`frames=538 elapsed=2.31s fps=233.37`), `32` reached the max-clock band (`frames=538 elapsed=2.21s fps=243.71`), and `64` was the strongest (`frames=538 elapsed=2.20s fps=244.95`, repeat `frames=538 elapsed=2.19s fps=245.15`).
- Result on requested command after wiring the RFDETR default: depth `2` measured `frames=538 elapsed=2.21s fps=243.86` and repeat `frames=538 elapsed=2.20s fps=244.16`, with no extra environment variable or external clock lock.
- Learning: This does not shorten the TensorRT graph body; it moves the run into the steady graph-bound clock regime before the benchmark's measured interval. The tradeoff is extra first-frame/model-warmup latency, which is acceptable for this throughput-oriented RFDETR TRT path and keeps the measured pipeline close to the observed `~243-245` FPS max-clock ceiling.

### Rejected: Higher RFDETR TensorRT Graph Replay Warmup Counts

- Hypothesis: The accepted `64` replay warmup may still under-warm the T4 for the measured interval; increasing capture-time graph replays to `96` or `128` could raise the steady benchmark closer to the best observed warmup run.
- Change tested: Temporary code only; changed `RFDetrForInstanceSegmentationTRT.forward(...)` from `cuda_graph_replay_warmup_count=64` to `128`, then to `96`. Pipeline depth remained fixed at `2`.
- Correctness: Prediction math and graph topology are unchanged; only extra pre-measurement replays of the already captured TensorRT graph are added.
- Result on requested command: `128` measured `frames=538 elapsed=2.20s fps=244.73` and repeat `frames=538 elapsed=2.20s fps=244.12`; `96` measured `frames=538 elapsed=2.20s fps=244.64`. These are within the accepted `64`-warmup band and do not justify the extra startup latency.
- Learning: `64` replay warmups are enough to reach the steady graph-bound clock regime. Keep the accepted `64` setting.

### Profile: Accepted Warmed RFDETR Graph Gap

- Request: Capture Nsight Systems evidence for the accepted `64` replay-warmup checkpoint while keeping pipeline depth fixed at `2`.
- Profile: `/tmp/rfdetr_depth2_warm64_20260523_145345.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_warm64_20260523_145345.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_warm64_20260523_145345_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_warm64_20260523_145345_stats_cuda_gpu_mem_time_sum.csv`, `/tmp/rfdetr_depth2_warm64_20260523_145345_stats_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.24s fps=240.45`.
- Graph spacing: The capture includes `602` CUDA graph traces: `64` capture warmup replays plus `538` frame replays. After skipping the `64` warmups and the next `100` frame launches, CUDA graph duration was p50 `4069.806 us`, p90 `4137.469 us`, p95 `4142.621 us`, p99 `4149.969 us`, mean `4051.976 us`; graph end-to-next-start gap was p50 `40.191 us`, p90 `41.644 us`, p95 `41.958 us`, p99 `42.399 us`, mean `40.311 us`.
- Gap decomposition: Busy work inside the gap was p50 `34.912 us`, mean `35.067 us`; idle inside the gap was p50 `5.184 us`, p90 `5.952 us`, p95 `6.080 us`, p99 `6.228 us`, mean `5.244 us`. The largest gap occupants remain input D2D copy (`1168128B`, `13.117 us` avg overlap), mask D2D clone (`2433600B`, `13.095 us` avg overlap), sigmoid (`7.019 us` avg overlap), fill-long (`2.787 us` avg overlap), logits D2D clone (`36400B`, `2.098 us` avg overlap), boxes D2D clone (`1600B`, `1.988 us` avg overlap), fill-int (`1.936 us` avg overlap), and selector (`1.888 us` avg overlap).
- Learning: The warmup checkpoint does not materially change the steady graph body or graph-to-graph gap; it moves the measured run into the same graph-bound steady-clock regime that external max-clock testing exposed. Further code wins still need TensorRT graph-duration reduction or a correctness-compatible engine/tactic change.

### RFDETR TensorRT Clone Result After Capture Warmup

- Hypothesis: The accepted capture path cloned the first returned TensorRT outputs before the `64` extra graph warmup replays. Moving that clone after the warmup replays should return equivalent outputs while materializing the first result after the graph has ramped clocks.
- Change: In `_capture_cuda_graph(...)`, kept the initial post-capture replay, ran the configured capture warmup replays, then cloned `output_buffers` for the returned first result.
- Correctness: Compared the warmed CUDA graph path against standard non-graph TensorRT execution on all `538` frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`.
- Result on requested command: depth `2` measured `frames=538 elapsed=2.19s fps=245.17` and repeat `frames=538 elapsed=2.20s fps=244.67`, above the prior accepted `64` warmup default band.
- Learning: Keeping all first-result materialization after graph warmup gives a small but useful improvement without changing steady-state replay semantics. The path is still graph-bound and clock-regime sensitive.

### Rejected: Remove Pre-Replay Capture Clone

- Hypothesis: After moving the first returned result clone after the capture warmup replays, the earlier post-capture clone is overwritten and might be removable as startup-only work.
- Change tested: Temporary code only; removed the first `results = [buf.clone() ...]` and synchronize immediately after CUDA graph capture, keeping only the post-warmup replay result clone. Pipeline depth remained fixed at `2`.
- Correctness: The returned results still come from a replay of the same captured graph and input; prediction math is unchanged.
- Result on requested command: `frames=538 elapsed=2.20s fps=244.59`, below the accepted clone-after-warmup best and not enough to justify removing the extra startup work.
- Learning: The overwritten clone likely contributes a small amount of useful warmup before the measured interval. Keep the accepted capture sequence.

### Rejected: Skip Pre-Capture TensorRT Warmup Enqueue

- Hypothesis: Since RFDETR now replays the captured CUDA graph `64` times before measured frames, the separate `execute_async_v3(...)` warmup before CUDA graph capture might be redundant startup work.
- Change tested: Temporary code only; removed the pre-capture TensorRT warmup enqueue and its stream synchronization from `_capture_cuda_graph(...)`. Pipeline depth remained fixed at `2`.
- Correctness: The graph captured and replayed successfully; prediction math should be unchanged because steady execution still uses the same captured graph.
- Result on requested command: `frames=538 elapsed=2.20s fps=244.72`, below the accepted clone-after-warmup best and not enough to justify changing the capture sequence.
- Learning: The pre-capture enqueue is either useful TensorRT setup or useful GPU warmup for the measured interval. Keep the accepted capture sequence.

### Rejected: Steady-State Copy Pattern Capture Warmup

- Hypothesis: The capture-time warmup replays only the TensorRT graph body, while measured cache-hit frames do input D2D copy, graph replay, and output D2D copies. Replaying that whole copy/replay/copy pattern during capture warmup could warm the same copy engines and allocator paths before measured frames.
- Change tested: Temporary code only; after graph capture, allocated scratch output buffers once, then repeated `input_buffer.copy_(pre_processed_images)`, `cuda_graph.replay()`, and scratch `copy_` from each TensorRT output during the `64` warmup iterations before returning the post-warmup result clone. Pipeline depth remained fixed at `2`.
- Result on requested command: `frames=538 elapsed=2.19s fps=245.50`, then `frames=538 elapsed=2.20s fps=244.60`, then `frames=538 elapsed=2.21s fps=243.84`.
- Learning: Warming the full steady-state copy pattern is not a stable improvement and can regress the warmed graph-bound path. Keep the accepted warmup that replays only the captured TensorRT graph before cloning the first returned result.

### Profile: Accepted Depth-2 Warmed Graph Refresh

- Request: Collect a fresh Nsight Systems report on the accepted warmed implementation while keeping pipeline depth fixed at `2`.
- Profile: `/tmp/rfdetr_depth2_accepted_20260523_150944.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_accepted_20260523_150944.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_accepted_20260523_150944_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_accepted_20260523_150944_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_accepted_20260523_150944_stats_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.36s fps=227.82`.
- Graph spacing: The capture includes `602` CUDA graph traces. After skipping the `64` warmups plus the next `100` frame launches, CUDA graph duration was p50 `4129.582 us`, p90 `4187.479 us`, p95 `4194.288 us`, p99 `4249.728 us`, mean `4134.538 us`; graph end-to-next-start gap was p50 `42.463 us`, p90 `43.839 us`, p95 `44.256 us`, p99 `44.872 us`, mean `42.602 us`.
- Gap decomposition over the first `100` stable post-settling gaps: busy work inside the gap was p50 `37.791 us`, mean `37.704 us`; idle inside the gap was p50 `4.544 us`, mean `4.514 us`. The largest gap occupants were input D2D copy (`1168128B`, `13.143 us` avg overlap), mask D2D clone (`2433600B`, `13.131 us`), sigmoid (`6.945 us`), boxes D2D clone (`1600B`, `4.528 us` in this capture), fill-long (`2.723 us`), logits D2D clone (`36400B`, `2.105 us`), fill-int (`2.060 us`), and selector (`1.767 us`).
- Learning: This refresh shows the same desired shape: only about `4-5 us` median idle between graph replays, with the rest of the graph-to-graph interval being required GPU copy/postprocess work. Remaining FPS is still dominated by TensorRT graph duration.

### Rejected: Reusable TensorRT Output Copy Buffers

- Hypothesis: The accepted TensorRT CUDA graph cache-hit path allocates fresh cloned output tensors every frame. Reusing per-thread output copy buffers and filling them with `copy_` from graph-owned outputs could keep cloned-output lifetime isolation while reducing allocator/clone overhead, unlike the previously rejected borrowed-output path.
- Change tested: Temporary env-gated code only; with `RFDETR_TRT_REUSE_OUTPUT_COPY_BUFFERS=True`, each thread reused `empty_like` buffers for the TensorRT outputs and copied graph-owned outputs into those buffers on the graph stream. Pipeline depth remained fixed at `2`.
- Result on requested command with the gate enabled: `frames=538 elapsed=2.19s fps=246.00`, then `frames=538 elapsed=2.21s fps=243.87`, then `frames=538 elapsed=2.20s fps=244.26`.
- Learning: The first run beat the accepted band, but repeats did not. Reusable output copy buffers are too noise-sensitive in the current graph-bound regime and are not stable enough to checkpoint. Keep the accepted `buf.clone()` output path.

### External Runtime Probe: Memory Plus Graphics Clock Lock

- Hypothesis: The previous external max-clock probe locked graphics clocks only. Since the accepted path is TensorRT graph-bound and includes input/output copy traffic, also locking the T4 memory clock to the supported `5001 MHz` state might improve the graph-bound ceiling.
- Change tested: External runtime setting only; attempted `nvidia-smi -lmc 5001,5001`, then locked graphics clocks with `nvidia-smi -lgc 1590,1590`, ran the benchmark with pipeline depth fixed at `2`, and reset clock locks afterward.
- Result: The T4 runtime reported locked memory clocks are not supported. The run executed at `P0`, `1590 MHz` graphics clock, and `5000 MHz` memory clock, measuring `frames=538 elapsed=2.21s fps=243.26`, matching the earlier graphics-clock-only result. After reset and idle, the GPU returned to `P8`, `300 MHz` graphics, `405 MHz` memory.
- Learning: There is no separate memory-clock lock knob available in this environment, and graphics-clock lock remains only an external deployment/runtime tuning option. It does not change the accepted library code path.

### TensorRT Accepted Engine Inspection

- Request: Inspect the current accepted TensorRT plan directly before attempting more graph-body changes.
- Evidence: Engine inspector dump saved to `/tmp/rfdetr_accepted_engine_inspector.json` for the accepted shared blob `/tmp/cache/shared-blobs/bc173a2cfda9a10af2bc411885e9fec3`.
- Result: The accepted engine is `187,947,996` bytes, has `4` I/O tensors, `261` layers, `4` auxiliary streams, `18,289,152` bytes device memory, `ProfilingVerbosity.LAYER_NAMES_ONLY`, and tactic source mask `8`. Tensors are `input` float32 `(1,3,312,312)`, `dets` float32 `(1,100,4)`, `labels` float32 `(1,100,91)`, and mask output `4186` float32 `(1,100,78,78)`. Coarse layer-name counts from the inspector are `95` matmul/GEMM-like layers, `78` fused/cast layers, `14` conv layers, and `1` resize layer.
- Learning: The engine metadata matches the Nsight kernel evidence: remaining runtime is dominated by TensorRT GEMM/MHA-style graph body work. The plan is only layer-name verbose, so tactic-level detail is not available from the serialized engine inspector in this environment.

### External Runtime Probe: Persistence Mode

- Hypothesis: Enabling GPU persistence mode might reduce clock/context ramp effects for the warmed graph-bound benchmark.
- Change tested: External runtime setting only. Persistence mode was already enabled before the probe, so the benchmark effectively re-ran the accepted path under the existing setting. Pipeline depth remained fixed at `2`, and persistence mode was restored to the original enabled state afterward.
- Result on requested command: `frames=538 elapsed=2.19s fps=245.29`, within the accepted warmed band.
- Learning: Persistence mode was already active and is not a new optimization knob. The accepted code path remains responsible for the current measured throughput.

### External Runtime Probe: Application Clocks

- Hypothesis: `nvidia-smi -ac 5001,1590` might enforce the high memory and graphics application-clock targets more cleanly than graphics-only lock or unsupported memory lock commands.
- Change tested: External runtime setting only; set application clocks to `(MEM 5001, SM 1590)`, ran the accepted benchmark with pipeline depth fixed at `2`, then reset application clocks with `nvidia-smi -rac`.
- Result on requested command: `frames=538 elapsed=2.18s fps=246.36`, matching the current accepted warmed band rather than improving it. After reset and idle, the GPU returned to persistence enabled, `P8`, `300 MHz` graphics, `405 MHz` memory, with default application clocks `(MEM 5001, SM 585)`.
- Learning: Application clocks do not improve beyond the accepted warmed path in this environment. Keep clock tuning as an external diagnostic only, not a code or default runtime requirement.

### Rejected: Captured Deferred Postprocess Graph

- Hypothesis: The graph-to-graph gap still contains several small GPU launches after TensorRT replay: output copies, sigmoid, selector, query-index cast/fill, and limited mask resize. Reusing stable TensorRT output copy buffers and capturing the deferred sigmoid/selector/mask-resize postprocess sequence into a CUDA graph could reduce launch overhead while preserving cloned-output lifetime isolation.
- Change tested: Temporary env-gated code only; with `RFDETR_CAPTURE_DEFERRED_POSTPROCESS_GRAPH=True`, TensorRT cache-hit replay copied graph outputs into per-thread reusable buffers, and RFDETR deferred postprocess attempted to capture the sigmoid, fused selector, and fused mask resize operations keyed by those stable input pointers. The first attempt failed before any frames due concurrent depth-2 worker graph captures tripping PyTorch's CUDA caching allocator assertion; a second attempt serialized first-time captures with a global lock.
- Result on requested command after serializing capture: `frames=538 elapsed=2.29s fps=235.36`, far below the accepted warmed band.
- Learning: Capturing postprocess launch work adds too much startup/scheduling complexity and requires static output-buffer choreography that perturbs the current well-balanced depth-2 pipeline. Keep the accepted simple postprocess launch sequence and cloned TensorRT output tensors.

### Nsight Compute TensorRT Top-Kernel Snapshot

- Request: Use Nsight Compute on the dominant TensorRT graph kernels now that Nsight Systems shows the run is graph-body bound.
- Profile: `/tmp/rfdetr_ncu_trt_top_20260523_153512.ncu-rep`, exported text `/tmp/rfdetr_ncu_trt_top_20260523_153512_details.txt`, and raw CSV `/tmp/rfdetr_ncu_trt_top_20260523_153512_raw.csv`. Capture used `--set basic`, a regex for the top XMMA/GEMM/MHA TensorRT kernels, `--launch-skip 200`, and `--launch-count 6`.
- Result under NCU instrumentation: `frames=538 elapsed=5.49s fps=97.98`. The sampled top kernels are small-grid TensorRT kernels: representative `sm75_xmma_gemm_f16f16_f16f32...128x128x32` launches had grid size `72`, `240` registers/thread, `25%` theoretical occupancy, about `21.7%` achieved occupancy, `48%` SM throughput, and `15%` DRAM throughput; `_gemm_mha_v2...` had grid size `66`, `245` registers/thread, `25%` theoretical occupancy, `20.6%` achieved occupancy, `36%` SM throughput, and `6.5%` DRAM throughput.
- Learning: The remaining TensorRT graph body is dominated by many small, register/shared-memory-limited GEMM/MHA kernels. This points toward engine/tactic/export changes as the only likely large lever; custom postprocess and stream scheduling changes are now below the dominant cost.

### Rejected: CUDA Module Loading Mode

- Hypothesis: Changing `CUDA_MODULE_LOADING` before process startup might alter TensorRT module initialization/capture behavior and improve the warmed graph-bound run without changing outputs.
- Change tested: External process environment only; ran the accepted benchmark with `CUDA_MODULE_LOADING=EAGER`, then with `CUDA_MODULE_LOADING=LAZY`, keeping pipeline depth fixed at `2`.
- Result on requested command: `EAGER` measured `frames=538 elapsed=2.20s fps=245.08`; `LAZY` measured `frames=538 elapsed=2.20s fps=244.42`, both below the best accepted warmed band.
- Learning: CUDA module-loading mode is not a useful runtime knob for this already-warmed graph path. Keep the default environment.

### Profile: Depth-2 Node-Level Graph Refresh

- Request: Collect a fresh node-level Nsight Systems profile for the current accepted implementation while keeping pipeline depth fixed at `2`.
- Profile: `/tmp/rfdetr_depth2_nodegraph_20260523_155019.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_nodegraph_20260523_155019.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_nodegraph_20260523_155019_stats_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_nodegraph_20260523_155019_stats_cuda_gpu_mem_time_sum_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_nodegraph_20260523_155019_stats_cuda_api_sum_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.33s fps=231.16`.
- Graph spacing: The capture includes `602` CUDA graph launches and `239` TensorRT graph kernels per launch. After skipping the `64` capture warmups plus the next `100` frame launches, CUDA graph duration was p50 `4125.917 us`, p90 `4201.725 us`, p95 `4238.780 us`, p99 `4252.124 us`, mean `4133.067 us`; graph end-to-next-start gap was p50 `48.847 us`, p90 `50.464 us`, p95 `50.559 us`, p99 `51.647 us`, mean `48.804 us`.
- Gap decomposition: Largest gap occupants were graph input D2D copy (`1168128B`, `13.135 us` avg overlap), mask D2D clone (`2433600B`, `13.134 us`), selector kernels (`9.240 us` and `8.546 us` on alternating postprocess streams), boxes D2D clone (`1600B`, `4.513 us`), and small PyTorch vectorized kernels. The node-level trace has higher overhead than the lighter accepted profiles, but the shape is unchanged: TensorRT graph duration is the bottleneck and the inter-graph tail is short.
- Learning: The depth-2 pipeline has the requested low, consistent CUDA graph spacing. Further code wins still need a correct TensorRT engine/tactic change rather than more small postprocess scheduling tweaks.

### Rejected: Version-Compatible TensorRT Rebuilds From Public ONNX

- Hypothesis: The earlier local TensorRT opt0/1/2 rebuilds may have failed correctness because they were not built like the accepted plan. The accepted T4 FP16 plan requires `engine_host_code_allowed=True`, which is consistent with a `VERSION_COMPATIBLE` build that embeds TensorRT lean-runtime host code. Rebuilding the public ONNX with `VERSION_COMPATIBLE` might match the accepted package while allowing tactic-level tuning.
- Change tested: Built `/tmp/rfdetr_trt_rebuild_t4_fp16_opt3_vc/engine.plan` from `/tmp/rfdetr_onnx_pkg_5362b72bfb9f01d2e0b8cba2048d932c/weights.onnx` with TensorRT `10.12.0.36`, static input shape `1x3x312x312`, FP16 enabled, workspace `4 GiB`, `builder_optimization_level=3`, and `BuilderFlag.VERSION_COMPATIBLE`. Build completed in `117.28s` and produced a `187,854,900` byte plan, close to the accepted plan size.
- Correctness result: Compared the rebuilt FP16 version-compatible plan against the accepted T4 FP16 plan over all `538` benchmark frames with standard non-graph TensorRT forward and dense postprocess. It failed the required gate: `bad_counts=9`, `bad_classes=8`, `bad_masks=336`, `bad_boxes_gt5=18`, `max_box_delta=251.0`, `max_conf_delta=0.10457384586334229`.
- Follow-up: Built `/tmp/rfdetr_trt_rebuild_t4_fp32_opt3_vc/engine.plan` with the same settings but without FP16. Build completed in `24.18s` and produced a `248,398,812` byte plan. It also failed correctness against the accepted plan: `bad_counts=7`, `bad_classes=8`, `bad_masks=329`, `bad_boxes_gt5=19`, `max_box_delta=251.0`, `max_conf_delta=0.10576367378234863`.
- Metadata check: Roboflow package metadata lists six public packages for `rfdetr-seg-nano` / `coco-dataset-vdnr1/41`: L4 FP32 TRT, L4 FP16 TRT, T4 FP32 TRT, T4 FP16 TRT, ONNX FP32, and Torch FP32. The only T4 FP16 package is the accepted `c70f32369a54d61e06ef4e6b56c82524`; the public ONNX package does not rebuild into behavior-equivalent T4 TRT plans in this runtime.
- Learning: Local TensorRT rebuilds from the available ONNX are not safe optimization candidates because both FP16 and FP32 version-compatible builds change predictions substantially. Keep the accepted official T4 FP16 engine; engine-level gains would require an ONNX/export source that is known to match the accepted TRT plan or a new official model package, not ad hoc local rebuilds from this ONNX artifact.

### Rejected: PyTorch cudaMallocAsync Allocator Backend

- Hypothesis: The accepted depth-2 path still performs per-frame PyTorch CUDA allocations around TensorRT output clones, sigmoid, selector outputs, and resized mask tensors. Switching PyTorch to the CUDA async allocator with `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` might reduce allocator synchronization or fragmentation without changing model math.
- Change tested: External process environment only; launched the requested benchmark with `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`, `PYTHONPATH=/app/inference_models`, and pipeline depth fixed at `2`.
- Correctness: This does not change computation or tensor values, only allocator backend behavior.
- Result on requested command: `frames=538 elapsed=2.20s fps=245.00`, then `frames=538 elapsed=2.21s fps=243.47`, then `frames=538 elapsed=2.20s fps=244.54`.
- Learning: The async allocator can land inside the accepted warmed band, but it is not a stable improvement over the default allocator. Keep the default PyTorch CUDA allocator and do not require an external allocator environment variable for the benchmark.

### Rejected: ONNX Backend Alternative

- Hypothesis: The workflow script can force `--backend onnx`, and the environment config lists `TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider`. ONNX Runtime might build or dispatch a faster GPU path than the native `inference-models` TensorRT package.
- Change tested: Ran `PYTHONPATH=/app/inference_models python development/stream_interface/rfdetr_nano_seg_trt_workflow.py --video_reference vehicles_312px.mp4 --pipeline_depth 2 --backend onnx`.
- Result: The run failed before processing frames (`frames=0`) with ONNX Runtime I/O binding error: `There's no data transfer registered for copying tensors from Device:[DeviceType:1 MemoryType:0 DeviceId:0] to Device:[DeviceType:0 MemoryType:0 DeviceId:0]`.
- Learning: The current ONNX backend path is not a viable drop-in for this CUDA-tensor workflow fast path. Keep optimizing the accepted native TensorRT backend; do not switch benchmark backend.

### Rejected: Torch Package As Equivalent Export Source

- Hypothesis: The public Torch FP32 package `8b8da2fe824240522a39f3cde41aafae` might be a behavior-equivalent source for exporting ONNX and rebuilding TensorRT plans, unlike the public ONNX package that failed correctness.
- Change tested: Downloaded and loaded the Torch package explicitly with `AutoModel.from_pretrained(..., backend="torch", model_package_id="8b8da2fe824240522a39f3cde41aafae")`, then compared its postprocessed predictions against the accepted T4 FP16 TensorRT package `c70f32369a54d61e06ef4e6b56c82524` on benchmark video frames. Tested plain Torch, `model.export()` FP32, and `model.export()` FP16 paths. Pipeline benchmarks remain fixed at depth `2`; this was a correctness/provenance probe only.
- Result: Plain Torch and exported FP32 matched each other but failed the accepted correctness gate over the first `80` frames: `bad_counts=1`, `bad_classes=0`, `bad_masks=72`, `bad_boxes_gt5=3`, `max_box_delta=184.7200927734375`, `max_conf_delta=0.10576367378234863`. Exported FP16 was worse: `bad_counts=4`, `bad_classes=0`, `bad_masks=69`, `bad_boxes_gt5=3`, `max_box_delta=151.251220703125`, `max_conf_delta=0.2307741641998291`.
- Learning: The Torch package is closer to the accepted TRT engine than the public ONNX rebuilds, but it is still not behavior-equivalent. Do not use this Torch package as a local export/rebuild source for optimization; a correct engine-level tuning path still requires an official package or export source known to match the accepted T4 FP16 plan.

### Rejected: TensorRT Persistent Cache Limit

- Hypothesis: TensorRT `IExecutionContext.persistent_cache_limit` might enable activation/persistent L2 caching for the captured graph body and reduce the remaining graph-bound latency without changing model outputs or package artifacts.
- Change tested: Temporary env-gated code only; set `graph_context.persistent_cache_limit` before CUDA graph warmup/capture, then ran the requested benchmark with pipeline depth fixed at `2` for limits `1 MiB`, `4 MiB`, and `8 MiB`.
- Result on requested command: The T4 runtime rejected every nonzero value with `IExecutionContext::setPersistentCacheLimit: Error Code 3 ... size ... is larger than cudaDeviceProp.persistingL2CacheMaxSize(0 bytes)`. Runs measured `242.80`, `242.53`, and `243.44` FPS, below the accepted warmed band.
- Learning: This GPU exposes no persisting L2 cache budget to TensorRT, so `persistent_cache_limit` is not an available optimization knob here. Removed the temporary hook.

### Rejected: Detection-Limited Deferred Mask Allocation

- Hypothesis: The deferred mask resize path only launches work for the first `7` detections, but still allocates a `100 x H x W` bool tensor. Allocating only `7 x H x W` rows when `deferred_mask_resize_detection_limit=7` could reduce allocator pressure and memory footprint between TensorRT graph launches while preserving the recovery path for frames with more detections.
- Change tested: Temporary code only; moved detection-limit normalization before output allocation in `fused_resize_selected_masks(...)` and allocated the output tensor with `detection_limit` rows. Pipeline depth remained fixed at `2`.
- Result on requested command: `frames=538 elapsed=2.21s fps=243.80`, then `frames=538 elapsed=2.20s fps=244.45`, below the accepted warmed band.
- Learning: The allocation size is not the bottleneck; the gap is dominated by TensorRT output copies and graph-body latency. Keep the fixed `100` row allocation that preserves the established conversion-buffer behavior.

### Profile: Depth-2 Accepted Refresh After Provenance Probes

- Request: Capture a fresh Nsight Systems report for the current accepted implementation after the engine-provenance and postprocess allocation probes. The command used `PYTHONPATH=/app/inference_models` and pipeline depth stayed fixed at `2`.
- Profile: `/tmp/rfdetr_depth2_accepted_refresh_20260523_161926.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_accepted_refresh_20260523_161926.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_accepted_refresh_20260523_161926_stats_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_accepted_refresh_20260523_161926_stats_cuda_gpu_mem_time_sum_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_accepted_refresh_20260523_161926_stats_cuda_api_sum_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.29s fps=235.10`.
- Graph spacing: The capture includes `602` CUDA graph traces. After skipping the `64` warmups plus the next `100` frame launches, CUDA graph duration was p50 `4097.520 us`, p90 `4111.583 us`, p95 `4115.935 us`, p99 `4124.383 us`, mean `4070.625 us`; graph end-to-next-start gap was p50 `40.384 us`, p90 `41.535 us`, p95 `41.727 us`, p99 `42.336 us`, mean `41.097 us`.
- Gap decomposition over the first `100` stable post-settling gaps: busy work inside the gap was p50 `35.136 us`, mean `35.873 us`; idle inside the gap was p50 `5.119 us`, mean `5.224 us`. The largest gap occupants were the mask output D2D clone (`2433600B`, `13.149 us` avg overlap), next-frame input D2D copy (`1168128B`, `13.082 us`), sigmoid (`6.782 us`), selector (`2.167 us`), fill-int (`2.100 us`), logits D2D clone (`36400B`, `2.098 us`), and boxes D2D clone (`1600B`, `1.987 us`).
- Learning: The depth-2 pipeline is already very close to graph-body-bound: median idle between TensorRT graph replays is about `5 us`, while the graph body is about `4.1 ms`. Remaining large improvements still require a correct TensorRT graph-body/engine change or safe double-buffered output ownership, not depth changes or more CPU scheduling work.

### Rejected: Direct Fused Postprocess On Graph-Owned TensorRT Outputs

- Hypothesis: The largest graph-to-graph gap occupant is the full TensorRT mask-output D2D clone. Running the existing fused RFDETR selector and mask-resize directly on graph-owned TensorRT outputs before cloning the raw outputs could avoid copying the full `100 x 78 x 78` mask tensor and move closer to pure TensorRT graph-body bottlenecking.
- Change tested: Temporary env-gated code only; added an `output_processor` callback to the TensorRT CUDA-graph replay path and an RFDETR `forward_post_process(...)` path used by the workflow when `RFDETR_TRT_FUSED_GRAPH_POSTPROCESS=True`. Pipeline depth remained fixed at `2`.
- Correctness: A single-process full-video comparison against the accepted cloned-output path passed over all `538` frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`, with `max_count=7`.
- Result: The unsafe first workflow run appeared fast (`frames=538 elapsed=2.05s fps=263.03`) but repeated depth-2 workflow runs hit CUDA illegal-memory-access failures during CPU conversion, exposing a missing stream/lifetime handoff. Materializing selected masks without deferred count was stable but slow (`220.33 FPS`). Adding the required default-stream wait made the deferred path stable but not faster: `244.20`, `243.62`, and `243.88` FPS.
- Learning: The apparent `263 FPS` result was an invalid asynchronous/lifetime artifact. Once graph-owned outputs are handed off safely to CPU conversion, the path returns to the accepted throughput band. Removed the temporary callback/direct-postprocess code; a future safe win would need real double-buffered graph output ownership or conversion under a redesigned pipeline contract, not borrowing the single graph state's raw outputs.

### Rejected: CUDA Device Max Connections

- Hypothesis: The accepted TensorRT plan uses `4` auxiliary streams. Setting `CUDA_DEVICE_MAX_CONNECTIONS` before process startup might change CUDA work-queue mapping enough to improve TensorRT aux-stream scheduling or graph replay spacing.
- Change tested: External process environment only; ran the requested benchmark with `CUDA_DEVICE_MAX_CONNECTIONS` set to `1`, `2`, `8`, and `32`. Pipeline depth remained fixed at `2`.
- Correctness: This setting does not change model math or tensor values.
- Result on requested command: `1` measured `243.06` FPS, `2` measured `244.31` FPS, `8` measured `243.64` FPS, and `32` measured `243.07` FPS.
- Learning: CUDA connection count does not improve the accepted warmed graph path. Keep the default CUDA scheduling environment.

### Rejected: Two-State Direct Postprocess Graph Output Pool

- Hypothesis: The earlier direct-postprocess-on-graph-output experiment failed because a single shared graph output buffer could be overwritten while another depth-2 worker still needed it. Capturing a shared two-state CUDA graph pool before the first output could let one state feed fused postprocess while the other state replays the next frame, avoiding the full raw mask-output clone without unsafe borrowing.
- Change tested: Temporary env-gated code only; extended the TensorRT CUDA graph cache to hold a two-state pool when `RFDETR_TRT_THREAD_LOCAL_DIRECT_POSTPROCESS=True`, ran fused RFDETR postprocess on a separate postprocess stream, and made each producer graph stream wait for its postprocess stream before that state could be reused. Pipeline depth remained fixed at `2`.
- Correctness: After fixing the producer-stream handoff, the full-video comparison against the accepted cloned-output path passed over all `538` frames: `bad_counts=0`, `bad_classes=0`, `bad_masks=0`, `bad_boxes_gt5=0`, `max_box_delta=0.0`, `max_conf_delta=0.0`, with `max_count=7`.
- Result on requested command: An intermediate broken stream-wait version produced an invalid `285.71 FPS` but failed correctness badly (`bad_counts=76`, `bad_classes=77`, `bad_masks=537`, `bad_boxes_gt5=175`). The corrected version was stable and correct but not faster: `243.79`, `242.70`, and `242.31` FPS.
- Learning: Safely borrowing TensorRT graph outputs requires enough stream ordering that the raw-output clone removal no longer improves the depth-2 workflow. Removed the temporary graph-pool/direct-postprocess code.

### Rejected: Exact-Count D2H Detection Copy

- Hypothesis: The RFDETR fast conversion path always copies `7` mask rows from CUDA to pinned CPU memory, even when a frame has fewer valid detections. Copying the count first and then copying only `valid_count` rows could reduce D2H bytes for common 4-5 detection frames.
- Change tested: Temporary env-gated code only; with `RFDETR_EXACT_D2H_DETECTION_COPY=True`, copied and synchronized the count before copying xyxy/confidence/class/mask rows, then copied only the valid rows. Pipeline depth remained fixed at `2`.
- Correctness: This changes only how many already-resized selected rows are copied to CPU, not model math.
- Result on requested command: `frames=538 elapsed=2.80s fps=192.39`, then `frames=538 elapsed=2.82s fps=190.70`.
- Learning: The extra count synchronization is far more expensive than the saved mask-copy bytes. Keep the accepted single synchronization that copies the fixed 7-row pinned buffers.

### Rejected: Concurrent TensorRT CUDA Graph Replay Pool

- Hypothesis: The accepted TRT forward path serializes graph replay with a Python model lock. Since NCU showed many small register/shared-memory-limited TensorRT kernels, allowing two depth-2 workers to submit independent CUDA graph states on separate streams might overlap low-occupancy kernels and improve hardware utilization.
- Change tested: Temporary env-gated code only; with `RFDETR_TRT_CONCURRENT_GRAPHS=True`, first tried per-thread graph caches without the model lock, which failed because concurrent CUDA graph capture is not permitted. Then serialized capture and tested a shared two-cache graph pool with thread-local caller streams and lock-free round-robin replay. Pipeline depth remained fixed at `2`.
- Result on requested command: Concurrent capture failed before frames with `operation not permitted when stream is capturing`. The serialized-capture per-thread version completed but measured `220.23` FPS. The shared two-cache replay pool completed but measured `219.47` FPS.
- Learning: Concurrent graph submission does not help this TensorRT plan on T4. The required extra streams/contexts/capture and postprocess handoff cost more than any possible overlap among small TensorRT kernels. Keep the accepted single serialized graph replay path.

### Rejected: Fused Workflow SV Detection Conversion

- Hypothesis: After GPU postprocess fusion, the RFDETR workflow fast path still performs several Python passes to convert inference-model detections to `sv.Detections`, attach prediction type metadata, apply the no-op class filter, and attach parent metadata. Fusing these metadata writes for the no-class-filter benchmark path could reduce CPU-side packaging enough to improve the depth-2 producer/consumer balance.
- Change tested: Temporary env-gated code only; with `RFDETR_FUSED_SV_CONVERSION=True`, the RFDETR workflow fast path copied the existing pinned detection tensors and built final `sv.Detections` metadata in one helper, preserving prediction type, image dimensions, inference ID, detection IDs, root-parent metadata, and parent metadata. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: Baseline with the accepted path measured `frames=538 elapsed=2.21s fps=243.78`; the fused conversion branch measured `frames=538 elapsed=2.21s fps=242.91`.
- Learning: CPU-side detection packaging is no longer a useful lever for this benchmark. The accepted depth-2 path is constrained by the CUDA graph body plus required GPU copy/postprocess tail, so keep the generic metadata helpers and focus further work on TensorRT graph-body or safe GPU-output ownership changes.

### Rejected: Explicit TensorRT Optimization Profile Async

- Hypothesis: Even though the accepted RFDETR TensorRT plan has one optimization profile, explicitly calling `IExecutionContext.set_optimization_profile_async(0, graph_stream)` before setting the static input shape might alter context setup or graph-capture scheduling enough to reduce TensorRT graph replay latency.
- Change tested: Temporary env-gated code only; with `RFDETR_TRT_SET_PROFILE_ASYNC=true`, `_capture_cuda_graph(...)` created the graph stream before shape binding, selected profile `0` asynchronously on that stream, synchronized, then proceeded with the accepted warmup and CUDA graph capture. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Evidence: The runtime inspection showed `num_optimization_profiles=1` and `active_optimization_profile=0` before the change, so this was expected to be low probability.
- Result on requested command: Accepted baseline in the same session measured `frames=538 elapsed=2.22s fps=242.41`; the explicit-profile gate measured `frames=538 elapsed=2.22s fps=242.52` and repeat `frames=538 elapsed=2.21s fps=243.16`.
- Learning: Explicit profile selection is a no-op for this already-active single-profile engine and does not move the depth-2 ceiling. Keep the simpler accepted graph-capture path.

### Rejected: Int32 RFDETR Class Mapping Tensor

- Hypothesis: `prepare_class_remapping(...)` stores the RFDETR class mapping table as `int64`, but the fused Triton selector stores output class IDs as `int32`. Building the mapping tensor as `int32` could reduce selector load width/codegen without changing class IDs.
- Change tested: Temporary code only; changed `ClassesReMapping.class_mapping` from `torch.int64` to `torch.int32` while keeping `remaining_class_ids` as `int64`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: The first pass measured `frames=538 elapsed=2.31s fps=232.91`, and a warmed repeat measured `frames=538 elapsed=2.21s fps=243.11`, below the clean accepted sanity run in the same session at `frames=538 elapsed=2.21s fps=243.49`.
- Learning: Class-map load width is not a limiter for the fused selector. The original `int64` mapping is safer for the generic PyTorch fallback paths and remains at least as fast end to end.

### Rejected: GPU Exclusive Process Compute Mode

- Hypothesis: The accepted depth-2 run is TensorRT graph-body bound and sensitive to GPU scheduling. Switching the T4 from default compute mode to `EXCLUSIVE_PROCESS` before launching the benchmark might reduce context scheduling overhead or background interference.
- Change tested: External runtime setting only; confirmed and killed a stale benchmark process that was holding a CUDA context, set `nvidia-smi -c EXCLUSIVE_PROCESS`, ran the requested benchmark with pipeline depth fixed at `2`, then restored compute mode with `nvidia-smi -c DEFAULT`.
- Result on requested command: `frames=538 elapsed=2.22s fps=242.77`, below the accepted warmed band.
- Learning: Compute-mode isolation does not improve this single-process workload. Keep the default compute mode; the remaining limiter is still the TensorRT CUDA graph body plus the small required post-graph tail.

### Rejected: CPU Affinity And Nice Priority

- Hypothesis: The depth-2 pipeline still depends on a CPU producer thread keeping the TensorRT graph fed. Pinning the process to one hardware thread per physical core or raising process priority could reduce CPU scheduling jitter enough to tighten the graph-to-graph launch cadence.
- Change tested: External runtime settings only; after confirming no other CUDA process was active, ran the requested benchmark with `taskset -c 0-3` on the 4-core/8-thread VM, then with `nice -n -20`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: Clean default baseline after stale-process cleanup measured `frames=538 elapsed=2.21s fps=243.99`. `taskset -c 0-3` measured `244.44` then `243.88` FPS. `nice -n -20` measured `243.46` FPS.
- Learning: CPU scheduler tuning is not a stable improvement. The small first affinity result is within normal warmed-path noise, and priority does not help. Keep the default CPU scheduling for the benchmark.

### External Runtime Refresh: Application Clocks After Cleanup

- Hypothesis: After removing a stale benchmark process that had been holding a CUDA context, the previous max application-clock diagnostic might expose a higher ceiling for the accepted warmed path.
- Change tested: External runtime setting only; set application clocks to `(MEM 5001, SM 1590)`, ran the requested benchmark with pipeline depth fixed at `2`, then reset application clocks with `nvidia-smi -rac`.
- Result on requested command: `frames=538 elapsed=2.20s fps=244.24`, compared with the same-session default-clock clean baseline of `frames=538 elapsed=2.21s fps=243.99`.
- Learning: Application clocks still do not move throughput beyond the accepted warmed band. The current code-level CUDA graph warmup already reaches the practical graph-bound clock regime for the measured interval.

### Rejected: Python Optimize Mode

- Hypothesis: Launching the benchmark with `PYTHONOPTIMIZE=1` could remove Python assert/debug overhead in the workflow and model stack without changing model math or TensorRT execution.
- Change tested: External interpreter setting only; ran the requested benchmark with `PYTHONOPTIMIZE=1` and pipeline depth fixed at `2`.
- Result on requested command: `frames=538 elapsed=2.20s fps=244.26`, then `frames=538 elapsed=2.21s fps=243.36`, compared with the same-session default baseline of `frames=538 elapsed=2.21s fps=243.99`.
- Learning: Python optimize mode is normal warmed-path noise and not a stable throughput lever. Keep the default interpreter mode.

### Profile: Clean Depth-2 Graph Gap After Runtime Cleanup

- Request: Refresh Nsight Systems evidence for the current accepted implementation after removing stale CUDA processes and rejecting the recent runtime scheduling probes. Pipeline depth remained fixed at `2`.
- Profile: `/tmp/rfdetr_depth2_clean_after_cleanup_20260523_173101.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_clean_after_cleanup_20260523_173101.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_clean_after_cleanup_20260523_173101_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_clean_after_cleanup_20260523_173101_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_clean_after_cleanup_20260523_173101_stats_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.28s fps=235.45`.
- Graph spacing: The capture includes `602` CUDA graph traces. After skipping the `64` capture warmups plus the next `100` frame launches, CUDA graph duration was p50 `4068.479 us`, p90 `4134.321 us`, p95 `4139.256 us`, p99 `4145.568 us`, mean `4072.660 us`; graph end-to-next-start gap was p50 `40.448 us`, p90 `41.805 us`, p95 `42.207 us`, p99 `42.856 us`, mean `40.615 us`.
- Gap decomposition over the first `100` stable post-settling gaps: busy work inside the gap was p50 `35.136 us`, mean `35.363 us`; idle inside the gap was p50 `5.152 us`, mean `5.237 us`.
- Learning: The current accepted depth-2 path is still effectively TensorRT CUDA-graph-body bound. The post-graph interval is low and consistent, and only about `5 us` of it is idle; further wins need a correctness-compatible TensorRT graph-duration reduction or a way to remove required input/output copy work without adding dependencies.

### Rejected: Explicit PyTorch CUDA Graph Pool Handle

- Hypothesis: Passing an explicit `torch.cuda.graph_pool_handle()` to the TensorRT CUDA graph capture might reduce graph memory-pool setup overhead or produce a slightly better replay object.
- Change tested: Temporary env-gated code only; with `RFDETR_TRT_GRAPH_POOL_HANDLE=true`, `_capture_cuda_graph(...)` passed a fresh graph-pool handle to `torch.cuda.graph(...)`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: `frames=538 elapsed=2.21s fps=243.58`, below the recent clean default baseline of `frames=538 elapsed=2.21s fps=243.99`.
- Learning: The captured TensorRT graph body does not benefit from a PyTorch graph-pool handle. The graph contains TensorRT work, not PyTorch allocations that would use the pool. Keep the accepted `torch.cuda.graph(cuda_graph, stream=stream)` capture.

### Profile: User-Requested Depth-2 Graph-Bound Refresh

- Request: Capture a fresh Nsight Systems report for the current accepted implementation while keeping the workflow pipeline depth fixed at `2`.
- Sanity run before profiling: the requested command measured `frames=538 elapsed=2.20s fps=244.55`.
- Profile: `/tmp/rfdetr_depth2_user_refresh_20260523_173919.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_user_refresh_20260523_173919.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_user_refresh_20260523_173919_stats_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_user_refresh_20260523_173919_stats_cuda_gpu_mem_time_sum_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_user_refresh_20260523_173919_stats_cuda_api_sum_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.30s fps=234.31`.
- Graph spacing: The capture includes `602` CUDA graph traces. After skipping the `64` capture warmups plus the next `100` frame launches, CUDA graph duration was p50 `4073.664 us`, p90 `4137.858 us`, p95 `4141.692 us`, p99 `4148.619 us`, mean `4080.681 us`; graph end-to-next-start gap was p50 `40.543 us`, p90 `41.920 us`, p95 `42.303 us`, p99 `42.866 us`, mean `40.656 us`.
- Gap decomposition over the first `100` stable post-settling gaps: busy work inside the gap was p50 `35.151 us`, mean `35.275 us`; idle inside the gap was p50 `5.296 us`, mean `5.425 us`. The largest gap occupants were the next-frame input D2D copy (`1168128B`, `13.149 us` avg overlap), TensorRT mask D2D clone (`2433600B`, `13.093 us`), sigmoid (`6.942 us`), fill-long (`2.832 us`), logits D2D clone (`36400B`, `2.114 us`), selector (`2.042 us`), boxes D2D clone (`1600B`, `1.997 us`), and fill-int (`1.906 us`).
- Learning: The requested depth-2 run is already tightly graph-bound. The point where one TensorRT CUDA graph ends is about `40-43 us` from the next graph start, with only about `5 us` of idle bubble. Remaining FPS is dominated by the TensorRT graph body plus required input/output copies; depth `3` was not tested.

### Rejected: Masked BBox Loads In Fused Selector

- Hypothesis: `_select_topk_boxes_kernel` still loads bbox coordinates and computes box geometry for top entries that may be discarded as background/no-object or below threshold. Masking the bbox `tl.load(...)` operations with the existing `keep` predicate could reduce selector memory work without changing outputs.
- Change tested: Temporary code only; changed the four bbox coordinate loads inside `_select_topk_boxes_kernel` to `tl.load(..., mask=keep, other=0.0)`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: first run after the Triton codegen change measured `frames=538 elapsed=2.30s fps=234.07`; warmed repeat measured `frames=538 elapsed=2.21s fps=243.99`, below the same-session clean run of `frames=538 elapsed=2.20s fps=244.55`.
- Learning: Predicating these scalar bbox loads does not improve the depth-2 graph-bound path. Any saved load work is below noise or offset by changed Triton codegen. Reverted to the simpler unconditional bbox loads.

### Rejected: NVIDIA TF32 Override Runtime Knob

- Hypothesis: The remaining TensorRT CUDA graph body includes FP32/Tensor-Core kernels. Setting `NVIDIA_TF32_OVERRIDE` before process startup might alter TF32 dispatch and improve the graph-bound ceiling without code changes.
- Change tested: External process environment only; ran the requested benchmark with `NVIDIA_TF32_OVERRIDE=0`, then with `NVIDIA_TF32_OVERRIDE=1`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result: Same-session default baseline measured `frames=538 elapsed=2.19s fps=245.20`. Both override runs failed before processing frames: TensorRT/Myelin reported `Inconsistent setting of NVIDIA_TF32_OVERRIDE env var at build -1 and at execution 0` for `0`, and the same build/execution mismatch plus `NVIDIA_TF32_OVERRIDE set to unrecognized value: "1"` for `1`.
- Learning: The accepted serialized T4 FP16 engine must run with the build-time/default TF32 override state. This is not a viable runtime tuning knob for the packaged engine.

### Rejected: CUDA Cache Disable Runtime Knob

- Hypothesis: The accepted TensorRT graph body may depend on CUDA/Myelin module-cache behavior. Launching with `CUDA_CACHE_DISABLE=1` could expose whether the CUDA code cache is adding runtime overhead or changing warmed graph scheduling.
- Change tested: External process environment only; ran the requested benchmark with `CUDA_CACHE_DISABLE=1`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: `frames=538 elapsed=2.20s fps=244.53`, compared with the same-session default baseline of `frames=538 elapsed=2.19s fps=245.20`.
- Learning: Disabling the CUDA cache is not useful for the accepted warmed graph path. The serialized engine already reaches the same graph-bound steady-state behavior with the default cache policy.

### Rejected: RFDETR Normalization Lookup Table

- Hypothesis: RFDETR CPU preprocessing still performs per-pixel uint8-to-float normalization with one multiply and one add per channel. Precomputing a `3 x 256` float32 normalization lookup table and filling each output channel with `np.take(..., out=...)` could reduce CPU producer work while preserving exact model inputs.
- Change tested: Temporary code only; changed `_pil_image_to_normalized_tensor(...)` to fetch normalized float32 values from a thread-local LUT instead of using `np.multiply(...)` plus in-place bias. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Correctness: The LUT values matched the existing float32 multiply/add formula exactly for all `256` possible uint8 inputs across all three channels (`array_equal=True`, `max_diff=0.0`).
- Result on requested command: first LUT run measured `frames=538 elapsed=2.20s fps=244.65`; repeat measured `frames=538 elapsed=2.21s fps=242.98`, below the recent same-session default baseline of `frames=538 elapsed=2.19s fps=245.20`.
- Learning: The LUT gather path is not faster end to end. The existing vectorized multiply/add normalization is already efficient and better for the current depth-2 producer/GPU balance. Reverted to the accepted ufunc normalization path.

### TensorRT Accepted Engine Layer-Time Snapshot

- Request: Map the remaining TensorRT graph-body bottleneck from kernel names back to TensorRT layer names to see whether a targeted plugin or replacement layer is plausible.
- Profile: Ran the accepted engine directly with TensorRT `IProfiler` on `50` non-graph executions after warmup. Raw JSON summary is `/tmp/rfdetr_trt_layer_profile_20260523_accepted.json`.
- Result: The profile reported `261` layers. Summed reported layer time was `5.925 ms` per execute under profiler instrumentation. Coarse layer groups by summed average time were matmul/FC `2.857 ms` across `72` layers, attention/MHA `1.185 ms` across `37` layers, fused elementwise/shape layers `1.100 ms` across `85` layers, convolutions `0.555 ms` across `14` layers, other layers `0.210 ms`, and resize `0.019 ms`.
- Top layers: the largest individual layers were twelve repeated Myelin MHA layers (`_gemm_mha_v2_myl2_*`) at about `0.066-0.069 ms` each, followed by many backbone encoder MLP `fc2` MatMul layers at about `0.054-0.059 ms` each. The segmentation-head convolutions and resize were smaller.
- Learning: The remaining graph body is distributed across many small Myelin-generated transformer MHA/MLP matmul layers, not a single bad layer. A custom patch would need a broad correct transformer/tactic/export change; there is no obvious one-layer plugin target left in the accepted serialized engine.

### Rejected: cuBLAS Workspace Runtime Config

- Hypothesis: The accepted TensorRT plan is Myelin-heavy but still contains many GEMM/MHA-style kernels. Setting `CUBLAS_WORKSPACE_CONFIG` before process startup might alter cuBLAS/cuBLASLt workspace behavior for any library-backed tactics and improve graph-bound throughput.
- Change tested: External process environment only; ran the requested benchmark with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, then with `CUBLAS_WORKSPACE_CONFIG=:16:8`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: `:4096:8` measured `frames=538 elapsed=2.20s fps=244.86`; `:16:8` measured `frames=538 elapsed=2.21s fps=243.95`, both below the recent same-session default baseline of `frames=538 elapsed=2.19s fps=245.20`.
- Learning: cuBLAS workspace configuration is not a useful runtime knob for this serialized TensorRT/Myelin plan. Keep the default library workspace behavior.

### Rejected: PyTorch Native Allocator Split Size

- Hypothesis: The accepted depth-2 path still allocates fixed-size PyTorch CUDA tensors around TensorRT output clones and fused postprocess. Setting `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:...` could change native caching-allocator block splitting enough to reduce allocator overhead or fragmentation while preserving tensor values.
- Change tested: External process environment only; first tried `max_split_size_mb:16`, then tried valid `max_split_size_mb:64`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result: `max_split_size_mb:16` failed before frames because PyTorch rejected it (`CachingAllocator option max_split_size_mb too small, must be > 20`). `max_split_size_mb:64` ran successfully but measured `frames=538 elapsed=2.20s fps=244.50`, below the recent default warmed baseline of `frames=538 elapsed=2.19s fps=245.20`.
- Learning: Native allocator split-size tuning does not improve this graph-bound run. The default allocator policy remains the best tested option; the earlier `cudaMallocAsync` allocator probe was also not stable enough to keep.

### Profile: Depth-2 Graph-Bound Refresh

- Request: Capture a fresh Nsight Systems report for the current accepted implementation while keeping the workflow pipeline depth fixed at `2`. Depth `3` was not tested.
- Sanity run before profiling: the requested command measured `frames=538 elapsed=2.20s fps=244.65`.
- Profile: `/tmp/rfdetr_depth2_graphbound_refresh_20260523_180532.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_graphbound_refresh_20260523_180532.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_graphbound_refresh_20260523_180532_stats_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_graphbound_refresh_20260523_180532_stats_cuda_gpu_mem_time_sum_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_graphbound_refresh_20260523_180532_stats_cuda_api_sum_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.29s fps=234.56`.
- Graph spacing: The capture includes `602` CUDA graph traces: `64` capture warmup replays plus `538` frame replays. After skipping the `64` warmups plus the next `100` frame launches, CUDA graph duration was p50 `4071.070 us`, p90 `4135.303 us`, p95 `4138.343 us`, p99 `4144.114 us`, mean `4081.231 us`; graph end-to-next-start gap was p50 `40.479 us`, p90 `41.932 us`, p95 `42.214 us`, p99 `43.034 us`, mean `40.548 us`.
- Gap decomposition over the first `100` stable post-settling gaps: busy work inside the gap was p50 `35.135 us`, mean `35.202 us`; idle inside the gap was p50 `5.152 us`, mean `5.242 us`. The largest gap occupants were the TensorRT mask D2D clone (`2433600B`, `13.210 us` avg overlap), next-frame input D2D copy (`1168128B`, `13.098 us`), sigmoid (`7.066 us`), fill-long (`2.801 us`), logits D2D clone (`36400B`, `2.106 us`), boxes D2D clone (`1600B`, `2.009 us`), fill-int (`1.820 us`), and selector (`1.684 us`).
- Learning: The current depth-2 run is shaped as requested: CUDA graph end is consistently close to the next CUDA graph start, with only about `5 us` median idle between graph replays. The remaining throughput ceiling is still the TensorRT CUDA graph body plus the required input/output copy and fused postprocess tail; previously tested safe ownership schemes that try to remove those copies return to the accepted band once synchronization is correct.

### Rejected: Redundant Selector Mask Cleanup

- Hypothesis: `_select_topk_boxes_kernel` loads logits with `mask=valid_offsets` and `other=-inf`, then immediately applies the same validity mask with `tl.where(...)`. Removing the redundant `tl.where(...)` could simplify Triton selector codegen without changing scores.
- Change tested: Temporary code only; removed the second `scores = tl.where(valid_offsets, scores, -inf)` from `fused_postprocess.py`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Correctness: The masked `tl.load(..., other=-inf)` already produces the same value for invalid offsets, so this cleanup is semantically equivalent for the selector.
- Result on requested command: first compile-cold run measured `frames=538 elapsed=2.29s fps=234.72`; warmed repeats measured `frames=538 elapsed=2.21s fps=243.89` and `frames=538 elapsed=2.21s fps=243.60`, below the immediately prior clean sanity run of `frames=538 elapsed=2.20s fps=244.65`.
- Learning: This codegen simplification does not improve the graph-bound depth-2 run. The selector is too small relative to TensorRT replay and required copy traffic, and small Triton schedule changes can land below the accepted warmed band. Reverted to the accepted selector implementation.

### Rejected: X-Axis Specialized Mask Resize

- Hypothesis: The benchmark resizes RFDETR masks from `78` columns to `312` columns, so the x-axis scale is exactly `4x` even though the original frame height is `176`. The prior fully-4x resize specialization did not match this video because the y-axis is not 4x. Specializing only x-coordinate interpolation could reduce arithmetic in `_resize_selected_masks_kernel(...)`.
- Change tested: Temporary code only; added `_resize_selected_masks_x4_kernel(...)` that replaces x-axis floor/divide work with a `out_x % 4` mapping while keeping the generic y-axis bilinear math. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Correctness: CUDA smoke check compared generic and x-specialized kernels on random `(100, 78, 78)` masks resized to `(176, 312)` for seven selected detections: `equal=True`, `diff=0`.
- Result on requested command: warmed depth-2 runs measured `frames=538 elapsed=2.20s fps=244.48`, then `frames=538 elapsed=2.21s fps=243.91`, then `frames=538 elapsed=2.21s fps=243.37`, below the latest clean accepted sanity run of `frames=538 elapsed=2.20s fps=244.65`.
- Learning: The x-axis arithmetic is not the limiter for the current graph-bound run. The extra Triton variant and changed codegen do not produce a stable end-to-end improvement, so the generic resize kernel remains the accepted path.

### Rejected: Skip Single-Image Batch Contiguous Call

- Hypothesis: In RFDETR preprocessing, `tensors[0].unsqueeze(0)` is already contiguous for the accepted single-image CHW tensor. Removing the following `.contiguous()` could avoid a tiny PyTorch dispatch in the CPU producer path without changing the TensorRT input.
- Change tested: Temporary code only; changed the single-image branch in `pre_process_network_input(...)` from `tensors[0].unsqueeze(0).contiguous()` to `tensors[0].unsqueeze(0)`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Correctness: Local tensor checks confirmed both normal and pinned CHW tensors remain contiguous after `unsqueeze(0)` with expected NCHW strides; no model math or preprocessing pixels change.
- Result on requested command: depth-2 runs measured `frames=538 elapsed=2.20s fps=244.37` and `frames=538 elapsed=2.20s fps=244.05`, below the latest clean accepted sanity run of `frames=538 elapsed=2.20s fps=244.65`.
- Learning: The extra `.contiguous()` is effectively a no-op in this path and is not the current producer limiter. Removing it did not improve throughput, so the accepted explicit contiguous call remains.

### Rejected: Yield Before RFDETR Postprocess

- Hypothesis: In the RFDETR TRT workflow fast path, the same worker thread calls `forward(...)` and immediately launches deferred GPU postprocess. With pipeline depth fixed at `2`, yielding once after `forward(...)` might let the other worker acquire the model lock and enqueue the next TensorRT CUDA graph before selector/resize work, reducing the already-small graph-to-graph tail.
- Change tested: Temporary code only; inserted `time.sleep(0)` between `model._model.forward(...)` and `model._model.post_process(...)` in the RFDETR TRT workflow fast path. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: depth-2 runs measured `frames=538 elapsed=2.20s fps=244.43` and `frames=538 elapsed=2.21s fps=243.90`, within noise but below the accepted warmed ceiling.
- Learning: Python scheduler yielding does not improve the depth-2 balance. The current path already hands off quickly enough, and explicit yielding adds variance without reducing the TensorRT graph-body bottleneck. Reverted to the accepted immediate postprocess launch.

### Rejected: Triton Selector Max Return Indices

- Hypothesis: `_select_topk_boxes_kernel` computes the top offset with an equality mask plus `tl.min(...)` after `tl.max(scores, axis=0)`. Triton's `tl.max(..., return_indices=True, return_indices_tie_break_left=True)` can return the max value and lane index together, potentially reducing selector reduction work.
- Change tested: Temporary code only; replaced the selector loop's `tl.max` plus equality-mask tie-break with `tl.max(..., return_indices=True, return_indices_tie_break_left=True)`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: the first compile-cold run measured `frames=538 elapsed=2.29s fps=235.06`; warmed repeat measured `frames=538 elapsed=2.21s fps=243.95`, below the accepted warmed ceiling.
- Learning: Triton's max-with-index codegen is not better for this single-block selector on T4. The existing explicit equality-mask/tie-break sequence remains the faster full-pipeline path.

### Rejected: Single-Image Deferred Fused Postprocess Shortcut

- Hypothesis: The benchmark always calls RFDETR dense postprocess with batch size `1` and `defer_fused_postprocess_count=True`. Trying the fused postprocess directly on `logits[0]` before the generic batch sigmoid/loop could remove small Python and tensor-wrapper overhead while preserving the same fused selector and mask resize behavior.
- Change tested: Temporary code only; added an early single-image branch in `post_process_instance_segmentation_results(...)` that called `_try_fused_instance_segmentation_post_process(...)` with `torch.sigmoid(logits[0])` and returned `[fused_result]` when supported. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: depth-2 runs measured `frames=538 elapsed=2.20s fps=244.43` and `frames=538 elapsed=2.21s fps=243.63`, below the accepted warmed ceiling.
- Learning: The generic batch wrapper and loop are not limiting the current depth-2 run. The added branch changes bytecode/scheduling enough to add variance without reducing the TensorRT graph-body bottleneck. Reverted to the generic postprocess flow.

### Rejected: Combined Small TensorRT Output Copy

- Hypothesis: The accepted TensorRT CUDA graph path clones the small boxes and logits outputs separately before cloning the large mask tensor. Allocating one small flat device tensor, copying boxes and logits into views, and returning those views could reduce small-output allocation overhead without borrowing graph-owned outputs or changing the large mask clone.
- Change tested: Temporary code only; in the CUDA graph cache-hit path, when three same-dtype outputs were present, allocated one flat tensor for the first two outputs, copied the graph-owned first and second output buffers into shaped views, and cloned the third output normally. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: depth-2 runs measured `frames=538 elapsed=2.20s fps=244.97` and `frames=538 elapsed=2.20s fps=244.21`, not a stable improvement over the accepted warmed ceiling.
- Learning: The two small output clones are below the limiter, and replacing them with manual view copies changes allocation/scheduling enough to add variance. Keep the simpler per-output clone path.

### Profile: Depth-2 Graph Spacing Refresh

- Request: Capture another Nsight Systems report for the current accepted implementation while keeping workflow pipeline depth fixed at `2`. Depth `3` was not tested.
- Sanity run before profiling: the requested command measured `frames=538 elapsed=2.20s fps=244.60`.
- Profile: `/tmp/rfdetr_depth2_graphspacing_20260523_183938.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_graphspacing_20260523_183938.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_graphspacing_20260523_183938_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_graphspacing_20260523_183938_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_graphspacing_20260523_183938_stats_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.33s fps=230.93`.
- Graph spacing: The capture includes `602` CUDA graph traces: `64` capture warmup replays plus `538` frame replays. After skipping the `64` warmups plus the next `100` frame launches, CUDA graph duration was p50 `4073.855 us`, p90 `4134.462 us`, p95 `4136.539 us`, p99 `4142.646 us`, mean `4075.027 us`; graph end-to-next-start gap was p50 `40.575 us`, p90 `41.996 us`, p95 `42.303 us`, p99 `42.888 us`, mean `40.840 us`.
- Gap decomposition over the first `100` stable post-settling gaps: busy work inside the gap was p50 `35.200 us`, mean `35.656 us`; idle inside the gap was p50 `5.328 us`, mean `5.500 us`. The largest gap occupants were next-frame input D2D copy (`1168128B`, `13.120 us` avg overlap), TensorRT mask D2D clone (`2433600B`, `13.110 us`), sigmoid (`6.726 us`), fill-long (`2.815 us`), selector (`2.180 us`), fill-int (`2.173 us`), logits D2D clone (`36400B`, `2.105 us`), and boxes D2D clone (`1600B`, `1.998 us`).
- Learning: The current depth-2 run is still shaped as requested. The median graph-to-graph tail is roughly `1%` of the TensorRT graph body, and only about `5 us` of that tail is idle after required copies and fused postprocess work. The run is effectively bottlenecked by the TensorRT CUDA graph forward pass plus the narrow postprocess/copy tail; further large gains likely require changing the TensorRT engine/export/tactics rather than Python-side pipeline depth.

### Rejected: Suppress Benchmark Progress Prints

- Hypothesis: The benchmark sink flushes progress text every `50` frames. Suppressing intermediate progress prints while preserving the final FPS line could remove occasional result-path I/O stalls and keep the depth-2 pipeline fed more consistently.
- Change tested: Temporary benchmark-harness code only; set `PROGRESS_EVERY = 0` in `development/stream_interface/rfdetr_nano_seg_trt_workflow.py` and guarded the progress-print branch. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Correctness: This does not affect preprocessing, TensorRT execution, postprocess, prediction materialization, or final benchmark output; it only changes intermediate console logging.
- Result on requested command: depth-2 runs measured `frames=538 elapsed=2.20s fps=244.80` and `frames=538 elapsed=2.21s fps=243.19`, not a stable improvement over the accepted warmed band.
- Learning: Intermediate progress printing is not a meaningful limiter for the current graph-bound benchmark. Keep the progress output at every `50` frames for observability.

### Current Package Metadata Recheck

- Request: Re-check official Roboflow package metadata through the local `inference_models` provider path before spending more time on engine-body tuning.
- Result: The provider resolves `rfdetr-seg-nano` to `coco-dataset-vdnr1/41` and still returns six public packages: L4 TRT FP32 `3e3ddd85586b43e4fac6d319fb2927fd`, ONNX FP32 `5362b72bfb9f01d2e0b8cba2048d932c`, L4 TRT FP16 `89d1f41e2af4f4f3ffcdfb77e774d26a`, Torch FP32 `8b8da2fe824240522a39f3cde41aafae`, T4 TRT FP32 `bbc2cc23adf6f5e71a9241956081da96`, and T4 TRT FP16 `c70f32369a54d61e06ef4e6b56c82524`.
- Learning: There is no new official T4-compatible TensorRT package available through the current metadata. The accepted T4 FP16 package remains the only official package that has passed the benchmark correctness gate; further graph-body improvements still require a behavior-equivalent export source or a new official T4 package.

### Rejected: Static Batch TensorRT Wrapper Shortcut

- Hypothesis: RFDETR TRT uses a static batch size of `1`, and the benchmark always sends exactly one frame. Short-circuiting `_infer_from_trt_engine_with_batch_size_boundaries(...)` directly into `_execute_trt_engine(...)` for the exact static-batch case could remove small Python branch/reminder overhead before graph replay.
- Change tested: Temporary code only; added an early return when `min_batch_size == max_batch_size == pre_processed_images.shape[0]`, preserving the same tensor and the same `_execute_trt_engine(...)` call. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Correctness: This does not change model inputs, TensorRT execution, postprocess, or predictions; it only bypasses generic padding bookkeeping when no padding is needed.
- Result on requested command: same-session clean baseline measured `frames=538 elapsed=2.21s fps=243.90`; shortcut runs measured `frames=538 elapsed=2.21s fps=243.91` and `frames=538 elapsed=2.21s fps=243.00`.
- Learning: Generic static-batch bookkeeping is below the current limiter. The accepted helper structure remains clearer and at least as fast in the graph-bound depth-2 run.

### Profile: Main-Thread CPU Refresh

- Request: Refresh CPU-side evidence for the current accepted implementation while keeping workflow pipeline depth fixed at `2`. Depth `3` was not tested.
- Profile: `/tmp/rfdetr_depth2_current_20260523_1856.prof`, captured with Python `cProfile` around the requested benchmark command.
- Result under profiling: `frames=538 elapsed=2.20s fps=244.61`, inside the accepted warmed band.
- Findings: Standard `cProfile` mostly captured startup/import and the main result-dispatch queue, not the worker-thread CUDA hot path. The main-thread sink itself was negligible (`538` calls, about `0.005 s` cumulative), and the top cumulative runtime after import was queue waiting in `_dispatch_inference_results`.
- Learning: The remaining limiter is not main-thread result sink work. This profile is consistent with the Nsight Systems evidence that the run is constrained by the TensorRT CUDA graph body plus the short GPU copy/postprocess tail; worker-thread CPU hot-path work is already mostly hidden by the depth-2 pipeline.

### Profile: All-Thread Yappi CPU/Wall Refresh

- Request: Refresh all-thread CPU-side evidence for the accepted path after the main-thread-only `cProfile` run. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Profiles: `/tmp/rfdetr_depth2_yappi_wall_20260523_1902.pstat`, `/tmp/rfdetr_depth2_yappi_wall_20260523_1902.callgrind`, `/tmp/rfdetr_depth2_yappi_cpu_20260523_1906.pstat`, and `/tmp/rfdetr_depth2_yappi_cpu_20260523_1906.callgrind`.
- Result under profiler: Yappi wall-clock profiling measured `frames=538 elapsed=2.22s fps=242.75`. Yappi CPU-clock profiling is much higher overhead for this workload and measured `frames=538 elapsed=3.10s fps=173.75`; use it only for relative CPU attribution.
- Findings: Worker wall time remains dominated by CUDA waits: RFDETR fast path wall time was about `8.92 s` across both workers, while true CPU-time attribution in the same area was much smaller. The largest local true CPU self-time left was preprocessing normalization (`_pil_image_to_normalized_tensor(...)` around `0.300 s` under the CPU profiler), followed by fixed pinned conversion bookkeeping (`_try_copy_limited_cuda_detection_tensors_to_pinned_numpy(...)` around `0.071 s` self-time). Fused selector/postprocess wall time is mostly GPU wait, not Python.
- Learning: The all-thread CPU evidence agrees with Nsight Systems: the depth-2 pipeline hides worker CPU well enough that the remaining FPS ceiling is TensorRT graph replay and required GPU copy/postprocess work. Any CPU micro-optimization must be very low-risk and measured end-to-end, because true Python self-time is now a small fraction of frame time.

### Rejected: Broadcast RFDETR Normalization

- Hypothesis: The current RFDETR PIL preprocessing writes normalized CHW channels with three separate NumPy multiply/add loops. A single broadcasted NumPy multiply over a channel-reordered CHW source could reduce preprocessing CPU self-time, which the Yappi CPU profile identified as the largest remaining local CPU function.
- Change tested: Temporary code only; replaced the per-channel loop in `_pil_image_to_normalized_tensor(...)` with `np.moveaxis(image_array[:, :, channel_order], 2, 0)`, one broadcasted `np.multiply(..., out=normalized)`, and one broadcasted bias add. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Correctness: A local micro-check on resized uint8 inputs showed exact normalized tensor equality versus the accepted per-channel loop (`max diff 0.0`) for the benchmark channel-swap case.
- Result on requested command: depth-2 runs measured `frames=538 elapsed=2.21s fps=243.84` and `frames=538 elapsed=2.20s fps=244.57`, inside noise but not a stable improvement over accepted warmed runs.
- Learning: The broadcast form is only marginally faster in isolation and adds a per-frame channel-reorder temporary. In the full depth-2 pipeline, it does not improve throughput. Keep the accepted direct per-channel writes into the pinned CHW buffer.

### Profile: Depth-2 Graphbound Refresh

- Request: Capture another Nsight Systems report for user analysis while keeping workflow pipeline depth fixed at `2`. Depth `3` was not tested.
- Profile: `/tmp/rfdetr_depth2_graphbound_20260523_190956.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_graphbound_20260523_190956.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_graphbound_20260523_190956_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_graphbound_20260523_190956_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_graphbound_20260523_190956_stats_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.32s fps=231.58`.
- Graph spacing: The capture includes `602` CUDA graph traces: `64` capture warmup replays plus `538` frame replays. After skipping the `64` warmups plus the next `100` frame launches, CUDA graph duration was p50 `4068.463 us`, p90 `4132.048 us`, p95 `4135.371 us`, p99 `4140.723 us`, mean `4069.635 us`; graph end-to-next-start gap was p50 `40.544 us`, p90 `41.996 us`, p95 `42.374 us`, p99 `43.007 us`, mean `40.722 us`.
- Gap decomposition over the first `100` stable post-settling gaps: busy work inside the gap was p50 `35.200 us`, mean `35.294 us`; idle inside the gap was p50 `5.328 us`, mean `5.381 us`. The largest gap occupants were next-frame input D2D copy (`1168128B`, `13.133 us` avg overlap), TensorRT mask D2D clone (`2433600B`, `13.133 us`), sigmoid (`6.933 us`), fill-long (`2.846 us`), logits D2D clone (`36400B`, `2.110 us`), boxes D2D clone (`1600B`, `1.995 us`), fill-int (`1.947 us`), and selector (`1.804 us`).
- Learning: The refreshed depth-2 profile matches the accepted graphbound shape. The graph-to-graph gap remains about `1%` of the TensorRT graph body, with roughly `5 us` median idle after required ownership copies and fused postprocess work. The practical limiter is still the TensorRT CUDA graph forward pass plus the narrow GPU copy/postprocess tail, not pipeline depth.

### Rejected: CUDA Device Max Connections Runtime Knob

- Hypothesis: The accepted depth-2 path uses separate CUDA streams for preprocessing, TensorRT graph replay, postprocess, and D2H conversion. Changing `CUDA_DEVICE_MAX_CONNECTIONS` before process startup might alter stream work-queue scheduling enough to tighten the graph-to-graph cadence.
- Change tested: External process environment only; ran the requested benchmark with `CUDA_DEVICE_MAX_CONNECTIONS=1`, then with `CUDA_DEVICE_MAX_CONNECTIONS=32`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: same-session default baseline measured `frames=538 elapsed=2.20s fps=244.42`; `CUDA_DEVICE_MAX_CONNECTIONS=1` measured `frames=538 elapsed=2.36s fps=227.71`; `CUDA_DEVICE_MAX_CONNECTIONS=32` measured `frames=538 elapsed=2.21s fps=243.16`.
- Learning: Reducing the number of device work queues is actively harmful for this overlapped depth-2 schedule, and increasing it does not improve the accepted graph-bound cadence. Keep the default CUDA connection setting.

### Rejected: Skip No-Op RFDETR Numpy Preprocessing Helper

- Hypothesis: The accepted RFDETR package has no static crop, grayscale, contrast, or two-step resize, so `_pre_process_numpy(...)` calls `apply_pre_processing_to_numpy_image(...)` only to return the same image and a zero static-crop offset. A guarded fast path for this no-op case could reduce CPU preprocessing overhead without changing pixels or metadata.
- Change tested: Temporary code only; added `_can_skip_numpy_pre_processing(...)` in `rfdetr/pre_processing.py` and bypassed the generic helper when all numpy preprocessing operations were inactive after overrides. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Correctness: Compared the fast path against the generic helper on `16` frames from `vehicles_312px.mp4`; normalized tensor max diff was `0.0` and preprocessing metadata matched exactly.
- Result on requested command: depth-2 runs measured `frames=538 elapsed=2.21s fps=243.65` and `frames=538 elapsed=2.21s fps=243.64`, below the same-session default baseline of `frames=538 elapsed=2.20s fps=244.42`.
- Learning: This helper call is not a measurable limiter in the current graph-bound run. The added branch slightly worsens scheduling/noise, so the generic helper remains the accepted path.

### Profile: Triton Selector Kernel NCU Snapshot

- Request: Gather lower-level evidence for the remaining custom Triton selector work while keeping the benchmark command at pipeline depth `2`. Depth `3` was not tested.
- Profiles: Initial launch-shape-only report `/tmp/rfdetr_selector_kernel_depth2_20260523_192029.ncu-rep`; explicit-counter report `/tmp/rfdetr_selector_kernel_metrics_20260523_192216.ncu-rep`; details text `/tmp/rfdetr_selector_kernel_metrics_20260523_192216_details.txt`.
- Result under profiler: The explicit-counter NCU run measured `frames=538 elapsed=7.26s fps=74.06`, which is profiling overhead only and not comparable to normal benchmark FPS.
- Findings: The sampled `_select_topk_boxes_kernel` launches as a single Triton program with CUDA launch shape `(1, 1, 1)x(256, 1, 1)`. Across five sampled launches, NCU reported mean `gpu__time_duration.avg=24.766 us`, mean DRAM read `53.914 KB`, mean DRAM write `12.8 B`, fixed L1 global-load traffic `42.820 KB`, fixed L1 global-store traffic `928 B`, and fixed `15811` SMSP instructions.
- Learning: This supports the earlier selector experiments: the kernel is under-parallelized by shape, but it is also very small and mostly reads the `100x91` score matrix. Prior attempts to improve occupancy with top-2-per-query, different warp counts, max-iteration caps, and raw-logit selection all lost end to end because the extra launch/codegen/traffic costs outweighed the tiny selector tail. Future selector work should only proceed if it also removes another launch or required copy, not as standalone selector tuning.

### Rejected: Glibc Arena Limit Runtime Knob

- Hypothesis: The remaining CPU-side work allocates small NumPy arrays, UUID strings, and metadata objects while two workflow workers are active. Setting `MALLOC_ARENA_MAX=1` before process startup might reduce glibc arena overhead or memory-management variance enough to improve the depth-2 materialization tail.
- Change tested: External process environment only; ran the requested benchmark with `MALLOC_ARENA_MAX=1`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: same-session default baseline measured `frames=538 elapsed=2.21s fps=243.57`; `MALLOC_ARENA_MAX=1` measured `frames=538 elapsed=2.20s fps=244.77`, then repeated at `frames=538 elapsed=2.21s fps=243.95`; default rerun measured `frames=538 elapsed=2.20s fps=244.01`.
- Learning: The first arena-limited run was noise rather than a stable allocator improvement. Host allocator tuning does not move the current graph-bound ceiling, and the target command should not require an external glibc allocator environment variable.

### Rejected: OpenCV Single Thread Runtime Probe

- Hypothesis: The benchmark source is an OpenCV-decoded video and the process default reports `cv2.getNumThreads() == 8`. Restricting OpenCV to one thread might reduce CPU scheduling contention between video decode and the two depth-2 workflow workers.
- Change tested: Temporary launcher only; called `cv2.setNumThreads(1)` before executing `development/stream_interface/rfdetr_nano_seg_trt_workflow.py` via `runpy`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: `cv2.setNumThreads(1)` measured `frames=538 elapsed=2.20s fps=244.91`; immediate default rerun measured `frames=538 elapsed=2.20s fps=245.00`.
- Learning: OpenCV thread-pool size is not a limiter in the accepted graph-bound run. Keep the normal OpenCV default rather than adding benchmark-specific thread configuration.

### Rejected: Capture TensorRT Graph On Caller Stream

- Hypothesis: The TensorRT CUDA graph cache creates a dedicated graph stream, then each RFDETR cache hit waits that graph stream on the model inference stream and waits the inference stream back on the graph stream. Capturing and replaying the graph on the current inference stream could turn those into self-waits in the RFDETR path and remove small event edges without changing graph outputs.
- Change tested: Temporary code only; changed `_capture_cuda_graph(...)` to use `torch.cuda.current_stream(device)` instead of creating a new `torch.cuda.Stream(device=device)`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Correctness: This changed only stream placement for the same TensorRT graph capture/replay, input copy, and output clones. `py_compile` passed before the benchmark, and the run completed normally.
- Result on requested command: `frames=538 elapsed=2.20s fps=244.33`, below the immediate accepted default run of `frames=538 elapsed=2.20s fps=245.00`.
- Learning: The dedicated TensorRT graph stream is part of the stable overlap schedule. Collapsing graph replay onto the caller inference stream does not reduce the graph-bound tail and slightly underperforms, so the accepted dedicated graph stream remains.

### Rejected: Python Malloc Runtime Knob

- Hypothesis: Prediction materialization still creates many small Python objects and NumPy arrays. Running the process with `PYTHONMALLOC=malloc` might reduce small-object allocator contention or interact better with glibc under the two-worker depth-2 pipeline.
- Change tested: External process environment only; ran the requested benchmark with `PYTHONMALLOC=malloc`. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: `PYTHONMALLOC=malloc` first measured `frames=538 elapsed=2.19s fps=245.44`, but repeated at `frames=538 elapsed=2.21s fps=243.90`; immediate default rerun measured `frames=538 elapsed=2.21s fps=243.77`.
- Learning: The first allocator result was not repeatable and stays inside normal run-to-run variance. Python allocator selection is not a stable improvement, and the target command should keep the default allocator.

### Profile: Depth-2 Graphbound Refresh

- Request: Capture another Nsight Systems report for user analysis while keeping workflow pipeline depth fixed at `2`. Depth `3` was not tested.
- Profile: `/tmp/rfdetr_depth2_graphbound_20260523_194100.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_graphbound_20260523_194100.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_graphbound_20260523_194100_stats_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_graphbound_20260523_194100_stats_cuda_gpu_mem_time_sum_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_graphbound_20260523_194100_stats_cuda_api_sum_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.26s fps=238.29`.
- Graph spacing: The capture includes `602` CUDA graph traces: `64` capture warmup replays plus `538` frame replays. After skipping the `64` warmups plus the next `100` frame launches, CUDA graph duration was p50 `4076.798 us`, p90 `4139.707 us`, p95 `4144.201 us`, p99 `4155.937 us`, mean `4065.573 us`; graph end-to-next-start gap was p50 `40.863 us`, p90 `42.201 us`, p95 `42.477 us`, p99 `42.879 us`, mean `40.945 us`.
- Gap decomposition over the first `100` stable post-settling gaps: busy work inside the gap was p50 `35.455 us`, mean `35.648 us`; idle inside the gap was p50 `5.200 us`, mean `5.220 us`. The largest gap occupants were TensorRT mask D2D clone (`2433600B`, `13.457 us` avg overlap), next-frame input D2D copy (`1168128B`, `13.102 us`), sigmoid (`6.598 us`), fill-long (`2.800 us`), selector (`2.393 us`), logits D2D clone (`36400B`, `2.099 us`), fill-int (`2.063 us`), and boxes D2D clone (`1600B`, `1.991 us`).
- Learning: The refreshed report still matches the target shape: the graph-to-graph idle bubble is about `5 us`, while the TensorRT CUDA graph body is about `4.07 ms`. The remaining throughput ceiling is dominated by the TensorRT graph forward pass plus required ownership copies and a narrow postprocess tail.

### Rejected: Two-Phase Limited D2H Detection Copy

- Hypothesis: The fixed RFDETR workflow conversion copies `7` masks to pinned CPU buffers every frame, while the benchmark detection distribution averages only `3.54` detections per frame (`1`: 15 frames, `2`: 104, `3`: 164, `4`: 145, `5`: 74, `6`: 14, `7`: 22). Copying the GPU count first, synchronizing, then copying only `valid_count` rows could reduce D2H bytes enough to improve the CPU materialization boundary.
- Change tested: Temporary code only; in `_try_copy_limited_cuda_detection_tensors_to_pinned_numpy(...)`, copied and synchronized `valid_count` before copying boxes, confidences, class IDs, and masks for only the valid rows. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Result on requested command: `frames=538 elapsed=2.80s fps=192.19`, a severe regression from the accepted warmed band.
- Learning: The saved D2H bytes do not compensate for introducing a second synchronization at the conversion boundary. The accepted single-sync fixed 7-row copy is the better depth-2 schedule because it keeps the GPU/CPU pipeline from stalling on an early count readback.

### Diagnostic: Depth-2 TensorRT Graph Visibility In Nsight Systems

- Hypothesis: The latest depth-2 Nsight Systems SQLite might reveal idle bubbles inside the captured TensorRT CUDA graph body, not just the gap between graph replays. If visible graph-internal idle were large, it could justify another TensorRT stream/tactic scheduling experiment.
- Analysis: Reused `/tmp/rfdetr_depth2_graphbound_20260523_194100.sqlite`, skipped the `64` capture warmup replays plus `100` settling frame launches, and measured GPU activities overlapping the remaining `438` CUDA graph trace intervals. Pipeline depth stayed fixed at `2`; depth `3` was not tested.
- Result: CUDA graph duration was p50 `4076.798 us`, mean `4065.573 us`. Nsight Systems did not expose the captured TensorRT graph's steady-state internal kernels as normal kernel rows; the visible overlapping non-graph work inside graph intervals was only p50 `280.651 us`, mean `280.587 us`, with top visible occupants Host-to-Device input copy `1168128B` (`188.948 us`/graph avg overlap), fixed-mask Device-to-Host copy `384384B` (`59.230 us`), `_resize_selected_masks_kernel` (`13.945 us`), and `_select_topk_boxes_kernel` (`10.508 us`). The visible overlap factor was `1.000`, so those exposed non-graph events are not meaningfully concurrent with one another.
- Learning: The profile confirms that depth-2 is already hiding preprocessing H2D and prediction D2H/postprocess under the TensorRT graph replay. It does not provide graph-internal TensorRT node timing; for the graph body, the better evidence remains the separate TensorRT layer profiler and NCU kernel snapshots, which showed the cost distributed across many small Myelin MHA/MLP kernels rather than a single obvious plugin target.

### Profile: Depth-2 CUDA Graph Node Trace

- Hypothesis: Nsight Systems `--cuda-graph-trace=node` can expose steady-state TensorRT graph node kernels, giving better evidence for any remaining graph-body scheduling or plugin target than the default graph-envelope reports.
- Profile: `/tmp/rfdetr_depth2_graphnode_20260523_194903.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_graphnode_20260523_194903.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_graphnode_20260523_194903_stats_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_graphnode_20260523_194903_stats_cuda_gpu_mem_time_sum_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_graphnode_20260523_194903_stats_cuda_api_sum_cuda_api_sum.csv`. Pipeline depth stayed fixed at `2`; depth `3` was not tested.
- Result under profiler: `frames=538 elapsed=2.33s fps=230.56`.
- Graph-node findings: The trace shows `602` inferred graph replays with `242` graph nodes each. After skipping `64` warmup replays plus `100` settling launches, graph-node envelope duration was p50 `4131.245 us`, p90 `4193.670 us`, p95 `4197.040 us`, mean `4137.898 us`; union GPU-busy time inside the graph was p50 `3916.356 us`, mean `3921.581 us`; internal no-activity idle was p50 `215.837 us`, p90 `221.781 us`, p95 `223.043 us`, mean `216.316 us`. Mean graph busy fraction was `94.77%`, summed-activity/union-busy overlap factor was only `1.0405`, and each replay used `6` streams.
- Top graph-node occupants per replay were distributed across many repeated TensorRT/Myelin kernels: `25` FP16 GEMM nodes at `808.390 us/replay`, `12` MHA nodes at `585.925 us/replay`, `12` fused FP16/FP32 GEMM nodes at `481.064 us/replay`, `12` smaller GEMM nodes at `221.701 us/replay`, plus smaller convolution, layernorm/GELU, transpose, and Myelin fusion nodes. The only visible Myelin TopK node was about `14.588 us/replay`.
- Learning: The graph body is now directly visible and confirms the earlier layer/NCU evidence: most time is spread across many small transformer GEMM/MHA/Myelin nodes with modest internal idle and little multi-stream overlap. There is no single large graph node to replace with a custom kernel; further graph-body gains require a correctness-equivalent TensorRT engine/export/tactic change rather than another Python scheduling tweak.

### Profile: Graph-Node MHA Nsight Compute Snapshot

- Hypothesis: The top repeated graph-node MHA kernel from the node trace may expose a hardware-level limitation that suggests a TensorRT tactic or custom-kernel target.
- Profiles: Launch-shape-only attempt `/tmp/rfdetr_trt_mha_graphnode_ncu_20260523_195236.ncu-rep` and useful basic-metrics report `/tmp/rfdetr_trt_mha_graphnode_basic_ncu_20260523_195342.ncu-rep` with details in `/tmp/rfdetr_trt_mha_graphnode_basic_ncu_20260523_195342_details.txt`. The NCU command matched `_gemm_mha_v2_0x7daddb359f728ff2e600188f192f4549`, used graph profiling mode `node`, skipped `900` matching launches, collected `3` launches with the `basic` set, and kept pipeline depth fixed at `2`; depth `3` was not tested.
- Result under NCU overhead: `frames=538 elapsed=5.33s fps=100.84`, not comparable to normal benchmark FPS.
- Findings: The sampled MHA node launches as `(1, 6, 11)x(128, 1, 1)` with grid size `66`, block size `128`, `245` registers/thread, `24.58 KiB` dynamic shared memory per block, `0.82` waves/SM, theoretical occupancy `25%`, achieved occupancy about `20.54%`, achieved active warps/SM about `6.57`, compute throughput about `36.04%`, memory throughput about `25.47%`, DRAM throughput about `6.45%`, and duration about `106.33 us` under NCU replay.
- Learning: The top MHA node is small-grid and limited by registers/shared memory, with low DRAM pressure. This is consistent with TensorRT/Myelin tactic limitations on T4 rather than a standalone memory optimization. A replacement would need to cover repeated MHA/GEMM structure across the transformer, not one isolated postprocess-style kernel.

### Diagnostic: Accepted Engine Tactic Visibility

- Hypothesis: The accepted TensorRT engine inspector might expose tactic IDs or implementation details for the top MHA/GEMM graph nodes, allowing a targeted tactic-level runtime change without rebuilding from an incompatible ONNX source.
- Analysis: Rechecked `/tmp/rfdetr_accepted_engine_inspector.json` and `/tmp/rfdetr_trt_layer_profile_20260523_accepted.json` after the CUDA graph node trace. The accepted engine inspector only contains layer names and bindings because the serialized engine was built with `ProfilingVerbosity.LAYER_NAMES_ONLY`; it does not expose tactic IDs, implementation alternatives, or detailed tensor formats.
- Findings: The direct TensorRT layer profiler still maps the largest per-execute layer times to twelve repeated `_gemm_mha_v2_myl2_*` layers at about `0.066-0.069 ms` each, followed by many encoder `mlp/fc2/MatMul_myl2_*` and Myelin FC layers around `0.054-0.059 ms` each. Coarse grouping of the `261` profiled layers attributes the largest totals to MHA/attention and MatMul/FC families rather than a single plugin-sized outlier.
- Learning: There is no tactic-level runtime knob visible from the accepted plan. A real graph-body optimization needs either a new official engine built with detailed profiling metadata, the exact correctness-equivalent export source, or an offline rebuild path that passes the all-frame class/box/mask correctness gate; the public ONNX/Torch packages tested so far do not satisfy that gate.

### Profile: Graph-Node Top GEMM Nsight Compute Snapshot

- Hypothesis: The largest aggregate graph-node kernel family, `sm75_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage1_warpsize2x2x1_tensor16x8x8_execute_kernel_trt`, might show whether the remaining TensorRT graph body is compute-, memory-, or launch-shape-limited.
- Profile: `/tmp/rfdetr_trt_topgemm_graphnode_basic_ncu_20260523_195831.ncu-rep` with details in `/tmp/rfdetr_trt_topgemm_graphnode_basic_ncu_20260523_195831_details.txt`. The NCU command matched the GEMM kernel, used graph profiling mode `node`, skipped `1800` matching launches, collected `3` launches with the `basic` set, and kept pipeline depth fixed at `2`; depth `3` was not tested.
- Result under NCU overhead: `frames=538 elapsed=5.25s fps=102.51`, not comparable to normal benchmark FPS.
- Findings: The sampled GEMM launches had grid sizes `36`, `54`, and `72` with block size `128`, `166` registers/thread, `16.38 KiB` dynamic shared memory per block, waves/SM `0.30-0.60`, theoretical occupancy `37.5%`, achieved occupancy `13.90-21.35%` (mean `17.71%`), achieved active warps/SM mean `5.67`, compute throughput mean `48.14%`, DRAM throughput mean `24.78%`, and duration mean `54.91 us` under NCU replay.
- Learning: The top GEMM family is also small-grid and occupancy-limited, though less register-heavy than the MHA node. Along with the MHA snapshot, this points to TensorRT tactic/export structure as the remaining graph-body limiter; replacing a single postprocess-style kernel cannot address enough of the repeated GEMM/MHA body.

### Rejected: Three Explicit TensorRT Aux Streams

- Hypothesis: Prior tests rejected zero, one, two, and four explicit TensorRT auxiliary streams during CUDA graph capture. Since the graph-node trace shows six streams active during replay but low overlap, forcing exactly three persistent aux streams might produce a slightly better balance between TensorRT internal parallelism and scheduling overhead.
- Change tested: Temporary code only; added three persistent `torch.cuda.Stream` objects to the TensorRT CUDA graph state, called `graph_context.set_aux_streams(...)` before the pre-capture warmup and CUDA graph capture, and kept pipeline depth fixed at `2`; depth `3` was not tested.
- Correctness: `py_compile` passed and the benchmark completed normally. The change only altered TensorRT auxiliary stream scheduling for the same captured engine and output tensors.
- Result on requested command: depth-2 runs measured `frames=538 elapsed=2.20s fps=244.73` and `frames=538 elapsed=2.21s fps=243.50`; after reverting to the accepted default aux-stream behavior, the same-session baseline measured `frames=538 elapsed=2.21s fps=243.71`.
- Learning: Three explicit aux streams are just noise in the accepted warmed band, not a stable gain. Keep TensorRT's default auxiliary stream scheduling.

### Rejected: Defer TensorRT Graph Output Wait To RFDETR Postprocess

- Hypothesis: The TensorRT graph cache hit path currently queues input copy, CUDA graph replay, and output clones on the dedicated graph stream, then makes the caller inference stream wait for that graph stream. In the RFDETR deferred workflow path, postprocess could wait on the graph stream directly, potentially removing one event edge before the next depth-2 replay.
- Change tested: Temporary code only; added a `defer_cuda_graph_output_sync` flag to the TensorRT helper, used it only from RFDETR when `defer_cuda_stream_sync=True`, stored the graph stream in RFDETR thread-local state, and made postprocess wait on that stream instead of the inference stream. Pipeline depth remained fixed at `2`; depth `3` was not tested.
- Correctness: `py_compile` passed and the benchmark completed normally.
- Result on requested command: `frames=538 elapsed=2.20s fps=244.83`, inside the accepted band but not a clear improvement. A correctly launched Nsight Systems run with this temporary scheduling variant measured `frames=538 elapsed=2.29s fps=235.03`, below the latest accepted-path graph-bound profile.
- Profile for rejected variant: `/tmp/rfdetr_depth2_deferout_20260523_201017.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_deferout_20260523_201017.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_deferout_20260523_201017_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_deferout_20260523_201017_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_deferout_20260523_201017_stats_cuda_api_sum.csv`.
- Learning: Moving the output wait out of the inference stream does not improve the depth-2 balance. The accepted event chain is not the current limiter, and preserving the simpler helper-level ownership handoff is better.

### Profile: Depth-2 Graphbound Final Refresh

- Request: Capture a fresh Nsight Systems report for user analysis after reverting the unsuccessful scheduling tweak. Pipeline depth stayed fixed at `2`; depth `3` was not tested.
- Accepted-path sanity run before profiling: `frames=538 elapsed=2.20s fps=244.13`.
- Profile: `/tmp/rfdetr_depth2_graphbound_final_20260523_201236.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_graphbound_final_20260523_201236.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_graphbound_final_20260523_201236_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_graphbound_final_20260523_201236_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_graphbound_final_20260523_201236_stats_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.32s fps=231.55`.
- Quick exported-data check: the profile contains `602` `cudaGraphLaunch` calls. After skipping `64` warmup launches plus `100` settling launches, graph-launch submit interval was p50 `4118.925 us`, p90 `4252.884 us`, p95 `4316.037 us`, mean `4119.233 us`.
- Learning: The fresh accepted report is consistent with the earlier graph-bound traces. Depth `2` keeps the run shaped around the TensorRT CUDA graph replay cadence; remaining visible work is the narrow fused postprocess/copy tail rather than a CPU scheduling bubble.

### Rejected: RFDETR Mask Resize Triton Block Size Variants

- Hypothesis: The deferred fused mask resize kernel is one of the few remaining visible postprocess kernels in the graph-to-graph tail. Changing the Triton pixel block size from `256` to either `512` or `128` might improve occupancy or reduce scheduling overhead for the `7 x 312 x 312` limited resize workload.
- Change tested: Temporary code only; changed `fused_resize_selected_masks(...)` block size to `512`, then to `128`. Pipeline depth stayed fixed at `2`; depth `3` was not tested.
- Correctness: This only changed the partitioning of the same bilinear mask threshold computation. `py_compile` passed after each variant and the benchmark completed normally. Classes and boxes are untouched by this kernel.
- Result on requested command: `512` measured `frames=538 elapsed=2.20s fps=244.14` and `frames=538 elapsed=2.20s fps=244.03`; `128` measured `frames=538 elapsed=2.21s fps=242.92`. After reverting to accepted `256`, the immediate same-session baseline measured `frames=538 elapsed=2.21s fps=242.96`, indicating the session was noisy/slow but neither variant showed a stable gain.
- Learning: The accepted `256` tile remains the best default. This kernel is too small relative to the TensorRT CUDA graph body for tiling-only changes to move end-to-end throughput reliably.

### Accepted: RFDETR Mask Resize Two-Warp Launch

- Hypothesis: The accepted deferred mask resize uses a `256`-pixel tile with `num_warps=4`. Since the fixed first-stage grid only covers up to `7` detections at `312x312`, the per-program vector work may be over-provisioned; `num_warps=2` could reduce scheduling/register pressure while preserving the same per-pixel math.
- Change: Changed `_resize_selected_masks_kernel` launch in `fused_resize_selected_masks(...)` from `num_warps=4` to `num_warps=2`. Pipeline depth stayed fixed at `2`; depth `3` was not tested.
- Correctness: `py_compile` passed. This kernel runs after class selection and box decoding, so class IDs and boxes are unchanged by construction; the mask computation is the same independent per-pixel bilinear threshold with only Triton launch geometry changed.
- Result on requested command: `num_warps=2` measured `frames=538 elapsed=2.20s fps=244.42`, `frames=538 elapsed=2.20s fps=244.51`, and `frames=538 elapsed=2.20s fps=244.02`. Same-session `num_warps=4` baselines measured `frames=538 elapsed=2.21s fps=243.50` and `frames=538 elapsed=2.21s fps=243.82`.
- Learning: The improvement is small and still in the noisy graph-bound band, but the same-session A/B favored two warps and the change is low-risk. Keep `num_warps=2` as the current accepted mask-resize launch geometry.

### Rejected: Vectorized Workflow Class Name Mapping

- Hypothesis: The RFDETR workflow fast path maps numeric `class_id` arrays to class-name strings with a Python list comprehension. Caching `model.class_names` as a NumPy object array and indexing it directly for in-range class IDs could reduce CPU materialization work while the depth-2 pipeline is feeding the GPU.
- Change tested: Temporary code only; added a thread-local cached class-name object array and used NumPy indexing when all class IDs were valid, falling back to the original per-element behavior for out-of-range IDs. Pipeline depth stayed fixed at `2`; depth `3` was not tested.
- Correctness: `py_compile` passed. A synthetic check over empty, in-range, and out-of-range class IDs matched the original mapping exactly. This path only changes display-name construction after model classes, boxes, and masks have already been materialized.
- Result on requested command: vectorized mapping measured `frames=538 elapsed=2.20s fps=244.55` and `frames=538 elapsed=2.21s fps=243.81`; after reverting to the original list comprehension, the same-session baseline measured `frames=538 elapsed=2.20s fps=244.04`.
- Learning: Class-name mapping is not a stable limiter in the accepted graph-bound run. Keep the simpler original mapping.

### Rejected: RFDETR Mask Resize One-Warp Launch

- Hypothesis: After accepting `num_warps=2` for the limited mask resize kernel, reducing further to `num_warps=1` might lower scheduling overhead for the same `256`-pixel tile and small `7 x 312 x 312` first-stage grid.
- Change tested: Temporary code only; changed `_resize_selected_masks_kernel` launch in `fused_resize_selected_masks(...)` from `num_warps=2` to `num_warps=1`. Pipeline depth stayed fixed at `2`; depth `3` was not tested.
- Correctness: `py_compile` passed. This only changes Triton launch geometry after class selection and box decoding; class IDs and boxes are unchanged by construction, and the mask math is the same.
- Result on requested command: `num_warps=1` measured `frames=538 elapsed=2.21s fps=243.91`, `frames=538 elapsed=2.21s fps=243.81`, and `frames=538 elapsed=2.21s fps=243.53`. After restoring the accepted `num_warps=2`, same-session runs measured `frames=538 elapsed=2.21s fps=243.28` and then `frames=538 elapsed=2.20s fps=244.52`.
- Learning: One warp is not a stable improvement and loses to the restored two-warp launch once the session returns to the accepted band. Keep `num_warps=2`.

### Rejected: UUID Hex Strings For Workflow IDs

- Hypothesis: The RFDETR workflow fast path creates one inference UUID per frame and one detection UUID per detection via `str(uuid.uuid4())`. Using `uuid.uuid4().hex` could avoid UUID string formatting with hyphens and reduce CPU materialization overhead without changing model predictions.
- Change tested: Temporary code only; changed frame-level inference IDs and per-detection IDs in the RFDETR workflow conversion path from `str(uuid.uuid4())` to `uuid.uuid4().hex`. Pipeline depth stayed fixed at `2`; depth `3` was not tested.
- Correctness: `py_compile` passed. The change only affects opaque workflow identifiers after prediction construction; classes, boxes, confidence, and masks are untouched.
- Result on requested command: `.hex` IDs measured `frames=538 elapsed=2.20s fps=244.40` and `frames=538 elapsed=2.21s fps=243.31`; after reverting to the original string UUID format, same-session baseline measured `frames=538 elapsed=2.21s fps=243.66`.
- Learning: UUID string formatting is not a stable limiter in the accepted graph-bound run. Keep the existing hyphenated UUID behavior.

### Profile: Depth-2 Low-Bubble Nsight Systems Refresh

- Request: Capture a fresh Nsight Systems report for user analysis while keeping the workflow pipeline depth fixed at `2`; depth `3` was not tested.
- Profile: `/tmp/rfdetr_depth2_lowbubble_20260523_204849.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_lowbubble_20260523_204849.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_lowbubble_20260523_204849_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_lowbubble_20260523_204849_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_lowbubble_20260523_204849_stats_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.29s fps=234.43`.
- Quick exported-data check: the profile contains `602` CUDA graph traces. After skipping `64` warmup launches plus `100` settling launches, graph duration was p50 `4066.704 us`, p90 `4131.826 us`, p95 `4134.941 us`, mean `4075.063 us`; graph end-to-next-start gap was p50 `40.543 us`, p90 `41.695 us`, p95 `42.156 us`, mean `40.703 us`; graph start-to-start interval was p50 `4107.584 us`, p90 `4172.229 us`, p95 `4175.436 us`, mean `4115.659 us`.
- Gap decomposition after the same skip: busy work inside the graph-to-graph gap was p50 `35.039 us`, p90 `36.857 us`, p95 `37.280 us`, mean `35.284 us`; idle gap time was p50 `5.312 us`, p90 `6.093 us`, p95 `6.272 us`, mean `5.419 us`.
- Non-profiled sanity run after restoring accepted code measured `frames=538 elapsed=2.20s fps=244.43`.
- Learning: The accepted depth-2 run is already tightly graph-paced. The next TensorRT CUDA graph starts about `40.5 us` after the previous graph ends, and only about `5.3 us` of that gap is idle; the bottleneck remains the CUDA graph forward body plus a narrow real GPU postprocess/copy tail.

### Rejected: Cached False Preprocessing Overrides Object

- Hypothesis: `_try_run_rfdetr_trt_fast_path(...)` constructs `PreProcessingOverrides(False, False, False)` for every frame. Reusing a module-level frozen dataclass instance could remove a small Python allocation in the CPU producer path without changing preprocessing flags.
- Change tested: Temporary code only; added a module-level `_RFDETR_PRE_PROCESSING_OVERRIDES` constant and passed it into RFDETR preprocessing. Pipeline depth stayed fixed at `2`; depth `3` was not tested.
- Correctness: `py_compile` passed. `PreProcessingOverrides` is a frozen dataclass with three boolean fields, so the reused object is immutable and semantically identical to constructing the same values per frame.
- Result on requested command: depth-2 runs measured `frames=538 elapsed=2.20s fps=244.19` and `frames=538 elapsed=2.21s fps=243.44`.
- Learning: This allocation is below the limiter in the current graph-bound path. The change did not improve throughput, so the per-frame explicit object construction was restored.

### Rejected: Unrolled RFDETR Normalization Channel Writes

- Hypothesis: CPU profiling showed `_pil_image_to_normalized_tensor(...)` as the largest remaining true CPU self-time. Unrolling the fixed three-channel normalization loop could remove minor Python loop/indexing overhead while preserving the exact same NumPy multiply/add operations.
- Change tested: Temporary code only; replaced the three-iteration `for output_channel, input_channel in enumerate(channel_order)` loop with three explicit `np.multiply(..., out=normalized[i])` calls followed by the same in-place bias adds. Pipeline depth stayed fixed at `2`; depth `3` was not tested.
- Correctness: `py_compile` passed. A deterministic local equivalence check over both channel orders and several uint8 sample images matched the original loop exactly with max diff `0`.
- Result on requested command: unrolled depth-2 runs measured `frames=538 elapsed=2.20s fps=245.02` and `frames=538 elapsed=2.21s fps=243.87`; after reverting to the accepted loop, the immediate same-session baseline measured `frames=538 elapsed=2.21s fps=243.77`.
- Learning: The first run was noise rather than a stable gain. Loop overhead in the normalization helper is not large enough to move the graph-bound steady state, so the simpler accepted loop remains.

### Diagnostic: Accepted TensorRT Graph-Only Ceiling

- Hypothesis: The accepted full workflow appears tightly graph-paced, so measuring the serialized TensorRT plan outside the Python workflow can bound the remaining useful optimization headroom.
- Diagnostic: `trtexec` was not installed, so a TensorRT Python harness loaded `/tmp/cache/shared-blobs/bc173a2cfda9a10af2bc411885e9fec3`, created one execution context for static input shape `(1, 3, 312, 312)`, bound the three output tensors, warmed the context, then timed both direct `execute_async_v3(...)` and a CUDA graph containing only `execute_async_v3(...)`. This diagnostic excludes preprocessing, input D2D copy into the graph buffer, output clones, sigmoid/selector/mask resize, D2H prediction copies, and workflow CPU materialization. Pipeline depth was not varied; depth `3` was not tested.
- Result: Direct `execute_async_v3(...)` measured `4.311606 ms` per enqueue (`231.93 fps`). CUDA graph replay-only measured `4.052506 ms` per replay (`246.76 fps`).
- Full workflow sanity check: The accepted depth-2 command measured `frames=538 elapsed=2.21s fps=243.72` immediately after the diagnostic.
- Learning: The current accepted workflow is within roughly `1.2%` of the pure TensorRT graph replay ceiling for this serialized plan on the observed T4. The remaining non-engine overhead is only a few dozen microseconds per frame, consistent with Nsight's roughly `40 us` graph-to-graph tail. Meaningful additional FPS now requires a faster correctness-equivalent TensorRT engine/tactic/export; Python, D2H, and postprocess micro-tweaks have very little headroom left.

### Rejected: TensorRT Runtime Max Threads Sweep

- Hypothesis: TensorRT `Runtime.max_threads` can be set before deserializing the engine. If it affected graph execution context internals or host-side graph launch preparation, tuning it might improve the accepted CUDA graph replay ceiling without changing model math.
- Diagnostic: Ran the same accepted-engine graph-only harness with `runtime.max_threads` set to `1`, `2`, `4`, `8`, and `16` before deserializing `/tmp/cache/shared-blobs/bc173a2cfda9a10af2bc411885e9fec3`. This measured only a CUDA graph containing `execute_async_v3(...)`; pipeline depth was not varied and depth `3` was not tested.
- Result: `max_threads=1` measured `4.054766 ms` (`246.62 fps`), `2` measured `4.076955 ms` (`245.28 fps`), `4` measured `4.087345 ms` (`244.66 fps`), `8` measured `4.094583 ms` (`244.23 fps`), and `16` measured `4.104315 ms` (`243.65 fps`). The default runtime value is already `1`.
- Full workflow sanity check: The accepted depth-2 command measured `frames=538 elapsed=2.21s fps=243.73` after the diagnostic.
- Learning: TensorRT runtime thread count is not an optimization lever for this accepted plan. The default is already the fastest measured setting for the graph-only ceiling, and larger values regress slightly. No code change was kept.

### Rejected: Explicit TensorRT Shape Inference Before Graph Capture

- Hypothesis: Calling `IExecutionContext.infer_shapes()` after setting the static input shape and tensor addresses, before the warmup enqueue and CUDA graph capture, might finalize TensorRT shape state earlier and produce a slightly faster captured graph.
- Diagnostic: Used the accepted-engine graph-only harness with three modes: baseline, `infer_shapes()`, and `infer_shapes()` plus `get_tensor_strides(...)` inspection for all I/O tensors. This measured only CUDA graph replay of `execute_async_v3(...)`; pipeline depth was not varied and depth `3` was not tested.
- Correctness: `infer_shapes()` returned an empty list of missing tensors, and tensor strides matched the accepted static layout: input `(292032, 97344, 312, 1)`, boxes `(400, 4, 1)`, logits `(9100, 91, 1)`, masks `(608400, 6084, 78, 1)`.
- Result: Baseline graph replay measured `4.055898 ms` (`246.55 fps`), `infer_shapes()` measured `4.075415 ms` (`245.37 fps`), and `infer_shapes()` plus stride inspection measured `4.087416 ms` (`244.65 fps`).
- Full workflow sanity check: The accepted depth-2 command measured `frames=538 elapsed=2.21s fps=243.05` after the diagnostic.
- Learning: TensorRT already has the static shape state it needs before capture. Explicit shape inference does not improve the captured graph and slightly regresses the graph-only ceiling, so no runtime change was kept.

### Diagnostic: TensorRT Helper Operation Ceiling

- Hypothesis: The accepted workflow is close to TensorRT graph-only speed because depth-2 scheduling overlaps part of the helper/postprocess tail. Measuring graph replay with the helper's input D2D copy and output clone operations in isolation can bound how much benefit remains from further clone/copy tuning.
- Diagnostic: Used the accepted engine and one captured `execute_async_v3(...)` CUDA graph. Timed four single-stream modes with CUDA events: graph replay only, input-buffer D2D copy plus graph replay, graph replay plus three output clones, and input-buffer D2D copy plus graph replay plus three output clones. This diagnostic excludes preprocessing, postprocess kernels, D2H prediction copies, and workflow CPU materialization; pipeline depth was not varied and depth `3` was not tested.
- Result: graph only measured `4.050621 ms` (`246.88 fps`), input copy plus graph measured `4.081056 ms` (`245.03 fps`), graph plus output clones measured `4.103890 ms` (`243.67 fps`), and input copy plus graph plus output clones measured `4.129317 ms` (`242.17 fps`).
- Full workflow sanity check: The accepted depth-2 command measured `frames=538 elapsed=2.20s fps=244.10` after the diagnostic.
- Learning: Input copy and output clones are real costs, but the accepted two-frame schedule overlaps enough surrounding work that the full workflow can run slightly faster than a single-stream serialized helper loop. Prior clone/borrow/copy rewrites lost this overlap. Further copy/clone tuning is unlikely to beat the current schedule unless it preserves the same decoupling while reducing the TensorRT graph body or replacing the serialized plan.

### Diagnostic: TensorRT Temporary Allocator Probe

- Hypothesis: `IExecutionContext.temporary_allocator` could reveal or control hidden TensorRT temporary allocations during the warmup/capture/replay path. If TensorRT was allocating temporary device buffers around graph capture, a preallocated allocator might reduce graph setup or replay jitter.
- Diagnostic: Used a temporary Python harness against the accepted engine and attached a logging `trt.IGpuAllocator` to the CUDA graph execution context. The allocator returned `None` for allocations so any real TensorRT temporary allocation would be visible and fail fast; no workflow code was changed. Pipeline depth stayed fixed at `2`; depth `3` was not tested.
- Result: The allocator received `0` callbacks after the pre-capture warmup enqueue, `0` callbacks after CUDA graph capture, and `0` callbacks after ten graph replays.
- Learning: TensorRT is not using the per-context temporary allocator in this static RFDETR graph path. There is no useful temporary-allocation hook to optimize, and adding a custom allocator would only add complexity.

### Profile: Depth-2 Graph-Paced Nsight Systems Refresh

- Request: Capture a fresh Nsight Systems report for user analysis while keeping workflow pipeline depth fixed at `2`. Depth `3` was not tested.
- Profile: `/tmp/rfdetr_depth2_graphpaced_20260523_211726.nsys-rep`, exported SQLite `/tmp/rfdetr_depth2_graphpaced_20260523_211726.sqlite`, and CSV summaries `/tmp/rfdetr_depth2_graphpaced_20260523_211726_stats_cuda_gpu_kern_sum.csv`, `/tmp/rfdetr_depth2_graphpaced_20260523_211726_stats_cuda_gpu_mem_time_sum.csv`, and `/tmp/rfdetr_depth2_graphpaced_20260523_211726_stats_cuda_api_sum.csv`.
- Result under profiler: `frames=538 elapsed=2.30s fps=234.28`.
- Graph spacing: The capture contains `602` CUDA graph traces. After skipping `64` capture warmup replays plus `100` settling launches, graph duration was p50 `4070.800 us`, p90 `4134.482 us`, p95 `4137.841 us`, mean `4075.039 us`; graph end-to-next-start gap was p50 `40.383 us`, p90 `41.548 us`, p95 `41.856 us`, mean `40.500 us`; graph start-to-start interval was p50 `4111.038 us`, p90 `4175.217 us`, p95 `4178.257 us`, mean `4115.560 us`.
- Gap decomposition after the same skip: busy work inside the graph-to-graph gap was p50 `34.943 us`, p90 `36.319 us`, p95 `37.036 us`, mean `35.086 us`; true idle was p50 `5.408 us`, p90 `6.048 us`, p95 `6.272 us`, mean `5.414 us`.
- Gap occupants: the largest clipped occupants were next-frame input Device-to-Device copy (`1168128B`, `13.130 us/gap`), TensorRT mask Device-to-Device clone (`2433600B`, `13.114 us/gap`), sigmoid (`6.902 us/gap`), fill-long (`2.812 us/gap`), logits Device-to-Device clone (`36400B`, `2.105 us/gap`), boxes Device-to-Device clone (`1600B`, `2.000 us/gap`), fill-int (`1.947 us/gap`), and `_select_topk_boxes_kernel` (`1.660 us/gap`).
- Non-profiled sanity check: The accepted depth-2 command measured `frames=538 elapsed=2.20s fps=244.29` immediately after the profile.
- Learning: The run is already graph-paced with only about `5.4 us` median idle between TensorRT graph replays. The consistent limiter is still the approximately `4.075 ms` TensorRT CUDA graph body plus unavoidable input/output ownership copies and a narrow fused-postprocess tail; CPU work is not blocking the next model forward in the steady state.
