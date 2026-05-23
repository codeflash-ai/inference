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
