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
