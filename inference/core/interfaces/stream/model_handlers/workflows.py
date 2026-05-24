from typing import Callable, Dict, List, Optional

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


class WorkflowRunner:
    def __init__(self) -> None:
        self._fast_path_cache: Dict[
            int, Optional[Callable[[List[VideoFrame]], List[dict]]]
        ] = {}

    def run_workflow(
        self,
        video_frames: List[VideoFrame],
        workflows_parameters: Optional[dict],
        execution_engine: ExecutionEngine,
        image_input_name: str,
        video_metadata_input_name: str,
        serialize_results: bool = False,
        _is_preview: bool = False,
    ) -> List[dict]:
        if not workflows_parameters and not serialize_results and not _is_preview:
            fast_path = self._get_single_step_fast_path(
                execution_engine=execution_engine,
                image_input_name=image_input_name,
            )
            if fast_path is not None:
                return fast_path(video_frames)
        if workflows_parameters is None:
            workflows_parameters = {}
        # TODO: pass fps reflecting each stream to workflows_parameters
        fps = video_frames[0].fps
        if video_frames[0].measured_fps:
            fps = video_frames[0].measured_fps
        if fps is None:
            # for FPS reporting we expect 0 when FPS cannot be determined
            fps = 0
        video_metadata_for_images = [
            VideoMetadata(
                video_identifier=(
                    str(video_frame.source_id)
                    if video_frame.source_id
                    else "default_source"
                ),
                frame_number=video_frame.frame_id,
                frame_timestamp=video_frame.frame_timestamp,
                fps=video_frame.fps,
                measured_fps=video_frame.measured_fps,
                comes_from_video_file=video_frame.comes_from_video_file,
            )
            for video_frame in video_frames
        ]
        workflows_parameters[image_input_name] = [
            {
                "type": "numpy_object",
                "value": video_frame.image,
                "video_metadata": video_metadata,
            }
            for video_frame, video_metadata in zip(
                video_frames, video_metadata_for_images
            )
        ]
        workflows_parameters[video_metadata_input_name] = video_metadata_for_images
        return execution_engine.run(
            runtime_parameters=workflows_parameters,
            fps=fps,
            serialize_results=serialize_results,
            _is_preview=_is_preview,
        )

    def _get_single_step_fast_path(
        self,
        execution_engine: ExecutionEngine,
        image_input_name: str,
    ) -> Optional[Callable[[List[VideoFrame]], List[dict]]]:
        cache_key = id(execution_engine)
        if cache_key not in self._fast_path_cache:
            self._fast_path_cache[cache_key] = self._build_single_step_fast_path(
                execution_engine=execution_engine,
                image_input_name=image_input_name,
            )
        return self._fast_path_cache[cache_key]

    @staticmethod
    def _build_single_step_fast_path(
        execution_engine: ExecutionEngine,
        image_input_name: str,
    ) -> Optional[Callable[[List[VideoFrame]], List[dict]]]:
        inner_engine = getattr(execution_engine, "_engine", None)
        compiled_workflow = getattr(inner_engine, "_compiled_workflow", None)
        if compiled_workflow is None:
            return None
        if compiled_workflow.input_substitutions:
            return None
        if len(compiled_workflow.steps) != 1:
            return None
        if len(compiled_workflow.workflow_definition.inputs) != 1:
            return None
        if len(compiled_workflow.workflow_definition.outputs) != 1:
            return None
        step_name, initialised_step = next(iter(compiled_workflow.steps.items()))
        manifest = initialised_step.manifest
        if (
            getattr(manifest, "type", None)
            != "roboflow_core/roboflow_instance_segmentation_model@v3"
        ):
            return None
        if getattr(manifest, "images", None) != f"$inputs.{image_input_name}":
            return None
        output = compiled_workflow.workflow_definition.outputs[0]
        if (
            output.name != "predictions"
            or output.selector != f"$steps.{step_name}.predictions"
        ):
            return None
        step = initialised_step.step

        def run_single_step_workflow(video_frames: List[VideoFrame]) -> List[dict]:
            workflow_images = []
            for idx, video_frame in enumerate(video_frames):
                video_metadata = VideoMetadata(
                    video_identifier=(
                        str(video_frame.source_id)
                        if video_frame.source_id
                        else "default_source"
                    ),
                    frame_number=video_frame.frame_id,
                    frame_timestamp=video_frame.frame_timestamp,
                    fps=video_frame.fps,
                    measured_fps=video_frame.measured_fps,
                    comes_from_video_file=video_frame.comes_from_video_file,
                )
                parent_id = f"{image_input_name}.[{idx}]"
                parent_metadata = ImageParentMetadata(parent_id=parent_id)
                workflow_images.append(
                    WorkflowImageData(
                        parent_metadata=parent_metadata,
                        workflow_root_ancestor_metadata=parent_metadata,
                        numpy_image=video_frame.image,
                        video_metadata=video_metadata,
                    )
                )
            step_results = step.run(
                images=workflow_images,
                model_id=manifest.model_id,
                confidence_mode=manifest.confidence_mode,
                custom_confidence=manifest.custom_confidence,
                class_agnostic_nms=manifest.class_agnostic_nms,
                class_filter=manifest.class_filter,
                iou_threshold=manifest.iou_threshold,
                max_detections=manifest.max_detections,
                max_candidates=manifest.max_candidates,
                mask_decode_mode=manifest.mask_decode_mode,
                tradeoff_factor=manifest.tradeoff_factor,
                disable_active_learning=manifest.disable_active_learning,
                active_learning_target_dataset=manifest.active_learning_target_dataset,
            )
            return [{"predictions": result["predictions"]} for result in step_results]

        return run_single_step_workflow
