"""CLI entry points for rv-segment, rv-pose, rv-pipeline."""

from __future__ import annotations

from pathlib import Path

import tyro


def segment_main(
    config: Path = Path("configs/grounded_sam2.yaml"),
    prompt: str = "",
    image: Path = Path("data/rgbd/demo/frame_000000/color.png"),
    output: Path = Path("data/masks/mask.png"),
) -> None:
    """Run Grounded SAM 2 on a single image."""
    from robotvision.segmentation.grounded_sam2 import GroundedSAM2
    from robotvision.utils.io import load_config

    import cv2

    cfg = load_config(config)
    segmenter = GroundedSAM2(cfg)
    text = prompt or cfg.get("default_prompt", "object")
    bgr = cv2.imread(str(image))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = segmenter.segment_image(rgb, text)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), (result.mask * 255).astype("uint8"))


def pose_main(
    config: Path = Path("configs/foundation_pose.yaml"),
    frame: Path = Path("data/rgbd/demo/frame_000000"),
    mask: Path = Path("data/masks/mask.png"),
) -> None:
    """Run FoundationPose on a single RGB-D frame + mask."""
    from robotvision.pose.foundation_pose import FoundationPoseEstimator
    from robotvision.utils.io import load_config, load_rgbd_frame, save_pose

    import cv2

    cfg = load_config(config)
    estimator = FoundationPoseEstimator(cfg)
    color, depth = load_rgbd_frame(frame)
    mask_arr = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    result = estimator.register(color, depth, mask_arr)
    out = Path(cfg.get("output_dir", "data/outputs")) / "pose"
    save_pose(result.object_in_camera, out)


def pipeline_main(
    config: Path = Path("configs/pipeline.yaml"),
) -> None:
    """Run the full Grounded SAM 2 → FoundationPose pipeline."""
    from robotvision.pipeline.combo import ComboPipeline

    pipeline = ComboPipeline.from_yaml(config)
    rgbd_dir = Path(pipeline.config["rgbd_dir"])
    mesh_path = Path(pipeline.config["mesh_path"])
    output_dir = Path(pipeline.config["output_dir"])
    prompt = pipeline.config["prompt"]
    pipeline.run_sequence(rgbd_dir, prompt, mesh_path, output_dir)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "segment": segment_main,
            "pose": pose_main,
            "pipeline": pipeline_main,
        }
    )
