"""Run the bundled BCV segmentation example without command-line arguments."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import torch

if __package__:
    from .run import run_case
else:
    from run import run_case


DEMO_DIR = Path(__file__).resolve().parent


def main(demo_dir: Path = DEMO_DIR, device: str | None = None) -> tuple[Path, Path]:
    demo_dir = Path(demo_dir)
    bcv_data = demo_dir / "data" / "bcv"
    # The manifest fixes the network settings and verifies the checkpoint hash.
    with (demo_dir / "models.json").open(encoding="utf-8") as config:
        spec = json.load(config)["bcv"]

    # Case 61 is the target; atlas 27 supplies the prior image and its labels.
    image_path = bcv_data / "case.nii.gz"
    atlas_image_path = bcv_data / "atlas_image.nii.gz"
    atlas_label_path = bcv_data / "atlas_label.nii.gz"
    checkpoint_path = demo_dir / "checkpoints" / "bcv.pth"

    # Keep each run's prediction and preview instead of overwriting earlier results.
    output_dir = demo_dir / "results" / f"bcv-{uuid4().hex}"
    # Prefer CUDA when available, but allow the bundled case to run on CPU.
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    prediction, preview = run_case(
        "bcv",
        image_path,
        atlas_image_path,
        atlas_label_path,
        checkpoint_path,
        output_dir,
        spec,
        selected_device,
    )
    print(f"Prediction: {prediction}")
    print(f"Preview: {preview}")
    return prediction, preview


if __name__ == "__main__":
    main()
