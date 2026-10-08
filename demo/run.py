"""Run one OPAL-Net segmentation case without a ground-truth label."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch

DEMO_DIR = Path(__file__).resolve().parent
REPO_DIR = DEMO_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from inference_seg import get_atlas_batch  # noqa: E402
from model_seg import AtlasGuidedSegModel  # noqa: E402


def _read_volume(path: Path) -> tuple[sitk.Image, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite 3D NIfTI volume: {path}")
    return image, array


def _verify_checkpoint(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for block in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")


def _write_preview(image: np.ndarray, labels: np.ndarray, path: Path, num_classes: int) -> None:
    foreground = np.count_nonzero(labels, axis=(1, 2))
    section = int(np.argmax(foreground)) if foreground.any() else labels.shape[0] // 2
    image_slice = image[section]
    low, high = np.percentile(image_slice, (1, 99))
    if high <= low:
        high = low + 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    for ax in axes:
        ax.imshow(image_slice, cmap="gray", vmin=low, vmax=high)
        ax.axis("off")
    axes[0].set_title("Input")
    axes[1].imshow(
        np.ma.masked_where(labels[section] == 0, labels[section]),
        cmap="turbo",
        vmin=1,
        vmax=max(2, num_classes - 1),
        alpha=0.55,
        interpolation="nearest",
    )
    axes[1].set_title("Prediction overlay")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_case(
    dataset: str,
    image_path: Path,
    atlas_image_path: Path,
    atlas_label_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    spec: dict,
    device: str,
) -> tuple[Path, Path]:
    dataset = dataset.lower()
    if dataset not in {"bcv", "oasis", "adni"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    image_path = Path(image_path)
    atlas_image_path = Path(atlas_image_path)
    atlas_label_path = Path(atlas_label_path)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    prediction_path = output_dir / "prediction.nii.gz"
    preview_path = output_dir / "overlay.png"
    if prediction_path.exists() or preview_path.exists():
        raise FileExistsError(f"Demo outputs already exist in {output_dir}")

    _verify_checkpoint(checkpoint_path, spec["sha256"])
    source_image, source = _read_volume(image_path)
    _, atlas_image = _read_volume(atlas_image_path)
    _, atlas_label = _read_volume(atlas_label_path)
    if atlas_image.shape != atlas_label.shape:
        raise ValueError("Atlas image and label shapes must match")
    if min(source.shape) < 8 or any(dim % 8 for dim in source.shape):
        raise ValueError("Input shape must be divisible by 8 in every dimension")
    if not np.array_equal(atlas_label, atlas_label.astype(np.int64)):
        raise ValueError("Atlas labels must be integers")

    architecture = spec["architecture"]
    num_classes = int(architecture["out_channels"])
    if atlas_label.min() < 0 or atlas_label.max() >= num_classes:
        raise ValueError(f"Atlas labels must be in [0, {num_classes - 1}]")
    if dataset == "oasis":
        source = np.maximum(source, 0)
        atlas_image = np.maximum(atlas_image, 0)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = AtlasGuidedSegModel(**architecture)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()

    target = torch.from_numpy(np.ascontiguousarray(source, dtype=np.float32))[None, None].to(device)
    atlas_image_cpu = torch.from_numpy(np.ascontiguousarray(atlas_image, dtype=np.float32))[None, None]
    atlas_label_cpu = torch.from_numpy(np.ascontiguousarray(atlas_label, dtype=np.float32))[None, None]
    with torch.inference_mode():
        atlas_image_batch, atlas_label_batch = get_atlas_batch(
            atlas_image_cpu, atlas_label_cpu, tuple(target.shape[2:]), device
        )
        output = model(
            x_real=target,
            atlas_img=atlas_image_batch,
            atlas_label=atlas_label_batch,
            is_training=False,
        )
        prediction = output["real_logits"].argmax(dim=1)[0].byte().cpu().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_image = sitk.GetImageFromArray(prediction)
    prediction_image.CopyInformation(source_image)
    sitk.WriteImage(prediction_image, str(prediction_path))
    _write_preview(source, prediction, preview_path, num_classes)
    return prediction_path, preview_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bcv", "oasis", "adni"), required=True)
    parser.add_argument("--image", type=Path, required=True, help="Preprocessed target NIfTI image")
    parser.add_argument("--atlas-image", type=Path, required=True)
    parser.add_argument("--atlas-label", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Overrides demo/checkpoints/<dataset>.pth")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the prediction and preview")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    args = parser.parse_args()

    with (DEMO_DIR / "models.json").open(encoding="utf-8") as f:
        spec = json.load(f)[args.dataset]
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    checkpoint = args.checkpoint or DEMO_DIR / "checkpoints" / f"{args.dataset}.pth"
    prediction, preview = run_case(
        args.dataset,
        args.image,
        args.atlas_image,
        args.atlas_label,
        checkpoint,
        args.output,
        spec,
        device,
    )
    print(f"Prediction: {prediction}")
    print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
