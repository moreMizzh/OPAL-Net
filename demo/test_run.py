import hashlib
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from demo.run import run_case
from model_seg import AtlasGuidedSegModel


def _write_image(path: Path, values: np.ndarray, reference: sitk.Image | None = None) -> None:
    image = sitk.GetImageFromArray(values)
    if reference is not None:
        image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))


@pytest.fixture
def case(tmp_path):
    shape = (16, 16, 16)
    target = sitk.GetImageFromArray(np.random.default_rng(7).random(shape).astype(np.float32))
    target.SetSpacing((1.2, 1.3, 1.4))
    target.SetOrigin((10.0, 20.0, 30.0))
    image_path = tmp_path / "image.nii.gz"
    sitk.WriteImage(target, str(image_path))

    atlas_image = tmp_path / "atlas_image.nii.gz"
    atlas_label = tmp_path / "atlas_label.nii.gz"
    _write_image(atlas_image, np.ones(shape, dtype=np.float32))
    _write_image(atlas_label, np.ones(shape, dtype=np.uint8))

    architecture = {
        "in_channels": 1,
        "out_channels": 2,
        "base_channels": 4,
        "dropout_prob": 0.0,
        "atlas_label_embed_channels": 2,
        "context_query_tokens": 2,
        "context_shuffle_factor": 2,
        "atlas_task_dropout": 0.0,
        "attn_heads": 4,
        "attn_dropout": 0.0,
        "attn_gate_init": -4.0,
        "enable_atlas_token_path": True,
        "share_atlas_encoder_with_seg": False,
        "bottleneck_ctx_residual_weights": [1.0, 1.0],
    }
    checkpoint = tmp_path / "model.pth"
    torch.save({"model_state_dict": AtlasGuidedSegModel(**architecture).state_dict()}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    spec = {"architecture": architecture, "sha256": digest}
    return image_path, atlas_image, atlas_label, checkpoint, spec, tmp_path / "results"


def test_single_case_writes_prediction_with_input_geometry(case):
    image_path, atlas_image, atlas_label, checkpoint, spec, output_dir = case

    run_case("bcv", image_path, atlas_image, atlas_label, checkpoint, output_dir, spec, "cpu")

    prediction = sitk.ReadImage(str(output_dir / "prediction.nii.gz"))
    source = sitk.ReadImage(str(image_path))
    assert prediction.GetSize() == source.GetSize()
    assert prediction.GetSpacing() == source.GetSpacing()
    assert prediction.GetOrigin() == source.GetOrigin()
    assert prediction.GetDirection() == source.GetDirection()
    assert (output_dir / "overlay.png").stat().st_size > 0


def test_checkpoint_digest_mismatch_fails_before_outputs(case):
    image_path, atlas_image, atlas_label, checkpoint, spec, output_dir = case
    spec["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256"):
        run_case("bcv", image_path, atlas_image, atlas_label, checkpoint, output_dir, spec, "cpu")

    assert not output_dir.exists()
