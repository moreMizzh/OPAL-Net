"""Run atlas-guided segmentation inference and compute evaluation metrics.

The script supports Dice, HD95, prediction-label export, and atlas-token
caching for repeated inference runs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from monai.metrics import HausdorffDistanceMetric
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
import sys
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import (  # noqa: E402
    ADNI_Split_Dataset,
    BCV_Dataset,
    OASIS_Dataset,
)
from model_seg import AtlasGuidedSegModel  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent


def _list_nifti_files(dir_path: Path) -> list[Path]:
    return sorted(p for p in dir_path.iterdir() if p.is_file() and p.name.endswith(".nii.gz"))




@dataclass(frozen=True)
class Config:
    """


    - `model_dir`  checkpoint
    - `export_train/val/test`  split
    - `save_pred_label*`  Dice
    - `hd95_mode / hd95_use_saved_pred / force_rebuild_atlas_token_cache`
       HD95  atlas token cache
    """
    model_dir: str = "./output/Seg/OASIS_seg_cascaded_reg"
    checkpoint_name: str = "best_dice"

    export_train: bool = False          #  train
    export_val: bool = False            #  val
    export_test: bool = True            #  test
    compute_hd95: bool = True              #  HD95
    hd95_mode: str = "monai_mean"           # "per_class_mean"  # HD95  MONAI
    save_pred_label: bool = True           #
    save_pred_label_train: bool = False      #  train
    save_pred_label_val: bool = False       #  val
    save_pred_label_test: bool = True      #  test
    hd95_use_saved_pred: bool = True        # True=False=
    atlas_token_cache_name: str = "inference_seg_atlas_token_cache.pt"  # atlas prior token
    force_rebuild_atlas_token_cache: bool = False  # True= cache atlas prior token

    downsample_factor: int = 1      # 1
    batch_size: int = 1             #  case  1
    num_workers: int = 0            #  DataLoader  worker
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  #
    wait_cuda_memory_release: bool = True       #  case  CUDA
    cuda_memory_wait_timeout_sec: float = 60.0  #  CUDA
    cuda_memory_wait_poll_sec: float = 0.1      #  CUDA
    cuda_memory_wait_tolerance_mb: float = 64.0  #

    dataset_key: str | None = None
    real_root_dir: str | None = None
    num_classes: int | None = None
    base_channels: int | None = None
    in_channels: int | None = None
    student_dropout: float | None = None
    atlas_label_embed_channels: int | None = None
    context_query_tokens: int | None = None
    context_shuffle_factor: int | None = None
    atlas_task_dropout: float | None = None
    attn_heads: int | None = None
    attn_dropout: float | None = None
    attn_gate_init: float | None = None
    enable_atlas_token_path: bool | None = None
    share_atlas_encoder_with_seg: bool | None = None
    bcv_atlas_id: int | None = None
    bottleneck_ctx_residual_weights: tuple[float, float] | None = None
    disable_atlas_token_path_infer: bool = False

    @classmethod
    def from_env(cls):
        """ `OPAL_SEG_INF_*` """

        cfg = cls()

        def _parse_float_tuple(raw: str) -> tuple[float, ...]:
            if raw.strip() == "":
                return ()
            return tuple(float(part.strip()) for part in raw.split(",") if part.strip() != "")

        env_map = {
            "OPAL_SEG_INF_MODEL_DIR": ("model_dir", str),
            "OPAL_SEG_INF_CHECKPOINT": ("checkpoint_name", str),
            "OPAL_SEG_INF_EXPORT_TRAIN": ("export_train", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_EXPORT_VAL": ("export_val", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_EXPORT_TEST": ("export_test", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_COMPUTE_HD95": ("compute_hd95", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_HD95_MODE": ("hd95_mode", str),
            "OPAL_SEG_INF_SAVE_PRED_LABEL": ("save_pred_label", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_SAVE_PRED_LABEL_TRAIN": ("save_pred_label_train", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_SAVE_PRED_LABEL_VAL": ("save_pred_label_val", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_SAVE_PRED_LABEL_TEST": ("save_pred_label_test", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_HD95_USE_SAVED_PRED": ("hd95_use_saved_pred", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_ATLAS_TOKEN_CACHE_NAME": ("atlas_token_cache_name", str),
            "OPAL_SEG_INF_FORCE_REBUILD_ATLAS_TOKEN_CACHE": ("force_rebuild_atlas_token_cache", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_DOWNSAMPLE_FACTOR": ("downsample_factor", int),
            "OPAL_SEG_INF_BATCH_SIZE": ("batch_size", int),
            "OPAL_SEG_INF_NUM_WORKERS": ("num_workers", int),
            "OPAL_SEG_INF_DEVICE": ("device", str),
            "OPAL_SEG_INF_WAIT_CUDA_MEMORY_RELEASE": ("wait_cuda_memory_release", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_CUDA_MEMORY_WAIT_TIMEOUT_SEC": ("cuda_memory_wait_timeout_sec", float),
            "OPAL_SEG_INF_CUDA_MEMORY_WAIT_POLL_SEC": ("cuda_memory_wait_poll_sec", float),
            "OPAL_SEG_INF_CUDA_MEMORY_WAIT_TOLERANCE_MB": ("cuda_memory_wait_tolerance_mb", float),
            "OPAL_SEG_INF_REAL_ROOT_DIR": ("real_root_dir", str),
            "OPAL_SEG_INF_NUM_CLASSES": ("num_classes", int),
            "OPAL_SEG_INF_BASE_CHANNELS": ("base_channels", int),
            "OPAL_SEG_INF_IN_CHANNELS": ("in_channels", int),
            "OPAL_SEG_INF_STUDENT_DROPOUT": ("student_dropout", float),
            "OPAL_SEG_INF_ATLAS_LABEL_EMBED_CHANNELS": ("atlas_label_embed_channels", int),
            "OPAL_SEG_INF_CONTEXT_QUERY_TOKENS": ("context_query_tokens", int),
            "OPAL_SEG_INF_CONTEXT_SHUFFLE_FACTOR": ("context_shuffle_factor", int),
            "OPAL_SEG_INF_ATLAS_TASK_DROPOUT": ("atlas_task_dropout", float),
            "OPAL_SEG_INF_ATTN_HEADS": ("attn_heads", int),
            "OPAL_SEG_INF_ATTN_DROPOUT": ("attn_dropout", float),
            "OPAL_SEG_INF_ATTN_GATE_INIT": ("attn_gate_init", float),
            "OPAL_SEG_INF_ENABLE_ATLAS_TOKEN_PATH": ("enable_atlas_token_path", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_SHARE_ATLAS_ENCODER_WITH_SEG": ("share_atlas_encoder_with_seg", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_INF_BCV_ATLAS_ID": ("bcv_atlas_id", int),
            "OPAL_SEG_INF_BOTTLENECK_CTX_RESIDUAL_WEIGHTS": ("bottleneck_ctx_residual_weights", _parse_float_tuple),
            "OPAL_SEG_INF_DISABLE_ATLAS_TOKEN_PATH_INFER": ("disable_atlas_token_path_infer", lambda v: v.lower() in {"1", "true", "yes", "on"}),
        }
        for env_name, (field_name, cast_fn) in env_map.items():
            raw_val = os.environ.get(env_name)
            if raw_val is not None and raw_val != "":
                object.__setattr__(cfg, field_name, cast_fn(raw_val))
        return cfg

    def resolve_from_training_config(self) -> None:
        """

         seg
         `base_channels / context_query_tokens / share_atlas_encoder_with_seg`
        checkpoint
        """

        cfg_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(cfg_path):
            missing = [
                name
                for name in (
                    "dataset_key",
                    "real_root_dir",
                    "num_classes",
                    "base_channels",
                    "in_channels",
                    "student_dropout",
                    "atlas_label_embed_channels",
                    "context_query_tokens",
                    "context_shuffle_factor",
                    "bottleneck_ctx_residual_weights",
                    "atlas_task_dropout",
                    "attn_heads",
                    "attn_dropout",
                    "attn_gate_init",
                )
                if getattr(self, name) is None
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing training config: {cfg_path}. Please provide {', '.join(missing)} explicitly."
                )
            return

        with open(cfg_path, "r", encoding="utf-8") as f:
            train_cfg = json.load(f)

        mapping = {
            "dataset_key": ("dataset_key", str),
            "real_root_dir": ("real_root_dir", str),
            "num_classes": ("num_classes", int),
            "base_channels": ("base_channels", int),
            "in_channels": ("in_channels", int),
            "student_dropout": ("student_dropout", float),
            "atlas_label_embed_channels": ("atlas_label_embed_channels", int),
            "context_query_tokens": ("context_query_tokens", int),
            "context_shuffle_factor": ("context_shuffle_factor", int),
            "bottleneck_ctx_residual_weights": ("bottleneck_ctx_residual_weights", tuple),
            "atlas_task_dropout": ("atlas_task_dropout", float),
            "attn_heads": ("attn_heads", int),
            "attn_dropout": ("attn_dropout", float),
            "attn_gate_init": ("attn_gate_init", float),
            "enable_atlas_token_path": ("enable_atlas_token_path", bool),
            "share_atlas_encoder_with_seg": ("share_atlas_encoder_with_seg", bool),
            "bcv_atlas_id": ("bcv_atlas_id", int),
        }
        for field_name, (cfg_name, cast_fn) in mapping.items():
            if getattr(self, field_name) is None:
                if cfg_name not in train_cfg:
                    if cfg_name == "share_atlas_encoder_with_seg":
                        object.__setattr__(self, field_name, True)
                        continue
                    if cfg_name == "bottleneck_ctx_residual_weights":
                        object.__setattr__(self, field_name, (0.0, 1.0))
                        continue
                    if cfg_name == "bcv_atlas_id":
                        object.__setattr__(self, field_name, None)
                        continue
                    raise KeyError(f"{cfg_name} is missing in training config.json")
                if cfg_name == "bcv_atlas_id" and train_cfg[cfg_name] is None:
                    object.__setattr__(self, field_name, None)
                    continue
                object.__setattr__(self, field_name, cast_fn(train_cfg[cfg_name]))

        valid_hd95_modes = {"per_class_mean", "monai_mean"}
        if self.hd95_mode not in valid_hd95_modes:
            raise ValueError(
                f"Unsupported hd95_mode={self.hd95_mode}. "
                f"Expected one of {sorted(valid_hd95_modes)}"
            )
        if self.downsample_factor <= 0:
            raise ValueError("downsample_factor must be positive")

        print(
            "Resolved inference config \n"
            f"dataset_key={self.dataset_key} | real_root_dir={self.real_root_dir} \n"
            f"num_classes={self.num_classes} \n"
            f"base_channels={self.base_channels} \n"
            f"in_channels={self.in_channels} \n"
            f"student_dropout={self.student_dropout} \n"
            f"hd95_mode={self.hd95_mode} \n"
            f"downsample_factor={self.downsample_factor} \n"
            f"context_query_tokens={self.context_query_tokens} \n"
            f"context_shuffle_factor={self.context_shuffle_factor} \n"
            f"bottleneck_ctx_residual_weights={self.bottleneck_ctx_residual_weights} \n"
            f"enable_atlas_token_path={self.enable_atlas_token_path} \n"
            f"share_atlas_encoder_with_seg={self.share_atlas_encoder_with_seg} \n"
            f"disable_atlas_token_path_infer={self.disable_atlas_token_path_infer} \n"
            f"hd95_use_saved_pred={self.hd95_use_saved_pred} \n"
            f"atlas_token_cache_name={self.atlas_token_cache_name} \n"
            f"force_rebuild_atlas_token_cache={self.force_rebuild_atlas_token_cache}"
        )


def build_atlas_cache_from_sample(sample: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    atlas_image_cpu = sample[0].unsqueeze(0).float().contiguous()
    atlas_label_cpu = sample[2].unsqueeze(0).float().contiguous()
    return atlas_image_cpu, atlas_label_cpu


def get_downsampled_shape(shape: tuple[int, int, int], factor: int) -> tuple[int, int, int]:
    if factor <= 1:
        return tuple(int(v) for v in shape)
    return tuple(max(1, int(v) // factor) for v in shape)


def downsample_image_label(
    image: torch.Tensor,
    label: torch.Tensor | None,
    factor: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if factor <= 1:
        return image, label
    target_shape = get_downsampled_shape(tuple(image.shape[2:]), factor)
    image = F.interpolate(image, size=target_shape, mode="trilinear", align_corners=True)
    if label is not None:
        label = F.interpolate(label.float(), size=target_shape, mode="nearest").long()
    return image, label


def get_atlas_batch(
    atlas_image_cpu: torch.Tensor,
    atlas_label_cpu: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = tuple(int(v) for v in spatial_shape)
    if tuple(atlas_image_cpu.shape[2:]) == key:
        atlas_img = atlas_image_cpu
        atlas_label = atlas_label_cpu
    else:
        atlas_img = F.interpolate(atlas_image_cpu, size=key, mode="trilinear", align_corners=True)
        atlas_label = F.interpolate(atlas_label_cpu, size=key, mode="nearest")
    return atlas_img.to(device, non_blocking=True), atlas_label.to(device, non_blocking=True)


def _atlas_shape_key(spatial_shape: tuple[int, int, int]) -> str:
    return "x".join(str(int(v)) for v in spatial_shape)


def atlas_token_cache_path(cfg: Config) -> str:
    """ atlas token """

    return os.path.join(cfg.model_dir, cfg.atlas_token_cache_name)


def build_atlas_token_cache_meta(cfg: Config, ckpt_path: str) -> dict[str, object]:
    """ cache

     checkpoint atlas token  cache
    """

    return {
        "checkpoint_path": os.path.abspath(ckpt_path),
        "checkpoint_mtime": float(os.path.getmtime(ckpt_path)),
        "dataset_key": cfg.dataset_key,
        "downsample_factor": int(cfg.downsample_factor),
        "enable_atlas_token_path": bool(cfg.enable_atlas_token_path),
        "share_atlas_encoder_with_seg": bool(cfg.share_atlas_encoder_with_seg),
        "disable_atlas_token_path_infer": bool(cfg.disable_atlas_token_path_infer),
    }


def load_atlas_token_cache(cache_path: str, expected_meta: dict[str, object]) -> dict[str, object]:
    """ atlas token cache cache"""

    if not os.path.exists(cache_path):
        return {"meta": dict(expected_meta), "tokens": {}}
    payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, dict):
        return {"meta": dict(expected_meta), "tokens": {}}
    cache_meta = payload.get("meta")
    cache_tokens = payload.get("tokens")
    if cache_meta != expected_meta or not isinstance(cache_tokens, dict):
        return {"meta": dict(expected_meta), "tokens": {}}
    normalized_tokens = {}
    for key, value in cache_tokens.items():
        if torch.is_tensor(value):
            normalized_tokens[str(key)] = value.detach().cpu().contiguous()
    return {"meta": dict(expected_meta), "tokens": normalized_tokens}


def save_atlas_token_cache(cache_path: str, cache_state: dict[str, object]) -> None:
    """ atlas prior tokens """

    payload = {
        "meta": dict(cache_state.get("meta", {})),
        "tokens": {
            str(key): value.detach().cpu().contiguous()
            for key, value in dict(cache_state.get("tokens", {})).items()
            if torch.is_tensor(value)
        },
    }
    torch.save(payload, cache_path)


def infer_share_atlas_encoder_from_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    independent_prefixes = (
        "atlas_encoder.enc1.",
        "atlas_encoder.enc2.",
        "atlas_encoder.enc3.",
        "atlas_encoder.bottleneck.",
        "atlas_encoder.pool1.",
        "atlas_encoder.pool2.",
        "atlas_encoder.pool3.",
    )
    return not any(key.startswith(independent_prefixes) for key in state_dict.keys())


def get_or_create_atlas_prior_tokens(
    model,
    atlas_image_cpu: torch.Tensor,
    atlas_label_cpu: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    cfg: Config,
    cache_state: dict[str, object] | None,
    cache_path: str | None,
):
    """ atlas prior tokens


    1.  /  cache
    2.
    3.  cache split/case
    """

    if not cfg.enable_atlas_token_path or cfg.disable_atlas_token_path_infer:
        return None
    if cache_state is None:
        cache_state = {"meta": {}, "tokens": {}}
    tokens_map = cache_state.setdefault("tokens", {})
    key = _atlas_shape_key(spatial_shape)
    cached = tokens_map.get(key)
    if torch.is_tensor(cached):
        return cached.to(cfg.device, non_blocking=True)

    atlas_img, atlas_label = get_atlas_batch(atlas_image_cpu, atlas_label_cpu, spatial_shape, cfg.device)
    with torch.inference_mode():
        atlas_info = model.build_atlas_prior_tokens(atlas_img, atlas_label, batch_size=1)
    atlas_prior_tokens = atlas_info["atlas_prior_tokens"].detach().cpu().contiguous()
    tokens_map[key] = atlas_prior_tokens
    if cache_path is not None:
        save_atlas_token_cache(cache_path, cache_state)
    return atlas_prior_tokens.to(cfg.device, non_blocking=True)


def collect_export_spatial_shapes(datasets: dict[str, object], cfg: Config) -> list[tuple[int, int, int]]:
    if not cfg.enable_atlas_token_path or cfg.disable_atlas_token_path_infer:
        return []
    split_names = []
    if cfg.export_train:
        split_names.append("train")
    if cfg.export_val:
        split_names.append("val")
    if cfg.export_test:
        split_names.append("test")
    shapes: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for split_name in split_names:
        dataset = datasets[split_name]
        for sample_idx in range(len(dataset)):
            sample = dataset[sample_idx]
            shape = tuple(int(v) for v in sample[1].shape[1:])
            shape = get_downsampled_shape(shape, cfg.downsample_factor)
            if shape not in seen:
                seen.add(shape)
                shapes.append(shape)
    return shapes


def prewarm_atlas_token_cache(
    model,
    atlas_image_cpu: torch.Tensor | None,
    atlas_label_cpu: torch.Tensor | None,
    datasets: dict[str, object],
    cfg: Config,
    cache_state: dict[str, object],
    cache_path: str,
):
    """ split  atlas token cache

     token
     case
    """

    if not cfg.enable_atlas_token_path or cfg.disable_atlas_token_path_infer:
        return
    if atlas_image_cpu is None or atlas_label_cpu is None:
        raise ValueError('atlas cache prewarm requires atlas_image_cpu and atlas_label_cpu')
    if cfg.force_rebuild_atlas_token_cache:
        cache_state["tokens"] = {}
        if os.path.exists(cache_path):
            os.remove(cache_path)
    shapes = collect_export_spatial_shapes(datasets, cfg)
    if not shapes:
        save_atlas_token_cache(cache_path, cache_state)
        print('Atlas token cache | no exported splits, skip prewarm')
        return
    print(f'Atlas token cache | preparing {len(shapes)} shape(s)')
    for spatial_shape in shapes:
        get_or_create_atlas_prior_tokens(
            model,
            atlas_image_cpu,
            atlas_label_cpu,
            spatial_shape,
            cfg,
            cache_state,
            cache_path,
        )
    print(f'Atlas token cache saved: {cache_path}')


def read_label_nifti_as_tensor(label_path: str, device: str) -> torch.Tensor:
    label_np = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.int64)
    return torch.from_numpy(label_np).to(device).unsqueeze(0).unsqueeze(0)


def write_label_nifti(label_tensor: torch.Tensor, ref_path: str, save_path: str) -> None:
    ref_img = sitk.ReadImage(ref_path)
    label_np = label_tensor.detach().cpu().squeeze().numpy().astype(np.int16)
    out_img = sitk.GetImageFromArray(label_np)
    out_img.SetSpacing(ref_img.GetSpacing())
    out_img.SetOrigin(ref_img.GetOrigin())
    out_img.SetDirection(ref_img.GetDirection())
    sitk.WriteImage(out_img, save_path)


def resolve_ckpt(cfg: Config) -> str:
    if cfg.checkpoint_name == "resume_latest.pth":
        return os.path.join(cfg.model_dir, cfg.checkpoint_name)
    if cfg.checkpoint_name == "best_dice":
        prefixes = ("best_real_dice_ep", "best_dice_ep")
    elif cfg.checkpoint_name == "best_loss":
        prefixes = ("best_loss_ep",)
    else:
        return os.path.join(cfg.model_dir, cfg.checkpoint_name)
    candidates = sorted(
        f for f in os.listdir(cfg.model_dir)
        if f.endswith(".pth") and any(f.startswith(prefix) for prefix in prefixes)
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for key: {cfg.checkpoint_name}")
    return os.path.join(cfg.model_dir, candidates[-1])


def build_datasets(cfg: Config) -> dict[str, object]:
    """Build ADNI, BCV, or OASIS segmentation inference datasets."""

    real_root = os.path.abspath(cfg.real_root_dir)
    root_name = os.path.basename(real_root)

    if "ADNI" in root_name:
        return {
            "train": ADNI_Split_Dataset(cfg.real_root_dir, is_training=True, inference=True),
            "val": ADNI_Split_Dataset(cfg.real_root_dir, is_training=False, is_val=True, inference=True),
            "test": ADNI_Split_Dataset(cfg.real_root_dir, is_training=False, is_val=False, inference=True),
        }
    if "OASIS" in root_name:
        return {
            "train": OASIS_Dataset(cfg.real_root_dir, is_training=True, inference=True),
            "val": OASIS_Dataset(cfg.real_root_dir, is_training=False, is_val=True, inference=True),
            "test": OASIS_Dataset(cfg.real_root_dir, is_training=False, is_val=False, inference=True),
        }
    if "bcv" in root_name.lower():
        return {
            "train": BCV_Dataset(cfg.real_root_dir, is_training=True, inference=True, atlas_id=cfg.bcv_atlas_id),
            "val": BCV_Dataset(cfg.real_root_dir, is_training=False, is_val=True, inference=True, atlas_id=cfg.bcv_atlas_id),
            "test": BCV_Dataset(cfg.real_root_dir, is_training=False, is_val=False, inference=True, atlas_id=cfg.bcv_atlas_id),
        }
    raise ValueError(
        f"Unsupported real_root_dir for OPAL-Net release inference: {cfg.real_root_dir}. "
        "Supported datasets are ADNI, BCV, and OASIS."
    )


def stem_from_name(file_name: str) -> str:
    stem = Path(file_name).stem
    if stem.endswith(".nii"):
        stem = Path(stem).stem
    return stem


def compute_per_class_dice(pred_label: np.ndarray, target_label: np.ndarray, num_classes: int) -> tuple[list[int], list[float]]:
    class_ids = list(range(1, num_classes))
    dice_values: list[float] = []
    for class_id in class_ids:
        pred_mask = pred_label == class_id
        target_mask = target_label == class_id
        union = pred_mask.sum() + target_mask.sum()
        if union == 0:
            dice_values.append(float("nan"))
            continue
        intersection = np.logical_and(pred_mask, target_mask).sum()
        dice_values.append(float((2.0 * intersection) / (union + 1e-8)))
    return class_ids, dice_values


def compute_per_class_hd95(pred_label: torch.Tensor, target_label: torch.Tensor, num_classes: int, hd95_metric: HausdorffDistanceMetric) -> list[float]:
    pred_onehot = F.one_hot(pred_label.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    target_onehot = F.one_hot(target_label.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    pred_onehot = pred_onehot[:, 1:]
    target_onehot = target_onehot[:, 1:]
    hd95_metric(y_pred=pred_onehot, y=target_onehot)
    hd_tensor = hd95_metric.aggregate()
    hd95_metric.reset()
    hd_array = hd_tensor.detach().cpu().numpy().reshape(-1)
    return [float(x) if np.isfinite(x) else float("nan") for x in hd_array.tolist()]


def compute_monai_mean_hd95(
    pred_label: torch.Tensor,
    target_label: torch.Tensor,
    num_classes: int,
    hd95_metric: HausdorffDistanceMetric,
) -> float:
    pred_onehot = F.one_hot(pred_label.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    target_onehot = F.one_hot(target_label.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    hd95_metric(y_pred=pred_onehot, y=target_onehot)
    hd_tensor = hd95_metric.aggregate()
    hd95_metric.reset()
    hd95 = float(hd_tensor.item())
    return hd95 if np.isfinite(hd95) else float("nan")


def nan_stats(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def build_empty_class_summary(class_ids: list[int]) -> dict[str, dict[str, float | None]]:
    return {str(class_id): {"mean": None, "std": None} for class_id in class_ids}


def overall_stats_from_class_summary(class_summary: dict[str, dict[str, float | None]]) -> dict[str, float | None]:
    class_means = []
    for stats in class_summary.values():
        mean_val = stats.get("mean")
        if mean_val is not None and np.isfinite(float(mean_val)):
            class_means.append(float(mean_val))
    return nan_stats(class_means)


def wait_for_cuda_memory_release(device: str, target_allocated: int, timeout_sec: float = 60.0, poll_sec: float = 0.1, tolerance_bytes: int = 8 * 1024 * 1024) -> None:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return
    device_obj = torch.device(device)
    torch.cuda.synchronize(device_obj)
    torch.cuda.empty_cache()
    start = time.time()
    while True:
        current_allocated = torch.cuda.memory_allocated(device_obj)
        if current_allocated <= target_allocated + tolerance_bytes:
            break
        if time.time() - start >= timeout_sec:
            current = current_allocated / (1024 ** 2)
            target = target_allocated / (1024 ** 2)
            print(
                f"warning | cuda memory did not fall back in time | current={current:.1f}MB | "
                f"target<={target:.1f}MB | tolerance={tolerance_bytes / (1024 ** 2):.1f}MB"
            )
            break
        time.sleep(poll_sec)
        torch.cuda.empty_cache()


def logits_to_pred_label(real_logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(real_logits, dim=1, keepdim=True)


def evaluate_split_dice(
    model,
    atlas_image_cpu: torch.Tensor,
    atlas_label_cpu: torch.Tensor,
    dataset,
    split_name: str,
    cfg: Config,
    save_pred_label: bool,
    token_cache_state: dict[str, object] | None,
    token_cache_file: str | None,
):
    """ split  Dice


    -  case
    -  per-class Dice
    -  split  per-class / overall
    -  `inference_seg_<split>_label/`
    """

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)
    pred_dir = os.path.join(cfg.model_dir, f"inference_seg_{split_name}_label")
    if save_pred_label:
        os.makedirs(pred_dir, exist_ok=True)

    class_ids = list(range(1, cfg.num_classes))
    dice_per_class_records = {class_id: [] for class_id in class_ids}
    case_dice_means: list[float] = []
    per_case = []

    with torch.inference_mode():
        for batch in tqdm(loader, total=len(loader), desc=f"dice {split_name}", leave=False):
            real_image = batch[1].to(cfg.device)
            real_label = batch[3].to(cfg.device)
            real_image, real_label = downsample_image_label(real_image, real_label, cfg.downsample_factor)
            if real_image.shape[0] != 1:
                raise ValueError("This inference script expects batch_size=1 for per-case metric logging")

            real_path = batch[4][0]
            real_name = batch[5][0]
            atlas_prior_tokens = get_or_create_atlas_prior_tokens(
                model,
                atlas_image_cpu,
                atlas_label_cpu,
                tuple(real_image.shape[2:]),
                cfg,
                token_cache_state,
                token_cache_file,
            )
            out = model(
                x_real=real_image,
                atlas_prior_tokens_override=atlas_prior_tokens,
                is_training=False,
                disable_atlas_token_path_infer=cfg.disable_atlas_token_path_infer,
            )
            pred_label = logits_to_pred_label(out["real_logits"])

            pred_np = pred_label.squeeze().cpu().numpy()
            target_np = real_label.squeeze().cpu().numpy()
            _, dice_values = compute_per_class_dice(pred_np, target_np, cfg.num_classes)
            dice_case_stats = nan_stats(dice_values)

            for class_id, dice_val in zip(class_ids, dice_values):
                if np.isfinite(dice_val):
                    dice_per_class_records[class_id].append(float(dice_val))
            if dice_case_stats["mean"] is not None:
                case_dice_means.append(float(dice_case_stats["mean"]))

            pred_path = None
            if save_pred_label:
                stem = stem_from_name(real_name)
                pred_path = os.path.join(pred_dir, f"{stem}_pred_label.nii.gz")
                write_label_nifti(pred_label, real_path, pred_path)

            per_case.append(
                {
                    "name": real_name,
                    "dice_per_class": {str(class_id): (None if not np.isfinite(val) else float(val)) for class_id, val in zip(class_ids, dice_values)},
                    "dice_mean": dice_case_stats["mean"],
                    "dice_std": dice_case_stats["std"],
                    "pred_label_path": pred_path,
                }
            )
            print(
                f"{split_name} | {real_name} | "
                f"dice_mean={0.0 if dice_case_stats['mean'] is None else dice_case_stats['mean']:.4f}"
            )

    dice_summary = {
        "split": split_name,
        "num_cases": len(per_case),
        "dice": {
            "per_class": {str(class_id): nan_stats(values) for class_id, values in dice_per_class_records.items()},
            "overall": overall_stats_from_class_summary(
                {str(class_id): nan_stats(values) for class_id, values in dice_per_class_records.items()}
            ),
            "overall_case_based": nan_stats(case_dice_means),
        },
        "per_case": per_case,
    }
    with open(os.path.join(cfg.model_dir, f"inference_seg_{split_name}_dice_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(dice_summary, f, indent=2, ensure_ascii=False)
    print(
        f"{split_name} dice summary | "
        f"dice_mean={0.0 if dice_summary['dice']['overall']['mean'] is None else dice_summary['dice']['overall']['mean']:.4f} | "
        f"dice_std={0.0 if dice_summary['dice']['overall']['std'] is None else dice_summary['dice']['overall']['std']:.4f}"
    )
    return dice_summary


def evaluate_split_hd95(
    model,
    atlas_image_cpu: torch.Tensor,
    atlas_label_cpu: torch.Tensor,
    dataset,
    split_name: str,
    cfg: Config,
    pred_dir: str,
    token_cache_state: dict[str, object] | None,
    token_cache_file: str | None,
):
    """ split  HD95


    -
    -  HD95
    """

    class_ids = list(range(1, cfg.num_classes))
    if not cfg.compute_hd95:
        summary = {
            "split": split_name,
            "num_cases": 0,
            "compute_hd95": False,
            "hd95_mode": cfg.hd95_mode,
            "hd95_source": "disabled",
            "hd95": {"per_class": build_empty_class_summary(class_ids), "overall": {"mean": None, "std": None}, "overall_case_based": {"mean": None, "std": None}},
            "per_case": [],
        }
        with open(os.path.join(cfg.model_dir, f"inference_seg_{split_name}_hd95_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)
    hd95_per_class_records = {class_id: [] for class_id in class_ids}
    case_hd95_means: list[float] = []
    per_case = []
    if cfg.hd95_mode == "per_class_mean":
        hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
    elif cfg.hd95_mode == "monai_mean":
        hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95)
    else:
        raise ValueError(f"Unsupported hd95_mode: {cfg.hd95_mode}")

    with torch.inference_mode():
        for batch in tqdm(loader, total=len(loader), desc=f"hd95 {split_name}", leave=False):
            case_memory_baseline = 0
            if torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
                case_memory_baseline = torch.cuda.memory_allocated(torch.device(cfg.device))

            real_image = batch[1].to(cfg.device)
            real_label = batch[3].to(cfg.device)
            real_image, real_label = downsample_image_label(real_image, real_label, cfg.downsample_factor)
            if real_image.shape[0] != 1:
                raise ValueError("This inference script expects batch_size=1 for per-case metric logging")

            real_path = batch[4][0]
            real_name = batch[5][0]
            stem = stem_from_name(real_name)
            pred_path = os.path.join(pred_dir, f"{stem}_pred_label.nii.gz")
            hd95_source = "saved_pred"

            if cfg.hd95_use_saved_pred and os.path.exists(pred_path):
                pred_label = read_label_nifti_as_tensor(pred_path, cfg.device)
            else:
                hd95_source = "rerun"
                atlas_prior_tokens = get_or_create_atlas_prior_tokens(
                    model,
                    atlas_image_cpu,
                    atlas_label_cpu,
                    tuple(real_image.shape[2:]),
                    cfg,
                    token_cache_state,
                    token_cache_file,
                )
                out = model(
                    x_real=real_image,
                    atlas_prior_tokens_override=atlas_prior_tokens,
                    is_training=False,
                    disable_atlas_token_path_infer=cfg.disable_atlas_token_path_infer,
                )
                pred_label = logits_to_pred_label(out["real_logits"])
                del out
                if cfg.save_pred_label or cfg.save_pred_label_train or cfg.save_pred_label_val or cfg.save_pred_label_test or cfg.hd95_use_saved_pred:
                    os.makedirs(pred_dir, exist_ok=True)
                    write_label_nifti(pred_label, real_path, pred_path)

            if cfg.hd95_mode == "per_class_mean":
                hd95_values = compute_per_class_hd95(pred_label, real_label, cfg.num_classes, hd95_metric)
                hd95_case_stats = nan_stats(hd95_values)
                for class_id, hd95_val in zip(class_ids, hd95_values):
                    if np.isfinite(hd95_val):
                        hd95_per_class_records[class_id].append(float(hd95_val))
            else:
                hd95_single = compute_monai_mean_hd95(pred_label, real_label, cfg.num_classes, hd95_metric)
                hd95_values = [float("nan")] * len(class_ids)
                hd95_case_stats = {
                    "mean": None if not np.isfinite(hd95_single) else float(hd95_single),
                    "std": 0.0 if np.isfinite(hd95_single) else None,
                }

            if hd95_case_stats["mean"] is not None:
                case_hd95_means.append(float(hd95_case_stats["mean"]))
            per_case.append(
                {
                    "name": real_name,
                    "hd95_per_class": {str(class_id): (None if not np.isfinite(val) else float(val)) for class_id, val in zip(class_ids, hd95_values)},
                    "hd95_mean": hd95_case_stats["mean"],
                    "hd95_std": hd95_case_stats["std"],
                    "pred_label_path": pred_path if os.path.exists(pred_path) else None,
                    "hd95_source": hd95_source,
                }
            )
            print(
                f"{split_name} | {real_name} | "
                f"hd95_mean={0.0 if hd95_case_stats['mean'] is None else hd95_case_stats['mean']:.4f} | "
                f"source={hd95_source}"
            )

            del pred_label, real_image, real_label
            if cfg.wait_cuda_memory_release and torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
                wait_for_cuda_memory_release(
                    cfg.device,
                    case_memory_baseline,
                    timeout_sec=cfg.cuda_memory_wait_timeout_sec,
                    poll_sec=cfg.cuda_memory_wait_poll_sec,
                    tolerance_bytes=int(cfg.cuda_memory_wait_tolerance_mb * 1024 * 1024),
                )

    hd95_summary = {
        "split": split_name,
        "num_cases": len(per_case),
        "compute_hd95": True,
        "hd95_mode": cfg.hd95_mode,
        "hd95_source": "saved_pred" if cfg.hd95_use_saved_pred else "rerun",
        "hd95": {
            "per_class": {str(class_id): nan_stats(values) for class_id, values in hd95_per_class_records.items()},
            "overall": overall_stats_from_class_summary(
                {str(class_id): nan_stats(values) for class_id, values in hd95_per_class_records.items()}
            ) if cfg.hd95_mode == "per_class_mean" else nan_stats(case_hd95_means),
            "overall_case_based": nan_stats(case_hd95_means),
        },
        "per_case": per_case,
    }
    with open(os.path.join(cfg.model_dir, f"inference_seg_{split_name}_hd95_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(hd95_summary, f, indent=2, ensure_ascii=False)
    print(
        f"{split_name} hd95 summary | "
        f"hd95_mean={0.0 if hd95_summary['hd95']['overall']['mean'] is None else hd95_summary['hd95']['overall']['mean']:.4f} | "
        f"hd95_std={0.0 if hd95_summary['hd95']['overall']['std'] is None else hd95_summary['hd95']['overall']['std']:.4f}"
    )
    return hd95_summary


def merge_split_metric_summaries(dice_summary: dict, hd95_summary: dict) -> dict:
    """ Dice  HD95  split """

    merged_cases: dict[str, dict] = {}
    for item in dice_summary.get("per_case", []):
        merged_cases[item["name"]] = dict(item)
    for item in hd95_summary.get("per_case", []):
        merged_cases.setdefault(item["name"], {"name": item["name"]})
        merged_cases[item["name"]].update(item)
    summary = {
        "split": dice_summary.get("split", hd95_summary.get("split")),
        "num_cases": max(int(dice_summary.get("num_cases", 0)), int(hd95_summary.get("num_cases", 0))),
        "compute_hd95": bool(hd95_summary.get("compute_hd95", False)),
        "hd95_mode": hd95_summary.get("hd95_mode"),
        "dice": dice_summary.get("dice"),
        "hd95": hd95_summary.get("hd95"),
        "per_case": list(merged_cases.values()),
    }
    return summary


def evaluate_split(
    model,
    atlas_image_cpu: torch.Tensor,
    atlas_label_cpu: torch.Tensor,
    dataset,
    split_name: str,
    cfg: Config,
    save_pred_label: bool,
    token_cache_state: dict[str, object] | None,
    token_cache_file: str | None,
):
    """ split

     Dice HD95 `inference_seg_<split>_metrics.json`
    """

    pred_dir = os.path.join(cfg.model_dir, f"inference_seg_{split_name}_label")
    save_pred_for_dice = bool(save_pred_label or (cfg.compute_hd95 and cfg.hd95_use_saved_pred))
    dice_summary = evaluate_split_dice(
        model,
        atlas_image_cpu,
        atlas_label_cpu,
        dataset,
        split_name,
        cfg,
        save_pred_label=save_pred_for_dice,
        token_cache_state=token_cache_state,
        token_cache_file=token_cache_file,
    )
    hd95_summary = evaluate_split_hd95(
        model,
        atlas_image_cpu,
        atlas_label_cpu,
        dataset,
        split_name,
        cfg,
        pred_dir=pred_dir,
        token_cache_state=token_cache_state,
        token_cache_file=token_cache_file,
    )
    summary = merge_split_metric_summaries(dice_summary, hd95_summary)
    with open(os.path.join(cfg.model_dir, f"inference_seg_{split_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(
        f"{split_name} summary | "
        f"dice_mean={0.0 if summary['dice']['overall']['mean'] is None else summary['dice']['overall']['mean']:.4f} | "
        f"dice_std={0.0 if summary['dice']['overall']['std'] is None else summary['dice']['overall']['std']:.4f} | "
        f"hd95_mean={0.0 if summary['hd95']['overall']['mean'] is None else summary['hd95']['overall']['mean']:.4f} | "
        f"hd95_std={0.0 if summary['hd95']['overall']['std'] is None else summary['hd95']['overall']['std']:.4f}"
    )
    return summary


def main(cfg: Config) -> None:
    """


    1.
    2.  checkpoint
    3.  atlas token cache
    4.  split
    5.  `inference_seg_metrics_summary.json`
    """

    object.__setattr__(cfg, "model_dir", str(Path(cfg.model_dir).resolve()))
    cfg.resolve_from_training_config()

    ckpt_path = resolve_ckpt(cfg)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    inferred_share = infer_share_atlas_encoder_from_state_dict(ckpt["model_state_dict"])
    if cfg.share_atlas_encoder_with_seg != inferred_share:
        print(
            "Checkpoint architecture override | "
            f"share_atlas_encoder_with_seg: {cfg.share_atlas_encoder_with_seg} -> {inferred_share}"
        )
        object.__setattr__(cfg, "share_atlas_encoder_with_seg", inferred_share)

    datasets = build_datasets(cfg)
    atlas_image_cpu = None
    atlas_label_cpu = None
    if cfg.enable_atlas_token_path:
        atlas_image_cpu, atlas_label_cpu = build_atlas_cache_from_sample(datasets["val"][0])
    model = AtlasGuidedSegModel(
        in_channels=cfg.in_channels,
        out_channels=cfg.num_classes,
        base_channels=cfg.base_channels,
        dropout_prob=cfg.student_dropout,
        atlas_label_embed_channels=cfg.atlas_label_embed_channels,
        context_query_tokens=cfg.context_query_tokens,
        context_shuffle_factor=cfg.context_shuffle_factor,
        atlas_task_dropout=cfg.atlas_task_dropout,
        attn_heads=cfg.attn_heads,
        attn_dropout=cfg.attn_dropout,
        attn_gate_init=cfg.attn_gate_init,
        enable_atlas_token_path=cfg.enable_atlas_token_path,
        share_atlas_encoder_with_seg=cfg.share_atlas_encoder_with_seg,
        bottleneck_ctx_residual_weights=cfg.bottleneck_ctx_residual_weights,
    ).to(cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # atlas token cache checkpoint +  + downsample_factor
    token_cache_file = atlas_token_cache_path(cfg)
    token_cache_meta = build_atlas_token_cache_meta(cfg, ckpt_path)
    token_cache_state = load_atlas_token_cache(token_cache_file, token_cache_meta)
    if token_cache_state.get("meta") != token_cache_meta:
        token_cache_state = {"meta": dict(token_cache_meta), "tokens": {}}
    prewarm_atlas_token_cache(
        model,
        atlas_image_cpu,
        atlas_label_cpu,
        datasets,
        cfg,
        token_cache_state,
        token_cache_file,
    )

    all_summary = {}
    split_plan = [
        ("train", cfg.export_train, cfg.save_pred_label or cfg.save_pred_label_train),
        ("val", cfg.export_val, cfg.save_pred_label or cfg.save_pred_label_val),
        ("test", cfg.export_test, cfg.save_pred_label or cfg.save_pred_label_test),
    ]
    for split_name, should_export, should_save_pred in split_plan:
        if not should_export:
            continue
        all_summary[split_name] = evaluate_split(
            model,
            atlas_image_cpu,
            atlas_label_cpu,
            datasets[split_name],
            split_name,
            cfg,
            save_pred_label=should_save_pred,
            token_cache_state=token_cache_state,
            token_cache_file=token_cache_file,
        )

    with open(os.path.join(cfg.model_dir, "inference_seg_metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"Inference metrics saved under: {cfg.model_dir}")


if __name__ == "__main__":
    main(Config.from_env())
