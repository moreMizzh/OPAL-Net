"""Train the OPAL-Net atlas-guided segmentation model.

The entry point combines synthetic registration outputs for supervision with
real labeled data for validation, testing, and regularization.
"""

from __future__ import annotations

import datetime
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import (  # noqa: E402
    ADNI_Split_Dataset,
    ADNI_Split_Syn_Dataset,
    BCV_Dataset,
    BCV_Syn_Dataset,
    OASIS_Dataset,
    OASIS_Syn_Dataset,
)
import inference_seg as seg_infer  # noqa: E402
import losses  # noqa: E402
import utils  # noqa: E402
from model_seg import AtlasGuidedSegModel  # noqa: E402


# Dataset presets provide neutral defaults for the public release.
# Explicit OPAL_SEG_* environment variables override these values.
DATASET_INFO = {
    "ADNI": {
        "real_root_dir": "../data/ADNI_Dataset",
        "syn_root_dir": "./output/Reg/ADNI_reference_reg",
        "exp_name": "ADNI_seg",
        "num_classes": 4,
        "batch_size": 1,
    },
    "BCV": {
        "real_root_dir": "../data/BCV/bcv_small",
        "syn_root_dir": "./output/Reg/BCV_cascaded_reg",
        "exp_name": "BCV_seg_cascaded_reg_hd",
        "num_classes": 5,
        "batch_size": 1,
    },
    "OASIS": {
        "real_root_dir": "../data/OASIS/OASIS",
        "syn_root_dir": "./output/Reg/OASIS_cascaded_reg",
        "exp_name": "OASIS_seg_cascaded_reg_atlas_task_unet_bica_nres",
        "num_classes": 36,
        "batch_size": 1,
    },
}


def apply_dataset_preset(cfg, dataset_key: str | None):
    if not dataset_key:
        return cfg
    preset = DATASET_INFO.get(dataset_key)
    if preset is None:
        raise ValueError(f"Unknown dataset_key: {dataset_key}. Available keys: {sorted(DATASET_INFO)}")
    for field_name, value in preset.items():
        if hasattr(cfg, field_name):
            object.__setattr__(cfg, field_name, value)
    object.__setattr__(cfg, "dataset_key", dataset_key)
    return cfg

@dataclass(frozen=True)
class Config:
    """Segmentation training configuration.

    The main groups are data paths, model structure switches, and regularizer
    weights. Values can be overridden with OPAL_SEG_* environment variables.
    """
    save_dir: str = ""                         #
    version: str = "seg"                     #  config.json
    syn_key: str = "final"                     # reg  syn final/coarse
    continue_train: bool = False               #  resume_latest.pth
    save_root_dir: str = "output/Seg"          # ss

    dataset_key: str = "BCV"                   #  dataset
    real_root_dir: str = ""                    #
    syn_root_dir: str = ""                     # reg  syn
    exp_name: str = ""                         #  save_dir
    num_classes: int = 0                       #
    bcv_atlas_id: int | None = None            # Optional BCV atlas ID; defaults to 27 in the dataset loader.

    base_channels: int = 16                    # UNet 163264128
    in_channels: int = 1                       # =1
    batch_size: int = 1                        # 3D  case
    epochs: int = 200                          # +
    epoch_start: int = 0                       #  epoch checkpoint
    save_every: int = 20                       #  checkpoint
    patience: int = 30                         # early stopping
    early_stop_min_delta: float = 0.0005      # Dice
    num_workers: int = 0                       # DataLoader 0=
    amp_enabled: bool = False # True                  #
    max_grad_norm: float = 3.0                #
    plot_overwrite: bool = True                #
    student_dropout: float = 0.0               # UNet  Dropout3d
    downsample_factor: int = 1               # //1
    lr: float = 5e-4#                             # Adam zz
    weight_decay: float = 1e-3# 1e-5 #                 #
    seed: int = 42 # random.randint(0, 2**31 - 1)   #

    #
    seg_weight: float =  0.5                  #
    ce_weight: float = 1.0                    # CE
    dice_weight: float = 1#1.0                # Dice

    atlas_label_embed_channels: int = 8        # atlas label embedding
    context_query_tokens: int = 48# 8          # context query token
    context_shuffle_factor: int = 2            # pixel shuffle  mask
    atlas_task_dropout: float = 0.0            # task encoder  dropout
    attn_heads: int = 8                        # cross-attention
    attn_dropout: float = 0.0                  # attention dropout
    attn_gate_init: float = -4.0               # attention gate =
    enable_atlas_token_path: bool = True       # True= atlas token pathFalse= token
    share_atlas_encoder_with_seg: bool = False  # True=atlas  seg encoderFalse=atlas  encoder
    bottleneck_ctx_residual_weights: tuple[float, float] = (1.0, 1.0)  # bottleneck  = a * image_feat + b * atlas_delta

    entropy_weight: float = 0.1# 0.05              # real >0
    prior_weight: float = 0.1                 # real >0
    prior_include_background: bool = False    # False
    prior_smoothing: float = 1e-4             #  batch  KL

    mmd_weight: tuple[float, ...] = (0.1, 0.1, 0.1)#     #  MMD  mmd_layer_idx
    mmd_layer_idx: tuple[int, ...] = (3, 4, 5)#          # MMD  bottleneck
    mmd_warmup_epochs: int = 0                 # MMD  N  MMD
    # -------------------

    auto_infer_after_train: bool = True        #  inference  python train_seg.py
    auto_infer_export_val: bool = False        #  val split
    auto_infer_export_test: bool = True        #  test split
    auto_infer_compute_hd95: bool = True       #  HD95
    auto_infer_checkpoint_name: str = "best_dice"  #  checkpoint
    auto_infer_save_pred_label: bool = False   #  nii

    @classmethod
    def from_env(cls):
        """ `OPAL_SEG_*`

         reg  `dataset_key`
        """

        cfg = cls()
        cfg = apply_dataset_preset(cfg, cfg.dataset_key)

        def _parse_int_tuple(raw: str) -> tuple[int, ...]:
            if raw.strip() == "":
                return ()
            return tuple(int(p.strip()) for p in raw.split(",") if p.strip() != "")

        def _parse_float_tuple(raw: str) -> tuple[float, ...]:
            if raw.strip() == "":
                return ()
            return tuple(float(p.strip()) for p in raw.split(",") if p.strip() != "")

        env_map = {
            "OPAL_SEG_SAVE_DIR": ("save_dir", str),
            "OPAL_SEG_VERSION": ("version", str),
            "OPAL_SEG_REAL_ROOT_DIR": ("real_root_dir", str),
            "OPAL_SEG_SYN_ROOT_DIR": ("syn_root_dir", str),
            "OPAL_SEG_SYN_KEY": ("syn_key", str),
            "OPAL_SEG_SAVE_ROOT_DIR": ("save_root_dir", str),
            "OPAL_SEG_EXP_NAME": ("exp_name", str),
            "OPAL_SEG_CONTINUE_TRAIN": ("continue_train", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_NUM_CLASSES": ("num_classes", int),
            "OPAL_SEG_BCV_ATLAS_ID": ("bcv_atlas_id", int),
            "OPAL_SEG_BASE_CHANNELS": ("base_channels", int),
            "OPAL_SEG_IN_CHANNELS": ("in_channels", int),
            "OPAL_SEG_STUDENT_DROPOUT": ("student_dropout", float),
            "OPAL_SEG_ATLAS_LABEL_EMBED_CHANNELS": ("atlas_label_embed_channels", int),
            "OPAL_SEG_CONTEXT_QUERY_TOKENS": ("context_query_tokens", int),
            "OPAL_SEG_CONTEXT_SHUFFLE_FACTOR": ("context_shuffle_factor", int),
            "OPAL_SEG_ATLAS_TASK_DROPOUT": ("atlas_task_dropout", float),
            "OPAL_SEG_ATTN_HEADS": ("attn_heads", int),
            "OPAL_SEG_ATTN_DROPOUT": ("attn_dropout", float),
            "OPAL_SEG_ATTN_GATE_INIT": ("attn_gate_init", float),
            "OPAL_SEG_ENABLE_ATLAS_TOKEN_PATH": ("enable_atlas_token_path", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_SHARE_ATLAS_ENCODER_WITH_SEG": ("share_atlas_encoder_with_seg", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_BOTTLENECK_CTX_RESIDUAL_WEIGHTS": ("bottleneck_ctx_residual_weights", _parse_float_tuple),
            "OPAL_SEG_BATCH_SIZE": ("batch_size", int),
            "OPAL_SEG_EPOCHS": ("epochs", int),
            "OPAL_SEG_EPOCH_START": ("epoch_start", int),
            "OPAL_SEG_SAVE_EVERY": ("save_every", int),
            "OPAL_SEG_PATIENCE": ("patience", int),
            "OPAL_SEG_NUM_WORKERS": ("num_workers", int),
            "OPAL_SEG_DOWNSAMPLE_FACTOR": ("downsample_factor", int),
            "OPAL_SEG_LR": ("lr", float),
            "OPAL_SEG_WEIGHT_DECAY": ("weight_decay", float),
            "OPAL_SEG_SEED": ("seed", int),
            "OPAL_SEG_AMP_ENABLED": ("amp_enabled", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_MAX_GRAD_NORM": ("max_grad_norm", float),
            "OPAL_SEG_EARLY_STOP_MIN_DELTA": ("early_stop_min_delta", float),
            "OPAL_SEG_SEG_WEIGHT": ("seg_weight", float),
            "OPAL_SEG_CE_WEIGHT": ("ce_weight", float),
            "OPAL_SEG_DICE_WEIGHT": ("dice_weight", float),
            "OPAL_SEG_ENTROPY_WEIGHT": ("entropy_weight", float),
            "OPAL_SEG_PRIOR_WEIGHT": ("prior_weight", float),
            "OPAL_SEG_PRIOR_INCLUDE_BACKGROUND": ("prior_include_background", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_PRIOR_SMOOTHING": ("prior_smoothing", float),
            "OPAL_SEG_MMD_WEIGHT": ("mmd_weight", _parse_float_tuple),
            "OPAL_SEG_MMD_LAYER_IDX": ("mmd_layer_idx", _parse_int_tuple),
            "OPAL_SEG_MMD_WARMUP_EPOCHS": ("mmd_warmup_epochs", int),
            "OPAL_SEG_PLOT_OVERWRITE": ("plot_overwrite", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_AUTO_INFER_AFTER_TRAIN": ("auto_infer_after_train", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_AUTO_INFER_EXPORT_VAL": ("auto_infer_export_val", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_AUTO_INFER_EXPORT_TEST": ("auto_infer_export_test", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_AUTO_INFER_COMPUTE_HD95": ("auto_infer_compute_hd95", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_SEG_AUTO_INFER_CHECKPOINT": ("auto_infer_checkpoint_name", str),
            "OPAL_SEG_AUTO_INFER_SAVE_PRED_LABEL": ("auto_infer_save_pred_label", lambda v: v.lower() in {"1", "true", "yes", "on"}),
        }
        for env_name, (field_name, cast_fn) in env_map.items():
            raw_val = os.environ.get(env_name)
            if raw_val is not None and raw_val != "":
                object.__setattr__(cfg, field_name, cast_fn(raw_val))
        return cfg

    def validate(self) -> None:
        """"""

        if len(self.bottleneck_ctx_residual_weights) != 2:
            raise ValueError(
                "bottleneck_ctx_residual_weights must contain exactly 2 values: (image_weight, delta_weight)"
            )
        if self.context_query_tokens <= 0:
            raise ValueError("context_query_tokens must be positive")
        if self.context_shuffle_factor <= 0:
            raise ValueError("context_shuffle_factor must be positive")
        if len(self.mmd_layer_idx) == 0:
            raise ValueError("mmd_layer_idx must contain at least one feature index")
        if len(self.mmd_weight) == 0:
            raise ValueError("mmd_weight must contain at least one value")
        if len(self.mmd_layer_idx) != len(self.mmd_weight):
            raise ValueError(
                f"mmd_layer_idx and mmd_weight length mismatch: {len(self.mmd_layer_idx)} vs {len(self.mmd_weight)}"
            )
        if self.downsample_factor <= 0:
            raise ValueError("downsample_factor must be positive")
        if self.prior_weight < 0:
            raise ValueError("prior_weight must be non-negative")
        if self.prior_smoothing <= 0:
            raise ValueError("prior_smoothing must be positive")

    def makedirs(self):
        save_dir = self.save_dir
        if not save_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            atlas = "T" if self.enable_atlas_token_path else "F"
            share = "S" if self.share_atlas_encoder_with_seg else "I"
            mmd = "T" if sum(self.mmd_weight) != 0 else "F"
            entropy = "T" if self.entropy_weight != 0 else "F"
            prior = "T" if self.prior_weight != 0 else "F"
            save_dir = os.path.join(self.save_root_dir, f"{self.exp_name}_{atlas}{share}{mmd}{entropy}{prior}_{self.syn_key}_{self.seed}_{ts}")
            object.__setattr__(self, "save_dir", save_dir)
        os.makedirs(save_dir, exist_ok=True)

    def save_json(self, file_name: str = "config.json"):
        with open(os.path.join(self.save_dir, file_name), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

        def _seed_worker(_worker_id: int):
            worker_seed = torch.initial_seed() % 2**32
            random.seed(worker_seed)
            np.random.seed(worker_seed)

        data_loader_generator = torch.Generator()
        data_loader_generator.manual_seed(self.seed)
        return _seed_worker, data_loader_generator


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


def build_base_datasets(cfg: Config):
    """Build ADNI, BCV, or OASIS segmentation train/val/test datasets."""

    real_root = os.path.abspath(cfg.real_root_dir)
    root_name = os.path.basename(real_root)

    if "ADNI" in root_name:
        train_set = ADNI_Split_Syn_Dataset(cfg.real_root_dir, cfg.syn_root_dir, syn_key=cfg.syn_key, is_training=True)
        val_set = ADNI_Split_Dataset(cfg.real_root_dir, is_training=False, is_val=True)
        test_set = ADNI_Split_Dataset(cfg.real_root_dir, is_training=False, is_val=False)
        return train_set, val_set, test_set

    if "OASIS" in root_name:
        train_set = OASIS_Syn_Dataset(cfg.real_root_dir, cfg.syn_root_dir, syn_key=cfg.syn_key)
        val_set = OASIS_Dataset(cfg.real_root_dir, is_training=False, is_val=True)
        test_set = OASIS_Dataset(cfg.real_root_dir, is_training=False, is_val=False)
        return train_set, val_set, test_set

    if "bcv" in root_name.lower():
        train_set = BCV_Syn_Dataset(
            cfg.real_root_dir,
            cfg.syn_root_dir,
            syn_key=cfg.syn_key,
            transform2img=None,
            transform2both=None,
            atlas_id=cfg.bcv_atlas_id,
        )
        val_set = BCV_Dataset(cfg.real_root_dir, is_training=False, is_val=True, atlas_id=cfg.bcv_atlas_id)
        test_set = BCV_Dataset(cfg.real_root_dir, is_training=False, is_val=False, atlas_id=cfg.bcv_atlas_id)
        return train_set, val_set, test_set

    raise ValueError(
        f"Unsupported real_root_dir for OPAL-Net release builder: {cfg.real_root_dir}. "
        "Supported datasets are ADNI, BCV, and OASIS."
    )


def get_effective_mmd_weights(cfg: Config, epoch: int) -> tuple[float, ...]:
    """ warmup  epoch  MMD """

    if cfg.mmd_warmup_epochs <= 0:
        return tuple(cfg.mmd_weight)
    ratio = min(1.0, float(epoch + 1) / float(cfg.mmd_warmup_epochs))
    return tuple(w * ratio for w in cfg.mmd_weight)


def get_effective_mmd_weight_sum(cfg: Config, epoch: int) -> float:
    return float(sum(get_effective_mmd_weights(cfg, epoch)))


def compute_entropy_minimization_loss(logits: torch.Tensor) -> torch.Tensor:
    """"""

    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)
    return entropy.mean()


def compute_class_ratio_prior_loss(
    logits: torch.Tensor,
    target_label: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
    smoothing: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """

     real  syn

    """

    probs = torch.softmax(logits, dim=1)
    pred_ratio = probs.mean(dim=(0, 2, 3, 4))

    if target_label.ndim == 5 and target_label.shape[1] == 1:
        target_index = target_label[:, 0].long()
    else:
        target_index = target_label.long()
    target_ratio = torch.bincount(target_index.reshape(-1), minlength=num_classes).to(dtype=probs.dtype, device=probs.device)

    if not include_background:
        pred_ratio = pred_ratio[1:]
        target_ratio = target_ratio[1:]

    pred_ratio = (pred_ratio + smoothing) / (pred_ratio.sum() + smoothing * pred_ratio.numel())
    target_ratio = (target_ratio + smoothing) / (target_ratio.sum() + smoothing * target_ratio.numel())
    return torch.sum(target_ratio * (torch.log(target_ratio.clamp_min(eps)) - torch.log(pred_ratio.clamp_min(eps))))


def compute_losses_seg(
    out: dict[str, torch.Tensor],
    syn_label: torch.Tensor,
    criterions: dict[str, torch.nn.Module],
    cfg: Config,
    effective_mmd_weights: tuple[float, ...],
):
    """ loss


    - `loss_seg`syn CE + Dice
    - `loss_mmd`syn/real
    - `loss_entropy`
    - `loss_prior`
    """

    syn_logits = out["syn_logits"]
    loss_ce = criterions["ce"](syn_logits, syn_label)
    loss_dice = criterions["dice"](syn_logits, syn_label)
    loss_seg = cfg.ce_weight * loss_ce + cfg.dice_weight * loss_dice

    loss_mmd = loss_seg.new_tensor(0.0)
    if any(w > 0 for w in effective_mmd_weights) and "real_features" in out:
        syn_features = out["syn_features"]
        real_features = out["real_features"]
        for layer_idx, layer_weight in zip(cfg.mmd_layer_idx, effective_mmd_weights):
            if layer_weight <= 0:
                continue
            if layer_idx < 0 or layer_idx >= len(syn_features) or layer_idx >= len(real_features):
                raise IndexError(
                    f"mmd_layer_idx={layer_idx} is out of range for features length {len(syn_features)}"
                )
            layer_loss = criterions["mmd"](syn_features[layer_idx], real_features[layer_idx])
            loss_mmd = loss_mmd + float(layer_weight) * layer_loss

    loss_entropy = loss_seg.new_tensor(0.0)
    if cfg.entropy_weight > 0 and "real_logits" in out:
        loss_entropy = compute_entropy_minimization_loss(out["real_logits"])

    loss_prior = loss_seg.new_tensor(0.0)
    if cfg.prior_weight > 0 and "real_logits" in out:
        loss_prior = compute_class_ratio_prior_loss(
            out["real_logits"],
            syn_label,
            num_classes=cfg.num_classes,
            include_background=cfg.prior_include_background,
            smoothing=cfg.prior_smoothing,
        )

    total = cfg.seg_weight * loss_seg + loss_mmd + cfg.entropy_weight * loss_entropy + cfg.prior_weight * loss_prior
    return {
        "loss": total,
        "loss_ce": loss_ce,
        "loss_dice": loss_dice,
        "loss_seg": loss_seg,
        "loss_mmd": loss_mmd,
        "loss_entropy": loss_entropy,
        "loss_prior": loss_prior,
        "effective_mmd_weight": total.new_tensor(float(sum(effective_mmd_weights))),
    }


def save_loss_plot(history: dict[str, list[float]], save_dir: str, overwrite: bool = True):
    """"""

    if plt is None:
        return
    img_dir = os.path.join(save_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].plot(history["train_seg"], label="train_seg")
    if any(history["train_mmd"]):
        axes[0].plot(history["train_mmd"], label="train_mmd")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(history["train_syn_dice"], label="train_syn_dice")
    axes[1].plot(history["val_real_dice"], label="val_real_dice")
    if history["test_real_dice"]:
        axes[1].plot(history["test_real_dice"], label="test_real_dice")
    axes[1].set_title("Dice")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    axes[2].plot(history["train_attn_gate"], label="train_attn_gate")
    axes[2].plot(history["train_syn_attn_delta_norm"], label="syn_delta")
    axes[2].plot(history["train_real_attn_delta_norm"], label="real_delta")
    axes[2].plot(history["train_atlas_prior_token_norm"], label="token_norm")
    axes[2].plot(history["effective_mmd_weight"], label="mmd_weight")
    axes[2].set_title("Attention")
    axes[2].grid(alpha=0.2)
    axes[2].legend()

    fig.tight_layout()
    if overwrite:
        file_name = "seg_latest.png"
    else:
        file_name = f"seg_{time.strftime('%Y-%m-%d-%H-%M', time.localtime())}.png"
    fig.savefig(os.path.join(img_dir, file_name), dpi=150)
    plt.close(fig)


def run_auto_inference(cfg: Config):
    """ `inference_seg.py`  val/test """

    infer_cfg = seg_infer.Config(
        model_dir=cfg.save_dir,
        checkpoint_name=cfg.auto_infer_checkpoint_name,
        export_val=cfg.auto_infer_export_val,
        export_test=cfg.auto_infer_export_test,
        compute_hd95=cfg.auto_infer_compute_hd95,
        save_pred_label=cfg.auto_infer_save_pred_label,
        downsample_factor=cfg.downsample_factor,
        batch_size=1,
        num_workers=cfg.num_workers,
        bcv_atlas_id=cfg.bcv_atlas_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(
        "Auto inference | "
        f"ckpt={infer_cfg.checkpoint_name} | "
        f"export_val={infer_cfg.export_val} | "
        f"export_test={infer_cfg.export_test} | "
        f"compute_hd95={infer_cfg.compute_hd95} | "
        f"save_pred_label={infer_cfg.save_pred_label} | "
        f"downsample_factor={infer_cfg.downsample_factor}"
    )
    seg_infer.main(infer_cfg)


def main(cfg: Config):
    """


    1.
    2.  atlas cache
    3.  train/val/test DataLoader
    4. epoch  val  test
    5.  best loss / best dice  checkpoint
    6.  `epochs=0` test
    """

    cfg.validate()
    cfg.makedirs()
    cfg.save_json()
    seed_worker_fn, data_loader_generator = cfg.set_seed()
    sys.stdout = utils.Logger(cfg.save_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("OPAL-Net segmentation training requires CUDA")
    device = "cuda"
    amp_enabled = cfg.amp_enabled

    #  seg atlas token path
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
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_set, val_set, test_set = build_base_datasets(cfg)
    atlas_image_cpu = None
    atlas_label_cpu = None
    # atlas token path  atlas image/label
    # //
    if cfg.enable_atlas_token_path:
        atlas_image_cpu, atlas_label_cpu = build_atlas_cache_from_sample(val_set[0])

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker_fn,
        generator=data_loader_generator,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    criterions = {
        "ce": losses.WeightedCrossEntropyLoss(),
        "dice": losses.DiceLoss(),
        "mmd": losses.MMDLoss(kernel_alphas=(0.1, 1.0, 10.0)),
    }
    items = [
        "train_loss",
        "val_loss",
        "train_seg",
        "train_mmd",
        "train_entropy",
        "train_prior",
        "train_ce",
        "train_dice_loss",
        "train_syn_dice",
        "val_real_ce",
        "val_real_dice_loss",
        "val_real_dice",
        "test_real_dice",
        "train_attn_gate",
        "train_syn_attn_delta_norm",
        "train_real_attn_delta_norm",
        "train_atlas_prior_token_norm",
        "effective_mmd_weight",
    ]
    history = {k: [] for k in items}

    ckpt_manager = utils.CheckpointManager(cfg.save_dir, amp_enabled)
    state = ckpt_manager.load_resume(
        continue_train=cfg.continue_train,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        defaults={
            "start_epoch": cfg.epoch_start,
            "best_dice": 0.0,
            "best_eval_loss": float("inf"),
            "wait": 0,
            "pth_best_loss": "",
            "pth_best_dice": "",
            "fig": history,
        },
        items=items,
    )
    start_epoch = state["start_epoch"]
    best_dice = state["best_dice"]
    best_eval_loss = state["best_eval_loss"]
    wait = state["wait"]
    pth_best_loss = state["pth_best_loss"]
    pth_best_dice = state["pth_best_dice"]
    history = state["fig"]

    print(f"Train cases: {len(train_set)} | Val cases: {len(val_set)} | Test cases: {len(test_set)}")
    print(
        "==================================================\n"
        f"  Using dataset_key={cfg.dataset_key} | syn_key={cfg.syn_key}\n"
        f"  enable_atlas_token_path = {cfg.enable_atlas_token_path}\n"
        f"  share_atlas_encoder_with_seg = {cfg.share_atlas_encoder_with_seg}\n"
        f"  atlas_shape = {None if atlas_image_cpu is None else tuple(atlas_image_cpu.shape[2:])}\n"
        f"  atlas_label_shape = {None if atlas_label_cpu is None else tuple(atlas_label_cpu.shape[2:])}\n"
        f"  context_query_tokens = {cfg.context_query_tokens}\n"
        f"  context_shuffle_factor = {cfg.context_shuffle_factor}\n"
        f"  bottleneck_ctx_residual_weights = {cfg.bottleneck_ctx_residual_weights}\n"
        f"  downsample_factor = {cfg.downsample_factor}\n"
        f"  entropy_weight = {cfg.entropy_weight}\n"
        f"  prior_weight = {cfg.prior_weight}\n"
        f"  prior_include_background = {cfg.prior_include_background}\n"
        f"  mmd_layers = {cfg.mmd_layer_idx}\n"
        f"  mmd_weights = {cfg.mmd_weight}\n"
        "==================================================\n"
    )

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        utils.adjust_learning_rate(optimizer, epoch, cfg.epochs, cfg.lr)

        train_loss_meter = utils.AverageMeter()
        train_seg_meter = utils.AverageMeter()
        train_mmd_meter = utils.AverageMeter()
        train_entropy_meter = utils.AverageMeter()
        train_prior_meter = utils.AverageMeter()
        train_ce_meter = utils.AverageMeter()
        train_dice_loss_meter = utils.AverageMeter()
        train_syn_dice_meter = utils.AverageMeter()
        train_attn_gate_meter = utils.AverageMeter()
        train_syn_delta_meter = utils.AverageMeter()
        train_real_delta_meter = utils.AverageMeter()
        train_token_norm_meter = utils.AverageMeter()

        effective_mmd_weights = get_effective_mmd_weights(cfg, epoch)
        effective_mmd_weight_sum = float(sum(effective_mmd_weights))
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False)
        for batch in pbar:
            # train  syn  MMD// real
            real_image = batch[0].to(device)
            syn_image = batch[2].to(device)
            syn_label = batch[3].to(device)
            real_image, _ = downsample_image_label(real_image, None, cfg.downsample_factor)
            syn_image, syn_label = downsample_image_label(syn_image, syn_label, cfg.downsample_factor)
            if cfg.enable_atlas_token_path:
                atlas_img, atlas_label = get_atlas_batch(atlas_image_cpu, atlas_label_cpu, real_image.shape[2:], device)
            else:
                atlas_img, atlas_label = None, None
            need_real_branch = effective_mmd_weight_sum > 0 or cfg.entropy_weight > 0 or cfg.prior_weight > 0
            model_real_input = real_image if need_real_branch else None

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(
                    x_syn=syn_image,
                    x_real=model_real_input,
                    atlas_img=atlas_img,
                    atlas_label=atlas_label,
                    is_training=True,
                )
                loss_dict = compute_losses_seg(out, syn_label, criterions, cfg, effective_mmd_weights)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            syn_logits = out["syn_logits"]
            syn_pred = torch.argmax(syn_logits, dim=1, keepdim=True)
            syn_dice = utils.dice_val(syn_pred.long(), syn_label.long(), cfg.num_classes)
            batch_weight = real_image.shape[0]

            train_loss_meter.update(loss.item(), batch_weight)
            train_seg_meter.update(loss_dict["loss_seg"].item(), batch_weight)
            train_mmd_meter.update(loss_dict["loss_mmd"].item(), batch_weight)
            train_entropy_meter.update(loss_dict["loss_entropy"].item(), batch_weight)
            train_prior_meter.update(loss_dict["loss_prior"].item(), batch_weight)
            train_ce_meter.update(loss_dict["loss_ce"].item(), batch_weight)
            train_dice_loss_meter.update(loss_dict["loss_dice"].item(), batch_weight)
            train_syn_dice_meter.update(syn_dice.item(), batch_weight)
            train_attn_gate_meter.update(out["attn_gate"].item(), batch_weight)
            train_syn_delta_meter.update(out["syn_attn_delta_norm"].item(), batch_weight)
            if "real_attn_delta_norm" in out:
                train_real_delta_meter.update(out["real_attn_delta_norm"].item(), batch_weight)
            else:
                train_real_delta_meter.update(0.0, batch_weight)
            train_token_norm_meter.update(out["atlas_prior_token_norm"].item(), batch_weight)

            pbar.set_postfix(
                loss=f"{train_loss_meter.avg:.4f}",
                seg=f"{train_seg_meter.avg:.4f}",
                mmd=f"{train_mmd_meter.avg:.4f}",
                ent=f"{train_entropy_meter.avg:.4f}",
                prior=f"{train_prior_meter.avg:.4f}",
                syn_dice=f"{train_syn_dice_meter.avg:.4f}",
                gate=f"{train_attn_gate_meter.avg:.4f}",
            )

        history["train_loss"].append(train_loss_meter.avg)
        history["train_seg"].append(train_seg_meter.avg)
        history["train_mmd"].append(train_mmd_meter.avg)
        history["train_entropy"].append(train_entropy_meter.avg)
        history["train_prior"].append(train_prior_meter.avg)
        history["train_ce"].append(train_ce_meter.avg)
        history["train_dice_loss"].append(train_dice_loss_meter.avg)
        history["train_syn_dice"].append(train_syn_dice_meter.avg)
        history["train_attn_gate"].append(train_attn_gate_meter.avg)
        history["train_syn_attn_delta_norm"].append(train_syn_delta_meter.avg)
        history["train_real_attn_delta_norm"].append(train_real_delta_meter.avg)
        history["train_atlas_prior_token_norm"].append(train_token_norm_meter.avg)
        history["effective_mmd_weight"].append(effective_mmd_weight_sum)

        #  epoch  val split  best checkpoint
        model.eval()
        val_loss_meter = utils.AverageMeter()
        val_ce_meter = utils.AverageMeter()
        val_dice_loss_meter = utils.AverageMeter()
        val_real_dice_meter = utils.AverageMeter()

        with torch.inference_mode():
            # val  checkpoint
            pbar_val = tqdm(val_loader, total=len(val_loader), desc=f"Val {epoch + 1}/{cfg.epochs}", leave=False)
            for batch in pbar_val:
                real_image = batch[1].to(device)
                real_label = batch[3].to(device)
                real_image, real_label = downsample_image_label(real_image, real_label, cfg.downsample_factor)
                if cfg.enable_atlas_token_path:
                    atlas_img, atlas_label = get_atlas_batch(atlas_image_cpu, atlas_label_cpu, real_image.shape[2:], device)
                else:
                    atlas_img, atlas_label = None, None

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    out = model(x_real=real_image, atlas_img=atlas_img, atlas_label=atlas_label, is_training=False)
                    real_logits = out["real_logits"]
                    loss_ce = criterions["ce"](real_logits, real_label)
                    loss_dice = criterions["dice"](real_logits, real_label)
                    val_loss = cfg.ce_weight * loss_ce + cfg.dice_weight * loss_dice

                pred = torch.argmax(real_logits, dim=1, keepdim=True)
                real_dice = utils.dice_val(pred.long(), real_label.long(), cfg.num_classes)
                batch_weight = real_image.shape[0]
                val_loss_meter.update(val_loss.item(), batch_weight)
                val_ce_meter.update(loss_ce.item(), batch_weight)
                val_dice_loss_meter.update(loss_dice.item(), batch_weight)
                val_real_dice_meter.update(real_dice.item(), batch_weight)

        history["val_loss"].append(val_loss_meter.avg)
        history["val_real_ce"].append(val_ce_meter.avg)
        history["val_real_dice_loss"].append(val_dice_loss_meter.avg)
        history["val_real_dice"].append(val_real_dice_meter.avg)

        current_val_loss = val_loss_meter.avg
        current_val_dice = val_real_dice_meter.avg
        improved_best_loss = current_val_loss < best_eval_loss
        improved_best_dice = current_val_dice > (best_dice + cfg.early_stop_min_delta)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_loss_meter.avg:.4f} | train_mmd={train_mmd_meter.avg:.4f} | "
            f"train_ent={train_entropy_meter.avg:.4f} | train_prior={train_prior_meter.avg:.4f} | train_syn_dice={train_syn_dice_meter.avg:.4f} | val_loss={current_val_loss:.4f} | "
            f"val_real_dice={current_val_dice:.4f} | attn_gate={train_attn_gate_meter.avg:.4f}"
        )

        if improved_best_loss:
            best_eval_loss = current_val_loss
            prev = pth_best_loss
            file_name = f"best_loss_ep{epoch + 1:03d}_loss{current_val_loss:.4f}.pth"
            pth_best_loss = ckpt_manager.save(
                file_name=file_name,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                best_dice=best_dice,
                best_eval_loss=best_eval_loss,
                pth_best_loss=pth_best_loss,
                pth_best_dice=pth_best_dice,
                wait=wait,
                history=history,
            )
            if prev and os.path.exists(prev) and prev != pth_best_loss:
                os.remove(prev)

        if improved_best_dice:
            best_dice = current_val_dice
            prev = pth_best_dice
            file_name = f"best_real_dice_ep{epoch + 1:03d}_dice{current_val_dice:.4f}.pth"
            pth_best_dice = ckpt_manager.save(
                file_name=file_name,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                best_dice=best_dice,
                best_eval_loss=best_eval_loss,
                pth_best_loss=pth_best_loss,
                pth_best_dice=pth_best_dice,
                wait=0,
                history=history,
            )
            if prev and os.path.exists(prev) and prev != pth_best_dice:
                os.remove(prev)

        if improved_best_loss or improved_best_dice:
            wait = 0
        else:
            wait += 1
            print(f"EarlyStopping counter: {wait} / {cfg.patience}")

        if (epoch + 1) % cfg.save_every == 0:
            ckpt_manager.save(
                file_name=f"periodic_ep{epoch + 1:03d}.pth",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                best_dice=best_dice,
                best_eval_loss=best_eval_loss,
                pth_best_loss=pth_best_loss,
                pth_best_dice=pth_best_dice,
                wait=wait,
                history=history,
            )

        ckpt_manager.save(
            file_name="resume_latest.pth",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch + 1,
            best_dice=best_dice,
            best_eval_loss=best_eval_loss,
            pth_best_loss=pth_best_loss,
            pth_best_dice=pth_best_dice,
            wait=wait,
            history=history,
        )
        save_loss_plot(history, cfg.save_dir, overwrite=cfg.plot_overwrite)

        if wait >= cfg.patience:
            print("Early stopping triggered.")
            break

    model.eval()
    test_real_dice_meter = utils.AverageMeter()
    with torch.inference_mode():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Test", leave=False):
            real_image = batch[1].to(device)
            real_label = batch[3].to(device)
            real_image, real_label = downsample_image_label(real_image, real_label, cfg.downsample_factor)
            if cfg.enable_atlas_token_path:
                atlas_img, atlas_label = get_atlas_batch(atlas_image_cpu, atlas_label_cpu, real_image.shape[2:], device)
            else:
                atlas_img, atlas_label = None, None
            out = model(x_real=real_image, atlas_img=atlas_img, atlas_label=atlas_label, is_training=False)
            real_logits = out["real_logits"]
            pred = torch.argmax(real_logits, dim=1, keepdim=True)
            real_dice = utils.dice_val(pred.long(), real_label.long(), cfg.num_classes)
            test_real_dice_meter.update(real_dice.item(), real_image.shape[0])

    history["test_real_dice"].append(test_real_dice_meter.avg)
    save_loss_plot(history, cfg.save_dir, overwrite=cfg.plot_overwrite)
    summary = {
        "syn_key": cfg.syn_key,
        "enable_atlas_token_path": cfg.enable_atlas_token_path,
        "share_atlas_encoder_with_seg": cfg.share_atlas_encoder_with_seg,
        "atlas_shape": None if atlas_image_cpu is None else list(atlas_image_cpu.shape[2:]),
        "downsample_factor": cfg.downsample_factor,
        "atlas_label_shape": None if atlas_label_cpu is None else list(atlas_label_cpu.shape[2:]),
        "context_query_tokens": cfg.context_query_tokens,
        "context_shuffle_factor": cfg.context_shuffle_factor,
        "bottleneck_ctx_residual_weights": cfg.bottleneck_ctx_residual_weights,
        "best_val_real_dice": best_dice,
        "best_val_loss": best_eval_loss,
        "test_real_dice": test_real_dice_meter.avg,
        "epochs_recorded": len(history["train_loss"]),
    }
    with open(os.path.join(cfg.save_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"test | real_dice={test_real_dice_meter.avg:.4f}")

    if cfg.auto_infer_after_train:
        run_auto_inference(cfg)


if __name__ == "__main__":
    main(Config.from_env())
