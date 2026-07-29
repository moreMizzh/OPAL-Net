"""Train the OPAL-Net cascaded registration model.

The pipeline predicts a coarse deformation, refines it with an ROI-conditioned
stage, and exports final warped images, labels, and flows.
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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dataset as dataset_lib
import inference_reg as reg_infer
import losses as losses_lib
import utils as utils_lib



import model_reg as reg_model_lib


#
#  `dataset_key`  `source_root_dir / exp_name / imgsize / num_classes`
#
DATASET_INFO = {
    "ADNI": {
        "source_root_dir": "../data/ADNI_Dataset",
        "exp_name": "ADNI_cascaded_reg_downsample1x",
        "imgsize": (160, 160, 160),
        "num_classes": 4,
    },
    "BCV": {
        "source_root_dir": "../data/BCV/bcv_small",
        "exp_name": "BCV_cascaded_reg_4x",
        "imgsize": (128, 128, 160),
        "num_classes": 5,
        "epochs": 400,
    },
    "OASIS": {
        "source_root_dir": "../data/OASIS/OASIS",
        "exp_name": "OASIS_cascaded_reg_downsample1x",
        "imgsize": (224, 192, 160),
        "num_classes": 36,
    },
}


def apply_dataset_preset(cfg, dataset_key: str | None):
    """"""

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
    """Cascaded registration experiment configuration.


    - `save_root_dir` `save_dir`
    - `dataset_key`
    - `imgsize / int_steps / enable_stage_b / roi_*`

    """

    save_dir: str = ""
    version: str = "cascaded_reg"
    save_root_dir: str = "output/Reg"
    continue_train: bool = False

    dataset_key: str = "BCV"
    source_root_dir: str = ""
    exp_name: str = ""
    imgsize: tuple[int, int, int] = (0, 0, 0)
    num_classes: int = 0
    bcv_atlas_id: int | None = None  # Optional BCV atlas ID; defaults to 27 in the dataset loader.

    # ==============================
    #
    # ==============================
    batch_size: int = 1  # 3D reg  case
    epochs: int = 200  #
    epoch_start: int = 0  #  checkpoint
    save_every: int = 20  #
    patience: int = 30  #  Dice / loss
    num_workers: int = 0  #  3D IO

    # ==============================
    #
    # ==============================
    lr: float = 1.5e-4  # Adam
    weight_decay: float = 0.0  # reg  L2
    int_steps: int = 7  # SVF -> flow  scaling-and-squaring
    enable_stage_b: bool = True  # False  A
    enable_roi_b: bool = True  # Enable the ROI refinement stage.
    roi_classes: tuple[int, ...] = ()   # ROI  foreground
    stage_a_downsample_factor: int = 4  # Coarse-stage downsampling factor.
    roi_dilation_radius: int = 3        # moving-side ROI
    roi_condition_channels: int = 3     # soft ROI / boundary / fixed edge
    detach_stage_a_for_b: bool = False  # True  B  A
    seed: int = 42  #  cudnn

    amp_enabled: bool = True  #  3D
    max_grad_norm: float = 3.0  #  case

    # ==============================
    #
    # ==============================
    coarse_ncc_weight: float = 1.0  # A  NCC
    coarse_smooth_weight: float = 1.0  # A  flow smoothness
    coarse_jac_weight: float = 0.00  # A  Jacobian penalty
    refine_ncc_weight: float = 1.0  # B  NCC
    refine_smooth_weight: float = 1.0 #  0.01  # B  residual smoothness
    refine_jac_weight: float = 0.00  # B  Jacobian penalty
    plot_overwrite: bool = True  # True
    early_stop_min_delta: float = 1e-4  # Dice

    # ==============================
    #
    # ==============================
    auto_infer_after_train: bool = True  # Run registration inference after training.
    auto_infer_save_train_outputs: bool = True  #  train split / imagelabelflow
    auto_infer_compute_test_metrics: bool = True  #  test split  test
    auto_infer_compute_hd95: bool = True  # test  HD95
    auto_infer_compute_folding_rate: bool = True  # test  Jacobian
    auto_infer_checkpoint_name: str = "best_dice"  #

    @classmethod
    def from_env(cls):
        """

         `dataset_key -> DATASET_INFO`
         `OPAL_REG_*`
        """

        cfg = cls()

        # `imgsize`  parser
        def _parse_imgsize(raw: str) -> tuple[int, int, int]:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) != 3:
                raise ValueError(f"OPAL_REG_IMGSIZE expects 'D,H,W', got: {raw}")
            return int(parts[0]), int(parts[1]), int(parts[2])

        def _parse_int_tuple(raw: str) -> tuple[int, ...]:
            if raw.strip() == "":
                return ()
            return tuple(int(p.strip()) for p in raw.split(",") if p.strip() != "")

        #  `OPAL_REG_* -> Config `
        #  shell
        env_map = {
            "OPAL_REG_SAVE_DIR": ("save_dir", str),
            "OPAL_REG_VERSION": ("version", str),
            "OPAL_REG_SOURCE_ROOT_DIR": ("source_root_dir", str),
            "OPAL_REG_SAVE_ROOT_DIR": ("save_root_dir", str),
            "OPAL_REG_EXP_NAME": ("exp_name", str),
            "OPAL_REG_CONTINUE_TRAIN": ("continue_train", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_IMGSIZE": ("imgsize", _parse_imgsize),
            "OPAL_REG_NUM_CLASSES": ("num_classes", int),
            "OPAL_REG_BCV_ATLAS_ID": ("bcv_atlas_id", int),
            "OPAL_REG_BATCH_SIZE": ("batch_size", int),
            "OPAL_REG_EPOCHS": ("epochs", int),
            "OPAL_REG_PATIENCE": ("patience", int),
            "OPAL_REG_NUM_WORKERS": ("num_workers", int),
            "OPAL_REG_LR": ("lr", float),
            "OPAL_REG_WEIGHT_DECAY": ("weight_decay", float),
            "OPAL_REG_INT_STEPS": ("int_steps", int),
            "OPAL_REG_ENABLE_STAGE_B": ("enable_stage_b", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_ENABLE_ROI_B": ("enable_roi_b", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_ROI_CLASSES": ("roi_classes", _parse_int_tuple),
            "OPAL_REG_STAGE_A_DOWNSAMPLE_FACTOR": ("stage_a_downsample_factor", int),
            "OPAL_REG_ROI_DILATION_RADIUS": ("roi_dilation_radius", int),
            "OPAL_REG_ROI_CONDITION_CHANNELS": ("roi_condition_channels", int),
            "OPAL_REG_DETACH_STAGE_A_FOR_B": ("detach_stage_a_for_b", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_SEED": ("seed", int),
            "OPAL_REG_AMP_ENABLED": ("amp_enabled", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_MAX_GRAD_NORM": ("max_grad_norm", float),
            "OPAL_REG_COARSE_NCC_WEIGHT": ("coarse_ncc_weight", float),
            "OPAL_REG_REFINE_NCC_WEIGHT": ("refine_ncc_weight", float),
            "OPAL_REG_PLOT_OVERWRITE": ("plot_overwrite", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_EARLY_STOP_MIN_DELTA": ("early_stop_min_delta", float),
            "OPAL_REG_AUTO_INFER_AFTER_TRAIN": ("auto_infer_after_train", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_AUTO_INFER_SAVE_TRAIN_OUTPUTS": (
                "auto_infer_save_train_outputs",
                lambda v: v.lower() in {"1", "true", "yes", "on"},
            ),
            "OPAL_REG_AUTO_INFER_COMPUTE_TEST_METRICS": (
                "auto_infer_compute_test_metrics",
                lambda v: v.lower() in {"1", "true", "yes", "on"},
            ),
            "OPAL_REG_AUTO_INFER_COMPUTE_HD95": ("auto_infer_compute_hd95", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_AUTO_INFER_COMPUTE_FOLDING_RATE": (
                "auto_infer_compute_folding_rate",
                lambda v: v.lower() in {"1", "true", "yes", "on"},
            ),
            "OPAL_REG_AUTO_INFER_CHECKPOINT": ("auto_infer_checkpoint_name", str),
        }
        # `dataset_key`
        raw_dataset_key = os.environ.get("OPAL_REG_DATASET_KEY")
        if raw_dataset_key is not None and raw_dataset_key != "":
            object.__setattr__(cfg, "dataset_key", raw_dataset_key)
        cfg = apply_dataset_preset(cfg, cfg.dataset_key)

        updates: dict[str, object] = {}
        for env_key, (field_name, caster) in env_map.items():
            env_value = os.getenv(env_key)
            if env_value is None or env_value == "":
                continue
            updates[field_name] = caster(env_value)
        if updates:
            merged = cls(**{**asdict(cfg), **updates})
            return merged
        return cfg

    def makedirs(self):
        """

         +
        """

        save_dir = self.save_dir
        if not save_dir:
            #
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(self.save_root_dir, f"{self.exp_name}_{ts}")
            object.__setattr__(self, "save_dir", save_dir)
        os.makedirs(save_dir, exist_ok=True)

    def save_json(self, file_name: str = "config.json"):
        """"""

        with open(os.path.join(self.save_dir, file_name), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

    def set_seed(self):
        """ Python / NumPy / Torch  DataLoader """

        # reg  Python / NumPy / Torch
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


class BCVTrainPlusValRegDataset(dataset_lib.BCV_Dataset):
    """BCV reg train=train+valeval=test"""

    def _resolve_target_ids(self, is_training: bool, is_val: bool) -> list[int]:
        if is_training or is_val:
            train_pool_ids = list(dataset_lib._bcv_train_pool_ids(self.atlas_id))
            return train_pool_ids + [37, 38, 39, 40]
        return list(range(61, 81))


def build_dataset(cfg: Config):
    """Build ADNI, BCV, or OASIS registration datasets."""

    source_root = os.path.abspath(cfg.source_root_dir)
    root_name = os.path.basename(source_root)

    if "ADNI" in root_name:
        return (
            dataset_lib.ADNI_Split_Dataset(cfg.source_root_dir, is_training=True),
            dataset_lib.ADNI_Split_Dataset(cfg.source_root_dir, is_training=False, is_val=True),
        )
    if "BCV" in root_name or "bcv_small" in root_name.lower() or "BCV" in cfg.exp_name:
        return (
            BCVTrainPlusValRegDataset(cfg.source_root_dir, is_training=True, atlas_id=cfg.bcv_atlas_id),
            BCVTrainPlusValRegDataset(cfg.source_root_dir, is_training=False, is_val=False, atlas_id=cfg.bcv_atlas_id),
        )
    if "OASIS" in root_name:
        return (
            dataset_lib.OASIS_Dataset(cfg.source_root_dir, is_training=True),
            dataset_lib.OASIS_Dataset(cfg.source_root_dir, is_training=False, is_val=True),
        )
    raise ValueError(
        f"Unsupported source_root_dir for OPAL-Net release builder: {cfg.source_root_dir}. "
        "Supported datasets are ADNI, BCV, and OASIS."
    )


def save_loss_plot(history: dict[str, list[float]], save_dir: str, epoch: int, overwrite: bool = True):
    """ v11 """

    save_img_dir = os.path.join(save_dir, "img")
    os.makedirs(save_img_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.ravel()

    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_title("Total Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(history["train_coarse_loss"], label="train_s1")
    axes[1].plot(history["val_coarse_loss"], label="val_s1")
    axes[1].plot(history["train_refine_loss"], label="train_s2")
    axes[1].plot(history["val_refine_loss"], label="val_s2")
    axes[1].set_title("Stage Loss")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    axes[2].plot(history["train_refine_smooth"], label="train_smooth")
    axes[2].plot(history["val_refine_smooth"], label="val_smooth")
    axes[2].plot(history["train_refine_jac"], label="train_jac")
    axes[2].plot(history["val_refine_jac"], label="val_jac")
    axes[2].set_title("ROI-B Regularization")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)

    axes[3].plot(history["val_coarse_dice"], label="val_coarse_dice")
    axes[3].plot(history["val_final_dice"], label="val_final_dice")
    axes[3].set_title("Dice")
    axes[3].grid(alpha=0.2)
    axes[3].legend()

    fig.tight_layout()
    fig_name = "reg_metrics_latest.png" if overwrite else f"reg_metrics_{epoch + 1:03d}_{time.strftime('%Y-%m-%d-%H-%M', time.localtime())}.png"
    fig.savefig(os.path.join(save_img_dir, fig_name), dpi=150)
    plt.close(fig)


def compute_losses(
    out: dict[str, torch.Tensor],
    target_img: torch.Tensor,
    criterions: dict[str, nn.Module],
    cfg: Config,
) -> dict[str, torch.Tensor]:
    """ v11 """

    loss_coarse_recon = criterions["ncc"](out["fixed_down"], out["coarse_warped_lowres"])
    loss_coarse_smooth = criterions["smooth"](out["coarse_svf_lowres"])
    loss_coarse_jac = criterions["jac"](out["coarse_flow_lowres"])
    loss_coarse = (
        cfg.coarse_ncc_weight * loss_coarse_recon
        + cfg.coarse_smooth_weight * loss_coarse_smooth
        + cfg.coarse_jac_weight * loss_coarse_jac
    )

    zero = loss_coarse_recon.new_tensor(0.0)
    if not cfg.enable_stage_b:
        return {
            "loss": loss_coarse,
            "loss_coarse": loss_coarse,
            "loss_refine": zero,
            "loss_coarse_recon": loss_coarse_recon,
            "loss_coarse_smooth": loss_coarse_smooth,
            "loss_coarse_jac": loss_coarse_jac,
            "loss_refine_recon": zero,
            "loss_refine_smooth": zero,
            "loss_refine_jac": zero,
        }

    loss_refine_recon = criterions["ncc"](target_img, out["final_warped"])
    loss_refine_smooth = criterions["smooth"](out["residual_svf"])
    loss_refine_jac = criterions["jac"](out["final_flow"])
    loss_refine = (
        cfg.refine_ncc_weight * loss_refine_recon
        + cfg.refine_smooth_weight * loss_refine_smooth
        + cfg.refine_jac_weight * loss_refine_jac
    )
    total_loss = loss_coarse + loss_refine
    return {
        "loss": total_loss,
        "loss_coarse": loss_coarse,
        "loss_refine": loss_refine,
        "loss_coarse_recon": loss_coarse_recon,
        "loss_coarse_smooth": loss_coarse_smooth,
        "loss_coarse_jac": loss_coarse_jac,
        "loss_refine_recon": loss_refine_recon,
        "loss_refine_smooth": loss_refine_smooth,
        "loss_refine_jac": loss_refine_jac,
    }


HISTORY_ITEMS = [
    "train_loss",
    "train_coarse_loss",
    "train_refine_loss",
    "train_coarse_recon",
    "train_coarse_smooth",
    "train_coarse_jac",
    "train_refine_recon",
    "train_refine_smooth",
    "train_refine_jac",
    "val_loss",
    "val_coarse_loss",
    "val_refine_loss",
    "val_coarse_recon",
    "val_coarse_smooth",
    "val_coarse_jac",
    "val_coarse_dice",
    "val_refine_recon",
    "val_refine_smooth",
    "val_refine_jac",
    "val_final_dice",
]

LOSS_KEY_MAP = {
    "loss": "loss",
    "coarse_loss": "loss_coarse",
    "refine_loss": "loss_refine",
    "coarse_recon": "loss_coarse_recon",
    "coarse_smooth": "loss_coarse_smooth",
    "coarse_jac": "loss_coarse_jac",
    "refine_recon": "loss_refine_recon",
    "refine_smooth": "loss_refine_smooth",
    "refine_jac": "loss_refine_jac",
}


def build_dataloaders(cfg: Config, seed_worker_fn, data_loader_generator):
    """ train/val  DataLoader"""

    train_set, val_set = build_dataset(cfg)
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        worker_init_fn=seed_worker_fn,
        generator=data_loader_generator,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    return train_set, val_set, train_loader, val_loader


def build_training_bundle(cfg: Config, device: str):
    """ reg  bundle


    - bundle:  checkpoint
    - reg_model:
    """

    bundle = reg_model_lib.build_cascaded_reg_model_bundle(
        inshape=cfg.imgsize,
        int_steps=cfg.int_steps,
        enable_stage_b=cfg.enable_stage_b,
        enable_roi_b=cfg.enable_roi_b,
        roi_classes=cfg.roi_classes,
        stage_a_downsample_factor=cfg.stage_a_downsample_factor,
        roi_dilation_radius=cfg.roi_dilation_radius,
        roi_condition_channels=cfg.roi_condition_channels,
    ).to(device)
    return bundle, bundle["reg_model"]


def build_criterions():
    """"""

    return {
        "ncc": losses_lib.NCC(win=9),
        "smooth": losses_lib.FlowGradLoss(),
        "jac": losses_lib.JacobianDeterminantLoss(),
    }


def create_meters():
    return {k: utils_lib.AverageMeter() for k in HISTORY_ITEMS}


def update_meters(
    meters: dict[str, utils_lib.AverageMeter],
    prefix: str,
    loss_dict: dict[str, torch.Tensor],
    batch_weight: int,
    coarse_dice: torch.Tensor | None = None,
    final_dice: torch.Tensor | None = None,
):
    for meter_suffix, loss_key in LOSS_KEY_MAP.items():
        meters[f"{prefix}_{meter_suffix}"].update(loss_dict[loss_key].item(), batch_weight)
    if coarse_dice is not None:
        meters[f"{prefix}_coarse_dice"].update(coarse_dice.item(), batch_weight)
    if final_dice is not None:
        meters[f"{prefix}_final_dice"].update(final_dice.item(), batch_weight)


def train_one_epoch(
    epoch: int,
    cfg: Config,
    bundle: nn.ModuleDict,
    reg_model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterions: dict[str, nn.Module],
    device: str,
):
    """


    -  coarse/refine
    -  loss
    -
    -  batch  epoch meter
    """

    meters = create_meters()
    reg_model.train()
    pbar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"train {epoch + 1}/{cfg.epochs}",
        file=getattr(sys.stdout, "terminal", sys.stdout),
        leave=False,
        position=0,
    )
    pbar_detail = tqdm(
        total=0,
        bar_format="{desc}",
        file=getattr(sys.stdout, "terminal", sys.stdout),
        leave=False,
        position=1,
    )
    for batch in pbar:
        moving = batch[0].to(device)
        fixed = batch[1].to(device)
        moving_label = batch[2].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg.amp_enabled):
            out = reg_model(
                moving=moving,
                fixed=fixed,
                moving_label=moving_label,
                detach_stage_a_for_b=cfg.detach_stage_a_for_b,
            )
            loss_dict = compute_losses(out, fixed, criterions, cfg)
        scaler.scale(loss_dict["loss"]).backward()
        scaler.unscale_(optimizer)
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(bundle.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        update_meters(meters, "train", loss_dict, moving.shape[0])
        pbar.set_postfix(
            s1=f"{meters['train_coarse_loss'].avg:.4f}",
            s2=f"{meters['train_refine_loss'].avg:.4f}",
        )
        pbar_detail.set_description_str(
            " "
            f"A[r/s/j]={meters['train_coarse_recon'].avg:.4f}/{meters['train_coarse_smooth'].avg:.4f}/{meters['train_coarse_jac'].avg:.4f} | "
            f"B[r/s/j]={meters['train_refine_recon'].avg:.4f}/{meters['train_refine_smooth'].avg:.4f}/{meters['train_refine_jac'].avg:.4f}"
        )
    pbar_detail.close()
    return meters


def validate_one_epoch(
    epoch: int,
    cfg: Config,
    reg_model: nn.Module,
    val_loader: DataLoader,
    criterions: dict[str, nn.Module],
    device: str,
):
    """


    -  loss
    -  coarse/final label  Dice
    -  checkpoint
    """

    meters = create_meters()
    reg_model.eval()
    with torch.inference_mode():
        pbar = tqdm(
            val_loader,
            total=len(val_loader),
            desc=f"val {epoch + 1}/{cfg.epochs}",
            file=getattr(sys.stdout, "terminal", sys.stdout),
            leave=False,
            position=0,
        )
        for batch in pbar:
            moving = batch[0].to(device)
            fixed = batch[1].to(device)
            moving_label = batch[2].to(device)
            fixed_label = batch[3].to(device)
            with torch.amp.autocast("cuda", enabled=cfg.amp_enabled):
                out = reg_model(
                    moving=moving,
                    fixed=fixed,
                    moving_label=moving_label,
                    detach_stage_a_for_b=cfg.detach_stage_a_for_b,
                )
                loss_dict = compute_losses(out, fixed, criterions, cfg)
            coarse_dice = utils_lib.dice_val(out["coarse_warped_label"].long(), fixed_label.long(), cfg.num_classes)
            final_dice = utils_lib.dice_val(out["final_warped_label"].long(), fixed_label.long(), cfg.num_classes)
            update_meters(meters, "val", loss_dict, moving.shape[0], coarse_dice=coarse_dice, final_dice=final_dice)
            pbar.set_postfix(
                s1=f"{meters['val_coarse_loss'].avg:.4f}",
                s2=f"{meters['val_refine_loss'].avg:.4f}",
                d=f"{meters['val_final_dice'].avg:.4f}",
            )
    return meters


def merge_epoch_meters(train_meters, val_meters):
    merged = create_meters()
    for key in HISTORY_ITEMS:
        source = train_meters if key.startswith("train_") else val_meters
        merged[key] = source[key]
    return merged


def append_history(history: dict[str, list[float]], meters: dict[str, utils_lib.AverageMeter]):
    for key in HISTORY_ITEMS:
        history[key].append(meters[key].avg)


def print_epoch_summary(epoch: int, cfg: Config, meters: dict[str, utils_lib.AverageMeter], best_dice: float):
    print(
        f"Epoch {epoch + 1}/{cfg.epochs} | "
        f"Train Loss: {meters['train_loss'].avg:.4f} | "
        f"Val Loss: {meters['val_loss'].avg:.4f} | "
        f"Val S1/S2 Dice: {meters['val_coarse_dice'].avg:.4f}/{meters['val_final_dice'].avg:.4f} | "
        f"Best Dice: {best_dice:.4f}"
    )
    print(
        f"  S1 Train/Val: {meters['train_coarse_loss'].avg:.4f}/{meters['val_coarse_loss'].avg:.4f} | "
        f"S2 Train/Val: {meters['train_refine_loss'].avg:.4f}/{meters['val_refine_loss'].avg:.4f}"
    )
    print(
        f"  S2 train[r/s/j]={meters['train_refine_recon'].avg:.4f}/{meters['train_refine_smooth'].avg:.4f}/{meters['train_refine_jac'].avg:.4f} | "
        f"S2 val[r/s/j]={meters['val_refine_recon'].avg:.4f}/{meters['val_refine_smooth'].avg:.4f}/{meters['val_refine_jac'].avg:.4f}"
    )
    print()


def save_checkpoint(
    ckpt_manager,
    file_name: str,
    bundle,
    optimizer,
    scaler,
    epoch: int,
    state: dict,
    history: dict[str, list[float]],
):
    return ckpt_manager.save(
        file_name=file_name,
        model=bundle,
        optimizer=optimizer,
        scaler=scaler,
        epoch=epoch,
        best_dice=state["best_dice"],
        best_eval_loss=state["best_eval_loss"],
        pth_best_loss=state["pth_best_loss"],
        pth_best_dice=state["pth_best_dice"],
        wait=state["wait"],
        history=history,
    )


def run_auto_inference(cfg: Config) -> None:
    """ reg inference"""

    if cfg.auto_infer_save_train_outputs:
        save_cfg = reg_infer.Config(
            model_dir=cfg.save_dir,
            checkpoint_name=cfg.auto_infer_checkpoint_name,
            export_train=True,
            export_val=False,
            export_test=False,
            compute_metrics=False,
            compute_hd95=False,
            compute_folding_rate=False,
            export_final=True,
            export_inputs=False,
            export_target_label=False,
            export_coarse=True,
            export_flow=True,
            bcv_atlas_id=cfg.bcv_atlas_id,
            device="cuda",
        )
        print(
            "Auto inference save | "
            f"ckpt={save_cfg.checkpoint_name} | "
            "split=train | compute_metrics=False | save coarse/final img+label+flow"
        )
        reg_infer.main(save_cfg)

    if cfg.auto_infer_compute_test_metrics:
        metric_cfg = reg_infer.Config(
            model_dir=cfg.save_dir,
            checkpoint_name=cfg.auto_infer_checkpoint_name,
            export_train=False,
            export_val=False,
            export_test=True,
            compute_metrics=True,
            compute_hd95=cfg.auto_infer_compute_hd95,
            compute_folding_rate=cfg.auto_infer_compute_folding_rate,
            export_final=True,
            export_inputs=False,
            export_target_label=False,
            export_coarse=True,
            export_flow=True,
            bcv_atlas_id=cfg.bcv_atlas_id,
            device="cuda",
        )
        print(
            "Auto inference metrics | "
            f"ckpt={metric_cfg.checkpoint_name} | "
            "split=test | "
            f"hd95={metric_cfg.compute_hd95} | "
            f"folding={metric_cfg.compute_folding_rate} | "
            "save_outputs=coarse/final img+label+flow"
        )
        reg_infer.main(metric_cfg)


def update_checkpoints(
    epoch: int,
    cfg: Config,
    meters: dict[str, utils_lib.AverageMeter],
    ckpt_manager,
    bundle,
    optimizer,
    scaler,
    state: dict,
    history: dict[str, list[float]],
):
    """ epoch  best-loss / best-dice / resume checkpoint


    - `best_loss`
    - `best_dice`
    - `resume_latest.pth`
    - `wait`  `patience`
    """

    current_eval_loss = meters["train_loss"].avg if cfg.dataset_key == "BCV" else meters["val_loss"].avg
    current_val_dice = meters["val_final_dice"].avg
    improved_best_dice = current_val_dice > (state["best_dice"] + cfg.early_stop_min_delta)
    improved_best_loss = current_eval_loss < state["best_eval_loss"]


    if improved_best_loss:
        state["best_eval_loss"] = current_eval_loss
        prev = state["pth_best_loss"]
        state["pth_best_loss"] = save_checkpoint(
            ckpt_manager,
            f"best_loss_ep{epoch + 1:03d}_loss{current_eval_loss:.4f}.pth",
            bundle,
            optimizer,
            scaler,
            epoch + 1,
            state,
            history,
        )
        if prev and os.path.exists(prev) and prev != state["pth_best_loss"]:
            os.remove(prev)

    if improved_best_dice:
        state["best_dice"] = current_val_dice
        prev = state["pth_best_dice"]
        state["pth_best_dice"] = save_checkpoint(
            ckpt_manager,
            f"best_dice_ep{epoch + 1:03d}_dice{current_val_dice:.4f}.pth",
            bundle,
            optimizer,
            scaler,
            epoch + 1,
            state,
            history,
        )
        if prev and os.path.exists(prev) and prev != state["pth_best_dice"]:
            os.remove(prev)
    if improved_best_loss or improved_best_dice:
        state["wait"] = 0
    else:
        state["wait"] += 1
        remaining_patience = max(cfg.patience - state["wait"], 0)
        print(f"  EarlyStop wait/patience: {state['wait']}/{cfg.patience} | remaining: {remaining_patience}")

    if (epoch + 1) % cfg.save_every == 0:
        save_checkpoint(ckpt_manager, f"epoch_{epoch + 1:03d}.pth", bundle, optimizer, scaler, epoch + 1, state, history)

    save_checkpoint(ckpt_manager, "resume_latest.pth", bundle, optimizer, scaler, epoch + 1, state, history)


def main(cfg: Config):
    """


    1.
    2.
    3.  CUDA
    4. loss  checkpoint
    5.  epoch /
    6.  best checkpoint
    7.  `auto_infer_after_train`
    """

    cfg.makedirs()
    cfg.save_json()
    seed_worker_fn, data_loader_generator = cfg.set_seed()
    sys.stdout = utils_lib.Logger(cfg.save_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("OPAL-Net reg training requires CUDA")
    device = "cuda"
    amp_enabled = cfg.amp_enabled

    bundle, reg_model = build_training_bundle(cfg, device)
    train_set, val_set, train_loader, val_loader = build_dataloaders(cfg, seed_worker_fn, data_loader_generator)
    if cfg.dataset_key == "BCV":
        print(f"Length of training/test set: {len(train_set)}/{len(val_set)}")
    else:
        print(f"Length of training/val set: {len(train_set)}/{len(val_set)}")
    coarse_stage = reg_model.module.coarse_stage if isinstance(reg_model, nn.DataParallel) else reg_model.coarse_stage
    print(
        "Reg930 setup | "
        f"stage_a_downsample_factor={cfg.stage_a_downsample_factor} | "
        f"stage_a_lowres_shape={coarse_stage.lowres_shape} | "
        f"enable_stage_b={cfg.enable_stage_b} | enable_roi_b={cfg.enable_roi_b}"
    )

    optimizer = optim.Adam(bundle.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterions = build_criterions()
    history = {k: [] for k in HISTORY_ITEMS}
    ckpt_manager = utils_lib.CheckpointManager(cfg.save_dir, amp_enabled)
    state = ckpt_manager.load_resume(
        continue_train=cfg.continue_train,
        model=bundle,
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
        items=HISTORY_ITEMS,
    )
    history = state["fig"]

    for epoch in range(state["start_epoch"], cfg.epochs):
        utils_lib.adjust_learning_rate(optimizer, epoch, cfg.epochs, cfg.lr)
        train_meters = train_one_epoch(
            epoch, cfg, bundle, reg_model, train_loader, optimizer, scaler, criterions, device
        )
        val_meters = validate_one_epoch(epoch, cfg, reg_model, val_loader, criterions, device)
        meters = merge_epoch_meters(train_meters, val_meters)
        append_history(history, meters)
        print_epoch_summary(epoch, cfg, meters, state["best_dice"])
        update_checkpoints(epoch, cfg, meters, ckpt_manager, bundle, optimizer, scaler, state, history)
        save_loss_plot(history, cfg.save_dir, epoch, overwrite=cfg.plot_overwrite)

        if state["wait"] >= cfg.patience:
            print(
                f"Early stop at epoch {epoch + 1}, "
                f"best_dice={state['best_dice']:.4f}, best_eval_loss={state['best_eval_loss']:.4f}"
            )
            break

    if cfg.auto_infer_after_train:
        run_auto_inference(cfg)


if __name__ == "__main__":
    main(Config.from_env())
