"""OPAL-Net 配准推理 / 导出入口脚本。

它既负责导出 warped image / label / flow，也负责按 split 计算
Dice、HD95 与 folding rate 等统计指标。
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dataset as dataset_lib



import model_reg as reg_model_lib


@dataclass(frozen=True)
class Config:
    """reg 推理配置，缺省项会尽量从训练目录回填。

    可把字段分成两类理解：
    - 训练结构绑定项：`imgsize / int_steps / enable_stage_b / roi_*`，默认必须和训练同构；
    - 当前推理行为项：`export_* / compute_* / checkpoint_name`，可按这次导出需求单独覆盖。
    """

    # ==============================
    # 权重与 split 选择
    # ==============================
    model_dir: str = "./output/Reg/BCV_cascaded_reg"  # 训练输出目录，默认从这里读取 config 和 checkpoint。
    checkpoint_name: str = "best_dice"  # 支持 best_dice / best_loss / resume_latest.pth / 显式文件名。

    export_train: bool = True  # 是否处理 train split。
    export_val: bool = False  # 是否处理 val split。
    export_test: bool = False  # 是否处理 test split。
    compute_metrics: bool = True  # False 时只做导出，不计算 Dice / HD95 / folding。
    compute_hd95: bool = False  # True 时额外计算 95% Hausdorff。
    compute_folding_rate: bool = True  # True 时计算负 Jacobian 占比。
    export_final: bool = True  # 保存最终 warped image / label。
    export_inputs: bool = False  # 保存原始 atlas / target 输入，主要用于排查数据链路。
    export_target_label: bool = False  # 保存 target label，通常只在调试导出时使用。
    export_coarse: bool = True  # 保存 A 阶段的粗配准结果。
    export_flow: bool = True  # 保存 coarse / residual / final flow。

    # ==============================
    # 训练配置回填项
    # ==============================
    data_root: str | None = None  # 留空时从 train config.json 回填。
    imgsize: tuple[int, int, int] | None = None  # 推理必须和训练时空间尺寸一致。
    num_classes: int | None = None  # 只用于标签指标统计。
    int_steps: int | None = None  # 必须和训练时积分步数一致，否则 flow 语义会偏。
    enable_stage_b: bool | None = None  # 与训练模型结构绑定。
    enable_roi_b: bool | None = None  # 与训练模型结构绑定。
    roi_classes: tuple[int, ...] | None = None  # 与训练模型结构绑定。
    stage_a_downsample_factor: int | None = None  # 与训练模型结构绑定。
    roi_dilation_radius: int | None = None  # 与训练模型结构绑定。
    roi_condition_channels: int | None = None  # 与训练模型结构绑定。
    bcv_atlas_id: int | None = None  # BCV 可选 atlas id；留空时数据类内部回落到默认 27。
    metrics_json_name: str = "reg_metrics.json"  # 指标汇总文件名。
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 默认优先 GPU。
    memory_release_tolerance_mb: int = 64  # case 间显存回落容忍值，防止 HD95 连续计算堆积。

    @classmethod
    def from_env(cls):
        """从 `OPAL_REG_INF_*` 环境变量读取推理配置。"""

        cfg = cls()
        # 这里保留纯环境变量入口，方便训练脚本尾端直接复用，不引入第二套参数系统。
        env_map = {
            "OPAL_REG_INF_MODEL_DIR": ("model_dir", str),
            "OPAL_REG_INF_CHECKPOINT": ("checkpoint_name", str),
            "OPAL_REG_INF_EXPORT_TRAIN": ("export_train", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_INF_EXPORT_VAL": ("export_val", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_INF_EXPORT_TEST": ("export_test", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_INF_COMPUTE_METRICS": ("compute_metrics", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_INF_COMPUTE_HD95": ("compute_hd95", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_INF_COMPUTE_FOLDING_RATE": ("compute_folding_rate", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_INF_EXPORT_INPUTS": ("export_inputs", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_INF_EXPORT_TARGET_LABEL": ("export_target_label", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_INF_EXPORT_COARSE": ("export_coarse", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_INF_EXPORT_FLOW": ("export_flow", lambda v: v.lower() in {"1", "true", "yes", "on"}),
            "OPAL_REG_INF_DATA_ROOT": ("data_root", str),
            "OPAL_REG_INF_BCV_ATLAS_ID": ("bcv_atlas_id", int),
            "OPAL_REG_INF_NUM_CLASSES": ("num_classes", int),
            "OPAL_REG_INF_METRICS_JSON": ("metrics_json_name", str),
            "OPAL_REG_INF_DEVICE": ("device", str),
            "OPAL_REG_INF_MEMORY_RELEASE_TOLERANCE_MB": ("memory_release_tolerance_mb", int),
        }
        for env_name, (field_name, cast_fn) in env_map.items():
            raw_val = os.environ.get(env_name)
            if raw_val is not None and raw_val != "":
                object.__setattr__(cfg, field_name, cast_fn(raw_val))
        return cfg

    def resolve_from_training_config(self) -> None:
        """尽量从训练目录回填推理所需的结构参数。

        设计原则是“推理默认与训练同构”：
        如果调用方没有显式覆盖，就优先读取 `model_dir/config.json`，
        避免 checkpoint 能加载但结构语义对不上的隐性错误。
        """

        train_cfg = {}
        cfg_path = os.path.join(self.model_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                train_cfg = json.load(f)
        elif any(v is None for v in [self.data_root, self.imgsize, self.num_classes, self.int_steps]):
            raise FileNotFoundError(
                f"Missing required training config: {cfg_path}. "
                "Please provide data_root/imgsize/num_classes/int_steps explicitly."
            )

        # 推理默认与训练同构：只要调用方没显式覆盖，就从训练配置补齐必要字段。
        if self.data_root is None:
            object.__setattr__(self, "data_root", train_cfg["source_root_dir"])
        if self.imgsize is None:
            object.__setattr__(self, "imgsize", tuple(int(x) for x in train_cfg["imgsize"]))
        if self.num_classes is None:
            object.__setattr__(self, "num_classes", int(train_cfg["num_classes"]))
        if self.int_steps is None:
            object.__setattr__(self, "int_steps", int(train_cfg["int_steps"]))
        if self.enable_stage_b is None:
            object.__setattr__(self, "enable_stage_b", bool(train_cfg.get("enable_stage_b", True)))
        if self.enable_roi_b is None:
            object.__setattr__(self, "enable_roi_b", bool(train_cfg.get("enable_roi_b", False)))
        if self.roi_classes is None:
            object.__setattr__(self, "roi_classes", tuple(int(x) for x in train_cfg.get("roi_classes", ())))
        if self.stage_a_downsample_factor is None:
            object.__setattr__(self, "stage_a_downsample_factor", int(train_cfg.get("stage_a_downsample_factor", 4)))
        if self.roi_dilation_radius is None:
            object.__setattr__(self, "roi_dilation_radius", int(train_cfg.get("roi_dilation_radius", 3)))
        if self.roi_condition_channels is None:
            object.__setattr__(self, "roi_condition_channels", int(train_cfg.get("roi_condition_channels", 3)))
        if self.bcv_atlas_id is None and "bcv_atlas_id" in train_cfg:
            object.__setattr__(self, "bcv_atlas_id", int(train_cfg["bcv_atlas_id"]))
        print(
            "Resolved inference config | "
            f"data_root={self.data_root} | imgsize={self.imgsize} | num_classes={self.num_classes} | "
            f"int_steps={self.int_steps} | enable_stage_b={self.enable_stage_b} | "
            f"enable_roi_b={self.enable_roi_b} | roi_classes={self.roi_classes} | "
            f"stage_a_downsample_factor={self.stage_a_downsample_factor} | bcv_atlas_id={self.bcv_atlas_id}"
        )


def write_nifti(
    tensor: torch.Tensor,
    save_path: str,
    reference_path: str | None = None,
):
    """按参考影像头信息写出张量。

    参考图像提供 spacing / origin / direction，避免导出的 warped 结果只有数组值，
    却丢失医学图像坐标系信息。
    """

    array = tensor.detach().cpu().squeeze().numpy()
    if array.ndim == 4 and array.shape[0] in {2, 3}:
        # flow 采用 [C, D, H, W]，写出时转成 SimpleITK 需要的 [D, H, W, C] vector image。
        out_img = sitk.GetImageFromArray(np.moveaxis(array, 0, -1), isVector=True)
    else:
        out_img = sitk.GetImageFromArray(array)
    if reference_path:
        ref_img = sitk.ReadImage(reference_path)
        out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, save_path)


def _extract_path(raw_item) -> str | None:
    if raw_item is None:
        return None
    if isinstance(raw_item, (list, tuple)):
        if len(raw_item) == 0:
            return None
        return _extract_path(raw_item[0])
    return str(raw_item)


def _resolve_reference_paths(dataset, batch) -> tuple[str | None, str | None]:
    """从 dataset / batch 中找出 atlas 与 target 的参考 NIfTI 路径。"""

    target_path = _extract_path(batch[4]) if len(batch) >= 5 else None
    atlas_path = None
    for attr_name in ("atlas_path", "atlas_img_path"):
        attr_val = getattr(dataset, attr_name, None)
        if isinstance(attr_val, str) and attr_val:
            atlas_path = attr_val
            break
    return atlas_path, target_path


def _compute_binary_dice(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    pred_sum = int(pred_mask.sum().item())
    target_sum = int(target_mask.sum().item())
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    intersection = int((pred_mask & target_mask).sum().item())
    return float((2.0 * intersection) / (pred_sum + target_sum + 1e-8))


def _compute_binary_hd95(pred_mask: np.ndarray, target_mask: np.ndarray) -> float | None:
    """二值器官 HD95；若某一侧为空，则返回 None 表示该类不可比。"""

    pred_sum = int(pred_mask.sum())
    target_sum = int(target_mask.sum())
    if pred_sum == 0 and target_sum == 0:
        return 0.0
    if pred_sum == 0 or target_sum == 0:
        return None

    pred_img = sitk.GetImageFromArray(pred_mask.astype(np.uint8))
    target_img = sitk.GetImageFromArray(target_mask.astype(np.uint8))
    pred_surface = sitk.LabelContour(pred_img)
    target_surface = sitk.LabelContour(target_img)
    pred_dist = sitk.Abs(sitk.SignedMaurerDistanceMap(pred_img, squaredDistance=False, useImageSpacing=True))
    target_dist = sitk.Abs(sitk.SignedMaurerDistanceMap(target_img, squaredDistance=False, useImageSpacing=True))

    pred_surface_arr = sitk.GetArrayViewFromImage(pred_surface) > 0
    target_surface_arr = sitk.GetArrayViewFromImage(target_surface) > 0
    pred_dist_arr = sitk.GetArrayViewFromImage(pred_dist)
    target_dist_arr = sitk.GetArrayViewFromImage(target_dist)

    dist_pred_to_target = target_dist_arr[pred_surface_arr].astype(np.float32, copy=True)
    dist_target_to_pred = pred_dist_arr[target_surface_arr].astype(np.float32, copy=True)
    all_surface_dist = np.concatenate([dist_pred_to_target, dist_target_to_pred], axis=0)
    hd95 = float(np.percentile(all_surface_dist, 95))

    del pred_img, target_img, pred_surface, target_surface, pred_dist, target_dist
    del pred_surface_arr, target_surface_arr, pred_dist_arr, target_dist_arr
    del dist_pred_to_target, dist_target_to_pred, all_surface_dist
    gc.collect()
    return hd95


def _compute_folding_rate(flow: torch.Tensor) -> float:
    """通过 Jacobian determinant < 0 的体素比例衡量形变折叠。"""

    if flow.ndim == 5:
        if flow.shape[0] != 1 or flow.shape[1] != 3:
            raise ValueError("flow must be [1, 3, D, H, W] or [3, D, H, W]")
        flow = flow.squeeze(0)
    if flow.ndim != 4 or flow.shape[0] != 3:
        raise ValueError("flow must be [3, D, H, W]")

    u = flow.float()
    duz_dz = u[0, 1:, :-1, :-1] - u[0, :-1, :-1, :-1]
    duz_dy = u[0, :-1, 1:, :-1] - u[0, :-1, :-1, :-1]
    duz_dx = u[0, :-1, :-1, 1:] - u[0, :-1, :-1, :-1]

    duy_dz = u[1, 1:, :-1, :-1] - u[1, :-1, :-1, :-1]
    duy_dy = u[1, :-1, 1:, :-1] - u[1, :-1, :-1, :-1]
    duy_dx = u[1, :-1, :-1, 1:] - u[1, :-1, :-1, :-1]

    dux_dz = u[2, 1:, :-1, :-1] - u[2, :-1, :-1, :-1]
    dux_dy = u[2, :-1, 1:, :-1] - u[2, :-1, :-1, :-1]
    dux_dx = u[2, :-1, :-1, 1:] - u[2, :-1, :-1, :-1]

    j11 = 1.0 + duz_dz
    j12 = duz_dy
    j13 = duz_dx
    j21 = duy_dz
    j22 = 1.0 + duy_dy
    j23 = duy_dx
    j31 = dux_dz
    j32 = dux_dy
    j33 = 1.0 + dux_dx

    det = (
        j11 * (j22 * j33 - j23 * j32)
        - j12 * (j21 * j33 - j23 * j31)
        + j13 * (j21 * j32 - j22 * j31)
    )
    return float((det < 0).float().mean().item())


def _compute_case_metrics(
    pred_label: torch.Tensor,
    target_label: torch.Tensor,
    flow: torch.Tensor | None,
    num_classes: int,
    compute_hd95: bool,
    compute_folding_rate: bool,
) -> dict[str, object]:
    """统计单个 case 的指标。

    输出既包含 case 级均值，也保留 per-class 结果，便于后续画图和排查异常类别。
    """

    dice_per_class: dict[str, float] = {}
    hd95_per_class: dict[str, float | None] = {}
    dice_values: list[float] = []
    hd95_values: list[float] = []
    skipped_empty_classes = 0
    invalid_hd95_classes = 0

    for cls_idx in range(1, num_classes):
        pred_mask = pred_label == cls_idx
        target_mask = target_label == cls_idx
        pred_sum = int(pred_mask.sum().item())
        target_sum = int(target_mask.sum().item())
        if pred_sum == 0 and target_sum == 0:
            skipped_empty_classes += 1
            continue

        dice_value = _compute_binary_dice(pred_mask, target_mask)
        hd95_value = None
        if compute_hd95:
            hd95_value = _compute_binary_hd95(
                pred_mask.numpy().astype(np.uint8, copy=False),
                target_mask.numpy().astype(np.uint8, copy=False),
            )
        dice_per_class[str(cls_idx)] = dice_value
        hd95_per_class[str(cls_idx)] = hd95_value
        dice_values.append(dice_value)
        if not compute_hd95:
            continue
        if hd95_value is None:
            invalid_hd95_classes += 1
        else:
            hd95_values.append(hd95_value)

        del pred_mask, target_mask
        gc.collect()

    folding_rate = _compute_folding_rate(flow) if compute_folding_rate and flow is not None else None

    return {
        "dice_mean": float(sum(dice_values) / len(dice_values)) if dice_values else None,
        "hd95_mean": float(sum(hd95_values) / len(hd95_values)) if hd95_values else None,
        "folding_rate": folding_rate,
        "valid_class_count": len(dice_values),
        "valid_hd95_class_count": len(hd95_values) if compute_hd95 else 0,
        "skipped_empty_class_count": skipped_empty_classes,
        "invalid_hd95_class_count": invalid_hd95_classes if compute_hd95 else 0,
        "dice_per_class": dice_per_class,
        "hd95_per_class": hd95_per_class if compute_hd95 else {},
    }


def _summarize_split(case_metrics: list[dict[str, object]]) -> dict[str, object]:
    """把一个 split 的 case 指标汇总成均值统计。"""

    dice_values = [item["dice_mean"] for item in case_metrics if item["dice_mean"] is not None]
    hd95_values = [item["hd95_mean"] for item in case_metrics if item["hd95_mean"] is not None]
    folding_values = [item["folding_rate"] for item in case_metrics if item["folding_rate"] is not None]
    return {
        "case_count": len(case_metrics),
        "dice_mean": float(sum(dice_values) / len(dice_values)) if dice_values else None,
        "hd95_mean": float(sum(hd95_values) / len(hd95_values)) if hd95_values else None,
        "folding_rate_mean": float(sum(folding_values) / len(folding_values)) if folding_values else None,
        "valid_dice_case_count": len(dice_values),
        "valid_hd95_case_count": len(hd95_values),
        "valid_folding_rate_case_count": len(folding_values),
        "cases": case_metrics,
    }


def _save_metrics_json(cfg: Config, metrics_payload: dict[str, object]) -> None:
    save_path = os.path.join(cfg.model_dir, cfg.metrics_json_name)
    merged_payload: dict[str, object] = {}
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                old_payload = json.load(f)
            if isinstance(old_payload, dict):
                merged_payload = old_payload
        except (json.JSONDecodeError, OSError):
            merged_payload = {}

    old_splits = merged_payload.get("splits", {})
    new_splits = metrics_payload.get("splits", {})
    if not isinstance(old_splits, dict):
        old_splits = {}
    if not isinstance(new_splits, dict):
        new_splits = {}

    merged_payload.update(metrics_payload)
    merged_payload["splits"] = {**old_splits, **new_splits}

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(merged_payload, f, indent=2, ensure_ascii=False)


def _current_cuda_memory(device: str) -> int:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return 0
    torch.cuda.synchronize(device=device)
    return int(torch.cuda.memory_allocated(device=device))


def _wait_cuda_memory_release(
    device: str,
    baseline_bytes: int,
    tolerance_bytes: int = 64 * 1024 * 1024,
    timeout_sec: float = 5.0,
) -> None:
    if baseline_bytes <= 0 or not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return

    target_bytes = baseline_bytes + max(int(tolerance_bytes), 0)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        gc.collect()
        torch.cuda.empty_cache()
        current_bytes = _current_cuda_memory(device)
        if current_bytes <= target_bytes:
            return
        time.sleep(0.05)


def _resolve_ckpt(cfg: Config) -> str:
    if cfg.checkpoint_name == "resume_latest.pth":
        return os.path.join(cfg.model_dir, cfg.checkpoint_name)

    if cfg.checkpoint_name == "best_dice":
        prefixes = ("best_dice_ep", "model_best_dice")
    elif cfg.checkpoint_name == "best_loss":
        prefixes = ("best_loss_ep", "model_best_loss")
    else:
        return os.path.join(cfg.model_dir, cfg.checkpoint_name)

    candidates = sorted(
        f for f in os.listdir(cfg.model_dir)
        if f.endswith(".pth") and any(f.startswith(prefix) for prefix in prefixes)
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for key: {cfg.checkpoint_name}")
    return os.path.join(cfg.model_dir, candidates[-1])


def _build_datasets(data_root: str, bcv_atlas_id: int | None = None):
    """Build ADNI, BCV, or OASIS registration inference datasets."""

    if "ADNI" in data_root:
        return (
            dataset_lib.ADNI_Split_Dataset(data_root, is_training=True, inference=True),
            dataset_lib.ADNI_Split_Dataset(data_root, is_training=False, is_val=True, inference=True),
            dataset_lib.ADNI_Split_Dataset(data_root, is_training=False, is_val=False, inference=True),
        )
    if "BCV" in data_root or "bcv_small" in data_root.lower():
        return (
            dataset_lib.BCV_Dataset(data_root, is_training=True, inference=True, atlas_id=bcv_atlas_id),
            dataset_lib.BCV_Dataset(data_root, is_training=False, is_val=True, inference=True, atlas_id=bcv_atlas_id),
            dataset_lib.BCV_Dataset(data_root, is_training=False, is_val=False, inference=True, atlas_id=bcv_atlas_id),
        )
    if "OASIS" in data_root:
        return (
            dataset_lib.OASIS_Dataset(data_root, is_training=True, inference=True),
            dataset_lib.OASIS_Dataset(data_root, is_training=False, is_val=True, inference=True),
            dataset_lib.OASIS_Dataset(data_root, is_training=False, is_val=False, inference=True),
        )
    raise ValueError(
        f"Unsupported data_root for OPAL-Net release inference: {data_root}. "
        "Supported datasets are ADNI, BCV, and OASIS."
    )


def _stem_from_batch(batch, fallback_idx: int) -> str:
    stem = f"sample_{fallback_idx:04d}"
    if len(batch) < 6:
        return stem
    raw_name = batch[5][0]
    stem = Path(raw_name).stem
    if stem.endswith(".nii"):
        stem = Path(stem).stem
    return stem


def _export_split(reg_model, dataset, split_name: str, cfg: Config):
    """执行一个 split 的完整推理流程：可选导出、可选统计指标。"""

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    save_outputs = any(
        [
            cfg.export_inputs,
            cfg.export_target_label,
            cfg.export_coarse,
            cfg.export_final,
            cfg.export_flow,
        ]
    )
    # 只有确实要保存结果时才创建目录，避免“纯算指标”时留下空目录。
    split_dir = os.path.join(cfg.model_dir, f"{split_name}_output") if save_outputs else None
    if split_dir is not None:
        os.makedirs(split_dir, exist_ok=True)
    case_metrics: list[dict[str, object]] = []

    for idx, batch in enumerate(tqdm(loader, total=len(loader), desc=f"export {split_name}", leave=False)):
        moving, fixed, moving_label, target_label = [t.to(cfg.device) for t in batch[:4]]
        cuda_baseline_bytes = _current_cuda_memory(cfg.device)
        stem = _stem_from_batch(batch, idx)
        atlas_ref_path, target_ref_path = _resolve_reference_paths(dataset, batch)
        out = reg_model(moving=moving, fixed=fixed, moving_label=moving_label)

        if cfg.export_inputs:
            write_nifti(moving, os.path.join(split_dir, f"{stem}_input_atlas_image.nii.gz"), atlas_ref_path)
            write_nifti(fixed, os.path.join(split_dir, f"{stem}_input_target_image.nii.gz"), target_ref_path)
            write_nifti(moving_label, os.path.join(split_dir, f"{stem}_input_atlas_label.nii.gz"), atlas_ref_path)
            if cfg.export_target_label:
                write_nifti(target_label.float(), os.path.join(split_dir, f"{stem}_input_target_label.nii.gz"), target_ref_path)

        if cfg.export_coarse:
            write_nifti(out["coarse_warped"], os.path.join(split_dir, f"{stem}_coarse_image.nii.gz"), target_ref_path)
            write_nifti(out["coarse_warped_label"], os.path.join(split_dir, f"{stem}_coarse_label.nii.gz"), target_ref_path)

        if cfg.export_final:
            write_nifti(out["final_warped"], os.path.join(split_dir, f"{stem}_final_image.nii.gz"), target_ref_path)
            write_nifti(out["final_warped_label"], os.path.join(split_dir, f"{stem}_final_label.nii.gz"), target_ref_path)

        if cfg.export_flow:
            write_nifti(out["coarse_flow"], os.path.join(split_dir, f"{stem}_coarse_flow.nii.gz"), target_ref_path)
            write_nifti(out["residual_flow"], os.path.join(split_dir, f"{stem}_residual_flow.nii.gz"), target_ref_path)
            write_nifti(out["final_flow"], os.path.join(split_dir, f"{stem}_final_flow.nii.gz"), target_ref_path)

        if cfg.compute_metrics:
            # 指标与导出解耦：train split 常用于保存可视化结果，test split 常用于正式统计。
            pred_label = out["final_warped_label"].detach().cpu().squeeze().long()
            target_label_cpu = target_label.detach().cpu().squeeze().long()
            final_flow_cpu = out["final_flow"].detach().cpu()
            metrics = _compute_case_metrics(
                pred_label,
                target_label_cpu,
                final_flow_cpu,
                cfg.num_classes,
                cfg.compute_hd95,
                cfg.compute_folding_rate,
            )
            case_metrics.append({"case": stem, "split": split_name, **metrics})
            del pred_label, target_label_cpu, final_flow_cpu

        del out, moving, fixed, moving_label, target_label, batch
        gc.collect()
        if torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
            torch.cuda.empty_cache()
            _wait_cuda_memory_release(
                cfg.device,
                cuda_baseline_bytes,
                tolerance_bytes=cfg.memory_release_tolerance_mb * 1024 * 1024,
            )

    if not cfg.compute_metrics:
        return None
    return _summarize_split(case_metrics)


def main(cfg: Config):
    """推理总控入口。

    主流程顺序：
    1. 解析并回填训练配置；
    2. 解析 checkpoint 路径并恢复模型；
    3. 构建 train/val/test 数据集；
    4. 按 `export_*` 开关逐个 split 执行导出；
    5. 若开启 `compute_metrics`，持续把 split 结果写入 `reg_metrics.json`。
    """

    object.__setattr__(cfg, "model_dir", str(Path(cfg.model_dir).resolve()))
    cfg.resolve_from_training_config()

    ckpt_path = _resolve_ckpt(cfg)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    bundle = reg_model_lib.build_cascaded_reg_model_bundle(
        inshape=cfg.imgsize,
        int_steps=cfg.int_steps,
        enable_stage_b=True if cfg.enable_stage_b is None else cfg.enable_stage_b,
        enable_roi_b=bool(cfg.enable_roi_b),
        roi_classes=tuple(cfg.roi_classes or ()),
        stage_a_downsample_factor=4 if cfg.stage_a_downsample_factor is None else int(cfg.stage_a_downsample_factor),
        roi_dilation_radius=3 if cfg.roi_dilation_radius is None else int(cfg.roi_dilation_radius),
        roi_condition_channels=3 if cfg.roi_condition_channels is None else int(cfg.roi_condition_channels),
    ).to(cfg.device)
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    bundle.load_state_dict(ckpt["model_state_dict"])
    reg_model = bundle["reg_model"]
    reg_model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    train_set, val_set, test_set = _build_datasets(cfg.data_root, cfg.bcv_atlas_id)
    metrics_payload: dict[str, object] | None = None
    if cfg.compute_metrics:
        metrics_payload = {
            "model_dir": cfg.model_dir,
            "checkpoint": ckpt_path,
            "data_root": cfg.data_root,
            "num_classes": cfg.num_classes,
            "enable_stage_b": cfg.enable_stage_b,
            "stage_a_downsample_factor": cfg.stage_a_downsample_factor,
            "metrics_unit": {
                "dice": "unitless",
                "hd95": "voxel" if cfg.compute_hd95 else None,
                "folding_rate": "ratio" if cfg.compute_folding_rate else None,
            },
            "compute_metrics": cfg.compute_metrics,
            "compute_hd95": cfg.compute_hd95,
            "compute_folding_rate": cfg.compute_folding_rate,
            "splits": {},
        }
    if cfg.export_train:
        split_summary = _export_split(reg_model, train_set, "train", cfg)
        if metrics_payload is not None and split_summary is not None:
            metrics_payload["splits"]["train"] = split_summary
            _save_metrics_json(cfg, metrics_payload)
    if cfg.export_val:
        split_summary = _export_split(reg_model, val_set, "val", cfg)
        if metrics_payload is not None and split_summary is not None:
            metrics_payload["splits"]["val"] = split_summary
            _save_metrics_json(cfg, metrics_payload)
    if cfg.export_test:
        split_summary = _export_split(reg_model, test_set, "test", cfg)
        if metrics_payload is not None and split_summary is not None:
            metrics_payload["splits"]["test"] = split_summary
            _save_metrics_json(cfg, metrics_payload)

    if metrics_payload is not None:
        _save_metrics_json(cfg, metrics_payload)
        print(f"Metrics saved to: {os.path.join(cfg.model_dir, cfg.metrics_json_name)}")
    print(f"Outputs saved under: {cfg.model_dir}")


if __name__ == "__main__":
    main(Config.from_env())
