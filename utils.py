"""Shared training utilities, metrics, and 3D warping helpers.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageMeter:
    """Track the current value, running mean, and standard deviation."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals: list[float] = []
        self.std = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  #  epoch
        self.vals.append(val)
        self.std = np.std(self.vals)      #


class SpatialTransformer(nn.Module):
    """N-D  `grid_sample`  backward warping"""

    def __init__(self, size, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode

        #  forward
        vectors = [torch.arange(0, s) for s in size]
        grid = torch.stack(torch.meshgrid(*vectors, indexing="ij")).unsqueeze(0).float()
        self.register_buffer("grid", grid)  # buffer  module.to(device)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # `grid_sample`  [-1, 1]
        for dim_idx, dim_size in enumerate(shape):
            denom = max(dim_size - 1, 1)
            new_locs[:, dim_idx, ...] = 2.0 * (new_locs[:, dim_idx, ...] / denom - 0.5)

        # `grid_sample`  3D  x/y/z
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class register_model(nn.Module):
    """ warp


    - `x[0]`:
    - `x[1]`:  / flow
    """

    def __init__(self, img_size=(64, 256, 256), mode: str = "bilinear"):
        super().__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img, flow = x  #
        return self.spatial_trans(img, flow)


class EarlyStopping:
    """"""

    def __init__(self, patience: int = 20, verbose: bool = False, delta: float = 0.0):
        self.patience = patience  #  epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta        #

    def __call__(self, val_score: float) -> None:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


class Logger:
    """ stdout """

    def __init__(self, save_dir: str):
        self.terminal = sys.stdout
        self.log = open(os.path.join(save_dir, "logfile.log"), "a", buffering=1)  #

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        if message.endswith("\n"):
            self.flush()  # epoch/batch

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()
        os.fsync(self.log.fileno())


class CheckpointManager:
    """ AMP """

    def __init__(self, save_dir: str, amp_enabled: bool):
        self.save_dir = save_dir
        self.amp_enabled = amp_enabled
        self.resume_path = os.path.join(save_dir, "resume_latest.pth")  #

    def _dump_model_states(self, model) -> dict:
        if isinstance(model, dict):
            return {f"model_{name}_state_dict": module.state_dict() for name, module in model.items()}
        return {"model_state_dict": model.state_dict()}

    def _dump_optimizer_states(self, optimizer) -> dict:
        if isinstance(optimizer, dict):
            return {f"optimizer_{name}_state_dict": opt.state_dict() for name, opt in optimizer.items()}
        return {"optimizer_state_dict": optimizer.state_dict()}

    def _load_model_states(self, ckpt: dict, model) -> None:
        if isinstance(model, dict):
            for name, module in model.items():
                key = f"model_{name}_state_dict"
                if key in ckpt:
                    module.load_state_dict(ckpt[key])
            return
        model.load_state_dict(ckpt["model_state_dict"])

    def _load_optimizer_states(self, ckpt: dict, optimizer) -> None:
        if isinstance(optimizer, dict):
            for name, opt in optimizer.items():
                key = f"optimizer_{name}_state_dict"
                if key in ckpt:
                    opt.load_state_dict(ckpt[key])
            return
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    def build_checkpoint(
        self,
        model,
        optimizer,
        scaler,
        epoch: int,
        best_dice: float,
        best_eval_loss: float,
        pth_best_loss: str,
        pth_best_dice: str,
        wait: int,
        history: dict[str, list[float]],
    ) -> dict:
        payload = {
            "epoch": epoch,
            "scaler_state_dict": scaler.state_dict() if self.amp_enabled else None,
            "best_dice": best_dice,
            "best_eval_loss": best_eval_loss,
            "pth_best_loss": pth_best_loss,
            "pth_best_dice": pth_best_dice,
            "wait": wait,
            "history": history,  # resume
        }
        payload.update(self._dump_model_states(model))
        payload.update(self._dump_optimizer_states(optimizer))
        return payload

    def save(
        self,
        file_name: str,
        model,
        optimizer,
        scaler,
        epoch: int,
        best_dice: float,
        best_eval_loss: float,
        pth_best_loss: str,
        pth_best_dice: str,
        wait: int,
        history: dict[str, list[float]],
    ) -> str:
        path = os.path.join(self.save_dir, file_name)
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(
            self.build_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_dice=best_dice,
                best_eval_loss=best_eval_loss,
                pth_best_loss=pth_best_loss,
                pth_best_dice=pth_best_dice,
                wait=wait,
                history=history,
            ),
            path,
        )
        return path

    def load_resume(
        self,
        continue_train: bool,
        model,
        optimizer,
        scaler,
        defaults: dict,
        items: list[str],
    ) -> dict:
        state = dict(defaults)
        if not continue_train or not os.path.exists(self.resume_path):
            return state

        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(self.resume_path, map_location=map_location)
        self._load_model_states(ckpt, model)
        self._load_optimizer_states(ckpt, optimizer)
        if self.amp_enabled and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        state["start_epoch"] = int(ckpt.get("epoch", state["start_epoch"]))
        state["best_dice"] = float(ckpt.get("best_dice", state["best_dice"]))
        state["best_eval_loss"] = float(ckpt.get("best_eval_loss", state["best_eval_loss"]))
        state["wait"] = int(ckpt.get("wait", state["wait"]))
        state["pth_best_loss"] = ckpt.get("pth_best_loss", state["pth_best_loss"])
        state["pth_best_dice"] = ckpt.get("pth_best_dice", state["pth_best_dice"])

        history = ckpt.get("history", {})
        fig = state["fig"]
        for item_name in items:
            if item_name in history and isinstance(history[item_name], list):
                fig[item_name] = history[item_name]
        state["fig"] = fig

        print(
            f"Resume from epoch {state['start_epoch']}, "
            f"best_loss={state['best_eval_loss']:.4f}, best_dice={state['best_dice']:.4f}"
        )
        return state


def adjust_learning_rate(optimizer, epoch: int, max_epochs: int, base_lr: float) -> float:
    """ lr"""

    lr = round(base_lr * (1 - epoch / max_epochs) ** 0.9, 8)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def dice_val(y_pred: torch.Tensor, y_true: torch.Tensor, num_clus: int) -> torch.Tensor:
    """ Dice"""

    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()

    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()

    y_pred = y_pred[:, 1:]  #
    y_true = y_true[:, 1:]
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.0 * intersection) / (union + 1e-5)

    #  foreground  1.0  0
    valid_mask = union > 0
    valid_count = valid_mask.sum(dim=1)
    dsc = dsc * valid_mask
    per_case_dice = torch.where(
        valid_count > 0,
        dsc.sum(dim=1) / valid_count.clamp_min(1).to(dsc.dtype),
        torch.ones_like(valid_count, dtype=dsc.dtype),
    )
    return torch.mean(per_case_dice)


def multi_class_dice_coefficient(
    label1,
    label2,
    num_classes: int,
    include_background: bool = False,
) -> tuple[list[float], float]:
    """ numpy  Dice  Dice"""

    dice_coefficients = []
    start_cls = 0 if include_background else 1
    for class_idx in range(start_cls, num_classes):
        label1_binary = (label1 == class_idx).astype(np.int32)
        label2_binary = (label2 == class_idx).astype(np.int32)
        intersection = np.sum(label1_binary * label2_binary)
        union = np.sum(label1_binary) + np.sum(label2_binary)
        if union == 0:
            continue
        dice_coefficient = (2.0 * intersection) / (union + 1e-8)
        dice_coefficients.append(float(dice_coefficient))

    avg_dice = float(np.mean(dice_coefficients)) if len(dice_coefficients) > 0 else 1.0
    return dice_coefficients, avg_dice


def warp_3d(src: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """ 3D /


    - `flow`
    -  `src`  warped image / warped label
    """

    batch_size, _, depth, height, width = flow.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(depth, device=flow.device, dtype=flow.dtype),
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    grid = torch.stack([zz, yy, xx], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    loc = grid + flow
    loc[:, 0] = 2.0 * (loc[:, 0] / max(depth - 1, 1) - 0.5)
    loc[:, 1] = 2.0 * (loc[:, 1] / max(height - 1, 1) - 0.5)
    loc[:, 2] = 2.0 * (loc[:, 2] / max(width - 1, 1) - 0.5)
    grid_norm = loc.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]  # `grid_sample`  x/y/z
    return F.grid_sample(src, grid_norm, mode=mode, align_corners=True)
