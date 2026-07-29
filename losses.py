"""Loss functions used by the OPAL-Net registration and segmentation models.

The module contains similarity, deformation regularization, Dice, feature
alignment, and auxiliary losses for 3D tensors shaped [B, C, D, H, W].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import warp_3d


# ==============================
#
# ==============================
class VxmDiffReconLoss(nn.Module):
    """
     MSE /


    -  moving  fixed
    -  KL


    - fixed_img:  [B, C, D, H, W]
    - warped_moving_img:  fixed_img


    -
    """

    def __init__(self, sigma_sq: float = 1.0, sample_count: int = 1):
        super().__init__()
        self.sigma_sq = sigma_sq          #
        self.sample_count = sample_count  #

    def forward(self, fixed_img: torch.Tensor, warped_moving_img: torch.Tensor) -> torch.Tensor:
        if fixed_img.shape != warped_moving_img.shape:
            raise ValueError("fixed_img and warped_moving_img must share the same shape")
        if self.sigma_sq <= 0:
            raise ValueError("sigma_sq must be positive")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")

        mse = (fixed_img - warped_moving_img).pow(2).mean()                 #
        return mse / (2.0 * self.sigma_sq * float(self.sample_count))       #


class VxmDiffKLLoss(nn.Module):
    """
     KL


    -
    -


    - v_mean:
    - v_log_sigma:  logvar
    """

    def __init__(self, prior_lambda: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.prior_lambda = prior_lambda  #
        self.eps = eps                    #  log(0)

    def forward(self, v_mean: torch.Tensor, v_log_sigma: torch.Tensor) -> torch.Tensor:
        if v_mean.shape != v_log_sigma.shape:
            raise ValueError("v_mean and v_log_sigma must share the same shape")
        if self.prior_lambda <= 0:
            raise ValueError("prior_lambda must be positive")

        var = torch.exp(2.0 * v_log_sigma)                                        #  log_sigma
        log_var = torch.log(var + self.eps)                                       #  log(var)
        kl = 0.5 * (self.prior_lambda * (v_mean.pow(2) + var) - log_var - 1.0)   # KL(q||p)
        return kl.mean()                                                           #


class NCC(nn.Module):
    """
    LNCC  loss


    -  CUDA  autocast float  LNCC
    """

    def __init__(self, win: int = 9):
        super().__init__()
        self.win = win  #

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        if y_true.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                return lncc_loss(y_true.float(), y_pred.float(), win=self.win)  # GPU  FP32
        return lncc_loss(y_true.float(), y_pred.float(), win=self.win)           # CPU


class ROIWeightedLNCC(nn.Module):
    """
    ROI  LNCC


    -  refinement  selected ROI / boundary
    -  flow
    """

    def __init__(self, win: int = 9, eps: float = 1e-6):
        super().__init__()
        self.win = win
        self.eps = eps

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must share the same shape")
        if weight.ndim != 5 or weight.shape[1] != 1:
            raise ValueError("weight must be [B, 1, D, H, W]")
        if weight.shape[0] != y_true.shape[0] or weight.shape[2:] != y_true.shape[2:]:
            raise ValueError("weight spatial shape must match y_true/y_pred")

        norm_weight = weight.float().clamp_min(0.0)
        norm_weight = norm_weight / norm_weight.mean(dim=(2, 3, 4), keepdim=True).clamp_min(self.eps)
        norm_weight = torch.sqrt(norm_weight.clamp_min(self.eps))
        if y_true.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                return lncc_loss(
                    y_true.float() * norm_weight,
                    y_pred.float() * norm_weight,
                    win=self.win,
                )
        return lncc_loss(y_true.float() * norm_weight, y_pred.float() * norm_weight, win=self.win)


class FlowGradLoss(nn.Module):
    """



    -  flow/SVF
    """

    def __init__(self):
        super().__init__()

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 5 or flow.shape[1] != 3:
            raise ValueError("flow must be [B, 3, D, H, W]")
        dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]   # z
        dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]   # y
        dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]   # x
        return (dz.square().mean() + dy.square().mean() + dx.square().mean()) / 3.0


class JacobianDeterminantLoss(nn.Module):
    """
    Jacobian


    -  det(J) < eps /


    -  flow  u Jacobian  I + u
    -
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  #  Jacobian

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 5 or flow.shape[1] != 3:
            raise ValueError("flow must be [B, 3, D, H, W]")

        u = flow
        duz_dz = u[:, 0, 1:, :-1, :-1] - u[:, 0, :-1, :-1, :-1]  # u_z/z
        duz_dy = u[:, 0, :-1, 1:, :-1] - u[:, 0, :-1, :-1, :-1]  # u_z/y
        duz_dx = u[:, 0, :-1, :-1, 1:] - u[:, 0, :-1, :-1, :-1]  # u_z/x

        duy_dz = u[:, 1, 1:, :-1, :-1] - u[:, 1, :-1, :-1, :-1]  # u_y/z
        duy_dy = u[:, 1, :-1, 1:, :-1] - u[:, 1, :-1, :-1, :-1]  # u_y/y
        duy_dx = u[:, 1, :-1, :-1, 1:] - u[:, 1, :-1, :-1, :-1]  # u_y/x

        dux_dz = u[:, 2, 1:, :-1, :-1] - u[:, 2, :-1, :-1, :-1]  # u_x/z
        dux_dy = u[:, 2, :-1, 1:, :-1] - u[:, 2, :-1, :-1, :-1]  # u_x/y
        dux_dx = u[:, 2, :-1, :-1, 1:] - u[:, 2, :-1, :-1, :-1]  # u_x/x

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
        )  # 3x3
        return torch.relu(self.eps - det).mean()  #  det < eps


class MaskedFlowGradLoss(nn.Module):
    """



    -
    - /


    - flow: [B, 3, D, H, W]
    - weight: [B, 1, D, H, W] [0,1]
    """

    def __init__(self):
        super().__init__()

    def forward(self, flow: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 5 or flow.shape[1] != 3:
            raise ValueError("flow must be [B, 3, D, H, W]")
        if weight.ndim != 5 or weight.shape[1] != 1:
            raise ValueError("weight must be [B, 1, D, H, W]")

        dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]  # z
        dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]  # y
        dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]  # x

        #
        wz = weight[:, :, 1:, :, :]
        wy = weight[:, :, :, 1:, :]
        wx = weight[:, :, :, :, 1:]

        #  flow.shape[1]
        dz_loss = (dz.square() * wz).sum() / (wz.sum() * flow.shape[1] + 1e-6)
        dy_loss = (dy.square() * wy).sum() / (wy.sum() * flow.shape[1] + 1e-6)
        dx_loss = (dx.square() * wx).sum() / (wx.sum() * flow.shape[1] + 1e-6)
        return (dz_loss + dy_loss + dx_loss) / 3.0


class KLDivergenceLoss(nn.Module):
    """
    KL


    -  nn.Module loss
    -  kl_divergence_loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, mu: torch.Tensor | None, logvar: torch.Tensor | None) -> torch.Tensor:
        return kl_divergence_loss(mu, logvar)  #


class MeanAbsLoss(nn.Module):
    """"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(x))


class GradientConsistencyLoss(nn.Module):
    """ 3D """

    def __init__(self):
        super().__init__()

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        dz_a = img_a[:, :, 1:, :, :] - img_a[:, :, :-1, :, :]
        dy_a = img_a[:, :, :, 1:, :] - img_a[:, :, :, :-1, :]
        dx_a = img_a[:, :, :, :, 1:] - img_a[:, :, :, :, :-1]
        dz_b = img_b[:, :, 1:, :, :] - img_b[:, :, :-1, :, :]
        dy_b = img_b[:, :, :, 1:, :] - img_b[:, :, :, :-1, :]
        dx_b = img_b[:, :, :, :, 1:] - img_b[:, :, :, :, :-1]
        return (
            F.l1_loss(dz_a, dz_b)
            + F.l1_loss(dy_a, dy_b)
            + F.l1_loss(dx_a, dx_b)
        ) / 3.0


class FeatureSeparationLoss(nn.Module):
    """ common/diff """

    def __init__(self):
        super().__init__()

    def forward(self, common_vec: torch.Tensor, diff_vec: torch.Tensor) -> torch.Tensor:
        common_vec = F.normalize(common_vec, dim=1)
        diff_vec = F.normalize(diff_vec, dim=1)
        return torch.abs((common_vec * diff_vec).sum(dim=1)).mean()


class CommonConsistencyLoss(nn.Module):
    """ common  moving/fixed """

    def __init__(self):
        super().__init__()

    def forward(self, common_a_vec: torch.Tensor, common_f_vec: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(common_a_vec, common_f_vec)


class CosineCommonConsistencyLoss(nn.Module):
    """ common """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, common_a_vec: torch.Tensor, common_f_vec: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(common_a_vec, common_f_vec, dim=1, eps=self.eps)
        return (1.0 - sim).mean()


class CosineCommonDiffLoss(nn.Module):
    """ common  common/diff """

    def __init__(
        self,
        common_weight: float = 1.0,
        separation_weight: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.common_weight = common_weight
        self.separation_weight = separation_weight
        self.eps = eps

    def forward(
        self,
        common_a_vec: torch.Tensor,
        common_f_vec: torch.Tensor,
        diff_a_vec: torch.Tensor,
        diff_f_vec: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        common_sim = F.cosine_similarity(common_a_vec, common_f_vec, dim=1, eps=self.eps)
        loss_common = (1.0 - common_sim).mean()

        common_a_norm = F.normalize(common_a_vec, dim=1, eps=self.eps)
        common_f_norm = F.normalize(common_f_vec, dim=1, eps=self.eps)
        diff_a_norm = F.normalize(diff_a_vec, dim=1, eps=self.eps)
        diff_f_norm = F.normalize(diff_f_vec, dim=1, eps=self.eps)
        loss_sep = 0.5 * (
            torch.abs((common_a_norm * diff_a_norm).sum(dim=1)).mean()
            + torch.abs((common_f_norm * diff_f_norm).sum(dim=1)).mean()
        )

        total = self.common_weight * loss_common + self.separation_weight * loss_sep
        return {
            "loss": total,
            "loss_common": loss_common,
            "loss_sep": loss_sep,
        }


class DiffRegularizationLoss(nn.Module):
    """ diff_gap """

    def __init__(self, tau: float = 0.01, act_weight: float = 1.0, var_weight: float = 0.1):
        super().__init__()
        self.tau = tau
        self.act_weight = act_weight
        self.var_weight = var_weight

    def forward(self, diff_gap: torch.Tensor) -> torch.Tensor:
        if diff_gap.ndim != 5:
            raise ValueError("diff_gap must be [B, C, D, H, W]")
        loss_act = F.relu(self.tau - diff_gap.abs().mean())
        loss_var = -diff_gap.var()
        return self.act_weight * loss_act + self.var_weight * loss_var


class MMDLoss(nn.Module):
    """ Maximum Mean Discrepancy (MMD)

     (Multi-kernel Gaussian)
    """

    def __init__(
        self,
        kernel_alphas: tuple[float, ...] = (0.1, 1.0, 10.0),
        pooled_size: tuple[int, ...] = (4, 4, 4),
    ):
        super().__init__()
        self.kernel_alphas = kernel_alphas
        self.pooled_size = pooled_size

    def _reshape_samples(self, feat: torch.Tensor) -> torch.Tensor:
        """ `[N, C]` """

        if feat.ndim == 5:
            feat = F.adaptive_avg_pool3d(feat, self.pooled_size)
            feat = feat.permute(0, 2, 3, 4, 1).reshape(-1, feat.shape[1])
            return feat
        if feat.ndim == 4:
            feat = F.adaptive_avg_pool2d(feat, self.pooled_size[:2])
            feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
            return feat
        if feat.ndim == 2:
            return feat
        raise ValueError("MMDLoss expects feature tensors with ndim in {2, 4, 5}")

    def _reshape_weights(self, weight: torch.Tensor, feat_ndim: int) -> torch.Tensor:
        if feat_ndim == 5:
            if weight.ndim == 4:
                weight = weight.unsqueeze(1)
            if weight.ndim != 5 or weight.shape[1] != 1:
                raise ValueError("3D MMD weights must have shape [B, 1, D, H, W] or [B, D, H, W]")
            weight = F.adaptive_avg_pool3d(weight, self.pooled_size)
            return weight.reshape(-1)
        if feat_ndim == 4:
            if weight.ndim == 3:
                weight = weight.unsqueeze(1)
            if weight.ndim != 4 or weight.shape[1] != 1:
                raise ValueError("2D MMD weights must have shape [B, 1, H, W] or [B, H, W]")
            weight = F.adaptive_avg_pool2d(weight, self.pooled_size[:2])
            return weight.reshape(-1)
        if feat_ndim == 2:
            if weight.ndim == 2 and weight.shape[1] == 1:
                weight = weight[:, 0]
            if weight.ndim != 1:
                raise ValueError("Vector MMD weights must have shape [N] or [N, 1]")
            return weight
        raise ValueError("MMDLoss expects feature tensors with ndim in {2, 4, 5}")

    def _normalize_sample_weight(
        self,
        feat: torch.Tensor,
        weight: torch.Tensor | None,
    ) -> torch.Tensor:
        sample_count = self._reshape_samples(feat).shape[0]
        if weight is None:
            return feat.new_full((sample_count,), 1.0 / max(sample_count, 1))

        weight = self._reshape_weights(weight.to(dtype=feat.dtype, device=feat.device), feat.ndim)
        if weight.shape[0] != sample_count:
            raise ValueError(
                f"MMD weight sample count mismatch: got {weight.shape[0]}, expected {sample_count}"
            )
        weight = weight.clamp_min(0.0)
        total = weight.sum()
        if total <= 0:
            return feat.new_full((sample_count,), 1.0 / max(sample_count, 1))
        return weight / total

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_weight: torch.Tensor | None = None,
        target_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_vec = self._reshape_samples(source)
        target_vec = self._reshape_samples(target)
        source_prob = self._normalize_sample_weight(source, source_weight)
        target_prob = self._normalize_sample_weight(target, target_weight)

        xx = torch.cdist(source_vec, source_vec, p=2).pow(2)
        yy = torch.cdist(target_vec, target_vec, p=2).pow(2)
        xy = torch.cdist(source_vec, target_vec, p=2).pow(2)

        mmd = source_vec.new_tensor(0.0)
        for alpha in self.kernel_alphas:
            k_xx = torch.exp(-alpha * xx)
            k_yy = torch.exp(-alpha * yy)
            k_xy = torch.exp(-alpha * xy)

            #  batch
            mmd += (
                (source_prob[:, None] * source_prob[None, :] * k_xx).sum()
                + (target_prob[:, None] * target_prob[None, :] * k_yy).sum()
                - 2.0 * (source_prob[:, None] * target_prob[None, :] * k_xy).sum()
            )

        return mmd


class UncertaintyRegressionLoss(nn.Module):
    """ uncertainty_map"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        uncertainty_map: torch.Tensor,
        warped_img: torch.Tensor,
        fixed_img: torch.Tensor,
    ) -> torch.Tensor:
        if uncertainty_map.ndim != 5 or uncertainty_map.shape[1] != 1:
            raise ValueError("uncertainty_map must be [B, 1, D, H, W]")
        error_map = torch.abs(warped_img - fixed_img).detach()
        error_min = error_map.amin(dim=(2, 3, 4), keepdim=True)
        error_max = error_map.amax(dim=(2, 3, 4), keepdim=True)
        target = (error_map - error_min) / (error_max - error_min + self.eps)
        return F.mse_loss(uncertainty_map, target)


class WeightedCrossEntropyLoss(nn.Module):
    """


    -  segmentation  CE
    -  uncertainty-weighted
    -  `nn.Module`  `criterions`


    - logits:  `[B, C, D, H, W]`
    - target:  `[B, D, H, W]`  `[B, 1, D, H, W]`
    - weight_map:  `[B, 1, D, H, W]`


    -
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # `F.cross_entropy`  target
        if target.ndim == 5 and target.shape[1] == 1:
            target = target[:, 0].long()
        else:
            target = target.long()

        #  CE
        ce_map = F.cross_entropy(logits, target, reduction="none")
        if weight_map is None:
            return ce_map.mean()

        # weight_map  `[B, D, H, W]`  ce_map
        weight = weight_map[:, 0].float().clamp(0.0, 1.0)
        return (ce_map * weight).sum() / (weight.sum() + 1e-6)


class DiceLoss(nn.Module):
    """`dice_loss()` """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(logits, target, eps=self.eps)


class ContrastiveFeatureLoss(nn.Module):
    """NT-Xent  CLMorph """

    def __init__(self, temperature: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        if feat_a.ndim != 2 or feat_b.ndim != 2:
            raise ValueError("feat_a and feat_b must be [B, C]")
        if feat_a.shape != feat_b.shape:
            raise ValueError("feat_a and feat_b must share the same shape")

        batch_size = feat_a.shape[0]
        if batch_size <= 1:
            return feat_a.new_zeros(())

        z_a = F.normalize(feat_a, dim=1, eps=self.eps)
        z_b = F.normalize(feat_b, dim=1, eps=self.eps)
        reps = torch.cat([z_a, z_b], dim=0)
        sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2)  #  2B x 2B

        pos_ab = torch.diag(sim, batch_size)
        pos_ba = torch.diag(sim, -batch_size)
        positives = torch.cat([pos_ab, pos_ba], dim=0)  #

        logits = torch.exp(sim / self.temperature)
        negatives_mask = ~torch.eye(batch_size * 2, dtype=torch.bool, device=feat_a.device)
        denom = (logits * negatives_mask).sum(dim=1).clamp_min(self.eps)
        numer = torch.exp(positives / self.temperature).clamp_min(self.eps)
        return (-torch.log(numer / denom)).mean()

class SymmetricMultiscaleLNCCLoss(nn.Module):
    """
     LNCC


    1)
    2)  half-flow  moving/fixed
    3)  LNCC
    """

    def __init__(self, win: int = 9, eps: float = 1e-15, scales: list[int] = None):
        super().__init__()
        self.win = win
        self.eps = eps
        self.scales = scales if scales is not None else [1, 2, 4]  #

    def forward(
        self,
        fixed_img: torch.Tensor,
        moving_img: torch.Tensor,
        flow_list: list[torch.Tensor],
    ) -> torch.Tensor:
        total = fixed_img.new_zeros(())        # /
        base_size = fixed_img.shape[2:]        #
        for i, flow in enumerate(flow_list[: len(self.scales)]):
            s = self.scales[i]
            if s == 1:
                f_s = fixed_img
                m_s = moving_img
                flow_s = flow
            else:
                size_s = tuple(max(1, d // s) for d in base_size)                    #
                f_s = F.interpolate(fixed_img, size=size_s, mode="trilinear", align_corners=True)
                m_s = F.interpolate(moving_img, size=size_s, mode="trilinear", align_corners=True)
                flow_s = F.interpolate(flow, size=size_s, mode="trilinear", align_corners=True)
                ratio = [size_s[0] / base_size[0], size_s[1] / base_size[1], size_s[2] / base_size[2]]
                flow_s[:, 0] *= ratio[0]                                                #
                flow_s[:, 1] *= ratio[1]
                flow_s[:, 2] *= ratio[2]

            half_flow = 0.5 * flow_s                                                    # half-way
            m_half = warp_3d(m_s, half_flow, mode="bilinear")                          # moving
            f_half = warp_3d(f_s, -half_flow, mode="bilinear")                         # fixed
            total = total + lncc_loss(f_half, m_half, win=self.win, eps=self.eps)     #
        return total

# ==============================
#
#  nn.Module
# ==============================


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice


    -  CE


    - target  [B,D,H,W] / [B,1,D,H,W]
    -  one-hot  [B,C,D,H,W]
    """

    probs = torch.softmax(logits, dim=1)  #  logits
    if target.ndim == logits.ndim - 1:
        target_oh = torch.nn.functional.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
    elif target.ndim == logits.ndim and target.shape[1] == 1:
        target_oh = torch.nn.functional.one_hot(target[:, 0].long(), num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
    else:
        target_oh = target.float()
    inter = (probs * target_oh).sum(dim=(2, 3, 4))                  #
    denom = probs.sum(dim=(2, 3, 4)) + target_oh.sum(dim=(2, 3, 4)) #
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()                                         #


def ncc_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
     NCC 1 - Pearson


    -
    -  LNCC
    """

    x_mean = x.mean(dim=(2, 3, 4), keepdim=True)
    y_mean = y.mean(dim=(2, 3, 4), keepdim=True)
    x_center = x - x_mean
    y_center = y - y_mean
    numerator = (x_center * y_center).mean(dim=(2, 3, 4))  #
    denominator = torch.sqrt(
        (x_center.square().mean(dim=(2, 3, 4)) + eps)
        * (y_center.square().mean(dim=(2, 3, 4)) + eps)
    )  #
    return (1.0 - numerator / denominator).mean()


def lncc_loss(x: torch.Tensor, y: torch.Tensor, win: int = 9, eps: float = 1e-5) -> torch.Tensor:
    """
     NCCLNCC


    1)  3D
    2)
    3)
    """

    if x.shape != y.shape:
        raise ValueError("x and y must share the same shape")
    if x.ndim != 5:
        raise ValueError("lncc_loss currently supports 3D tensors [B, C, D, H, W]")

    channel = x.shape[1]
    filt = torch.ones(channel, 1, win, win, win, device=x.device, dtype=x.dtype)  #
    pad = win // 2

    def _sum_pool(t: torch.Tensor) -> torch.Tensor:
        return F.conv3d(t, filt, padding=pad, groups=channel)  # groups=channel

    x2 = x * x
    y2 = y * y
    xy = x * y

    x_sum = _sum_pool(x)
    y_sum = _sum_pool(y)
    x2_sum = _sum_pool(x2)
    y2_sum = _sum_pool(y2)
    xy_sum = _sum_pool(xy)

    win_size = float(win ** 3)
    x_mean = x_sum / win_size
    y_mean = y_sum / win_size

    cross = xy_sum - y_mean * x_sum - x_mean * y_sum + x_mean * y_mean * win_size
    x_var = x2_sum - 2 * x_mean * x_sum + x_mean * x_mean * win_size
    y_var = y2_sum - 2 * y_mean * y_sum + y_mean * y_mean * win_size
    x_var = torch.clamp(x_var, min=eps)                              #  0
    y_var = torch.clamp(y_var, min=eps)
    ncc = (cross * cross) / (x_var * y_var + eps)                    #
    ncc = torch.nan_to_num(ncc, nan=0.0, posinf=0.0, neginf=0.0)     #
    return 1.0 - ncc.mean()


def kl_divergence_loss(mu: torch.Tensor | None, logvar: torch.Tensor | None) -> torch.Tensor:
    """
     VAE KL


    -  mu/logvar  VAE
    """

    if mu is None or logvar is None:
        ref = mu if mu is not None else logvar
        if ref is None:
            return torch.tensor(0.0)                              #  CPU
        return torch.zeros((), device=ref.device, dtype=ref.dtype) # /
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


def smoothness_loss(svf: torch.Tensor) -> torch.Tensor:
    """
     FlowGradLoss
    """

    if svf.ndim != 5 or svf.shape[1] != 3:
        raise ValueError("svf must be [B, 3, D, H, W]")
    dz = svf[:, :, 1:, :, :] - svf[:, :, :-1, :, :]
    dy = svf[:, :, :, 1:, :] - svf[:, :, :, :-1, :]
    dx = svf[:, :, :, :, 1:] - svf[:, :, :, :, :-1]
    return (dz.square().mean() + dy.square().mean() + dx.square().mean()) / 3.0


def soft_dice_cycle_loss(
    atlas_label: torch.Tensor,
    phi_fwd: torch.Tensor,
    phi_bak: torch.Tensor,
    num_classes: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
     Dice


    -  atlas_label  phi_fwd phi_bak
    -
    """

    if atlas_label.ndim == 5 and atlas_label.shape[1] == 1:
        label_index = atlas_label[:, 0].long()
    elif atlas_label.ndim == 4:
        label_index = atlas_label.long()
    else:
        raise ValueError("atlas_label must be [B,1,D,H,W] or [B,D,H,W]")

    label_oh = F.one_hot(label_index, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    y_fwd = warp_3d(label_oh, phi_fwd, mode="bilinear")   #
    y_recon = warp_3d(y_fwd, phi_bak, mode="bilinear")    #

    inter = (label_oh * y_recon).sum(dim=(2, 3, 4))
    denom = label_oh.sum(dim=(2, 3, 4)) + y_recon.sum(dim=(2, 3, 4))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def reg_unet_vae_terms(
    producer_out: dict[str, torch.Tensor | None],
    atlas_img: torch.Tensor,
    atlas_label: torch.Tensor,
    real_img: torch.Tensor | None = None,
    num_classes: int | None = None,
    lncc_win: int = 9,
) -> dict[str, torch.Tensor]:
    """
     Reg-UNet-VAE


    - loss_sim:
    - loss_kl: VAE KL
    - loss_smooth: SVF
    - loss_cycle:
    """

    x_syn = producer_out["x_syn"]                 #
    phi_fwd = producer_out["phi_fwd"]             #
    phi_bak = producer_out["phi_bak"]             #
    svf = producer_out["svf"]                     #
    mu = producer_out.get("mu", None)             #
    logvar = producer_out.get("logvar", None)     #

    target_img = real_img if real_img is not None else atlas_img   #  atlas
    loss_sim = lncc_loss(x_syn, target_img, win=lncc_win)
    loss_kl = kl_divergence_loss(mu, logvar)
    if isinstance(loss_kl, torch.Tensor) and loss_kl.device != loss_sim.device:
        loss_kl = loss_kl.to(device=loss_sim.device, dtype=loss_sim.dtype)  #
    loss_smooth = smoothness_loss(svf)

    if num_classes is None:
        num_classes = int(atlas_label.max().detach().item()) + 1   #
        num_classes = max(num_classes, 2)                           #
    loss_cycle = soft_dice_cycle_loss(
        atlas_label=atlas_label,
        phi_fwd=phi_fwd,
        phi_bak=phi_bak,
        num_classes=num_classes,
    )
    return {
        "loss_sim": loss_sim,
        "loss_kl": loss_kl,
        "loss_smooth": loss_smooth,
        "loss_cycle": loss_cycle,
    }
