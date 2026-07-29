"""
损失函数模块（losses.py）

设计目标：
1) 提供配准与分割场景常用损失（NCC/LNCC、平滑、Jacobian、KL、Dice）；
2) 提供模块化封装（nn.Module）与函数式实现，便于在不同训练脚本复用；
3) 在概率配准与循环一致性场景中，提供可组合的辅助项。

说明：
- 本文件不涉及数据读取与文件写入；重点是数值计算与训练信号构建。
- 输入张量默认采用 3D 医学图像格式 [B, C, D, H, W]。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import warp_3d


# ==============================
# 配准主损失与正则
# ==============================
class VxmDiffReconLoss(nn.Module):
    """
    基于 MSE 的重建/相似度损失。

    目的：
    - 约束形变后 moving 图像与 fixed 图像在体素级别的差异；
    - 作为概率模型中的“重建项”，与 KL 项共同优化。

    输入：
    - fixed_img: 固定图像，形状 [B, C, D, H, W]
    - warped_moving_img: 形变后的移动图像，形状需与 fixed_img 一致

    输出：
    - 标量损失（越小越好）。
    """

    def __init__(self, sigma_sq: float = 1.0, sample_count: int = 1):
        super().__init__()
        self.sigma_sq = sigma_sq          # 似然方差超参数（高斯噪声假设）
        self.sample_count = sample_count  # 蒙特卡洛样本数（用于缩放重建项）

    def forward(self, fixed_img: torch.Tensor, warped_moving_img: torch.Tensor) -> torch.Tensor:
        if fixed_img.shape != warped_moving_img.shape:
            raise ValueError("fixed_img and warped_moving_img must share the same shape")
        if self.sigma_sq <= 0:
            raise ValueError("sigma_sq must be positive")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")

        mse = (fixed_img - warped_moving_img).pow(2).mean()                 # 体素级均方误差
        return mse / (2.0 * self.sigma_sq * float(self.sample_count))       # 高斯负对数似然的比例形式


class VxmDiffKLLoss(nn.Module):
    """
    概率配准中的 KL 散度损失。

    目的：
    - 约束模型预测分布接近先验分布，避免隐空间发散；
    - 保留采样多样性，支持不确定性建模。

    输入：
    - v_mean: 隐变量均值
    - v_log_sigma: 隐变量对数标准差（注意不是 logvar）
    """

    def __init__(self, prior_lambda: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.prior_lambda = prior_lambda  # 先验权重系数
        self.eps = eps                    # 防止 log(0)

    def forward(self, v_mean: torch.Tensor, v_log_sigma: torch.Tensor) -> torch.Tensor:
        if v_mean.shape != v_log_sigma.shape:
            raise ValueError("v_mean and v_log_sigma must share the same shape")
        if self.prior_lambda <= 0:
            raise ValueError("prior_lambda must be positive")

        var = torch.exp(2.0 * v_log_sigma)                                        # 由 log_sigma 恢复方差
        log_var = torch.log(var + self.eps)                                       # 数值稳定的 log(var)
        kl = 0.5 * (self.prior_lambda * (v_mean.pow(2) + var) - log_var - 1.0)   # KL(q||p) 逐元素表达式
        return kl.mean()                                                           # 标量化


class NCC(nn.Module):
    """
    LNCC 的模块封装（接口风格与其他 loss 统一）。

    关键逻辑：
    - 在 CUDA 情况下禁用 autocast，统一使用 float 计算 LNCC，避免低精度导致分母数值不稳定。
    """

    def __init__(self, win: int = 9):
        super().__init__()
        self.win = win  # 局部窗口边长

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        if y_true.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                return lncc_loss(y_true.float(), y_pred.float(), win=self.win)  # GPU 下强制 FP32
        return lncc_loss(y_true.float(), y_pred.float(), win=self.win)           # CPU 路径


class ROIWeightedLNCC(nn.Module):
    """
    ROI 加权 LNCC。

    目的：
    - 让 refinement 在 selected ROI / boundary 附近获得更高训练权重；
    - 不裁剪图像，也不限制 flow，只改变图像相似性损失的空间关注度。
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
    形变场全局平滑损失（扩散正则）。

    目的：
    - 限制 flow/SVF 在空间上剧烈震荡，提升形变场可解释性与可逆性。
    """

    def __init__(self):
        super().__init__()

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 5 or flow.shape[1] != 3:
            raise ValueError("flow must be [B, 3, D, H, W]")
        dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]   # z 方向一阶差分
        dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]   # y 方向一阶差分
        dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]   # x 方向一阶差分
        return (dz.square().mean() + dy.square().mean() + dx.square().mean()) / 3.0


class JacobianDeterminantLoss(nn.Module):
    """
    Jacobian 行列式惩罚。

    目的：
    - 对 det(J) < eps 的位置施加惩罚，抑制折叠/翻转（局部拓扑破坏）。

    注意：
    - 该实现将 flow 看作位移场 u，真实 Jacobian 为 I + ∇u。
    - 使用前向差分，边界采用裁剪后的公共体素区域。
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # 判定“不可接受 Jacobian”的阈值

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 5 or flow.shape[1] != 3:
            raise ValueError("flow must be [B, 3, D, H, W]")

        u = flow
        duz_dz = u[:, 0, 1:, :-1, :-1] - u[:, 0, :-1, :-1, :-1]  # ∂u_z/∂z
        duz_dy = u[:, 0, :-1, 1:, :-1] - u[:, 0, :-1, :-1, :-1]  # ∂u_z/∂y
        duz_dx = u[:, 0, :-1, :-1, 1:] - u[:, 0, :-1, :-1, :-1]  # ∂u_z/∂x

        duy_dz = u[:, 1, 1:, :-1, :-1] - u[:, 1, :-1, :-1, :-1]  # ∂u_y/∂z
        duy_dy = u[:, 1, :-1, 1:, :-1] - u[:, 1, :-1, :-1, :-1]  # ∂u_y/∂y
        duy_dx = u[:, 1, :-1, :-1, 1:] - u[:, 1, :-1, :-1, :-1]  # ∂u_y/∂x

        dux_dz = u[:, 2, 1:, :-1, :-1] - u[:, 2, :-1, :-1, :-1]  # ∂u_x/∂z
        dux_dy = u[:, 2, :-1, 1:, :-1] - u[:, 2, :-1, :-1, :-1]  # ∂u_x/∂y
        dux_dx = u[:, 2, :-1, :-1, 1:] - u[:, 2, :-1, :-1, :-1]  # ∂u_x/∂x

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
        )  # 3x3 行列式展开
        return torch.relu(self.eps - det).mean()  # 仅惩罚 det < eps 的区域


class MaskedFlowGradLoss(nn.Module):
    """
    掩码平滑损失（空间加权版平滑正则）。

    目的：
    - 在不同空间位置施加不同平滑约束强度；
    - 常用于“不确定/病理区域放松约束，健康区域保持平滑”的场景。

    输入：
    - flow: [B, 3, D, H, W]
    - weight: [B, 1, D, H, W]，通常取值 [0,1]
    """

    def __init__(self):
        super().__init__()

    def forward(self, flow: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 5 or flow.shape[1] != 3:
            raise ValueError("flow must be [B, 3, D, H, W]")
        if weight.ndim != 5 or weight.shape[1] != 1:
            raise ValueError("weight must be [B, 1, D, H, W]")

        dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]  # z 梯度
        dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]  # y 梯度
        dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]  # x 梯度

        # 权重需要与差分张量在空间维度对齐
        wz = weight[:, :, 1:, :, :]
        wy = weight[:, :, :, 1:, :]
        wx = weight[:, :, :, :, 1:]

        # 分母中乘 flow.shape[1] 是为了与通道维度归一化一致
        dz_loss = (dz.square() * wz).sum() / (wz.sum() * flow.shape[1] + 1e-6)
        dy_loss = (dy.square() * wy).sum() / (wy.sum() * flow.shape[1] + 1e-6)
        dx_loss = (dx.square() * wx).sum() / (wx.sum() * flow.shape[1] + 1e-6)
        return (dz_loss + dy_loss + dx_loss) / 3.0


class KLDivergenceLoss(nn.Module):
    """
    KL 函数式实现的模块包装器。

    目的：
    - 统一工程调用习惯（与其他 nn.Module loss 一致）；
    - 实际计算由 kl_divergence_loss 执行。
    """

    def __init__(self):
        super().__init__()

    def forward(self, mu: torch.Tensor | None, logvar: torch.Tensor | None) -> torch.Tensor:
        return kl_divergence_loss(mu, logvar)  # 直接转发，避免重复实现


class MeanAbsLoss(nn.Module):
    """返回输入张量绝对值均值。"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(x))


class GradientConsistencyLoss(nn.Module):
    """比较两幅 3D 图像的一阶梯度一致性。"""

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
    """约束 common/diff 向量低相关（近似正交）。"""

    def __init__(self):
        super().__init__()

    def forward(self, common_vec: torch.Tensor, diff_vec: torch.Tensor) -> torch.Tensor:
        common_vec = F.normalize(common_vec, dim=1)
        diff_vec = F.normalize(diff_vec, dim=1)
        return torch.abs((common_vec * diff_vec).sum(dim=1)).mean()


class CommonConsistencyLoss(nn.Module):
    """约束 common 向量在 moving/fixed 间保持一致。"""

    def __init__(self):
        super().__init__()

    def forward(self, common_a_vec: torch.Tensor, common_f_vec: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(common_a_vec, common_f_vec)


class CosineCommonConsistencyLoss(nn.Module):
    """基于余弦相似度约束 common 向量方向一致。"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, common_a_vec: torch.Tensor, common_f_vec: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(common_a_vec, common_f_vec, dim=1, eps=self.eps)
        return (1.0 - sim).mean()


class CosineCommonDiffLoss(nn.Module):
    """统一约束 common 对齐与 common/diff 解耦。"""

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
    """约束 diff_gap 既非零塌缩，又具有空间分布。"""

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
    """计算两组特征的 Maximum Mean Discrepancy (MMD) 以衡量分布差异。

    使用多核高斯 (Multi-kernel Gaussian) 提升对不同尺度分布差异的捕捉能力。
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
        """把不同维度特征统一压平成 `[N, C]` 样本表征，便于后续做核距离。"""

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

            # 使用概率权重的有偏估计，兼容单样本 batch 和空间置信度加权。
            mmd += (
                (source_prob[:, None] * source_prob[None, :] * k_xx).sum()
                + (target_prob[:, None] * target_prob[None, :] * k_yy).sum()
                - 2.0 * (source_prob[:, None] * target_prob[None, :] * k_xy).sum()
            )

        return mmd


class UncertaintyRegressionLoss(nn.Module):
    """用配准误差代理监督 uncertainty_map。"""

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
    """支持可选空间权重图的交叉熵损失。

    目的：
    - 统一 segmentation 任务中的 CE 口径；
    - 同时兼容普通监督与 uncertainty-weighted 伪监督；
    - 保持接口风格与其他 `nn.Module` 类一致，便于放入 `criterions` 字典统一管理。

    输入：
    - logits: 分割网络输出，形状 `[B, C, D, H, W]`
    - target: 标签，形状可为 `[B, D, H, W]` 或 `[B, 1, D, H, W]`
    - weight_map: 可选空间权重图，形状 `[B, 1, D, H, W]`

    输出：
    - 标量损失
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # `F.cross_entropy` 需要类别索引，因此先统一 target 的格式。
        if target.ndim == 5 and target.shape[1] == 1:
            target = target[:, 0].long()
        else:
            target = target.long()

        # 先得到逐体素 CE，后续再决定是否做空间加权。
        ce_map = F.cross_entropy(logits, target, reduction="none")
        if weight_map is None:
            return ce_map.mean()

        # weight_map 为单通道体素权重图，与 `[B, D, H, W]` 的 ce_map 对齐后逐点加权。
        weight = weight_map[:, 0].float().clamp(0.0, 1.0)
        return (ce_map * weight).sum() / (weight.sum() + 1e-6)


class DiceLoss(nn.Module):
    """`dice_loss()` 的模块封装器。"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(logits, target, eps=self.eps)


class ContrastiveFeatureLoss(nn.Module):
    """NT-Xent 风格对比损失，用于 CLMorph 类特征约束。"""

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
        sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2)  # 构造 2B x 2B 相似度矩阵

        pos_ab = torch.diag(sim, batch_size)
        pos_ba = torch.diag(sim, -batch_size)
        positives = torch.cat([pos_ab, pos_ba], dim=0)  # 对角偏移位置正好是正样本对

        logits = torch.exp(sim / self.temperature)
        negatives_mask = ~torch.eye(batch_size * 2, dtype=torch.bool, device=feat_a.device)
        denom = (logits * negatives_mask).sum(dim=1).clamp_min(self.eps)
        numer = torch.exp(positives / self.temperature).clamp_min(self.eps)
        return (-torch.log(numer / denom)).mean()

class SymmetricMultiscaleLNCCLoss(nn.Module):
    """
    对称多尺度 LNCC 损失。

    核心思想（分步骤）：
    1) 在多个尺度上评估配准质量；
    2) 对每个尺度使用 half-flow 将 moving/fixed 双向拉近（对称）；
    3) 逐尺度求 LNCC 并累加，提升对大形变与细节的一致约束。
    """

    def __init__(self, win: int = 9, eps: float = 1e-15, scales: list[int] = None):
        super().__init__()
        self.win = win
        self.eps = eps
        self.scales = scales if scales is not None else [1, 2, 4]  # 默认三尺度

    def forward(
        self,
        fixed_img: torch.Tensor,
        moving_img: torch.Tensor,
        flow_list: list[torch.Tensor],
    ) -> torch.Tensor:
        total = fixed_img.new_zeros(())        # 使用输入设备/精度创建标量累加器
        base_size = fixed_img.shape[2:]        # 原始体素空间尺寸
        for i, flow in enumerate(flow_list[: len(self.scales)]):
            s = self.scales[i]
            if s == 1:
                f_s = fixed_img
                m_s = moving_img
                flow_s = flow
            else:
                size_s = tuple(max(1, d // s) for d in base_size)                    # 尺度下采样尺寸
                f_s = F.interpolate(fixed_img, size=size_s, mode="trilinear", align_corners=True)
                m_s = F.interpolate(moving_img, size=size_s, mode="trilinear", align_corners=True)
                flow_s = F.interpolate(flow, size=size_s, mode="trilinear", align_corners=True)
                ratio = [size_s[0] / base_size[0], size_s[1] / base_size[1], size_s[2] / base_size[2]]
                flow_s[:, 0] *= ratio[0]                                                # 位移幅值按缩放比重标定
                flow_s[:, 1] *= ratio[1]
                flow_s[:, 2] *= ratio[2]

            half_flow = 0.5 * flow_s                                                    # half-way 变换
            m_half = warp_3d(m_s, half_flow, mode="bilinear")                          # moving 前向半步
            f_half = warp_3d(f_s, -half_flow, mode="bilinear")                         # fixed 反向半步
            total = total + lncc_loss(f_half, m_half, win=self.win, eps=self.eps)     # 累加该尺度损失
        return total

# ==============================
# 函数式实现
# 方便在脚本里直接组合，不强制走 nn.Module 接口。
# ==============================


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice 损失。

    目的：
    - 在类别不平衡时比 CE 更稳定，直接优化区域重叠。

    输入兼容：
    - target 可为索引标签 [B,D,H,W] / [B,1,D,H,W]；
    - 也可为 one-hot 概率图 [B,C,D,H,W]。
    """

    probs = torch.softmax(logits, dim=1)  # 将 logits 转换为类别概率
    if target.ndim == logits.ndim - 1:
        target_oh = torch.nn.functional.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
    elif target.ndim == logits.ndim and target.shape[1] == 1:
        target_oh = torch.nn.functional.one_hot(target[:, 0].long(), num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
    else:
        target_oh = target.float()
    inter = (probs * target_oh).sum(dim=(2, 3, 4))                  # 交集
    denom = probs.sum(dim=(2, 3, 4)) + target_oh.sum(dim=(2, 3, 4)) # 并集替代表达
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()                                         # 转为“越小越好”的损失


def ncc_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    全局 NCC 损失（1 - Pearson 相关系数）。

    适用：
    - 偏向对齐全局统计趋势，计算便宜；
    - 精细配准一般建议优先使用 LNCC。
    """

    x_mean = x.mean(dim=(2, 3, 4), keepdim=True)
    y_mean = y.mean(dim=(2, 3, 4), keepdim=True)
    x_center = x - x_mean
    y_center = y - y_mean
    numerator = (x_center * y_center).mean(dim=(2, 3, 4))  # 协方差项
    denominator = torch.sqrt(
        (x_center.square().mean(dim=(2, 3, 4)) + eps)
        * (y_center.square().mean(dim=(2, 3, 4)) + eps)
    )  # 方差归一化项
    return (1.0 - numerator / denominator).mean()


def lncc_loss(x: torch.Tensor, y: torch.Tensor, win: int = 9, eps: float = 1e-5) -> torch.Tensor:
    """
    局部 NCC（LNCC）损失。

    关键算法：
    1) 利用分组 3D 卷积在滑窗内做求和（代替逐窗循环）；
    2) 根据二阶矩公式计算局部协方差与方差；
    3) 得到局部相关系数后取均值并转为损失。
    """

    if x.shape != y.shape:
        raise ValueError("x and y must share the same shape")
    if x.ndim != 5:
        raise ValueError("lncc_loss currently supports 3D tensors [B, C, D, H, W]")

    channel = x.shape[1]
    filt = torch.ones(channel, 1, win, win, win, device=x.device, dtype=x.dtype)  # 每通道独立窗口核
    pad = win // 2

    def _sum_pool(t: torch.Tensor) -> torch.Tensor:
        return F.conv3d(t, filt, padding=pad, groups=channel)  # groups=channel 实现逐通道卷积

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
    x_var = torch.clamp(x_var, min=eps)                              # 防止分母接近 0
    y_var = torch.clamp(y_var, min=eps)
    ncc = (cross * cross) / (x_var * y_var + eps)                    # 相关系数平方形式
    ncc = torch.nan_to_num(ncc, nan=0.0, posinf=0.0, neginf=0.0)     # 屏蔽异常数值
    return 1.0 - ncc.mean()


def kl_divergence_loss(mu: torch.Tensor | None, logvar: torch.Tensor | None) -> torch.Tensor:
    """
    标准 VAE KL 损失函数。

    技术约束：
    - 允许 mu/logvar 为空（例如某些路径不启用 VAE），返回零损失以保持训练流程可复用。
    """

    if mu is None or logvar is None:
        ref = mu if mu is not None else logvar
        if ref is None:
            return torch.tensor(0.0)                              # 完全无参考时返回 CPU 标量
        return torch.zeros((), device=ref.device, dtype=ref.dtype) # 保持设备/精度一致
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


def smoothness_loss(svf: torch.Tensor) -> torch.Tensor:
    """
    函数式平滑损失（与 FlowGradLoss 等价）。
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
    软 Dice 循环一致性损失。

    业务逻辑：
    - 将 atlas_label 先经前向形变 phi_fwd，再经反向形变 phi_bak；
    - 要求重建标签接近原始标签，以约束前后向映射互逆一致性。
    """

    if atlas_label.ndim == 5 and atlas_label.shape[1] == 1:
        label_index = atlas_label[:, 0].long()
    elif atlas_label.ndim == 4:
        label_index = atlas_label.long()
    else:
        raise ValueError("atlas_label must be [B,1,D,H,W] or [B,D,H,W]")

    label_oh = F.one_hot(label_index, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    y_fwd = warp_3d(label_oh, phi_fwd, mode="bilinear")   # 正向形变
    y_recon = warp_3d(y_fwd, phi_bak, mode="bilinear")    # 反向重建

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
    汇总 Reg-UNet-VAE 训练所需损失项。

    输出字典：
    - loss_sim: 图像相似度
    - loss_kl: VAE KL
    - loss_smooth: SVF 平滑
    - loss_cycle: 标签循环一致性
    """

    x_syn = producer_out["x_syn"]                 # 合成图像
    phi_fwd = producer_out["phi_fwd"]             # 前向形变
    phi_bak = producer_out["phi_bak"]             # 反向形变
    svf = producer_out["svf"]                     # 速度场
    mu = producer_out.get("mu", None)             # 变分均值（可选）
    logvar = producer_out.get("logvar", None)     # 变分对数方差（可选）

    target_img = real_img if real_img is not None else atlas_img   # 无真实图时退化为 atlas 自重建
    loss_sim = lncc_loss(x_syn, target_img, win=lncc_win)
    loss_kl = kl_divergence_loss(mu, logvar)
    if isinstance(loss_kl, torch.Tensor) and loss_kl.device != loss_sim.device:
        loss_kl = loss_kl.to(device=loss_sim.device, dtype=loss_sim.dtype)  # 统一设备与精度
    loss_smooth = smoothness_loss(svf)

    if num_classes is None:
        num_classes = int(atlas_label.max().detach().item()) + 1   # 由标签动态推断类别数
        num_classes = max(num_classes, 2)                           # 至少保留二分类语义
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
