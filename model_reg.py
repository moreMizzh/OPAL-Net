"""OPAL-Net reg 模型实现。

设计目标：
- 以 `v8` 主链为基础迁移到 `v9`；
- reg 保留 `A + B` 的 coarse-to-fine 主体，但移除 B 的 domain / uncertainty 结构；
- 去掉实验性兼容层，保持主链扁平。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# Reg 基础组件
# ==============================
class SpatialTransformer(nn.Module):
    """3D 重采样层（backward warping，基于 grid_sample）。"""

    def __init__(self, size, mode: str = "bilinear", padding_mode: str = "border"):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(*vectors, indexing="ij")
        grid = torch.stack(grids).unsqueeze(0).float()  # [1, 3, D, H, W]
        self.register_buffer("grid", grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        # `grid_sample` 使用归一化坐标，因此这里把 voxel 坐标映射到 [-1, 1]。
        for i in range(len(shape)):
            denom = max(shape[i] - 1, 1)
            new_locs[:, i, ...] = 2.0 * (new_locs[:, i, ...] / denom - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return F.grid_sample(
            src,
            new_locs,
            mode=self.mode,
            align_corners=True,
            padding_mode=self.padding_mode,
        )


class VecInt(nn.Module):
    """SVF -> flow 的 scaling-and-squaring 积分器。"""

    def __init__(self, inshape, nsteps: int = 7):
        super().__init__()
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ConvBlock(nn.Module):
    """基础 3D 卷积块：Conv3d + IN + LeakyReLU。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedEncoder(nn.Module):
    """多尺度共享编码器（每层 stride=2 下采样）。"""

    def __init__(self, in_channels: int = 1, channels: tuple[int, int, int, int] = (16, 32, 64, 96)):
        super().__init__()
        prev = in_channels
        self.layers = nn.ModuleList()
        for ch in channels:
            self.layers.append(ConvBlock(prev, ch, stride=2))
            prev = ch

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class DualDecoder(nn.Module):
    """Stage A 解码器：融合 moving/fixed skip，逐级恢复分辨率。"""

    def __init__(
        self,
        bottleneck_channels: int,
        skip_channels: tuple[int, int, int] = (64, 32, 16),
        out_channels: tuple[int, int, int, int] = (64, 32, 16, 16),
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(bottleneck_channels + skip_channels[0] * 2, out_channels[0]),
                ConvBlock(out_channels[0] + skip_channels[1] * 2, out_channels[1]),
                ConvBlock(out_channels[1] + skip_channels[2] * 2, out_channels[2]),
                ConvBlock(out_channels[2] + 2, out_channels[3]),
            ]
        )
        self.out_channels = out_channels[-1]

    def forward(
        self,
        bottleneck: torch.Tensor,
        feats_a: list[torch.Tensor],
        feats_b: list[torch.Tensor],
        img_a: torch.Tensor,
        img_b: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(bottleneck)
        x = self.blocks[0](torch.cat([x, feats_a[-2], feats_b[-2]], dim=1))
        x = self.up(x)
        x = self.blocks[1](torch.cat([x, feats_a[-3], feats_b[-3]], dim=1))
        x = self.up(x)
        x = self.blocks[2](torch.cat([x, feats_a[-4], feats_b[-4]], dim=1))
        x = self.up(x)
        x = self.blocks[3](torch.cat([x, img_a, img_b], dim=1))
        return x


class DualDecoder3Level(nn.Module):
    """三层版 Stage A 解码器，配合浅层 coarse encoder 使用。"""

    def __init__(
        self,
        bottleneck_channels: int,
        skip_channels: tuple[int, int] = (32, 16),
        out_channels: tuple[int, int, int] = (32, 16, 16),
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(bottleneck_channels + skip_channels[0] * 2, out_channels[0]),
                ConvBlock(out_channels[0] + skip_channels[1] * 2, out_channels[1]),
                ConvBlock(out_channels[1] + 2, out_channels[2]),
            ]
        )
        self.out_channels = out_channels[-1]

    def forward(
        self,
        bottleneck: torch.Tensor,
        feats_a: list[torch.Tensor],
        feats_b: list[torch.Tensor],
        img_a: torch.Tensor,
        img_b: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(bottleneck)
        x = self.blocks[0](torch.cat([x, feats_a[-2], feats_b[-2]], dim=1))
        x = self.up(x)
        x = self.blocks[1](torch.cat([x, feats_a[-3], feats_b[-3]], dim=1))
        x = self.up(x)
        x = self.blocks[2](torch.cat([x, img_a, img_b], dim=1))
        return x


class GuidedDecoder(nn.Module):
    """Stage B 解码器：common skip + guided diff skip。"""

    def __init__(
        self,
        bottleneck_channels: int,
        guide_channels: tuple[int, int, int] = (64, 32, 16),
        out_channels: tuple[int, int, int, int] = (64, 32, 16, 16),
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(bottleneck_channels + guide_channels[0] * 2, out_channels[0]),
                ConvBlock(out_channels[0] + guide_channels[1] * 2, out_channels[1]),
                ConvBlock(out_channels[1] + guide_channels[2] * 2, out_channels[2]),
                ConvBlock(out_channels[2] + 2, out_channels[3]),
            ]
        )
        self.out_channels = out_channels[-1]

    def forward(
        self,
        bottleneck: torch.Tensor,
        common_skips: list[torch.Tensor],
        guided_diff_skips: list[torch.Tensor],
        img_a: torch.Tensor,
        img_b: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(bottleneck)
        x = self.blocks[0](torch.cat([x, common_skips[-2], guided_diff_skips[-2]], dim=1))
        x = self.up(x)
        x = self.blocks[1](torch.cat([x, common_skips[-3], guided_diff_skips[-3]], dim=1))
        x = self.up(x)
        x = self.blocks[2](torch.cat([x, common_skips[-4], guided_diff_skips[-4]], dim=1))
        x = self.up(x)
        x = self.blocks[3](torch.cat([x, img_a, img_b], dim=1))
        return x


# ==============================
# Reg v8 主干
# ==============================
class CoarseRegistrationStage(nn.Module):
    """Stage A：全局粗配准。"""

    def __init__(
        self,
        inshape,
        enc_channels: tuple[int, int, int, int] = (16, 32, 64, 96),
        int_steps: int = 7,
        flow_output_mode: str = "svf",
    ):
        super().__init__()
        if flow_output_mode not in {"svf", "disp"}:
            raise ValueError(f"flow_output_mode must be 'svf' or 'disp', got {flow_output_mode}")
        self.flow_output_mode = flow_output_mode
        self.encoder = SharedEncoder(1, enc_channels)
        self.decoder = DualDecoder(bottleneck_channels=enc_channels[-1] * 2)
        self.flow_head = nn.Conv3d(self.decoder.out_channels, 3, kernel_size=3, padding=1)
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_head.bias)
        self.integrate = VecInt(inshape, nsteps=int_steps) if flow_output_mode == "svf" else None
        self.warp = SpatialTransformer(inshape)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> dict[str, torch.Tensor]:
        feats_m = self.encoder(moving)
        feats_f = self.encoder(fixed)
        dec = self.decoder(torch.cat([feats_m[-1], feats_f[-1]], dim=1), feats_m, feats_f, moving, fixed)
        coarse_raw = self.flow_head(dec)
        if self.flow_output_mode == "svf":
            coarse_svf = coarse_raw
            coarse_flow = self.integrate(coarse_svf)
        else:
            coarse_svf = coarse_raw
            coarse_flow = coarse_raw
        coarse_warped = self.warp(moving, coarse_flow)
        return {
            "coarse_feat": dec,
            "coarse_svf": coarse_svf,
            "coarse_flow": coarse_flow,
            "coarse_warped": coarse_warped,
        }


class RegVM0000Model(nn.Module):
    """v0000：最基础的单阶段 VoxelMorph 风格配准。"""

    def __init__(
        self,
        inshape: tuple[int, int, int],
        int_steps: int = 7,
        flow_output_mode: str = "svf",
        enc_channels: tuple[int, int, int, int] = (16, 32, 64, 96),
    ):
        super().__init__()
        self.coarse_stage = CoarseRegistrationStage(
            inshape=inshape,
            enc_channels=enc_channels,
            int_steps=int_steps,
            flow_output_mode=flow_output_mode,
        )
        self.label_warp = SpatialTransformer(inshape, mode="nearest")

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        moving_label: torch.Tensor | None = None,
        detach_stage_a_for_b: bool = False,
    ) -> dict[str, torch.Tensor]:
        del detach_stage_a_for_b
        out = self.coarse_stage(moving, fixed)
        coarse_svf = out["coarse_svf"]
        coarse_flow = out["coarse_flow"]
        coarse_warped = out["coarse_warped"]
        result = {
            "moving": moving,
            "fixed": fixed,
            "coarse_feat": out["coarse_feat"],
            "coarse_svf": coarse_svf,
            "coarse_flow": coarse_flow,
            "coarse_warped": coarse_warped,
            "final_svf": coarse_svf,
            "final_flow": coarse_flow,
            "final_warped": coarse_warped,
        }
        if moving_label is not None:
            coarse_warped_label = self.label_warp(moving_label.float(), coarse_flow)
            result["coarse_warped_label"] = coarse_warped_label
            result["final_warped_label"] = coarse_warped_label
        return result


class DualBRefineStage(nn.Module):
    """Stage B：dual-encoder + common-guided residual refinement。"""

    def __init__(
        self,
        inshape,
        enc_channels: tuple[int, int, int, int] = (16, 32, 64, 96),
        int_steps: int = 7,
        flow_output_mode: str = "svf",
        fusion_mode: str = "film",
        attn_dim: int = 24,
        attn_pool_size: tuple[int, int, int] = (4, 4, 4),
    ):
        super().__init__()
        if flow_output_mode not in {"svf", "disp"}:
            raise ValueError(f"flow_output_mode must be 'svf' or 'disp', got {flow_output_mode}")
        self.flow_output_mode = flow_output_mode
        self.enc_channels = enc_channels
        self.fusion_mode = fusion_mode
        self.attn_dim = attn_dim
        self.attn_pool_size = attn_pool_size

        self.common_encoder = SharedEncoder(1, enc_channels)
        self.diff_encoder = SharedEncoder(1, enc_channels)
        self.deep_ch = enc_channels[-1]

        # Stage B 只预测 residual deformation，所以 bottleneck 里只组织 refine 所需信息。
        self.decoder = GuidedDecoder(bottleneck_channels=self.deep_ch * 3)
        self.flow_head = nn.Conv3d(self.decoder.out_channels, 3, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.integrate = VecInt(inshape, nsteps=int_steps) if flow_output_mode == "svf" else None
        self.warp = SpatialTransformer(inshape)

        self.level_gates = nn.ModuleList(
            [nn.Sequential(nn.Conv3d(ch, ch, kernel_size=1), nn.Sigmoid()) for ch in enc_channels]
        )
        self.level_film_gamma = nn.ModuleList([nn.Conv3d(ch, ch, kernel_size=1) for ch in enc_channels])
        self.level_film_beta = nn.ModuleList([nn.Conv3d(ch, ch, kernel_size=1) for ch in enc_channels])

        self.attn_q = nn.Conv3d(self.deep_ch, attn_dim, kernel_size=1)
        self.attn_k = nn.Conv3d(self.deep_ch, attn_dim, kernel_size=1)
        self.attn_v = nn.Conv3d(self.deep_ch, attn_dim, kernel_size=1)
        self.attn_proj = nn.Conv3d(attn_dim, self.deep_ch, kernel_size=1)
        self.attn_pool = nn.AdaptiveAvgPool3d(attn_pool_size)

        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_head.bias)

        valid_modes = {"concat", "gate", "film", "attn_lite", "common_only", "diff_only"}
        if self.fusion_mode not in valid_modes:
            raise ValueError(f"fusion_mode must be one of {sorted(valid_modes)}, got {self.fusion_mode}")

    def _apply_attn_lite(self, common_ctx: torch.Tensor, diff_res: torch.Tensor) -> torch.Tensor:
        pooled_common = self.attn_pool(common_ctx)
        pooled_diff = self.attn_pool(diff_res)
        q = self.attn_q(pooled_common).flatten(2).transpose(1, 2)
        k = self.attn_k(pooled_diff).flatten(2).transpose(1, 2)
        v = self.attn_v(pooled_diff).flatten(2).transpose(1, 2)
        scale = float(self.attn_dim) ** -0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)  # 轻量 self-attention 权重
        guided = torch.bmm(attn, v).transpose(1, 2)                              # 用 common 上下文引导 diff 特征
        guided = guided.reshape(
            pooled_diff.shape[0],
            self.attn_dim,
            self.attn_pool_size[0],
            self.attn_pool_size[1],
            self.attn_pool_size[2],
        )
        guided = self.attn_proj(guided)
        return F.interpolate(guided, size=diff_res.shape[2:], mode="trilinear", align_corners=True)

    def _fuse_level(
        self,
        common_feat: torch.Tensor,
        diff_feat: torch.Tensor,
        level_idx: int,
        use_attn_lite: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 返回 guided diff 特征和 control map，后者主要用于解释 B 到底在“放大/抑制”哪里。
        if self.fusion_mode == "concat":
            return diff_feat, torch.ones_like(torch.abs(diff_feat))
        if self.fusion_mode == "gate" or use_attn_lite:
            control_map = self.level_gates[level_idx](common_feat)
            return control_map * diff_feat, control_map
        if self.fusion_mode == "film":
            gamma = self.level_film_gamma[level_idx](common_feat)
            beta = self.level_film_beta[level_idx](common_feat)
            return gamma * diff_feat + beta, torch.sigmoid(gamma)
        if self.fusion_mode == "common_only":
            return torch.zeros_like(diff_feat), torch.zeros_like(torch.abs(diff_feat))
        return diff_feat, torch.ones_like(torch.abs(diff_feat))  # diff_only

    def compose_flows(self, coarse_flow: torch.Tensor, residual_flow: torch.Tensor) -> torch.Tensor:
        """把 coarse flow 与 residual flow 组合成最终形变场。"""

        warped_coarse_flow = self.warp(coarse_flow, residual_flow)  # 先把 coarse flow 映射到 residual 坐标系
        return residual_flow + warped_coarse_flow

    def forward(
        self,
        moving: torch.Tensor,
        coarse_warped: torch.Tensor,
        fixed: torch.Tensor,
        coarse_flow: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # A 已经完成一次全局对齐，因此 B 直接对 `coarse_warped` 和 `fixed` 做残差建模。
        common_feats_a = self.common_encoder(coarse_warped)
        common_feats_f = self.common_encoder(fixed)
        diff_feats_a = self.diff_encoder(coarse_warped)
        diff_feats_f = self.diff_encoder(fixed)

        common_a = common_feats_a[-1]
        common_f = common_feats_f[-1]
        diff_a = diff_feats_a[-1]
        diff_f = diff_feats_f[-1]

        common_mean = 0.5 * (common_a + common_f)  # 两侧共有的稳定结构语义。
        diff_res = diff_a - diff_f                 # 仍未对齐的残差信号。
        diff_mag = torch.abs(diff_res)             # 只保留幅值，便于做差异强度正则。
        common_skip_means = [0.5 * (a + f) for a, f in zip(common_feats_a, common_feats_f)]
        guided_diff_skips: list[torch.Tensor] = []

        if self.fusion_mode == "common_only":
            guided_diff = torch.zeros_like(diff_res)
            diff_mag = torch.zeros_like(diff_mag)
            control_map = torch.zeros_like(diff_mag)
            guided_diff_skips = [torch.zeros_like(feat) for feat in common_skip_means]
        elif self.fusion_mode == "diff_only":
            common_mean = torch.zeros_like(common_mean)
            common_skip_means = [torch.zeros_like(feat) for feat in common_skip_means]
            guided_diff = diff_res
            control_map = torch.ones_like(diff_mag)
            guided_diff_skips = [a - f for a, f in zip(diff_feats_a, diff_feats_f)]
        else:
            guided_source = self._apply_attn_lite(common_mean, diff_res) if self.fusion_mode == "attn_lite" else diff_res
            guided_diff, control_map = self._fuse_level(common_mean, guided_source, level_idx=len(self.enc_channels) - 1)
            for level_idx, (common_skip, diff_skip_a, diff_skip_f) in enumerate(
                zip(common_skip_means, diff_feats_a, diff_feats_f)
            ):
                diff_skip = diff_skip_a - diff_skip_f
                guided_skip, _ = self._fuse_level(
                    common_skip,
                    diff_skip,
                    level_idx=level_idx,
                    use_attn_lite=self.fusion_mode == "attn_lite",
                )
                guided_diff_skips.append(guided_skip)

        # B 的解码器只接三部分输入：共性语义、受控差异、差异幅值。
        bottleneck = torch.cat([common_mean, guided_diff, diff_mag], dim=1)
        dec = self.decoder(bottleneck, common_skip_means, guided_diff_skips, coarse_warped, fixed)

        residual_raw = self.flow_head(dec)
        if self.flow_output_mode == "svf":
            residual_svf = residual_raw
            residual_flow = self.integrate(residual_svf)
        else:
            residual_svf = residual_raw
            residual_flow = residual_raw
        final_flow = self.compose_flows(coarse_flow, residual_flow)
        final_warped = self.warp(moving, final_flow)

        return {
            "common_a_vec": self.pool(common_a).flatten(1),
            "common_f_vec": self.pool(common_f).flatten(1),
            "diff_a_vec": self.pool(diff_a).flatten(1),
            "diff_f_vec": self.pool(diff_f).flatten(1),
            "diff_gap": diff_mag,
            "residual_svf": residual_svf,
            "residual_flow": residual_flow,
            "final_flow": final_flow,
            "final_warped": final_warped,
            "control_map": control_map,
        }


class RegDualBModel(nn.Module):
    """v8 reg 主模型：Stage A coarse + dual-B refine。"""

    def __init__(
        self,
        inshape: tuple[int, int, int],
        int_steps: int = 7,
        stage_b_fusion_mode: str = "film",
        enable_stage_b: bool = True,
        flow_output_mode: str = "svf",
    ):
        super().__init__()
        self.enable_stage_b = enable_stage_b
        self.coarse_stage = CoarseRegistrationStage(inshape, int_steps=int_steps, flow_output_mode=flow_output_mode)
        self.stage_b = DualBRefineStage(
            inshape=inshape,
            int_steps=int_steps,
            flow_output_mode=flow_output_mode,
            fusion_mode=stage_b_fusion_mode,
        )
        self.warp_label = SpatialTransformer(inshape, mode="nearest")

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        moving_label: torch.Tensor | None = None,
        detach_stage_a_for_b: bool = False,
    ) -> dict[str, torch.Tensor]:
        # 主链设计很直接：A 做 coarse，B 做 residual refine。
        coarse = self.coarse_stage(moving, fixed)
        coarse_warped_label = None
        if moving_label is not None:
            coarse_warped_label = self.warp_label(moving_label.float(), coarse["coarse_flow"])
        if self.enable_stage_b:
            # 某些实验会切断 B -> A 的梯度，观察 refine 是否过度依赖 coarse 主干。
            coarse_warped_for_b = coarse["coarse_warped"].detach() if detach_stage_a_for_b else coarse["coarse_warped"]
            coarse_flow_for_b = coarse["coarse_flow"].detach() if detach_stage_a_for_b else coarse["coarse_flow"]
            stage_b_out = self.stage_b(moving, coarse_warped_for_b, fixed, coarse_flow_for_b)
        else:
            batch_size = moving.shape[0]
            pooled_shape = (batch_size, self.stage_b.deep_ch)
            # 关闭 B 时仍返回完整键集合，这样 train / infer 主链不需要再写 if-else。
            stage_b_out = {
                "common_a_vec": moving.new_zeros(pooled_shape),
                "common_f_vec": moving.new_zeros(pooled_shape),
                "diff_a_vec": moving.new_zeros(pooled_shape),
                "diff_f_vec": moving.new_zeros(pooled_shape),
                "diff_gap": torch.zeros_like(coarse["coarse_feat"]),
                "residual_svf": torch.zeros_like(coarse["coarse_svf"]),
                "residual_flow": torch.zeros_like(coarse["coarse_flow"]),
                "final_flow": coarse["coarse_flow"],
                "final_warped": coarse["coarse_warped"],
                "control_map": torch.zeros_like(coarse["coarse_feat"]),
            }
        out = {**coarse, **stage_b_out}
        if moving_label is not None:
            out["coarse_warped_label"] = coarse_warped_label
            out["final_warped_label"] = self.warp_label(moving_label.float(), stage_b_out["final_flow"])
        return out


class ROIConditionBuilder910(nn.Module):
    """910 专用 ROI 条件构造器。"""

    def __init__(self, roi_classes: tuple[int, ...] = (), dilation_radius: int = 3, condition_channels: int = 3):
        super().__init__()
        self.roi_classes = tuple(int(c) for c in roi_classes)
        self.dilation_radius = int(dilation_radius)
        self.condition_channels = int(condition_channels)

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask.float()
        kernel_size = radius * 2 + 1
        return F.max_pool3d(mask.float(), kernel_size=kernel_size, stride=1, padding=radius)

    @staticmethod
    def _erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask.float()
        kernel_size = radius * 2 + 1
        inv = 1.0 - mask.float()
        return 1.0 - F.max_pool3d(inv, kernel_size=kernel_size, stride=1, padding=radius)

    @staticmethod
    def _fixed_edge_map(fixed: torch.Tensor) -> torch.Tensor:
        dz = F.pad(torch.abs(fixed[:, :, 1:, :, :] - fixed[:, :, :-1, :, :]), (0, 0, 0, 0, 0, 1))
        dy = F.pad(torch.abs(fixed[:, :, :, 1:, :] - fixed[:, :, :, :-1, :]), (0, 0, 0, 1, 0, 0))
        dx = F.pad(torch.abs(fixed[:, :, :, :, 1:] - fixed[:, :, :, :, :-1]), (0, 1, 0, 0, 0, 0))
        edge = dz + dy + dx
        edge_mean = edge.mean(dim=(2, 3, 4), keepdim=True)
        edge_std = edge.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        return torch.clamp((edge - edge_mean) / edge_std, min=0.0, max=3.0) / 3.0

    def forward(self, coarse_warped_label: torch.Tensor, fixed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """输出 910 的 ROI 条件图与 ROI 加权图。"""

        if self.condition_channels <= 0:
            empty = fixed.new_zeros(fixed.shape[0], 0, *fixed.shape[2:])
            return empty, fixed.new_ones(fixed.shape[0], 1, *fixed.shape[2:])

        label = coarse_warped_label.round().long()
        if self.roi_classes:
            roi_mask = torch.zeros_like(label, dtype=torch.bool)
            for class_id in self.roi_classes:
                roi_mask = roi_mask | (label == class_id)
        else:
            roi_mask = label > 0

        roi_mask = roi_mask.float()
        soft_roi = self._dilate_mask(roi_mask, self.dilation_radius).clamp(0.0, 1.0)
        boundary_radius = max(1, min(self.dilation_radius, 2))
        boundary = (
            self._dilate_mask(roi_mask, boundary_radius) - self._erode_mask(roi_mask, boundary_radius)
        ).clamp(0.0, 1.0)
        fixed_edge = self._fixed_edge_map(fixed).detach()

        condition_parts = [soft_roi.detach(), boundary.detach(), fixed_edge]
        condition = torch.cat(condition_parts[: self.condition_channels], dim=1)
        if condition.shape[1] < self.condition_channels:
            pad_channels = self.condition_channels - condition.shape[1]
            condition = torch.cat(
                [condition, condition.new_zeros(condition.shape[0], pad_channels, *condition.shape[2:])],
                dim=1,
            )
        roi_weight = 1.0 + soft_roi.detach()
        return condition, roi_weight


class ROIRefineDecoder910(nn.Module):
    """910 专用单流解码器：只接收单编码器 skip 与 ROI 条件。"""

    def __init__(
        self,
        bottleneck_channels: int,
        skip_channels: tuple[int, int, int] = (64, 32, 16),
        condition_channels: int = 0,
        out_channels: tuple[int, int, int, int] = (64, 32, 16, 16),
    ):
        super().__init__()
        self.condition_channels = condition_channels
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(bottleneck_channels + skip_channels[0], out_channels[0]),
                ConvBlock(out_channels[0] + skip_channels[1], out_channels[1]),
                ConvBlock(out_channels[1] + skip_channels[2], out_channels[2]),
                ConvBlock(out_channels[2] + 2 + condition_channels, out_channels[3]),
            ]
        )
        self.out_channels = out_channels[-1]

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: list[torch.Tensor],
        coarse_warped: torch.Tensor,
        fixed: torch.Tensor,
        roi_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.up(bottleneck)
        x = self.blocks[0](torch.cat([x, skips[-2]], dim=1))
        x = self.up(x)
        x = self.blocks[1](torch.cat([x, skips[-3]], dim=1))
        x = self.up(x)
        x = self.blocks[2](torch.cat([x, skips[-4]], dim=1))
        x = self.up(x)
        tail_inputs = [x, coarse_warped, fixed]
        if self.condition_channels > 0:
            if roi_condition is None:
                roi_condition = coarse_warped.new_zeros(
                    coarse_warped.shape[0],
                    self.condition_channels,
                    *coarse_warped.shape[2:],
                )
            tail_inputs.append(roi_condition)
        x = self.blocks[3](torch.cat(tail_inputs, dim=1))
        return x


class ROIBRefineStage910(nn.Module):
    """910 专用 B：单编码器 ROI-conditioned residual refine。"""

    def __init__(
        self,
        inshape,
        enc_channels: tuple[int, int, int, int] = (16, 32, 64, 96),
        int_steps: int = 7,
        condition_channels: int = 0,
    ):
        super().__init__()
        self.condition_channels = condition_channels
        self.deep_ch = enc_channels[-1]
        self.encoder = SharedEncoder(2 + condition_channels, enc_channels)
        self.decoder = ROIRefineDecoder910(
            bottleneck_channels=self.deep_ch,
            skip_channels=(enc_channels[-2], enc_channels[-3], enc_channels[-4]),
            condition_channels=condition_channels,
        )
        self.flow_head = nn.Conv3d(self.decoder.out_channels, 3, kernel_size=3, padding=1)
        self.integrate = VecInt(inshape, nsteps=int_steps)
        self.warp = SpatialTransformer(inshape)

        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_head.bias)

    def compose_flows(self, coarse_flow: torch.Tensor, residual_flow: torch.Tensor) -> torch.Tensor:
        """把 coarse flow 与 residual flow 组合成最终形变场。"""

        warped_coarse_flow = self.warp(coarse_flow, residual_flow)
        return residual_flow + warped_coarse_flow

    def forward(
        self,
        moving: torch.Tensor,
        coarse_warped: torch.Tensor,
        fixed: torch.Tensor,
        coarse_flow: torch.Tensor,
        roi_condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.condition_channels > 0:
            if roi_condition is None:
                roi_condition = coarse_warped.new_zeros(
                    coarse_warped.shape[0],
                    self.condition_channels,
                    *coarse_warped.shape[2:],
                )
            encoder_input = torch.cat([coarse_warped, fixed, roi_condition], dim=1)
            control_map = roi_condition[:, :1]
        else:
            encoder_input = torch.cat([coarse_warped, fixed], dim=1)
            control_map = coarse_warped.new_zeros(coarse_warped.shape[0], 1, *coarse_warped.shape[2:])

        feats = self.encoder(encoder_input)
        dec = self.decoder(feats[-1], feats, coarse_warped, fixed, roi_condition)
        residual_svf = self.flow_head(dec)
        residual_flow = self.integrate(residual_svf)
        final_flow = self.compose_flows(coarse_flow, residual_flow)
        final_warped = self.warp(moving, final_flow)
        return {
            "residual_svf": residual_svf,
            "residual_flow": residual_flow,
            "final_flow": final_flow,
            "final_warped": final_warped,
            "control_map": control_map,
        }


class RegROIBv910Model(nn.Module):
    """910 专用 reg 主模型：A coarse + 单编码器 ROI-B refine。"""

    def __init__(
        self,
        inshape: tuple[int, int, int],
        int_steps: int = 7,
        enable_stage_b: bool = True,
        enable_roi_b: bool = True,
        roi_classes: tuple[int, ...] = (),
        roi_dilation_radius: int = 3,
        roi_condition_channels: int = 3,
    ):
        super().__init__()
        self.enable_stage_b = enable_stage_b
        self.enable_roi_b = enable_roi_b
        self.roi_condition_channels = int(roi_condition_channels if enable_roi_b else 0)
        self.coarse_stage = CoarseRegistrationStage(inshape, int_steps=int_steps)
        self.roi_builder = ROIConditionBuilder910(
            roi_classes=roi_classes,
            dilation_radius=roi_dilation_radius,
            condition_channels=self.roi_condition_channels,
        )
        self.stage_b = ROIBRefineStage910(
            inshape=inshape,
            int_steps=int_steps,
            condition_channels=self.roi_condition_channels,
        )
        self.warp_label = SpatialTransformer(inshape, mode="nearest")

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        moving_label: torch.Tensor | None = None,
        detach_stage_a_for_b: bool = False,
    ) -> dict[str, torch.Tensor]:
        coarse = self.coarse_stage(moving, fixed)
        coarse_warped_label = None
        roi_condition = None
        roi_weight = None
        coarse_warped_label = self.warp_label(moving_label.float(), coarse["coarse_flow"])
        if self.enable_roi_b and self.enable_stage_b:
            roi_condition, roi_weight = self.roi_builder(coarse_warped_label, fixed)

        if self.enable_stage_b:
            coarse_warped_for_b = coarse["coarse_warped"].detach() if detach_stage_a_for_b else coarse["coarse_warped"]
            coarse_flow_for_b = coarse["coarse_flow"].detach() if detach_stage_a_for_b else coarse["coarse_flow"]
            stage_b_out = self.stage_b(
                moving=moving,
                coarse_warped=coarse_warped_for_b,
                fixed=fixed,
                coarse_flow=coarse_flow_for_b,
                roi_condition=roi_condition,
            )
        else:
            stage_b_out = {
                "residual_svf": torch.zeros_like(coarse["coarse_svf"]),
                "residual_flow": torch.zeros_like(coarse["coarse_flow"]),
                "final_flow": coarse["coarse_flow"],
                "final_warped": coarse["coarse_warped"],
                "control_map": coarse["coarse_warped"].new_zeros(
                    coarse["coarse_warped"].shape[0], 1, *coarse["coarse_warped"].shape[2:]
                ),
            }

        out = {**coarse, **stage_b_out}
        if roi_condition is not None and roi_weight is not None:
            out["roi_condition"] = roi_condition
            out["roi_weight"] = roi_weight
        if moving_label is not None:
            out["coarse_warped_label"] = coarse_warped_label
            out["final_warped_label"] = self.warp_label(moving_label.float(), stage_b_out["final_flow"])
        return out


def build_v9_model_bundle(
    inshape: tuple[int, int, int],
    int_steps: int = 7,
    stage_b_fusion_mode: str = "film",
    enable_stage_b: bool = True,
    flow_output_mode: str | None = None,
) -> nn.ModuleDict:
    if flow_output_mode not in {None, "svf", "disp"}:
        raise ValueError(f"Unsupported flow_output_mode: {flow_output_mode}")
    resolved_flow_mode = "svf" if flow_output_mode is None else flow_output_mode
    reg_model = RegDualBModel(
        inshape=inshape,
        int_steps=int_steps,
        stage_b_fusion_mode=stage_b_fusion_mode,
        enable_stage_b=enable_stage_b,
        flow_output_mode=resolved_flow_mode,
    )
    return nn.ModuleDict({"reg_model": reg_model})


def build_v0000_vm_model_bundle(
    inshape: tuple[int, int, int],
    int_steps: int = 7,
    flow_output_mode: str | None = None,
) -> nn.ModuleDict:
    """v0000 builder：最基础的单阶段 VM。"""

    if flow_output_mode not in {None, "svf", "disp"}:
        raise ValueError(f"Unsupported flow_output_mode: {flow_output_mode}")
    resolved_flow_mode = "svf" if flow_output_mode is None else flow_output_mode
    reg_model = RegVM0000Model(
        inshape=inshape,
        int_steps=int_steps,
        flow_output_mode=resolved_flow_mode,
    )
    return nn.ModuleDict({"reg_model": reg_model})


def build_v9_1_0_roib_model_bundle(
    inshape: tuple[int, int, int],
    int_steps: int = 7,
    enable_stage_b: bool = True,
    enable_roi_b: bool = True,
    roi_classes: tuple[int, ...] = (),
    roi_dilation_radius: int = 3,
    roi_condition_channels: int = 3,
) -> nn.ModuleDict:
    """910 专用模型构建函数。"""

    reg_model = RegROIBv910Model(
        inshape=inshape,
        int_steps=int_steps,
        enable_stage_b=enable_stage_b,
        enable_roi_b=enable_roi_b,
        roi_classes=roi_classes,
        roi_dilation_radius=roi_dilation_radius,
        roi_condition_channels=roi_condition_channels,
    )
    return nn.ModuleDict({"reg_model": reg_model})


# ==============================
# Reg v9.3.0 下采样 Stage-A + 原分辨率 ROI-B
# ==============================
class DownsampledCoarseRegistrationStage930(nn.Module):
    """930 的 Stage A：先在低分辨率估计 coarse flow，再映射回原分辨率。"""

    def __init__(
        self,
        inshape: tuple[int, int, int],
        downsample_factor: int = 8,
        enc_channels: tuple[int, int, int, int] = (16, 32, 64, 96),
        int_steps: int = 7,
    ):
        super().__init__()
        self.full_shape = tuple(int(s) for s in inshape)
        self.downsample_factor = max(int(downsample_factor), 1)
        self.required_multiple = 2 ** len(enc_channels)
        self.lowres_shape = self._choose_lowres_shape(
            full_shape=self.full_shape,
            downsample_factor=self.downsample_factor,
            required_multiple=self.required_multiple,
        )
        self.encoder = SharedEncoder(1, enc_channels)
        self.decoder = DualDecoder(bottleneck_channels=enc_channels[-1] * 2)
        self.flow_head = nn.Conv3d(self.decoder.out_channels, 3, kernel_size=3, padding=1)
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_head.bias)
        self.integrate = VecInt(self.lowres_shape, nsteps=int_steps)
        self.warp_lowres = SpatialTransformer(self.lowres_shape)
        self.warp_fullres = SpatialTransformer(self.full_shape)

    @staticmethod
    def _resize_image(img: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        if tuple(img.shape[2:]) == tuple(target_shape):
            return img
        return F.interpolate(img, size=target_shape, mode="trilinear", align_corners=True)

    @staticmethod
    def _choose_axis_size(full_size: int, downsample_factor: int, required_multiple: int) -> int:
        target = max(int(round(float(full_size) / float(downsample_factor))), required_multiple)
        lower = max((target // required_multiple) * required_multiple, required_multiple)
        upper = max(((target + required_multiple - 1) // required_multiple) * required_multiple, required_multiple)
        if abs(target - lower) <= abs(upper - target):
            return lower
        return upper

    @classmethod
    def _choose_lowres_shape(
        cls,
        full_shape: tuple[int, int, int],
        downsample_factor: int,
        required_multiple: int,
    ) -> tuple[int, int, int]:
        return tuple(
            cls._choose_axis_size(int(axis_size), int(downsample_factor), int(required_multiple))
            for axis_size in full_shape
        )

    @staticmethod
    def _resize_flow(flow: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        if tuple(flow.shape[2:]) == tuple(target_shape):
            return flow
        src_shape = flow.shape[2:]
        resized = F.interpolate(flow, size=target_shape, mode="trilinear", align_corners=True)

        def _coord_scale(target_size: int, src_size: int) -> float:
            if src_size <= 1 or target_size <= 1:
                return float(target_size) / max(float(src_size), 1.0)
            return float(target_size - 1) / float(src_size - 1)

        scale_z = _coord_scale(target_shape[0], src_shape[0])
        scale_y = _coord_scale(target_shape[1], src_shape[1])
        scale_x = _coord_scale(target_shape[2], src_shape[2])
        resized[:, 0] = resized[:, 0] * scale_z
        resized[:, 1] = resized[:, 1] * scale_y
        resized[:, 2] = resized[:, 2] * scale_x
        return resized

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> dict[str, torch.Tensor]:
        moving_down = self._resize_image(moving, self.lowres_shape)
        fixed_down = self._resize_image(fixed, self.lowres_shape)

        feats_m = self.encoder(moving_down)
        feats_f = self.encoder(fixed_down)
        dec = self.decoder(torch.cat([feats_m[-1], feats_f[-1]], dim=1), feats_m, feats_f, moving_down, fixed_down)
        coarse_svf_lowres = self.flow_head(dec)
        coarse_flow_lowres = self.integrate(coarse_svf_lowres)
        coarse_warped_lowres = self.warp_lowres(moving_down, coarse_flow_lowres)

        coarse_flow_fullres = self._resize_flow(coarse_flow_lowres, self.full_shape)
        coarse_warped_fullres = self.warp_fullres(moving, coarse_flow_fullres)
        return {
            "coarse_feat": dec,
            "moving_down": moving_down,
            "fixed_down": fixed_down,
            "coarse_svf_lowres": coarse_svf_lowres,
            "coarse_flow_lowres": coarse_flow_lowres,
            "coarse_warped_lowres": coarse_warped_lowres,
            "coarse_svf": coarse_svf_lowres,
            "coarse_flow": coarse_flow_fullres,
            "coarse_warped": coarse_warped_fullres,
        }


class RegROIBv930Model(nn.Module):
    """930 主模型：A 在低分辨率估 flow，B 在原分辨率 refine，并在原图空间合成 flow。"""

    def __init__(
        self,
        inshape: tuple[int, int, int],
        int_steps: int = 7,
        enable_stage_b: bool = True,
        enable_roi_b: bool = True,
        roi_classes: tuple[int, ...] = (),
        roi_dilation_radius: int = 3,
        roi_condition_channels: int = 3,
        stage_a_downsample_factor: int = 8,
    ):
        super().__init__()
        self.enable_stage_b = enable_stage_b
        self.enable_roi_b = enable_roi_b
        self.roi_condition_channels = int(roi_condition_channels if enable_roi_b else 0)
        self.coarse_stage = DownsampledCoarseRegistrationStage930(
            inshape=inshape,
            downsample_factor=stage_a_downsample_factor,
            int_steps=int_steps,
        )
        self.roi_builder = ROIConditionBuilder910(
            roi_classes=roi_classes,
            dilation_radius=roi_dilation_radius,
            condition_channels=self.roi_condition_channels,
        )
        self.stage_b = ROIBRefineStage910(
            inshape=inshape,
            int_steps=int_steps,
            condition_channels=self.roi_condition_channels,
        )
        self.warp_label = SpatialTransformer(inshape, mode="nearest")

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        moving_label: torch.Tensor | None = None,
        detach_stage_a_for_b: bool = False,
    ) -> dict[str, torch.Tensor]:
        coarse = self.coarse_stage(moving, fixed)
        coarse_warped_label = None
        roi_condition = None
        roi_weight = None
        if moving_label is not None:
            coarse_warped_label = self.warp_label(moving_label.float(), coarse["coarse_flow"])
        if self.enable_roi_b and self.enable_stage_b and coarse_warped_label is not None:
            roi_condition, roi_weight = self.roi_builder(coarse_warped_label, fixed)

        if self.enable_stage_b:
            coarse_warped_for_b = coarse["coarse_warped"].detach() if detach_stage_a_for_b else coarse["coarse_warped"]
            coarse_flow_for_b = coarse["coarse_flow"].detach() if detach_stage_a_for_b else coarse["coarse_flow"]
            stage_b_out = self.stage_b(
                moving=moving,
                coarse_warped=coarse_warped_for_b,
                fixed=fixed,
                coarse_flow=coarse_flow_for_b,
                roi_condition=roi_condition,
            )
        else:
            stage_b_out = {
                "residual_svf": coarse["coarse_flow"].new_zeros(coarse["coarse_flow"].shape),
                "residual_flow": coarse["coarse_flow"].new_zeros(coarse["coarse_flow"].shape),
                "final_flow": coarse["coarse_flow"],
                "final_warped": self.stage_b.warp(moving, coarse["coarse_flow"]),
                "control_map": coarse["coarse_warped"].new_zeros(
                    coarse["coarse_warped"].shape[0], 1, *coarse["coarse_warped"].shape[2:]
                ),
            }

        out = {**coarse, **stage_b_out}
        if roi_condition is not None and roi_weight is not None:
            out["roi_condition"] = roi_condition
            out["roi_weight"] = roi_weight
        if moving_label is not None:
            out["coarse_warped_label"] = coarse_warped_label
            out["final_warped_label"] = self.warp_label(moving_label.float(), stage_b_out["final_flow"])
        return out


def build_downsampled_roib_reg_model_bundle(
    inshape: tuple[int, int, int],
    int_steps: int = 7,
    enable_stage_b: bool = True,
    enable_roi_b: bool = True,
    roi_classes: tuple[int, ...] = (),
    roi_dilation_radius: int = 3,
    roi_condition_channels: int = 3,
    stage_a_downsample_factor: int = 8,
) -> nn.ModuleDict:
    """930 专用模型构建函数。"""

    reg_model = RegROIBv930Model(
        inshape=inshape,
        int_steps=int_steps,
        enable_stage_b=enable_stage_b,
        enable_roi_b=enable_roi_b,
        roi_classes=roi_classes,
        roi_dilation_radius=roi_dilation_radius,
        roi_condition_channels=roi_condition_channels,
        stage_a_downsample_factor=stage_a_downsample_factor,
    )
    return nn.ModuleDict({"reg_model": reg_model})


# ==============================
# Reg v9.3.1 下采样 Stage-A(浅层) + 原分辨率 ROI-B
# ==============================
class DownsampledShallowCoarseRegistrationStage931(nn.Module):
    """931 的 Stage A：三层 encoder/decoder，兼容更激进的下采样比例。"""

    def __init__(
        self,
        inshape: tuple[int, int, int],
        downsample_factor: int = 8,
        enc_channels: tuple[int, int, int] = (16, 32, 64),
        int_steps: int = 7,
    ):
        super().__init__()
        self.full_shape = tuple(int(s) for s in inshape)
        self.downsample_factor = max(int(downsample_factor), 1)
        self.required_multiple = 2 ** len(enc_channels)
        self.lowres_shape = self._choose_lowres_shape(
            full_shape=self.full_shape,
            downsample_factor=self.downsample_factor,
            required_multiple=self.required_multiple,
        )
        self.encoder = SharedEncoder(1, enc_channels)
        self.decoder = DualDecoder3Level(bottleneck_channels=enc_channels[-1] * 2)
        self.flow_head = nn.Conv3d(self.decoder.out_channels, 3, kernel_size=3, padding=1)
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_head.bias)
        self.integrate = VecInt(self.lowres_shape, nsteps=int_steps)
        self.warp_lowres = SpatialTransformer(self.lowres_shape)
        self.warp_fullres = SpatialTransformer(self.full_shape)

    @staticmethod
    def _resize_image(img: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        if tuple(img.shape[2:]) == tuple(target_shape):
            return img
        return F.interpolate(img, size=target_shape, mode="trilinear", align_corners=True)

    @staticmethod
    def _choose_axis_size(full_size: int, downsample_factor: int, required_multiple: int) -> int:
        target = max(int(round(float(full_size) / float(downsample_factor))), required_multiple)
        lower = max((target // required_multiple) * required_multiple, required_multiple)
        upper = max(((target + required_multiple - 1) // required_multiple) * required_multiple, required_multiple)
        if abs(target - lower) <= abs(upper - target):
            return lower
        return upper

    @classmethod
    def _choose_lowres_shape(
        cls,
        full_shape: tuple[int, int, int],
        downsample_factor: int,
        required_multiple: int,
    ) -> tuple[int, int, int]:
        return tuple(
            cls._choose_axis_size(int(axis_size), int(downsample_factor), int(required_multiple))
            for axis_size in full_shape
        )

    @staticmethod
    def _resize_flow(flow: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
        if tuple(flow.shape[2:]) == tuple(target_shape):
            return flow
        src_shape = flow.shape[2:]
        resized = F.interpolate(flow, size=target_shape, mode="trilinear", align_corners=True)

        def _coord_scale(target_size: int, src_size: int) -> float:
            if src_size <= 1 or target_size <= 1:
                return float(target_size) / max(float(src_size), 1.0)
            return float(target_size - 1) / float(src_size - 1)

        scale_z = _coord_scale(target_shape[0], src_shape[0])
        scale_y = _coord_scale(target_shape[1], src_shape[1])
        scale_x = _coord_scale(target_shape[2], src_shape[2])
        resized[:, 0] = resized[:, 0] * scale_z
        resized[:, 1] = resized[:, 1] * scale_y
        resized[:, 2] = resized[:, 2] * scale_x
        return resized

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor) -> dict[str, torch.Tensor]:
        moving_down = self._resize_image(moving, self.lowres_shape)
        fixed_down = self._resize_image(fixed, self.lowres_shape)

        feats_m = self.encoder(moving_down)
        feats_f = self.encoder(fixed_down)
        dec = self.decoder(torch.cat([feats_m[-1], feats_f[-1]], dim=1), feats_m, feats_f, moving_down, fixed_down)
        coarse_svf_lowres = self.flow_head(dec)
        coarse_flow_lowres = self.integrate(coarse_svf_lowres)
        coarse_warped_lowres = self.warp_lowres(moving_down, coarse_flow_lowres)

        coarse_flow_fullres = self._resize_flow(coarse_flow_lowres, self.full_shape)
        coarse_warped_fullres = self.warp_fullres(moving, coarse_flow_fullres)
        return {
            "coarse_feat": dec,
            "moving_down": moving_down,
            "fixed_down": fixed_down,
            "coarse_svf_lowres": coarse_svf_lowres,
            "coarse_flow_lowres": coarse_flow_lowres,
            "coarse_warped_lowres": coarse_warped_lowres,
            "coarse_svf": coarse_svf_lowres,
            "coarse_flow": coarse_flow_fullres,
            "coarse_warped": coarse_warped_fullres,
        }


class RegROIBv931Model(nn.Module):
    """931 主模型：浅层低分辨率 A + 原分辨率 ROI-B。"""

    def __init__(
        self,
        inshape: tuple[int, int, int],
        int_steps: int = 7,
        enable_stage_b: bool = True,
        enable_roi_b: bool = True,
        roi_classes: tuple[int, ...] = (),
        roi_dilation_radius: int = 3,
        roi_condition_channels: int = 3,
        stage_a_downsample_factor: int = 4,
    ):
        super().__init__()
        self.enable_stage_b = enable_stage_b
        self.enable_roi_b = enable_roi_b
        self.roi_condition_channels = int(roi_condition_channels if enable_roi_b else 0)
        self.coarse_stage = DownsampledShallowCoarseRegistrationStage931(
            inshape=inshape,
            downsample_factor=stage_a_downsample_factor,
            int_steps=int_steps,
        )
        self.roi_builder = ROIConditionBuilder910(
            roi_classes=roi_classes,
            dilation_radius=roi_dilation_radius,
            condition_channels=self.roi_condition_channels,
        )
        self.stage_b = ROIBRefineStage910(
            inshape=inshape,
            int_steps=int_steps,
            condition_channels=self.roi_condition_channels,
        )
        self.warp_label = SpatialTransformer(inshape, mode="nearest")

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        moving_label: torch.Tensor | None = None,
        detach_stage_a_for_b: bool = False,
    ) -> dict[str, torch.Tensor]:
        coarse = self.coarse_stage(moving, fixed)
        coarse_warped_label = None
        roi_condition = None
        roi_weight = None
        if moving_label is not None:
            coarse_warped_label = self.warp_label(moving_label.float(), coarse["coarse_flow"])
        if self.enable_roi_b and self.enable_stage_b and coarse_warped_label is not None:
            roi_condition, roi_weight = self.roi_builder(coarse_warped_label, fixed)

        if self.enable_stage_b:
            coarse_warped_for_b = coarse["coarse_warped"].detach() if detach_stage_a_for_b else coarse["coarse_warped"]
            coarse_flow_for_b = coarse["coarse_flow"].detach() if detach_stage_a_for_b else coarse["coarse_flow"]
            stage_b_out = self.stage_b(
                moving=moving,
                coarse_warped=coarse_warped_for_b,
                fixed=fixed,
                coarse_flow=coarse_flow_for_b,
                roi_condition=roi_condition,
            )
        else:
            stage_b_out = {
                "residual_svf": coarse["coarse_flow"].new_zeros(coarse["coarse_flow"].shape),
                "residual_flow": coarse["coarse_flow"].new_zeros(coarse["coarse_flow"].shape),
                "final_flow": coarse["coarse_flow"],
                "final_warped": self.stage_b.warp(moving, coarse["coarse_flow"]),
                "control_map": coarse["coarse_warped"].new_zeros(
                    coarse["coarse_warped"].shape[0], 1, *coarse["coarse_warped"].shape[2:]
                ),
            }

        out = {**coarse, **stage_b_out}
        if roi_condition is not None and roi_weight is not None:
            out["roi_condition"] = roi_condition
            out["roi_weight"] = roi_weight
        if moving_label is not None:
            out["coarse_warped_label"] = coarse_warped_label
            out["final_warped_label"] = self.warp_label(moving_label.float(), stage_b_out["final_flow"])
        return out


def build_v9_3_1_roib_downsampled_shallowa_model_bundle(
    inshape: tuple[int, int, int],
    int_steps: int = 7,
    enable_stage_b: bool = True,
    enable_roi_b: bool = True,
    roi_classes: tuple[int, ...] = (),
    roi_dilation_radius: int = 3,
    roi_condition_channels: int = 3,
    stage_a_downsample_factor: int = 4,
) -> nn.ModuleDict:
    """931 专用模型构建函数。"""

    reg_model = RegROIBv931Model(
        inshape=inshape,
        int_steps=int_steps,
        enable_stage_b=enable_stage_b,
        enable_roi_b=enable_roi_b,
        roi_classes=roi_classes,
        roi_dilation_radius=roi_dilation_radius,
        roi_condition_channels=roi_condition_channels,
        stage_a_downsample_factor=stage_a_downsample_factor,
    )
    return nn.ModuleDict({"reg_model": reg_model})


# ==============================
# Reg v9.3.2 下采样 Stage-A + 原版 Dual-B
# ==============================
class RegDownsampledDualBv932Model(nn.Module):
    """932 主模型：930 的低分辨率 A，接 900 的 Dual-B refine。"""

    def __init__(
        self,
        inshape: tuple[int, int, int],
        int_steps: int = 7,
        stage_b_fusion_mode: str = "film",
        enable_stage_b: bool = True,
        stage_a_downsample_factor: int = 4,
    ):
        super().__init__()
        self.enable_stage_b = enable_stage_b
        self.coarse_stage = DownsampledCoarseRegistrationStage930(
            inshape=inshape,
            downsample_factor=stage_a_downsample_factor,
            int_steps=int_steps,
        )
        self.stage_b = DualBRefineStage(
            inshape=inshape,
            int_steps=int_steps,
            fusion_mode=stage_b_fusion_mode,
        )
        self.warp_label = SpatialTransformer(inshape, mode="nearest")

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        moving_label: torch.Tensor | None = None,
        detach_stage_a_for_b: bool = False,
    ) -> dict[str, torch.Tensor]:
        coarse = self.coarse_stage(moving, fixed)
        coarse_warped_label = None
        if moving_label is not None:
            coarse_warped_label = self.warp_label(moving_label.float(), coarse["coarse_flow"])
        if self.enable_stage_b:
            coarse_warped_for_b = coarse["coarse_warped"].detach() if detach_stage_a_for_b else coarse["coarse_warped"]
            coarse_flow_for_b = coarse["coarse_flow"].detach() if detach_stage_a_for_b else coarse["coarse_flow"]
            stage_b_out = self.stage_b(moving, coarse_warped_for_b, fixed, coarse_flow_for_b)
        else:
            batch_size = moving.shape[0]
            pooled_shape = (batch_size, self.stage_b.deep_ch)
            stage_b_out = {
                "common_a_vec": moving.new_zeros(pooled_shape),
                "common_f_vec": moving.new_zeros(pooled_shape),
                "diff_a_vec": moving.new_zeros(pooled_shape),
                "diff_f_vec": moving.new_zeros(pooled_shape),
                "diff_gap": torch.zeros_like(coarse["coarse_feat"]),
                "residual_svf": torch.zeros_like(coarse["coarse_svf"]),
                "residual_flow": torch.zeros_like(coarse["coarse_flow"]),
                "final_flow": coarse["coarse_flow"],
                "final_warped": coarse["coarse_warped"],
                "control_map": torch.zeros_like(coarse["coarse_feat"]),
            }
        out = {**coarse, **stage_b_out}
        if moving_label is not None:
            out["coarse_warped_label"] = coarse_warped_label
            out["final_warped_label"] = self.warp_label(moving_label.float(), stage_b_out["final_flow"])
        return out


def build_v9_3_2_downsampled_dualb_model_bundle(
    inshape: tuple[int, int, int],
    int_steps: int = 7,
    stage_b_fusion_mode: str = "film",
    enable_stage_b: bool = True,
    stage_a_downsample_factor: int = 4,
) -> nn.ModuleDict:
    """932 专用模型构建函数。"""

    reg_model = RegDownsampledDualBv932Model(
        inshape=inshape,
        int_steps=int_steps,
        stage_b_fusion_mode=stage_b_fusion_mode,
        enable_stage_b=enable_stage_b,
        stage_a_downsample_factor=stage_a_downsample_factor,
    )
    return nn.ModuleDict({"reg_model": reg_model})


# ==============================
# Reg v9.1.1 / v9.1.2 显式 ROI-B
# ==============================
class ExplicitROIBuilder911(nn.Module):
    """显式 ROI 条件构造器：按类别生成 moving-side ROI。"""

    def __init__(
        self,
        roi_classes: tuple[int, ...],
        dilation_radius: int = 3,
        include_fixed_edge: bool = False,
    ):
        super().__init__()
        self.roi_classes = tuple(int(c) for c in roi_classes)
        self.dilation_radius = int(dilation_radius)
        self.include_fixed_edge = bool(include_fixed_edge)

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask.float()
        kernel_size = radius * 2 + 1
        return F.max_pool3d(mask.float(), kernel_size=kernel_size, stride=1, padding=radius)

    @staticmethod
    def _fixed_edge_map(fixed: torch.Tensor) -> torch.Tensor:
        dz = F.pad(torch.abs(fixed[:, :, 1:, :, :] - fixed[:, :, :-1, :, :]), (0, 0, 0, 0, 0, 1))
        dy = F.pad(torch.abs(fixed[:, :, :, 1:, :] - fixed[:, :, :, :-1, :]), (0, 0, 0, 1, 0, 0))
        dx = F.pad(torch.abs(fixed[:, :, :, :, 1:] - fixed[:, :, :, :, :-1]), (0, 1, 0, 0, 0, 0))
        edge = dz + dy + dx
        edge_mean = edge.mean(dim=(2, 3, 4), keepdim=True)
        edge_std = edge.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        return torch.clamp((edge - edge_mean) / edge_std, min=0.0, max=3.0) / 3.0

    def forward(
        self,
        coarse_warped: torch.Tensor,
        coarse_warped_label: torch.Tensor,
        fixed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self.roi_classes:
            raise ValueError("Explicit ROI models require non-empty roi_classes")

        label = coarse_warped_label.round().long()
        roi_parts: list[torch.Tensor] = []
        for class_id in self.roi_classes:
            class_mask = (label == class_id).float()
            roi_parts.append(self._dilate_mask(class_mask, self.dilation_radius).clamp(0.0, 1.0))

        class_roi = torch.cat(roi_parts, dim=1)
        union_roi = class_roi.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        roi_sum = class_roi.sum(dim=1, keepdim=True)
        class_weight = torch.where(
            roi_sum > 0,
            class_roi / roi_sum.clamp_min(1e-6),
            torch.zeros_like(class_roi),
        )
        abs_diff = torch.abs(coarse_warped - fixed)
        fixed_edge = self._fixed_edge_map(fixed).detach() if self.include_fixed_edge else None
        roi_weight = 1.0 + union_roi.detach()
        return {
            "abs_diff": abs_diff,
            "class_roi": class_roi.detach(),
            "class_weight": class_weight.detach(),
            "union_roi": union_roi.detach(),
            "fixed_edge": fixed_edge,
            "roi_weight": roi_weight,
        }


class ConditionedResidualFeatureStage(nn.Module):
    """显式 ROI-B 的基础特征提取器：单 encoder + 条件解码。"""

    def __init__(
        self,
        condition_channels: int,
        enc_channels: tuple[int, int, int, int] = (16, 32, 64, 96),
    ):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.deep_ch = enc_channels[-1]
        self.encoder = SharedEncoder(2 + self.condition_channels, enc_channels)
        self.decoder = ROIRefineDecoder910(
            bottleneck_channels=self.deep_ch,
            skip_channels=(enc_channels[-2], enc_channels[-3], enc_channels[-4]),
            condition_channels=self.condition_channels,
        )

    def forward(
        self,
        coarse_warped: torch.Tensor,
        fixed: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.condition_channels > 0:
            if condition is None:
                condition = coarse_warped.new_zeros(
                    coarse_warped.shape[0],
                    self.condition_channels,
                    *coarse_warped.shape[2:],
                )
            encoder_input = torch.cat([coarse_warped, fixed, condition], dim=1)
        else:
            condition = None
            encoder_input = torch.cat([coarse_warped, fixed], dim=1)
        feats = self.encoder(encoder_input)
        return self.decoder(feats[-1], feats, coarse_warped, fixed, condition)


class ExplicitROIOnceStage911(nn.Module):
    """911: 一次前向输出 global residual 与 class-wise residual。"""

    def __init__(
        self,
        inshape,
        class_count: int,
        condition_channels: int,
        int_steps: int = 7,
        enable_global_residual: bool = True,
        class_flow_scale: float = 1.0,
    ):
        super().__init__()
        self.class_count = int(class_count)
        self.enable_global_residual = bool(enable_global_residual)
        self.class_flow_scale = float(class_flow_scale)
        self.feature_stage = ConditionedResidualFeatureStage(condition_channels=condition_channels)
        self.global_head = (
            nn.Conv3d(self.feature_stage.decoder.out_channels, 3, kernel_size=3, padding=1)
            if self.enable_global_residual
            else None
        )
        self.class_head = nn.Conv3d(
            self.feature_stage.decoder.out_channels,
            self.class_count * 3,
            kernel_size=3,
            padding=1,
        )
        self.integrate = VecInt(inshape, nsteps=int_steps)

        if self.global_head is not None:
            nn.init.normal_(self.global_head.weight, mean=0.0, std=1e-5)
            nn.init.zeros_(self.global_head.bias)
        nn.init.normal_(self.class_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.class_head.bias)

    def forward(
        self,
        coarse_warped: torch.Tensor,
        fixed: torch.Tensor,
        condition: torch.Tensor,
        class_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feat = self.feature_stage(coarse_warped, fixed, condition)
        if self.global_head is not None:
            global_svf = self.global_head(feat)
            global_flow = self.integrate(global_svf)
        else:
            global_svf = coarse_warped.new_zeros(coarse_warped.shape[0], 3, *coarse_warped.shape[2:])
            global_flow = torch.zeros_like(global_svf)

        bsz = coarse_warped.shape[0]
        spatial_shape = coarse_warped.shape[2:]
        class_svf = self.class_head(feat).view(bsz, self.class_count, 3, *spatial_shape)
        class_flow = self.integrate(class_svf.reshape(bsz * self.class_count, 3, *spatial_shape)).view(
            bsz,
            self.class_count,
            3,
            *spatial_shape,
        )
        class_weight_expanded = class_weight.unsqueeze(2)
        fused_class_svf = (class_svf * class_weight_expanded).sum(dim=1)
        fused_class_flow = (class_flow * class_weight_expanded).sum(dim=1)
        residual_svf = global_svf + self.class_flow_scale * fused_class_svf
        residual_flow = global_flow + self.class_flow_scale * fused_class_flow
        return {
            "global_residual_svf": global_svf,
            "global_residual_flow": global_flow,
            "class_residual_svf": class_svf,
            "class_residual_flow": class_flow,
            "fused_class_residual_svf": fused_class_svf,
            "fused_class_residual_flow": fused_class_flow,
            "residual_svf": residual_svf,
            "residual_flow": residual_flow,
            "class_flow_mag": class_flow.abs().mean(),
        }


class ConditionedResidualPredictor(nn.Module):
    """显式 ROI-B 的单个 residual predictor。"""

    def __init__(
        self,
        inshape,
        condition_channels: int,
        int_steps: int = 7,
    ):
        super().__init__()
        self.feature_stage = ConditionedResidualFeatureStage(condition_channels=condition_channels)
        self.flow_head = nn.Conv3d(self.feature_stage.decoder.out_channels, 3, kernel_size=3, padding=1)
        self.integrate = VecInt(inshape, nsteps=int_steps)
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_head.bias)

    def forward(
        self,
        coarse_warped: torch.Tensor,
        fixed: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feat = self.feature_stage(coarse_warped, fixed, condition)
        residual_svf = self.flow_head(feat)
        residual_flow = self.integrate(residual_svf)
        return {"residual_svf": residual_svf, "residual_flow": residual_flow}


class ROIPerClassStage912(nn.Module):
    """912: 每类单独前向，再做 class-wise 融合。"""

    def __init__(
        self,
        inshape,
        class_count: int,
        global_condition_channels: int,
        class_condition_channels: int,
        int_steps: int = 7,
        enable_global_residual: bool = True,
        class_flow_scale: float = 1.0,
    ):
        super().__init__()
        self.class_count = int(class_count)
        self.enable_global_residual = bool(enable_global_residual)
        self.class_flow_scale = float(class_flow_scale)
        self.global_predictor = (
            ConditionedResidualPredictor(inshape, global_condition_channels, int_steps=int_steps)
            if self.enable_global_residual
            else None
        )
        self.class_predictor = ConditionedResidualPredictor(
            inshape,
            class_condition_channels,
            int_steps=int_steps,
        )

    def forward(
        self,
        coarse_warped: torch.Tensor,
        fixed: torch.Tensor,
        global_condition: torch.Tensor,
        class_conditions: list[torch.Tensor],
        class_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.global_predictor is not None:
            global_out = self.global_predictor(coarse_warped, fixed, global_condition)
            global_svf = global_out["residual_svf"]
            global_flow = global_out["residual_flow"]
        else:
            global_svf = coarse_warped.new_zeros(coarse_warped.shape[0], 3, *coarse_warped.shape[2:])
            global_flow = torch.zeros_like(global_svf)

        class_svf_parts: list[torch.Tensor] = []
        class_flow_parts: list[torch.Tensor] = []
        for condition_k in class_conditions:
            class_out = self.class_predictor(coarse_warped, fixed, condition_k)
            class_svf_parts.append(class_out["residual_svf"])
            class_flow_parts.append(class_out["residual_flow"])

        class_svf = torch.stack(class_svf_parts, dim=1)
        class_flow = torch.stack(class_flow_parts, dim=1)
        class_weight_expanded = class_weight.unsqueeze(2)
        fused_class_svf = (class_svf * class_weight_expanded).sum(dim=1)
        fused_class_flow = (class_flow * class_weight_expanded).sum(dim=1)
        residual_svf = global_svf + self.class_flow_scale * fused_class_svf
        residual_flow = global_flow + self.class_flow_scale * fused_class_flow
        return {
            "global_residual_svf": global_svf,
            "global_residual_flow": global_flow,
            "class_residual_svf": class_svf,
            "class_residual_flow": class_flow,
            "fused_class_residual_svf": fused_class_svf,
            "fused_class_residual_flow": fused_class_flow,
            "residual_svf": residual_svf,
            "residual_flow": residual_flow,
            "class_flow_mag": class_flow.abs().mean(),
        }


class RegExplicitROIV911Model(nn.Module):
    """911 主模型：一次前向的显式 class-wise ROI-B。"""

    def __init__(
        self,
        inshape: tuple[int, int, int],
        roi_classes: tuple[int, ...],
        int_steps: int = 7,
        enable_stage_b: bool = True,
        enable_global_residual: bool = True,
        roi_dilation_radius: int = 3,
        include_fixed_edge: bool = False,
        class_flow_scale: float = 1.0,
    ):
        super().__init__()
        self.enable_stage_b = enable_stage_b
        self.enable_global_residual = enable_global_residual
        self.class_count = len(tuple(int(c) for c in roi_classes))
        self.coarse_stage = CoarseRegistrationStage(inshape, int_steps=int_steps)
        self.roi_builder = ExplicitROIBuilder911(
            roi_classes=roi_classes,
            dilation_radius=roi_dilation_radius,
            include_fixed_edge=include_fixed_edge,
        )
        condition_channels = 1 + self.class_count + (1 if include_fixed_edge else 0)
        self.stage_b = ExplicitROIOnceStage911(
            inshape=inshape,
            class_count=self.class_count,
            condition_channels=condition_channels,
            int_steps=int_steps,
            enable_global_residual=enable_global_residual,
            class_flow_scale=class_flow_scale,
        )
        self.warp = SpatialTransformer(inshape)
        self.warp_label = SpatialTransformer(inshape, mode="nearest")

    @staticmethod
    def compose_flows(
        coarse_flow: torch.Tensor,
        residual_flow: torch.Tensor,
        warp: SpatialTransformer,
    ) -> torch.Tensor:
        return residual_flow + warp(coarse_flow, residual_flow)

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        moving_label: torch.Tensor | None = None,
        detach_stage_a_for_b: bool = False,
    ) -> dict[str, torch.Tensor]:
        coarse = self.coarse_stage(moving, fixed)
        out = dict(coarse)
        if moving_label is None:
            raise ValueError("Explicit ROI models require moving_label")

        coarse_flow_for_b = coarse["coarse_flow"].detach() if detach_stage_a_for_b else coarse["coarse_flow"]
        coarse_warped_for_b = coarse["coarse_warped"].detach() if detach_stage_a_for_b else coarse["coarse_warped"]
        coarse_warped_label = self.warp_label(moving_label.float(), coarse_flow_for_b)
        out["coarse_warped_label"] = coarse_warped_label

        if not self.enable_stage_b:
            zero_flow = torch.zeros_like(coarse["coarse_flow"])
            out.update(
                {
                    "global_residual_svf": zero_flow,
                    "global_residual_flow": zero_flow,
                    "class_residual_svf": zero_flow.unsqueeze(1).repeat(1, self.class_count, 1, 1, 1, 1),
                    "class_residual_flow": zero_flow.unsqueeze(1).repeat(1, self.class_count, 1, 1, 1, 1),
                    "fused_class_residual_svf": zero_flow,
                    "fused_class_residual_flow": zero_flow,
                    "residual_svf": zero_flow,
                    "residual_flow": zero_flow,
                    "final_flow": coarse["coarse_flow"],
                    "final_warped": coarse["coarse_warped"],
                    "final_warped_label": self.warp_label(moving_label.float(), coarse["coarse_flow"]),
                    "class_flow_mag": coarse["coarse_flow"].new_tensor(0.0),
                    "control_map": coarse["coarse_warped"].new_zeros(
                        coarse["coarse_warped"].shape[0],
                        1,
                        *coarse["coarse_warped"].shape[2:],
                    ),
                }
            )
            return out

        roi_pack = self.roi_builder(coarse_warped_for_b, coarse_warped_label, fixed)
        condition_parts = [roi_pack["abs_diff"], roi_pack["class_roi"]]
        if roi_pack["fixed_edge"] is not None:
            condition_parts.append(roi_pack["fixed_edge"])
        condition = torch.cat(condition_parts, dim=1)
        stage_b_out = self.stage_b(
            coarse_warped=coarse_warped_for_b,
            fixed=fixed,
            condition=condition,
            class_weight=roi_pack["class_weight"],
        )
        final_flow = self.compose_flows(coarse_flow_for_b, stage_b_out["residual_flow"], self.warp)
        final_warped = self.warp(moving, final_flow)
        out.update(stage_b_out)
        out.update(
            {
                "explicit_condition": condition,
                "class_roi": roi_pack["class_roi"],
                "class_weight": roi_pack["class_weight"],
                "roi_weight": roi_pack["roi_weight"],
                "control_map": roi_pack["union_roi"],
                "final_flow": final_flow,
                "final_warped": final_warped,
                "final_warped_label": self.warp_label(moving_label.float(), final_flow),
            }
        )
        return out


class RegPerClassROIV912Model(nn.Module):
    """912 主模型：按类别多次前向的显式 class-wise ROI-B。"""

    def __init__(
        self,
        inshape: tuple[int, int, int],
        roi_classes: tuple[int, ...],
        int_steps: int = 7,
        enable_stage_b: bool = True,
        enable_global_residual: bool = True,
        roi_dilation_radius: int = 3,
        include_fixed_edge: bool = False,
        class_flow_scale: float = 1.0,
    ):
        super().__init__()
        self.enable_stage_b = enable_stage_b
        self.enable_global_residual = enable_global_residual
        self.roi_classes = tuple(int(c) for c in roi_classes)
        self.class_count = len(self.roi_classes)
        self.include_fixed_edge = include_fixed_edge
        self.coarse_stage = CoarseRegistrationStage(inshape, int_steps=int_steps)
        self.roi_builder = ExplicitROIBuilder911(
            roi_classes=roi_classes,
            dilation_radius=roi_dilation_radius,
            include_fixed_edge=include_fixed_edge,
        )
        global_condition_channels = 2 + (1 if include_fixed_edge else 0)
        class_condition_channels = 3 + (1 if include_fixed_edge else 0)
        self.stage_b = ROIPerClassStage912(
            inshape=inshape,
            class_count=self.class_count,
            global_condition_channels=global_condition_channels,
            class_condition_channels=class_condition_channels,
            int_steps=int_steps,
            enable_global_residual=enable_global_residual,
            class_flow_scale=class_flow_scale,
        )
        self.warp = SpatialTransformer(inshape)
        self.warp_label = SpatialTransformer(inshape, mode="nearest")

    @staticmethod
    def compose_flows(
        coarse_flow: torch.Tensor,
        residual_flow: torch.Tensor,
        warp: SpatialTransformer,
    ) -> torch.Tensor:
        return residual_flow + warp(coarse_flow, residual_flow)

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        moving_label: torch.Tensor | None = None,
        detach_stage_a_for_b: bool = False,
    ) -> dict[str, torch.Tensor]:
        coarse = self.coarse_stage(moving, fixed)
        out = dict(coarse)
        if moving_label is None:
            raise ValueError("Explicit ROI models require moving_label")

        coarse_flow_for_b = coarse["coarse_flow"].detach() if detach_stage_a_for_b else coarse["coarse_flow"]
        coarse_warped_for_b = coarse["coarse_warped"].detach() if detach_stage_a_for_b else coarse["coarse_warped"]
        coarse_warped_label = self.warp_label(moving_label.float(), coarse_flow_for_b)
        out["coarse_warped_label"] = coarse_warped_label

        if not self.enable_stage_b:
            zero_flow = torch.zeros_like(coarse["coarse_flow"])
            out.update(
                {
                    "global_residual_svf": zero_flow,
                    "global_residual_flow": zero_flow,
                    "class_residual_svf": zero_flow.unsqueeze(1).repeat(1, self.class_count, 1, 1, 1, 1),
                    "class_residual_flow": zero_flow.unsqueeze(1).repeat(1, self.class_count, 1, 1, 1, 1),
                    "fused_class_residual_svf": zero_flow,
                    "fused_class_residual_flow": zero_flow,
                    "residual_svf": zero_flow,
                    "residual_flow": zero_flow,
                    "final_flow": coarse["coarse_flow"],
                    "final_warped": coarse["coarse_warped"],
                    "final_warped_label": self.warp_label(moving_label.float(), coarse["coarse_flow"]),
                    "class_flow_mag": coarse["coarse_flow"].new_tensor(0.0),
                    "control_map": coarse["coarse_warped"].new_zeros(
                        coarse["coarse_warped"].shape[0],
                        1,
                        *coarse["coarse_warped"].shape[2:],
                    ),
                }
            )
            return out

        roi_pack = self.roi_builder(coarse_warped_for_b, coarse_warped_label, fixed)
        global_condition_parts = [roi_pack["abs_diff"], roi_pack["union_roi"]]
        if roi_pack["fixed_edge"] is not None:
            global_condition_parts.append(roi_pack["fixed_edge"])
        global_condition = torch.cat(global_condition_parts, dim=1)

        class_conditions: list[torch.Tensor] = []
        for class_idx in range(self.class_count):
            class_code = roi_pack["class_roi"].new_full(
                (
                    roi_pack["class_roi"].shape[0],
                    1,
                    *roi_pack["class_roi"].shape[2:],
                ),
                fill_value=float(class_idx + 1) / float(max(self.class_count, 1)),
            )
            parts = [roi_pack["abs_diff"], roi_pack["class_roi"][:, class_idx : class_idx + 1], class_code]
            if roi_pack["fixed_edge"] is not None:
                parts.append(roi_pack["fixed_edge"])
            class_conditions.append(torch.cat(parts, dim=1))

        stage_b_out = self.stage_b(
            coarse_warped=coarse_warped_for_b,
            fixed=fixed,
            global_condition=global_condition,
            class_conditions=class_conditions,
            class_weight=roi_pack["class_weight"],
        )
        final_flow = self.compose_flows(coarse_flow_for_b, stage_b_out["residual_flow"], self.warp)
        final_warped = self.warp(moving, final_flow)
        out.update(stage_b_out)
        out.update(
            {
                "class_roi": roi_pack["class_roi"],
                "class_weight": roi_pack["class_weight"],
                "roi_weight": roi_pack["roi_weight"],
                "control_map": roi_pack["union_roi"],
                "global_condition": global_condition,
                "final_flow": final_flow,
                "final_warped": final_warped,
                "final_warped_label": self.warp_label(moving_label.float(), final_flow),
            }
        )
        return out


def build_v9_1_1_explicit_roi_model_bundle(
    inshape: tuple[int, int, int],
    roi_classes: tuple[int, ...],
    int_steps: int = 7,
    enable_stage_b: bool = True,
    enable_global_residual: bool = True,
    roi_dilation_radius: int = 3,
    include_fixed_edge: bool = False,
    class_flow_scale: float = 1.0,
) -> nn.ModuleDict:
    reg_model = RegExplicitROIV911Model(
        inshape=inshape,
        roi_classes=roi_classes,
        int_steps=int_steps,
        enable_stage_b=enable_stage_b,
        enable_global_residual=enable_global_residual,
        roi_dilation_radius=roi_dilation_radius,
        include_fixed_edge=include_fixed_edge,
        class_flow_scale=class_flow_scale,
    )
    return nn.ModuleDict({"reg_model": reg_model})


def build_v9_1_2_perclass_roi_model_bundle(
    inshape: tuple[int, int, int],
    roi_classes: tuple[int, ...],
    int_steps: int = 7,
    enable_stage_b: bool = True,
    enable_global_residual: bool = True,
    roi_dilation_radius: int = 3,
    include_fixed_edge: bool = False,
    class_flow_scale: float = 1.0,
) -> nn.ModuleDict:
    reg_model = RegPerClassROIV912Model(
        inshape=inshape,
        roi_classes=roi_classes,
        int_steps=int_steps,
        enable_stage_b=enable_stage_b,
        enable_global_residual=enable_global_residual,
        roi_dilation_radius=roi_dilation_radius,
        include_fixed_edge=include_fixed_edge,
        class_flow_scale=class_flow_scale,
    )
    return nn.ModuleDict({"reg_model": reg_model})


# ==============================
# Reg v9.2.0 单器官 ROI-aware 实验模型
# ==============================
class SingleOrganROIBuilder920(nn.Module):
    """920 专用单器官 ROI 构造器：从 moving label 提取目标类 ROI。"""

    def __init__(self, target_class: int = 1, dilation_radius: int = 5, condition_channels: int = 3):
        super().__init__()
        self.target_class = int(target_class)
        self.dilation_radius = int(dilation_radius)
        self.condition_channels = int(condition_channels)

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask.float()
        kernel_size = radius * 2 + 1
        return F.max_pool3d(mask.float(), kernel_size=kernel_size, stride=1, padding=radius)

    @staticmethod
    def _erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask.float()
        kernel_size = radius * 2 + 1
        inv = 1.0 - mask.float()
        return 1.0 - F.max_pool3d(inv, kernel_size=kernel_size, stride=1, padding=radius)

    @staticmethod
    def _fixed_edge_map(fixed: torch.Tensor) -> torch.Tensor:
        dz = F.pad(torch.abs(fixed[:, :, 1:, :, :] - fixed[:, :, :-1, :, :]), (0, 0, 0, 0, 0, 1))
        dy = F.pad(torch.abs(fixed[:, :, :, 1:, :] - fixed[:, :, :, :-1, :]), (0, 0, 0, 1, 0, 0))
        dx = F.pad(torch.abs(fixed[:, :, :, :, 1:] - fixed[:, :, :, :, :-1]), (0, 1, 0, 0, 0, 0))
        edge = dz + dy + dx
        edge_mean = edge.mean(dim=(2, 3, 4), keepdim=True)
        edge_std = edge.std(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
        return torch.clamp((edge - edge_mean) / edge_std, min=0.0, max=3.0) / 3.0

    def forward(self, moving: torch.Tensor, moving_label: torch.Tensor, fixed: torch.Tensor) -> dict[str, torch.Tensor]:
        label = moving_label.round().long()
        roi_mask = (label == self.target_class).float()
        soft_roi = self._dilate_mask(roi_mask, self.dilation_radius).clamp(0.0, 1.0)
        boundary_radius = max(1, min(self.dilation_radius, 2))
        boundary = (
            self._dilate_mask(roi_mask, boundary_radius) - self._erode_mask(roi_mask, boundary_radius)
        ).clamp(0.0, 1.0)
        fixed_edge = self._fixed_edge_map(fixed).detach()
        roi_image = moving * soft_roi.detach()
        roi_valid = (roi_mask.flatten(1).sum(dim=1) > 0).float().view(-1, 1, 1, 1, 1)

        if self.condition_channels <= 0:
            condition = moving.new_zeros(moving.shape[0], 0, *moving.shape[2:])
        else:
            condition_parts = [soft_roi.detach(), boundary.detach(), fixed_edge]
            condition = torch.cat(condition_parts[: self.condition_channels], dim=1)
            if condition.shape[1] < self.condition_channels:
                pad_channels = self.condition_channels - condition.shape[1]
                condition = torch.cat(
                    [condition, condition.new_zeros(condition.shape[0], pad_channels, *condition.shape[2:])],
                    dim=1,
                )

        roi_weight = 1.0 + soft_roi.detach()
        return {
            "roi_mask": roi_mask,
            "soft_roi": soft_roi.detach(),
            "boundary": boundary.detach(),
            "fixed_edge": fixed_edge,
            "roi_image": roi_image,
            "condition": condition,
            "roi_weight": roi_weight,
            "roi_valid": roi_valid,
        }


class ROIRegistrationDecoder920(nn.Module):
    """920 专用解码器：在尾部只回注 ROI-only moving 与 fixed。"""

    def __init__(
        self,
        bottleneck_channels: int,
        skip_channels: tuple[int, int, int] = (96, 48, 24),
        condition_channels: int = 0,
        out_channels: tuple[int, int, int, int] = (96, 48, 24, 24),
    ):
        super().__init__()
        self.condition_channels = int(condition_channels)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(bottleneck_channels + skip_channels[0], out_channels[0]),
                ConvBlock(out_channels[0] + skip_channels[1], out_channels[1]),
                ConvBlock(out_channels[1] + skip_channels[2], out_channels[2]),
                ConvBlock(out_channels[2] + 2 + self.condition_channels, out_channels[3]),
            ]
        )
        self.out_channels = out_channels[-1]

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: list[torch.Tensor],
        roi_image: torch.Tensor,
        fixed: torch.Tensor,
        roi_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.up(bottleneck)
        x = self.blocks[0](torch.cat([x, skips[-2]], dim=1))
        x = self.up(x)
        x = self.blocks[1](torch.cat([x, skips[-3]], dim=1))
        x = self.up(x)
        x = self.blocks[2](torch.cat([x, skips[-4]], dim=1))
        x = self.up(x)
        tail_inputs = [x, roi_image, fixed]
        if self.condition_channels > 0:
            if roi_condition is None:
                roi_condition = roi_image.new_zeros(roi_image.shape[0], self.condition_channels, *roi_image.shape[2:])
            tail_inputs.append(roi_condition)
        x = self.blocks[3](torch.cat(tail_inputs, dim=1))
        return x


class RegSingleOrganROI920Model(nn.Module):
    """920 主模型：moving 侧只使用 ROI 图像的单阶段配准。"""

    def __init__(
        self,
        inshape: tuple[int, int, int],
        int_steps: int = 7,
        target_class: int = 1,
        roi_dilation_radius: int = 5,
        roi_condition_channels: int = 3,
        enc_channels: tuple[int, int, int, int] = (24, 48, 96, 144),
    ):
        super().__init__()
        self.target_class = int(target_class)
        self.roi_condition_channels = int(roi_condition_channels)
        self.roi_builder = SingleOrganROIBuilder920(
            target_class=target_class,
            dilation_radius=roi_dilation_radius,
            condition_channels=roi_condition_channels,
        )
        self.encoder = SharedEncoder(2 + self.roi_condition_channels, enc_channels)
        self.decoder = ROIRegistrationDecoder920(
            bottleneck_channels=enc_channels[-1],
            skip_channels=(enc_channels[-2], enc_channels[-3], enc_channels[-4]),
            condition_channels=self.roi_condition_channels,
            out_channels=(enc_channels[-2], enc_channels[-3], enc_channels[-4], enc_channels[-4]),
        )
        self.flow_head = nn.Conv3d(self.decoder.out_channels, 3, kernel_size=3, padding=1)
        self.integrate = VecInt(inshape, nsteps=int_steps)
        self.warp = SpatialTransformer(inshape)
        self.warp_label = SpatialTransformer(inshape, mode="nearest")
        nn.init.normal_(self.flow_head.weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.flow_head.bias)

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        moving_label: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if moving_label is None:
            raise ValueError("RegSingleOrganROI920Model requires moving_label")

        roi_pack = self.roi_builder(moving, moving_label, fixed)
        # reg920 的实验前提是 moving 侧不再输入整图，只保留目标器官 ROI 图像。
        encoder_input = torch.cat([roi_pack["roi_image"], fixed, roi_pack["condition"]], dim=1)
        feats = self.encoder(encoder_input)
        dec = self.decoder(
            feats[-1],
            feats,
            roi_pack["roi_image"],
            fixed,
            roi_pack["condition"],
        )
        final_svf = self.flow_head(dec)
        final_flow = self.integrate(final_svf)
        final_warped = self.warp(roi_pack["roi_image"], final_flow)
        final_warped_label = self.warp_label(moving_label.float(), final_flow)
        return {
            "final_svf": final_svf,
            "final_flow": final_flow,
            "final_warped": final_warped,
            "final_warped_label": final_warped_label,
            "roi_weight": roi_pack["roi_weight"],
            "roi_valid": roi_pack["roi_valid"],
            "control_map": roi_pack["soft_roi"],
            "roi_image": roi_pack["roi_image"],
            "roi_mask": roi_pack["roi_mask"],
        }


def build_v9_2_0_singleorgan_model_bundle(
    inshape: tuple[int, int, int],
    int_steps: int = 7,
    target_class: int = 1,
    roi_dilation_radius: int = 5,
    roi_condition_channels: int = 3,
    enc_channels: tuple[int, int, int, int] = (24, 48, 96, 144),
) -> nn.ModuleDict:
    reg_model = RegSingleOrganROI920Model(
        inshape=inshape,
        int_steps=int_steps,
        target_class=target_class,
        roi_dilation_radius=roi_dilation_radius,
        roi_condition_channels=roi_condition_channels,
        enc_channels=enc_channels,
    )
    return nn.ModuleDict({"reg_model": reg_model})


# ==============================
# Seg v8 基线（UNet + MMD）
# ==============================
class SegConvBlock(nn.Module):
    """分割基础卷积块：双 Conv3d + IN + LeakyReLU。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DistillSegUNet(nn.Module):
    """轻量 3D UNet，返回 logits 与多层特征。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 16,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        self.dropout = nn.Dropout3d(dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.enc1 = SegConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = SegConvBlock(c1, c2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = SegConvBlock(c2, c3)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = SegConvBlock(c3, c4)
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = SegConvBlock(c3 + c3, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = SegConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = SegConvBlock(c1 + c1, c1)
        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.dropout(self.bottleneck(self.pool3(e3)))     # bottleneck 负责汇聚全局语义
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # 逐级拼接 skip，恢复空间细节
        d2 = self.dropout(self.dec2(torch.cat([self.up2(d3), e2], dim=1)))
        d1 = self.dropout(self.dec1(torch.cat([self.up1(d2), e1], dim=1)))
        logits = self.head(d1)
        features = [e1, e2, e3, b, d3, d2, d1]
        return logits, features


class SingleSegUNetMMD(nn.Module):
    """v8 seg：单网络 UNet（baseline）+ 中层特征 MMD。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 16,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.unet = DistillSegUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            dropout_prob=dropout_prob,
        )

    def forward(
        self,
        x_syn: torch.Tensor | None = None,
        x_real: torch.Tensor | None = None,
        is_training: bool = True,
    ) -> dict[str, torch.Tensor]:
        if not is_training:
            if x_real is None:
                raise ValueError("x_real is required during inference")
            real_logits, _ = self.unet(x_real)
            return {"real_logits": real_logits}

        if x_syn is None:
            raise ValueError("x_syn is required during training")

        syn_logits, syn_features = self.unet(x_syn)
        out = {
            "syn_logits": syn_logits,
            "syn_features": syn_features,  # 供 MMD 在指定层对齐 syn / real 中间分布
        }
        if x_real is not None:
            real_logits, real_features = self.unet(x_real)
            out["real_logits"] = real_logits
            out["real_features"] = real_features
        return out


class SingleSegUNetBasic(nn.Module):
    """v0000：最基础的单网络 3D UNet。"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 16,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.unet = DistillSegUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, _ = self.unet(x)
        return {"logits": logits}


class DualSyntheticTeacherNet(nn.Module):
    """v1110 教师网络：联合读取 reg/vae 两路 synthetic 输入，输出两份 soft logits。"""

    def __init__(
        self,
        num_classes: int,
        image_channels: int = 1,
        base_channels: int = 16,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        in_channels = image_channels * 2 + num_classes * 2
        out_channels = num_classes * 2
        self.num_classes = num_classes
        self.unet = DistillSegUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            dropout_prob=dropout_prob,
        )

    def forward(
        self,
        reg_syn_image: torch.Tensor,
        reg_syn_label_onehot: torch.Tensor,
        vae_syn_image: torch.Tensor,
        vae_syn_label_onehot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        teacher_input = torch.cat(
            [reg_syn_image, reg_syn_label_onehot, vae_syn_image, vae_syn_label_onehot],
            dim=1,
        )
        logits, features = self.unet(teacher_input)
        reg_teacher_logits, vae_teacher_logits = torch.split(logits, self.num_classes, dim=1)
        return {
            "reg_teacher_logits": reg_teacher_logits,
            "vae_teacher_logits": vae_teacher_logits,
            "teacher_features": features,
        }


class FlowEncoder3D(nn.Module):
    """3D flow VAE 编码器。"""

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, base_channels, stride=2),
            ConvBlock(base_channels, base_channels * 2, stride=2),
            ConvBlock(base_channels * 2, base_channels * 4, stride=2),
            ConvBlock(base_channels * 4, base_channels * 6, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class FlowDecoder3D(nn.Module):
    """3D flow VAE 解码器。"""

    def __init__(self, out_channels: int = 3, base_channels: int = 16):
        super().__init__()
        hidden = base_channels * 6
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(hidden, base_channels * 4),
                ConvBlock(base_channels * 4, base_channels * 2),
                ConvBlock(base_channels * 2, base_channels),
                ConvBlock(base_channels, base_channels),
            ]
        )
        self.head = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, output_shape: tuple[int, int, int]) -> torch.Tensor:
        for block in self.blocks:
            x = self.up(x)
            x = block(x)
        flow = self.head(x)
        if flow.shape[2:] != output_shape:
            flow = F.interpolate(flow, size=output_shape, mode="trilinear", align_corners=True)
        return flow


class FlowVAE1110(nn.Module):
    """v1110：最小可行的 3D deformation flow VAE。"""

    def __init__(self, in_channels: int = 3, base_channels: int = 16, latent_dim: int = 128):
        super().__init__()
        self.encoder = FlowEncoder3D(in_channels=in_channels, base_channels=base_channels)
        hidden_channels = base_channels * 6
        self.hidden_channels = hidden_channels
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc_mu = nn.Linear(hidden_channels, latent_dim)
        self.fc_logvar = nn.Linear(hidden_channels, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden_channels)
        self.decoder = FlowDecoder3D(out_channels=in_channels, base_channels=base_channels)

    def encode(self, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(flow)
        pooled = self.pool(feat).flatten(1)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, output_shape: tuple[int, int, int]) -> torch.Tensor:
        feat = self.fc_decode(z).view(z.shape[0], self.hidden_channels, 1, 1, 1)
        return self.decoder(feat, output_shape=output_shape)

    def forward(self, flow: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(flow)
        z = self.reparameterize(mu, logvar)
        recon_flow = self.decode(z, output_shape=tuple(flow.shape[2:]))
        return {
            "recon_flow": recon_flow,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def sample(self, num_samples: int, output_shape: tuple[int, int, int], device: str | torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.fc_mu.out_features, device=device)
        return self.decode(z, output_shape=output_shape)


__all__ = [
    "SpatialTransformer",
    "CoarseRegistrationStage",
    "RegVM0000Model",
    "DualBRefineStage",
    "RegDualBModel",
    "build_v0000_vm_model_bundle",
    "build_v9_model_bundle",
    "ROIConditionBuilder910",
    "ROIRefineDecoder910",
    "ROIBRefineStage910",
    "RegROIBv910Model",
    "build_v9_1_0_roib_model_bundle",
    "DownsampledCoarseRegistrationStage930",
    "RegROIBv930Model",
    "build_downsampled_roib_reg_model_bundle",

    "build_cascaded_reg_model_bundle",
    "SingleOrganROIBuilder920",
    "ROIRegistrationDecoder920",
    "RegSingleOrganROI920Model",
    "build_v9_2_0_singleorgan_model_bundle",
    "DistillSegUNet",
    "SingleSegUNetMMD",
    "SingleSegUNetBasic",
    "DualSyntheticTeacherNet",
    "FlowVAE1110",
]


def build_cascaded_reg_model_bundle(*args, **kwargs):
    """Public cascaded_reg builder, equivalent to the reg930 4x-downsample variant."""

    return build_downsampled_roib_reg_model_bundle(*args, **kwargs)
