from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegConvBlock(nn.Module):
    """Plain 3D UNet block for the public seg backbone."""

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


class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = int(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.upscale_factor
        if r == 1:
            return x
        b, c, d, h, w = x.shape
        divisor = r ** 3
        if c % divisor != 0:
            raise ValueError(f"PixelShuffle3D expects channels divisible by {divisor}, got {c}")
        out_c = c // divisor
        x = x.view(b, out_c, r, r, r, d, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return x.view(b, out_c, d * r, h * r, w * r)


class PixelUnshuffle3D(nn.Module):
    def __init__(self, downscale_factor: int):
        super().__init__()
        self.downscale_factor = int(downscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.downscale_factor
        if r == 1:
            return x
        b, c, d, h, w = x.shape
        if d % r != 0 or h % r != 0 or w % r != 0:
            raise ValueError(f"PixelUnshuffle3D expects spatial dims divisible by {r}, got {(d, h, w)}")
        out_d, out_h, out_w = d // r, h // r, w // r
        x = x.view(b, c, out_d, r, out_h, r, out_w, r)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return x.view(b, c * (r ** 3), out_d, out_h, out_w)


class AtlasPriorEncoder(nn.Module):
    """Build atlas prior tokens from a shared image encoder feature and atlas labels."""

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 16,
        atlas_label_embed_channels: int = 8,
        context_query_tokens: int = 8,
        context_shuffle_factor: int = 2,
        atlas_task_dropout: float = 0.0,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.context_query_tokens = int(context_query_tokens)
        self.context_shuffle_factor = int(context_shuffle_factor)
        self.atlas_label_embed_channels = int(atlas_label_embed_channels)

        c4 = base_channels * 8
        self.out_channels = c4

        self.pixel_shuffle = PixelShuffle3D(self.context_shuffle_factor)
        self.pixel_unshuffle = PixelUnshuffle3D(self.context_shuffle_factor)
        shuffled_channels = c4 // (self.context_shuffle_factor ** 3)
        if shuffled_channels <= 0:
            raise ValueError('context_shuffle_factor is too large for atlas bottleneck channels')
        if c4 % (self.context_shuffle_factor ** 3) != 0:
            raise ValueError(
                f'context_shuffle_factor={self.context_shuffle_factor} is incompatible with channels={c4}'
            )

        self.label_embedding = nn.Embedding(self.num_classes, self.atlas_label_embed_channels)
        self.merge_conv = nn.Sequential(
            nn.Conv3d(shuffled_channels + self.atlas_label_embed_channels, shuffled_channels, kernel_size=1),
            nn.InstanceNorm3d(shuffled_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.context_queries = nn.Parameter(torch.randn(1, self.context_query_tokens, c4) * 0.02)
        self.query_norm = nn.LayerNorm(c4)
        self.kv_norm = nn.LayerNorm(c4)
        self.context_cross_attn = nn.MultiheadAttention(embed_dim=c4, num_heads=attn_heads, dropout=attn_dropout, batch_first=True)
        self.context_self_norm = nn.LayerNorm(c4)
        self.context_self_attn = nn.MultiheadAttention(embed_dim=c4, num_heads=attn_heads, dropout=attn_dropout, batch_first=True)
        self.task_dropout = nn.Dropout(atlas_task_dropout) if atlas_task_dropout > 0 else nn.Identity()

    def build_foreground_tokens(self, feat: torch.Tensor, atlas_label: torch.Tensor) -> torch.Tensor:
        label_down = F.interpolate(atlas_label.float(), size=feat.shape[2:], mode='nearest').long()
        global_token = feat.mean(dim=(2, 3, 4))
        tokens = []
        for class_id in range(1, self.num_classes):
            mask = (label_down == class_id).float()
            denom = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
            pooled = (feat * mask).sum(dim=(2, 3, 4)) / denom
            valid = (mask.sum(dim=(2, 3, 4)) > 0).expand_as(pooled)
            tokens.append(torch.where(valid, pooled, global_token))
        return torch.stack(tokens, dim=1)

    def build_context_tokens(self, feat: torch.Tensor, atlas_label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shuffled_feat = self.pixel_shuffle(feat)
        label_up = F.interpolate(atlas_label.float(), size=shuffled_feat.shape[2:], mode='nearest')
        label_up = label_up[:, 0].long().clamp_(0, self.num_classes - 1)
        label_emb = self.label_embedding(label_up).permute(0, 4, 1, 2, 3).contiguous()
        merged = self.merge_conv(torch.cat([shuffled_feat, label_emb], dim=1))
        merged = self.pixel_unshuffle(merged)
        merged_tokens = merged.flatten(2).transpose(1, 2)

        query = self.context_queries.expand(feat.shape[0], -1, -1)
        context, _ = self.context_cross_attn(self.query_norm(query), self.kv_norm(merged_tokens), self.kv_norm(merged_tokens), need_weights=False)
        self_delta, _ = self.context_self_attn(self.context_self_norm(context), self.context_self_norm(context), self.context_self_norm(context), need_weights=False)
        context = self.task_dropout(self_delta + context)
        return context, merged

    def forward(self, atlas_feat: torch.Tensor, atlas_label: torch.Tensor) -> dict[str, torch.Tensor]:
        foreground_tokens = self.build_foreground_tokens(atlas_feat, atlas_label)
        context_tokens, merged_feat = self.build_context_tokens(atlas_feat, atlas_label)
        atlas_prior_tokens = torch.cat([foreground_tokens, context_tokens], dim=1)
        return {
            'atlas_feature': atlas_feat,
            'atlas_merged_feature': merged_feat,
            'foreground_tokens': foreground_tokens,
            'context_tokens': context_tokens,
            'atlas_prior_tokens': atlas_prior_tokens,
        }


class AtlasImageTaskEncoder(nn.Module):
    """Encode atlas image/label into prior tokens with an independent image encoder."""

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 16,
        atlas_label_embed_channels: int = 8,
        context_query_tokens: int = 8,
        context_shuffle_factor: int = 2,
        atlas_task_dropout: float = 0.0,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.context_query_tokens = int(context_query_tokens)
        self.context_shuffle_factor = int(context_shuffle_factor)
        self.atlas_label_embed_channels = int(atlas_label_embed_channels)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        self.out_channels = c4

        self.enc1 = SegConvBlock(1, c1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = SegConvBlock(c1, c2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = SegConvBlock(c2, c3)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = SegConvBlock(c3, c4)

        self.pixel_shuffle = PixelShuffle3D(self.context_shuffle_factor)
        self.pixel_unshuffle = PixelUnshuffle3D(self.context_shuffle_factor)
        shuffled_channels = c4 // (self.context_shuffle_factor ** 3)
        if shuffled_channels <= 0:
            raise ValueError('context_shuffle_factor is too large for atlas bottleneck channels')
        if c4 % (self.context_shuffle_factor ** 3) != 0:
            raise ValueError(
                f'context_shuffle_factor={self.context_shuffle_factor} is incompatible with channels={c4}'
            )

        self.label_embedding = nn.Embedding(self.num_classes, self.atlas_label_embed_channels)
        self.merge_conv = nn.Sequential(
            nn.Conv3d(shuffled_channels + self.atlas_label_embed_channels, shuffled_channels, kernel_size=1),
            nn.InstanceNorm3d(shuffled_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.context_queries = nn.Parameter(torch.randn(1, self.context_query_tokens, c4) * 0.02)
        self.query_norm = nn.LayerNorm(c4)
        self.kv_norm = nn.LayerNorm(c4)
        self.context_cross_attn = nn.MultiheadAttention(embed_dim=c4, num_heads=attn_heads, dropout=attn_dropout, batch_first=True)
        self.context_self_norm = nn.LayerNorm(c4)
        self.context_self_attn = nn.MultiheadAttention(embed_dim=c4, num_heads=attn_heads, dropout=attn_dropout, batch_first=True)
        self.task_dropout = nn.Dropout(atlas_task_dropout) if atlas_task_dropout > 0 else nn.Identity()

    def encode_image(self, atlas_img: torch.Tensor) -> torch.Tensor:
        x = self.enc1(atlas_img)
        x = self.enc2(self.pool1(x))
        x = self.enc3(self.pool2(x))
        x = self.bottleneck(self.pool3(x))
        return x

    def build_foreground_tokens(self, feat: torch.Tensor, atlas_label: torch.Tensor) -> torch.Tensor:
        label_down = F.interpolate(atlas_label.float(), size=feat.shape[2:], mode='nearest').long()
        global_token = feat.mean(dim=(2, 3, 4))
        tokens = []
        for class_id in range(1, self.num_classes):
            mask = (label_down == class_id).float()
            denom = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
            pooled = (feat * mask).sum(dim=(2, 3, 4)) / denom
            valid = (mask.sum(dim=(2, 3, 4)) > 0).expand_as(pooled)
            tokens.append(torch.where(valid, pooled, global_token))
        return torch.stack(tokens, dim=1)

    def build_context_tokens(self, feat: torch.Tensor, atlas_label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shuffled_feat = self.pixel_shuffle(feat)
        label_up = F.interpolate(atlas_label.float(), size=shuffled_feat.shape[2:], mode='nearest')
        label_up = label_up[:, 0].long().clamp_(0, self.num_classes - 1)
        label_emb = self.label_embedding(label_up).permute(0, 4, 1, 2, 3).contiguous()
        merged = self.merge_conv(torch.cat([shuffled_feat, label_emb], dim=1))
        merged = self.pixel_unshuffle(merged)
        merged_tokens = merged.flatten(2).transpose(1, 2)

        query = self.context_queries.expand(feat.shape[0], -1, -1)
        context, _ = self.context_cross_attn(self.query_norm(query), self.kv_norm(merged_tokens), self.kv_norm(merged_tokens), need_weights=False)
        self_delta, _ = self.context_self_attn(self.context_self_norm(context), self.context_self_norm(context), self.context_self_norm(context), need_weights=False)
        context = self.task_dropout(self_delta + context)
        return context, merged

    def forward(self, atlas_img: torch.Tensor, atlas_label: torch.Tensor) -> dict[str, torch.Tensor]:
        atlas_feat = self.encode_image(atlas_img)
        foreground_tokens = self.build_foreground_tokens(atlas_feat, atlas_label)
        context_tokens, merged_feat = self.build_context_tokens(atlas_feat, atlas_label)
        atlas_prior_tokens = torch.cat([foreground_tokens, context_tokens], dim=1)
        return {
            'atlas_feature': atlas_feat,
            'atlas_merged_feature': merged_feat,
            'foreground_tokens': foreground_tokens,
            'context_tokens': context_tokens,
            'atlas_prior_tokens': atlas_prior_tokens,
        }


class BottleneckAtlasPriorBiCrossAttention(nn.Module):
    """Bidirectional cross-attention fusion with configurable residual blending."""

    def __init__(
        self,
        channels: int,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        residual_weights: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__()
        self.image_norm = nn.LayerNorm(channels)
        self.task_norm = nn.LayerNorm(channels)
        self.image_residual_weight = float(residual_weights[0])
        self.delta_residual_weight = float(residual_weights[1])
        self.image_to_task_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=attn_heads, dropout=attn_dropout, batch_first=True)
        self.task_to_image_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=attn_heads, dropout=attn_dropout, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image_feat: torch.Tensor, atlas_prior_tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_tokens = image_feat.flatten(2).transpose(1, 2)
        img_norm = self.image_norm(img_tokens)
        task_norm = self.task_norm(atlas_prior_tokens)

        img_from_task, _ = self.image_to_task_attn(img_norm, task_norm, task_norm, need_weights=False)
        task_from_img, _ = self.task_to_image_attn(task_norm, img_norm, img_norm, need_weights=False)

        img_from_task = img_from_task.transpose(1, 2).reshape_as(image_feat)
        task_summary = task_from_img.mean(dim=1)
        task_summary = task_summary[:, :, None, None, None].expand_as(image_feat)
        fused_delta = self.fuse(torch.cat([img_from_task, task_summary], dim=1))
        feat_ctx = self.image_residual_weight * image_feat + self.delta_residual_weight * fused_delta

        attn_delta_norm = fused_delta.norm(dim=1).mean()
        reverse_delta_norm = task_from_img.norm(dim=-1).mean()
        return feat_ctx, {
            'attn_gate': image_feat.new_tensor(self.delta_residual_weight),
            'attn_delta_norm': attn_delta_norm,
            'reverse_attn_delta_norm': reverse_delta_norm,
        }


class AtlasGuidedSegUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 16,
        dropout_prob: float = 0.0,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        enable_atlas_token_path: bool = True,
        bottleneck_ctx_residual_weights: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        self.enable_atlas_token_path = bool(enable_atlas_token_path)
        self.dropout = nn.Dropout3d(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.enc1 = SegConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = SegConvBlock(c1, c2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = SegConvBlock(c2, c3)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = SegConvBlock(c3, c4)
        self.atlas_attn = BottleneckAtlasPriorBiCrossAttention(
            channels=c4,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
            residual_weights=bottleneck_ctx_residual_weights,
        ) if self.enable_atlas_token_path else None
        self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = SegConvBlock(c3 + c3, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = SegConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = SegConvBlock(c1 + c1, c1)
        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)

    def encode_backbone(self, x: torch.Tensor, apply_dropout: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bottleneck_raw = self.bottleneck(self.pool3(e3))
        if apply_dropout:
            bottleneck_raw = self.dropout(bottleneck_raw)
        return e1, e2, e3, bottleneck_raw

    def forward_one(self, x: torch.Tensor, atlas_prior_tokens: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        e1, e2, e3, bottleneck_raw = self.encode_backbone(x, apply_dropout=True)
        if self.enable_atlas_token_path and atlas_prior_tokens is not None:
            bottleneck_ctx, attn_info = self.atlas_attn(bottleneck_raw, atlas_prior_tokens)
        else:
            zero = bottleneck_raw.new_tensor(0.0)
            bottleneck_ctx = bottleneck_raw
            attn_info = {'attn_gate': zero, 'attn_delta_norm': zero, 'reverse_attn_delta_norm': zero}
        d3 = self.dec3(torch.cat([self.up3(bottleneck_ctx), e3], dim=1))
        d2 = self.dropout(self.dec2(torch.cat([self.up2(d3), e2], dim=1)))
        d1 = self.dropout(self.dec1(torch.cat([self.up1(d2), e1], dim=1)))
        logits = self.head(d1)
        aux = {
            'features': [e1, e2, e3, bottleneck_ctx, d3, d2, d1],
            'bottleneck_raw': bottleneck_raw,
            'bottleneck_ctx': bottleneck_ctx,
            'attn_gate': attn_info['attn_gate'],
            'attn_delta_norm': attn_info['attn_delta_norm'],
            'reverse_attn_delta_norm': attn_info['reverse_attn_delta_norm'],
            'token_path_enabled': bottleneck_raw.new_tensor(1.0 if self.enable_atlas_token_path else 0.0),
        }
        return logits, aux


class AtlasGuidedSegModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        base_channels: int = 16,
        dropout_prob: float = 0.0,
        atlas_label_embed_channels: int = 8,
        context_query_tokens: int = 8,
        context_shuffle_factor: int = 2,
        atlas_task_dropout: float = 0.0,
        attn_heads: int = 4,
        attn_dropout: float = 0.0,
        attn_gate_init: float = -4.0,
        enable_atlas_token_path: bool = True,
        share_atlas_encoder_with_seg: bool = True,
        bottleneck_ctx_residual_weights: tuple[float, float] = (1.0, 1.0),
    ):
        super().__init__()
        del attn_gate_init
        self.enable_atlas_token_path = bool(enable_atlas_token_path)
        self.share_atlas_encoder_with_seg = bool(share_atlas_encoder_with_seg)
        if self.enable_atlas_token_path:
            atlas_encoder_cls = AtlasPriorEncoder if self.share_atlas_encoder_with_seg else AtlasImageTaskEncoder
            self.atlas_encoder = atlas_encoder_cls(
                num_classes=out_channels,
                base_channels=base_channels,
                atlas_label_embed_channels=atlas_label_embed_channels,
                context_query_tokens=context_query_tokens,
                context_shuffle_factor=context_shuffle_factor,
                atlas_task_dropout=atlas_task_dropout,
                attn_heads=attn_heads,
                attn_dropout=attn_dropout,
            )
        else:
            self.atlas_encoder = None
        self.seg_net = AtlasGuidedSegUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            dropout_prob=dropout_prob,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
            enable_atlas_token_path=enable_atlas_token_path,
            bottleneck_ctx_residual_weights=bottleneck_ctx_residual_weights,
        )

    @staticmethod
    def _expand_tensor_to_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
        if x.shape[0] == batch_size:
            return x
        if x.shape[0] == 1:
            return x.expand(batch_size, *([-1] * (x.ndim - 1)))
        raise ValueError(f'Cannot expand atlas batch {x.shape[0]} to target batch {batch_size}')

    def _resolve_batch_size(self, x_syn: torch.Tensor | None, x_real: torch.Tensor | None) -> int:
        batch_sizes = [x.shape[0] for x in (x_syn, x_real) if x is not None]
        if not batch_sizes:
            raise ValueError('At least one of x_syn or x_real must be provided')
        if len(set(batch_sizes)) != 1:
            raise ValueError(f'x_syn/x_real batch size mismatch: {batch_sizes}')
        return batch_sizes[0]

    def _build_disabled_token_stats(self, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = ref.new_tensor(0.0)
        return {'token_path_enabled': ref.new_tensor(0.0), 'atlas_prior_token_norm': zero}

    def build_atlas_prior_tokens(
        self,
        atlas_img: torch.Tensor,
        atlas_label: torch.Tensor,
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.enable_atlas_token_path:
            raise ValueError('build_atlas_prior_tokens requires enable_atlas_token_path=True')
        if batch_size is None:
            batch_size = int(atlas_img.shape[0])
        atlas_img = self._expand_tensor_to_batch(atlas_img, batch_size)
        atlas_label = self._expand_tensor_to_batch(atlas_label, batch_size)
        if self.share_atlas_encoder_with_seg:
            _, _, _, atlas_feat = self.seg_net.encode_backbone(atlas_img, apply_dropout=False)
            atlas_info = self.atlas_encoder(atlas_feat, atlas_label)
        else:
            atlas_info = self.atlas_encoder(atlas_img, atlas_label)
        atlas_prior_tokens = atlas_info['atlas_prior_tokens']
        atlas_info = dict(atlas_info)
        atlas_info.update({
            'atlas_prior_tokens': atlas_prior_tokens,
            'token_path_enabled': atlas_prior_tokens.new_tensor(1.0),
            'atlas_prior_token_norm': atlas_prior_tokens.norm(dim=-1).mean(),
        })
        return atlas_info

    def forward(
        self,
        x_syn: torch.Tensor | None = None,
        x_real: torch.Tensor | None = None,
        atlas_img: torch.Tensor | None = None,
        atlas_label: torch.Tensor | None = None,
        atlas_prior_tokens_override: torch.Tensor | None = None,
        is_training: bool = True,
        disable_atlas_token_path_infer: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size = self._resolve_batch_size(x_syn, x_real)
        atlas_prior_tokens = None
        out: dict[str, torch.Tensor] = {}
        if atlas_prior_tokens_override is not None:
            atlas_prior_tokens = self._expand_tensor_to_batch(atlas_prior_tokens_override, batch_size)
            out.update({
                'atlas_prior_tokens': atlas_prior_tokens,
                'token_path_enabled': atlas_prior_tokens.new_tensor(1.0),
                'atlas_prior_token_norm': atlas_prior_tokens.norm(dim=-1).mean(),
            })
        elif self.enable_atlas_token_path:
            if atlas_img is None or atlas_label is None:
                raise ValueError('atlas_img and atlas_label are required when enable_atlas_token_path=True')
            atlas_info = self.build_atlas_prior_tokens(atlas_img, atlas_label, batch_size=batch_size)
            atlas_prior_tokens = atlas_info['atlas_prior_tokens']
            out.update({
                'atlas_prior_tokens': atlas_prior_tokens,
                'foreground_tokens': atlas_info['foreground_tokens'],
                'context_tokens': atlas_info['context_tokens'],
                'token_path_enabled': atlas_prior_tokens.new_tensor(1.0),
                'atlas_prior_token_norm': atlas_prior_tokens.norm(dim=-1).mean(),
            })
        else:
            ref = x_syn if x_syn is not None else x_real
            out.update(self._build_disabled_token_stats(ref))

        if not is_training:
            if x_real is None:
                raise ValueError('x_real is required during inference')
            infer_tokens = None if disable_atlas_token_path_infer else atlas_prior_tokens
            real_logits, real_aux = self.seg_net.forward_one(x_real, infer_tokens)
            out.update({
                'real_logits': real_logits,
                'real_features': real_aux['features'],
                'real_ctx': real_aux['bottleneck_ctx'],
                'real_raw': real_aux['bottleneck_raw'],
                'attn_gate': real_aux['attn_gate'],
                'real_attn_gate': real_aux['attn_gate'],
                'real_attn_delta_norm': real_aux['attn_delta_norm'],
                'real_reverse_attn_delta_norm': real_aux['reverse_attn_delta_norm'],
                'disable_atlas_token_path_infer': real_aux['bottleneck_raw'].new_tensor(1.0 if disable_atlas_token_path_infer else 0.0),
            })
            return out

        if x_syn is None:
            raise ValueError('x_syn is required during training')
        syn_logits, syn_aux = self.seg_net.forward_one(x_syn, atlas_prior_tokens)
        out.update({
            'syn_logits': syn_logits,
            'syn_features': syn_aux['features'],
            'syn_ctx': syn_aux['bottleneck_ctx'],
            'syn_raw': syn_aux['bottleneck_raw'],
            'syn_attn_gate': syn_aux['attn_gate'],
            'syn_attn_delta_norm': syn_aux['attn_delta_norm'],
            'syn_reverse_attn_delta_norm': syn_aux['reverse_attn_delta_norm'],
        })
        if x_real is not None:
            real_logits, real_aux = self.seg_net.forward_one(x_real, atlas_prior_tokens)
            out.update({
                'real_logits': real_logits,
                'real_features': real_aux['features'],
                'real_ctx': real_aux['bottleneck_ctx'],
                'real_raw': real_aux['bottleneck_raw'],
                'real_attn_gate': real_aux['attn_gate'],
                'real_attn_delta_norm': real_aux['attn_delta_norm'],
                'real_reverse_attn_delta_norm': real_aux['reverse_attn_delta_norm'],
            })
            out['attn_gate'] = 0.5 * (syn_aux['attn_gate'] + real_aux['attn_gate'])
        else:
            out['attn_gate'] = syn_aux['attn_gate']
        return out
