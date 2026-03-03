from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class LesionContextGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class MSVEPAFP(nn.Module):
    def __init__(self, out_channels: int = 160, d_model: int = 256, grid_size: int = 14, pretrained: bool = False):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_v2_s(weights=weights)
        self.features = backbone.features
        self.capture_idxs = [2, 3, 5, 7]

        feat_channels = self._infer_channels()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in feat_channels])
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in feat_channels])

        self.lcg = nn.ModuleList([LesionContextGate(out_channels) for _ in feat_channels])
        self.scale_logits = nn.Parameter(torch.zeros(len(feat_channels)))

        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.proj = nn.Linear(out_channels, d_model)

    def _infer_channels(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            feats = []
            for idx, block in enumerate(self.features):
                x = block(x)
                if idx in self.capture_idxs:
                    feats.append(x)
            return [f.shape[1] for f in feats]

    def _build_pyramid(self, feats):
        lat = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(lat) - 2, -1, -1):
            lat[i] = lat[i] + F.interpolate(lat[i + 1], size=lat[i].shape[-2:], mode="nearest")
        out = [s(x) for s, x in zip(self.smooth, lat)]
        return out

    def forward(self, x: torch.Tensor):
        feats = []
        cur = x
        for idx, block in enumerate(self.features):
            cur = block(cur)
            if idx in self.capture_idxs:
                feats.append(cur)

        pyramid = self._build_pyramid(feats)
        weights = torch.softmax(self.scale_logits, dim=0)

        fused = 0.0
        for i, p in enumerate(pyramid):
            gated = self.lcg[i](p)
            pooled = self.pool(gated)
            fused = fused + weights[i] * pooled

        b, c, h, w = fused.shape
        tokens = fused.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)
        return tokens
