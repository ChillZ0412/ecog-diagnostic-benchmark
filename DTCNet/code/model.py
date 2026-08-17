"""
DTCNet: Dilated-Transposed Convolution Network
Paper: Wang et al. (2025), Frontiers in Computational Neuroscience
DOI:  10.3389/fncom.2025.1627819

Final architecture (faithful reproduction of the paper):
  FeatureReduction: 1×1 bottleneck + 3×1 conv → 48ch
  Encoder: [48, 64, 96, 128, 128] (dilation 1,2,3,1,2)
  AvgPool
  Normalization: per-channel z-score + median removal (done in preprocess_raw.py)
  Input: (batch, channels, freqs, time)
  Output: per-time-step trajectory (batch, 5, T)  [output_mode='trajectory']
        end-point (batch, 5)        [output_mode='single', ablation]
"""

import torch
import torch.nn as nn


class FeatureReduction(nn.Module):
    """Two-stage reduction: 1×1 bottleneck → 3×1 temporal conv.

    Conv1d(ch×freq → 48, k=1) + Conv1d(48 → 48, k=3, pad=1)
    """
    def __init__(self, in_features, out_ch=48):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_features, out_ch, kernel_size=1)
        self.conv = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        B, C, F, T = x.shape
        x = x.reshape(B, C * F, T)
        x = self.bottleneck(x)
        x = self.conv(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1, pool=2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=padding)
        self.norm = nn.LayerNorm(out_ch)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(pool, pool)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.gelu(x)
        x = self.dropout(x)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, x_ch, skip_ch, out_ch, kernel_size=3, scale=2):
        super().__init__()
        concat_ch = x_ch + skip_ch
        self.conv = nn.Conv1d(concat_ch, x_ch, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(x_ch)
        self.gelu = nn.GELU()
        self.up = nn.ConvTranspose1d(x_ch, out_ch, kernel_size=scale, stride=scale)

    def forward(self, x, skip):
        t = min(x.shape[2], skip.shape[2])
        x = x[:, :, :t]
        skip = skip[:, :, :t]
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.gelu(x)
        return self.up(x)


class DTCNet(nn.Module):
    def __init__(self, n_channels=62, n_freqs=40, dropout=0.1, output_mode='trajectory'):
        super().__init__()
        self.output_mode = output_mode  # 'trajectory' per-time-step | 'single' end-point (ablation)
        in_features = n_channels * n_freqs

        # Module 1: Feature Reduction → 48 ch
        self.feat_reduce = FeatureReduction(in_features, out_ch=48)

        # Module 2: Encoder [48, 64, 96, 128, 128]
        enc_cfg = [
            (48,  64,  7, 1),
            (64,  96,  7, 2),
            (96,  128, 5, 3),
            (128, 128, 5, 1),
            (128, 128, 5, 2),
        ]
        self.encoders = nn.ModuleList([
            EncoderBlock(ci, co, k, d, dropout) for ci, co, k, d in enc_cfg
        ])

        # Module 3: Decoder
        dec_cfg = [
            (128, 128, 96),
            (96,  128, 64),
            (64,  128, 48),
            (48,  96,  48),
            (48,  64,  32),
        ]
        self.decoders = nn.ModuleList([
            DecoderBlock(xc, sc, oc) for xc, sc, oc in dec_cfg
        ])

        # Module 4: Output
        self.output_conv = nn.Conv1d(32, 5, kernel_size=1)

    def forward(self, x):
        x = self.feat_reduce(x)
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)
        skips = skips[::-1]
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[i])
        x = self.output_conv(x)
        if self.output_mode == 'single':
            return x.mean(dim=2)  # (batch, 5) end-point (ablation)
        return x  # (batch, 5, T) per-time-step trajectory (main)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    for ch in [62, 48, 64]:
        m = DTCNet(n_channels=ch, n_freqs=40)
        x = torch.randn(4, ch, 40, 256)
        y = m(x)
        print(f"ch={ch:2d}: params={m.get_param_count():,}  out={y.shape}  ✓")
