import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )

    def forward(self, x):
        return F.silu(x + self.block(x))


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            ResidualBlock(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            ResidualBlock(out_channels),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class TinyImageEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, feat_dim=128):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            ResidualBlock(c),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.SiLU(),
            ResidualBlock(c * 2),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c * 4),
            nn.SiLU(),
            ResidualBlock(c * 4),
        )
        self.head = nn.Linear(c * 4, feat_dim)

    def forward(self, x):
        feat_map = self.stem(x)
        pooled = feat_map.mean(dim=(2, 3))
        feat = self.head(pooled)
        return feat_map, feat


class TinyResidualBridge(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, n_components=8):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            ResidualBlock(c),
        )
        self.down1 = DownsampleBlock(c, c * 2)
        self.down2 = DownsampleBlock(c * 2, c * 4)
        self.bottleneck = nn.Sequential(
            ResidualBlock(c * 4),
            ResidualBlock(c * 4),
        )
        self.up1 = UpsampleBlock(c * 4, c * 2, c * 2)
        self.up2 = UpsampleBlock(c * 2, c, c)
        self.out = nn.Sequential(
            ResidualBlock(c),
            nn.Conv2d(c, n_components * in_channels, kernel_size=3, padding=1),
        )
        self.n_components = n_components
        self.in_channels = in_channels

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        h = self.bottleneck(s2)
        h = self.up1(h, s1)
        h = self.up2(h, s0)
        out = self.out(h)
        b, _, h_out, w_out = out.shape
        return out.view(b, self.n_components, self.in_channels, h_out, w_out)


class ImageSpaceLightSB(nn.Module):
    """
    A LightSB-inspired image-space bridge:
    - learned component prototypes in image feature space
    - component-conditioned residual image means
    - LightSB-style objective using log_potential(y) and log_C(x)
    """

    def __init__(
        self,
        image_channels=3,
        n_components=8,
        feature_dim=128,
        base_channels=32,
        epsilon=0.5,
        noise_std=0.05,
        residual_scale=0.35,
    ):
        super().__init__()
        self.n_components = n_components
        self.image_channels = image_channels
        self.epsilon = epsilon
        self.noise_std = noise_std
        self.residual_scale = residual_scale

        self.feature_encoder = TinyImageEncoder(
            in_channels=image_channels,
            base_channels=base_channels,
            feat_dim=feature_dim,
        )
        self.bridge = TinyResidualBridge(
            in_channels=image_channels,
            base_channels=base_channels,
            n_components=n_components,
        )

        self.component_prototypes = nn.Parameter(
            torch.randn(n_components, feature_dim) / math.sqrt(feature_dim)
        )
        self.log_alpha_raw = nn.Parameter(torch.zeros(n_components))
        self.input_gate = nn.Linear(feature_dim, n_components)

    def encode_features(self, x):
        _, feat = self.feature_encoder(x)
        feat = F.normalize(feat, dim=-1)
        return feat

    def component_scores(self, x):
        feat = self.encode_features(x)
        proto = F.normalize(self.component_prototypes, dim=-1)
        return feat @ proto.t() + self.log_alpha_raw[None, :]

    def get_component_params(self, x):
        residuals = torch.tanh(self.bridge(x)) * self.residual_scale
        means = torch.clamp(x[:, None, :, :, :] + residuals, -1.0, 1.0)
        feat = self.encode_features(x)
        gate_logits = self.input_gate(feat) + self.log_alpha_raw[None, :]
        return gate_logits, means

    def get_log_potential(self, y):
        scores = self.component_scores(y) / self.epsilon
        return torch.logsumexp(scores, dim=1)

    def get_log_C(self, x):
        gate_logits, means = self.get_component_params(x)
        b, k, c, h, w = means.shape
        flat_means = means.view(b * k, c, h, w)
        mean_scores = self.component_scores(flat_means).view(b, k, k)
        aligned_scores = torch.diagonal(mean_scores, dim1=1, dim2=2)
        return torch.logsumexp((gate_logits + aligned_scores) / self.epsilon, dim=1)

    def transport_regularization(self, x):
        _, means = self.get_component_params(x)
        residuals = means - x[:, None, :, :, :]
        l2 = residuals.pow(2).mean()
        tv_h = (residuals[:, :, :, 1:, :] - residuals[:, :, :, :-1, :]).abs().mean()
        tv_w = (residuals[:, :, :, :, 1:] - residuals[:, :, :, :, :-1]).abs().mean()
        return l2 + 0.25 * (tv_h + tv_w)

    def training_loss(self, x, y):
        return (-self.get_log_potential(y) + self.get_log_C(x)).mean()

    @torch.no_grad()
    def forward(self, x, deterministic=False):
        gate_logits, means = self.get_component_params(x)
        probs = torch.softmax(gate_logits / max(self.epsilon, 1e-6), dim=1)
        if deterministic:
            inds = probs.argmax(dim=1)
        else:
            inds = torch.multinomial(probs, num_samples=1).squeeze(1)
        chosen = means[torch.arange(x.shape[0], device=x.device), inds]
        if deterministic:
            return chosen
        noisy = chosen + self.noise_std * torch.randn_like(chosen)
        return torch.clamp(noisy, -1.0, 1.0)
