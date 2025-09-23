from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, act: nn.Module | None = None
    ) -> None:
        super().__init__()
        if act is None:
            act = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding), nn.BatchNorm2d(out_ch), act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SimpleUNet(nn.Module):
    """A lightweight U-Net for translation tasks."""

    def __init__(self, in_ch: int, out_ch: int, base: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.final = ConvBlock(base, out_ch, padding=0, kernel_size=1)

    @staticmethod
    def _crop_and_concat(upsampled: torch.Tensor, bypass: torch.Tensor) -> torch.Tensor:
        diff_y = bypass.size()[2] - upsampled.size()[2]
        diff_x = bypass.size()[3] - upsampled.size()[3]
        if diff_y != 0 or diff_x != 0:
            bypass = bypass[
                :,
                :,
                diff_y // 2 : bypass.size(2) - (diff_y - diff_y // 2),
                diff_x // 2 : bypass.size(3) - (diff_x - diff_x // 2),
            ]
        return torch.cat([upsampled, bypass], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self._crop_and_concat(d3, e3)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = self._crop_and_concat(d2, e2)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = self._crop_and_concat(d1, e1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out


class DnCNN(nn.Module):
    """A simple DnCNN-like architecture to remove Gaussian noise."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, num_layers: int = 17, features: int = 64) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        layers.extend([nn.Conv2d(in_ch, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)])
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Conv2d(features, features, kernel_size=3, padding=1),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(features, out_ch, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict noise residual r; dnoised = x - r
        r = self.net(x)
        return x - r


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 64) -> torch.Tensor:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, 1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> None:
        return self.model(x)
