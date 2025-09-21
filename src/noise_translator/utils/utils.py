from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image  # type: ignore[import]


def weights_init(m: nn.Conv2d | nn.Linear) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


def save_sample_grid(
    clean: torch.Tensor,
    real_noisy: torch.Tensor,
    translated: torch.Tensor,
    denoised: torch.Tensor,
    save_path: Path,
    n: int = 4,
) -> None:
    """Save sample grid with clumns: clean | real noisy | trainslated | denoised."""
    batch_size = clean.size(0)
    to_save = []
    for idx in range(min(n, batch_size)):
        to_save.extend([clean[idx], real_noisy[idx], translated[idx], denoised[idx]])
    grid = torch.stack(to_save, dim=0)  # torch.cat → torch.stack
    save_image(grid, save_path, nrow=4)
