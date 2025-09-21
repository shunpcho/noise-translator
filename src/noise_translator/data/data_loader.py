from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchvision.transforms.functional as f
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

if TYPE_CHECKING:
    from collections.abc import Callable


class PairedNoisyDataset(Dataset):
    """Dataset returning (clean, real noisy) pairs.

    The user should prepare paired data or synthetic approximateions.
    For denonstration we'll reusea clean dataset and synthesize 'real' noise by corrupting with complex noise if need.
    """

    def __init__(
        self,
        root_data: Path,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        crop_size: tuple[int, int] | None = None,
    ) -> None:
        self.noisy_path = root_data
        self.transform = transform
        self.crop_size = crop_size
        self.noisy_files = sorted(self.noisy_path.glob("*_real.png"))

    def __len__(self) -> int:
        return len(self.noisy_files)

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
        noisy_path = self.noisy_files[idx]
        clean_path = _noisy_to_clean(noisy_path)
        noisy_img: Image.Image = Image.open(noisy_path).convert("RGB")
        clean_img: Image.Image = Image.open(clean_path).convert("RGB")
        if self.crop_size is not None:
            i: int
            j: int
            w: int
            h: int
            i, j, h, w = transforms.RandomCrop.get_params(noisy_img, output_size=self.crop_size)
            noisy_img: Image.Image = f.crop(noisy_img, i, j, h, w)
            clean_img: Image.Image = f.crop(clean_img, i, j, h, w)

        if self.transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            noisy_img = self.transform(noisy_img)
            torch.manual_seed(seed)
            clean_img = self.transform(clean_img)
        return noisy_img, clean_img


def _noisy_to_clean(noisy_path: Path) -> Path:
    return noisy_path.with_name(noisy_path.name.replace("_real", "_mean"))
