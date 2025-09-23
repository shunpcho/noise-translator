from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torchvision.transforms.functional as f
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class PairedNoisyDataset(Dataset):
    """Dataset returning (clean, real noisy) pairs.

    The user should prepare paired data or synthetic approximateions.
    For denonstration we'll reusea clean dataset and synthesize 'real' noise by corrupting with complex noise if need.
    """

    def __init__(
        self,
        clean_files: list[Path],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        crop_size: tuple[int, int] | None = None,
        noise_level: float = 0.0,
    ) -> None:
        self.clean_files = clean_files
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.crop_size = crop_size
        self.noise_level = noise_level

    def __len__(self) -> int:
        return len(self.clean_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean_path = self.clean_files[idx]
        if self.noise_level > 0:
            # Add synthetic noise case
            clean_img: Image.Image = Image.open(clean_path).convert("RGB")
            if self.crop_size is not None:
                top: int
                left: int
                height: int
                width: int
                top, left, height, width = transforms.RandomCrop.get_params(clean_img, output_size=self.crop_size)
                clean_img: Image.Image = f.crop(clean_img, top, left, height, width)
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            clean_img = self.transform(clean_img)
            stds = torch.rand(1) * self.noise_level
            noise = torch.randn_like(clean_img) * stds / 255.0
            noisy_img = clean_img + noise
            noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
            return noisy_img, clean_img

        noisy_path = clean_to_noisy(clean_path)
        noisy_img: Image.Image = Image.open(noisy_path).convert("RGB")
        clean_img: Image.Image = Image.open(clean_path).convert("RGB")

        if self.crop_size is not None:
            top: int
            left: int
            height: int
            width: int
            top, left, width, height = transforms.RandomCrop.get_params(clean_img, output_size=self.crop_size)
            clean_img: Image.Image = f.crop(clean_img, top, left, height, width)
            noisy_img: Image.Image = f.crop(noisy_img, top, left, height, width)

        # Generate consistent random crop and transformation
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        clean_img = self.transform(clean_img)
        torch.manual_seed(seed)
        noisy_img = self.transform(noisy_img)

        return noisy_img, clean_img


def clean_to_noisy(clean_path: Path) -> Path:
    return clean_path.with_name(clean_path.name.replace("_mean", "_real"))


def create_dataloader(
    root_data: Path,
    transform: Callable[[Image.Image], torch.Tensor] | None = None,
    crop_size: tuple[int, int] | None = None,
    batch_size: int = 4,
    test_size: float = 0.2,
    noise_level: float = 0.0,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    img_files = sorted(root_data.glob("*_mean.png"))
    train_files, test_files = train_test_split(img_files, test_size=test_size, random_state=42)
    train_dataset = PairedNoisyDataset(train_files, transform=transform, crop_size=crop_size, noise_level=noise_level)
    test_dataset = PairedNoisyDataset(test_files, transform=transform, crop_size=crop_size, noise_level=noise_level)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
