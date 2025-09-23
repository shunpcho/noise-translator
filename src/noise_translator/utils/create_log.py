import logging
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate PSNR between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Simplified SSIM calculation."""
    # This is a simplified version. For accurate SSIM, use torchmetrics or skimage
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.var(img1)
    sigma2_sq = torch.var(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))

    c1 = 0.01**2
    c2 = 0.03**2

    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim.item()


def get_logger(out_dir: Path, name: str = "train"):
    log_file = out_dir / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # file
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)

    # tensorboard
    writer = SummaryWriter(log_dir=str(out_dir / "logs"))

    return logger, writer
