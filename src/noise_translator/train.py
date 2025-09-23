from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

from noise_translator.data.data_loader import create_dataloader
from noise_translator.models.models import DnCNN, PatchDiscriminator, SimpleUNet
from noise_translator.models.NAFNet_arch import NAFNet
from noise_translator.utils.create_log import calculate_psnr, calculate_ssim, get_logger
from noise_translator.utils.utils import save_sample_grid, weights_init


def pretrain_denoiser(
    denoiser: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    iteration: int = 10000,
    lr: float = 1e-3,
    out_dir: Path = Path("results"),
    eval_iter: int = 1000,
    writer: SummaryWriter | None = None,
    logger: logging.Logger | None = None,
) -> nn.Module:
    optim = torch.optim.Adam(denoiser.parameters(), lr=lr)
    criterion = nn.MSELoss()
    denoiser.train()

    model_out_dir = out_dir / "models" / "denoiser"
    model_out_dir.mkdir(exist_ok=True, parents=True)

    step = 0
    pbar = tqdm(total=iteration, desc="Denoiser Pretrain")

    while step < iteration:
        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            out_img = denoiser(noisy)
            loss = criterion(out_img, clean)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if writer:
                train_ssim = calculate_ssim(out_img, clean)
                train_psnr = calculate_psnr(out_img, clean)
                writer.add_scalar("Denoiser/train_loss", loss.item(), step)
                writer.add_scalar("Denoiser/train_ssim", train_ssim, step)
                writer.add_scalar("Denoiser/train_psnr", train_psnr, step)
            if logger and step % 100 == 0:
                logger.info(f"[Denoiser][Step {step}] Train Loss={loss.item():.6f}")

            step += 1
            pbar.update(1)
            pbar.set_postfix(train_loss=loss.item())

            # checkpoint & validation
            if step % eval_iter == 0 or step == iteration:
                denoiser.eval()
                val_loss, val_ssim, val_psnr = 0.0, 0.0, 0.0

                pbar_eval = tqdm(total=len(test_loader), desc="Denoiser Eval")
                with torch.no_grad():
                    for noisy_val, clean_val in test_loader:
                        noisy_val = noisy_val.to(device)
                        clean_val = clean_val.to(device)
                        out_val = denoiser(noisy_val)
                        val_loss += criterion(out_val, clean_val).item()
                        val_ssim += calculate_ssim(out_val, clean_val)
                        val_psnr += calculate_psnr(out_val, clean_val)

                        pbar_eval.update(1)
                        pbar_eval.set_postfix(val_loss=val_loss)

                val_loss /= len(test_loader)
                val_ssim /= len(test_loader)
                val_psnr /= len(test_loader)

                pbar_eval.close()

                if writer:
                    writer.add_scalar("Denoiser/val_loss", val_loss, step)
                    writer.add_scalar("Denoiser/val_ssim", val_ssim, step)
                    writer.add_scalar("Denoiser/val_psnr", val_psnr, step)

                    # sample image
                    sample_noisy, sample_clean = noisy[:4], clean[:4]
                    sample_out = denoiser(sample_noisy)
                    grid = make_grid(torch.cat([sample_clean, sample_noisy, sample_out], dim=0), nrow=4)
                    writer.add_image("Denoiser/samples", grid, step)

                if logger:
                    logger.info(
                        f"[Denoiser][Step {step}] Val Loss={val_loss:.6f}, SSIM={val_ssim:.4f}, PSNR={val_psnr:.2f}"
                    )

                # save model
                torch.save(
                    {
                        "step": step,
                        "model": denoiser.state_dict(),
                        "optimizer": optim.state_dict(),
                        "val_loss": val_loss,
                    },
                    model_out_dir / f"denoiser_step{step}.pth",
                )

                # log tensorboard
                if writer:
                    sample_noisy = noisy[:4]
                    sample_clean = clean[:4]
                    sample_out = denoiser(sample_noisy)
                    grid = make_grid(torch.cat([sample_clean, sample_noisy, sample_out], dim=0), nrow=4)
                    writer.add_image("Denoiser/samples", grid, step)

                denoiser.train()

            if step >= iteration:
                break

    pbar.close()
    return denoiser


def train_translator(
    translator: nn.Module,
    denoiser: nn.Module,
    dsc: nn.Module | None,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    iteration: int = 50,
    lr: float = 1e-4,
    lambda_recon: float = 1.0,
    lambda_gan: float = 0.1,
    out_dir: Path = Path("results"),
    eval_iter: int = 10,
    writer: SummaryWriter | None = None,
    logger: logging.Logger | None = None,
) -> nn.Module:
    optim_t = torch.optim.Adam(translator.parameters(), lr=lr)
    optim_d = torch.optim.Adam(dsc.parameters(), lr=lr) if dsc is not None else None
    l1 = nn.L1Loss()
    adversarial_loss = nn.BCEWithLogitsLoss()

    translator.train()
    denoiser.eval()
    if dsc is not None:
        dsc.train()

    model_out_dir = out_dir / "models" / "translator"
    model_out_dir.mkdir(exist_ok=True, parents=True)
    fig_out_dir = out_dir / "figs"
    fig_out_dir.mkdir(exist_ok=True, parents=True)

    step = 0
    pbar = tqdm(total=iteration, desc="Translator Train")

    while step < iteration:
        for real_noisy, clean in train_loader:
            real_noisy = real_noisy.to(device)
            clean = clean.to(device)

            translated = torch.sigmoid(translator(real_noisy))
            denoised = denoiser(translated)
            recon_loss = l1(denoised, clean)

            if dsc is not None:
                # train discriminator
                sigma = 0.05
                synthetic_noisy = (clean + torch.randn_like(clean) * sigma).clamp(0, 1)

                optim_d.zero_grad()
                pred_real = dsc(synthetic_noisy)
                loss_real = adversarial_loss(pred_real, torch.ones_like(pred_real))
                pred_fake = dsc(translated.detach())
                loss_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
                d_loss = 0.5 * (loss_real + loss_fake)
                d_loss.backward()
                optim_d.step()

                # train translator
                optim_t.zero_grad()
                pred_fake_for_trans = dsc(translated)
                gan_loss_trans = adversarial_loss(pred_fake_for_trans, torch.ones_like(pred_fake_for_trans))
                total_loss = lambda_recon * recon_loss + lambda_gan * gan_loss_trans
                total_loss.backward()
                optim_t.step()

                pbar.set_postfix(
                    recon=float(recon_loss.detach()),
                    gan=float(gan_loss_trans.detach()),
                    d=float(d_loss.detach()),
                )

            else:
                optim_t.zero_grad()
                total_loss = lambda_recon * recon_loss
                total_loss.backward()
                optim_t.step()
                pbar.set_postfix(recon=float(recon_loss.detach()))

                # log to tensorboard
                if writer:
                    train_ssim = calculate_ssim(denoised, clean)
                    train_psnr = calculate_psnr(denoised, clean)
                    writer.add_scalar("Translator/train_loss", total_loss.item(), step)
                    writer.add_scalar("Translator/train_ssim", train_ssim, step)
                    writer.add_scalar("Translator/train_psnr", train_psnr, step)
                    if dsc:
                        writer.add_scalar("Translator/gan_loss", gan_loss_trans.item(), step)
                        writer.add_scalar("Translator/d_loss", d_loss.item(), step)
            if logger and step % 10 == 0:
                logger.info(f"[Translator][Step {step}] Loss={total_loss.item():.6f}")

            step += 1
            pbar.update(1)

            # checkpoint & validation
            if step % eval_iter == 0 or step == iteration:
                translator.eval()
                val_loss, val_ssim, val_psnr = 0.0, 0.0, 0.0
                pabar_eval = tqdm(total=len(test_loader), desc="Translator Eval")
                with torch.no_grad():
                    for real_noisy_val, clean_val in test_loader:
                        real_noisy_val, clean_val = real_noisy_val.to(device), clean_val.to(device)
                        translated_val = torch.sigmoid(translator(real_noisy_val))
                        denoised_val = denoiser(translated_val)
                        val_loss += l1(denoised_val, clean_val).item()
                        val_ssim += calculate_ssim(denoised_val, clean_val)
                        val_psnr += calculate_psnr(denoised_val, clean_val)
                val_loss /= len(test_loader)
                val_ssim /= len(test_loader)
                val_psnr /= len(test_loader)

                pabar_eval.close()

                if writer:
                    writer.add_scalar("Translator/val_loss", val_loss, step)
                    writer.add_scalar("Translator/val_ssim", val_ssim, step)
                    writer.add_scalar("Translator/val_psnr", val_psnr, step)

                    # sample images
                    sample_real_noisy = real_noisy[:4]
                    sample_clean = clean[:4]
                    sample_translated = torch.sigmoid(translator(sample_real_noisy))
                    sample_denoised = denoiser(sample_translated)
                    save_sample_grid(
                        sample_clean,
                        sample_real_noisy,
                        sample_translated,
                        sample_denoised,
                        fig_out_dir / f"sample_step{step}.png",
                        n=4,
                    )
                    grid = make_grid(
                        torch.cat([sample_clean, sample_real_noisy, sample_translated, sample_denoised], dim=0), nrow=4
                    )
                    writer.add_image("Translator/samples", grid, step)

                if logger:
                    logger.info(
                        f"[Translator][Step {step}] Val Loss={val_loss:.6f}, SSIM={val_ssim:.4f}, PSNR={val_psnr:.2f}"
                    )

                # save model checkpoint
                torch.save(
                    {
                        "step": step,
                        "translator": translator.state_dict(),
                        "optimizer_t": optim_t.state_dict(),
                        "optimizer_d": optim_d.state_dict() if optim_d else None,
                        "val_loss": val_loss,
                    },
                    model_out_dir / f"translator_step{step}.pth",
                )

                # sample visualization
                with torch.no_grad():
                    sample_real_noisy = real_noisy[:4]
                    sample_clean = clean[:4]
                    sample_translated = torch.sigmoid(translator(sample_real_noisy))
                    sample_denoised = denoiser(sample_translated)
                    save_sample_grid(
                        sample_clean,
                        sample_real_noisy,
                        sample_translated,
                        sample_denoised,
                        fig_out_dir / f"sample_step{step}.png",
                        n=4,
                    )

                # log to tensorboard
                if writer:
                    grid = make_grid(
                        torch.cat([sample_clean, sample_real_noisy, sample_translated, sample_denoised], dim=0), nrow=4
                    )
                    writer.add_image("Samples/clean_noisy_translated_denoised", grid, step)

                translator.train()
                if dsc is not None:
                    dsc.train()

            if step >= iteration:
                break

    pbar.close()
    return translator


def set_translator(model: str) -> nn.Module:
    if model == "unet":
        translator = SimpleUNet(in_ch=3, out_ch=3)
    elif model == "nafnet":
        img_channel = 3
        width = 32
        enc_blks = [2, 2, 4, 8]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2]
        translator = NAFNet(
            img_channel=img_channel,
            width=width,
            middle_blk_num=middle_blk_num,
            enc_blk_nums=enc_blks,
            dec_blk_nums=dec_blks,
        )
    else:
        msg = f"Model {model} is not implemented."
        raise NotImplementedError(msg)
    return translator


def run(
    data_dir: Path = Path("data"),
    device_str: str = "cuda",
    crop_size: tuple[int, int] | None = None,
    iter_denoiser: int = 50,
    iter_translator: int = 50,
    eval_iter_denoiser: int = 10,
    eval_iter_translator: int = 10,
    use_discriminator: bool = True,
) -> None:
    """Main training pipeline."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # Initialize models
    denoiser = DnCNN(in_ch=3).to(device)
    translator = set_translator("nafnet").to(device)
    discriminator = PatchDiscriminator(in_ch=3).to(device) if use_discriminator else None

    # Initialize weights
    denoiser.apply(weights_init)
    translator.apply(weights_init)
    if discriminator is not None:
        discriminator.apply(weights_init)

    # Create output directory
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True, parents=True)
    logger, writer = get_logger(out_dir)

    logger.info(f"Using device: {device}")

    # Pretrain denoiser
    print("Pretraining denoiser...")
    train_denoiser_loader, test_denoiser_loader = create_dataloader(
        data_dir, batch_size=4, transform=transform, crop_size=crop_size, test_size=0.2
    )
    denoiser = pretrain_denoiser(
        denoiser,
        train_denoiser_loader,
        test_denoiser_loader,
        device,
        iteration=iter_denoiser,
        eval_iter=eval_iter_denoiser,
        lr=1e-3,
        out_dir=out_dir,
        writer=writer,
        # logger=logger,
    )

    # Train translator
    print("Training translator...")
    train_translator_loader, test_translator_loader = create_dataloader(
        data_dir, batch_size=4, transform=transform, crop_size=crop_size, test_size=0.2
    )
    translator = train_translator(
        translator,
        denoiser,
        discriminator,
        train_translator_loader,
        test_translator_loader,
        device,
        iteration=iter_translator,
        eval_iter=eval_iter_translator,
        lr=1e-4,
        lambda_recon=1.0,
        lambda_gan=0.1,
        out_dir=out_dir,
        writer=writer,
        # logger=logger,
    )

    print("Dnoe. Check", out_dir)


if __name__ == "__main__":
    run(
        data_dir=Path("data/CC15"),
        device_str="cpu",
        crop_size=(64, 64),
        iter_denoiser=100,
        iter_translator=200,
        eval_iter_denoiser=10,
        eval_iter_translator=10,
    )
