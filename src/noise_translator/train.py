from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from noise_translator.data.data_loader import PairedNoisyDataset
from noise_translator.models.models import DnCNN, PatchDiscriminator, SimpleUNet
from noise_translator.utils.utils import save_sample_grid, weights_init


def pretrain_denoiser(
    denoiser: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    out_dir: Path = Path("results"),
) -> nn.Module:
    optim = torch.optim.Adam(denoiser.parameters(), lr=lr)
    criterion = nn.MSELoss()
    denoiser.train()

    model_out_dir = out_dir / "models" / "denoiser"
    model_out_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"Denoiser Pretrain {epoch:3}/{epochs}")
        running = 0.0
        for noisy, clean in pbar:
            noisy = clean.to(device)
            clean = clean.to(device)

            out_img = denoiser(noisy)
            loss = criterion(out_img, clean)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
            pbar.set_postfix(loss=running / (pbar.n + 1))

        torch.save(denoiser.state_dict(), model_out_dir / f"denoiser_epoch{epoch}.pth")
    return denoiser


def train_translator(
    translator: nn.Module,
    denoiser: nn.Module,
    dsc: nn.Module | None,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-4,
    lambda_recon: float = 1.0,
    lambda_gan: float = 0.1,
    out_dir: Path = Path("results"),
) -> nn.Module:
    optim_t = torch.optim.Adam(translator.parameters(), lr=lr)
    optim_d = torch.optim.Adam(dsc.parameters(), lr=lr) if dsc is not None else None
    l1 = nn.L1Loss()
    adversarial_loss = nn.BCEWithLogitsLoss()

    translator.train()
    denoiser.eval()
    if dsc is not None:
        dsc.train()

    model_out_dir = out_dir / "models" / "trainslator"
    model_out_dir.mkdir(exist_ok=True, parents=True)
    fig_out_dir = out_dir / "figs"
    fig_out_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"Translator Train {epoch:3}/{epochs}")
        for clean, real_noisy in pbar:
            clean = clean.to(device)
            real_noisy = real_noisy.to(device)

            translated = torch.sigmoid(translator(real_noisy))
            denoised = denoiser(translated)
            recon_loss = l1(denoised, clean)

            # GAN loss: discriminator shoud think translated ~ synthetic_noisy
            if dsc is not None:
                sigma = 0.05
                synthetic_noisy = (clean + torch.randn_like(clean) * sigma).clamp(0, 1)

                optim_d.zero_grad()
                # label -> real, fake = 1, 0
                pred_real = dsc(synthetic_noisy)
                loss_real = adversarial_loss(pred_real, torch.ones_like(pred_real))
                pred_fake = dsc(translated.detach())
                loss_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
                d_loss = 0.5 * (loss_real + loss_fake)
                d_loss.backward()
                optim_d.step()

                optim_t.zero_grad()
                pred_fake_for_trans = dsc(translated)
                gan_loss_trans = adversarial_loss(pred_fake_for_trans, torch.ones_like(pred_fake_for_trans))
                total_loss = lambda_recon * recon_loss + lambda_gan * gan_loss_trans
                total_loss.backward()
                optim_t.step()

                pbar.set_postfix(
                    recon=float(recon_loss.detach()), gan=float(gan_loss_trans.detach()), d=float(d_loss.detach())
                )

            else:
                optim_t.zero_grad()
                total_loss = lambda_recon * recon_loss
                total_loss.backward()
                optim_t.step()
                pbar.set_postfix(recon=float(recon_loss.detach()))

        torch.save(translator.state_dict(), model_out_dir / f"translator_epoch{epoch}.pth")

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
                fig_out_dir / f"sample_epoch{epoch}.png",
                n=4,
            )

    return translator


def run(
    data_dir: Path = Path("data"), device_str: str = "cuda", epochs_denoiser: int = 10, epochs_translator: int = 10
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = PairedNoisyDataset(data_dir, transform=transform, crop_size=(64, 64))
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    denoiser = DnCNN(in_ch=3).to(device)
    translator = SimpleUNet(in_ch=3, out_ch=3).to(device)
    discriminator = PatchDiscriminator(in_ch=3).to(device)

    denoiser.apply(weights_init)
    translator.apply(weights_init)
    discriminator.apply(weights_init)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Pretraining denoiser:")
    pretrain_loder = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=4)
    denoiser = pretrain_denoiser(denoiser, pretrain_loder, device, epochs=epochs_denoiser, lr=1e-3, out_dir=out_dir)

    print("Training translator:")
    translator = train_translator(
        translator,
        denoiser,
        discriminator,
        loader,
        device,
        epochs=epochs_translator,
        lr=1e-4,
        lambda_recon=1.0,
        lambda_gan=0.1,
        out_dir=out_dir,
    )

    print("Dnoe. Check", out_dir)


if __name__ == "__main__":
    run(data_dir=Path("data/CC15"), device_str="cpu", epochs_denoiser=5, epochs_translator=5)
