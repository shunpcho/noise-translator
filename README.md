# noise-translator

## Minimal PyTorch implementation of the "Learning to Translate Noise for Robust Image Denoising" idea.

This script provides:

- A simple DnCNN-like denoiser (pretrain on synthetic Gaussian noise)
- A UNet-like NoiseTranslator that maps _real_ noisy images -> images with noise distribution the denoiser expects
- A PatchGAN discriminator (optional adversarial training)
- Training loops for:
  1. Pretrain denoiser on clean + Gaussian noise
  2. Train translator with denoiser fixed (and optional discriminator) so that denoiser(T(y_real)) ~= x_clean

Usage:

- Prepare dataset returning (clean, real_noisy) pairs for translator training.
- For denoiser pretraining you can use clean images alone and add synthetic Gaussian noise on the fly.

Notes:

- This is a pedagogical minimal example, not a production-ready training regime.
- Improve: stronger architectures, data augmentation, perceptual losses, cycle losses, better GAN stabilization.

Requirements:
torch, torchvision, tqdm
