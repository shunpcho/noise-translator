# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

"""Simple Baselines for Image Restoration.

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
"""

import math

import torch
import torch.nn.functional as f
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
    ) -> torch.Tensor:
        ctx.eps = eps
        _, channel, _, _ = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, channel, 1, 1) * y + bias.view(1, channel, 1, 1)
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        eps = ctx.eps

        _, channel, _, _ = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, channel, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


def exists(x: object) -> bool:
    return x is not None


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(
        self, channel: int, net_bias: bool = True, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0
    ) -> None:
        super().__init__()
        dw_channel = channel * dw_expand
        self.norm1 = LayerNorm(channel)
        # self.norm1 = LayerNorm2d(channel)
        self.conv1 = nn.Conv2d(
            in_channels=channel,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=net_bias,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=net_bias,
        )
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=net_bias,
            ),
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=net_bias,
        )
        self.beta = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)
        self.norm2 = LayerNorm(channel)
        # self.norm2 = LayerNorm2d(channel)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * channel
        self.conv4 = nn.Conv2d(
            in_channels=channel,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=net_bias,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=net_bias,
        )
        self.gamma = nn.Parameter(torch.zeros((1, channel, 1, 1)), requires_grad=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x *= self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        # print(x.grad)

        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 12,
        enc_blk_nums: list[int] = [2, 2, 4, 8],
        dec_blk_nums: list[int] = [2, 2, 2, 2],
        net_bias: bool = True,
    ) -> None:
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=net_bias,
        )
        self.ending = nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=net_bias,
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.mask = {}

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, net_bias) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2, bias=net_bias))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan, net_bias) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, net_bias) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, channel, h, w = inp.shape
        inp = self.check_image_size(inp)

        # for gray
        if channel == 1:
            inp = inp.repeat(1, 3, 1, 1)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs, strict=False):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1], strict=False):
            x = up(x)
            x += enc_skip
            x = decoder(x)

        x = self.ending(x)
        x += inp
        # x = inp - x

        # for gray
        if channel == 1:
            x = x.mean(dim=1).unsqueeze(dim=1)

        return torch.clamp(x[:, :, :h, :w], 0, 1)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        # mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        # mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # x = f.pad(x, (0, mod_pad_w, 0, mod_pad_h))

        h_padded, w_padded = (
            ((h + self.padder_size) // self.padder_size) * self.padder_size,
            ((w + self.padder_size) // self.padder_size) * self.padder_size,
        )
        mod_pad_h = h_padded - h if h % self.padder_size != 0 else 0
        mod_pad_w = w_padded - w if w % self.padder_size != 0 else 0
        x = f.pad(x, (0, mod_pad_w, 0, mod_pad_h), padding_mode="reflect")
        return x


if __name__ == "__main__":
    img_channel = 3
    # width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    width = 32
    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    # width = 32
    # enc_blks = [2, 2, 2, 2]
    # middle_blk_num = 2
    # dec_blks = [2, 2, 2, 2]

    # width = 32
    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    # width = 64
    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]

    net = NAFNet(
        img_channel=img_channel,
        width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blks,
        dec_blk_nums=dec_blks,
    )

    with torch.no_grad():
        inp_shape = (3, 256, 256)

        from ptflops import get_model_complexity_info  # pyright: ignore[reportMissingImports]
        from torchsummary import summary as summary_  # pyright: ignore[reportMissingImports]

        summary_(net.cuda(), (3, 256, 256), batch_size=1)

        macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)

        params = float(params[:-3])
        macs = float(macs[:-4])

        print(macs, params)
