from functools import reduce
from sys import path
from PIL.Image import alpha_composite
import torch
from torch._C import set_flush_denormal
import torch.nn as nn
from torch.nn.modules import linear
from torch.nn.modules.activation import LeakyReLU, Tanh
import numpy as np


def _progressive_gan_builder(transition_iters, latent_channels,
                             image_channels, base_channels, growth_ratio,
                             num_lods, init_fn=None):
    prog_gen = ProgressiveGenerator(latent_channels, image_channels,
                                    base_channels, growth_ratio, num_lods)
    prog_disc = ProgressiveDiscriminator(image_channels, base_channels,
                                         growth_ratio, num_lods)
    if init_fn is not None:
        prog_gen.apply(init_fn)
        prog_disc.apply(init_fn)

    model = {'gen': prog_gen, 'disc': prog_disc}
    scheduler = ProgressiveScheduleCallback(model, num_lods, transition_iters)
    return model, scheduler


class ProgressiveScheduleCallback():
    def __init__(self, model, num_lods, transition_iters) -> None:
        self.fadeout_iters, self.stable_iters = transition_iters
        self.model = model
        self.num_lods = num_lods
        self.lod = 1
        self.iteration = 0
        self.callback = self.fadeout_callback
        self.set_model_output_mode('fadeout')
        self.update_alpha()

    def set_model_output_mode(self, mode):
        for model in self.model.values():
            model.change_lod(self.lod, mode)

    def update_alpha(self):
        alpha = np.clip(self.iteration/self.fadeout_iters, 0.0, 1.0)
        for model in self.model.values():
            model.update_alpha(alpha)

    def fadeout_callback(self):
        self.iteration += 1
        self.update_alpha()
        if self.iteration >= self.fadeout_iters:
            self.iteration = 0
            self.set_model_output_mode('rgb')
            self.callback = self.stable_callback

    def stable_callback(self):
        if self.lod == self.num_lods:
            return
        self.iteration += 1
        if self.iteration >= self.stable_iters:
            self.iteration = 0
            self.lod += 1
            self.set_model_output_mode('fadeout')
            self.callback = self.fadeout_callback

    def __call__(self):
        self.callback()


class ProgressiveGenerator(nn.Module):
    def __init__(self, latent_channels, image_channels,
                 base_channels, growth_ratio, num_lods):
        super().__init__()
        out_channels = [base_channels*growth_ratio **
                        lod for lod in range(num_lods)]
        self.to_rgb = nn.ModuleList(
            [nn.Conv2d(num_c, image_channels, 1, 1) for num_c in out_channels])
        self.lods = nn.ModuleList([LatentProjection(
            latent_channels, out_channels[-1], (self.to_rgb[-1],))])
        for i in range(len(out_channels)-2, -1, -1):
            self.lods.append(ProgressiveUpsample(
                out_channels[i+1], out_channels[i],
                (self.to_rgb[i+1], self.to_rgb[i])))
        self.lod = 1

    def update_alpha(self, alpha):
        self.lods[self.lod-1].set_alpha(alpha)

    def change_lod(self, lod, output_mode='fadeout'):
        self.lod = min(lod, len(self.lods))
        for i in range(self.lod-1):
            self.lods[i].output_mode('intermediate')
        self.lods[self.lod-1].output_mode(output_mode)

    def forward(self, x):
        for i in range(self.lod):
            x = self.lods[i](x)
        return x


class ProgressiveDiscriminator(nn.Module):
    def __init__(self, image_channels,
                 base_channels, growth_ratio, num_lods):
        super().__init__()
        out_channels = [base_channels*growth_ratio **
                        lod for lod in range(num_lods)]
        self.from_rgb = nn.ModuleList(
            [nn.Conv2d(image_channels, num_c, 1, 1) for num_c in out_channels])
        self.lods = nn.ModuleList([DiscriminatorCritic(
            out_channels[-1], (self.from_rgb[-1],))])
        for i in range(len(out_channels)-2, -1, -1):
            self.lods.append(ProgressiveDownsample(
                out_channels[i], out_channels[i+1],
                (self.from_rgb[i+1], self.from_rgb[i])))
        self.lod = 1

    def update_alpha(self, alpha):
        self.lods[self.lod-1].set_alpha(alpha)

    def change_lod(self, lod, output_mode='fadeout'):
        self.lod = min(lod, len(self.lods))
        for i in range(self.lod-1):
            self.lods[i].output_mode('intermediate')
        self.lods[self.lod-1].output_mode(output_mode)

    def forward(self, x):
        for i in range(self.lod-1, -1, -1):
            x = self.lods[i](x)
        return x


class PixelwiseNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x/(x.norm(p=2, dim=1, keepdim=True)+self.epsilon)


class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, _, h, w = x.shape
        batch_std = x.std(0, keepdim=True).mean(
            (1, 2, 3), keepdim=True).tile(h, w).repeat(b, 1, 1, 1)
        return torch.cat((x, batch_std), dim=1)


class LatentProjection(nn.Module):
    def __init__(self, latent_channels, latent_proj_channels, to_rgb):
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, latent_proj_channels, 4, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(latent_proj_channels,
                      latent_proj_channels, 3, 1, 1),
            PixelwiseNorm(),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.to_rgb = to_rgb
        self.forward_fn = self.rgb_forward

    def set_alpha(self, alpha):
        pass

    def output_mode(self, mode):
        if mode == 'intermediate':
            self.forward_fn = self.intermediate_forward
        elif mode == 'rgb' or mode == 'fadeout':
            self.forward_fn = self.rgb_forward

    def intermediate_forward(self, x):
        return self.spatial_proj(x)

    def rgb_forward(self, x):
        return self.to_rgb[0](self.spatial_proj(x))

    def forward(self, *args):
        return self.forward_fn(*args)


class DiscriminatorCritic(nn.Module):
    def __init__(self, in_channels, from_rgb):
        super().__init__()
        self.net = nn.Sequential(
            MinibatchStdDev(),
            nn.Conv2d(in_channels+1, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, 4, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, 1, 1, 1)
        )
        self.from_rgb = from_rgb
        self.forward_fn = self.rgb_forward

    def set_alpha(self, alpha):
        pass

    def output_mode(self, mode):
        if mode == 'intermediate':
            self.forward_fn = self.intermediate_forward
        elif mode == 'rgb' or mode == 'fadeout':
            self.forward_fn = self.rgb_forward

    def intermediate_forward(self, x):
        return self.net(x)

    def rgb_forward(self, x):
        return self.net(self.from_rgb[0](x))

    def forward(self, *args):
        return self.forward_fn(*args)


class ProgressiveUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, to_rgb):
        super().__init__()

        self.net = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                 PixelwiseNorm(),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                 PixelwiseNorm(),
                                 nn.LeakyReLU(0.2, inplace=True))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.to_rgb = to_rgb
        self.forward_fn = self.lerp_forward
        self.alpha = 0.0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def lerp_forward(self, x):
        x_up = self.upsample(x)
        x_conv = self.to_rgb[1](self.net(x_up))
        x_up = self.to_rgb[0](x_up)
        return (1-self.alpha)*x_up + self.alpha*x_conv

    def rgb_forward(self, x):
        return self.to_rgb[1](self.net(self.upsample(x)))

    def output_mode(self, mode):
        if mode == 'intermediate':
            self.forward_fn = self.intermediate_forward
        elif mode == 'rgb':
            self.forward_fn = self.rgb_forward
        elif mode == 'fadeout':
            self.forward_fn = self.lerp_forward

    def intermediate_forward(self, x):
        return self.net(self.upsample(x))

    def forward(self, *args):
        return self.forward_fn(*args)


class ProgressiveDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, from_rgb):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                 nn.LeakyReLU(0.2, inplace=True))
        self.downsample = nn.AvgPool2d(2, 2)
        self.from_rgb = from_rgb
        self.forward_fn = self.lerp_forward
        self.alpha = 0.0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def lerp_forward(self, x):
        x_down = self.from_rgb[0](self.downsample(x))
        x_conv = self.downsample(self.net(self.from_rgb[1](x)))
        return (1-self.alpha)*x_down+self.alpha*x_conv

    def rgb_forward(self, x):
        return self.downsample(self.net(self.from_rgb[1](x)))

    def output_mode(self, mode):
        if mode == 'intermediate':
            self.forward_fn = self.intermediate_forward
        elif mode == 'rgb':
            self.forward_fn = self.rgb_forward
        elif mode == 'fadeout':
            self.forward_fn = self.lerp_forward

    def intermediate_forward(self, x):
        return self.downsample(self.net(x))

    def forward(self, *args):
        return self.forward_fn(*args)
