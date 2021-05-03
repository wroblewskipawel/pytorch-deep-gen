import numpy as np
import torch
import torch.nn as nn
from . import dcgan


def vanilla_psgan(latent_channels, image_channels, hidden_layers, **kwargs):
    return _psgan_builder(dcgan.StridedConvUpsample,
                          dcgan.StridedConvDownsample,
                          latent_channels, image_channels,
                          hidden_layers, **kwargs)


def _psgan_builder(gen_build_fn, disc_build_fn, latent_channels,
                   image_channels, hidden_layers, base_channels=64,
                   growth_ratio=2, periodic_hidden=60, init_fn=None,
                   gen_output_act=(nn.Tanh, {})):
    generator_arch = [
        {'in_channels': sum(latent_channels),
         'out_channels': base_channels*growth_ratio**(hidden_layers),
         'use_bnorm': True}]
    generator_arch.extend([{
        'in_channels': base_channels*growth_ratio**layer,
        'out_channels': base_channels*growth_ratio**(layer-1),
        'use_bnorm': True
    } for layer in range(hidden_layers, 0, -1)])
    generator_arch.append({
        'in_channels': base_channels,
        'out_channels': image_channels,
        'activation': gen_output_act,
        'use_bnorm': False,
    })

    discriminator_arch = [{
        'in_channels': image_channels,
        'out_channels': base_channels,
        'use_bnorm': False}]
    discriminator_arch.extend([{
        'in_channels': base_channels*growth_ratio**layer,
        'out_channels': base_channels*growth_ratio**(layer+1),
        'use_bnorm': True
    } for layer in range(hidden_layers)])
    discriminator_arch.append({
        'in_channels': base_channels*growth_ratio**(hidden_layers),
        'out_channels': 1,
        'activation': None,
        'use_bnorm': False
    })

    gen = nn.Sequential(
        PeriodicSampler(latent_channels, periodic_hidden),
        *[gen_build_fn(**layer_kwargs)
          for layer_kwargs in generator_arch])

    disc = nn.Sequential(
        *[disc_build_fn(**layer_kwargs)
          for layer_kwargs in discriminator_arch])

    if init_fn is not None:
        gen[1].apply(init_fn)
        disc.apply(init_fn)

    return {'gen': gen, 'disc': disc}


class PeriodicSampler(nn.Module):
    def __init__(self, latent_dims, hidden):
        super().__init__()
        self.d_l, self.d_g, self.d_p = latent_dims
        self.hidden = hidden
        self.W = nn.Conv2d(self.d_g, self.hidden, 1, 1)
        self.W_1 = nn.Conv2d(self.hidden, self.d_p, 1, 1)
        self.W_2 = nn.Conv2d(self.hidden, self.d_p, 1, 1)
        self.activaton = nn.ReLU()

        nn.init.normal_(self.W.weight.data, 0.0, 0.02)
        nn.init.normal_(self.W.bias.data, 0.0, 0.02)
        nn.init.normal_(self.W_1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.W_2.weight.data, 0.0, 0.02)
        c = torch.linspace(0.0, 1.0, self.d_p) * np.pi
        self.W_1.bias.data.copy_(c + torch.randn(self.d_p)*0.02*c)
        self.W_2.bias.data.copy_(c + torch.randn(self.d_p)*0.02*c)

    def forward(self, z_g):
        b, _, h, w = z_g.shape
        x = self.activaton(self.W(z_g))
        wave_vectors = torch.stack(
            [self.W_1(x), self.W_2(x)], dim=-1).unsqueeze(-2)
        grid = torch.stack(torch.meshgrid(
            torch.arange(h), torch.arange(w)), dim=-1
        ).unsqueeze(-1).to(dtype=z_g.dtype, device=z_g.device)
        z_p = torch.matmul(wave_vectors, grid).view(b, self.d_p, h, w)
        z_p += 2*np.pi*torch.rand(b, self.d_p, 1, 1, device=z_g.device)
        z_p = torch.sin(z_p)
        z_l = torch.rand(b, self.d_l, h, w, device=z_g.device)*2 - 1
        return torch.cat([z_l, z_g, z_p], dim=1)
