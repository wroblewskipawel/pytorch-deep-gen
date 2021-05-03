import torch.nn as nn


def vanilla_dcgan(latent_channels, image_channels, hidden_layers, **kwargs):
    return _dcgan_builder(StridedConvUpsample, StridedConvDownsample,
                          latent_channels, image_channels,
                          hidden_layers, **kwargs)


def _dcgan_builder(gen_build_fn, disc_build_fn, latent_channels,
                   image_channels, hidden_layers, base_channels=64,
                   growth_ratio=2, init_fn=None, gen_output_act=(nn.Tanh, {})):
    generator_arch = [
        {'in_channels': latent_channels,
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
        *[gen_build_fn(**layer_kwargs)
          for layer_kwargs in generator_arch])

    disc = nn.Sequential(
        *[disc_build_fn(**layer_kwargs)
          for layer_kwargs in discriminator_arch])

    if init_fn is not None:
        gen.apply(init_fn)
        disc.apply(init_fn)

    return {'gen': gen, 'disc': disc}


class StridedConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=(nn.ReLU, {'inplace': True}), use_bnorm=True,
                 dropout_p=None):
        super().__init__()
        net = []
        net.append(nn.ConvTranspose2d(in_channels, out_channels,
                   kernel_size=5, stride=2, padding=2))
        if use_bnorm:
            net.append(nn.BatchNorm2d(out_channels))
        if dropout_p is not None:
            net.append(nn.Dropout2d(dropout_p))
        if activation is not None:
            net.append(activation[0](**activation[1]))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class StridedConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=(nn.LeakyReLU, {
                     'negative_slope': 0.2, 'inplace': True}),
                 use_bnorm=False, dropout_p=None):
        super().__init__()
        net = []
        net.append(nn.Conv2d(in_channels, out_channels,
                   kernel_size=5, stride=2, padding=2))
        if use_bnorm:
            net.append(nn.BatchNorm2d(out_channels))
        if dropout_p is not None:
            net.append(nn.Dropout2d(dropout_p))
        if activation is not None:
            net.append(activation[0](**activation[1]))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResizeConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=(nn.ReLU, {'inplace': True}),
                 use_bnorm=True, dropout_p=None):
        super().__init__()
        net = []
        net.append(nn.Upsample(2, mode='nearest'))
        net.append(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1))
        if use_bnorm is True:
            net.append(nn.BatchNorm2d(out_channels))
        if dropout_p is not None:
            net.append(nn.Dropout2d(dropout_p))
        if activation is not None:
            net.append(activation[0](**activation[1]))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResizeConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=(nn.LeakyReLU, {
                     'negative_slope': 0.2, 'inplace': True}),
                 use_bnorm=False, dropout_p=None):
        super().__init__()
        net = []
        net.append(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1))
        if use_bnorm is True:
            net.append(nn.BatchNorm2d(out_channels))
        if dropout_p is not None:
            net.append(nn.Dropout2d(dropout_p))
        if activation is not None:
            net.append(activation[0](**activation[1]))
        net.append(nn.AvgPool2d(2, 2))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
