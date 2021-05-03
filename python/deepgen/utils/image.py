import torch


class Unnormalize():
    def __init__(self, channels_mean, channels_std) -> None:
        self.mean = channels_mean[..., None, None]
        self.std = channels_std[..., None, None]

    def __call__(self, tensor_image):
        return tensor_image*self.std+self.mean
