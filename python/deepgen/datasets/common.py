import os
from PIL import Image
import torch
import torchvision.transforms as tf
from torchvision.datasets import CIFAR10
import torch
from torchvision.transforms import transforms


def cifar10n(transform=None):
    if transform is None:
        transform = tf.ToTensor()
    cifar10 = CIFAR10('./data/cifar10', transform=transform, download=True)
    data = torch.cat([s[0].unsqueeze(0) for s in cifar10], dim=0)
    mean, std = compute_mean(cifar10)
    cifar10_n = (data - mean[..., None, None])/std[..., None, None]
    dataset = DiscardLabels(torch.utils.data.TensorDataset(cifar10_n))
    return dataset, mean, std


def compute_mean(dataset):
    mean = torch.mean(torch.stack(
        [s[0].mean((1, 2)) for s in dataset]), dim=0)
    mean_2 = torch.mean(torch.stack(
        [(s[0]**2).mean((1, 2)) for s in dataset]), dim=0)
    std = torch.sqrt(mean_2 - mean**2)
    return mean, std


class DiscardLabels():
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0]


class SingleImageDataset():
    def __init__(self, image_path, iterations,
                 PIL_image_mode='RGB', transform=None):
        self.image_path = os.path.abspath(image_path)
        self.image = Image.open(self.image_path).convert(PIL_image_mode)
        self.transform = transform
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        return self.transform(self.image) if self.transform is not None \
            else self.image
