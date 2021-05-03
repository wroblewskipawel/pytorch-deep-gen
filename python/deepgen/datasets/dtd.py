import os
from PIL import Image
import random
import itertools


class DescripableTextureDataset():
    def __init__(self, root, transform=None, repeat=1):
        self.images_dir = os.path.join(root, 'images')
        self.texture_classes = {
            texture_class: os.listdir(
                os.path.join(self.images_dir, texture_class))
            for texture_class in os.listdir(self.images_dir)
        }
        self.transform = transform
        self.repeat = repeat
        self.chosen_classes = set(self.texture_classes.keys())

    def chose_subset(self, chosen_classes):
        assert set(chosen_classes) <= set(self.texture_classes.keys()), \
            "chosen_classes is not subset of texture_classes"
        self.chosen_classes = chosen_classes
        self.samples = list(itertools.chain.from_iterable(
            list(itertools.repeat(
                list(itertools.chain.from_iterable(
                    [[os.path.join(self.images_dir, texture_class, file_name)
                      for file_name in self.texture_classes[texture_class]]
                     for texture_class in self.chosen_classes])), self.repeat))
        ))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image
