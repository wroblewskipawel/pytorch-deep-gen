import os
import json
import itertools
import random
import torch
from PIL import Image


class TextureHeavenDataset():
    def __init__(self, root, lod=None, transform=None, repeat=1):
        data_info_path = os.path.join(root, 'data_info.json')
        assert os.path.exists(data_info_path), \
            "Could not find TextureHeaven dataset at specified path"
        self.root = root
        with open(data_info_path, 'r') as f:
            self.info = json.load(f)
        self.texture_classes = {os.path.split(key)[1]: list(
            item.keys()) for key, item in self.info.items()}
        self.texture_data_paths = {}
        for texture_class in self.info:
            for name, data in self.info['texture_class'].items():
                self.texture_data_paths[name] = {
                    file_name[len(name):]: os.path.join(
                        self.root, texture_class, name, file_name
                    ) for file_name in data['rendered']
                }
        self.texture_environments = \
            set(self.texture_data_paths[
                list(self.texture_data_paths.keys())[0]].keys())
        self.chosen_environemnts = list(self.texture_environments)
        self.chose_classes = set(self.texture_classes.keys())
        self.lod = lod
        self.transform = transform
        self.repeat = repeat
        self.chose_subset(self.chosen_classes, self.chosen_environemnts)

    @property
    def output_resolution(self):
        sample = self.__getitem__(0)
        return tuple(sample.shape[-2:]) if isinstance(sample, torch.Tensor) \
            else sample.size

    def chose_subset(self, chosen_classes, chosen_environments):
        assert set(chosen_classes) <= set(self.texture_classes.keys()), \
            "chosen_classes is not subset of available texture_classes"
        assert set(chosen_environments) <= self.texture_environments, \
            "chosen_environments is not subset" + \
            " of available texture_environments"
        self.chosen_environemnts = set(chosen_environments)
        self.chose_classes = set(chosen_classes)
        self.samples = list(itertools.chain.from_iterable(list(
            itertools.repeat(
                list(itertools.chain.from_iterable(
                    [itertools.product(self.texture_classes[c],
                                       self.chosen_environemnts)
                     for c in self.chose_classes])), self.repeat
            ))))
        random.shuffle(self.samples)

    def __getitem__(self, idx):
        texture_name, texture_environment = self.samples[idx]
        image_path = self.texture_data_paths[texture_name][texture_environment]
        image = Image.open(image_path).convert('RGB')
        if self.lod is not None:
            image = image.reduce(self.lod)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.samples)
