import torch
import torch.nn as nn

class Initializer():
    def __init__(self, init_map):
        self.init_map = init_map
    
    def __call__(self, module):
        for type, init_dict in self.init_map.items():
            if isinstance(module, type):
                for param, (init_fn, init_params) in init_dict.items():
                    if hasattr(module, param):
                        init_fn(getattr(module, param).data, **init_params)
