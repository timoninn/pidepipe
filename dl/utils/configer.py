from pathlib import Path
import importlib

import torch

from .reader import load_yaml
from .model import Model


class Configer:

    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)

    def get_custom_object(self, name: str):
        sub_config = self.config[name]

        return get_custom_object(sub_config)

    @property
    def seed(self):
        return self.config['experiment']['seed']

    @property
    def activation(self):
        return self.config['model']['activation']

    @property
    def model(self) -> Model:
        sub_config = self.config['model']

        return get_custom_object(sub_config)

    @property
    def criterion(self):
        sub_config = self.config['criterion']

        return get_custom_object(sub_config)

    def get_optimizer(self, parameters):
        sub_config = self.config['optimizer']

        p = {'params': parameters}
        return get_object(torch.optim, sub_config, **p)

    def get_scheduler(self, optimizer):
        sub_config = self.config['scheduler']

        p = {'optimizer': optimizer}
        return get_object(torch.optim.lr_scheduler, sub_config, **p)


def get_custom_object(sub_config, **kwargs):
    m = importlib.import_module(sub_config['py'])
    o = get_object(m, sub_config, **kwargs)

    return o


def get_object(module, sub_config, **kwargs):
    c = getattr(module, sub_config['class'])

    args = sub_config.get('args')

    if args is not None:
        o = c(**args, **kwargs)
    else:
        o = c(**kwargs)

    return o
