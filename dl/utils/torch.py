from typing import Any

import torch
from torch import nn


def set_requires_grad(module: nn.Module, requires: bool):
    for param in module.parameters():
        param.requires_grad = requires


def get_available_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(device: torch.device, value: Any) -> Any:
    if torch.is_tensor(value):
        return value.to(device)

    elif isinstance(value, tuple):
        return tuple(to_device(device, v) for v in value)

    elif isinstance(value, list):
        return list(to_device(device, v) for v in value)

    else:
        return value
