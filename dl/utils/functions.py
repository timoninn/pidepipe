from typing import Any

import torch
from torch import nn


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


def get_activation_func(name: str == 'none'):
    if name == 'none':
        return lambda x: x

    elif name == 'sigmoid':
        return nn.Sigmoid()

    elif name == 'softmax2d':
        return nn.Softmax2d()

    else:
        raise NotImplementedError(
            'Only "none", "sigmoid" and "softmax2d" activations were implemented'
        )


def dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = None,
    eps: float = 1e-7,
    activation: str = 'none',
    reduction: str = 'mean'
) -> float:
    """Calculate dise loss.

    Arguments:
        logits {torch.Tensor} -- Predicted logits. Shape (N, C, H, W)
        target {torch.Tensor} -- Target. Shape (N, C, H, W)

        N - batch size

    Keyword Arguments:
        threshold {float} -- Threshold for output binarization (default: None)
        eps {float} -- Epsilon for numeric stability (default: {1e-7})
        activation {str} -- Torch activation function applied for logits.
            One of 'none', 'sogmoid', 'softmax2d' (default: {'sigmoid'})
        reduction {str} -- Result value reduction.

    Returns:
        float -- Dice score
    """
    assert logits.dim() == target.dim() == 4, 'Dice requires 4D tensors as input'

    batch_size = logits.size(0)

    activation_func = get_activation_func(activation)
    predicted = activation_func(logits)

    if threshold is not None:
        predicted = (predicted > threshold).float()

    batch_dice = 0.0
    for p, t in zip(predicted, target):
        dice = flat_dice(
            predicted=p,
            target=t,
            eps=eps
        )

        batch_dice += dice

    if reduction == 'none':
        result = batch_dice

    elif reduction == 'mean':
        result = batch_dice / batch_size

    else:
        raise NotImplementedError(
            'Only "none" and "mean" reductions were implemented'
        )

    return result


def flat_dice(
    predicted: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> float:
    assert predicted.dim() == target.dim() == 3, 'Flat dice requires 3D tensors as input'

    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice = 2.0 * intersection / (union + eps)

    return dice


    # dd = 0

    # for p, t in zip(predicted, target):
    #     intersection = torch.sum(p * t)
    #     union = torch.sum(p) + torch.sum(t)
    #     dice = 2.0 * intersection / (union + eps)

    #     if union == 0:
    #         dd += 1
    #     else:
    #         dd += dice

    # return dd / 4