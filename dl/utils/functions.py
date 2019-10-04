from typing import Any

import torch
from torch import nn
from torch import Tensor


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


def to_predictions(
    output: Tensor,
    threshold: float = 0.5,
    activation: str = None,
) -> Tensor:
    activation_func = get_activation_func(activation)
    predicted = activation_func(output)

    if threshold is not None:
        predicted = (predicted > threshold).int()

    return predicted


def f_score(
    output: Tensor,
    target: Tensor,
    beta: float = 1.0,
    threshold: float = 0.5,
    activation: str = None,
    eps: float = 1e-7
) -> Tensor:
    assert output.dim() == target.dim() == 4, '4D tensors as input required'

    predicted = to_predictions(
        output=output,
        threshold=threshold,
        activation=activation
    )

    tp = torch.sum(predicted * target)
    fp = torch.sum(predicted) - tp
    fn = torch.sum(predicted) - tp

    numerator = (1 + beta**2) * tp
    denominator = numerator + (beta**2 * fn) + fp

    return numerator / (denominator + eps)


def accuracy(
    output: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    activation: str = None,
) -> Tensor:
    assert output.dim() == target.dim() == 4, '4D tensors as input required'

    predicted = to_predictions(
        output=output,
        threshold=threshold,
        activation=activation
    )

    length = output.size(1)

    correct = (predicted == target.int()).sum().float()

    return correct / length
