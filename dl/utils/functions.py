import torch
from torch import nn


def get_available_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_activation_func(name: str == 'none'):
    if name == 'none':
        return lambda x: x

    elif name == 'sigmoid':
        return nn.Sigmoid()

    elif name == 'softmax2d':
        return nn.Softmax2d()

    else:
        raise NotImplementedError(
            'Only "sigmoid" and "softmax2d" activation was implemented'
        )


def dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = None,
    eps: float = 1e-7,
    activation: str = 'sigmoid',
    reduction: str = 'mean'
) -> float:
    """Calculate dise loss.

    Arguments:
        logits {torch.Tensor} -- Predicted logits. Shape (N, ...)
        target {torch.Tensor} -- Target. Shape (N, ...)

        N - batch size

    Keyword Arguments:
        threshold {float} -- Threshold for output binarization (default: None)
        eps {float} -- Epsilon for numeric stability (default: {1e-7})
        activation {str} -- Torch activation function applied for logits.
            One of 'none', 'sogmoid', 'softmax2d' (default: {'sigmoid'})

    Returns:
        float -- Dice score
    """
    batch_size = logits.size(0)

    activation_func = get_activation_func(activation)
    predicted = activation_func(logits)

    if threshold is not None:
        predicted = (predicted > threshold).float()


    sum_dice = 0.0
    for p, t in zip(predicted, target):
        intersection = torch.sum(p * t)
        union = torch.sum(p) + torch.sum(t)
        dice = 2.0 * intersection / (union + eps)

        sum_dice += dice

    if reduction == 'none':
        result = sum_dice

    elif reduction == 'mean':
        result = sum_dice / batch_size

    else:
        raise NotImplementedError(
            'Only "none" and "mean" reductions were implemented'
        )

    return result
