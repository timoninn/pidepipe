from torch import nn
import torch


def get_available_device() -> torch.device:

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    activation: str = 'sigmoid'
) -> float:
    """Calculate dise loss.

    Arguments:
        logits {torch.Tensor} -- Predicted logits
        target {torch.Tensor} -- Target

    Keyword Arguments:
        threshold {float} -- Threshold for output binarization (default: None)
        eps {float} -- Epsilon for numeric stability (default: {1e-7})
        activation {str} -- Torch activation function applied for logits.
            One of 'none', 'sogmoid', 'softmax2d' (default: {'sigmoid'})

    Returns:
        float -- Dice score
    """
    activation_func = get_activation_func(activation)
    predicted = activation_func(logits)

    if threshold is not None:
        predicted = (predicted > threshold).float()

    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)

    dice = 2.0 * intersection / (union + eps)

    return dice
