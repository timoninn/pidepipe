import numpy as np
import torch
import torch.nn as nn


from .functions import dice, get_activation_func


class DiceLoss(nn.Module):

    def __init__(
        self,
        eps: float = 1e-7,
        activation: str = 'sigmoid',
        reduction: str = 'mean'
    ):
        super().__init__()

        self.eps = eps
        self.activation = activation
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        return dice(
            logits=logits,
            target=target,
            eps=self.eps,
            activation=self.activation,
            reduction=self.reduction
        )

class BCEDiceLoss(nn.Module):

    def __init__(
        self,
        activation: str = 'sigmoid'
    ):
        super().__init__()

        self.bce = nn.BCELoss()
        self.dice = DiceLoss(activation=activation)
        self.activation = get_activation_func(activation)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> float:

        predicted = self.activation(input=logits)

        return self.bce(predicted, target) - torch.log(self.dice(logits, target))



class BCEDiceWeightedLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 0.5
    ):
        super().__init__()

        self.alpha = alpha

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        return self.alpha * self.bce(logits, target) - (1 - self.alpha) * self.dice(logits, target) + 1
