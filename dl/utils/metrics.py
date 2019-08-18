import torch.nn as nn
import torch
from functions import dice


class DiceMetric(nn.Module):

    def __init__(
        self,
        threshold: float = 0.5,
        eps: float = 1e-7,
        activation: str = 'sigmoid'
    ):
        self.threshold = threshold
        self.eps = eps
        self.activation = activation

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        return dice(
            logits=logits,
            target=target,
            threshold=self.threshold,
            eps=self.eps,
            activation=self.activation
        )
