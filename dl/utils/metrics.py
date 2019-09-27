import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from .functions import dice

class AccuracyMetric(nn.Module):
    def __init__(
        self,
        threshold: float = 0.5
    ):
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ) -> float:
        batch_size = output.size(0)
        length = output.size(1)

        predicted = (output > self.threshold).int()
        correct = (predicted == target.int()).sum().float()

        accuracy = correct / length

        return accuracy / batch_size

class DiceMetric(nn.Module):

    def __init__(
        self,
        threshold: float = 0.5,
        eps: float = 1e-7,
        activation: str = 'none'
    ):
        super().__init__()

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
