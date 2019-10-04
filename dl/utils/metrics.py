from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from .functions import dice, f_score, accuracy


class Metric(nn.Module):
    def __init__(
        self,
        metric_fn: Callable,
        **metric_params
    ):
        super().__init__()

        self.metric_fn = metric_fn
        self.metric_params = metric_params

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor
    ) -> Tensor:
        batch_size = output.size(0)
        metric_value = self.metric_fn(output, target, **self.metric_params)

        return metric_value / batch_size


class FScoreMetric(Metric):

    def __init__(
        self,
        beta: float = 1.0,
        threshold: float = 0.5,
        activation: str = None,
        eps: float = 1e-7
    ):
        super().__init__(
            metric_fn=f_score,
            beta=beta,
            threshold=threshold,
            activation=activation,
            eps=eps
        )

class AccuracyMetric2(Metric):

    def __init__(
        self,
        threshold: float = 0.5,
        activation: str = None
    ):
        super().__init__(
            metric_fn=accuracy,
            threshold=threshold,
            activation=activation
        )

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
