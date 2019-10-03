from typing import Dict

import torch
from torch import nn

from ..core.callback import Callback
from ..core.state import State
from ..core.meter import Monitor


class MetricsCallback(Callback):

    def __init__(
        self,
        metrics: Dict[str, nn.Module]
    ):
        self.metrics = metrics

    def on_batch_end(self, state: State):
        for name, metric in self.metrics.items():
            value = metric(state.output, state.target)

            state.meter.add_batch_value(
                phase=state.phase,
                metric_name=name,
                value=value.item(),
                batch_size=state.input.size(0)
            )
