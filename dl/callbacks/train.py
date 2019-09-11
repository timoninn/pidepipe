from typing import Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from ..core.callback import Callback
from ..core.state import State
from ..utils.functions import get_available_device


class TrainCallback(Callback):

    def __init__(
        self,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ):
        self.optimizer = optimizer
        self.criterion = criterion

    def on_begin(self, state: State):
        state.optimizer = self.optimizer
        state.criterion = self.criterion

    def on_batch_begin(self, state: State):
        with torch.set_grad_enabled(state.is_train_phase):

            # Criterion may be None for infer and valid phases.
            if state.is_infer_phase == False and state.criterion is not None:
                loss = state.criterion(state.output, state.target)

                state.meter.add_batch_value(
                    phase=state.phase,
                    metric_name='loss',
                    value=loss.item(),
                    batch_size=state.input.size(0)
                )

                if state.is_train_phase:
                    state.optimizer.zero_grad()
                    loss.backward()
                    state.optimizer.step()
