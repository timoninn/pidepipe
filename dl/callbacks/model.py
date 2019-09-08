from typing import Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from ..core.callback import Callback
from ..core.state import State
from ..utils.functions import get_available_device, to_device

Scheduler = _LRScheduler


class ModelCallback(Callback):

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None
    ):
        self.model = model

        self.device = device if device is not None else get_available_device()

    def on_begin(self, state: State):
        state.model = self.model
        state.device = self.device

        # Fix optimizer loading 3
        # Model to device before optimizer loading state dict
        state.model.to(state.device)

    def on_phase_begin(self, state: State):
        state.model.train(state.is_train_phase)

    def on_batch_begin(self, state: State):
        with torch.set_grad_enabled(state.is_train_phase):
            input = to_device(device=state.device, value=state.input)
            target = to_device(device=state.device, value=state.target)

            state.batch[0] = input
            state.batch[1] = target

            # state.batch = (input, target)
            state.output = state.model(input)
