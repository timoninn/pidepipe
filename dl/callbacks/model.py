from typing import Dict, Any
from functools import partial

import torch
from torch import nn
from torch import Tensor

from ..core.callback import Callback
from ..core.state import State
from ..utils.torch import get_available_device, to_device, flip
from ..utils.functions import get_activation_func


class ModelCallback(Callback):

    def __init__(
        self,
        model: nn.Module,
        activation: str,
        device: torch.device = None
    ):
        self.model = model
        self.activation = get_activation_func(activation)

        self.device = device if device is not None else get_available_device()

    def _predict(self, model: nn.Module, input: torch.Tensor) -> Tensor:
        output = model(input)
        output = self.activation(output)

        return output

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

            state.batch['input'] = input
            state.batch['target'] = target

            state.output = self._predict(model=state.model, input=input)


class TtaModelCallback(ModelCallback):

    def __init__(
        self,
        model: nn.Module,
        activation: str,
        ttas: [str] = ['none'],
        apply_reverse_tta: bool = False,
        device: torch.device = None
    ):
        super().__init__(
            model=model,
            activation=activation,
            device=device
        )

        self.apply_reverse_tta = apply_reverse_tta
        self.ttas = ttas

    def _predict_tta(
        self,
        model: nn.Module,
        input: torch.Tensor,
        tta: str
    ) -> Tensor:
        last_dim = len(input.shape) - 1

        if tta == 'none':
            def tta_method(x): return x

        elif tta == 'horizontal':
            tta_method = partial(flip, dim=last_dim)

        elif tta == 'vertical':
            tta_method = partial(flip, dim=last_dim-1)

        else:
            raise NotImplementedError

        input = tta_method(input)
        output = model(input)
        if self.apply_reverse_tta:
            output = tta_method(output)
        output = self.activation(output)

        return output

    def _predict(self, model: nn.Module, input: torch.Tensor):
        outputs = [self._predict_tta(model, input, tta) for tta in self.ttas]

        outputs = torch.stack(outputs)
        output = torch.mean(outputs, dim=0)

        return output
