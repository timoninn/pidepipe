from typing import Dict, Any

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from .callback import Callback
from .state import State
from ..utils.functions import get_available_device

Scheduler = _LRScheduler


class Runner():

    def __init__(
        self,
        device: torch.device = None
    ):
        self.device = device if device is not None else get_available_device()
        self.state: State = None

    def _run(self):
        # Fix optimizer loading 3
        # Model to device before optimizer loading state dict
        self.state.model.to(self.device)

        self._run_event('begin')

        for epoch in range(self.state.num_epochs):
            self._run_epoch(epoch)

            if self.state.stop_train:
                break

        self._run_event('end')

    def _run_epoch(self, epoch):

        self._run_event('epoch_begin')

        for phase in self._get_phases():
            self._run_phase(phase=phase, epoch=epoch)

        self._run_event('epoch_end')

    def _run_phase(self, phase: str, epoch: int):
        loader = self._get_loader(phase=phase)

        self.state.phase = phase
        self.state.epoch = epoch
        self.state.num_batches = len(loader)

        self.state.meter.begin_phase(phase=phase)

        self.state.model.train(self.state.is_train_phase)

        self._run_event('phase_begin')

        with torch.set_grad_enabled(self.state.is_train_phase):
            for idx, batch in enumerate(loader):
                self.state.batch_idx = idx

                self._run_batch(batch)

        self.state.meter.end_phase(phase=phase)

        self._run_event('phase_end')

    def _run_batch(self, batch):
        self._run_event('batch_begin')

        input, target = batch

        input = self._to_device(input)

        # Targer is None for infer phase.
        target = self._to_device(target)

        # Check where to zero gradients before or after model call.
        if self.state.is_train_phase:
            self.state.optimizer.zero_grad()

        output = self.state.model(input)

        self.state.input = input
        self.state.target = target
        self.state.output = output

        # Criterion may be None for infer and valid phases.
        if self.state.is_infer_phase == False and self.state.criterion is not None:
            loss = self.state.criterion(output, target)

            self.state.meter.add_batch_value(
                phase=self.state.phase,
                metric_name='loss',
                value=loss.item(),
                batch_size=self.state.input.size(0)
            )

            if self.state.is_train_phase:
                loss.backward()
                self.state.optimizer.step()

        self._run_event('batch_end')

    def _to_device(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(self.device)
        else:
            return value

    def _get_loader(self, phase: str) -> DataLoader:
        return self.loaders[phase]

    def _get_phases(self) -> [str]:
        return self.loaders.keys()

    def _run_event(self, name: str):
        if self.callbacks is not None:
            for callback in self.callbacks:
                getattr(callback, f'on_{name}')(self.state)

    def train(
        self,

        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Scheduler,

        train_loader: DataLoader,
        valid_loader: DataLoader,

        log_dir: str,
        num_epochs: int,

        callbacks: [Callback]
    ):
        self.callbacks = callbacks

        self.loaders: Dict[str, DataLoader] = {
            'train': train_loader,
            'valid': valid_loader
        }

        self.state = State(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,

            epoch=0,
            num_epochs=num_epochs,

            log_dir=log_dir
        )

        self._run()

    def eval(
        self,
        model: nn.Module,
        loader: DataLoader,
        callbacks: [Callback]
    ):
        self.callbacks = callbacks

        self.loaders: Dict[str, DataLoader] = {
            'valid': loader
        }

        self.state = State(
            model=model,
            optimizer=None,
            scheduler=None,
            criterion=None,

            epoch=0,
            num_epochs=1
        )

        self._run()
