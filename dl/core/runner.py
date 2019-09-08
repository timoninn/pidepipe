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

    def __init__(self):
        self.state: State = None

    def _run(self):
        self._send_event('begin')

        for epoch in range(1, self.state.num_epochs + 1):
            if self.state.stop_running:
                break

            self._run_epoch(epoch)

        self._send_event('end')

    def _run_epoch(self, epoch):
        self.state.epoch = epoch

        self._send_event('epoch_begin')

        for phase in self._get_phases():
            self._run_phase(phase=phase)

        self._send_event('epoch_end')

    def _run_phase(self, phase: str):
        loader = self._get_loader(phase=phase)

        self.state.phase = phase
        self.state.num_batches = len(loader)

        self.state.meter.begin_phase(phase=phase)

        self._send_event('phase_begin')

        for idx, batch in enumerate(loader):
            self.state.batch_idx = idx
            self._run_batch(batch)

        self.state.meter.end_phase(phase=phase)

        self._send_event('phase_end')

    def _run_batch(self, batch):
        self.state.batch = batch

        self._send_event('batch_begin')
        self._send_event('batch_end')

    def _get_loader(self, phase: str) -> DataLoader:
        return self.loaders[phase]

    def _get_phases(self) -> [str]:
        return self.loaders.keys()

    def _send_event(self, name: str):
        if self.callbacks is not None:
            for callback in self.callbacks:
                getattr(callback, f'on_{name}')(self.state)

    def run2(
        self,
        loaders: Dict[str, DataLoader],

        num_epochs: int,
        callbacks: [Callback]
    ):
        self.loaders = loaders
        self.callbacks = callbacks

        self.state = State(
            model=None,
            optimizer=None,
            scheduler=None,
            criterion=None,
            epoch=0,
            num_epochs=num_epochs
        )

        self._run()

    def run(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Scheduler,

        loaders: Dict[str, DataLoader],

        num_epochs: int,
        callbacks: [Callback]
    ):
        self.loaders = loaders
        self.callbacks = callbacks

        self.state = State(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epoch=0,
            num_epochs=num_epochs
        )

        self._run()

    def train(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Scheduler,

        train_loader: DataLoader,
        valid_loader: DataLoader,

        num_epochs: int,
        callbacks: [Callback]
    ):
        loaders = {
            'train': train_loader,
            'valid': valid_loader
        }

        self.run(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            num_epochs=1,
            callbacks=callbacks
        )

    def eval(
        self,
        model: nn.Module,
        loader: DataLoader,
        callbacks: [Callback]
    ):
        self.run(
            model=model,
            criterion=None,
            optimizer=None,
            scheduler=None,
            loaders={'valid': loader},
            num_epochs=1,
            callbacks=callbacks
        )

    def infer(
        self,
        model: nn.Module,
        loader: DataLoader,
        callbacks: [Callback]
    ):
        self.run(
            model=model,
            criterion=None,
            optimizer=None,
            scheduler=None,
            loaders={'infer': loader},
            num_epochs=1,
            callbacks=callbacks
        )
