from typing import Dict

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
        self._run_event('begin')

        self.model.to(self.device)

        for epoch in range(self.state.num_epochs):
            self._run_epoch(epoch)

            if self.state.stop_train:
                break

        self._run_event('end')

    def _run_epoch(self, epoch):

        self._run_event('epoch_begin')

        for phase in self._get_phases():
            self._run_phase(phase=phase, epoch=epoch)

        # self._run_train_phase(epoch)
        # self._run_valid_phase(epoch)

        self._run_event('epoch_end')

    def _run_phase(self, phase: str, epoch: int):
        loader = self._get_loader(phase=phase)
        is_train = self.state.is_train_phase

        self.state.phase = phase
        self.state.epoch = epoch
        self.state.num_batches = len(loader)

        self.state.meter.begin_phase(phase=phase)

        self.state.model.train(is_train)

        self._run_event('phase_begin')

        with torch.set_grad_enabled(is_train):
            for idx, batch in enumerate(loader):
                self.state.batch_idx = idx

                self._run_batch(batch)

        self.state.meter.end_phase(phase=phase)

        self._run_event('phase_end')

    def _run_batch(self, batch):
        self._run_event('batch_begin')

        input, target = batch

        input = input.to(self.device)
        target = target.to(self.device)

        # Check where to zero gradients before or after model call.
        if self.state.is_train_phase():
            self.state.optimizer.zero_grad()

        output = self.state.model(input)

        self.state.input = input
        self.state.target = target
        self.state.output = output

        # Only for non-infer phase
        loss = self.state.criterion(output, target)

        if self.state.is_train_phase():
            loss.backward()
            self.state.optimizer.step()

        self.state.meter.add_batch_value(
            phase=self.state.phase,
            metric_name='loss',
            value=loss.item()
        )
        # Only for non-infer phase

        self._run_event('batch_end')

    def _run_train_phase(self, epoch: int):
        self.state.phase = 'train'
        self.state.epoch = epoch
        self.state.num_batches = len(self.valid_loader)

        self.state.meter.begin_phase(phase='train')
        self.state.model.train()

        self._run_event('phase_begin')

        for idx, batch in enumerate(self.train_loader):
            self.state.batch_idx = idx

            self._run_train_batch(batch)

        self.state.meter.end_phase(phase='train')

        self._run_event('phase_end')

    def _run_valid_phase(self, epoch):
        self.state.phase = 'valid'
        self.state.epoch = epoch
        self.state.num_batches = len(self.valid_loader)

        self.state.meter.begin_phase(phase='valid')
        self.state.model.eval()

        self._run_event('phase_begin')

        with torch.no_grad():
            for idx, batch in enumerate(self.valid_loader):
                self.state.batch_idx = idx

                self._run_valid_batch(batch)

        self.state.meter.end_phase(phase='valid')

        self._run_event('phase_end')

    def _run_train_batch(self, batch):
        self._run_event('batch_begin')

        images, masks = batch

        self.state.optimizer.zero_grad()

        images = images.to(self.device)
        masks = masks.to(self.device)

        outputs = self.state.model(images)

        self.state.input = images
        self.state.target = masks
        self.state.output = outputs

        loss = self.state.criterion(outputs, masks)

        loss.backward()
        self.state.optimizer.step()

        self.state.meter.add_batch_value(
            phase=self.state.phase,
            metric_name='loss',
            value=loss.item()
        )

        self._run_event('batch_end')

    def _run_valid_batch(self, batch):
        self._run_event('batch_begin')

        images, masks = batch

        images = images.to(self.device)
        masks = masks.to(self.device)

        outputs = self.state.model(images)

        self.state.input = images
        self.state.target = masks
        self.state.output = outputs

        loss = self.state.criterion(outputs, masks)

        self.state.meter.add_batch_value(
            phase=self.state.phase,
            metric_name='loss',
            value=loss.item()
        )

        self._run_event('batch_end')

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

        valid_loader: DataLoader,
        train_loader: DataLoader,

        log_dir: str,
        num_epochs: int,

        callbacks: [Callback]
    ):
        # self.train_loader = train_loader
        # self.valid_loader = valid_loader

        self.loaders: Dict[str, DataLoader] = {
            'train': train_loader,
            'valid': valid_loader
        }

        self.callbacks = callbacks

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

    def infer(
        self,
        model: nn.Module,
        loader: DataLoader,
        callbacks: [Callback]
    ):
        self.callbacks = callbacks

        self.loaders: Dict[str, DataLoader] = {
            'infer': loader
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
