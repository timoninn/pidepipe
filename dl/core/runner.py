from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

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

    def _run_epoch(self, epoch):

        self._run_event('epoch_begin')

        self._run_train_phase(epoch)
        self._run_valid_phase(epoch)

        self._run_event('epoch_end')

    def _run_train_phase(self, epoch: int):

        self.state.phase = 'train'
        self.state.epoch = epoch
        self.state.model.train()

        self._run_event('phase_begin')

        num_batches = len(self.train_loader)
        tk = tqdm(self.train_loader, total=num_batches)

        for batch in tk:
            self._run_train_batch(batch)

        self._run_event('phase_end')

    def _run_valid_phase(self, epoch):

        self.state.phase = 'valid'
        self.state.epoch = epoch
        self.state.model.eval()

        self._run_event('phase_begin')

        with torch.no_grad():
            num_batches = len(self.valid_loader)
            tk = tqdm(self.valid_loader, total=num_batches)

            for batch in tk:
                self._run_valid_batch(batch)

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
            value=loss.value()
        )

        self._run_event('batch_end')

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
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.callbacks = callbacks

        model.to(self.device)

        self.state = State(
            phase=None,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            metrics=[],
            log_dir=log_dir,
            epoch=0,
            num_epochs=num_epochs,
            stop_train=False
        )

        for epoch in range(num_epochs):
            if self.state.stop_train:
                break

            self._run_epoch(epoch)

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        callbacks: [Callback],
        checkpoint_path: str
    ):
        pass

    def infer(
        self,
        model: nn.Module,
        loaders: {str: DataLoader},
        checkpoint_path: str,
        save_logits_path: str
    ):
        # Infer phase
        checkpoint = torch.load(checkpoint_path)
        model_state_dict = checkpoint['model_state_dict']

        model.load_state_dict(model_state_dict)
        model.to(self.device)

        with torch.no_grad():
            model.eval()
            infer_loader = loaders['infer']
            num_batches = len(infer_loader)

            tq = tqdm(infer_loader, total=num_batches)

            result = []
            for itr, batch in enumerate(tq):
                # Infer phase batch.
                images, _ = batch

                images = images.to(self.device)

                outputs = model(images)

                for arr in outputs.cpu().numpy():
                    result.extend(arr)

            np.save(save_logits_path + 'logits.npy', result)
