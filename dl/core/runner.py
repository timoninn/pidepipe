from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

from callback import Callback
from state import State
from ..utils.functions import get_available_device

Scheduler = _LRScheduler


class Runner():

    def __init__(
        self,
        device: torch.device = None
    ):
        self.device = device if device is not None else get_available_device()

        self.best_valid_loss = float('inf')

        self.callbacks: [Callback] = None
        self.state: State = None

    def _run_epoch(self, epoch):

        self._run_event('epoch_begin')

        self._run_train_phase(epoch)
        self._run_valid_phase(epoch)

        self._run_event('epoch_end')

    def _run_train_phase(self, epoch: int):

        self._run_event('phase_begin')

        print(f'{epoch}/{num_epochs} Epoch {epoch} (train)')

        self.model.train()

        num_batches = len(self.train_loader)
        tk = tqdm(self.train_loader, total=num_batches)

        running_loss = 0.0

        for itr, batch in enumerate(tk):
            loss = self._run_train_batch(batch)
            running_loss += loss
            tk.set_postfix({'loss': running_loss / (itr + 1)})

        epoch_loss = running_loss / num_batches
        print(f'Loss: {epoch_loss:.4}')

        self._save_state(epoch, epoch_loss, 'last.pth')

        self._run_event('phase_end')

    def _run_valid_phase(self, epoch):

        self._run_event('phase_begin')

        print(f'{epoch}/{num_epochs} Epoch {epoch} (valid)')
        with torch.no_grad():
            self.model.eval()

            num_batches = len(self.valid_loader)
            tk = tqdm(self.valid_loader, total=num_batches)

            running_loss = 0.0
            for itr, batch in enumerate(tk):
                loss = self._run_valid_batch(batch)
                running_loss += loss

        epoch_loss = running_loss / num_batches
        print(f'Loss: {epoch_loss:.4}')

        if epoch_loss < self.best_valid_loss:
            print('New optimal found. Saving state')
            self.best_valid_loss = epoch_loss

            self._save_state(epoch, epoch_loss, 'best.pth')

        self._run_event('phase_end')

    def _run_train_batch(self, batch):

        self._run_event('batch_begin')

        images, masks = batch

        self.optimizer.zero_grad()

        images = images.to(self.device)
        masks = masks.to(self.device)

        outputs = self.model(images)
        loss = self.criterion(outputs, masks)
        loss.backward()
        self.optimizer.step()

        self._run_event('batch_end')

        return loss.item()

    def _run_valid_batch(self, batch):

        self._run_event('batch_begin')

        images, masks = batch

        images = images.to(self.device)
        masks = masks.to(self.device)

        outputs = self.model(images)
        loss = self.criterion(outputs, masks)

        self._run_event('batch_end')

        return loss.item()

    def _run_event(self, name: str):

        if self.callbacks is not None:
            for callback in self.callbacks:
                getattr(callback, f'on_{name}')(self.state)

    def _save_state(self, epoch: int, epoch_loss: float, name: str):

        state = {
            "epoch": epoch,
            "loss": epoch_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(state, self.log_dir + name)

    def train(
        self,
        model: nn.Module,

        criterion: nn.Module,
        metrics: [nn.Module],

        optimizer: optim.Optimizer,
        scheduler: Scheduler,

        loaders: {str: DataLoader},

        valid_loader: DataLoader,
        train_loader: DataLoader,

        log_dir: str,
        num_epochs: int,

        callbacks: [Callback]
    ):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.model = model
        self.callbacks = callbacks
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer

        self.log_dir = log_dir

        model.to(self.device)

        for epoch in range(num_epochs):
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
