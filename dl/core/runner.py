from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

from callback import Callback
from state import State

Scheduler = _LRScheduler


class Runner():

    def __init__(
        self,
        device=None
    ):
        self.device = device
        self.best_valid_loss = float('inf')

        self.callbacks: [Callback] = None
        self.state: State = None

    def _run_event(self, name: str):

        if self.callbacks is not None:
            for callback in self.callbacks:
                getattr(callback, f'on_{name}')(self.state)

    def _run_train_batch(self, batch):
        images, masks = batch

        self.optimizer.zero_grad()

        images = images.to(self.device)
        masks = masks.to(self.device)

        outputs = self.model(images)

        loss = self.criterion(outputs, masks)

        loss.backward()

        self.optimizer.step()

        return loss.intem()

    def _run_valid_batch(self):
        pass

    def _run_epoch(self):
        pass

    def _run_phase(self):
        pass

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



        self.model = model
        self.callbacks = callbacks
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer


        model.to(self.device)

        for epoch in range(num_epochs):

            # One batch

            # Train phase

            self._run_event('epoch_begin')

            self._run_event('phase_begin')

            print(f'{epoch}/{num_epochs} Epoch {epoch} (train)')

            model.train()
            train_loader = loaders['train']

            num_batches = len(train_loader)
            tk = tqdm(train_loader, total=num_batches)

            # Init metric/loss
            running_loss = 0.0

            for itr, batch in enumerate(tk):
                self._run_event('batch_begin')

                loss = self._run_train_batch(batch)

                running_loss += loss

                self._run_event('batch_end')
                tk.set_postfix({'loss': running_loss / (itr + 1)})

            # Train phase loss.

            # Get current metric/loss value
            epoch_loss = running_loss / num_batches
            print(f'Loss: {epoch_loss:.4}')

            state = {
                "epoch": epoch,
                "loss": epoch_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(state, log_dir + 'last.pth')

            self._run_event('phase_end')

            # Valid pahse

            self._run_event('phase_begin')

            print(f'{epoch}/{num_epochs} Epoch {epoch} (valid)')
            with torch.no_grad():
                model.eval()
                valid_loader = loaders['valid']

                num_batches = len(valid_loader)
                tk = tqdm(valid_loader, total=num_batches)

                running_loss = 0.0

                for itr, batch in enumerate(tk):
                    self._run_event('batch_begin')

                    # Valid phase one batch

                    images, masks = batch

                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = model(images)

                    loss = criterion(outputs, masks)

                    running_loss += loss.item()

                    self._run_event('batch_end')

            # Valid phase loss.
            epoch_loss = running_loss / num_batches
            print(f'Loss: {epoch_loss:.4}')

            if epoch_loss < self.best_valid_loss:
                print('New optimal found. Saving state')
                self.best_valid_loss = epoch_loss

                torch.save(state, log_dir + 'best.pth')

            self._run_event('phase_end')

            self._run_event('epoch_end')

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
