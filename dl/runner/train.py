from typing import Dict, List
from pathlib import Path

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from ..core.runner import Runner
from ..core.callback import Callback
from pidepipe.dl.callbacks.logging import ConsoleLoggingCallback, FileLoggingCallback, TensorboardLoggingCallback
from pidepipe.dl.callbacks.metrics import MetricsCallback
from pidepipe.dl.callbacks.model import ModelCallback
from pidepipe.dl.callbacks.scheduler import SchedulerCallback
from pidepipe.dl.callbacks.checkpoint import SaveCheckpointCallback, LoadCheckpointCallback
from pidepipe.dl.callbacks.train import TrainCallback


class TrainRunner(Runner):

    def __init__(
        self
    ):
        pass

    def train(
        self,

        model: nn.Module,
        activation: str,

        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: lr_scheduler._LRScheduler,

        train_loader: Dict[str, DataLoader],
        valid_loader: Dict[str, DataLoader] = None,

        num_epochs: int = 1,
        metrics: Dict[str, nn.Module] = None,

        log_dir: str = None,
        resume_dir: str = None,

        monitor: str = 'train_loss',

        callbacks: List[Callback] = None,
    ):

        callbacks = [
            ModelCallback(model=model, activation=activation),
            TrainCallback(criterion=criterion, optimizer=optimizer)
        ]

        if scheduler is not None:
            callbacks.append(
                SchedulerCallback(scheduler=scheduler, monitor=monitor)
            )

        if resume_dir is not None:
            resume_path = Path(resume_dir) / 'checkpoints' / 'best.pt'

            callbacks.append(
                LoadCheckpointCallback(path=resume_path)
            )

        if metrics is not None:
            callbacks.append(MetricsCallback(metrics=metrics))

        callbacks.append(ConsoleLoggingCallback())

        if log_dir is not None:
            log_path = Path(log_dir)

            callbacks.extend(
                FileLoggingCallback(log_dir=log_path / 'logs'),

                SaveCheckpointCallback(
                    path=log_path / 'checkpoints',
                    monitor=monitor,
                    minimize=True
                )
            )

        loaders = {
            'train': train_loader,
            'valid': valid_loader
        }

        self.run2(
            loaders=loaders,
            num_epochs=num_epochs,
            callbacks=callbacks
        )

    def eval(
        self,

        model: nn.Module,
        activation: str,

        loader: Dict[str, DataLoader],
        metrics: Dict[str, nn.Module],
        resume_dir: str
    ):
        pass

    def infer(
        self,

        model: nn.Module,
        activation: str,

        loader: Dict[str, DataLoader],
        out_dir: str,
        resume_dir: str,
    ):
        pass
