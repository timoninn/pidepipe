from typing import Dict, List
from pathlib import Path

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from ..core.runner import Runner
from ..core.callback import Callback
from pidepipe.dl.callbacks.train import TrainCallback
from pidepipe.dl.callbacks.checkpoint import SaveCheckpointCallback, LoadCheckpointCallback
from pidepipe.dl.callbacks.scheduler import SchedulerCallback
from pidepipe.dl.callbacks.model import ModelCallback
from pidepipe.dl.callbacks.metrics import MetricsCallback
from pidepipe.dl.callbacks.logging import ConsoleLoggingCallback, FileLoggingCallback, TensorboardLoggingCallback
from pidepipe.dl.callbacks.infer import InferCallback, FilesInferCallback


class TrainRunner(Runner):

    def __init__(
        self
    ):
        pass

    def _get_checkpoints_path(self, log_dir: str) -> Path:
        return Path(log_dir) / 'checkpoints'

    def _get_logs_path(self, log_dir: str) -> Path:
        return Path(log_dir) / 'logs'

    def _get_infer_path(self, log_dir: str) -> Path:
        return Path(log_dir) / 'infer'

    def train(
        self,

        model: nn.Module,
        activation: str,

        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: lr_scheduler._LRScheduler,

        train_loader: DataLoader,
        valid_loader: DataLoader,

        num_epochs: int = 1,
        metrics: Dict[str, nn.Module] = None,
        monitor: str = 'train_loss',

        log_dir: str = None,
        resume_dir: str = None,

        callbacks: List[Callback] = None,
    ):
        loaders = {
            'train': train_loader,
            'valid': valid_loader
        }

        callbacks = [
            ModelCallback(model=model, activation=activation),
            TrainCallback(criterion=criterion, optimizer=optimizer)
        ]

        if scheduler is not None:
            callbacks.append(
                SchedulerCallback(scheduler=scheduler, monitor=monitor)
            )

        if resume_dir is not None:
            resume_path = self._get_checkpoints_path(resume_dir) / 'best.pt'

            callbacks.append(LoadCheckpointCallback(path=resume_path))

        if metrics is not None:
            callbacks.append(MetricsCallback(metrics=metrics))

        callbacks.append(ConsoleLoggingCallback())

        if log_dir is not None:
            callbacks.extend(
                [
                    FileLoggingCallback(log_dir=self._get_logs_path(log_dir)),

                    SaveCheckpointCallback(
                        path=self._get_checkpoints_path(log_dir),
                        monitor=monitor,
                        minimize=True
                    )
                ]
            )

        self.run2(
            loaders=loaders,
            num_epochs=num_epochs,
            callbacks=callbacks
        )

    def eval(
        self,

        model: nn.Module,
        activation: str,

        loader: DataLoader,
        metrics: Dict[str, nn.Module],
        resume_dir: str
    ):
        loaders = {'valid': loader}

        callbacks = [
            ModelCallback(model=model, activation=activation),

            LoadCheckpointCallback(
                path=self._get_checkpoints_path(resume_dir) / 'best.pt'
            ),

            MetricsCallback(metrics=metrics),

            ConsoleLoggingCallback()
        ]

        self.run2(
            loaders=loaders,
            num_epochs=1,
            callbacks=callbacks
        )

    def infer(
        self,

        model: nn.Module,
        activation: str,

        loader: DataLoader,
        out_dir: str,
        resume_dir: str,

        one_file_output: bool = False
    ):
        loaders = {'infer': loader}

        if one_file_output:
            infer_callback = InferCallback(
                out_dir=self._get_infer_path(out_dir)
            )
        else:
            infer_callback = FilesInferCallback(
                out_dir=self._get_infer_path(out_dir)
            )

        callbacks = [
            ModelCallback(model=model, activation=activation),

            LoadCheckpointCallback(
                path=self._get_checkpoints_path(resume_dir) / 'best.pt'
            ),

            infer_callback,

            ConsoleLoggingCallback()
        ]

        self.run2(
            loaders=loaders,
            num_epochs=1,
            callbacks=callbacks
        )