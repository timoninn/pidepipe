from typing import Dict, Any, abstractmethod
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from .meter import Meter


class State:

    def __init__(
        self,
        phase: str,

        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: _LRScheduler,

        criterion: nn.Module,
        metrics: [nn.Module],

        log_dir: str,

        epoch: int,
        num_epochs: int,

        stop_train: bool = False
    ):

        self.phase: str = phase

        self.model = model
        self.optimizer = optimizer

        self.criterion = criterion
        self.metrics = metrics

        self.log_dir: str = log_dir

        self.epoch: int = epoch
        self.num_epochs: int = num_epochs

        self.batch_idx: int = None
        self.num_batches: int = None

        self.stop_train: bool = stop_train

        self.input = None
        self.target = None

        self.output = None

        self.meter = Meter()
