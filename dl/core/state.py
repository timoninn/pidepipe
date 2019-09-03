from typing import Dict, Any, abstractmethod
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from .meter import Meter


class State:

    def __init__(
        self,

        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: _LRScheduler,
        criterion: nn.Module,

        epoch: int,
        num_epochs: int
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.epoch: int = epoch
        self.num_epochs: int = num_epochs

        self.phase: str = None
        self.stop_train: bool = False

        self.batch_idx: int = None
        self.num_batches: int = None

        self.input = None
        self.target = None

        self.output = None

        self.meter = Meter()

    @property
    def is_train_phase(self) -> bool:
        return (self.phase == 'train')

    @property
    def is_infer_phase(self) -> bool:
        return (self.phase == 'infer')

