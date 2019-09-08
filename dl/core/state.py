from typing import Dict, Tuple, Any, abstractmethod
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
        self.stop_running: bool = False

        self.batch_idx: int = None
        self.num_batches: int = None
        self.batch: Any = None

        self.output = None

        self.device: torch.device = None

        self.meter = Meter()

    @property
    def input(self) -> torch.Tensor:
        return self.batch[0]

    @property
    def target(self) -> torch.Tensor:
        return self.batch[1]

    @property
    def is_train_phase(self) -> bool:
        return (self.phase == 'train')

    @property
    def is_infer_phase(self) -> bool:
        return (self.phase == 'infer')
