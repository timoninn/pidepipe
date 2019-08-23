
from typing import Dict, Any, abstractmethod

from state import State

class Callback:

    def on_epoch_begin(self, state: State):
        pass

    def on_epoch_end(self, state: State):
        pass

    def on_phase_begin(self, state: State):
        pass

    def on_phase_end(self, state: State):
        pass

    def on_batch_begin(self, state: State):
        pass

    def on_batch_end(self, state: State):
        pass


class EarlyStoppingCallback(Callback):

    def __init__(
        self,
        monitor: str = 'val_loss', # train_dice
        patience: int = 2
    ):
        pass

class CheckpointCallback(Callback):

    def __init__(
        self,
        path: str,
        save_n_best: int = 3,
        monitor: str = 'train_loss',
    ):
        pass