
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


class LoggingCallback(Callback):

    def on_epoch_begin(self, state: State):
        print(f'{state.epoch} / {state.num_epoch} Begin epoch ({state.phase})')

    def on_epoch_end(self, state: State):
        print(f'{state.epoch} / {state.num_epoch} End epoch ({state.phase})')

    def on_phase_begin(self, state: State):
        print(f'Begin {state.phase} phase')

    def on_phase_end(self, state: State):
        print(f'End {state.phase} phase')

    def on_batch_begin(self, state: State):
        print('Batch begin')

    def on_batch_end(self, state: State):
        print('Batch end')


class EarlyStoppingCallback(Callback):

    def __init__(
        self,
        monitor: str = 'val_loss',  # train_dice
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
        self.monitor = Monitor(monitor)


class Monitor:

    def __init__(self, str: str):
        components = str.split('_')

        self.str = str
        self.phase = components[0]
        self.metric_name = components[1]
