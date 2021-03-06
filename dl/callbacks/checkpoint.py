from typing import Dict, Any
from pathlib import Path

import torch

from ..core.callback import Callback
from ..core.state import State
from ..core.meter import Monitor


class SaveCheckpointCallback(Callback):

    def __init__(
        self,
        path: str,
        monitor: str = 'train_loss',
        minimize: bool = True
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.minimize = minimize
        self.monitor = Monitor(monitor)

    def _is_last_value_best(self, state: State):
        return state.meter.is_last_epoch_value_best(
            phase=self.monitor.phase,
            metric_name=self.monitor.metric_name,
            minimize=self.minimize
        )

    def _save_state(self, state: State, name: str):
        last_loss = state.meter.get_last_epoch_value(
            phase=self.monitor.phase,
            metric_name=self.monitor.metric_name
        )

        state = {
            "epoch": state.epoch,
            "train_loss": last_loss,
            "model_state_dict": state.model.state_dict(),
            "optimizer_state_dict": state.optimizer.state_dict()
        }

        torch.save(state, self.path / name)

    def on_epoch_end(self, state: State):
        if self._is_last_value_best(state):
            print('Saving state')
            self._save_state(state, 'best.pt')

        self._save_state(state, 'last.pt')


class LoadCheckpointCallback(Callback):

    def __init__(
        self,
        path: str
    ):
        self.path = Path(path)

    def _load(self) -> Dict[str, Any]:
        print(f'Load checkpoint at path: {self.path}')
        return torch.load(self.path)

    def on_begin(self, state: State):
        checkpoint = self._load()

        state.model.load_state_dict(checkpoint['model_state_dict'])

        # if state.optimizer is not None:
        # state.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Fix optimizer loading 1
        # for state in state.optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda()

        # Fix optimizer loading 2
        # Model to device before optimizer creation
