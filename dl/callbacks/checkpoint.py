from pathlib import Path

import torch

from ..core.callback import Callback
from ..core.state import State
from ..core.meter import Monitor


class CheckpointCallback(Callback):

    def __init__(
        self,
        path: str,
        save_n_best: int = 3,
        monitor: str = 'train_loss'
    ):
        self.path = Path(path)
        self.save_n_best = save_n_best

        self.monitor = Monitor(monitor)

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

    def on_epoch_begin(self, state: State):
        pass

    def on_epoch_end(self, state: State):
        if state.meter.is_last_epoch_value_best:
            print('Last value is best')
            self._save_state(state, 'best.pt')

        self._save_state(state, 'last.pt')

    def on_phase_begin(self, state: State):
        pass

    def on_phase_end(self, state: State):
        pass

    def on_batch_begin(self, state: State):
        pass

    def on_batch_end(self, state: State):
        pass
