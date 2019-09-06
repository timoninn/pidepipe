from typing import Dict

from torch.optim import lr_scheduler

from ..core.callback import Callback
from ..core.state import State
from ..core.meter import Monitor


class SchedulerCallback(Callback):

    def __init__(
        self,
        monitor: str = 'train_loss'
    ):
        self.monitor = Monitor(monitor)

    def _step(self, state: State):
        if isinstance(state.scheduler, lr_scheduler.ReduceLROnPlateau):
            metric_value = state.meter.get_last_epoch_value(
                phase=self.monitor.phase,
                metric_name=self.monitor.metric_name
            )

            state.scheduler.step(metric_value, epoch=state.epoch)
        else:
            state.scheduler.step(epoch=state.epoch)

    def on_epoch_begin(self, state: State):
        pass

    def on_epoch_end(self, state: State):
        self._step(state)

    def on_phase_begin(self, state: State):
        pass

    def on_phase_end(self, state: State):
        pass

    def on_batch_begin(self, state: State):
        pass

    def on_batch_end(self, state: State):
        pass
