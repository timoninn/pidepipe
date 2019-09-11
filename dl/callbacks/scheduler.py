from typing import Dict

from torch.optim import lr_scheduler

from ..core.callback import Callback
from ..core.state import State
from ..core.meter import Monitor

Scheduler = lr_scheduler._LRScheduler


class SchedulerCallback(Callback):

    def __init__(
        self,
        scheduler: Scheduler,
        monitor: str = 'train_loss'
    ):
        self.scheduler = scheduler
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

    def _get_lr(self, scheduler: Scheduler):
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            return None
        else:
            return scheduler.get_lr()

    def on_begin(self, state: State):
        state.scheduler = self.scheduler

    def on_epoch_begin(self, state: State):
        state.lr = self._get_lr(state.scheduler)

    def on_epoch_end(self, state: State):
        # Where to make scheduler step on epoch begin or end ?
        self._step(state)
