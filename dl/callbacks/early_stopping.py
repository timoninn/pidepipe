from ..core.callback import Callback
from ..core.state import State
from ..core.meter import Monitor


class EarlyStoppingCallback(Callback):

    def __init__(
        self,
        monitor: str = 'val_loss',
        minimize: bool = True,
        patience: int = 3,
        delta: float = 1e-6
    ):
        assert delta >= 0, 'Delta should be greater than or equal to zero'
        assert patience > 0, 'Patience should be greater than zero'

        self.monitor = Monitor(monitor)
        self.minimize = minimize
        self.patience = patience
        self.delta = delta

        self.best_value = float('+inf') if minimize else float('-inf')
        self.num_fails = 0

    def _check_best_value(self, value: float) -> bool:
        if self.minimize:
            return (self.best_value - value) >= self.delta
        else:
            return (value - self.best_value) >= self.delta

    def _check_early_stopping(self, state: State):
        value = state.meter.get_last_epoch_value(
            phase=self.monitor.phase,
            metric_name=self.monitor.metric_name
        )

        if self._check_best_value(value):
            self.num_fails = 0
            self.best_value = value
        else:
            self.num_fails += 1

        if self.num_fails == self.patience:
            print('Early stopping')
            state.stop_train = True

    def on_epoch_end(self, state: State):
        if state.is_train_phase:
            self._check_early_stopping(state)
