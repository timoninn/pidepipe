import logging
from tqdm import tqdm
from pathlib import Path

from ..core.callback import Callback
from ..core.state import State


class ConsoleLoggingCallback(Callback):

    def __init__(self):
        self.tqdm = None

    def _update_tqdm(self, metrics_values):
        self.tqdm.set_postfix(
            {k: f'{v:3.8f}' for k, v in sorted(metrics_values.items())}
        )

        self.tqdm.update()

    def on_phase_begin(self, state: State):
        self.tqdm = tqdm(
            desc=f'Epoch {state.epoch} / {state.num_epochs} Phase {state.phase}',
            total=state.num_batches,
            leave=True,
            ncols=0
        )

    def on_batch_end(self, state: State):
        metrics_values = state.meter.get_all_last_batch_values(
            phase=state.phase
        )

        self._update_tqdm(metrics_values)

    def on_phase_end(self, state: State):
        metrics_values = state.meter.get_current_epoch_metrics_values(
            phase=state.phase
        )

        self._update_tqdm(metrics_values)

        self.tqdm.close()
        self.tqdm = None


class FileLoggingCallback(Callback):

    def __init__(
        self,
        log_dir: str
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(self.log_dir / 'log.txt')
        handler.setLevel(logging.INFO)

        logger = logging.getLogger(name='state_logger')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        self.logger = logger

    def on_phase_end(self, state: State):
        metrics_values = state.meter.get_current_epoch_metrics_values(
            phase=state.phase
        )

        msg = f'Epoch {state.epoch} / {state.num_epochs} Phase: {state.phase} Metrics: {metrics_values}'
        self.logger.info(msg)
