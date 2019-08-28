from ..core.callback import Callback
from ..core.state import State

from tqdm import tqdm


class LoggingCallback(Callback):

    def __init__(self):
        self.tqdm = None

    def on_begin(self, state: State):
        pass

    def on_end(self, state: State):
        pass

    def on_epoch_begin(self, state: State):
        pass

    def on_epoch_end(self, state: State):
        pass

    def on_phase_begin(self, state: State):
        self.tqdm = tqdm(
            desc=f'Epoch {state.epoch + 1} / {state.num_epochs} Phase {state.phase}',
            total=state.num_batches,
            leave=True,
            ncols=0
        )

    def on_phase_end(self, state: State):
        self.tqdm.close()
        self.tqdm = None

        # for name in state.meter.get_all_metric_names(state.phase):
        #     value = state.meter.get_last_epoch_value(
        #         phase=state.phase,
        #         metric_name=name
        #     )

        #     print(f'{state.phase}_{name}: {value}')

    def on_batch_begin(self, state: State):
        pass

    def on_batch_end(self, state: State):
        metrics_values = state.meter.get_current_epoch_metrics_values(
            phase=state.phase
        )

        self.tqdm.set_postfix(
            { k: f'{v:3.8f}' for k, v in sorted(metrics_values.items()) }
        )

        self.tqdm.update()
