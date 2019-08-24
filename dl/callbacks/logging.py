from ..core.callback import Callback
from ..core.state import State

from tqdm import tqdm


class LoggingCallback(Callback):

    def __init__(self):
        self.tqdm = None

    def on_epoch_begin(self, state: State):
        pass

    def on_epoch_end(self, state: State):
        pass

    def on_phase_begin(self, state: State):
        self.tqdm = tqdm(
            desc=f'Epoch {state.epoch + 1} / {state.num_epochs} Phase {state.phase}',
            total=state.num_batches
        )

    def on_phase_end(self, state: State):
        self.tqdm.close()
        self.tqdm = None

        loss = state.meter.get_last_epoch_value(
            phase=state.phase,
            metric_name='loss'
        )

        print(f'Loss: {loss}\n')

    def on_batch_begin(self, state: State):
        pass

    def on_batch_end(self, state: State):
        self.tqdm.update()
