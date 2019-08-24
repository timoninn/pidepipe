from ..core.callback import Callback
from ..core.state import State


class LoggingCallback(Callback):

    def on_epoch_begin(self, state: State):
        print(f'{state.epoch + 1} / {state.num_epochs} Begin epoch')

    def on_epoch_end(self, state: State):
        print(f'{state.epoch + 1} / {state.num_epochs} End epoch')

    def on_phase_begin(self, state: State):
        print(f'Begin {state.phase} phase')

    def on_phase_end(self, state: State):
        print(f'End {state.phase} phase')

    def on_batch_begin(self, state: State):
        print('Batch begin')

    def on_batch_end(self, state: State):
        print('Batch end')
