from typing import Dict, Any, abstractmethod

from .state import State


class Callback:

    def on_begin(self, state: State):
        pass

    def on_end(self, state: State):
        pass

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
