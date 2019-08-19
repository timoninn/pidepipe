
from typing import Dict, Any, abstractmethod

State = Dict[str, Any]


class Callback:

    def on_fold_begin(self, state: State):
        pass

    def on_fold_end(self, state: State):
        pass

    def on_epoch_begin(self, state: State):
        pass

    def on_epoch_end(self, state: State):
        pass

    def on_batch_begin(self, state: State):
        pass

    def on_batch_end(self, state: State):
        pass

    def on_train_batch_begin(self, state: State):
        pass

    def on_train_batch_end(self, state: State):
        pass

    def on_valid_batch_begin(self, state: State):
        pass

    def on_valid_batch_end(self, state: State):
        pass

    def on_infer_batch_begin(self, state: State):
        pass

    def on_infer_batch_end(self, state: State):
        pass
