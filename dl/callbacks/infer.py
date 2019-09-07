from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..core.callback import Callback
from ..core.state import State


class InferCallback(Callback):

    def __init__(
        self,
        out_dir: str
    ):
        self.out_dir = Path(out_dir)
        self.predictions = []

    def on_epoch_begin(self, state: State):
        pass

    def on_epoch_end(self, state: State):
        pass

    def on_phase_begin(self, state: State):
        pass

    def on_phase_end(self, state: State):
        np.save(self.out_dir / 'infer.npy', self.predictions)

    def on_batch_begin(self, state: State):
        pass

    def on_batch_end(self, state: State):
        for arr in state.output.cpu().numpy():
            self.predictions.append(arr)
