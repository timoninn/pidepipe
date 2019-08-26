from ..core.callback import Callback
from ..core.state import State


class EarlyStoppingCallback(Callback):

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 3
    ):
        pass
