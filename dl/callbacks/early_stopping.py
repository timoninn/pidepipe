from ..core.callback import Callback
from ..core.state import State


class EarlyStoppingCallback(Callback):

    def __init__(
        self,
        monitor: str = 'val_loss',  # train_dice
        patience: int = 2
    ):
        pass
