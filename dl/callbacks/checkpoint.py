from ..core.callback import Callback
from ..core.state import State


class CheckpointCallback(Callback):

    def __init__(
        self,
        path: str,
        save_n_best: int = 3,
        monitor: str = 'train_loss',
    ):
        self.monitor = Monitor(monitor)

    # def _save_state(self, epoch: int, epoch_loss: float, name: str):

    #     state = {
    #         "epoch": epoch,
    #         "loss": epoch_loss,
    #         "model_state_dict": self.model.state_dict(),
    #         "optimizer_state_dict": self.optimizer.state_dict(),
    #     }

    #     torch.save(state, self.log_dir + name)
