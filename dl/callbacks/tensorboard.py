from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

from ..core.callback import Callback
from ..core.state import State

# $tensorboard --logdir=runs


class TensorboardCallback(Callback):

    def __init__(
        self,
        log_dir: str,
        comment: str
    ):
        self.log_dir = Path(log_dir)

        self.tb = SummaryWriter(
            log_dir=self.log_dir / comment
        )

        self.is_g = True

    def on_begin(self, state: State):
        print(f'Torch version: {torch.__version__}')
        print(f'Torchvision version: {torchvision.__version__}')

    def on_end(self, state: State):
        self.tb.close()

    def on_epoch_begin(self, state: State):
        pass

    def on_epoch_end(self, state: State):
        pass

    def on_phase_begin(self, state: State):
        pass

    def on_phase_end(self, state: State):

        # metrics_values = state.meter.get_current_epoch_metrics_values(
        #     phase=state.phase
        # )

        # self.tb.add_scalars(
        #     main_tag=state.phase,
        #     tag_scalar_dict=metrics_values,
        #     global_step=state.epoch
        # )

        for name in state.meter.get_all_metric_names(state.phase):
            value = state.meter.get_last_epoch_value(
                phase=state.phase,
                metric_name=name
            )

            self.tb.add_scalar(
                # tag=f'{state.phase}_{name}',
                tag=f'{name}/{state.phase}',
                scalar_value=value,
                global_step=state.epoch
            )

    def on_batch_begin(self, state: State):
        pass

    def on_batch_end(self, state: State):
        pass
