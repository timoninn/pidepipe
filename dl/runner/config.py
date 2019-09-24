from typing import Dict

from torch import nn
from torch.utils.data import DataLoader

from .train import TrainRunner
from ..utils.configer import Configer
from ..utils.experiment import set_global_seed


class ConfigRunner(TrainRunner):

    def __init__(self, config_path: str):
        self.configer = Configer(config_path)
        set_global_seed(self.configer.seed)

    def train(
        self,

        train_loader: DataLoader,
        valid_loader: DataLoader,

        metrics: Dict[str, nn.Module],

        log_dir: str = None,
        resume_dir: str = None,
    ):
        model = self.configer.model

        # Add only parameters which require grad. Use for layer freezing.
        # trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())

        trainable_parameters = model.trainable_parameters()
        optimizer = self.configer.get_optimizer(trainable_parameters)

        super().train(
            model=model,
            activation=self.configer.activation,

            criterion=self.configer.criterion,
            optimizer=optimizer,
            scheduler=self.configer.get_scheduler(optimizer),

            train_loader=train_loader,
            valid_loader=valid_loader,

            num_epochs=self.configer.config['train']['num_epochs'],
            early_stopping=self.configer.config['train']['early_stopping'],

            metrics=metrics,
            monitor=self.configer.config['train']['monitor'],

            log_dir=log_dir,
            resume_dir=resume_dir
        )

    def eval(
        self,
        loader: DataLoader,
        metrics: Dict[str, nn.Module],
        resume_dir: str = None
    ):
        super().eval(
            model=self.configer.model,
            activation=self.configer.activation,
            loader=loader,
            metrics=metrics,
            resume_dir=resume_dir
        )

    def infer(
        self,
        loader: DataLoader,
        out_dir: str,
        resume_dir: str,
        one_file_output: bool = False
    ):
        super().infer(
            model=self.configer.model,
            activation=self.configer.activation,
            loader=loader,
            out_dir=out_dir,
            resume_dir=resume_dir,
            one_file_output=one_file_output
        )
