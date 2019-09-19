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
        metrics
    ):
        model = self.configer.model

        # Add only parameters which require grad. Use for layer freezing.
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = self.configer.get_optimizer(parameters)

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

            log_dir=self.configer.log_dir,
            resume_dir=self.configer.resume_dir
        )

    def eval(self, loader: DataLoader, metrics):
        super().eval(
            model=self.configer.model,
            activation=self.configer.activation,
            loader=loader,
            metrics=metrics,
            resume_dir=self.configer.resume_dir
        )

    def infer(self, loader: DataLoader):
        super().infer(
            model=self.configer.model,
            activation=self.configer.activation,
            loader=loader,
            out_dir=self.configer.config['infer']['out_dir'],
            resume_dir=self.configer.resume_dir,
            one_file_output=self.configer.config['infer']['one_file_output']
        )
