from typing import Dict, List
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader, Dataset

from .kfold import KFolds
from ..runner.config import ConfigRunner


class KFoldsRunner():

    def __init__(
        self,
        kfolds: KFolds,
        exp_config: str,
        metrics: Dict[str, nn.Module],
        log_dir: str,
        resume_dir: str
    ):
        self.kfolds = kfolds
        self.metrics = metrics
        self.log_path = Path(log_dir)
        self.resume_dir = resume_dir

        self.runner = ConfigRunner(config_path=exp_config)

    def run(self):
        for idx, loaders in enumerate(self.kfolds):
            self.train_fold(idx, loaders)

    def train_fold(self, idx: int, loaders: [DataLoader]):
        fold_log_dir = self.log_path / f'fold_{idx}'

        self.runner.train(
            train_loader=loaders[0],
            valid_loader=loaders[1],
            metrics=self.metrics,
            log_dir=fold_log_dir,
            resume_dir=self.resume_dir
        )
