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
        self.resume_dir = Path(resume_dir)

        self.runner = ConfigRunner(config_path=exp_config)

    def _get_fold_log_dir(self, idx: int):
        return self.log_path / f'fold_{idx}'

    def _get_fold_resume_dir(self, idx: int):
        return self.resume_dir / f'fold_{idx}'

    def _train_fold(self, idx: int, loaders: [DataLoader]):
        print(f'Train fold {idx}')

        self.runner.train(
            train_loader=loaders[0],
            valid_loader=loaders[1],
            metrics=self.metrics,
            log_dir=self._get_fold_log_dir(idx),
            resume_dir=self._get_fold_resume_dir(idx)
        )

    def _eval_fold(self, idx: int, loaders: [DataLoader]):
        print(f'Evaluate fold {idx}')

        print('Train set:')
        self.runner.eval(
            loader=loaders[0],
            metrics=self.metrics,
            resume_dir=self._get_fold_resume_dir(idx)
        )

        print('Valid set:')
        self.runner.eval(
            loader=loaders[1],
            metrics=self.metrics,
            resume_dir=self._get_fold_resume_dir(idx)
        )

    def train(self):
        for idx, loaders in enumerate(self.kfolds):
            self._train_fold(idx, loaders)
            break

    def eval(self):
        for idx, loaders in enumerate(self.kfolds):
            self._eval_fold(idx, loaders)
            break
