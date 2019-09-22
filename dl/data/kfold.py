from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold


class KFolds():

    def __init__(self, num_splits: int):
        # self._curr_idx = 0
        self.num_splits = num_splits

    def prepare_dataloaders(self, idx: int) -> [DataLoader]:
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.prepare_dataloaders(idx)

    # def __iter__(self):
    #     # self._curr_idx = 0
    #     return iter(self)

    # def __next__(self):
    #     if self._curr_idx == self.num_splits:
    #         raise StopIteration

    #     self._curr_idx += 1

    #     return self[self._curr_idx-1]

    def __len__(self):
        return self.num_splits
