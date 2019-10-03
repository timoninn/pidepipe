import math

import numpy as np
from torch.utils.data import Sampler
from pidepipe.dl.utils.experiment import set_global_seed


def get_num_after_point(x: float) -> int:
    balance_int = str(x)
    if not '.' in balance_int:
        return 0

    return len(balance_int) - balance_int.index('.') - 1


def gcd(arr: [int]) -> int:
    result = arr[0]
    for i in arr[1:]:
        result = math.gcd(result, i)

    return result


class BalanceSampler(Sampler):

    def __init__(
        self,
        labels: [int],
        balance: [float],
        shuffle=True,
        seed: int = None
    ):
        labels = np.array(labels)
        balance = np.array(balance)

        assert np.sum(balance) == 1, 'Sum of balances should be equal to 1'

        samples_per_class = np.array([
            np.sum(class_idx == labels) for class_idx in np.unique(labels)
        ])

        assert balance.shape == samples_per_class.shape, f'Number of balances ({balance.shape[0]}) should be equal to number of classes ({samples_per_class.shape[0]})'

        # Calculate min number of samples for balance.
        num_after_point_vec = np.vectorize(get_num_after_point)(balance)
        num_after_point = np.max(num_after_point_vec)
        balance_int = balance * 10**num_after_point
        balance_int = balance_int.astype(np.int64)
        min_balance_int = balance_int // gcd(balance_int)

        # Calculate max number of samples for balance.
        count = 0
        while (samples_per_class - min_balance_int >= 0).all():
            samples_per_class -= min_balance_int
            count += 1

        self.samples_counts = count * min_balance_int
        self.len = np.sum(self.samples_counts)
        self.labels_idxs = [
            np.arange(labels.size)[labels == label].tolist() for label in np.unique(labels)
        ]
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        indices = []
        for label_idxs, samples_count in zip(self.labels_idxs, self.samples_counts):
            replace = samples_count > len(label_idxs)

            idxs = np.random.choice(
                label_idxs,
                size=samples_count,
                replace=replace
            )

            indices.extend(idxs.tolist())

        assert (len(indices) == self.len)

        if self.shuffle:
            np.random.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return self.len


if __name__ == "__main__":
    set_global_seed(9)

    a = [0] * 1
    b = [1] * 2
    c = [2] * 2000
    d = [3] * 7

    sampler = BalanceSampler(
        labels=a + b + c + d,
        balance=[0.1, 0.2, 0, 0.7]
    )

    for i in sampler:
        print(i)

    # idxs = next(iter(sampler))

    # print(idxs)
