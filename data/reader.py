import pandas as pd
from scipy.sparse import csr_matrix, hstack


def read_npz(paths):
    """
    Read a bunch sparse files and concatenate over 1 axis.
    """
    return hstack([sparse.load_npz(path) for path in paths]).tocsr()


def read_csv(paths):
    """
    Read a bunch of .csv files and concatenate over 1 axis.
    """
    _data = pd.concat([pd.read_csv(f, parse_dates=True)
                       for f in paths], axis=1)
    attrs = _data.columns.values

    return _data, attrs


def write_csv(data, path):

    data.to_csv(path, index=False, header=True)
