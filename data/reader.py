import pandas as pd
from scipy.sparse import hstack, load_npz
from pathlib import Path


def read_npz(paths):
    """
    Read a bunch sparse files and concatenate over 1 axis.
    """

    return hstack([load_npz(path) for path in paths]).tocsr()


def read_csv(paths):
    """
    Read a bunch of .csv files and concatenate over 1 axis.
    """

    csvs = [pd.read_csv(f, parse_dates=True) for f in paths]
    _data = pd.concat(csvs, axis=1)

    attrs = _data.columns.values

    return _data, attrs


def write_csv(data, path, index=False, header=True):
    """Write data frame.

    Arguments:
        data {DataFrame} -- Data frame to save.
        path {String} -- Path to save
    """

    data.to_csv(path, index=index, header=header)


def read_all_data(
    features: [str],
    path: Path,
    train_folder: str = 'train',
    test_folder: str = 'test'
):

    train_path = path / train_folder
    test_path = path / test_folder

    X_train, attrs = read_csv([train_path / (f + '.csv') for f in features])
    X_test, _ = read_csv([test_path / (f + '.csv') for f in features])

    return X_train, X_test, attrs
