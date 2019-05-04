import numpy as np
import pandas as pd
from scipy import stats


# work correctly only without constant columns
def remove_outliers_z(X, y=None, attrs=None, theshold=3):

    check_examples_count(X, y)

    print('Input data shape:' + str(X.shape))

    data = X if attrs is None else X[attrs]

    z = np.abs(stats.zscore(data))

    idxs = (z < theshold).all(axis=1)

    result_X = X[idxs]
    result_y = y[idxs]

    print('Removed outliers data shape:' + str(result_X.shape))

    return result_X, result_y


def check_examples_count(lhs, rhs=None):
    """Check that examples count equal on rhs and lsh.

    :lhs: TODO
    :rhs: TODO
    :returns: TODO

    """

    if rhs is not None:
        assert lhs.shape[0] == rhs.shape[0], 'Rhs and lhs number of examples mismatch'
