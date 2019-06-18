import numpy as np
from scipy import stats


def remove_outliers_z(X, y=None, attrs=None, theshold=3):
    """Remove outliers using z-score.
    Works correctly only without constant columns.

    Arguments:
        X {DataFrame} -- Data frame to clean.

    Keyword Arguments:
        y {DataFrame} -- Rows deleted as in X. (default: {None})
        attrs {Array} -- Attributes to check outliers. (default: {None})
        theshold {int} -- z-score treshold (default: {3})

    Returns:
        tuple -- Cleaned X, y.
    """

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
    """ Check that examples count equal on rhs and lsh.

    Arguments:
        lhs {DataFrame}

    Keyword Arguments:
        rhs {DataFrame} -- (default: {None})
    """

    if rhs is not None:
        assert lhs.shape[0] == rhs.shape[0], 'Rhs and lhs number of examples mismatch'
