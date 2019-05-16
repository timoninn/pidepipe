"""Embedded (shrinkage methods)
inbuilt(automathic) methods of selecting features (regularization)
- lasso regression
- ridge regression
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Lasso


def calculate_feature_importances(X, y, estimator='tree', refit=False):

    if estimator == 'tree':
        model = ExtraTreesClassifier(n_estimators=250)
        print('Using tree estimator')

    elif estimator == 'linear':
        model = Lasso(alpha=0.5)
        print('Using lasso estimator')

    else:
        model = estimator
        print('Using custom estimator')

    model.fit(X, y)

    try:
        scores = model.feature_importances_

    except AttributeError:
        scores = np.abs(model.coef_)

    return _handle_scores(scores, X.columns)


def _handle_scores(scores, features):

    indices = np.argsort(scores)[::-1]

    print('Feature scores:')

    for f in range(features.shape[0]):
        idx = indices[f]

        print("%d. %s (%f)" % (f + 1, features[idx], scores[idx]))

    return [features[idx] for idx in indices]
