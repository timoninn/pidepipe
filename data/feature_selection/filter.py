"""Filter
evaluate predictive power of each individual feature (with respect to target)

- correlation with target variable
- domain knowledge
- chi square test (for categorical features)
- information gain
"""

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

import numpy as np


def select_features(X, y):

    selector = SelectKBest(score_func=chi2, k=5)

    selector.fit(X, y)

    scores = selector.scores_
    features = X.columns.values

    indices = np.flip(np.argsort(scores), axis=0)

    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" %
              (f + 1, features[indices[f]], scores[indices[f]]))
