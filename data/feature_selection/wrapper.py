"""Wrapper
evaluate predictive power of variables combinations to find the best subset
- subset selection (2^k tries)
- stepwise selection
  - forward selection (1 + k(k+1)/2)
  - backward selection (1 + k(k+1)/2)
  - hybrid
"""
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def select_sequentially(X, y, estimator, k_features="best", forward=True, floating=False, scoring=None, cv=None):

    selector = SFS(
        estimator=estimator,
        k_features=k_features,
        forward=forward,
        floating=floating,
        verbose=2,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )

    direction = "forward" if forward else "backward"
    floating = "floating" if floating else "not floating"

    print("\nSequentual %s %s feature selection\n" % (direction, floating))

    selector.fit(X, y)

    print('\n\nBest score: %.6f' % selector.k_score_)
    print('Best subset (indices):', selector.k_feature_idx_)
    print('Best subset (names):', selector.k_feature_names_, "\n")

    return selector.k_feature_idx_, selector.k_feature_names_
