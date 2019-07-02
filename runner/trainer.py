import lightgbm as lgb
import pandas as pd
import numpy as np


class Trainer():

    def __init__(
        self,
        model: lgb.sklearn.LGBMRegressor = None
    ):
        self.model = model

    def train(
        self,
        model: lgb.sklearn.LGBMRegressor,
        eval_metric,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        folds: None,
        logdir: str,
        save_path: str = None,
        verbose: bool = False,
        early_stopping_rounds=200
    ):
        result = {}
        oof = np.zeros(X.shape[0])
        valid_scores = []
        predictions = np.zeros((X_test.shape[0]))

        for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X)):
            print(f'Fold {fold_n + 1} started')

            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]

            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric='mae',
                verbose=verbose,
                early_stopping_rounds=early_stopping_rounds
            )

            y_pred_valid = model.predict(X_valid)

            y_pred_test = model.predict(X_test)
            predictions += y_pred_test

            oof[valid_idx] = y_pred_valid.reshape(-1)

            score = eval_metric(y_valid, y_pred_valid)
            valid_scores.append(score)

        predictions /= folds.n_splits

        print('CV mean score: {0:.4f}, std: {1:.4f}'.format(
            np.mean(valid_scores), np.std(valid_scores)))

        result['oof'] = oof
        result['valid_scores'] = valid_scores
        result['predctions'] = pd.DataFrame(predictions, columns=['scalar_coupling_constant'])

        return result

    def infer(
        self,
        X: pd.DataFrame
    ):
        y_pred = self.model.predict(X)

        return y_pred
