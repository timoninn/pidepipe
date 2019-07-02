import pandas as pd

def split(X: pd.DataFrame, feature_groups: [[str]], names: [str], path):

    assert len(feature_groups) == len(names), 'Groups and names count mismatch'

    for group, name in zip(feature_groups, names):
        print('Split features')
        print(f'Process group: {name}')

        group_data = X[group]
        group_data.to_csv(path / (name + '.csv'), index=False, header=True)