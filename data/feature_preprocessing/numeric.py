from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import pandas as pd


def standard_scale(X, attrs=None, mode='train', path=None):

    _attrs = X.columns if attrs == None else attrs

    data = X if attrs == None else X[_attrs]

    print('Standard scaling')
    print('Mode:', mode)
    print('Input shape:', data.shape)

    if mode == 'train':
        encoder = StandardScaler()
        encoder.fit(data)

        dump(encoder, path)

    elif mode == 'test':
        encoder = load(path)

    result = encoder.transform(data)

    print('Result shape:', result.shape)

    return pd.DataFrame(result, columns=_attrs)
