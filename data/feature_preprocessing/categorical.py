from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load


# for tree models
def lable_encoding():
    pass

# for tree and non tree models


def frequency_encoding():
    pass

# for linear models and NN
# interactions of cotegorical features for linear models and KNN


def one_hot_encoding(X, attrs=None, sparse=True, mode='train', path=None):

    data = X if attrs == None else X[attrs]

    print('One-hot encoding')
    print('Mode:', mode)
    print('Input shape:', data.shape)

    if mode == 'train':
        encoder = OneHotEncoder(
            sparse=sparse,
            handle_unknown='ignore'
        )
        encoder.fit(data)

        dump(encoder, path)

    elif mode == 'test':
        encoder = load(path)

    result = encoder.transform(data)

    print('Result shape:', result.shape)

    return result
