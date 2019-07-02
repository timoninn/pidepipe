from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from joblib import dump, load
import pandas as pd

# for tree models


def lable_encoding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    attrs: [str] = None
):
    print('Lable encodind')

    attributes = attrs if attrs is not None else X_train.columns.values
    new_attributes = [attr + '_le' for attr in attributes]

    train = X_train[attributes]
    test = X_test[attributes]

    print('Input train shape:', train.shape)
    print('Input test shape:', test.shape)
    print('Attributes:', attributes)

    train_encods = []
    test_encods = []

    for attr, new_attr in zip(attributes, new_attributes):
        encoder = LabelEncoder()

        encoder.fit(train[attr])

        train_encoded = encoder.transform(train[attr])
        test_encoded = encoder.transform(test[attr])

        train_encoded_df = pd.DataFrame(train_encoded, columns=[new_attr])
        test_encoded_df = pd.DataFrame(test_encoded, columns=[new_attr])

        train_encods.append(train_encoded_df)
        test_encods.append(test_encoded_df)

    return pd.concat(train_encods, axis=1), pd.concat(test_encods, axis=1), new_attributes

# for tree and non tree models


def frequency_encoding():
    pass

# for linear models and NN
# interactions of cotegorical features for linear models and KNN


def one_hot_encoding(X, attrs=None, sparse=True, mode='train', path=None):

    data = X if attrs is None else X[attrs]

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
