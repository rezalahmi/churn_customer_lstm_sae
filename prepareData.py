import pandas as pd
import numpy as np


def load_data(dataset_path):
    return pd.read_csv(dataset_path)


def split_train_test(rate, dataset_path):
    data = load_data(dataset_path)
    data = data.iloc[:, 2:]
    Xtrain = data.loc[:100000 * rate - 1, :]
    Xtest = data.loc[100000 * rate:, :]
    Xtrain = np.asarray(Xtrain.to_numpy())
    Xtrain = Xtrain.reshape((Xtrain.shape[0], 1, Xtrain.shape[1]))
    Xtest = np.asarray(Xtest.to_numpy())
    Xtest = Xtest.reshape((Xtest.shape[0], 1, Xtest.shape[1]))
    return Xtrain, Xtest
