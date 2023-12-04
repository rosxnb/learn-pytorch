import torch
import pandas as pd
import scipy.stats as stats
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


def get_dataframe() -> pd.DataFrame:
    filepath = 'wineQualityRed.csv'
    df = pd.read_csv(filepath, sep=';')

    # remove the outliers
    df = df[df['total sulfur dioxide'] < 200]

    # data normalization
    cols2zscore = df.keys()
    cols2zscore = cols2zscore.drop('quality')
    df[cols2zscore] = df[cols2zscore].apply(stats.zscore)

    # binarize wine quality labels
    df['binaryQuality'] = 0
    df['binaryQuality'][df['quality'] > 5] = 1

    return df


def get_data_and_label() -> tuple(torch.Tensor, torch.Tensor):
    df = get_dataframe()
    dataColumns = df.keys()
    dataColumns = dataColumns.drop('quality')
    dataColumns = dataColumns.drop('binaryQuality')

    data = torch.Tensor(df[dataColumns].values).float()
    label = torch.Tensor(df['binaryQuality'].values).float()
    label = label[:, None]  # make it column vector

    return data, label


def get_datasets(train_size=0.8) -> tuple(TensorDataset, TensorDataset):
    data, label = get_data_and_label()
    x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=train_size)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    return train_dataset, test_dataset
