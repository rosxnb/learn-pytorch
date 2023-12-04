import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


class SugarPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(11, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.output(x)


def train_model(train_loader, test_loader, epochs):
    model = SugarPredictor()
    lossFunc = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainAcc = []
    testAcc = []
    trainLosses = []
    testLosses = []
    for epoch in range(epochs):
        model.train()

        batchAcc = []
        batchLosses = []
        for data, label in train_loader:
            predictions = model(data)
            loss = lossFunc(predictions, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchLosses.append(loss.item())
            r = np.corrcoef(predictions.detach().T, label.T)[0, 1]
            batchAcc.append(r)

        trainAcc.append(np.mean(batchAcc))
        trainLosses.append(np.mean(batchLosses))

        model.eval()

        data, label = next(iter(test_loader))
        with torch.no_grad():
            predictions = model(data)

        testLosses.append(lossFunc(predictions, label).item())
        r = np.corrcoef(predictions.detach().T, label.T)[0, 1]
        testAcc.append(r)

    return trainLosses, testLosses, trainAcc, testAcc, model


def main():
    filepath = 'wineQualityRed.csv'
    df = pd.read_csv(filepath, sep=';')

    df = df[df['total sulfur dioxide'] < 200]  # remove outliers
    df = df.apply(stats.zscore)  # standardize whole dataset

    data = torch.Tensor(df.drop('residual sugar', axis='columns').values)
    label = torch.Tensor(df['residual sugar'].values)
    label = label[:, None]

    x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.9)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.tensors[0].shape[0])

    epochs = 1000
    train_losses, test_losses, train_accuracies, test_accuracies, model = train_model(train_loader, test_loader, epochs)

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))

    ax[0].plot(train_losses, label='Train loss')
    ax[0].plot(test_losses, label='Test loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('MSE loss')
    ax[0].set_title('Train/Test Losses')
    ax[0].legend()

    ax[1].plot(train_accuracies, label='Train accuracy')
    ax[1].plot(test_accuracies, label='Test accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy (mean correlation)')
    ax[1].set_title('Train/Test accuracy')
    ax[1].legend()

    training_predictions = model(x_train)
    testing_predictions = model(x_test)
    ax[2].plot(training_predictions.detach(), y_train, 'ro')
    ax[2].plot(testing_predictions.detach(), y_test, 'b^')
    ax[2].set_xlabel('Model sugar predictions')
    ax[2].set_ylabel('True sugar values')
    ax[2].set_title('Predictions vs True Value')

    train_r = np.corrcoef(training_predictions.detach().T, y_train.T)[0, 1]
    test_r = np.corrcoef(testing_predictions.detach().T, y_test.T)[0, 1]
    ax[2].legend([f'Train r={train_r:.3f}', f'Test r={test_r:.3f}'])

    plt.show()

    # visualize the correlation matrix
    plt.figure(figsize=(15, 9))
    heatmap = sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, cmap='BrBG')
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 14})
    plt.show()


if __name__ == '__main__':
    main()
