from typing_extensions import override
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F


def create_data(nPerCluster=100):
    blur = 1
    a = [2, 2]
    b = [2, 7]

    x = [a[0] + np.random.randn(nPerCluster) * blur, a[1] + np.random.randn(nPerCluster) * blur]
    y = [b[0] + np.random.randn(nPerCluster) * blur, b[1] + np.random.randn(nPerCluster) * blur]
    
    data_np = np.hstack((x, y)).T
    labels_np = np.vstack((np.zeros((nPerCluster, 1)), np.ones((nPerCluster, 1))))
    data = torch.tensor(data_np, dtype=torch.float)
    labels = torch.tensor(labels_np, dtype=torch.float)

    return data, labels


class QwertyClassifier(nn.Module):
    # model = nn.Sequential(
    #     nn.Linear(2, 12),
    #     nn.ReLU(),
    #     nn.Linear(12, 1),
    #     nn.ReLU(),
    #     nn.Linear(1, 1)
    # )
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 12)
        self.hidden_layer = nn.Linear(12, 1)
        self.output_layer = nn.Linear(1, 1)

    @override
    def forward(self, x):
        x = F.relu( self.input_layer(x) )
        x = F.relu( self.hidden_layer(x) )
        x = self.output_layer(x)
        return x



def create_and_train_model(lr, trainEpoches=500):
    data, labels = create_data()

    model = QwertyClassifier()

    lossFunc = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    trainLosses = np.zeros(trainEpoches)
    for i in range(trainEpoches):
        yhat = model(data)
        loss = lossFunc(yhat, labels)
        trainLosses[i] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = model(data)
    trainAccuracy = 100 * torch.mean( ((predictions > 0) == labels).float() )

    return trainLosses, predictions, trainAccuracy


if __name__ == '__main__':
    trainEpoches = 500
    learningRates = np.linspace(0.001, 0.1, 50)
    losses = np.zeros((len(learningRates), trainEpoches))
    accuracies  = np.zeros(len(learningRates))

    for lri, lr in enumerate(learningRates):
        trainLosses, predictions, trainAccuracy = create_and_train_model(lr, trainEpoches)
        losses[lri, :] = trainLosses
        accuracies[lri] = trainAccuracy

    fig, ax = plt.subplots(1, 2, figsize=(16, 5), num='Multilayer QWERTY')
    ax[0].set_title('Accuracy vs LR')
    ax[0].plot(learningRates, accuracies, 's-')
    ax[0].set_xlabel('Leanring Rate')
    ax[0].set_ylabel('Accuracy')

    ax[1].set_title('Losses per LR')
    ax[1].plot(losses.T)
    ax[1].set_ylabel('Loss')
    plt.show()

