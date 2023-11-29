# We have been trying to solve QWERTY with non-linear models
# in past few python scipts.
# Now let's try to solve with a linear model since the problem
# itself is a linear problem

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


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


def visualize_data():
    data, labels = create_data()
    plt.title('Generated Data Visualization')
    plt.plot(data[np.where(labels == 0)[0], 0], data[np.where(labels == 0)[0], 1], 'bs', label='label 0')
    plt.plot(data[np.where(labels == 1)[0], 0], data[np.where(labels == 1)[0], 1], 'rs', label='label 1')
    plt.show()


def create_and_train_model(lr, trainEpoches=500):
    model = nn.Sequential(
        nn.Linear(2, 12),
        nn.Linear(12, 1),
        nn.Linear(1, 1)
    )

    lossFunc = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    data, labels = create_data()
    trainLosses = np.zeros(trainEpoches)

    for i in range(trainEpoches):
        yhat = model(data)
        loss = lossFunc(yhat, labels)
        trainLosses[i] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = model(data)
    trainAccuracy = torch.mean(((predictions > 0) == labels).float()) * 100
    
    return trainLosses, predictions, trainAccuracy


if __name__ == '__main__':
    # visualize_data()
    trainEpoches = 500
    learningRates = np.linspace(0.001, 0.1, 50)
    losses = np.zeros((len(learningRates), trainEpoches))
    accuracies = []

    for lri, lr in enumerate(learningRates):
        trainLosses, predictions, trainAccuracy = create_and_train_model(lr, trainEpoches)
        losses[lri, :] = trainLosses
        accuracies.append(trainAccuracy)

    fig, ax = plt.subplots(1, 2, figsize=(16, 5), num='QWERTY Linear Classification')
    ax[0].set_title('Accuracy vs LR')
    ax[0].plot(learningRates, accuracies, 'rs-')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Learning Rate')
    ax[1].set_title('Losses per LR')
    ax[1].plot(losses.T)
    ax[1].set_ylabel('Loss')
    plt.show()
