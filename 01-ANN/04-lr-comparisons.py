# Parametric Expriment: Observe the effect of learning rates
# Data used for exprimentation is Qwerties data from last program (03)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def create_data(nPerCluster=100):
    # generate data
    a = [1, 1]
    b = [5, 1]
    blur = 1/2

    x = [a[0] + np.random.randn(nPerCluster) * blur, a[1] + np.random.randn(nPerCluster) * blur]
    y = [b[0] + np.random.randn(nPerCluster) * blur, b[1] + np.random.randn(nPerCluster) * blur]

    labels_np = np.vstack((np.zeros((nPerCluster, 1)), np.ones((nPerCluster, 1))))
    data_np = np.hstack((x, y)).T
    data = torch.tensor(data_np, dtype=torch.float)
    labels = torch.tensor(labels_np, dtype=torch.float)

    return data, labels


def create_model(lr):
    model = nn.Sequential(
        nn.Linear(2, 1),
        nn.ReLU(),
        nn.Linear(1, 1)
    ) # notice we are not using sigmoid as final activation

    lossFunc = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, lossFunc, optimizer


def create_and_train_model(lr, data, labels, trainEpoches):
    model, lossFunc, optimizer = create_model(lr)

    trainLosses = np.zeros(trainEpoches)
    for epochi in range(trainEpoches):
        yhat = model(data)
        loss = lossFunc(yhat, labels)
        trainLosses[epochi] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = model(data)
    trainAccuracy = 100 * torch.mean( ((predictions > 0) == labels).float() )

    return predictions, trainLosses, trainAccuracy


if __name__ == '__main__':
    data, labels = create_data()
    trainEpoches = 1000
    learningRates = np.linspace(0.001, 0.1, 40)

    # result spaces
    accuracyByLr = []
    losses = np.zeros( (len(learningRates), trainEpoches))

    # expriment loop
    for lri, lr in enumerate(learningRates):
        predictions, trainLosses, trainAccuracy = create_and_train_model(lr, data, labels, trainEpoches)
        accuracyByLr.append(trainAccuracy)
        losses[lri, :] = trainLosses

    # visualize
    fig, ax = plt.subplots(1, 3, figsize=(20, 5), num='Learing Rates Comparision')
    ax[0].plot(learningRates, accuracyByLr, 'bs-')
    ax[0].set_xlabel('Learning Rate')
    ax[0].set_ylabel('Accuracy')
    
    ax[1].plot(losses.T)
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Loss per LR')

    # conduct the expriment multiple time
    # this take around 7 mins
    nExpriments = 50
    trainEpoches = 500
    accuracies = np.zeros((nExpriments, len(learningRates)))

    for expi in range(nExpriments):
        for lri, lr in enumerate(learningRates):
            predictions, trainLosses, trainAccuracy = create_and_train_model(lr, data, labels, trainEpoches)
            accuracies[expi, lri] = trainAccuracy

    # visualize mean accuracies vs lr
    ax[2].plot(learningRates, np.mean(accuracies, axis=0), 's-')
    ax[2].set_xlabel('Learning Rate')
    ax[2].set_ylabel('Mean Accuracy')
    ax[2].set_title('Meta Expriment')
    plt.show()

