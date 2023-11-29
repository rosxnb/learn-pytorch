# Generating manaual data and seperating 3 class labels

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def create_data(nPerCluster=50):
    blur = 1
    a = [ 3, -2 ]
    b = [ 3,  3 ]
    c = [ 6,  0 ]

    x = [ a[0] + np.random.randn(nPerCluster) * blur, a[1] + np.random.randn(nPerCluster) * blur ]
    y = [ b[0] + np.random.randn(nPerCluster) * blur, b[1] + np.random.randn(nPerCluster) * blur ]
    z = [ c[0] + np.random.randn(nPerCluster) * blur, c[1] + np.random.randn(nPerCluster) * blur ]

    labels_np = np.vstack( (np.zeros((nPerCluster, 1)), np.ones((nPerCluster, 1)), 1 + np.ones((nPerCluster, 1))) )
    data_np = np.hstack( (x, y, z) ).T

    data = torch.tensor(data_np, dtype=torch.float)
    # labels = torch.tensor(labels_np, dtype=torch.float)
    labels = torch.squeeze(torch.tensor(labels_np).long())
    # print(torch.tensor(labels_np).shape)
    # print(torch.squeeze(torch.tensor(labels_np)).shape)

    return data, labels


def visualize_data(nPerCluster=50):
    data, labels = create_data(nPerCluster)
    plt.plot( data[np.where(labels == 0)[0], 0], data[np.where(labels == 0)[0], 1], 'rs' )
    plt.plot( data[np.where(labels == 1)[0], 0], data[np.where(labels == 1)[0], 1], 'g^' )
    plt.plot( data[np.where(labels == 2)[0], 0], data[np.where(labels == 2)[0], 1], 'bo' )
    plt.legend()
    plt.title('Generated Data')
    plt.show()
    return


def create_and_train_model(lr, trainEcpoches):
    model = nn.Sequential(
        nn.Linear(2, 6),
        nn.ReLU(),
        nn.Linear(6, 4),
        nn.ReLU(),
        nn.Linear(4, 3)
    )

    lossFunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    data, labels = create_data()
    trainLosses = np.zeros(trainEcpoches)
    trainAccuraies = []
    for i in range(trainEcpoches):
        yhat = model(data)
        loss = lossFunc(yhat, labels)
        trainLosses[i] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictionLabels = torch.argmax(yhat, dim=1)
        currentAccuracy = 100 * torch.mean( (predictionLabels == labels).float() )
        trainAccuraies.append(currentAccuracy)

    predictions = model(data)
    predictionLabels = torch.argmax(predictions, dim=1)
    trainAccuracy = torch.mean( (predictionLabels == labels).float() ) * 100

    return predictions, trainLosses, trainAccuraies, trainAccuracy


if __name__ == '__main__':
    predictions, trainLosses, trainAccuraies, trainAccuracy = create_and_train_model(0.03, 1000)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5), num='Multiclass Qwerty Classification')
    ax[0].set_title('Training Loss Progress')
    ax[0].plot(trainLosses, 'rs')
    ax[0].set_ylabel('Losses')

    ax[1].set_title(f'Final Accuracy = {trainAccuracy}%')
    ax[1].plot(trainAccuraies, 'bo-')
    ax[1].set_ylabel('Accuracy')

    ax[2].set_title('Model Predictions')
    ax[2].legend(['qwerty 1', 'qwerty 2', 'qwerty 3'])
    ax[2].set_xlabel('Stimulus Number')
    ax[2].set_ylabel('Probability')
    softmax = nn.Softmax(dim=1)
    predictions = softmax(predictions) # change values into probailities

    colorshape = [ 'bs', 'ko', 'r^' ]
    for i in range(3):
        ax[2].plot(predictions[:, i].data, colorshape[i], markerfacecolor='w')

    plt.show()
