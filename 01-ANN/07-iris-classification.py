# Model to clasify multiple class as output

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_data():
    iris = sns.load_dataset('iris')
    # print(f'Head of dataset ({type(iris)}):')
    # print(iris.head())

    data = torch.tensor(iris[iris.columns[0:4]].values ).float()
    labels = torch.zeros(len(data), dtype=torch.long)
    # labels[iris.species == 'setosa'] = 0
    labels[iris.species == 'vesicolor'] = 1
    labels[iris.species == 'virginica'] = 2

    # sns.pairplot(iris, hue='species')
    # plt.show()
    return data, labels


def create_and_train_model(lr, trainEpoches=1000):
    data, labels = create_data()

    model = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    )

    lossFunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    accuracies = []
    losses = np.zeros(trainEpoches)
    for i in range(trainEpoches):
        yhat = model(data)
        loss = lossFunc(yhat, labels)
        losses[i] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        currentAccuracy = torch.mean( (torch.argmax(yhat, dim=1) == labels).float() ) * 100
        accuracies.append(currentAccuracy)

    predictions = model(data)
    predLabels = torch.argmax(predictions, dim=1)
    
    accuracy = torch.mean((predLabels == labels).float()) * 100

    return losses, accuracies, predLabels, accuracy


if __name__ == '__main__':
    losses, accuracies, predLabels, accuracy = create_and_train_model(0.01)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 5), num='Iris Classification')
    ax[0].set_title('Losses per training epoch')
    ax[0].plot(losses, 'ro-')
    ax[1].set_title(f'Accuracies per training epoch ({accuracy}%)')
    ax[1].plot(accuracies, 'bo-')
    plt.show()

