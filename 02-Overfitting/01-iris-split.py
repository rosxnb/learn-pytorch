import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def create_data():
    iris = sns.load_dataset('iris')

    data = torch.tensor(iris[iris.columns[0:4]].values ).float()
    labels = torch.zeros(len(data), dtype=torch.long)
    labels[iris.species == 'vesicolor'] = 1
    labels[iris.species == 'virginica'] = 2

    return data, labels


def create_and_train_model(trainprop, trainEpoches=1000):
    data, labels = create_data()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=trainprop)

    model = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    )

    lr = 0.01
    lossFunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    trainAccuracies = []
    testAccuracies = []

    for _ in range(trainEpoches):
        yhat = model(X_train)
        loss = lossFunc(yhat, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trainPredLabels = torch.argmax(yhat, dim=1)
        trainAccuracy = torch.mean((trainPredLabels == y_train).float()) * 100
        trainAccuracies.append(trainAccuracy)

        testPredLabels = torch.argmax(model(X_test), dim=1)
        testAccuracy = torch.mean((testPredLabels == y_test).float()) * 100
        testAccuracies.append(testAccuracy)

    return trainAccuracies, testAccuracies


if __name__ == '__main__':
    trainEpoches = 250
    trainSetProps = np.linspace(0.2, 0.95, 10)

    trainAccuracies = np.zeros( (len(trainSetProps), trainEpoches) )
    testAccuracies = np.zeros( (len(trainSetProps), trainEpoches) )

    for i, trainprop in enumerate(trainSetProps):
        trainAcc, testAcc = create_and_train_model(trainprop, trainEpoches)

        trainAccuracies[i, :] = trainAcc
        testAccuracies[i, :] = testAcc

    # Visualization part
    fig, ax = plt.subplots(1, 2, figsize=(16, 5), num='Varying Test Data Sizes')
    ax[0].imshow(trainAccuracies,aspect='auto',
                 vmin=50,vmax=90, extent=[0,trainEpoches,trainSetProps[-1],trainSetProps[0]])
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Training size proportion')
    ax[0].set_title('Training accuracy')

    p = ax[1].imshow(testAccuracies,aspect='auto',
                 vmin=50,vmax=90, extent=[0,trainEpoches,trainSetProps[-1],trainSetProps[0]])
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Training size proportion')
    ax[1].set_title('Test accuracy')
    fig.colorbar(p,ax=ax[1])

    plt.show()

