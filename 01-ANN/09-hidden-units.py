# Parametric Expriment: check model performance by only changing the hidden units

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_data():
    iris = sns.load_dataset('iris')
    data = torch.tensor( iris[ iris.columns[:4] ].values, dtype=torch.float )

    labels = torch.zeros( len(data), dtype=torch.long )
    labels[ iris[ iris.columns[4] ] == 'versicolor' ] = 1
    labels[ iris[ iris.columns[4] ] == 'virginica' ] = 2

    return data, labels


def create_and_train_model(nHiddenUnit, lr = 0.03, trainEpoches=150):
    model = nn.Sequential(
        nn.Linear(4, nHiddenUnit),
        nn.ReLU(),
        nn.Linear(nHiddenUnit, nHiddenUnit),
        nn.ReLU(),
        nn.Linear(nHiddenUnit, 3)
    )

    lossFunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    data, labels = get_data()

    for _ in range(trainEpoches):
        yhat = model(data)
        loss = lossFunc(yhat, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    predictions = model(data)
    predictionLabels = torch.argmax( predictions, dim=1 )
    trainAccuracy = 100 * torch.mean( (predictionLabels == labels).float() )
    return trainAccuracy


if __name__ == '__main__':
    nHiddenUnits = np.arange(1, 129) # vary the hidden units count
    trainAccuracies = []

    for nHiddenunit in nHiddenUnits:
        trainAccuracy = create_and_train_model(nHiddenunit)
        trainAccuracies.append(trainAccuracy)

    plt.figure(figsize=(12, 5), num='Hidden Units Variation Expriment')
    plt.plot(nHiddenUnits, trainAccuracies, 'ko-', markerfacecolor='w')
    plt.ylabel('Accuracy')
    plt.xlabel('Hidden Units')
    plt.title('Accuracy per Hidden Units')

    plt.plot([1, 128], [33, 33], '--', color=[.8, .8, .8]) # model not getting anything right boundary
    plt.plot([1, 128], [67, 67], '--', color=[.8, .8, .8]) # model only getting 2 labels right boundary

    plt.show()


