## L2 regularization on iris dataset

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def create_data():
    iris = sns.load_dataset('iris')

    data = torch.tensor(iris[iris.columns[0:4]].values ).float()
    labels = torch.zeros(len(data), dtype=torch.long)
    labels[iris.species == 'vesicolor'] = 1
    labels[iris.species == 'virginica'] = 2

    return data, labels


def create_and_train_model(l2lambda, trainprop=0.8, trainEpoches=1000):
    data, labels = create_data()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=trainprop)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataset = DataLoader(train_dataset, batch_size=4)
    test_dataset = DataLoader(test_dataset, batch_size=test_dataset.tensors[0].shape[0])

    model = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 3)
    )

    lr = 0.005
    lossFunc = nn.CrossEntropyLoss()

    # apply ridge regression
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2lambda)

    trainAccuracies = []
    testAccuracies = []

    for _ in range(trainEpoches):
        batchAccuray = []

        for batch_data, batch_label in train_dataset:
            yhat = model(batch_data)
            loss = lossFunc(yhat, batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchAccuray.append( 100 * torch.mean( (torch.argmax( yhat, dim=1 ) == batch_label).float() ) )

        trainAccuracies.append( np.mean(batchAccuray) )
        
        X, y = next(iter(test_dataset)) # extract the data and label from test_dataset
        predictions = model(X)
        testAccuracies.append( 100 * torch.mean( (torch.argmax( predictions, dim=1 ) == y).float()) )

    return trainAccuracies, testAccuracies


# 1D smooting filter
def smooth(x,k):
  return np.convolve(x,np.ones(k)/k,mode='same')


if __name__ == '__main__':
    trainEpoches = 500
    trainProp = 0.8

    l2lambdas = np.linspace(0, 0.1, 10)
    accuracyResultsTrain = np.zeros( (trainEpoches, len(l2lambdas)) )
    accuracyResultsTest  = np.zeros( (trainEpoches, len(l2lambdas)) )

    for li in range(len(l2lambdas)):

        trainAccuracies, testAccuracies = create_and_train_model(l2lambdas[li], trainProp, trainEpoches)

        accuracyResultsTrain[:, li] = smooth( trainAccuracies, 10 )
        accuracyResultsTest[:, li]  = smooth( testAccuracies, 10 )

    fig,ax = plt.subplots(1, 2, figsize=(17,7))

    ax[0].plot(accuracyResultsTrain)
    ax[0].set_title('Train accuracy')
    ax[1].plot(accuracyResultsTest)
    ax[1].set_title('Test accuracy')

    # make the legend easier to read
    leglabels = [np.round(i,2) for i in l2lambdas]

    # common features
    for i in range(2):
        ax[i].legend(leglabels)
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('Accuracy (%)')
        ax[i].set_ylim([50,101])
        ax[i].grid()

    plt.show()

