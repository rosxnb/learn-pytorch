## L1 regularization on iris dataset

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


def create_and_train_model(l1lambda, trainprop=0.8, trainEpoches=1000):
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
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    trainAccuracies = []
    testAccuracies = []

    for _ in range(trainEpoches):
        batchAccuray = []

        # count total number of weights
        nweights = 0
        for pname, weight in model.named_parameters():
            if 'bias' not in pname:
                nweights += weight.numel()

        for batch_data, batch_label in train_dataset:
            yhat = model(batch_data)
            loss = lossFunc(yhat, batch_label)

            # initialize L1 term
            L1_term = torch.tensor( 0., requires_grad=True )

            # calculate L1 term
            for pname, weight in model.named_parameters():
                if 'bias' not in pname:
                    L1_term = L1_term + torch.sum( torch.abs(weight) )

            # add L1 term to loss
            loss = loss + l1lambda * L1_term / nweights

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
    trainEpoches = 1000
    trainProp = 0.8

    l1lambdas = np.linspace(0, 0.005, 10)
    accuracyResultsTrain = np.zeros( (trainEpoches, len(l1lambdas)) )
    accuracyResultsTest  = np.zeros( (trainEpoches, len(l1lambdas)) )

    for li in range(len(l1lambdas)):

        trainAccuracies, testAccuracies = create_and_train_model(l1lambdas[li], trainProp, trainEpoches)

        accuracyResultsTrain[:, li] = smooth( trainAccuracies, 10 )
        accuracyResultsTest[:, li]  = smooth( testAccuracies, 10 )

    fig,ax = plt.subplots(1, 2, figsize=(17,7))

    ax[0].plot(accuracyResultsTrain)
    ax[0].set_title('Train accuracy')
    ax[1].plot(accuracyResultsTest)
    ax[1].set_title('Test accuracy')

    # make the legend easier to read
    leglabels = [np.round(i,4) for i in l1lambdas]

    # common features
    for i in range(2):
        ax[i].legend(leglabels)
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('Accuracy (%)')
        ax[i].set_ylim([50,101])
        ax[i].grid()

    plt.show()


