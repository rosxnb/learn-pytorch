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


def create_and_train_model(trainprop, trainEpoches=1000):
    data, labels = create_data()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=trainprop)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_dataset = DataLoader(train_dataset, batch_size=4)
    test_dataset = DataLoader(test_dataset, batch_size=test_dataset.tensors[0].shape[0])

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
        testAccuracies.append( 100 * torch.mean( (torch.argmax( predictions, dim=1 ) == y).float() ) )

    return trainAccuracies, testAccuracies


if __name__ == '__main__':
    trainEpoches = 500
    trainProp = 0.8
    trainAccuracies, testAccuracies = create_and_train_model(trainProp, trainEpoches)

    # Visualization part
    plt.figure( figsize=(10, 5) )
    plt.plot( trainAccuracies, 'ro-' )
    plt.plot( testAccuracies, 'bs-' )
    plt.xlabel('Epoches')
    plt.ylabel('Accuracies')
    plt.legend(['Train', 'Test'])

    plt.show()

