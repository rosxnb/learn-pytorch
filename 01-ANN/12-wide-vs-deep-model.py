from typing_extensions import override
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


def get_data():
    iris = sns.load_dataset('iris')
    data = torch.tensor( iris[iris.columns[:4]].values, dtype=torch.float )

    labels = torch.zeros( len(data), dtype=torch.long )
    labels[ iris.species == 'versicolor' ] = 1
    labels[ iris.species == 'virginica' ] = 2

    return data, labels


class AnnIris(nn.Module):
    def __init__(self, nUnits, nLayers):
        super().__init__()

        # create dictionary to store layers
        self.layersDict = nn.ModuleDict()
        self.nLayers = nLayers

        # input layer
        self.layersDict['input_layer'] = nn.Linear(4, nUnits)

        # hidden layers
        for i in range(nLayers):
            self.layersDict[f'hidden_layer_{i}'] = nn.Linear(nUnits, nUnits)

        # output layer
        self.layersDict['output_layer'] = nn.Linear(nUnits, 3)

    @override
    def forward(self, x):
        # input layer
        x = F.relu( self.layersDict['input_layer'](x) )

        # hidden layers
        for i in range(self.nLayers):
            x = F.relu( self.layersDict[f'hidden_layer_{i}'](x) )

        # output layer
        x = self.layersDict['output_layer'](x) # notice tha activation is not applied
        return x


def test_run_model():
    nUnitsPerLayer = 12
    nLayers = 4
    model = AnnIris(nUnitsPerLayer, nLayers)
    print('Model Architecture:')
    print(model)
    print()

    dummyData = torch.randn(10, 4)
    prediction = model(dummyData)
    print(f'Shape of output: {prediction.shape}')
    print('Model test prediction values:')
    print(prediction)


def create_and_train_model(nUnits, nLayers, lr, trainEpoches):
    model = AnnIris(nUnits, nLayers)
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
    predLabels = torch.argmax( predictions, dim=1 )
    trainAccuracy = torch.mean( (predLabels == labels).float() ) * 100

    numOfTrainableParameters = sum( p.numel() for p in model.parameters() if p.requires_grad )

    return trainAccuracy, numOfTrainableParameters


if __name__ == '__main__':
    numOfUnits = np.arange(4, 101, 3)
    numOfLayers = range(1, 6)

    lr = 0.01
    trainEpoches = 500
    trainAccuracies = np.zeros( (len(numOfUnits), len(numOfLayers)) )
    trainableParameters = np.zeros( (len(numOfUnits), len(numOfLayers)) )

    for ithUnit, nUnits in enumerate(numOfUnits):
        for ithLayer, nLayers in enumerate(numOfLayers):
            trainAccuracy, numOfTrainableParameters = create_and_train_model(nUnits, nLayers, lr, trainEpoches)

            trainAccuracies[ithUnit, ithLayer] = trainAccuracy
            trainableParameters[ithUnit, ithLayer] = numOfTrainableParameters

    # visulaize relation between accuracy and trainable parameters
    fig, ax = plt.subplots(1, 2, figsize=(16, 5), num='Does num of trainable parameters effect model accuracy?')

    ax[0].set_title('Accuracy per NumOfUnits')
    ax[0].plot(numOfUnits, trainAccuracies, 'o-', markersize=8, markerfacecolor='w')
    ax[0].plot(numOfUnits[[0, -1]], [33, 33], '--', color=[.8, .8, .8], label='')
    ax[0].plot(numOfUnits[[0, -1]], [67, 67], '--', color=[.8, .8, .8], label='')
    ax[0].set_xlabel('Number of Hidden Units')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(numOfLayers)


    parameters = trainableParameters.flatten()
    accuracies = trainAccuracies.flatten()
    r = np.corrcoef(parameters, accuracies)[0, 1]

    ax[1].set_title(f'Accuracy per NumOfParameters. r = {r}')
    ax[1].plot(parameters, accuracies, 'o', markerfacecolor='w')
    ax[1].set_xlabel('Number of Parameters')
    ax[1].set_ylabel('Accuracy')

    plt.show()
