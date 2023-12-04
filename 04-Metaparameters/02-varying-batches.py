import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import time
import numpy as np
import matplotlib.pyplot as plt
from wine_data import get_datasets


class WineModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = nn.Linear(11, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, data) -> torch.Tensor:
        data = F.relu(self.input(data))
        data = F.relu(self.fc1(data))
        data = F.relu(self.fc2(data))

        return self.output(data)


def train_model(train_loader, test_loader, epochs=1000):
    model = WineModel()
    lossFunc = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainAccu = []
    testAccu = []
    trainLoss = torch.zeros(epochs)

    for epoch in range(epochs):
        model.train()  # switch to training mode

        batchAcc = []
        batchLoss = []
        for data, label in train_loader:
            predictions = model(data)
            loss = lossFunc(predictions, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchLoss.append(loss.item())

            truths = (predictions > 0) == label
            batchAcc.append(100 * torch.mean(truths.float()).item())

        trainLoss[epoch] = np.mean(batchLoss)
        trainAccu.append(np.mean(batchAcc))

        model.eval()  # switch off the training mode

        data, label = next(iter(test_loader))  # because we only have 1 batch for testing
        with torch.no_grad():
            testPredictions = model(data)
        testAccu.append(100 * torch.mean(((testPredictions > 0) == label).float()).item())

    return trainAccu, testAccu, trainLoss


if __name__ == '__main__':
    numEpochs = 1000
    batchSizes = 2 ** np.arange(1, 10, 2)

    trainDataset, testDataset = get_datasets(train_size=0.9)
    testLoader = DataLoader(testDataset, batch_size=testDataset.tensors[0].shape[0])

    trainAccRes = np.zeros((numEpochs, len(batchSizes)))
    testAccRes = np.zeros((numEpochs, len(batchSizes)))
    trainLossRes = np.zeros((numEpochs, len(batchSizes)))
    compTime = np.zeros(len(batchSizes))

    for idx, batch in enumerate(batchSizes):
        trainLoader = DataLoader(trainDataset, batch_size=int(batch), shuffle=True, drop_last=True)

        startTime = time.process_time()
        trainAcc, testAcc, trainLosses = train_model(trainLoader, testLoader, numEpochs)
        endTime = time.process_time()

        trainAccRes[:, idx] = trainAcc
        testAccRes[:, idx] = testAcc
        trainLossRes[:, idx] = trainLosses
        compTime[idx] = endTime - startTime

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[0].plot(trainAccRes)
    ax[0].set_title('Train Accuracies')
    ax[1].plot(testAccRes)
    ax[1].set_title('Test Accuracies')

    for i in range(2):
        ax[i].legend(batchSizes)
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel('Accuracy (%)')
        ax[i].set_ylim([0, 100])
        ax[i].grid()

    ax[2].plot(trainLossRes)
    ax[2].set_title('Train Loss')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Loss')
    ax[2].legend(batchSizes)
    plt.show()

    plt.bar(range(len(compTime)), compTime, tick_label=batchSizes)
    plt.xlabel('Mini-batch size')
    plt.ylabel('Computation time (s)')
    plt.show()
