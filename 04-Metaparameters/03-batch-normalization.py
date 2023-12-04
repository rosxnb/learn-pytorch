import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from wine_data import get_datasets


class WineModelWithBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(11, 16)

        self.bNorm1 = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(16, 32)

        self.bNorm2 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 20)

        self.output = nn.Linear(20, 1)

    def forward(self, data, do_batch_norm):
        data = F.relu(self.input(data))

        if do_batch_norm:
            data = self.bNorm1(data)
            data = F.relu(self.fc1(data))
            data = self.bNorm2(data)
            data = F.relu(self.fc2(data))
        else:
            data = F.relu(self.fc1(data))
            data = F.relu(self.fc2(data))

        return self.output(data)


def train_model(train_loader, test_loader, do_batch_norm=True, epochs=1000):
    model = WineModelWithBatchNorm()
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
            predictions = model(data, do_batch_norm)
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
            testPredictions = model(data, do_batch_norm)
        testAccu.append(100 * torch.mean(((testPredictions > 0) == label).float()).item())

    return trainAccu, testAccu, trainLoss


if __name__ == '__main__':
    train_dataset, test_dataset = get_datasets()

    trainLoader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    testLoader = DataLoader(test_dataset, batch_size=test_dataset.tensors[0].shape[0])

    trainAccNo, testAccNo, lossesNo = train_model(trainLoader, testLoader, do_batch_norm=False)
    trainAccWith, testAccWith, lossesWith = train_model(trainLoader, testLoader, do_batch_norm=True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    ax[0].plot(lossesWith, label='WITH batch norm')
    ax[0].plot(lossesNo, label='NO batch norm')
    ax[0].set_title('Losses')
    ax[0].legend()

    ax[1].plot(trainAccWith, label='WITH batch norm')
    ax[1].plot(trainAccNo, label='NO batch norm')
    ax[1].set_title('Train accuracy')
    ax[1].legend()

    ax[2].plot(testAccWith, label='WITH batch norm')
    ax[2].plot(testAccNo, label='NO batch norm')
    ax[2].set_title('Test accuracy')
    ax[2].legend()

    plt.show()
