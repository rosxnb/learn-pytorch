import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn.model_selection import train_test_split


class MnistClassifier(nn.Module):
    def __int__(self):
        super.__init__()

        self.input = nn.Linear(786, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 10)

    def forward(self, data):
        data = F.relu(self.input(data))
        data = F.relu(self.fc1(data))
        data = F.relu(self.fc2(data))

        return F.log_softmax(data)


def main():
    filepath = 'data/mnist_train_small.csv'
    whole_data = np.loadtxt(open(filepath, 'rb'), delimiter=',')
    data = whole_data[:, 1:]
    label = whole_data[:, 0]

    # normalize the data
    data = data / max(data)


if __name__ == '__main__':
    pass
