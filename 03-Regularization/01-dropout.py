import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def dropout_demo():
    x: torch.Tensor = torch.ones(10)

    p = 0.5  # independent probability for each node to get dropped
    dropout = nn.Dropout(p=p)

    y = dropout(x)

    print(f'Original Data: {x}')
    print(f'Data after dropout: {y}')
    print(f'Data after dropout but scaling reversed: {y * (1 - p)}')
    print('\n  dropped out values = 0 \n  left out values = 2 (x * w and pytorch uses w = w * (1/1-p)) ie scales '
          'weights during training')

    dropout.eval()  # this doesn't allow any dropout to happen
    y_ = dropout(x)
    print(f'Data after dropout(eval mode on): {y_}')

    y_ = nn.functional.dropout(x, training=False)  # equivalent of turning on eval() for object function Dropout


def create_data():
    n_per_cluster = 200

    thetas = np.linspace(0, 4 * np.pi, n_per_cluster)
    radi1 = 10
    radi2 = 15

    a = [radi1 * np.cos(thetas) + np.random.randn(n_per_cluster) * 3,
         radi1 * np.sin(thetas) + np.random.randn(n_per_cluster)]
    b = [radi2 * np.cos(thetas) + np.random.randn(n_per_cluster),
         radi2 * np.sin(thetas) + np.random.randn(n_per_cluster) * 3]

    data_np = np.hstack((a, b)).T
    labels_np = np.vstack((np.zeros((n_per_cluster, 1)), np.ones((n_per_cluster, 1))))

    data = torch.tensor(data_np).float()
    labels = torch.tensor(labels_np).float()

    return data, labels


def visualize_data():
    plt.figure(figsize=(10, 5), num="Visualize Generated Data")

    data, labels = create_data()
    plt.plot(data[np.where(labels == 0)[0], 0], data[np.where(labels == 0)[0], 1], 'bs')
    plt.plot(data[np.where(labels == 1)[0], 0], data[np.where(labels == 1)[0], 1], 'k^')
    plt.title('Qwerty Doughnuts')
    plt.xlabel('Dimension 1')
    plt.xlabel('Dimension 2')
    plt.show()


def get_data(train_proportion=0.8):
    data, labels = create_data()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=train_proportion)

    # train_data = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    # test_data = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)

    batch_size = 16  # int(train_data.tensors[0].shape[0]/4) -- Hard-coding is better to avoid huge batches!
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

    return train_loader, test_loader


class CircularQwerty(nn.Module):
    def __init__(self, dropout_prob):
        super().__init__()

        self.dropoutRate = dropout_prob

        self.input_layer = nn.Linear(2, 128)
        self.hidden_layer = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.dropout(x, p=self.dropoutRate, training=self.training)

        x = F.relu(self.hidden_layer(x))
        x = F.dropout(x, p=self.dropoutRate, training=self.training)

        x = self.output_layer(x)

        return x


def create_and_train_model(dr, lr=0.002, train_epochs=1000):
    model = CircularQwerty(dr)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_loader, test_loader = get_data()

    train_accuracies = []
    test_accuracies = []

    for _ in range(train_epochs):
        model.train()
        batch_accuracies = []

        for x, y in train_loader:
            y_hat = model(x)
            loss = loss_func(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_acc = 100 * torch.mean(((y_hat > 0) == y).float())
            batch_accuracies.append(batch_acc)

        train_accuracies.append(np.mean(batch_accuracies))

        model.eval()
        x, y = next(iter(test_loader))
        y_hat = model(x)
        test_accuracies.append(100 * torch.mean(((y_hat > 0) == y).float()))

    return train_accuracies, test_accuracies


if __name__ == '__main__':
    dropoutRates = np.arange(10) / 10
    accuracies = np.zeros((len(dropoutRates), 2))

    for i in range(len(dropoutRates)):
        trainAccuracy, testAccuracy = create_and_train_model(dropoutRates[i])

        # store only last 100 epochs accuracy
        accuracies[i, 0] = np.mean(trainAccuracy[-100:])
        accuracies[i, 1] = np.mean(testAccuracy[-100:])

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), num='Regularization is not always beneficial')

    ax[0].plot(dropoutRates, accuracies, 'o-')
    ax[0].set_xlabel('Dropout proportion')
    ax[0].set_ylabel('Average accuracy')
    ax[0].legend(['Train', 'Test'])

    ax[1].plot(dropoutRates, -np.diff(accuracies, axis=1), 'o-')
    ax[1].plot([0, .9], [0, 0], 'k--')
    ax[1].set_xlabel('Dropout proportion')
    ax[1].set_ylabel('Train-test difference (acc%)')

    plt.show()
