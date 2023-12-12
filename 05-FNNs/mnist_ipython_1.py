# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import torch
filepath = 'data/mnist_train_small.csv'
whole_data = np.loadtxt(open(filepath, 'rb'), delimiter=',')
whole_data.shape
data = whole_data[:, 1:]
label = whole_data[:, 0]
label.shape
data.shape
import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
data = torch.tensor( data, dtype=torch.float )
label = torch.tensor( label, dtype=torch.long )
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1)
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
batch_size = 32
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=train_dataset.tensors[0].shape[0])
learning_rate=0.01
epochs = 60
class MnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(784, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 10)

    def forward(self, data):
        data = F.relu(self.input(data))
        data = F.relu(self.fc1(data))
        data = F.relu(self.fc2(data))
        return torch.log_softmax(self.output(data), axis=1)

def train_model():
    r'''
        Batch trains the model for epochs with learning_rate

        Parameters:
            None

        Return value:
            losses[], train_accuracies, test_accuracies, model
    '''

    model = MnistClassifier()
    lossFunc = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_accuracies = []
    test_accuracies = []
    losses = []

    for epoch in range(epochs):
        batch_accuracy = []
        batch_loss = []

        for X, y in train_loader:
            predictions = model(X)
            loss = lossFunc(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
            predicted_labels = torch.argmax(predictions, axis=1)
            batch_acc = (predicted_labels == y).float()
            batch_acc = 100 * torch.mean(batch_acc)
            batch_accuracy.append(batch_acc)

        losses.append(np.mean(batch_loss))
        train_accuracies.append(np.mean(batch_accuracy))

        test_X, test_y = next(iter(test_loader))
        predictions = model(test_X)
        predicted_labels = torch.argmax(predictions, axis=1)
        test_acc = (predicted_labels == test_y).float()
        test_acc = 100 * torch.mean(test_acc)
        test_accuracies.append(test_acc)

    return losses, train_accuracies, test_accuracies, model
    
losses_train, accuracy_train, accuracy_test, mnist_model = train_model()
plt.plot(losses_train)
plt.xlabel('epochs')
plt.ylabel('losses')
plt.title('Training Losses')
plt.show()
plt.plot(accuracy_train, label='train acc')
plt.plot(accuracy_test, label='test acc')
plt.xlabel('epochs')
plt.ylabel('accuracy (%)')
plt.show()
accuracy_test[-1]
help(torch.max)
