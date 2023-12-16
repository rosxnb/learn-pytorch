import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_data():
    filepath = '../05-FNNs/data/mnist_train_small.csv'
    data = np.loadtxt( open(filepath, 'rb'), delimiter=',' )
    
    label = data[:, 0]
    data = data[:, 1:]
    data = data / np.max(data)

    data = data.reshape( data.shape[0], 1, 28, 28 ) # convert to 2D image

    label = torch.tensor( label ).long()
    data = torch.tensor( data ).float()

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1)
    batch_size = 32

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.tensors[0].shape[0])

    return train_loader, test_loader


def get_model(toggle_print=False):
    class MnistCNN(nn.Module):
        def __init__(self, toggle_print):
            super().__init__()

            self.print = toggle_print
            self.max_pool_N = 2

            self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=1)
            # size: np.floor( (28 + 2*1 - 5) / 1 ) + 1 = 26
            # after pooling: 26 / 2 = 13 (/2 b/c maxpool)

            self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=1)
            # size: np.floor( (13 + 2*1 - 5) / 1 ) + 1 = 11
            # after pooling: 11 / 2 = 5 (/2 b/c maxpool)

            expected_size = np.floor( (5 + 2 * 0 - 1) / 1 ) + 1 # fc1 layer has no padding or kernel, so set to 0/1
            expected_size = 20 * int(expected_size ** 2) # since the image is square

            self.fc1 = nn.Linear(expected_size, 50)
            self.output = nn.Linear(50, 10)

        def forward(self, data):
            if self.print: print(f'Input: {data.shape}')

            # convolution -> max-pool -> relu
            data = F.relu( F.max_pool2d( self.conv1(data), self.max_pool_N ) )
            if self.print: print(f'conv1 and max-pool: {data.shape}')

            data = F.relu( F.max_pool2d( self.conv2(data), self.max_pool_N ) )
            if self.print: print(f'conv1 and max-pool: {data.shape}')

            # reshape for linear layer
            units_count = int(data.shape.numel() / data.shape[0])
            data = data.view(-1, units_count)
            if self.print: print(f'vectorized: {data.shape}')

            data = F.relu( self.fc1(data) )
            if self.print: print(f'fc1: {data.shape}')

            data = self.output(data)
            if self.print: print(f'output: {data.shape}')

            return data

    model = MnistCNN(toggle_print)
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, lossfunc, optimizer


def test_run_model():
    model, lossfunc, _ = get_model(True)
    train_loader = get_data()[0]

    x, y = next(iter(train_loader))
    y_hat = model(x)
    print(f'\ny: {y.shape} \t y_hat: {y_hat.shape}')

    loss = lossfunc(y_hat, y)
    print(f'loss = {loss}')

    print()
    summary(model, (1, 28, 28))


def train_model(train_loader, test_loader):
    numepochs = 10

    model, lossfun, optimizer = get_model()

    # initialize losses
    losses    = torch.zeros(numepochs)
    train_accuracies  = []
    test_accuracies   = []


    # loop over epochs
    for epochi in range(numepochs):

        # loop over training data batches
        model.train()
        batchAcc  = []
        batchLoss = []
        for X,y in train_loader:

            # forward pass and loss
            yHat = model(X)
            loss = lossfun(yHat,y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batchLoss.append(loss.item())

            # compute accuracy
            matches = torch.argmax(yHat,axis=1) == y     # booleans (false/true)
            matchesNumeric = matches.float()             # convert to numbers (0/1)
            accuracyPct = 100*torch.mean(matchesNumeric) # average and x100
            batchAcc.append( accuracyPct )               # add to list of accuracies
        # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        train_accuracies.append( np.mean(batchAcc) )

        # and get average losses across the batches
        losses[epochi] = np.mean(batchLoss)

        # test accuracy
        model.eval()
        X,y = next(iter(test_loader)) # extract X,y from test dataloader
        with torch.no_grad(): # deactivates autograd
            yHat = model(X)
          
        # compare the following really long line of code to the training accuracy lines
        test_accuracies.append( 100*torch.mean((torch.argmax(yHat,axis=1)==y).float()) )
    # end epochs

    # function output
    return train_accuracies, test_accuracies, losses, model


def main():
    train_loader, test_loader = get_data()
    train_accuracies, test_accuracies, losses, _ = train_model(train_loader, test_loader)

    _, ax = plt.subplots(1, 2, figsize=(16,6), num='MNIST CNN')
    ax[0].plot(losses, 's-')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Model loss')

    ax[1].plot(train_accuracies, 's-', label='Train')
    ax[1].plot(test_accuracies, 'o-', label='Test')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy (%)')
    ax[1].set_title(f'Final model test accuracy: {test_accuracies[-1]:.2f}%')
    ax[1].legend()

    plt.show()
    

if __name__ == '__main__':
    main()

