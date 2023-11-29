import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

if __name__ == '__main__':
    # generate data
    N = 30
    x = torch.randn(N, 1)
    y = x + torch.randn(N, 1)

    # visualize the data
    plt.plot(x, y, 'ks')
    plt.title('Generated Data')
    plt.show()

    # specify the model architecture
    AnnReg = nn.Sequential(
        nn.Linear(1, 1),
        nn.ReLU(),
        nn.Linear(1, 1)
    )
    print('Model Description: ')
    print(AnnReg)

    # prepare loss and optimizer function
    lr = 0.05
    lossFn = nn.MSELoss()
    optimizer = torch.optim.SGD(AnnReg.parameters(), lr=lr)

    # train the model
    trainEpochs = 500
    losses = torch.zeros(trainEpochs)

    for epochi in range(trainEpochs):
        # forward pass
        yHat = AnnReg(x)

        # loss computation
        testLoss = lossFn(yHat, y)
        losses[epochi] = testLoss

        # back prop
        optimizer.zero_grad()
        testLoss.backward()
        optimizer.step()
        print(f'test loss {epochi}:', testLoss)

    # get the predictions
    predictions = AnnReg(x)
    testLoss = (predictions - y).pow(2).mean()

    # visualize how the loss decreased over training epoches
    plt.plot(losses.data, 'o', markerfacecolor='w', linewidth=.1) # plot the losses
    plt.plot(trainEpochs, testLoss.item(), 'rs') # plot the final loss point
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title(f'Final loss = {testLoss.data}')
    plt.show()

    # plot the predicted line
    correl = np.corrcoef(y.T, predictions.data.T)[0, 1]
    plt.plot(x, y, 'ks')
    plt.plot(x, predictions.data, 'go')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Correlation coefficient = {correl}')
    plt.show()
