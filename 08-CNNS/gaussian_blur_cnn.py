import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_data():
    nPerClass = 1000
    imgSize   = 91
    
    x = np.linspace(-4, 4, imgSize)
    xx, yy = np.meshgrid(x, x)
    
    # the two widths (a.u.)
    widths = [1.8, 2.4]
    
    # initialize tensors containing images and labels
    images = torch.zeros(2*nPerClass, 1, imgSize, imgSize)
    labels = torch.zeros(2*nPerClass)
    
    for i in range(2*nPerClass):
        # create the gaussian with random centers
        ro = 2*np.random.randn(2) # ro = random offset
        G  = np.exp( -( (xx-ro[0])**2 + (yy-ro[1])**2) / (2*widths[i%2]**2) )

        # and add noise
        G  = G + np.random.randn(imgSize, imgSize) / 5

        # add to the tensor
        images[i,:,:,:] = torch.Tensor(G).view(1, imgSize, imgSize)
        labels[i] = i % 2
    
    labels = labels[:, None]

    # plot_gaussian_images(nPerClass, images, labels)

    train_data,test_data, train_labels,test_labels = train_test_split(images, labels, test_size=.1)

    train_data = TensorDataset(train_data,train_labels)
    test_data  = TensorDataset(test_data,test_labels)

    batchsize    = 32
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_data, batch_size=test_data.tensors[0].shape[0])

    return train_loader, test_loader


def get_model():
    class GausNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.enc = nn.Sequential(
                nn.Conv2d(1, 6, 3, padding=1),  # output size: (91+2*1-3)/1 + 1 = 91
                nn.ReLU(),                      # note that relu is treated like a "layer"
                nn.AvgPool2d(2, 2),             # output size: 91/2 = 45 
                nn.Conv2d(6, 4, 3, padding=1),  # output size: (45+2*1-3)/1 + 1 = 45
                nn.ReLU(),                    
                nn.AvgPool2d(2, 2),             # output size: 45/2 = 22
                nn.Flatten(),                   # vectorize conv output
                nn.Linear(22*22*4, 50),         # output size: 50
                nn.Linear(50, 1),               # output size: 1
            )

        def forward(self,x):
            return self.enc(x)

    model = GausNet()
    lossfunc = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, lossfunc, optimizer


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
            matches = (yHat > 0) == y     # booleans (false/true)
            matchesNumeric = matches.float()             # convert to numbers (0/1)
            accuracyPct = 100 * torch.mean(matchesNumeric) # average and x100
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
        test_accuracies.append(100 * torch.mean( ((yHat > 0) == y).float() ))
    # end epochs

    # function output
    return train_accuracies, test_accuracies, losses, model

def main():
    train_loader, test_loader = get_data()
    train_accuracies, test_accuracies, losses, model = train_model(train_loader, test_loader)

    _, ax = plt.subplots(1, 2, figsize=(16,6), num='Losses and Accuray plot')
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

    # visualize some images
    X, y = next(iter(test_loader)) # extract X,y from test dataloader
    yHat = model(X)

    _, axs = plt.subplots(2, 10, figsize=(18,7), num='Visualizing some images')

    for i,ax in enumerate(axs.flatten()):
        G = torch.squeeze( X[i,0,:,:] ).detach()
        ax.imshow(G, vmin=-1, vmax=1, cmap='jet')
        t = ( int(y[i].item()) , int(yHat[i].item()>0) )
        ax.set_title('T:%s, P:%s'%t)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

    # visualize the filters (kernels)
    layer1W = model.enc[0].weight
    layer3W = model.enc[3].weight

    _, axs = plt.subplots(1, 6, figsize=(16,4), num='filters learned')
    for i,ax in enumerate(axs.flatten()):
        ax.imshow( torch.squeeze(layer1W[i,:,:,:]).detach(), cmap='Purples')
        ax.axis('off')
    plt.suptitle('First convolution layer filters')
    plt.show()

    _, axs = plt.subplots(4, 6, figsize=(15,9), num='filters learned')
    for i in range(6*4):
        idx = np.unravel_index(i, (4,6))
        axs[idx].imshow( torch.squeeze(layer3W[idx[0],idx[1],:,:]).detach(), cmap='Purples')
        axs[idx].axis('off')
    plt.suptitle('Second convolution layer filters')
    plt.show()

    img_size = 91
    summary(model, (1, img_size, img_size))

def plot_gaussian_images(nPerClass, images, labels):
    _, axs = plt.subplots(3, 7, figsize=(13,6))

    for _, ax in enumerate(axs.flatten()):
        whichpic = np.random.randint(2 * nPerClass)
        G = np.squeeze( images[whichpic,:,:] )
        ax.imshow(G, vmin=-1, vmax=1, cmap='jet')
        ax.set_title('Class %s' %int(labels[whichpic].item()))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


if __name__ == '__main__':
    main()

