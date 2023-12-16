# create 3 custom loss functions

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from occlusion_cnn import get_data


img_size = 91
nSample = 1_000


# L1 loss function
class MyL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
      
    def forward(self,yHat,y):
        l = torch.mean( torch.abs(yHat-y) )
        return l


# L2+average loss function
class MyL2AveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,yHat,y):
        # MSE part
        l = torch.mean( (yHat-y)**2 )

        # average part
        a = torch.abs(torch.mean(yHat))

        # sum together
        return l + a


# correlation loss function
class MyCorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,yHat,y):
        meanx = torch.mean(yHat)
        meany = torch.mean(y)

        num = torch.sum( (yHat-meanx)*(y-meany) )
        den = (torch.numel(y)-1) * torch.std(yHat) * torch.std(y)
        return -num / den


def get_model():
    class GausNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.enc = nn.Sequential(
                nn.Conv2d(1,6,3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(6,4,3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2,2)  
            )

            self.dec = nn.Sequential(
                nn.ConvTranspose2d(4,6,3,2),
                nn.ReLU(),
                nn.ConvTranspose2d(6,1,3,2),
            )
          
        def forward(self,x):
            return self.dec( self.enc(x) )

    model = GausNet()
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)

    # lossfunc = nn.MSELoss()
    # lossfunc = MyL1Loss()
    # lossfunc = MyL2AveLoss()
    lossfunc = MyCorLoss()

    return model, lossfunc, optimizer


def train_model():
    epochs = 1_000
    images, imagesOccluded = get_data()
    net, lossfun, optimizer = get_model()

    losses = torch.zeros(epochs)

    for epochi in range(epochs):

        # pick a set of images at random
        pics2use = np.random.choice(nSample, size=32, replace=False)

        # get the input (has occlusions) and the target (no occlusions)
        X = imagesOccluded[pics2use, :, :, :]
        Y = images[pics2use, :, :, :]

        # forward pass and loss
        yHat = net(X)
        loss = lossfun(yHat, Y)
        losses[epochi] = loss.item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # end epochs

    # function output
    return losses, net


def main():
    losses, model = train_model()

    plt.plot(losses,'s-',label='Train')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Model loss')
    plt.show()

    # visualize some images
    _, imagesOccluded = get_data()
    pics2use = np.random.choice(nSample,size=32, replace=False)
    X = imagesOccluded[pics2use, :, :, :]
    yHat = model(X)

    _, axs = plt.subplots(2, 10, figsize=(15,3))
    for i in range(10):
        G = torch.squeeze( X[i, 0, :, :] ).detach()
        O = torch.squeeze( yHat[i, 0, :, :] ).detach()

        axs[0,i].imshow(G, vmin=-1, vmax=1, cmap='jet')
        axs[0,i].axis('off')
        axs[0,i].set_title('Model input', fontsize=10)

        axs[1,i].imshow(O, vmin=-1, vmax=1, cmap='jet')
        axs[1,i].axis('off')
        axs[1,i].set_title('Model output', fontsize=10)
    plt.show()


if __name__ == '__main__':
    main()

