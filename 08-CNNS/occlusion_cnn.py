# Remove occlusion from manually generated gaussian blur images

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


img_size = 91
nSample = 1_000

def get_data(visualize=False):
    x = np.linspace(-4, 4, img_size)
    xx, yy = np.meshgrid(x, x)

    images = torch.zeros(nSample, 1, img_size, img_size)
    imagesOccluded = torch.zeros(nSample, 1, img_size, img_size)
    widths = np.linspace(2, 20, nSample)

    for i in range(nSample):
        # create gaussian with random center
        center = np.random.randn(2) * 1.5
        gaussian = np.exp( -( (xx - center[0]) ** 2 + (yy - center[1]) ** 2 ) / widths[i] )

        # add noise
        gaussian = gaussian + np.random.randn(img_size, img_size) / 5
        images[i, 0, :, :] = torch.Tensor(gaussian).view(1, img_size, img_size)

        # add random bar
        i1 = np.random.choice( range(2, 28) )
        i2 = np.random.choice( range(2, 6) )
        if np.random.randn() > 0:
            gaussian[ i1 : i1 + i2, ] = 1
        else:
            gaussian[ :, i1 : i1 + i2 ] = 1

        # store as image
        imagesOccluded[i, 0, :, :] = torch.Tensor(gaussian).view(1, img_size, img_size)

    if not visualize: return images, imagesOccluded

    _, ax = plt.subplots(2, 10, figsize=(15,3), num='Generated Images')
    for i in range(10):
        pic = np.random.randint(nSample)
        ax[0,i].imshow(np.squeeze( images[pic,:,:] ),vmin=-1,vmax=1,cmap='jet')
        ax[0,i].set_xticks([]), ax[0,i].set_yticks([])

        ax[1,i].imshow(np.squeeze( imagesOccluded[pic,:,:] ),vmin=-1,vmax=1,cmap='jet')
        ax[1,i].set_xticks([]), ax[1,i].set_yticks([])
    plt.show()

    return images, imagesOccluded


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
    lossfunc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)

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

