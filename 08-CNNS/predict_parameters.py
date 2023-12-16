import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


nGauss  = 1000
imgSize = 91

def get_data():
    x = np.linspace(-4,4,imgSize)
    X,Y = np.meshgrid(x,x)

    # initialize tensors containing images and labels
    images = torch.zeros(nGauss,1,imgSize,imgSize)
    labels = torch.zeros(nGauss,3)

    for i in range(nGauss):
        # location and width parameters
        loc = np.max(x)/2 * np.random.randn(2) # center coordinate
        wid = np.random.rand()*10 + 5 # width of Gaussian

        # create the gaussian with random centers
        G  = np.exp( -( (X-loc[0])**2 + (Y-loc[1])**2) / wid )
        G  = G + np.random.randn(imgSize,imgSize)/10

        # add to the tensor
        images[i,:,:,:] = torch.Tensor(G).view(1,imgSize,imgSize)
        labels[i,:] = torch.Tensor( [loc[0],loc[1],wid] )

    # use scikitlearn to split the data
    train_data,test_data, train_labels,test_labels = train_test_split(images, labels, test_size=.1)

    # convert into PyTorch Datasets
    train_data = TensorDataset(train_data,train_labels)
    test_data  = TensorDataset(test_data,test_labels)

    # translate into dataloader objects
    batchsize    = 16
    train_loader = DataLoader(train_data,batch_size=batchsize,shuffle=True,drop_last=True)
    test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

    return train_loader, test_loader


def get_model():
    class gausnet(nn.Module):
        def __init__(self):
            super().__init__()

            # all layers in one go using nn.Sequential
            self.enc = nn.Sequential(

                # conv/pool block 1
                nn.Conv2d(1,6,3,padding=1),  # output size: (91+2*1-3)/1 + 1 = 91
                nn.ReLU(),                   # 
                nn.AvgPool2d(2,2),           # output size: 91/2 = 45

                # conv/pool block 2
                nn.Conv2d(6,4,3,padding=1),  # output size: (45+2*1-3)/1 + 1 = 45
                nn.ReLU(),                   # 
                nn.AvgPool2d(2,2),           # output size: 45/2 = 22

                # linear decision layer
                nn.Flatten(),                # vectorize conv2 block output
                nn.Linear(22*22*4,50),       # output size: 50
                nn.Linear(50,3),             # output size: 3
            )

        def forward(self,x):
            return self.enc(x)

    # create the model instance
    net = gausnet()

    # loss function
    lossfun = nn.MSELoss()

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(),lr=.001)

    return net,lossfun,optimizer


def train_model():
    # number of epochs
    numepochs = 30

    train_loader, test_loader = get_data()

    # create a new model
    net,lossfun,optimizer = get_model()

    # initialize losses
    trainLoss = torch.zeros(numepochs)
    testLoss  = torch.zeros(numepochs)


    # loop over epochs
    for epochi in range(numepochs):

        # loop over training data batches
        batchLoss = []
        for X,y in train_loader:
            # forward pass and loss
            yHat = net(X)
            loss = lossfun(yHat,y)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batchLoss.append(loss.item())
            # end of batch loop...

        # and get average losses across the batches
        trainLoss[epochi] = np.mean(batchLoss)

        # test accuracy
        X,y = next(iter(test_loader)) # extract X,y from test dataloader
        with torch.no_grad(): # deactivates autograd
            yHat = net(X)
            loss = lossfun(yHat,y)

        # extract the loss for this test epoch
        testLoss[epochi] = loss.item()
    # end epochs

    # function output
    return trainLoss,testLoss,net


def main():
    _, test_loader = get_data()
    trainLoss,testLoss,net = train_model()

    plt.figure(figsize=(16, 5), num='Losses')
    plt.plot(trainLoss,'s-',label='Train')
    plt.plot(testLoss,'o-',label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Model loss (final test loss: %.2f)'%testLoss[-1])

    X, Y = next(iter(test_loader)) # extract X,y from test dataloader
    yHat = net(X)

    th = np.linspace(0,2*np.pi)

    _, axs = plt.subplots(2,10,figsize=(16,6), num='Predicted Gaussian Blur')
    for i,ax in enumerate(axs.flatten()):

        # get the Gaussian and draw it, and draw the white guide-lines
        G = torch.squeeze( X[i,0,:,:] ).detach()
        ax.imshow(G,vmin=-1,vmax=1,cmap='jet',extent=[-4,4,-4,4],origin='lower')
        ax.plot([-4,4],[0,0],'w--')
        ax.plot([0,0],[-4,4],'w--')

        # compute the model's prediction
        cx = yHat[i][0].item() # center X
        cy = yHat[i][1].item() # center Y
        rd = yHat[i][2].item() # radius

        # and draw it
        x = cx + np.cos(th)*np.sqrt(rd)
        y = cy + np.sin(th)*np.sqrt(rd)
        ax.plot(x,y,'b')
        ax.plot(cx,cy,'bo')

        # some final plotting niceties
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([-4,4])
        ax.set_ylim([-4,4])
    plt.tight_layout()

    paramNames = ['Cx','Cy','rad.']

    plt.figure(figsize=(10,5), num='Corrrelation Plot')
    for i in range(3):
        # extract parameters and compute correlation
        yy = Y[:,i].detach()
        yh = yHat[:,i].detach()
        cr = np.corrcoef(yy,yh)[0,1]

        # plot with label
        plt.plot(yy,yh,'o',label=f'{paramNames[i]}, r={cr:.3f}')

    plt.legend()
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.grid()
    plt.show()
    

if __name__ == '__main__':
    main()

