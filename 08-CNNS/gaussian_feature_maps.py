# visualize the feature maps of each convolution

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

from gaussian_blur_cnn import get_data


def get_model():
    class GaussNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
            # output size = (91 + 2*1 - 3) / 1 + 1 = 91
            # post pooling = 91 / 2 = 45

            self.conv2 = nn.Conv2d(6, 4, kernel_size=3, stride=1, padding=1)
            # output size = (45 + 2*1 - 3) / 1 + 1 = 45
            # post pooling = 45 / 2 = 22

            self.fc1 = nn.Linear( 4*22*22, 50 )
            self.output = nn.Linear( 50, 1 )

        def forward(self, data):
            conv1act = F.relu( self.conv1(data) )
            data = F.max_pool2d( conv1act, kernel_size=2 )

            conv2act = F.relu( self.conv2(data) )
            data = F.max_pool2d( conv2act, kernel_size=2 )

            data = data.reshape( data.shape[0], -1 )
            data = F.relu( self.fc1(data) )
            data = self.output(data)

            return data, conv1act, conv2act
    
    model = GaussNet()
    lossfunc = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    return model, lossfunc, optimizer


def test_run_model():
    # test the model with one batch
    model = get_model()[0]

    train_loader = get_data()[0]

    # test that the model runs and can compute a loss
    X, _ = next(iter(train_loader))
    yHat, featmap1, featmap2 = model(X)

    # check sizes of outputs
    print('Predicted category:')
    print(yHat.shape)
    print('\nFeature map after conv1')
    print(featmap1.shape)
    print('\nFeature map after conv2')
    print(featmap2.shape)

    summary(model, (1, 91, 91))


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
            yHat = model(X)[0]
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
            yHat = model(X)[0]
          
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

    # Feature maps from the conv1 layer
    X, y = next(iter(test_loader)) # extract X,y from test dataloader
    _, featmap1, featmap2 = model(X)
    _, axs = plt.subplots(7,10,figsize=(20, 9), num='1st feature maps')

    for pici in range(10):
        # show the original picture
        img = X[pici,0,:,:].detach()
        axs[0,pici].imshow(img,cmap='jet',vmin=0,vmax=1)
        axs[0,pici].axis('off')
        axs[0,pici].text(2,2,'T:%s'%int(y[pici].item()),ha='left',va='top',color='w',fontweight='bold')

        for feati in range(6):
            # extract the feature map from this image
            img = featmap1[pici,feati,:,:].detach()
            axs[feati+1,pici].imshow(img,cmap='inferno',vmin=0,vmax=torch.max(img)*.9)
            axs[feati+1,pici].axis('off')
            axs[feati+1,pici].text(-5,45,feati,ha='right') if pici==0 else None
    plt.tight_layout()
    plt.suptitle('First set of feature map activations for 10 test images',x=.5,y=1.01)
    plt.show()

    # Repeat for feature2 maps
    fig,axs = plt.subplots(5,10,figsize=(20, 9), num='2nd feature maps')
    for pici in range(10):
        img = X[pici,0,:,:].detach()
        axs[0,pici].imshow(img,cmap='jet',vmin=0,vmax=1)
        axs[0,pici].axis('off')

        for feati in range(4):
            img = featmap2[pici,feati,:,:].detach()
            axs[feati+1,pici].imshow(img,cmap='inferno',vmin=0,vmax=torch.max(img)*.9)
            axs[feati+1,pici].axis('off')
            axs[feati+1,pici].text(-5,22,feati,ha='right') if pici==0 else None
    plt.tight_layout()
    plt.suptitle('Second set of feature map activations for 10 test images',x=.5,y=1.01)
    plt.show()

    ### correlations across the SECOND convolution layer
    # convenient variables
    nStim = featmap2.shape[0]
    nMaps = featmap2.shape[1]
    nCors = (nMaps*(nMaps-1))//2

    # initialze the matrix of all correlation values
    allrs = np.zeros((nStim,nCors))
    Call  = np.zeros((nMaps,nMaps))

    # loop over each stimulus
    for i in range(nStim):
        # extract the vectorized feature maps from this image
        featmaps = featmap2[i,:,:,:].view(nMaps,-1).detach()

        # compute the correlation matrix
        C = np.corrcoef(featmaps)
        Call += C

        # extract the unique correlations from the matrix
        idx = np.nonzero(np.triu(C,1))
        allrs[i,:] = C[idx]


    # define the x-axis labels
    xlab = [] * nCors

    for i in range(nCors):
        xlab.append('%s-%s' %(idx[0][i],idx[1][i]))

    # now visualize the correlations
    fig = plt.figure(figsize=(20, 9), num='Correlation across 2nd conv layer')
    ax0 = fig.add_axes([.1,.1,.55,.9]) # [left, bottom, width, height]
    ax1 = fig.add_axes([.68,.1,.3,.9])
    cax = fig.add_axes([.98,.1,.01,.9])

    for i in range(nCors):
        ax0.plot(i+np.random.randn(nStim)/30,allrs[:,i],'o',markerfacecolor='w',markersize=10)

    # make the plot more interpretable
    ax0.set_xlim([-.5,nCors-.5])
    ax0.set_ylim([-1.05,1.05])
    ax0.set_xticks(range(nCors))
    ax0.set_xticklabels(xlab)
    ax0.set_xlabel('Feature map pair')
    ax0.set_ylabel('Correlation coefficient')
    ax0.set_title('Correlations for each image')

    # now show the average correlation matrix
    h = ax1.imshow(Call/nStim,vmin=-1,vmax=1)
    ax1.set_title('Correlation matrix')
    ax1.set_xlabel('Feature map')
    ax1.set_ylabel('Feature map')
    
    # add a colorbar
    fig.colorbar(h,cax=cax)
    plt.show()

    ### correlations across the FIRST convolution layer
    # convenient variables
    nStim = featmap1.shape[0]
    nMaps = featmap1.shape[1]
    nCors = (nMaps*(nMaps-1))//2

    # initialze the matrix of all correlation values
    allrs = np.zeros((nStim,nCors))
    Call  = np.zeros((nMaps,nMaps))

    # loop over each stimulus
    for i in range(nStim):
        # extract the vectorized feature maps from this image
        featmaps = featmap1[i,:,:,:].view(nMaps,-1).detach()

        # compute the correlation matrix
        C = np.corrcoef(featmaps)
        Call += C

        # extract the unique correlations from the matrix
        idx = np.nonzero(np.triu(C,1))
        allrs[i,:] = C[idx]

    # define the x-axis labels
    xlab = []*nCors
    for i in range(nCors):
        xlab.append('%s-%s' %(idx[0][i],idx[1][i]))

    # now visualize the correlations
    fig = plt.figure(figsize=(20, 9), num='Correlation accross 1st conv layer')
    ax0 = fig.add_axes([.1,.1,.55,.9]) # [left, bottom, width, height]
    ax1 = fig.add_axes([.68,.1,.3,.9])
    cax = fig.add_axes([.98,.1,.01,.9])

    for i in range(nCors):
        ax0.plot(i+np.random.randn(nStim)/30,allrs[:,i],'o',markerfacecolor='w',markersize=10)

    # make the plot more interpretable
    ax0.set_xlim([-.5,nCors-.5])
    ax0.set_ylim([-1.05,1.05])
    ax0.set_xticks(range(nCors))
    ax0.set_xticklabels(xlab)
    ax0.set_xlabel('Feature map pair')
    ax0.set_ylabel('Correlation coefficient')
    ax0.set_title('Correlations for each image')

    # now show the average correlation matrix
    h = ax1.imshow(Call/nStim,vmin=-1,vmax=1)
    ax1.set_title('Correlation matrix')
    ax1.set_xlabel('Feature map')
    ax1.set_ylabel('Feature map')

    # add a colorbar
    fig.colorbar(h,cax=cax)
    plt.show()


if __name__ == '__main__':
    main()

