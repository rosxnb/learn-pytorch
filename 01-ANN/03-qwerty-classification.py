# ANN for classifying "qwerties"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # generate data
    a = [1, 1]
    b = [5, 1]
    nPerCluster = 50
    blur = 1

    x = [a[0] + np.random.randn(nPerCluster) * blur, a[1] + np.random.randn(nPerCluster) * blur]
    y = [b[0] + np.random.randn(nPerCluster) * blur, b[1] + np.random.randn(nPerCluster) * blur]

    labels_np = np.vstack((np.zeros((nPerCluster, 1)), np.ones((nPerCluster, 1))))
    data_np = np.hstack((x, y)).T
    data = torch.tensor(data_np, dtype=torch.float)
    label = torch.tensor(labels_np, dtype=torch.float)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5), num='Qwerty Classification')
    # fig.canvas.manager.set_window_title('Qwerty Classification')

    # visualize the data
    ax[0].plot( data[ np.where( label == 0 )[0], 0 ], data[ np.where( label == 0 )[0], 1 ], 'bs', label='label 0' )
    ax[0].plot( data[ np.where( label == 1 )[0], 0 ], data[ np.where( label == 1 )[0], 1 ], 'ko', label='label 1' )
    ax[0].set_title('Qwerties')
    ax[0].set_xlabel('Querty Dim1')
    ax[0].set_ylabel('Querty Dim2')
    ax[0].legend()

    # architect model
    AnnClassifier = nn.Sequential(
        nn.Linear(2, 1),
        nn.ReLU(),
        nn.Linear(1, 1),
        nn.Sigmoid() # depricated, rather use nn.BCEWithLogitsLoss which does sigmoid internally
    )
    print('Model architecture:')
    print(AnnClassifier)

    lr = 0.01
    lossFunc = nn.BCELoss()
    optimizer = torch.optim.SGD(AnnClassifier.parameters(), lr=lr)

    # model training
    trainEpoches = 1000
    trainLosses = np.zeros(trainEpoches)
    
    for epochi in range(trainEpoches):
        yhat = AnnClassifier(data)
        loss = lossFunc(yhat, label)
        trainLosses[epochi] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # visulize the losses
    ax[1].plot(trainLosses, 'bo')
    ax[1].set_ylabel('Training Losses')
    ax[1].set_xlabel('Training Epoch')
    ax[1].set_title('Loss vs Training Epoches')

    # visualize the misclassified labels
    prediction = AnnClassifier(data)
    predlabels = prediction > 0.5
    misclassified = np.where( predlabels != label )[0]
    totalacc = 100 - 100 * len(misclassified) / (2 * nPerCluster)

    # visualize misclassified labels
    ax[2].plot(data[misclassified, 0] ,data[misclassified, 1], 'rx', markersize=12, markeredgewidth=3)
    ax[2].plot(data[np.where(~predlabels)[0],0], data[np.where(~predlabels)[0],1], 'bs')
    ax[2].plot(data[np.where(predlabels)[0],0], data[np.where(predlabels)[0],1], 'ko')
    ax[2].legend(['Misclassified','blue','black'], bbox_to_anchor=(1, 1))
    ax[2].set_title(f'{totalacc}% correct')

    plt.show()
