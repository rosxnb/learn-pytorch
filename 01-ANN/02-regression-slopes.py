# Parametric Expriment: varying slopes and observing the its effect on model
# Exprimemt can take some time to conduct

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def build_and_trian_model(x, y):
    model = nn.Sequential(
        nn.Linear(1, 1),
        nn.ReLU(),
        nn.Linear(1, 1)
    )

    lr = 0.01
    lossFunc = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    trainEpoches = 500
    trainLosses = torch.zeros(trainEpoches)
    for epochi in range(trainEpoches):
        yHat = model(x)

        loss = lossFunc(yHat, y)
        trainLosses[epochi] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = model(x)
    return predictions, trainLosses


def create_data(slope, N=50):
    x = torch.randn(N, 1)
    y = slope * x + torch.randn(N, 1) / 2
    return x, y


# use this function to visualize how m effects data
# m = 0 vs m = 2
def visualize_data_with_slope(slope):
    x, y = create_data(slope)
    plt.plot(x, y, 'ko')
    plt.ylim([-6, 6])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Data with slope {slope}')
    plt.show()


if __name__ == '__main__':
    slopes = np.linspace(-2, 2, 21) # create slopes in range -2 to 2 in 21 steps
    nExpriments = 50
    results = np.zeros((len(slopes), nExpriments, 2))

    for slopeidx, slope in enumerate(slopes):
        for nexpriment in range(nExpriments):
            x, y = create_data(slope)
            yhat, losses = build_and_trian_model(x, y)

            results[slopeidx, nexpriment, 0] = losses[-1]
            results[slopeidx, nexpriment, 1] = np.corrcoef(y.T, yhat.data.T)[0, 1]

    # correlation can be 0 if model didn't do well
    results[np.isnan(results)] = 0

    # visualize
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(slopes, np.mean(results[:, :, 0], axis=1), 'ko-', markerfacecolor='w', markersize=10)
    ax[0].set_xlabel('Slope')
    ax[0].set_title('Loss')
    ax[1].plot(slopes, np.mean(results[:, :, 1], axis=1), 'ms-', markerfacecolor='w', markersize=10)
    ax[1].set_xlabel('Slope')
    ax[1].set_ylabel('Correlation')
    ax[1].set_title('Model Performance')
    plt.show()
