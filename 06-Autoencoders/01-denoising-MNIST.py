import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


def get_model():
    class MyAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input      = nn.Linear(784, 250)
            self.encoder    = nn.Linear(250, 50)
            self.latent     = nn.Linear(50, 250)
            self.decoder    = nn.Linear(250, 784)

        def forward(self, x):
            x = F.relu( self.input(x) )
            x = F.relu( self.encoder(x) )
            x = F.relu( self.latent(x) )

            # return with sigmoid because data will be pass normalized [0, 1]
            return F.sigmoid( self.decoder(x) )

    mymodel = MyAutoencoder()
    lossfunction = nn.MSELoss()
    optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.001)

    return mymodel, lossfunction, optimizer


def get_data():
    filepath = '../05-FNNs/data/mnist_train_small.csv'
    data = np.loadtxt(open(filepath, 'rb'), delimiter=',')

    labelT = torch.Tensor(data[:, 0])
    labelT = labelT.reshape((-1, 1))

    dataT = torch.Tensor(data[:, 1:])
    dataT = dataT / torch.max(dataT)

    return dataT, labelT


def train_model():
    data, _ = get_data()
    model, lossfunc, optimizer = get_model()

    epoches = 10_000
    losses = []

    for _ in range(epoches):
        # select random samples to prevent rote learning
        sample_size = 32
        randomIdx = np.random.choice(data.shape[0], size=sample_size)
        x = data[randomIdx, :]

        outputs = model(x)
        loss = lossfunc(outputs, x) # compare with original data to get loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append( loss.item() )

    return losses, model


def plot_images(original, predicted, wintitle='Original vs Model generated'):
    img_count = original.shape[0]
    _, ax = plt.subplots(2, img_count, figsize=(16, 7), num=wintitle)

    for i in range(5):
        ax[0, i].imshow( original[i, :].view(28, 28).detach(), cmap='gray' )
        ax[1, i].imshow( predicted[i, :].view(28, 28).detach(), cmap='gray' )
        ax[0, i].set_xticks([]), ax[0, i].set_yticks([])
        ax[1, i].set_xticks([]), ax[1, i].set_yticks([])

    plt.show()


def denoise_data_demo(trained_model):
    sample_size = 5
    data, _ = get_data()
    pollute_idx = np.random.choice(data.shape[0], size=sample_size)
    pollute_data = data[pollute_idx, :]
    pollute_data = pollute_data + torch.randn(pollute_data.shape) / 4

    outputs = trained_model(pollute_data)
    plot_images(pollute_data, outputs, 'Noisy vs Model generated')


def main():
    train_losses, model = train_model()

    plt.plot(train_losses, '.-')
    plt.title(f'Final loss = {train_losses[-1]}')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.show()

    data, _ = get_data()
    test_data = data[35:40, :]
    model_output = model(test_data)

    plot_images(test_data, model_output)
    denoise_data_demo(model)


if __name__ == '__main__':
    main()

