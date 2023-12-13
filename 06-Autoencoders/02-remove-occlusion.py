import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


def get_data():
    filepath = '../05-FNNs/data/mnist_train_small.csv'
    data = np.loadtxt( open(filepath, 'rb'), delimiter=',' )

    label = data[:, 0]
    data = data[:, 1:]
    data = data / np.max(data)

    data = torch.Tensor( data )
    label = torch.Tensor( label )

    return data, label


def get_model():
    class MyAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(784, 170)
            self.encoder = nn.Linear(170, 30)
            self.latent = nn.Linear(30, 170)
            self.decoder = nn.Linear(170, 784)

        def forward(self, data):
            data = F.relu( self.input(data) )
            data = F.relu( self.encoder(data) )
            data = F.relu( self.latent(data) )
            
            return F.sigmoid( self.decoder(data) )

    model = MyAutoencoder()
    lossfunc = nn.MSELoss()
    optimizer = torch.optim.Adam( model.parameters(), lr=0.001 )

    return model, lossfunc, optimizer


def plot_images(original, predicted, wintitle='Original vs Model generated'):
    img_count = original.shape[0]
    _, ax = plt.subplots(2, img_count, figsize=(16, 7), num=wintitle)

    for i in range(5):
        ax[0, i].imshow( original[i, :].view(28, 28).detach(), cmap='gray' )
        ax[1, i].imshow( predicted[i, :].view(28, 28).detach(), cmap='gray' )
        ax[0, i].set_xticks([]), ax[0, i].set_yticks([])
        ax[1, i].set_xticks([]), ax[1, i].set_yticks([])

    plt.show()


def train_model():
    data = get_data()[0]
    model, lossfunc, optimizer = get_model()

    epochs = 5
    batch_size = 32
    batch_count = int( data.shape[0] / batch_size )

    losses = []
    for _ in range(epochs):
        random_data_indices = np.random.permutation(data.shape[0]).astype(int)

        for batch in range(batch_count):
            batch_range = range(batch * batch_size, batch * batch_size + batch_size)
            batch_data = data[ random_data_indices[batch_range], : ]

            outputs = model(batch_data)
            loss = lossfunc(outputs, batch_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append( loss.item() )

    return losses, model


def main():
    data = get_data()[0]
    losses, model = train_model()

    plt.plot(losses, '.-')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.show()

    random_test_indices = np.random.choice(data.shape[0], 5)
    test_data = data[random_test_indices, :]
    output = model(test_data)
    plot_images(test_data, output)

    # distort the images
    for sample in range(test_data.shape[0]):
        img = test_data[sample, :].view(28, 28)

        pos = np.random.choice(np.arange(10, 21))
        if (pos & 1):
            img[ pos : pos + 1, : ] = 1
        else:
            img[ :, pos : pos + 1 ] = 1

    deocclude = model(test_data)
    plot_images(test_data, deocclude, 'Occluded img to Deocclusion')


if __name__ == '__main__':
    main()

