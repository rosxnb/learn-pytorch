import torch
import numpy as np
import matplotlib.pyplot as plt

from mnist_cnn import get_data, train_model


def shift_loader(data_loader):
    for i in range(data_loader.dataset.tensors[0].shape[0]):
        img = data_loader.dataset.tensors[0][i, :, :]

        # reshape and roll by 10 pixels
        randroll = np.random.randint(-10, 11)
        img = torch.roll( img, randroll, dims = 1 )

        # re-vectorize and put back to data_loader
        data_loader.dataset.tensors[0][i, :, :] = img


def plot_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def main():
    train_loader, test_loader = get_data()
    shift_loader(train_loader)
    shift_loader(test_loader)

    # img = train_loader.dataset.tensors[0][29, :, :]
    # img = torch.squeeze(img).detach()
    # plot_image(img)

    train_accuracies, test_accuracies, losses, _ = train_model(train_loader, test_loader)
    _, ax = plt.subplots(1, 2, figsize=(16,6), num='MNIST CNN')
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


if __name__ == '__main__':
    main()
