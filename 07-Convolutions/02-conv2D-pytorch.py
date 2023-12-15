import numpy as np
import matplotlib.pyplot as plt

from imageio import imread

import torch
import torch.nn.functional as F


def main():
    image = imread('https://upload.wikimedia.org/wikipedia/commons/6/61/De_nieuwe_vleugel_van_het_Stedelijk_Museum_Amsterdam.jpg')

    # transform 3D image to 2D for convinence (not necessary for convolution)
    image = np.mean(image, axis=2)
    image = image / np.max(image)

    # manual kernel like traditional ways
    vertical_kernel = np.array([ [1, 0, -1],
                                 [1, 0, -1],
                                 [1, 0, -1] ])

    horizontal_kernel = np.array([ [ 1,  1,  1],
                                   [ 0,  0,  0],
                                   [-1, -1, -1] ])

    VK = torch.tensor( vertical_kernel ).view(1, 1, 3, 3).double()
    HK = torch.tensor( horizontal_kernel ).view(1, 1, 3, 3).double()
    imgT = torch.tensor( image ).view(1, 1, image.shape[0], image.shape[1])

    vConv = F.conv2d(imgT, VK)
    hConv = F.conv2d(imgT, HK)

    _, ax = plt.subplots(2, 2, figsize=(18, 7))
    ax[0, 0].imshow(vertical_kernel)
    ax[0, 0].set_title('Vertical Kernel')
    ax[0, 1].imshow(horizontal_kernel)
    ax[0, 1].set_title('Horizontal Kernel')
    ax[1, 0].imshow( torch.squeeze(vConv).detach(), cmap='gray', vmin=0, vmax=.09 )
    ax[1, 0].set_title('Vertical Convolution')
    ax[1, 1].imshow( torch.squeeze(hConv).detach(), cmap='gray', vmin=0, vmax=.09 )
    ax[1, 1].set_title('Horizontal Convolution')
    plt.show()


if __name__ == '__main__':
    main()

