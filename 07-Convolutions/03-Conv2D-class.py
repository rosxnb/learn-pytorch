import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt


input_channels = 3
output_channels = 15 # each output feature map requires a kernel so 15 kernels
kernelN = 5 # (5x5) matrix
stride = 1 # default
padding = 0


def visualize_convolution_kernels():
    conv = nn.Conv2d(input_channels, output_channels, kernelN, stride, padding)

    print(f'Size of convolution weight: {conv.weight.shape}')
    print(f'Size of convolution bias: {conv.bias.shape}')

    _, axs = plt.subplots(3, 5, figsize=(20, 8), num='Convolution Kernels')
    for idx, ax in enumerate(axs.flatten()):
        ax.imshow( torch.squeeze( conv.weight[idx, 0, :, :] ).detach(), cmap='Purples' ) # we are only looking at first color channel
        ax.set_title('L1(0) -> L(%s)' %idx)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def convolve_image():
    img_dimensions = (1, 3, 64, 64) # batchsize, RGB, height, width
    image = torch.rand(img_dimensions)

    img_for_plt = image.permute(2, 3, 1, 0).numpy()
    print( f'Image dimension for pytorch:    {img_dimensions}' )
    print( f'Image dimension for matplotlib: {img_for_plt.shape}' )

    plt.imshow(np.squeeze(img_for_plt))
    plt.title('Generated image')
    plt.show()

    conv = nn.Conv2d(input_channels, output_channels, kernelN, stride, padding)
    output_image = conv(image)

    print(f'Size of convolution output of image: {output_image.shape}')

    _, axs = plt.subplots(3, 5, figsize=(20, 8), num=f'{output_channels} feature maps produced as convolution')
    for idx, ax in enumerate(axs.flatten()):
        feature_map = output_image[0, idx, :, :]
        feature_map = torch.squeeze(feature_map).detach()

        ax.imshow(feature_map, cmap='Purples')
        ax.set_title(f'feature map {idx}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    convolve_image()


if __name__ == '__main__':
    main()

