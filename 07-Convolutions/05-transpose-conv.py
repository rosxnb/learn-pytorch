import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


inChannels = 3
outChannels = 15
kernelN = 7
stride = (2, 3)
padding = 1

def main():
    conv = nn.ConvTranspose2d(inChannels, outChannels, kernelN, stride, padding)

    print(f'Transpose Convolution:\n {conv}')
    print()
    print(f'Weight shape: {conv.weight.shape}')
    print(f'Bias shape: {conv.bias.shape}')

    imageSize = (1, 3, 64, 64)
    img = torch.rand(imageSize)

    output = conv(img)

    _, axx = plt.subplots(3, 5, figsize=(18, 6), num='Transpose Convolution')
    for idx, ax in enumerate(axx.flatten()):
        feature_map = output[0, idx, :, :].detach()
        feature_map = np.squeeze(feature_map)

        ax.imshow(feature_map, cmap='Purples')
        ax.axis('off')
        ax.set_title(f'{idx}th feature-map')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

