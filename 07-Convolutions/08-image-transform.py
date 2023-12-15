import torch
import numpy as np
import matplotlib.pyplot as plt

# The list of datasets that come with torchvision: https://pytorch.org/vision/stable/index.html
import torchvision

# transformations list: https://pytorch.org/vision/stable/transforms.html
import torchvision.transforms as T


def main():
    cdata = torchvision.datasets.CIFAR10(root='cifar10', download=True)
    print(cdata)
    print()
    print(f'Shape of data:    {cdata.data.shape}')
    print(f'Categories label: {cdata.classes}')

    # inspect a few random images

    _, axs = plt.subplots(5, 5, figsize=(18, 10), num='Sample images')

    for ax in axs.flatten():

      # select a random picture
      randidx = np.random.choice(len(cdata.targets))

      # extract that image
      pic = cdata.data[randidx, :, :, :]

      # and its label
      label = cdata.classes[cdata.targets[randidx]]

      # and show!
      ax.imshow(pic)
      ax.text(16, 0, label, ha='center', fontweight='bold', color='k', backgroundcolor='y')
      ax.axis('off')

    plt.tight_layout()
    plt.show()


    # prepare the transformation
    Ts = T.Compose([ T.ToTensor(),                       # convert img to tensor while normalizing ([0, 255] to [0.0, 1.0])
                     T.Resize(32 * 4, antialias=None),   # scale image by factor of 4
                     T.Grayscale(num_output_channels=1)  # convert from RGB to Grayscale color scale
        ]) 
    
    # include the transform in the dataset
    cdata.transform = Ts
    
    # you can also apply the transforms immediately when loading in the data
    # cdata = torchvision.datasets.CIFAR10(root='cifar10', download=True, transform=Ts)
    
    # Important! Adding a transform doesn't change the image data:
    print(f'After transformation shape of an image: {cdata.data[123,:,:,:].shape}')
    # transformation is only applied when we want explicitly


    # apply the transform
    # option 1a: apply the transform "externally" to an image
    img1 = Ts( cdata.data[123,:,:,:] )

    # option 1b: use the embedded transform
    img2 = cdata.transform( cdata.data[123,:,:,:] )

    # let's see what we've done!
    _, ax = plt.subplots(1, 3, figsize=(10,3), num='Original image and Transformed images')
    ax[0].imshow(cdata.data[123,:,:,:])
    ax[1].imshow(torch.squeeze(img1))
    ax[2].imshow(torch.squeeze(img2), cmap='gray')

    plt.show()


if __name__ == '__main__':
    main()

