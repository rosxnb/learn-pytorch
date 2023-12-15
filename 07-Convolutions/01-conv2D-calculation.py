import numpy as np
import matplotlib.pyplot as plt


imageN = 20
kernelN = 7

def get_data():
    np.random.seed(32)
    image = np.random.randn( imageN, imageN )

    xx, yy = np.meshgrid( np.linspace(-3, 3, kernelN), np.linspace(-3, 3, kernelN) )
    kernel = np.exp( -(xx ** 2 + yy ** 2) / kernelN )

    return image, kernel


def plot_data():
    img, kernel = get_data()

    _, ax = plt.subplots(1, 2, figsize=(16, 7), num='Image and Kernel')
    ax[0].imshow(img)
    ax[0].set_title('Image')
    ax[1].imshow(kernel)
    ax[1].set_title('Convolution Image')
    plt.show()


def manual_2D_convolution():
    img, kernel = get_data()

    convolution =  np.zeros( (imageN, imageN) )
    kernel_center = kernelN // 2

    for row in range(kernel_center, imageN - kernel_center):
        for col in range(kernel_center, imageN - kernel_center):

            # cut out piece of img for dot product with kernel
            image_portion = img[ row - kernel_center : row + kernel_center + 1, : ] # select the rows
            image_portion = image_portion[ :, col - kernel_center : col + kernel_center + 1 ]

            # dot product with kernel (flip kernel both row and col wise for "real convolution" and not the cross correlation)
            dotprod = np.sum( image_portion * kernel[::-1, ::-1] )

            # store dot product in central pos relative to kernel
            convolution[row, col] = dotprod

    return convolution


def use_scipy():
    from scipy.signal import convolve2d

    img, kernel = get_data()
    convolution = convolve2d(img, kernel, mode='valid')

    return convolution


def main():
    manual = manual_2D_convolution()
    scpy = use_scipy()

    plot_data()

    _, ax = plt.subplots(1, 2, figsize=(16, 7), num='Manual and Scipy')
    ax[0].imshow(manual)
    ax[0].set_title('Manual convolution')
    ax[1].imshow(scpy)
    ax[1].set_title('Using scipy convolution')
    plt.show()


if __name__ == '__main__':
    main()

