import torch
import torch.nn as nn


def main():
    poolSize = 3
    stride = 3

    p2 = nn.MaxPool2d(poolSize, stride)
    p3 = nn.MaxPool3d(poolSize, stride)

    print(f'2D pooling: \n{p2}\n')
    print(f'3D pooling: \n{p3}\n')

    # create a 2D and a 3D image
    img2 = torch.randn(1,1,30,30)
    img3 = torch.randn(1,3,30,30)
    
    
    # all combinations of image and maxpool dimensionality
    img2Pool2 = p2(img2)
    print(f'2D image, 2D maxpool: {img2Pool2.shape}\n' )
    
    # img2Pool3 = p3(img2)
    # print(f'2D image, 3D maxpool: {img2Pool3.shape}\n' )
    
    img3Pool2 = p2(img3)
    print(f'3D image, 2D maxpool: {img3Pool2.shape}\n' )
    
    img3Pool3 = p3(img3)
    print(f'3D image, 3D maxpool: {img3Pool3.shape}\n' )


if __name__ == '__main__':
    main()

