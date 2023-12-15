import torch
import torch.nn as nn


def main():
    littlenet = nn.Sequential(

        ## the conv-pool block
        nn.Conv2d(3,10,5,3,2), # convolution layer
        nn.ReLU(),             # activation function
        nn.AvgPool3d(3,3),     # average-pool

        ## the FFN block
        nn.Flatten(),          # vectorize to get from image to linear
        nn.Linear(588,1),      # FC linear layer
        nn.Sigmoid()           # output activation
    )

    img = torch.rand(1,3,128,128)
    output = littlenet(img)
    print(output)


if __name__ == '__main__':
    main()

