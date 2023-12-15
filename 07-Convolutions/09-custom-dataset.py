# review paper on data augmentation in DL:
# https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0

import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset,DataLoader


class CustomDataset(Dataset):
    def __init__(self, tensors, transform=None):
        assert all(t.size(0) == tensors[0].size(0) for t in tensors), 'Size mismatch between tensors provided for dataset'

        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, idx):
        x = self.tensors[0][idx]
        y = self.tensors[1][idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def main():
    filepath = '../05-FNNs/data/mnist_train_small.csv'
    data = np.loadtxt( open(filepath, 'rb'), delimiter=',' )

    labels = data[:8,0]
    data   = data[:8,1:]

    # normalize the data to a range of [0 1]
    dataNorm = data / np.max(data)

    # reshape to 2D!
    dataNorm = dataNorm.reshape(dataNorm.shape[0],1,28,28)

    # check sizes
    print(dataNorm.shape)
    print(labels.shape)

    # convert to torch tensor format
    dataT   = torch.tensor( dataNorm ).float()
    labelsT = torch.tensor( labels ).long()

    # create a list of transforms to apply to the image
    imgtrans = T.Compose([ 
        T.ToPILImage(),
        T.RandomVerticalFlip(p=.5),
        # T.RandomRotation(90), 
        T.ToTensor()
    ])

    # convert into PyTorch Datasets
    # NOTE: we have no test data here, but you should apply the same transformations to the test data
    train_data = CustomDataset((dataT, labelsT), imgtrans)

    # translate into dataloader objects
    dataLoaded = DataLoader(train_data, batch_size=8, shuffle=False)

    # import data from the dataloader, just like during training
    X, _ = next(iter(dataLoaded))


    # create a figure
    _, axs = plt.subplots(2, 8, figsize=(16,4), num='Transform Images with CustomDataset')


    # loop over images in the dataset
    for i in range(8):

      # draw images
      axs[0,i].imshow(dataT[i,0,:,:].detach(), cmap='gray')
      axs[1,i].imshow(X[i,0,:,:].detach(), cmap='gray')

      # some niceties
      for row in range(2):
        axs[row,i].set_xticks([])
        axs[row,i].set_yticks([])

    # row labels
    axs[0,0].set_ylabel('Original')
    axs[1,0].set_ylabel('torch dataset')

    plt.show()

    print(len(train_data))


if __name__ == '__main__':
    main()

