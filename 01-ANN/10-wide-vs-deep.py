import numpy as np
import torch.nn as nn


if __name__ == '__main__':
    wideModel = nn.Sequential(
        nn.Linear(2, 4),
        nn.Linear(4, 3)
    )

    deepModel = nn.Sequential(
        nn.Linear(2, 2),
        nn.Linear(2, 2),
        nn.Linear(2, 3)
    )

    print('---------------------------------------------------------')
    print('Wide Model:')
    print(wideModel)
    print()

    print('Wide Model Parameters:')
    for p in wideModel.named_parameters():
        print(p)
        print()

    print('---------------------------------------------------------')
    print('Deep Model:')
    print(deepModel)
    print()

    print('Deep Model Parameters:')
    for p in deepModel.named_parameters():
        print(p)
        print()
    
    print('---------------------------------------------------------')
    print('Nodes Count In Wide vs Deep Networks')
    print('---------------------------------------------------------')

    nNodesInWideNet = 0
    for name, weight_matrix in wideModel.named_parameters():
        # count nodes by counting bias
        if 'bias' in name:
            nNodesInWideNet += len(weight_matrix)

    nNodesInDeepNet = 0
    for name, weight_matrix in deepModel.named_parameters():
        # count nodes by counting bias
        if 'bias' in name:
            nNodesInDeepNet += len(weight_matrix)

    print(f'\t Nodes count in wide net: {nNodesInWideNet}')
    print(f'\t Nodes count in wide net: {nNodesInDeepNet}')
    
    print('---------------------------------------------------------')
    print('Parameters Count In Wide vs Deep Networks')
    print('---------------------------------------------------------')

    # filtering by 'requires_grad' to count only the trianable parameters
    nParamsInWideNet = np.sum([ p.numel() for p in wideModel.parameters() if p.requires_grad ])
    nParamsInDeepNet = np.sum([ p.numel() for p in deepModel.parameters() if p.requires_grad ])

    print(f'\t Parameters count in wide net: {nParamsInWideNet}')
    print(f'\t Parameters count in wide net: {nParamsInDeepNet}')
    
