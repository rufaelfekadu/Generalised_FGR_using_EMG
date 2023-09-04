from torch import nn
import torch

class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropoutRate=0.5, kernLength=64, F1=8,
                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()

        if dropoutType == 'SpatialDropout2D':
            self.dropoutType = nn.Dropout2d
        elif dropoutType == 'Dropout':
            self.dropoutType = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (Chans, 1), bias=False, groups=F1),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            self.dropoutType(p=dropoutRate)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 8), padding=(0, 7), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            self.dropoutType(p=dropoutRate)
        )
         
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(F2 * (Samples // 32), nb_classes),
            nn.ReLU(),
            nn.Linear(nb_classes, nb_classes)
        )

    def forward(self, x):
        x = self.block1(x) 
        print(x.shape)
        x = self.block2(x) 

        x = self.flatten(x)
        print(x.shape)
        x = self.dense(x) #input shape: (batch_size, F2 * (Samples // 32)
        return x 


if __name__ == '__main__':
    model = EEGNet(2)
    print(model)
    x = torch.randn(1, 1, 64, 128)
    print(model(x).shape)



