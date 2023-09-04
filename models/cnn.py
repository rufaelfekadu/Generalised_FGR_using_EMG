import torch.nn as nn
import torch.nn.functional as functional
import torch
import random


class simpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.conv_1 = nn.Conv2d(1, 2, 2, padding='same')
        self.batch_norm_1 = nn.BatchNorm2d(2)
        self.conv_2 = nn.Conv2d(2, 4, 2, padding='same')
        self.batch_norm_2 = nn.BatchNorm2d(4)
        self.fc_1 = nn.Linear(4*4*4, 40)  # 4*4 from image dimension, 4 from num of filters
        self.batch_norm_3 = nn.BatchNorm1d(40)
        self.fc_2 = nn.Linear(40, 20)
        self.batch_norm_4 = nn.BatchNorm1d(20)
        self.fc_3 = nn.Linear(20, num_classes)

    def forward(self, x, feat=False):
        # add white gaussian noise to the input only during training
        if self.training and random.random() < 0:  # % chance to add noise to the batch (adjust to your needs)
            noise = torch.randn(x.shape) * 0.1 * (float(torch.max(x)) - float(torch.min(x)))  # up to 10% noise
            # move noise to the same device as x - super important!
            noise = noise.to(x.device)
            # add the noise to x
            x = x + noise
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = functional.relu(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = functional.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        if feat:
            return x
        x = self.batch_norm_3(x)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_2(x)
        x = self.batch_norm_4(x)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_3(x)
        x = functional.softmax(x, dim=1)
        return x
    

class simpleMLP(torch.nn.Module):
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.fc_1 = nn.Linear(4*4, 40)
        self.batch_norm_1 = nn.BatchNorm1d(40)
        self.fc_2 = nn.Linear(40, 20)
        self.batch_norm_2 = nn.BatchNorm1d(20)
        self.fc_3 = nn.Linear(20, num_classes)
    
    def forward(self, x):
        # add white gaussian noise to the input only during training
        if self.training and random.random() < 0:
            noise = torch.randn(x.shape) * 0.1 * (float(torch.max(x)) - float(torch.min(x)))
            noise = noise.to(x.device)
            x = x + noise
        
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.batch_norm_1(x)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_2(x)
        x = self.batch_norm_2(x)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_3(x)
        x = functional.softmax(x, dim=1)
        return x
