import torch
from torch import  nn
from torch.nn import functional
import random

class Net(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(Net, self).__init__()

        self.dropout_rate = 0.3

        self.encoder = nn.Sequential(
                 nn.Conv2d(1, 2, 2, padding='same'),
                    nn.BatchNorm2d(2),
                    nn.ReLU(),
                    nn.Conv2d(2, 4, 2, padding='same'),
                    nn.BatchNorm2d(4),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(4*4*4, 40),
                    nn.BatchNorm1d(40),
                    nn.ReLU(),
                    nn.Dropout(p=self.dropout_rate))
        
        self.classifier = nn.Sequential(
                nn.Linear(40, 20),
                nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(20, num_classes),
                nn.Softmax(dim=1)
            )
        
    def forward(self, x, feat=False):
        # add white gaussian noise to the input only during training
        if self.training and random.random() < 0:  # % chance to add noise to the batch (adjust to your needs)
            noise = torch.randn(x.shape) * 0.1 * (float(torch.max(x)) - float(torch.min(x)))  # up to 10% noise
            # move noise to the same device as x - super important!
            noise = noise.to(x.device)
            # add the noise to x
            x = x + noise

        x = self.encoder(x)
        if feat:
            return x
        
        x = self.classifier(x)
        return x
    
class Classifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(40, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.softmax(x, dim=1)
        return x
    
class FeatureExtractor(nn.Module):
    def __init__(self, dropout_rate: float = 0.3, target=False):
            super().__init__()
            self.dropout_rate = dropout_rate

            self.encoder = nn.Sequential(
                 nn.Conv2d(1, 2, 2),
                    nn.BatchNorm2d(2),
                    nn.ReLU(),
                    nn.Conv2d(2, 4, 2),
                    nn.BatchNorm2d(4),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(4*4*4, 40),
                    nn.BatchNorm1d(40),
                    nn.ReLU(),
                    nn.Dropout(p=self.dropout_rate))
    def forward(self, x):
        # add white gaussian noise to the input only during training
        if self.training and random.random() < 0:
            noise = torch.randn(x.shape) * 0.1 * (float(torch.max(x)) - float(torch.min(x)))
            noise = noise.to(x.device)
            x = x + noise
        x = self.encoder(x)
        return x


            
# specify model
class simpleCNN(torch.nn.Module):
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3, target=False):
            super().__init__()
            self.dropout_rate = dropout_rate

            self.encoder = nn.Sequential(
                 nn.Conv2d(1, 2, 2, padding='same'),
                    nn.BatchNorm2d(2),
                    nn.ReLU(),
                    nn.Conv2d(2, 4, 2, padding='same'),
                    nn.BatchNorm2d(4),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(4*4*4, 40),
                    nn.BatchNorm1d(40),
                    nn.ReLU(),
                    nn.Dropout(p=self.dropout_rate))
            
            self.classifier = nn.Sequential(
                nn.Linear(40, 20),
                nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(20, num_classes),
                nn.Softmax(dim=1)
            )

            if target:
                for param in self.classifier.parameters():
                    param.requires_grad = False



    def forward(self, x):
        # add white gaussian noise to the input only during training
        if self.training and random.random() < 0:  # % chance to add noise to the batch (adjust to your needs)
            noise = torch.randn(x.shape) * 0.1 * (float(torch.max(x)) - float(torch.min(x)))  # up to 10% noise
            # move noise to the same device as x - super important!
            noise = noise.to(x.device)
            # add the noise to x
            x = x + noise
       
        x = self.encoder(x)
        x = self.classifier(x)
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
        # x = self.fc_3(x)
        # x = functional.softmax(x, dim=1)
        return x

def get_adv(args):

    source_cnn = simpleCNN(num_classes=args.num_classes, dropout_rate=args.dropout_rate, target=args.target)
    target_cnn = simpleCNN(num_classes=args.num_classes, dropout_rate=args.dropout_rate, target=args.target)
    discriminator = Classifier()

    return source_cnn.to(args.device), target_cnn.to(args.device), discriminator.to(args.device)
