import torch
from torch import nn
from cnn import simpleCNN, simpleMLP
from eegnet import EEGNet
from adv_network import Classifier, Net

model_dict = {
    'cnn': simpleCNN,
    'mlp': simpleMLP,
    'eegnet': EEGNet
}


class BaseModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = model_dict[model_name](num_classes, dropout_rate)
    
    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def train(model, train_loader, optimizer, criterion, device):
        model.train()
        train_loss = 0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        train_loss /= len(train_loader.dataset)
        accuracy = 100. * correct / len(train_loader.dataset)
        
        return train_loss, accuracy

class BaseAdvModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.classifier = model_dict[model_name](num_classes, dropout_rate)
        self.discriminator = nn.Sequential(
            nn.Linear(4*4, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

    @staticmethod
    def train(model, train_loader, optimizer, criterion, device):

        pass