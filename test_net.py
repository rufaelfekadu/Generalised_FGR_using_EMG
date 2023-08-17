import torch
import torch.nn as nn
import torch.nn.functional as functional
import random
from pathlib import Path
import torchvision.models as md
import numpy as np

from torch.utils.data import Dataset, DataLoader


import sys
sys.path.append('../Fingers-Gesture-Recognition')
import Source.fgr.models as models
from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager

from models import make_model, vision, Net
from dataset import emgdata

from utils import preprocess_data, get_logger



# train function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, (target, _)) in enumerate(train_loader): # data: (batch_size, 3, 512, 512)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # set gradient to zero
        output = model(data) # output: (batch_size, 10)
        loss = torch.nn.functional.cross_entropy(output, target) # loss: (batch_size)
        loss.backward() # back propagation
        optimizer.step() # update parameters
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item() # sum up correct predictions


    #print loss and accuracy
    print('Train Epoch: {} \tLoss: {:.6f} \tAccuracy: {}/{} ({:.00f}%)'.format(
        epoch, loss.item(), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    
def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): # no gradient calculation
        for data, (target,_) in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # sum up correct predictions
            total += target.size(0)
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.00f}%)\n'.format(test_loss, correct, total, (correct/total)*100) )# print loss and accuracy

    return correct/total


def main():

    #setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pipeline definition and data manager creation
    data_path = Path('../data/doi_10')
    subject = 1

    datest = emgdata(data_path, subject)

    train_data, test_data = torch.utils.data.random_split(datest, [int(len(datest)*0.8), len(datest)-int(len(datest)*0.8)])
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

    # model definition
    # model = vision(image_size=4, 
    #                 patch_size=2, 
    #                 num_classes=10, 
    #                 hidden_dim=64, 
    #                 num_layers=1, num_heads=2, mlp_dim=40, attention_dropout=0.1).to(device)
    model = Net(num_classes=10).to(device)
    print(model)
    #print number of parameters
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    #optimizer definition
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(100):
        
        # train and test
        train(model, device, train_loader, optimizer, i)
        test(model, test_loader, device)
    
    #save model
    torch.save(model.state_dict(), './outputs/model_cnn.pt')




if __name__ == '__main__':
    main()