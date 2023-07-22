import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import torch
from torch import nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder
import logging
from trainner import adversarial_domain, train_target_cnnP_domain

import sys
sys.path.append('/home/rufael.marew/Documents/projects/tau/Fingers-Gesture-Recognition')

import numpy as np
import Source.fgr.models as models

from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = functional.softmax(x, dim=1)
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
    



# train function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): # data: (batch_size, 3, 512, 512)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # set gradient to zero
        output = model(data) # output: (batch_size, 10)
        loss = torch.nn.functional.cross_entropy(output, target) # loss: (batch_size)
        loss.backward() # back propagation
        optimizer.step() # update parameters

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( # print loss
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): # no gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # sum up correct predictions
            total += target.size(0)
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.00f}%)\n'.format(test_loss, correct, total, (correct/total)*100) )# print loss and accuracy

    return correct/total

def preprocess_data(dataset):

    data = dataset[0]
    labels = dataset[1]

    data = torch.Tensor(data)
    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(np.char.strip(labels, '_0123456789'))
    labels = torch.Tensor(labels).to(torch.int64)
  
    dataset = TensorDataset(data, labels)

    # split data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # data loader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    return train_loader, test_loader

# main function
def main(args):

    # data_dir = '../data/doi_10/emg'


    logger = logging.getLogger(__name__)
    # get data
    train_transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                ])
            
    # dataset = datasets.ImageFolder(data_dir+'/001_1', transform=train_transform)
    
    # pipeline definition and data manager creation
    data_path = Path('../data/doi_10')
    pipeline = Data_Pipeline(base_data_files_path=data_path)  # configure the data pipeline you would like to use (check pipelines module for more info)
    subject = 1
    dm = Data_Manager([subject], pipeline)
    print(dm.data_info())

    source_dataset = dm.get_dataset(experiments=[f'{subject:03d}_1_1'])
    target_datset = dm.get_dataset(experiments=[f'{subject:03d}_1_2'])

    source_train_loader, source_test_loader = preprocess_data(source_dataset)
    target_train_loader, target_test_loader = preprocess_data(target_datset)

    # for i in range(100):
    #     #generate images for the first 10 samples
    #     Path(f'../../data/doi_10/emg_test/001_1/{train_dataset[i][1]}').mkdir(parents=True, exist_ok=True)
    #     img = train_dataset[i][0].numpy().reshape(4,4)
    #     image = Image.fromarray(img, mode="L")
    #     image.save(f'../../data/doi_10/emg_test/001_1/{train_dataset[i][1]}/{i}.png')


    # specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    source_cnn = simpleCNN().to(device)

    # train target CNN
    target_cnn = simpleCNN(target=True).to(device)
    optimizer = torch.optim.Adam(
            target_cnn.encoder.parameters(), 
            lr=args.lr, betas=args.betas, 
            weight_decay=args.weight_decay)

    discriminator = Discriminator().to(device)
    criterion = nn.CrossEntropyLoss()
    d_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=args.d_lr, betas=args.betas, weight_decay=args.weight_decay)
    best_acc, best_class, classNames = train_target_cnnP_domain(
        source_cnn, target_cnn, discriminator,
        criterion, optimizer, d_optimizer,
        source_train_loader, target_train_loader, target_test_loader,
        logger, args=args)
    bestClassWiseDict = {}
    for cls_idx, clss in enumerate(classNames):
        bestClassWiseDict[clss] = best_class[cls_idx].item()
    logger.info('Best acc.: {}'.format(best_acc))
    logger.info('Best acc. (Classwise):')
    logger.info(bestClassWiseDict)

    # specify optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # train
    # for epoch in range(1, 100):
    #     train(model, device, train_loader, optimizer, epoch)
    #     test(model, test_loader, device=device)

    adversarial_domain()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # NN
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--trained', type=str, default='')
    parser.add_argument('--slope', type=float, default=0.2)
    # train
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--d_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2.5e-5)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--betas', type=float, nargs='+', default=(.5, .999))
    parser.add_argument('--lam', type=float, default=0.25)
    parser.add_argument('--thr', type=float, default=0.79)
    parser.add_argument('--thr_domain', type=float, default=0.87)
    parser.add_argument('--num_val', type=int, default=3)  # number of val. within each epoch
    # misc
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='outputs/sgada_domain')
    # office dataset categories
    parser.add_argument('--src_cat', type=str, default='mscoco')
    parser.add_argument('--tgt_cat', type=str, default='flir')
    parser.add_argument('--tgt_conf_cat', type=str, default='flir_confident')
    parser.add_argument('--message', type=str, default='altinel')  # to track parallel device outputs

    args, unknown = parser.parse_known_args()
    main(args)
