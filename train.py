import torch
from pathlib import Path
import os
#import model stuff
from models import make_model

#import data stuff
import sys
sys.path.append('../Fingers-Gesture-Recognition')
import Source.fgr.models as models
from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager

from utils import preprocess_data, get_logger, arg_parse
from trainner import Trainner

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

def train_adv

def main(args):

    logger = get_logger(os.path.join(args.logdir, 'train_sgada.log'))
    logger.info(args)

            
    # dataset = datasets.ImageFolder(data_dir+'/001_1', transform=train_transform)
    
    # pipeline definition and data manager creation
    data_path = Path('../data/doi_10')
    pipeline = Data_Pipeline(base_data_files_path=data_path)  # configure the data pipeline you would like to use (check pipelines module for more info)
    subject = 1
    dm = Data_Manager([subject], pipeline)
    print(dm.data_info())


    source_dataset = dm.get_dataset(experiments=[f'{subject:03d}_*_1'])
    source_train_loader, source_test_loader, class_info = preprocess_data(source_dataset)
    if args.model == 'adv':
        target_datset = dm.get_dataset(experiments=[f'{subject:03d}_*_2'])
        target_train_loader, target_test_loader, _ = preprocess_data(target_datset)

    args.classInfo = class_info

    model = make_model(args)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    
    if args.model == 'adv':
        d_optimizer = torch.optim.Adam(
                model[2].parameters(),
                lr=args.d_lr, betas=args.betas, weight_decay=args.weight_decay)
        trainner = Trainner(args, model, 
                        (source_train_loader, source_test_loader, target_train_loader, target_test_loader ),
                        criterion=criterion,
                        optimizer=optimizer, d_optimizer=d_optimizer, logger=logger)
    else:
        d_optimizer = None
        trainner = Trainner(args, model, 
                        (source_train_loader, source_test_loader, None, None),
                        criterion=criterion,
                        optimizer=optimizer, d_optimizer=d_optimizer, logger=logger)
    
    
    trainner.train(args)

if __name__ == '__main__':
    args = arg_parse()
    main(args)