import torch
from pathlib import Path
import os
#import model stuff
from models import make_model, Net

#import data stuff
import sys
sys.path.append('../Fingers-Gesture-Recognition')
import Source.fgr.models as models
from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager

from utils import preprocess_data, get_logger, arg_parse, AverageMeter
from dataset import emgdata, load_saved_data
from trainner import Trainner

from torch.utils.data import DataLoader



# train_epoch function
def train_epoch(model, device, train_loader, optimizer, criterion, epoch):

    model.train()

    total_loss = AverageMeter()
    accuracy = AverageMeter()

    for batch_idx, (data, (target, pos)) in enumerate(train_loader): # data: (batch_size, 3, 512, 512)

        data, target, pos = data.to(device), target.to(device), pos.to(device)

        optimizer.zero_grad()

        #forward pass
        output = model(data)

        loss = criterion(output, target) # loss: (batch_size)
        total_loss.update(loss.item(), data.size(0))

        loss.backward() # back propagation
        optimizer.step() # update parameters

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()

        accuracy.update(correct, data.size(0))
    
    output = {
        "train_loss": total_loss.avg,
        "train_acc": accuracy.avg,
    }
    
    return output

def test(model, test_loader, device, criterion):

    model.eval()

    test_accuracy = AverageMeter()
    test_loss = AverageMeter()

    with torch.no_grad(): # no gradient calculation
        for data, (target,pos) in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

            test_accuracy.update(pred.eq(target.view_as(pred)).sum().item(), data.size(0))
            test_loss.update(criterion(output, target).item(), data.size(0))

    output = {
        "test_loss": test_loss,
        "test_acc": test_accuracy,
    }

    return output


def main(args):
            
    #setup logger
    logger = get_logger(os.path.join(args.logdir, 'adv_train.log'))
    logger.info(args)

    #setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load data
    train_data =  load_saved_data('./outputs/train_data.pt')
    test_data = load_saved_data('./outputs/test_data.pt')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)


    model = Net(num_classes=10).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
   
    
    for epoch in range(1,args.epochs+1):

        train_output = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        
        if epoch % args.test_freq == 0:
            logger.info('Train Epoch: {} \tTotal Loss: {:.4f}\tAccuracy: {:.2f}%'.format(
                epoch, train_output["total_loss"], train_output["train_acc"]*100))
                    
            test_output = test(model, test_loader, device=device, criterion=criterion)
            logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                test_output["test_loss"].avg, test_output["test_acc"].sum, len(test_loader.dataset), (test_output["test_acc"].avg)*100))
        
if __name__ == '__main__':

    args = arg_parse()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    
    main(args)