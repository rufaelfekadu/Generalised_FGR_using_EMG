import torch
from pathlib import Path
import os
#import model stuff
from models import make_model, Net
import numpy as np

#import data stuff
import sys
sys.path.append('../Fingers-Gesture-Recognition')
import Source.fgr.models as models
from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager

from utils import preprocess_data, get_logger, arg_parse, AverageMeter
from dataset import emgdata, load_saved_data

from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold

# set random seed
torch.manual_seed(0)
np.random.seed(0)

def train(args, model, device, train_loader, test_loader, optimizer, criterion, logger):

    logger.info(f"{'Epoch' : <10}{'Train Loss' : ^20}{'Train Accuracy' : ^20}{'Test Loss' : ^20}{'Test Accuracy' : >10}")
    for epoch in range(1,args.epochs+1):

        train_output = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        log_string = ""
        if epoch % args.test_freq == 0:
            log_string = '{:<10}{:^20.4f}{:^20.2f} '.format(epoch, train_output["train_loss"].avg, train_output["train_acc"].avg*100)
            test_output = test(model, test_loader, device=device, criterion=criterion)
            log_string += '{:^20.4f}{:>10.2f}'.format(test_output["test_loss"].avg, test_output["test_acc"].avg*100)
            logger.info(log_string)

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
        "train_loss": total_loss,
        "train_acc": accuracy,
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
    logger = get_logger(os.path.join(args.logdir, 'cnn.log'))
    logger.info(args)

    #setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #setup kfold
    k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
    dataset = emgdata(args.data_path, subjects=[1], sessions=[1,2], pos=[1,2,3])

    #load data
    # train_data =  load_saved_data(args.data_path+'/train_data.pt')
    # test_data = load_saved_data(args.data_path+'/test_data.pt')

    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    # test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

    results = {}

    for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset)):

        logger.info(f"{'Fold' : <10}{fold+1}")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_subsampler)

        model = Net(num_classes=10).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    

        train(args, model, device, train_loader, test_loader, optimizer, criterion, logger)

        #final test
        test_output = test(model, test_loader, device=device, criterion=criterion)
        results[fold] = test_output
    
    #print final result
    logger.info(f"\n\n{'Fold' : <10}{'Test Loss' : ^20}{'Test Accuracy' : ^20}")
    sum = 0.0
    for fold, result in results.items():
        logger.info(f"{fold+1:<10}{result['test_loss'].avg:^20.4f}{result['test_acc'].avg*100:^20.2f}")
        sum += result['test_acc'].avg*100
    logger.info(f"\n{'Average' : <10}{sum/len(results.items()):^20.4f}")


    
        
if __name__ == '__main__':

    args = arg_parse()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    main(args)