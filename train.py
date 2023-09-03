import torch
from pathlib import Path
import os
#import model stuff
from models import make_model, Net
import numpy as np
import pandas as pd
import argparse

#import data stuff
from config import config as cfg
# sys.path.append('/Users/rufaelmarew/Documents/tau/project/Fingers-Gesture-Recognition')
# import Source.fgr.models as models
# from Source.fgr.pipelines import Data_Pipeline
# from Source.fgr.data_manager import Data_Manager

from utils import preprocess_data, get_logger, arg_parse, AverageMeter
from dataset import emgdata, load_saved_data

from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, StratifiedKFold

# set random seed
torch.manual_seed(0)
np.random.seed(0)


def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train(cfg, model, device, train_loader, test_loader, optimizer, criterion, logger):

    logger.info(f"{'Epoch' : <10}{'Train Loss' : ^20}{'Train Accuracy' : ^20}{'Test Loss' : ^20}{'Test Accuracy' : >10}")
    for epoch in range(1,cfg.TRAIN.NUM_EPOCHS+1):

        train_output = train_epoch(model, device, train_loader, optimizer, criterion, epoch)
        log_string = ""
        if epoch % cfg.TRAIN.TEST_FREQ == 0:
            log_string = '{:<10}{:^20.4f}{:^20.2f} '.format(epoch, train_output["train_loss"].avg, train_output["train_acc"].avg*100)
            test_output = test(model, test_loader, device=device, criterion=criterion)
            log_string += '{:^20.4f}{:>10.2f}'.format(test_output["test_loss"].avg, test_output["test_acc"].avg*100)
            logger.info(log_string)

# train_epoch function
def train_epoch(model, device, train_loader, optimizer, criterion, epoch):

    model.train()

    total_loss = AverageMeter()
    accuracy = AverageMeter()

    for batch_idx, (data, (target, pos, label)) in enumerate(train_loader): # data: (batch_size, 3, 512, 512)

        data, target, pos = data.to(device), target.to(device), pos.to(device)

        optimizer.zero_grad()

        #forward pass
        output = model(data)

        loss = criterion(output, target) # loss: (batch_size)
        total_loss.update(loss.item(), data.size(0))

        loss.backward() # back propagation
        optimizer.step() # update parameters

        # pred = output.argmax(dim=1, keepdim=True)
        pred = output.max(dim=1, keepdim=True)[1]  
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

    class_acc = torch.zeros(len(test_loader.dataset.classes))
    class_len = torch.zeros(len(test_loader.dataset.classes))

    with torch.no_grad(): # no gradient calculation
        for data, (target,pos, label) in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            # pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            pred = output.max(dim=1, keepdim=True)[1]
            test_accuracy.update(pred.eq(target.view_as(pred)).sum().item(), data.size(0))
            test_loss.update(criterion(output, target).item(), data.size(0))
            
            for class_id in test_loader.dataset.classes:
                idx = torch.nonzero(label==class_id.to(label.device), as_tuple=False)
                class_acc[class_id] += pred[idx].eq(target[idx].view_as(pred[idx])).sum().item()
                class_len[class_id] += len(idx)
    
    class_acc /= class_len
        
    output = {
        "test_loss": test_loss,
        "test_acc": test_accuracy,
        "perclass": class_acc,
        "class_len": class_len,
    }

    return output


def main(cfg):
            
    #setup logger
    logger = get_logger(os.path.join(cfg.OUTPUT.LOG_DIR, 'train.log'))

    logger.info(cfg)
    

    #setup device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(cfg.TRAIN.DEVICE)

    #setup kfold
    k_fold = StratifiedKFold(n_splits=cfg.TRAIN.N_SPLITS, shuffle=True, random_state=0)
    
    dataset = emgdata(cfg.DATA.DATA_PATH,
                        subjects=cfg.DATA.SUBJECT,
                        sessions=cfg.DATA.SESSION,
                        pos=cfg.DATA.POSITION,
                        checkpoint=cfg.DATA.CHECKPOINT)
    dataset.print_info()
    results = {}
    for fold, (train_ids, test_ids) in enumerate(k_fold.split(dataset, dataset.label)):

        logger.info(f"\n{'Fold' : <10}{fold+1}")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = DataLoader(dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                    sampler=train_subsampler,
                                    num_workers=cfg.TRAIN.NUM_WORKERS)
        test_loader = DataLoader(dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                    sampler=test_subsampler,
                                    num_workers=cfg.TRAIN.NUM_WORKERS)
        model = Net(num_classes=10).to(device)
        model.apply(reset_weights)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
       

        train(cfg, model, device, train_loader, test_loader, optimizer, criterion, logger)

        #final test
        test_output = test(model, test_loader, device=device, criterion=criterion)
        logger.info(f"{'': <10}{'Test Loss' : ^20}{'Test Accuracy' : ^20}")
        logger.info(f"{'': <10}{test_output['test_loss'].avg:^20.4f}{test_output['test_acc'].avg*100:^20.2f}")
        
        logger.info(f"{'Fold':<10}{'class': <20}{'Test Accuracy' : ^20}{'class length' : ^20}")
        for i, (acc,length) in enumerate(zip(test_output['perclass'], test_output['class_len'])):
            logger.info(f"{fold:<10}{test_loader.dataset.classNames[i]:<20}{acc:^20.2f}{length:^20.0f}")
        
        results[fold] = test_output
    
    
    #print final result
    logger.info(f"\n\n{'': <10}{'Fold' : <20}{'Test Loss' : ^20}{'Test Accuracy' : ^20}")
    sum = 0.0
    for fold, result in results.items():
        logger.info(f"{fold+1:<10}{result['test_loss'].avg:^20.4f}{result['test_acc'].avg*100:^20.2f}")
        sum += result['test_acc'].avg*100

    logger.info(f"\n{'Average' : <10}{sum/len(results.items()):^20.4f}")
    results['average'] = sum/len(results.items())
    #save the results
    torch.save(results, os.path.join(args.logdir, f'results.pt'))

    
        
if __name__ == '__main__':

    # args = arg_parse()

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='configuration file')
    parser.add_argument('opts', default=None,help='modify config options using the command-line', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_list(args.opts)

    if isinstance(cfg.DATA.SUBJECT, int):
        cfg.DATA.SUBJECT = [cfg.DATA.SUBJECT]
    if isinstance(cfg.DATA.SESSION, int):
        cfg.DATA.SESSION = [cfg.DATA.SESSION]
    if isinstance(cfg.DATA.POSITION, int):
        cfg.DATA.POSITION = [cfg.DATA.POSITION]

    log_dir = os.path.join(cfg.OUTPUT.LOG_DIR,f'subject_{cfg.DATA.SUBJECT}' )
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT.LOG_DIR = log_dir
    data_path = Path(cfg.DATA.PATH)
    cfg.DATA.DATA_PATH = data_path

    main(cfg)