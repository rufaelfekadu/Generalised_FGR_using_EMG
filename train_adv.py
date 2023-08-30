import pandas as pd
from pathlib import Path
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils import preprocess_data, get_logger, AverageMeter, arg_parse
import os
import sys
sys.path.append('/home/rufael.marew/Documents/projects/tau/Fingers-Gesture-Recognition')

# import data stuff
import numpy as np
from dataset import emgdata, load_saved_data
# import Source.fgr.models as models

from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager

from models import make_model, vision, Net, Classifier, simpleMLP, FeatureExtractor

from sklearn.model_selection import KFold
#ignore warnings
import warnings
warnings.filterwarnings("ignore")

# set random seed
torch.manual_seed(0)
np.random.seed(0)


#### TODO:
# 1. train the discriminator every other epoch
# 2. have a weight for the discriminator loss


def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train(args, model, device, train_loader, test_loader, optimizer, criterion, logger):
    logger.info(f"{'Epoch' : <10}{'Train Loss' : ^20}{'Train disc_Loss' : ^20}{'Train Accuracy' : ^20}{'Test Loss' : ^20}{'Test Accuracy' : >10}{'Test disc_Accuracy' : >5}")
    #train
    for epoch in range(1, args.epochs + 1):

        train_output = train_epoch(args, model, device, train_loader, optimizer, criterion, epoch)
        log_string = ""
        if epoch % args.test_freq == 0:

            log_string = '{:<10}{:^20.4f}{:^20.4f}{:^20.2f}'.format(epoch, train_output["total_loss"].avg, train_output["discriminator_loss"].avg, train_output["train_acc"].avg*100)
            test_output = test(model, test_loader, device=device, criterion=criterion[0])

            log_string += '{:^20.4f}{:^20.2f}{:>5.2f}'.format(test_output["test_class_loss"].avg, test_output["test_acc"].avg*100, test_output["test_disc_acc"].avg*100)
            logger.info(log_string)
    
    return model

# train_epoch function
def train_epoch(args, model, device, train_loader, optimizer, criterion, epoch):
    # model.train()
    classifier_optimizer, discriminator_optimizer = optimizer
    classifier, discriminator = model
    classifier_loss, adverserial_loss = criterion

    classifier.train()
    discriminator.train()



    total_loss = AverageMeter()
    discriminator_loss = AverageMeter()
    accuracy = AverageMeter()
    class_loss = AverageMeter()

    for batch_idx, (data, (target, pos)) in enumerate(train_loader): # data: (batch_size, 3, 512, 512)
        correct = 0
        data, target, pos = data.to(device), target.to(device), pos.to(device)


        #forward pass
        feature = classifier.encoder(data)
        domain_output = discriminator(feature.detach())
        domain_loss = adverserial_loss(domain_output, pos)
        discriminator_loss.update(domain_loss.item(), data.size(0))

        # train classifier
        classifier_optimizer.zero_grad() # set gradient to zero
        class_output = classifier.classifier(feature)
        classification_loss = classifier_loss(class_output, target) # loss: (batch_size)
        class_loss.update(classification_loss.item(), data.size(0))

        loss = classification_loss - args.alpha*domain_loss
        # loss = classification_loss
        total_loss.update(loss.item(), data.size(0))

        loss.backward() # back propagation
        classifier_optimizer.step() # update parameters

        if epoch%args.disc_train_freq == 0:
            # train discriminator
            discriminator_optimizer.zero_grad() # set gradient to zero
            domain_output = discriminator(feature.detach())
            domain_loss = adverserial_loss(domain_output, pos)
            domain_loss.backward()
            discriminator_optimizer.step() # update parameters

        pred = class_output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy.update(correct, data.size(0))

    
    output = {
        "classifier_loss": class_loss,
        "total_loss": total_loss,
        "discriminator_loss": discriminator_loss,
        "train_acc": accuracy,
    }
    
    return output

def test(model, test_loader, device, criterion):

    classifier, discriminator = model

    classifier.eval()
    discriminator.eval()

    test_accuracy = AverageMeter()
    test_loss = AverageMeter()
    test_disc_accuracy = AverageMeter()

    with torch.no_grad(): # no gradient calculation
        for data, (target,pos) in test_loader:

            data, target, pos = data.to(device), target.to(device), pos.to(device)

            #forward pass
            feat = classifier.encoder(data)
            output = classifier.classifier(feat)
            disc_output = discriminator(feat)

            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            disc_pred = disc_output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

            test_accuracy.update(pred.eq(target.view_as(pred)).sum().item(), data.size(0))
            test_loss.update(criterion(output, target).item(), data.size(0))
            test_disc_accuracy.update(disc_pred.eq(pos.view_as(disc_pred)).sum().item(), data.size(0))

    # test_loss /= len(test_loader.dataset)
    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.00f}%)\n'.format(test_loss, correct, total, (correct/total)*100) )# print loss and accuracy
    output = {
        "test_class_loss": test_loss,
        "test_acc": test_accuracy,
        "test_disc_acc": test_disc_accuracy,
    }
    return output



# main function
def main(args):

    # data_dir = '../data/doi_10/emg'

    #setup logger
    logger = get_logger(os.path.join(args.logdir, 'adv_train.log'))
    logger.info(args)

    # specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #setup kfold
    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    dataset = emgdata(args.data_path, 
                      subjects=[1],
                      pos=[1,2,3],
                      sessions=[1],
                      transform=None,
                      target_transform=None,
                      train=True,
                      checkpoint=True)
    
    results = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        logger.info(f"\n{'Fold' : <10}{fold+1}")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # define data loaders
        train_loader = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    sampler=train_subsampler,
                                    num_workers=args.n_workers,
                                    pin_memory=True)

        test_loader = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    sampler=test_subsampler,
                                    num_workers=args.n_workers,
                                    pin_memory=True)
        
        # define models
        discriminator = Classifier(num_classes=3)
        discriminator.apply(reset_weights)
        classifier = Net(num_classes=10)
        classifier.apply(reset_weights)
        # define optimizers
        classifier_optimizer = torch.optim.Adam(
                classifier.parameters(), 
                lr=args.lr, betas=args.betas, 
                weight_decay=args.weight_decay)
    
        discriminator_optimizer = torch.optim.Adam(
                discriminator.parameters(), 
                lr=args.d_lr, betas=args.betas, 
                weight_decay=args.weight_decay)
    
        # define loss functions
        adverserial_loss = nn.CrossEntropyLoss()
        classifier_loss = nn.CrossEntropyLoss()

        # define model
        model = (classifier.to(device), discriminator.to(device))
        optimizer = (classifier_optimizer, discriminator_optimizer)
        criterion = (classifier_loss, adverserial_loss)
    
        train(args, model, device, train_loader, test_loader, optimizer, criterion, logger)

        #final test
        test_output = test(model, test_loader, device=device, criterion=criterion[0])
        logger.info(f"{'': <10}{'Test Loss' : ^20}{'Test Accuracy' : ^20}{'Test disc_Accuracy' : >5}")
        logger.info(f"{'': <10}{test_output['test_class_loss'].avg:^20.4f}{test_output['test_acc'].avg*100:^20.2f}{test_output['test_disc_acc'].avg*100:^5.2f}")

        results[fold] = test_output

    #print final result
    logger.info(f"\n\n{'': <10}{'Fold' : <20}{'Test Loss' : ^20}{'Test Accuracy' : ^20}{'Test disc_Accuracy' : >5}")
    sum = 0.0
    for fold, result in results.items():
        logger.info(f"{fold+1:<10}{result['test_class_loss'].avg:^20.4f}{result['test_acc'].avg*100:^20.2f}{result['test_disc_acc'].avg*100:^5.2f}")
        sum += result['test_acc'].avg*100
    logger.info(f"\n{'Average' : <10}{sum/len(results.items()):^20.4f}")
    
    # #save model
    # torch.save(classifier, os.path.join(args.logdir,'classifier.pt'))
    # torch.save(discriminator, os.path.join(args.logdir,'discriminator.pt'))




if __name__ == '__main__':

    args = arg_parse()

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data_path)
    args.__setattr__('data_path', data_path)
    
    main(args)
