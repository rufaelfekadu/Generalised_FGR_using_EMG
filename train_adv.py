import pandas as pd
from pathlib import Path
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils import preprocess_data, get_logger, AverageMeter
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

# train_epoch function
def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    # model.train()
    feature_exteractor_optimizer, classifier_optimizer, discriminator_optimizer = optimizer
    feature_extractor, classifier, discriminator = model
    classifier_loss, adverserial_loss = criterion

    feature_extractor.train()
    classifier.train()
    discriminator.train()



    total_loss = AverageMeter()
    discriminator_loss = AverageMeter()
    accuracy = AverageMeter()
    class_loss = AverageMeter()

    for batch_idx, (data, (target, pos)) in enumerate(train_loader): # data: (batch_size, 3, 512, 512)
        correct = 0
        data, target, pos = data.to(device), target.to(device), pos.to(device)

        feature_exteractor_optimizer.zero_grad() # set gradient to zero
        classifier_optimizer.zero_grad() # set gradient to zero
        discriminator_optimizer.zero_grad() # set gradient to zero

        #forward pass
        feature = feature_extractor(data)

        # train discriminator
        domain_output = discriminator(feature.detach())
        domain_loss = adverserial_loss(domain_output, pos)

        # train classifier
        class_output = classifier(feature)
        classification_loss = classifier_loss(class_output, target) # loss: (batch_size)
        class_loss.update(classification_loss.item(), data.size(0))

        loss = classification_loss +0.25*domain_loss
        total_loss.update(loss.item(), data.size(0))

        loss.backward() # back propagation
        feature_exteractor_optimizer.step() # update parameters
        classifier_optimizer.step() # update parameters

        #train_discriminator
        discriminator_optimizer.zero_grad() # set gradient to zero
        domain_output = discriminator(feature.detach())
        domain_loss = adverserial_loss(domain_output, pos)
        discriminator_loss.update(domain_loss.item(), data.size(0))
        domain_loss.backward()
        discriminator_optimizer.step()

        pred = class_output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy.update(correct, data.size(0))

    # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( # print loss
    #     epoch, batch_idx * len(data), len(train_loader.dataset),
    #     100. * batch_idx / len(train_loader), loss.item()))
    
    output = {
        "classifier_loss": class_loss.avg,
        "total_loss": total_loss.avg,
        "discriminator_loss": discriminator_loss.avg,
        "train_acc": accuracy.avg,
    }
    
    return output

def test(model, test_loader, device, criterion):

    feature_extractor, classifier, discriminator = model

    feature_extractor.eval()
    classifier.eval()
    discriminator.eval()

    test_accuracy = AverageMeter()
    test_loss = AverageMeter()
    test_disc_accuracy = AverageMeter()

    with torch.no_grad(): # no gradient calculation
        for data, (target,pos) in test_loader:

            data, target, pos = data.to(device), target.to(device), pos.to(device)

            #forward pass
            feat = feature_extractor(data)
            output = classifier(feat)
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
    
    #load data
    train_data =  load_saved_data('./outputs/train_data.pt')
    test_data = load_saved_data('./outputs/test_data.pt')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)

    # specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define models
    feature_extractor = FeatureExtractor()
    discriminator = Classifier(num_classes=3)
    classifier = Classifier()

    # define optimizers
    feature_extractor_optimizer = torch.optim.Adam(
            feature_extractor.parameters(), 
            lr=args.lr, betas=args.betas, 
            weight_decay=args.weight_decay)
    classifier_optimizer = torch.optim.Adam(
            classifier.parameters(), 
            lr=args.lr, betas=args.betas, 
            weight_decay=args.weight_decay)
    discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(), 
            lr=args.lr, betas=args.betas, 
            weight_decay=args.weight_decay)
    
    # define loss functions
    adverserial_loss = nn.CrossEntropyLoss()
    classifier_loss = nn.CrossEntropyLoss()

    # define model
    model = (feature_extractor.to(device), classifier.to(device), discriminator.to(device))
    optimizer = (feature_extractor_optimizer, classifier_optimizer, discriminator_optimizer)
    criterion = (classifier_loss, adverserial_loss)

    #train
    for epoch in range(1, args.epochs + 1):

        train_output = train_epoch(model, device, train_loader, optimizer, criterion, epoch)

        if epoch % 10 == 0:
            logger.info('Train Epoch: {} \tTotal Loss: {:.4f}\tDiscriminator Loss: {:.4f}\tAccuracy: {:.2f}%'.format(
                epoch, train_output["total_loss"], train_output["discriminator_loss"], train_output["train_acc"]*100))
                    
            test_output = test(model, test_loader, device=device, criterion=criterion[0])
            logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) Dicriminator Accuracy: {:.2f}%'.format(
                test_output["test_class_loss"].avg, test_output["test_acc"].sum, len(test_loader.dataset), (test_output["test_acc"].avg)*100, test_output["test_disc_acc"].avg*100))
        
    #save model
    torch.save(feature_extractor, os.path.join(args.logdir,'feature_extractor.pt'))
    torch.save(classifier, os.path.join(args.logdir,'classifier.pt'))
    torch.save(discriminator, os.path.join(args.logdir,'discriminator.pt'))




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
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--betas', type=float, nargs='+', default=(.5, .999))
    parser.add_argument('--lam', type=float, default=0.25)
    parser.add_argument('--thr', type=float, default=0.79)
    parser.add_argument('--thr_domain', type=float, default=0.87)
    parser.add_argument('--num_val', type=int, default=3)  # number of val. within each epoch
    # misc
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='outputs/adv_model')
    # office dataset categories
    parser.add_argument('--src_cat', type=str, default='mscoco')
    parser.add_argument('--tgt_cat', type=str, default='flir')
    parser.add_argument('--tgt_conf_cat', type=str, default='flir_confident')
    parser.add_argument('--message', type=str, default='altinel')  # to track parallel device outputs

    args, unknown = parser.parse_known_args()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    main(args)
