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

from utils import preprocess_data, get_logger, arg_parse, AverageMeter
from trainner import Trainner


# train_epoch function
def train_epoch(model, device, train_loader, optimizer, epoch):
    # model.train()
    feature_exteractor_optimizer, classifier_optimizer, discriminator_optimizer = optimizer
    feature_extractor, classifier, discriminator = model

    feature_extractor.train()
    classifier.train()
    discriminator.train()



    total_loss = AverageMeter()
    discriminator_loss = AverageMeter()
    accuracy = AverageMeter()

    for batch_idx, (data, (target, pos)) in enumerate(train_loader): # data: (batch_size, 3, 512, 512)

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

        loss = classification_loss + domain_loss
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

        accuracy.update(class_output.argmax(dim=1, keepdim=True).eq(target.view_as(class_output)).sum().item(), data.size(0))



    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( # print loss
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
    
    output = {
        "total_loss": total_loss.avg,
        "discriminator_loss": discriminator_loss.avg,
        "train_acc": accuracy.avg,
    }
    
    return output

def test(model, test_loader, device):
    feature_extractor, classifier, discriminator = model
    model.eval()
    # test_loss = 0
    # correct = 0
    # total = 0

    test_accuracy = AverageMeter()
    test_loss = AverageMeter()
    with torch.no_grad(): # no gradient calculation
        for data, (target,pos) in test_loader:
            data, target = data.to(device), target.to(device)
            feat = feature_extractor(data)
            output = classifier(feat)
            # test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item() # sum up correct predictions
            # total += target.size(0)
            test_accuracy.update(pred.eq(target.view_as(pred)).sum().item(), data.size(0))
            test_loss.update(torch.nn.functional.cross_entropy(output, target, reduction='sum').item(), data.size(0))
    # test_loss /= len(test_loader.dataset)
    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.00f}%)\n'.format(test_loss, correct, total, (correct/total)*100) )# print loss and accuracy
    output = {
        "test_loss": test_loss.avg,
        "test_acc": test_accuracy.avg,
    }
    return output


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