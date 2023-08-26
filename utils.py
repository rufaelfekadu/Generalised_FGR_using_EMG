import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import argparse
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value
       https://github.com/pytorch/examples/blob/master/imagenet/main.py#L296
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val 
        self.count += n
        self.avg = self.sum / self.count

def preprocess_data(dataset):

    data = dataset[0]
    labels = dataset[1]

    data = torch.Tensor(data)
    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(np.char.strip(labels, '_0123456789'))
    labels = torch.Tensor(labels).to(torch.int64)
    class_info = {'classes': torch.unique(labels),
                    'classNames': labelencoder.classes_}
    dataset = TensorDataset(data, labels)

    # split data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # save test data
    # torch.save(test_dataset, 'outputs/test_dataset_cnn.pt')
    # data loader
    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=96, shuffle=True)

    return train_loader, test_loader, class_info

def get_logger(log_file):
    from logging import getLogger, FileHandler, StreamHandler, Formatter, DEBUG, INFO  # noqa
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    sh = StreamHandler()
    sh.setLevel(INFO)
    for handler in [fh, sh]:
        formatter = Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
    logger = getLogger('adda')
    logger.setLevel(INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='transformer')
    # NN
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--trained', type=str, default='')
    parser.add_argument('--slope', type=float, default=0.2)

    #transformer
    parser.add_argument('--image_size', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--attention_dropout', type=float, default=0.1)

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
    parser.add_argument('--save_freq', type=int, default=5)
    # misc
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--logdir', type=str, default='outputs/')
    parser.add_argument('--test_freq', type=int, default=10)

    # discriminator
    parser.add_argument('--disc_train_freq', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.25)

    args, unknown = parser.parse_known_args()

    return args