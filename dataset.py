import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from sklearn.preprocessing import LabelEncoder

import numpy as np
import sys
sys.path.append('/home/rufael.marew/Documents/projects/tau/Fingers-Gesture-Recognition')
from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager
from sklearn.model_selection import train_test_split
import os


scenario = {
    0:{'subject':[1,2,3,4,5,6,7,8,9,10], 'sessions':[1,2], 'pos':[1,2,3]},
    1:{'subject':[1,2,3,4,5,6,7,8,9,10], 'sessions':[1,2], 'pos':[1,2,3]},
    2:{'subject':[1,2,3,4,5,6,7,8,9,10], 'sessions':[1,2], 'pos':[1,2,3]},
}
#torch dataset classs
class emgdata(Dataset):
    def __init__(self, data_dir, subjects=[1], pos=[1,2,3], sessions=[1,2], transform=None, target_transform=None, train=True, checkpoint=True, scenario=0):

        self.pipeline = Data_Pipeline(base_data_files_path=data_dir)  # configure the data pipeline you would like to use (check pipelines module for more info)
        self.subjects = subjects
        self.sessions = sessions
        self.pos = pos
        self.transform = transform
        self.target_transform = target_transform

        name = f"subject_{subjects[0]}-{subjects[-1]}_sessions_{sessions[0]}-{sessions[-1]}_positions_{pos[0]}-{pos[-1]}.pt"
        if checkpoint:
            try:
                dataset = self.load_saved_data(os.path.join(data_dir,name))
                self.data = dataset.data
                self.target = dataset.target
                self.pos = dataset.pos
                self.label = dataset.label
                self.classes = dataset.classes
                self.classNames = dataset.classNames
                del dataset
            except FileNotFoundError:
                print('dataset not found, creating new dataset')
                self.create_dataset()
                torch.save(self, os.path.join(data_dir,name))
            
        else:
            self.create_dataset()

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], (self.target[idx], self.pos[idx], self.label[idx])
    
    def create_dataset(self):
        dm = Data_Manager(self.subjects, self.pipeline)
        print(dm.data_info())

        experiments = []
        for subject in self.subjects:
            for session in self.sessions:
                for p in self.pos:
                    experiments.append(f'{subject:03d}_{session}_{p}')
        
        dataset = dm.get_dataset(experiments=experiments)

        data = dataset[0]
        labels = dataset[1]
        labelencoder = LabelEncoder()
        self.target = labelencoder.fit_transform(np.char.strip(labels, '_0123456789'))
        self.pos = torch.from_numpy(labelencoder.fit_transform(np.array([i.split('_')[2] for i in labels]))).long()

        data = data.reshape(data.shape[0],1,4,4)
        self.data = data

        if self.transform is not None:
            self.data = self.transform(self.data) #t
        else:
            self.data = torch.from_numpy(self.data).float()
        
        if self.target_transform is not None:
            self.target = self.target_transform(self.target)
        else:
            self.target = torch.from_numpy(self.target).long()
        
        self.label = torch.from_numpy(labelencoder.fit_transform(dataset[1]))
        self.classes = torch.unique(self.label)
        self.classNames = labelencoder.classes_
        
    
    def senario_1(self):
        pass
    def senario_2(self):
        pass

    def print_info(self):
        # number of images in each class
        print('----------Number of images in each class------')
        print('')
        for i in np.unique(self.target):
            idx = self.target == i
            print(f'Class {i} has {idx.sum()} samples')
        print('----------------------------------------------')

        # number of classes
        print('number of classes: ', len(np.unique(self.label)))
    
    def load_saved_data(self, path):
        dataset = torch.load(path)
        return dataset

def load_saved_data(path):
    dataset = torch.load(path)
    return dataset


if __name__ == '__main__':
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/rufael.marew/Documents/projects/tau/data/doi_10')
    parser.add_argument('--subject', type=int, default=[1])
    parser.add_argument('--sessions', type=int, default=[1,2])
    parser.add_argument('--positions', type=int, default=[1,2,3])
    parser.add_argument('checkpoint', type=bool, default=True)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    dataset = emgdata(args.data_path, subjects=args.subject, sessions=args.sessions, pos=args.positions)

    print(dataset.__len__())