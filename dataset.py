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

#torch dataset classs
class emgdata(Dataset):
    def __init__(self, data_dir, subjects=[1], pos=[1,2,3], sessions=[1,2], transform=None, target_transform=None, train=True, checkpoint=True):

        self.pipeline = Data_Pipeline(base_data_files_path=data_dir)  # configure the data pipeline you would like to use (check pipelines module for more info)
        self.subjects = subjects
        self.sessions = sessions
        self.pos = pos

        if checkpoint:
            try:
                dataset = load_saved_data(os.path.join(data_dir,'/dataset.pt'))
            except FileNotFoundError:
                print('dataset not found, creating new dataset')
                dataset = self.create_dataset()
                torch.save(dataset, data_dir+'/dataset.pt')
            self.data = dataset.data
            self.target = dataset.target
            self.pos = dataset.pos
        else:
            self.create_dataset()

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], (self.target[idx], self.pos[idx])
    
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
    
    def load_saved_data(self, path):
        dataset = torch.load(path)
        return dataset

def load_saved_data(path):
    dataset = torch.load(path)
    return dataset

def make_dataset(data_path, save_path, subject, sessions, positions, test_size=0.3):

    dataset = emgdata(data_path, subjects=subject, sessions=sessions, pos=positions)
    # train_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=test_size, shuffle=True, stratify=dataset.target)
    # # train_data, test_data = random_split(dataset, [int(len(dataset)*(1-test_size)), len(dataset)-int(len(dataset)*(1-test_size))])

    # train_data = torch.utils.data.Subset(dataset, train_idx)
    # test_data = torch.utils.data.Subset(dataset, test_idx)

    # torch.save(train_data, save_path+'/train_data.pt')
    # torch.save(test_data, save_path+'/test_data.pt')

    torch.save(dataset, save_path+'/dataset.pt')
    del dataset

    return train_data, test_data

if __name__ == '__main__':
    from pathlib import Path
    data_path = Path('/home/rufael.marew/Documents/projects/tau/data/doi_10')
    # save_path = '/home/rufael.marew/Documents/projects/tau/data/emgdataset/'

    subject = [1]
    sessions = [1]
    positions = [1,2,3]

    # train_data, test_data = make_dataset(data_path, save_path, subject, sessions, positions)

    dataset = emgdata(data_path, subjects=subject, sessions=sessions, pos=positions)

    print(dataset.__len__())