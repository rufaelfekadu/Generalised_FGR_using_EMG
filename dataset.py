import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

import numpy as np
import sys
sys.path.append('../Fingers-Gesture-Recognition')
from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager


#torch dataset classs
class emgdata(Dataset):
    def __init__(self, data_dir, subject, pos=1, transform=None, target_transform=None):

        pipeline = Data_Pipeline(base_data_files_path=data_dir)  # configure the data pipeline you would like to use (check pipelines module for more info)
        self.subject = subject
        dm = Data_Manager([subject], pipeline)
        print(dm.data_info())

        dataset = dm.get_dataset(experiments=[f'{subject:03d}_1_*'])
        data = dataset[0]
        labels = dataset[1]
        labelencoder = LabelEncoder()

        self.target = labelencoder.fit_transform(np.char.strip(labels, '_0123456789'))
        # self.target  = np.char.strip(labels, '_0123456789')
        # self.pos = np.array([i.split('_')[2] for i in dataset[1]])
        self.pos = labelencoder.fit_transform(np.array([i.split('_')[2] for i in labels]))

        data = data.reshape(data.shape[0],1,4,4)
        self.data = data

        if transform is not None:
            self.data = transform(self.data)
        else:
            self.data = torch.from_numpy(self.data).float()
        
        if target_transform is not None:
            self.target = target_transform(self.target)
        else:
            self.target = torch.from_numpy(self.target).long()
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        
        return self.data[idx], (self.target[idx], self.pos[idx])
    
if __name__ == '__main__':
    from pathlib import Path
    subject = 1
    data_dir = Path('../data/doi_10')
    dataset = emgdata(data_dir, subject)

    print(dataset.__len__())