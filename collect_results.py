import torch
from torch import nn
import os

def collect(path):
    result = torch.load(path)
    return result

def analize(result):
    print(result.keys())
    
if __name__ == '__main__':

    path = 'output/scenario_1/cnn/subject_[1]/results.pt'

    result = collect(path)
    analize(result)
