######## Read and retrive data from edf, video and csv file ########

import os
import numpy as np
import pandas as pd
import torch
import mne
import cv2
from torch.utils.data import Dataset
from config import config
from datetime import datetime


class exp_times:
    manus_start_time = datetime.strptime('2023-10-02 14:59:20.799216', '%Y-%m-%d %H:%M:%S.%f')
    emg_start_time = datetime.strptime('2023-10-02 14:59:55.627000', '%Y-%m-%d %H:%M:%S.%f')
    video_Start_time = datetime.strptime('2023-10-02 14:59:55.628000', '%Y-%m-%d %H:%M:%S.%f')
    # emg_start_time = '2023-10-02 14:59:55.627'
    # video_Start_time = '2023-10-02 14:59:55:628'


def read_edf_file(file_path):
    # function to read edf file
    # input: file path
    # output: raw data
    raw = mne.io.read_raw_edf(file_path, preload=True)

    #crop data before exp start time
    raw.crop(tmin=exp_times.emg_start_time)
    #convert to numpy array
    raw = raw.get_data()

    return raw

def read_video_file(file_path):
    # function to read video file
    # input: file path
    # output: video data
    cap = cv2.VideoCapture(file_path)
    # crop video before exp start time
    cap.set(cv2.CAP_PROP_POS_MSEC, exp_times.video_Start_time)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def read_csv_file(file_path):
    # function to read csv file
    # input: file path
    # output: csv data
    data = pd.read_csv(file_path)
    # create a new coulmn of timestamp starting from manus_start_time and incrementing by the elapsed time between rows
    data['Timestamp'] = exp_times.manus_start_time
    data['Timestamp'] = data['Timestamp'].apply(lambda x: x + pd.DateOffset(milliseconds=1))
    
    # crop data before exp start time
    # data = data.loc[data['Timestamp'] >= exp_times.emg_start_time]
    return data



class EMGDataset(Dataset):
    def __init__(self, cfg,transform=None):
        self.data_info = cfg.DATA
        self.transform = transform


def main():
    video_path = 'dataset/video_2023-10-02 14-59-55-628.avi'
    edf_path = 'dataset/test 2023-10-02 14-59-55-627.edf'
    manus_data = 'dataset/Untitled_2023-10-02_15-24-12_YH_lab_R.csv'

    # video_data = read_video_file(video_path)
    # edf_data = read_edf_file(edf_path)

    manus_data = read_csv_file(manus_data)
    print(manus_data.columns)

if __name__ == '__main__':
    main()
    # print(exp_times.video_Start_time-exp_times.emg_start_time)