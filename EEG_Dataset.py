import torch
from torch.utils.data import Dataset
import itertools
import numpy as np
import os
import config as cfg
from scipy.signal import hilbert, chirp
from scipy.signal import butter, lfilter

class EEG_Dataset(Dataset):
    def __init__(self, files):
        self.len = cfg.eeglen
        self.pairs = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            self.pairs += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]
        # get 3840
        self.seg = np.zeros((len(self.pairs),1))
        for index in range(len(self.pairs)):
            env = np.load(self.pairs[index][1]).astype(np.float32)
            if len(env)<self.len:
                self.seg[index] = 0
            else:
                self.seg[index] = int((len(env))/self.len)

        self.alleeg = []
        self.allmel = []
        self.allenv = []

        for index in range(len(self.pairs)):

            # print("{0}/{1}".format(index, len(self.pairs)))

            alleeg = np.load(self.pairs[index][0]).astype(np.float32)
            allenv = np.load(self.pairs[index][1]).astype(np.float32)
            allmel = np.load(self.pairs[index][2]).astype(np.float32)


            self.alleeg.append(alleeg)
            self.allenv.append(allenv)
            self.allmel.append(allmel)


    def __len__(self):
        return int(np.sum(self.seg))

    def find_index(self, target):
        acc = 0
        for i, num in enumerate(self.seg):
            acc += num
            if acc > target:
                return i, int(target - acc + num)

    def __getitem__(self, index):
        nparry_index, time_index = self.find_index(index)

        alleeg = self.alleeg[nparry_index]
        allmel = self.allmel[nparry_index]
        allenv = self.allenv[nparry_index]
        eeg = alleeg[self.len*time_index:self.len*time_index+self.len,:]
        mel = allmel[self.len * time_index:self.len * time_index+self.len, :]
        env = allenv[self.len * time_index:self.len * time_index+self.len, :]

        eeg = torch.tensor(eeg, dtype=torch.float32)
        eeg_mean = torch.mean(eeg, dim=0, keepdim=True)
        eeg_std = torch.std(eeg, dim=0, keepdim=True)
        eeg = (eeg - eeg_mean) / eeg_std

        mel = torch.tensor(mel, dtype=torch.float32)
        # mel_mean = torch.mean(mel, dim=0, keepdim=True)
        # mel_std = torch.std(mel, dim=0, keepdim=True)
        # mel = (mel - mel_mean) / mel_std

        env = torch.tensor(env, dtype=torch.float32)
        # env_mean = torch.mean(env, dim=0, keepdim=True)
        # env_std = torch.std(env, dim=0, keepdim=True)
        # env = (env - env_mean) / env_std


        return eeg,env,mel



