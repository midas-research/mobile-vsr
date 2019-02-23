# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import torch

from cvtransforms import *


def load_file(filename):
    cap = np.load(filename)
    arrays = np.stack([cv2.cvtColor(cap[_], cv2.COLOR_RGB2GRAY) for _ in range(29)], axis=0)
    arrays = arrays / 255.
    return arrays


class LRW():
    def __init__(self, folds, path):

        self.folds = folds  # ['train', 'val', 'test']
        self.path = path
        self.istrain = (folds == 'train')

        with open('../label_sorted.txt') as myfile:
            self.data_dir = myfile.read().splitlines()

        self.data_files = glob.glob(self.path+'*/'+self.folds+'/*.npy')
        self.list = {}

        for i, x in enumerate(self.data_files):
            target = x.split('/')[-3]
            for j, elem in enumerate(self.data_dir):
                if elem == target:
                    self.list[i] = [x]
                    self.list[i].append(j)
        print('Load {} part'.format(self.folds))

    def __getitem__(self, idx):

        inputs = load_file(self.list[idx][0])        
        labels = self.list[idx][1]
        return inputs, labels

    def __len__(self):
        return len(self.data_files)


def prepare_train_batch(batch, device=None, non_blocking=False):
    inputs, targets = batch

    batch_img = RandomCrop(inputs.numpy(), (88, 88))
    batch_img = ColorNormalize(batch_img)
    batch_img = HorizontalFlip(batch_img)

    batch_img = np.reshape(batch_img, (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
    inputs = torch.from_numpy(batch_img)
    inputs = inputs.float().permute(0, 4, 1, 2, 3)

    inputs, targets = inputs.to(device), targets.to(device)
    return inputs, targets

def prepare_val_batch(batch, device=None, non_blocking=False):
    inputs, targets = batch
    
    batch_img = CenterCrop(inputs.numpy(), (88, 88))
    batch_img = ColorNormalize(batch_img)
    
    batch_img = np.reshape(batch_img, (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
    inputs = torch.from_numpy(batch_img)
    inputs = inputs.float().permute(0, 4, 1, 2, 3)

    inputs, targets = inputs.to(device), targets.to(device)
    return inputs, targets