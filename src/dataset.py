# author: Daulet Baimukashev
# Created on: 6.05.20

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ControlDataset(Dataset):
    """Control dataset """
    def __init__(self, x, y, transform = None):

        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        x_shape = self.x.shape
        return x_shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read feature
        features = self.x[idx, :, :]

        # read label
        labels = self.y[idx, :]
        labels = np.array([labels])
        labels = labels.astype('float').reshape(-1, 2)

        sample = {'features':features, 'labels':labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, labels = sample['features'], sample['labels']

        return {'features': torch.from_numpy(features),
                'labels': torch.from_numpy(labels),
               }
