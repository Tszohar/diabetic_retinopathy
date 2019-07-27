import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class BDDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.labels.iloc[idx]['id_code']) + '.png'
        image = cv2.imread(img_name,)
       # image = Image.open(img_name)
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32)
        binary_label = []
        if ((self.labels.iloc[idx]['diagnosis'] == 0) | (self.labels.iloc[idx]['diagnosis'] == 1)):
            binary_label = 0
        else:
            binary_label = 1
        sample = {'image': image, 'label': np.array(binary_label), 'name': img_name}



        if self.transform:
            sample = self.transform(sample)
            # transform_list = [transforms.ColorJitter(),
            #                     #transforms.RandomAffine(),
            #                     #transforms.RandomCrop(),
            #                     transforms.RandomHorizontalFlip(p=0.5),
            #                     transforms.RandomResizedCrop(256),
            #                     transforms.RandomRotation((0, 90))
            #                   ]
            # transforms.RandomApply(transform_list, p=0.5)
            # transforms.ToTensor()

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        sample['image'] = torch.from_numpy(image)
        sample['label'] = torch.from_numpy(sample['label'])

        return sample


