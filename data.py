import os
from collections import Sized

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


class BDDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.labels.iloc[idx]['id_code'])
        if os.path.isfile(img_name + '.png'):
            img_name = img_name + '.png'
        else:
            img_name = img_name + '.jpeg'

        image = Image.open(img_name).convert('RGB')
        image = image.resize([224, 224])
        sample = {'image': image, 'name': img_name,
                  'diagnosis': self.labels["diagnosis"][idx]}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        sample['image'] = transforms.ToTensor()(sample['image'])
        return sample


class Balanced_Sampler(Sampler):
    def __init__(self, data_source: Sized):
        super().__init__(data_source)
        self.num_samples = len(data_source)
        self.num_of_classes = 5
        self.samples_per_class = len(data_source) / self.num_of_classes
        self.data = data_source

    def __len__(self):
        return len(self.num_samples)

    def __iter__(self):
        print('bla')
        batch = []

        batch.append(idx)

        return iter(range(self.num_samples))