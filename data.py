import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
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
        image = Image.open(img_name).convert('RGB')
        image = image.resize([256, 256])
        sample = {'image': image, 'name': img_name,
                  'diagnosis': self.labels["diagnosis"][idx]}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        sample['image'] = transforms.ToTensor()(sample['image'])
        return sample

