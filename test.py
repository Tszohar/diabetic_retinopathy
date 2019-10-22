import torch
from PIL import Image
from torch.utils.data import DataLoader
import os

from torchvision.transforms import transforms

import parameters
from convert_model2csv import model2csv
from data import BDDataset
from functions import predict_label
from network import BDNetwork, Outputs
import pandas as pd
from torch.utils.data import Dataset


class TestDataset(Dataset):
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
        sample = {'image': image, 'name': img_name}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        sample['image'] = transforms.ToTensor()(sample['image'])
        return sample


if __name__ == '__main__':

    batch_size = 32
    classifier_type = Outputs.MULTI_CLASS

    model_path = "/media/guy/Files 3/Tsofit/blindness detection/results/20191002 (10:50:00.518925)_MULTI_CLASS_32/model_epoch_99.pth"
    test_csv = '/media/guy/Files 3/Tsofit/blindness detection/test.csv'
    test_images = '/media/guy/Files 3/Tsofit/blindness detection/test_images'
    submission_csv = '/media/guy/Files 3/Tsofit/blindness detection/sample_submission.csv'

    model = torch.load(model_path)
    net = BDNetwork(classifier_type)
    net.load_state_dict(model)
    net.eval()
    net.to(parameters.device)

    test_dataset = TestDataset(csv_file=test_csv, data_dir=test_images)
    train_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)


    collect = []
    for i_batch, sample_batched in enumerate(train_dataloader):
        outputs = net(sample_batched['image'].to(parameters.device)).detach()
        predicted_labels = predict_label(outputs, classifier_type)
        for i in range(len(sample_batched['name'])):
            id_code = os.path.basename(sample_batched['name'][i]).split('.')[0]
            collect.append({'id_code': id_code, 'diagnosis': predicted_labels[i].item()})
    dataframe = pd.DataFrame(collect)
    dataframe.to_csv(submission_csv)
    print ('Done!')