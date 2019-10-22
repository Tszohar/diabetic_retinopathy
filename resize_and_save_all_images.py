import os

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import shutil
import parameters
from data import BDDataset

if __name__ == '__main__':

    batch_size = 1
    train_dataset = BDDataset(csv_file=parameters.train_csv, data_dir=parameters.data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    if not os.path.isdir(os.path.dirname(parameters.resized_data_dir)):
        os.mkdir(parameters.resized_data_dir)

    for i_batch, sample_batched in enumerate(train_dataloader):
        img = sample_batched['image'][i_batch]
        trans = transforms.ToPILImage()
        pil_img = trans(sample_batched['image'][i_batch])
     #   img_name = os.path.basename(sample_batched['name'][i_batch])
      #  img.save()
