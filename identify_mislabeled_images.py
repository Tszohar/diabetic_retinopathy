import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

model_path = "/media/guy/Files 3/Tsofit/blindness detection/results/20191028 (11:23:39.097861)_MULTI_CLASS_100_Resnet_and_augmentation/model_epoch_49.pth"
output_path = os.path.join(os.path.dirname(model_path), 'images_analysis.csv')

dataframe = pd.read_csv(output_path)
mislabeled = dataframe[dataframe['evaluation'] == False]
mislabeled = mislabeled.reset_index()

for i in range(len(mislabeled)):
    print('bla')
    img = plt.imread(mislabeled['id_code'][i])
    plt.imshow(img)
    plt.title('label: {}, predicted: {}'.format(mislabeled['diagnosis'][i], mislabeled['predicted_label'][i]))

