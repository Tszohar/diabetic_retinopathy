
import os
import shutil
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from data import BDDataset, ToTensor
from model import get_model

from torch.utils.tensorboard import SummaryWriter


root_folder = '/home/guy/tsofit/blindness detection'
data_dir = os.path.join(root_folder, 'train_images')
validation_csv = os.path.join(root_folder, 'validation.csv')
output_dir = os.path.join(root_folder, 'mislabeled_images/')
output_correct_label_dir = os.path.join(root_folder, 'correct_label/')

def copy_images_to_subfolder(validation_df):
    os.makedirs(output_dir)
    os.makedirs(output_correct_label_dir)
    for img in range(len(validation_df)):
        if validation_df['predicted_label'][img] != validation_df['label'][img]:
            shutil.copy(validation_df['full_path'][img],
                        os.path.join(output_dir,
                        validation_df['id_code'][img] + '.png'))
        else:
            shutil.copy(validation_df['full_path'][img],
                        os.path.join(output_correct_label_dir,
                        validation_df['id_code'][img] + '.png'))

def mislabeled_csv(validation_df):
    mis_csv_path = os.path.join(output_dir, 'mislabeled.csv')
    mis_df = pd.DataFrame(columns=['id_code', 'predicted_label', 'label', 'score_0', 'score_1', 'score_2', 'score_3', 'score_4', 'full_path'])
    mis_list = []
    for img in range(len(validation_df)):
        if validation_df['predicted_label'][img] != validation_df['label'][img]:
            mis_list.append({ 'id_code': validation_df['id_code'][img],
                            'predicted_label': np.array(validation_df['predicted_label'][img]),
                            'label': np.array(validation_df['label'][img]),
                            'score_0': np.array(validation_df['score_0'][img]),
                            'score_1': np.array(validation_df['score_1'][img]),
                            'score_2': np.array(validation_df['score_2'][img]),
                            'score_3': np.array(validation_df['score_3'][img]),
                            'score_4': np.array(validation_df['score_4'][img]),
                            'full_path': validation_df['full_path'][img]
                            })
    mis_df = pd.DataFrame(mis_list)
    mis_df.to_csv(mis_csv_path)

def recall_by_class(validation_df):
    recall_accuracy = {}
    for current_class in range(2):
        class_num_images = len(validation_df[validation_df['label'] == current_class])
        correct_class_labels = len(validation_df[(validation_df['predicted_label'] == validation_df['label'])
                                                 & (validation_df['label'] == current_class)])
        recall_accuracy[current_class] = (correct_class_labels / class_num_images)
        print('Class {} recall: {}'.format(current_class, recall_accuracy[current_class]))

def accuracy_by_class(validation_df):
    class_accuracy = {}
    for current_class in range(2):
        correct_class_labels = len(validation_df[((validation_df['predicted_label'] == validation_df['label'])
                                                  & (validation_df['label'] == current_class))
                                                 | ((validation_df['predicted_label'] != current_class)
                                                    & (validation_df['label'] != current_class))])
        class_accuracy[current_class] = (correct_class_labels / len(validation_df))
        print('Class {} accuracy: {}'.format(current_class, class_accuracy[current_class]))

if __name__ == '__main__':

    model_path = '/home/guy/tsofit/blindness detection/results/24-07-2019 (21:54:32.272135)/model_epoch_99.pth'
    output_path = '/home/guy/tsofit/blindness detection/dataset.csv'
    batch_size = 1

    #####################################################################################
    # START, Post process - loading data and export to csv, only for the first time
    #####################################################################################

    # device = torch.device("cuda")
    # validation_dataset = BDDataset(csv_file=validation_csv, data_dir=data_dir, transform=ToTensor())
    # validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    # validation_df = pd.DataFrame(columns=['id_code', 'predicted_label', 'label', 'scores', 'full_path'])
    # model = torch.load(model_path)
    # net = get_model()
    # net.load_state_dict(model)
    # net.to(device)
    # net.eval()
    #
    # collect = []
    # with torch.no_grad():
    #     for i_batch, sampled_batch in enumerate(validation_dataloader):
    #         outputs = net(sampled_batch['image'].to(device))
    #         collect.append({'id_code': os.path.basename(sampled_batch['name'][0]).split('.')[0],
    #                         'predicted_label': np.argmax(np.array(outputs[0].cpu())),
    #                         'label': np.array(sampled_batch['label'][0]),
    #                         'score_0': np.array(outputs[0][0].cpu()),
    #                         'score_1': np.array(outputs[0][1].cpu()),
    #                         # 'score_2': np.array(outputs[0][2].cpu()),
    #                         # 'score_3': np.array(outputs[0][3].cpu()),
    #                         # 'score_4': np.array(outputs[0][4].cpu()),
    #                         'full_path': sampled_batch['name'][0],
    #                         })
    #
    # validation_df = pd.DataFrame(collect)
    # validation_df.to_csv(output_path)

    #####################################################################################
    # END, Post process - loading data and export to csv, only for the first time -
    #####################################################################################

    validation_df = pd.read_csv(output_path)
    #copy_images_to_subfolder(validation_df)
    #mislabeled_csv(validation_df)
    recall_by_class(validation_df)
    accuracy_by_class(validation_df)
    con_metrix = confusion_matrix(validation_df['label'], validation_df['predicted_label'])
    print(con_metrix)


