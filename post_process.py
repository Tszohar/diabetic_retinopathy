
import os
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from data import BDDataset
from model import get_model


def copy_images_by_predicate(dataframe, output_dir, predicate):
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for img in range(len(validation_df)):
        if predicate(dataframe.iloc[img]):
            src = os.path.join(dataframe['data_dir'][img], dataframe['id_code'][img] + '.png')
            dst = os.path.join(output_dir, dataframe['id_code'][img] + '_l{}.png'.format(dataframe['diagnosis'][img]))
            shutil.copy(src=src, dst=dst)


def copy_images_to_subfolder(dataframe, analysis_dir):
    tp_dir = os.path.join(analysis_dir, "tn")
    tp_pred = lambda x: x["predicted_label"] == 1 and x["label"] == 1
    copy_images_by_predicate(dataframe, tp_dir, tp_pred)

    tn_dir = os.path.join(analysis_dir, "tp")
    tn_pred = lambda x: x["predicted_label"] == 0 and x["label"] == 0
    copy_images_by_predicate(dataframe, tn_dir, tn_pred)

    fp_dir = os.path.join(analysis_dir, "fn")
    fp_pred = lambda x: x["predicted_label"] == 1 and x["label"] == 0
    copy_images_by_predicate(dataframe, fp_dir, fp_pred)

    fn_dir = os.path.join(analysis_dir, "fp")
    fn_pred = lambda x: x["predicted_label"] == 0 and x["label"] == 1
    copy_images_by_predicate(dataframe, fn_dir, fn_pred)


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

    model_path = '/home/guy/tsofit/blindness detection/results/29-07-2019 (08:31:40.256328)/model_epoch_99.pth'
    model_path = "/media/guy/Files 3/Tsofit/blindness detection/results/04-08-2019 (13:26:05.166282)/model_epoch_47.pth"
    output_path = '/home/guy/tsofit/blindness detection/dataset29-7.csv'
    batch_size = 1

    root_folder = '/home/guy/tsofit/blindness detection'
    root_folder = '/media/guy/Files 3/Tsofit/blindness detection'
    data_dir = os.path.join(root_folder, 'train_images')
    validation_csv = os.path.join(root_folder, 'validation.csv')
    analysis_dir = os.path.join(root_folder, 'analysis_dir/')

    go_through_data = False

    # CR: Split to 2 scripts, one for generating the CSV files and going over the data (evaluation), and another for post processing

    #####################################################################################
    # START, Post process - loading data and export to csv, only for the first time
    #####################################################################################
    if go_through_data:
        device = torch.device("cuda")

        # CR: This doesn't have to be validation dataset, it can also be the training dataset, change naming
        validation_dataset = BDDataset(csv_file=validation_csv, data_dir=data_dir)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        validation_df = pd.DataFrame(columns=['id_code', 'predicted_label', 'label', 'scores', 'full_path'])
        model = torch.load(model_path)
        net = get_model()
        net.load_state_dict(model)
        net.to(device)
        net.eval()

        collect = []
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(validation_dataloader):
                if i_batch % 10 == 0:
                    print("batch {}/{}".format(i_batch, len(validation_dataloader)))
                outputs = net(sampled_batch['image'].to(device))
                collect.append({'id_code': os.path.basename(sampled_batch['name'][0]).split('.')[0],
                                'predicted_label': np.argmax(np.array(outputs[0].cpu())),
                                'label': np.array(sampled_batch['label'][0]),
                                'score_0': np.array(outputs[0][0].cpu()),
                                'score_1': np.array(outputs[0][1].cpu()),
                                # 'score_2': np.array(outputs[0][2].cpu()),
                                # 'score_3': np.array(outputs[0][3].cpu()),
                                # 'score_4': np.array(outputs[0][4].cpu()),
                                "diagnosis": np.array(sampled_batch['diagnosis'][0]),
                                "data_dir": data_dir
                                })

        validation_df = pd.DataFrame(collect)
        validation_df.to_csv(output_path)

    #####################################################################################
    # END, Post process - loading data and export to csv, only for the first time -
    #####################################################################################

    validation_df = pd.read_csv(output_path)
    copy_images_to_subfolder(validation_df, analysis_dir=analysis_dir)

    # Save mislabeled CSV
    mis_df = validation_df[validation_df['predicted_label'] != validation_df['label']]
    mis_df.to_csv(os.path.join(analysis_dir, "mislabeled.csv"))

    recall_by_class(validation_df)
    accuracy_by_class(validation_df)
    conf_matrix = confusion_matrix(validation_df['label'], validation_df['predicted_label'])
    print(conf_matrix)


