
import os
import shutil

import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import torch.nn as nn

from convert_model2csv import model2csv


def copy_images_by_predicate(dataframe, output_dir, predicate):
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for img in range(len(dataframe)):
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


def recall_by_class(dataframe):
    recall_accuracy = {}
    for current_class in range(2):
        class_num_images = len(dataframe[dataframe['label'] == current_class])
        correct_class_labels = len(dataframe[(dataframe['predicted_label'] == dataframe['label'])
                                             & (dataframe['label'] == current_class)])
        recall_accuracy[current_class] = (correct_class_labels / class_num_images)
        print('Class {} recall: {}'.format(current_class, recall_accuracy[current_class]))


def accuracy_by_class(dataframe):
    class_accuracy = {}
    for current_class in range(2):
        correct_class_labels = len(dataframe[((dataframe['predicted_label'] == dataframe['label'])
                                              & (dataframe['label'] == current_class))
                                             | ((dataframe['predicted_label'] != current_class)
                                                & (dataframe['label'] != current_class))])
        class_accuracy[current_class] = (correct_class_labels / len(dataframe))
        print('Class {} accuracy: {}'.format(current_class, class_accuracy[current_class]))


def loss_by_sample(dataframe):
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    loss = criterion(dataframe['predicted_label'], dataframe['label'])
    return loss

if __name__ == '__main__':

    batch_size = 1

   # root_folder = '/home/guy/tsofit/blindness detection'
    root_folder = '/media/guy/Files 3/Tsofit/blindness detection'
    model_path = os.path.join( root_folder, "/results/04-08-2019 (13:26:05.166282)/model_epoch_47.pth")
                                            #'/29-07-2019 (08:31:40.256328)/model_epoch_99.pth'
    output_csv = os.path.join(root_folder, '/dataset29-7.csv')
    data_dir = os.path.join(root_folder, 'train_images')
    data_csv = os.path.join(root_folder, 'validation.csv')
    analysis_dir = os.path.join(root_folder, 'analysis_dir/')

    #Converting the dataset into csv file
    model2csv(data_csv, data_dir, model_path, output_csv)

    dataframe = pd.read_csv(output_csv)
    copy_images_to_subfolder(dataframe, analysis_dir=analysis_dir)

    # Save mislabeled CSV
    mis_df = dataframe[dataframe['predicted_label'] != dataframe['label']]
    mis_df.to_csv(os.path.join(analysis_dir, "mislabeled.csv"))

    recall_by_class(dataframe)
    accuracy_by_class(dataframe)
    conf_matrix = confusion_matrix(dataframe['label'], dataframe['predicted_label'])
    print('confusion matrix:')
    print(conf_matrix)
    kappa_score = cohen_kappa_score(dataframe['label'], dataframe['predicted_label'])
    print('kappa score: {}'.format(kappa_score))

    loss = loss_by_sample(dataframe)



