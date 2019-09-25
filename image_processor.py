

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import parameters

#
# def loss_by_sample(dataframe):
#     criterion = nn.CrossEntropyLoss()
#     criterion.to(device)
#     loss = criterion(dataframe['predicted_label'], dataframe['label'])
#     return loss




class CopyImages:
    def __init__(self):

        # self.image_src = dataset['name']
        # self.image_name = os.path.split(self.image_src)
        self.output_dir = parameters.analysis_dir

    def by_predicate(self, dataset, predicate):
        dst_folder = os.path.join(self.output_dir, 'by_predicate')
        if os.path.isdir(dst_folder):
            shutil.rmtree(dst_folder)
        os.makedirs(dst_folder)
        for img in range(len(dataset)):
            image_path = dataset['name'][img]
            image_name = os.path.split(image_path)[1]
            predicted_class = predicate[img][predicate[img].argmax().item()].item().__round__()
            correct_class = dataset['diagnosis'][img].item()
            dst = os.path.join(dst_folder, image_name + '_D{}_P{}.png'.format(correct_class, predicted_class))
            shutil.copy(src=image_path, dst=dst)


class ConvertDataset2Csv:
    def __init__(self):
        self.output_path = os.path.join(parameters.analysis_dir, 'results_csv.csv')
        self.collect = []

    def convert(self, dataset, predicate):
        for img in range(len(dataset)):
            img_data = ConvertDataset2Csv.extract_data(self, dataset, predicate, img)
            self.collect.append(img_data)
            scores = ConvertDataset2Csv.split_scores(self, predicate)
            self.collect.append(scores)

    def extract_data(self, dataset, predicate, idx):
        image_path = dataset['name'][idx]
        image_name = os.path.split(image_path)[1]
        predicted_class = predicate[idx][predicate[idx].argmax().item()].item().__round__()
        correct_class = dataset['diagnosis'][idx].item()
        image_data = {'image_path': image_path, 'image_name': image_name, 'predicted_class': predicted_class,
                      'correct_class': correct_class}
        return image_data

    def split_scores(self, predicate):
        for score in range(len(predicate[1])):
            scores = {score: predicate[score]}
            return scores

    def save(self, data):
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(self.output_path)