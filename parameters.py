import os
import torch

from blindness_loss import RegressorLoss, MultipleBinaryLoss, BinaryLoss, MultiClassLoss
from network import Outputs
from post_process import RegressorAnalyzer, MultiBinaryAnalyzer, BinaryAnalyzer, MultiClassAnalyzer

root_folder = '/media/guy/Files 3/Tsofit/blindness detection'
data_dir = os.path.join(root_folder, 'train_images_resized')
data2015_dir = os.path.join(root_folder, 'train_images_2015_resized')
data_dir_all = os.path.join(root_folder, 'all_train_images_resized')

train_csv = os.path.join(root_folder, 'train.csv')
train2015_csv = os.path.join(root_folder, 'train2015.csv')
train_sick = os.path.join(root_folder, 'train_sick.csv')

train_over_csv = os.path.join(root_folder, 'train_over.csv')
validation_csv = os.path.join(root_folder, 'validation.csv')
validation_csv_sick = os.path.join(root_folder, 'validation_sick.csv')
base_log_dir = os.path.join(root_folder, 'results')
analysis_dir = os.path.join(root_folder, 'analysis_dir/')

device = torch.device("cuda")

loss_dict = {Outputs.REGRESSOR: RegressorLoss,
             Outputs.MULTI_BINARY: MultipleBinaryLoss,
             Outputs.BINARY: BinaryLoss,
             Outputs.MULTI_CLASS: MultiClassLoss,
             }

num_classes_dict = {'BINARY': 2,
                    'MULTI_CLASS': 5,
                    'REGRESSOR': 5,
                    'MULTI_BINARY': 4,
                    }

analyzer_dict = {Outputs.REGRESSOR: RegressorAnalyzer,
                 Outputs.MULTI_BINARY: MultiBinaryAnalyzer,
                 Outputs.BINARY: BinaryAnalyzer,
                 Outputs.MULTI_CLASS: MultiClassAnalyzer,
                 }

