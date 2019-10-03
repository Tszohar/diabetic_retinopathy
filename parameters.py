import os
import torch

from blindness_loss import RegressorLoss, MultipleBinaryLoss, BinaryLoss, MultiClassLoss
from network import Outputs
from post_process import RegressorAnalyzer, MultiBinaryAnalyzer, BinaryAnalyzer, MultiClassAnalyzer

root_folder = '/media/guy/Files 3/Tsofit/blindness detection'
data_dir = os.path.join(root_folder, 'train_images')
train_csv = os.path.join(root_folder, 'train.csv')
validation_csv = os.path.join(root_folder, 'validation.csv')
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

