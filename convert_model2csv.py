import torch

import os
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score
from data import BDDataset
from model import get_model

def model2csv(data_csv, data_dir, model_path, output_path):

    #####################################################################################
    # loading data according to specific model and export to csv
    #####################################################################################

    batch_size = 1

    device = torch.device("cuda")

    dataset = BDDataset(csv_file=data_csv, data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    dataframe = pd.DataFrame(columns=['id_code', 'predicted_label', 'label', 'scores', 'full_path'])
    model = torch.load(model_path)
    net = get_model()
    net.load_state_dict(model)
    net.to(device)
    net.eval()

    collect = []
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(dataloader):
            if i_batch % 10 == 0:
                print("batch {}/{}".format(i_batch, len(dataloader)))
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

    dataframe = pd.DataFrame(collect)
    dataframe.to_csv(output_path)



