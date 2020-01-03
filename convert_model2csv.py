import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import parameters
from data import BDDataset
from functions import predict_label
from network import BDNetwork
from resnet import ResNet


def model2csv(data_csv, data_dir, model_path, output_path, classifier_type):
    """

    :param data_csv:
    :param data_dir:
    :param model_path:
    :param output_path:
    :param classifier_type:
    :return:
    """
    #####################################################################################
    # loading data according to specific model and export to csv
    #####################################################################################

    batch_size = 32

    device = torch.device("cuda")

    dataset = BDDataset(csv_file=data_csv, data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    model = torch.load(model_path)
    net = ResNet(num_outputs=5)
    net.to(parameters.device)
    net.load_state_dict(model, strict=False)
    net.eval()
    blindness_loss = parameters.loss_dict[classifier_type]().to(parameters.device)

   # analyze = parameters.analyzer_dict[classifier_type]()

    collect = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch % 10 == 0:
                print("batch {}/{}".format(i_batch, len(dataloader)))
            outputs = net(sample_batched['image'].to(device))
            blindness_loss(outputs, sample_batched['diagnosis'].to(parameters.device))
            predicted_labels = predict_label(outputs, classifier_type)
            for i in range(len(sample_batched['name'])):
                collect.append({'id_code': os.path.basename(sample_batched['name'][i]),
                                'predicted_label': predicted_labels[i].item(),
                                'converted_label': blindness_loss.converted_label[i].item(),
                                "diagnosis": sample_batched['diagnosis'][i].item(),
                                'scores': np.array(outputs[i].cpu()),
                })
        dataframe = pd.DataFrame(collect)
        if os.path.isdir(os.path.dirname(output_path)):
            pass
        else:
            os.makedirs(os.path.dirname(output_path))
        dataframe.to_csv(output_path)



