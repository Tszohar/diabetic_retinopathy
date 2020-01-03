

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader


import parameters
from blindness_loss import MultiClassLoss
from data import BDDataset
from functions import predict_label
from network import Outputs
from resnet import ResNet


batch_size = 64
model_path = "/media/guy/Files 3/Tsofit/blindness detection/results/20191112 (14:59:57.119645)_MULTI_CLASS_64_OnlySick_weights_model_21/model_epoch_63.pth"
model_path_binary = '/media/guy/Files 3/Tsofit/blindness detection/results/20191111 (16:28:16.076528)_MULTI_CLASS_64_HealthyAndSick/model_epoch_21.pth'
model_path = '/media/guy/Files 3/Tsofit/blindness detection/results/20191111 (16:28:16.076528)_MULTI_CLASS_64_HealthyAndSick/model_epoch_21.pth'
binary_model_path = "/media/guy/Files 3/Tsofit/blindness detection/results/20191114 (11:50:48.136229)_BINARY_64_ResNetBinary_all/model_epoch_9.pth"
model_path = '/media/guy/Files 3/Tsofit/blindness detection/results/20191114 (14:01:59.115208)_BINARY_64_ResNetSick/model_epoch_2.pth'
model_path = '/media/guy/Files 3/Tsofit/blindness detection/results/20191117 (12:30:55.408262)_MULTI_CLASS_64_multi_smallDS/model_epoch_46.pth'
validation_dataset = BDDataset(csv_file=parameters.validation_csv, data_dir=parameters.data_dir)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

classifier_type = Outputs.MULTI_CLASS
net = ResNet(num_outputs=5)
blindness_loss_multi = parameters.loss_dict[classifier_type]().to(parameters.device)
model = torch.load(model_path)
net.load_state_dict(model, strict=False)
net.to(parameters.device)
net.eval()
torch.no_grad()
torch.set_grad_enabled(False)

# net_all = ResNet(num_outputs=5)
# model_binary = torch.load(model_path_binary)
# net_all.load_state_dict(model_binary)
# net_all.to(parameters.device)
# net_all.eval()
# torch.no_grad()
# torch.set_grad_enabled(False)

correct_labels_validation = 0
for i_batch, sample_batched in enumerate(validation_dataloader):
    outputs = net(sample_batched['image'].to(parameters.device))
    predicted_labels_all = predict_label(outputs, classifier_type)
    # predicted_labels_sick = predicted_labels_all - 1
    # sample_batched['diagnosis'] = sample_batched['diagnosis'] - 1
    # sample_batched['diagnosis'][sample_batched['diagnosis'] > 0] = 1
    multi_loss = blindness_loss_multi(outputs, sample_batched['diagnosis'].long().to(parameters.device))

    correct_labels_validation += accuracy_score(predicted_labels_all.cpu(), sample_batched['diagnosis'].cpu(),
                                                    normalize=False)

# Validation Accuracy calculation
validation_accuracy = (correct_labels_validation / len(validation_dataset))
print('Validation accuracy: ' + str(validation_accuracy))
