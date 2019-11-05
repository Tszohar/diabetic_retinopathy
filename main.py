import datetime
import os
import shutil
import numpy as np

import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import parameters
from blindness_loss import MultipleBinaryLoss, RegressorLoss, BinaryLoss, MultiClassLoss
from data import BDDataset
from functions import predict_label
from network import BDNetwork, Outputs
from resnet import ResNet
from timer import Timer

loss_dict = {Outputs.REGRESSOR: RegressorLoss,
             Outputs.MULTI_BINARY: MultipleBinaryLoss,
             Outputs.BINARY: BinaryLoss,
             Outputs.MULTI_CLASS: MultiClassLoss,
             }

if __name__ == '__main__':

    ##############################   Network and train details   #######################################################
    batch_size = 256
    epoch_size = 100
    augmentation = False
    classifier_type = Outputs.MULTI_CLASS
    net = ResNet()

    #net = BDNetwork(classifier_type)

    binary2multi = False
    if binary2multi:
        # Loads trained binary model and convert positive images to multi class classifier
        model_path = "/media/guy/Files 3/Tsofit/blindness detection/results/20190919 (11:56:16.840499)_BINARY_32/model_epoch_78.pth"
        binary_net = BDNetwork(Outputs.BINARY)
        model = torch.load(model_path)
        binary_net.load_state_dict(model)
        binary_net.to(parameters.device)
    net.to(parameters.device)
    blindness_loss = loss_dict[classifier_type]().to(parameters.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    comment = 'Binary2Multi'
    experiment_name = "{}_{}_{}".format(classifier_type.name, batch_size, comment)

    ##############################   Data augmentation & loading  ######################################################
    transforms = []
    transforms.append(torchvision.transforms.RandomResizedCrop(size=256, scale=(1., 1.), ratio=(1.0, 1.0)))
    transforms.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
    # transforms.append(torchvision.transforms.RandomRotation((0, 90)))
    # transforms.append(torchvision.transforms.ColorJitter(brightness=0.2))
    if augmentation:
        transform = torchvision.transforms.Compose(transforms)
    else:
        transform = None

    train_dataset = BDDataset(csv_file=parameters.train_csv, data_dir=parameters.data_dir, transform=transform)
    train2015_dataset = BDDataset(csv_file=parameters.train2015_csv, data_dir=parameters.data2015_dir, transform=transform)
    dataset_combined = train_dataset + train2015_dataset
    train_dataloader = DataLoader(dataset_combined, batch_size=batch_size, shuffle=True, num_workers=8)

    validation_dataset = BDDataset(csv_file=parameters.validation_csv, data_dir=parameters.data_dir)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    ####################################################################################################################
    timers = {"load_data": Timer(), "train": Timer()}
    train_desc = datetime.datetime.now()

    log_dir = os.path.join(parameters.base_log_dir, "{}_{}".format(train_desc.strftime("%Y%m%d (%H:%M:%S.%f)"),
                                                                   experiment_name))
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    print("Outputs dir: {}".format(log_dir))
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_validation = SummaryWriter(log_dir=os.path.join(log_dir, 'validation'))
    run_counter = 0

    ##############################   Training  #########################################################################

    for epoch in range(epoch_size):
        running_loss = 0.0
        correct_labels_train = 0
        correct_labels_validation = 0
        timers["load_data"].start()
        net.train()
        params = np.zeros(len(train_dataset))
        for i_batch, sample_batched in enumerate(train_dataloader):
            run_counter += 1
            timers["load_data"].stop()
            optimizer.zero_grad()
            timers["train"].start()
            if binary2multi:
                outputs = binary_net(sample_batched['image'].to(parameters.device))
                binary_classes = outputs.argmax(axis=1)
                positive_outputs = sample_batched['diagnosis'][binary_classes == 1]
                outputs = net(sample_batched['image'][positive_outputs].to(parameters.device))
                loss = blindness_loss(outputs, sample_batched['diagnosis'][positive_outputs].to(parameters.device))
            else:
                outputs = net(sample_batched['image'].to(parameters.device))
                loss = blindness_loss(outputs, sample_batched['diagnosis'].to(parameters.device))
            loss.backward()
            optimizer.step()
            timers["train"].stop()

            # print statistics
            running_loss += loss.item()
            writer_train.add_scalar(tag='loss', scalar_value=loss.item(), global_step=run_counter)
            if i_batch % 5 == 0:
                print('[epoch: {}, batch: {}] loss: {:.3f}'.format(epoch + 1, i_batch + 1, loss.item()))
                print("Timings: " + ", ".join(["{}: {:.2f}".format(k, v.average_time) for k, v in timers.items()]))
                running_loss = 0.0
                writer_train.add_image("input", sample_batched['image'][0], i_batch)
            timers["load_data"].start()
            writer_train.file_writer.flush()

        # # Train Accuracy calculation
        # if classifier_type == Outputs.BINARY or classifier_type == Outputs.MULTI_BINARY or classifier_type == Outputs.MULTI_CLASS:
        #     for i_batch, sample_batched in enumerate(train_dataloader):
        #         if binary2multi:
        #             outputs = binary_net(sample_batched['image'].to(parameters.device))
        #             binary_classes = outputs.argmax(axis=1)
        #             positive_outputs = sample_batched['diagnosis'][binary_classes == 1]
        #             outputs = net(sample_batched['image'][positive_outputs].to(parameters.device))
        #             loss = blindness_loss(outputs, sample_batched['diagnosis'][positive_outputs].to(parameters.device))
        #         else:
        #             outputs = net(sample_batched['image'].to(parameters.device).to(parameters.device))
        #             loss = blindness_loss(outputs, sample_batched['diagnosis'].to(parameters.device))
        #         predicted_labels = predict_label(outputs, classifier_type)
        #         correct_labels_train += accuracy_score(predicted_labels.cpu(), blindness_loss.converted_label.cpu(), normalize=False)
        # train_accuracy = (correct_labels_train / len(train_dataset))
        #
        # print('Train accuracy: ' + str(train_accuracy))
        # writer_train.add_scalar(tag='accuracy', scalar_value=train_accuracy, global_step=epoch)

##############################   Validation  ###################################################################

        net.eval()

        correct_labels_validation = 0
        for i_batch, sample_batched in enumerate(validation_dataloader):
            if binary2multi:
                outputs = binary_net(sample_batched['image'].to(parameters.device))
                binary_classes = outputs.argmax(axis=1)
                positive_outputs = sample_batched['diagnosis'][binary_classes == 1]
                outputs = net(sample_batched['image'][positive_outputs].to(parameters.device))
                loss_val = blindness_loss(outputs, sample_batched['diagnosis'][positive_outputs].to(parameters.device))
            else:
                outputs = net(sample_batched['image'].to(parameters.device)).detach()
                loss_val = blindness_loss.forward(outputs, sample_batched['diagnosis'].to(parameters.device))
            writer_validation.add_scalar(tag='loss', scalar_value=loss_val, global_step=run_counter)
            predicted_labels = predict_label(outputs, classifier_type)
            if classifier_type == Outputs.BINARY or classifier_type == Outputs.MULTI_BINARY or classifier_type == Outputs.MULTI_CLASS:
                correct_labels_validation += accuracy_score(predicted_labels.cpu(), blindness_loss.converted_label.cpu(), normalize=False)

        # Validation Accuracy calculation
        if classifier_type == Outputs.BINARY or classifier_type == Outputs.MULTI_BINARY or classifier_type == Outputs.MULTI_CLASS:
            validation_accuracy = (correct_labels_validation / len(validation_dataset))
            print('Validation accuracy: ' + str(validation_accuracy))
            writer_validation.add_scalar(tag='accuracy', scalar_value=validation_accuracy, global_step=epoch)
            writer_validation.file_writer.flush()

        torch.save(net.state_dict(), os.path.join(log_dir, 'model_epoch_{}.pth'.format(epoch)))
