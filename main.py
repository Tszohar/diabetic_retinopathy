import os
import shutil

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import datetime


from timer import Timer
from model import get_model
from data import BDDataset, ToTensor

root_folder = '/home/guy/tsofit/blindness detection'
data_dir = os.path.join(root_folder, 'train_images')
train_csv = os.path.join(root_folder, 'train.csv')
validation_csv = os.path.join(root_folder, 'validation.csv')


if __name__ == '__main__':
    batch_size = 64
    device = torch.device("cuda")
    net = get_model()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(net.parameters())

    timers = {"load_data": Timer(), "train": Timer()}
    base_log_dir = '/home/guy/tsofit/blindness detection/results'
    train_desc = datetime.datetime.now()
    log_dir = os.path.join(base_log_dir, train_desc.strftime("%d-%m-%Y (%H:%M:%S.%f)"))
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    writer_train = SummaryWriter(log_dir=log_dir, comment='test comment')
    writer_validation = SummaryWriter(log_dir=os.path.join(log_dir, 'validation'), comment='test comment')
    run_counter = 0

    train_dataset = BDDataset(csv_file=train_csv, data_dir=data_dir, transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    validation_dataset = BDDataset(csv_file=validation_csv, data_dir=data_dir, transform=ToTensor())
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    for epoch in range(100):
        running_loss = 0.0
        correct_labels_train = 0
        correct_labels_validation = 0
        timers["load_data"].start()
        net.train()
        for i_batch, sample_batched in enumerate(train_dataloader):
            run_counter += 1
            timers["load_data"].stop()
            optimizer.zero_grad()
            timers["train"].start()
            outputs = net.forward(sample_batched['image'].to(device))

            loss = criterion(outputs, sample_batched['label'].to(device))
            loss.backward()
            optimizer.step()
            timers["train"].stop()

            # print statistics
            running_loss += loss.item()
            writer_train.add_scalar(tag='Loss', scalar_value=loss.item(), global_step=run_counter)
            if i_batch % 5 == 0:  # print every 2000 mini-batches
                print('[epoch: {}, batch: {}] loss: {:.3f}'.format(epoch + 1, i_batch + 1, loss.item()))
                print("Timings: " + ", ".join(["{}: {:.2f}".format(k, v.average_time) for k, v in timers.items()]))
                running_loss = 0.0
            timers["load_data"].start()
            writer_train.file_writer.flush()

        net.eval()

        for i_batch, sample_batched in enumerate(train_dataloader):
            outputs = net(sample_batched['image'].to(device)).detach().cpu()
            correct_labels_train += accuracy_score(outputs.argmax(1), sample_batched['label'], normalize=False)

        train_accuracy = (correct_labels_train / len(train_dataset))
        print('Train accuracy: ' + str(train_accuracy))
        writer_train.add_scalar(tag='train_acc', scalar_value=train_accuracy, global_step=run_counter)
        for i_batch, sample_batched in enumerate(validation_dataloader):
            outputs = net(sample_batched['image'].to(device)).detach().cpu()
            loss_val = criterion(outputs, sample_batched['label'])
            correct_labels_validation += accuracy_score(outputs.argmax(1), sample_batched['label'], normalize=False)

        validation_accuracy = (correct_labels_validation / len(validation_dataset))
        print('Validation accuracy: ' + str(validation_accuracy))
        writer_validation.add_scalar(tag='val_acc', scalar_value=validation_accuracy, global_step=run_counter)
        writer_validation.add_scalar(tag='val_loss', scalar_value=loss_val, global_step=run_counter)
        writer_validation.file_writer.flush()
        torch.save(net.state_dict(), os.path.join(log_dir, 'model_epoch_{}.pth'.format(epoch)))


