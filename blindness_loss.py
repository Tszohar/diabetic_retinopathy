import numpy as np
import torch
import torch.nn as nn
import parameters


class RegressorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.criterion.to(parameters.device)
        self.converted_label = 0

    def forward(self, predictions, targets):
        converted_label = self.convert_label(targets)
        self.converted_label = converted_label[:, None]
        return self.criterion(predictions, self.converted_label)

    def convert_label(self, ground_truth):
        return ground_truth.float()


class BinaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(parameters.device)
        self.converted_label = 0

    def forward(self, predictions, targets):
        converted_label = self.convert_label(targets)
        self.converted_label = converted_label
        return self.criterion(predictions, self.converted_label.to(parameters.device).squeeze())

    def convert_label(self, ground_truth):
        return (ground_truth > 1).long()


class MultipleBinaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = 4
        self.converted_label = 0

    def forward(self, predictions, targets):
        loss = torch.zeros([self.num_classes, 1], dtype=torch.float32)
        for idx in range(0, self.num_classes):
            converted_label = self.convert_label(targets, threshold=idx)
            loss[idx] = self.criterion(predictions, converted_label)
        self.converted_label = torch.sum(loss)
        return self.converted_label

    def convert_label(self, ground_truth, threshold):
        return (ground_truth > threshold).long()


class MultiClassLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(parameters.device)
        self.converted_label = 0

    def forward(self, predictions, targets):
        self.converted_label = targets
        return self.criterion(predictions, targets)

