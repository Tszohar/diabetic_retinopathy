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
        return (ground_truth > 0).long()


class MultipleBinaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.num_classes = 4
        self.converted_label = 0

    def forward(self, predictions, targets):
        loss = torch.zeros([len(predictions), self.num_classes], dtype=torch.float32)
        num_of_scores = 8
        idx = 0
        for score in range(0, num_of_scores, 2):
            converted_label = self.convert_label(targets, threshold=idx)
            loss[:, idx] = self.criterion(predictions[:, score:score+2], converted_label)
            idx += 1
        #self.converted_label = torch.sum(loss, axis=1)[:, None]
        self.converted_label = targets
        return torch.mean(loss)
        #return torch.sum(loss)

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

