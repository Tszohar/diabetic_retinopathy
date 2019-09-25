#from enum import Enum
from aenum import Enum, NoAlias
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


class Outputs(Enum):
    _settings_ = NoAlias

    BINARY = 2
    MULTI_CLASS = 5
    REGRESSOR = 1
    MULTI_BINARY = 2



class BDNetwork(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.backbone = Backbone()
        self.head = Head(num_outputs=num_outputs.value)

    def forward(self, input):
        bb_out = self.backbone(input)
        head_out = self.head(bb_out)

        return head_out


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, input):
        return self.model(input)


class Head(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.model = nn.Sequential(
            Flatten(),
            nn.Linear(8 * 8 * 256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_outputs)
        )

    def forward(self, input):
        return self.model(input)






