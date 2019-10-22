import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


class Conv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def forward(self, input):
        return self.model(input)


class Conv2(nn.Module):
    def __init__(self, input_size, output_size, stride=1):
        super().__init__()
        self.fix_dimension = False
        if stride > 1:
            self.fix_dimension = True
            self.in_size = input_size
            self.out_size = output_size
        self.model = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, stride=stride, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_size, output_size, 3, stride=1, padding=1),
            nn.BatchNorm2d(output_size)
        )

    def forward(self, input):
        if self.fix_dimension:
            conv_1x1 = nn.Conv2d(self.in_size, self.out_size, 1, stride=2).to('cuda')
            converted_input = conv_1x1(input)
            self.fix_dimension = False
            return self.model(input) + converted_input
        return self.model(input) + input


class Block(nn.Module):
    def __init__(self, input_size, output_size, stride=1):
        super().__init__()
        self.conv = nn.Sequential(Conv2(input_size=input_size, output_size=output_size, stride=stride),
                                  nn.ReLU(inplace=True),
                                  Conv2(input_size=output_size, output_size=output_size), nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1()
        self.conv2 = Block(input_size=64, output_size=64)
        self.conv3 = Block(input_size=64, output_size=128, stride=2)
        self.conv4 = Block(input_size=128, output_size=256, stride=2)
        self.conv5 = Block(input_size=256, output_size=512, stride=2)
        self.end = nn.Sequential(nn.AvgPool2d(3), Flatten(), nn.Linear(2048, 1000), nn.Linear(1000, 1))

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        final_layer = self.end(conv5)
        return final_layer

