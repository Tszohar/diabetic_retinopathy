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
    def __init__(self, input_size, output_size, stride=1, fix_dimension=False):
        super().__init__()
        self.fix_dimension = False
        if fix_dimension:
            self.fix_dimension = True
            self.in_size = input_size
            self.out_size = output_size
            self.conv_1x1 = nn.Conv2d(self.in_size, self.out_size, 1, stride=2).to('cuda')
        self.model = nn.Sequential(
            nn.Conv2d(input_size, output_size, 3, stride=stride, padding=1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_size, output_size, 3, stride=1, padding=1),
            nn.BatchNorm2d(output_size)
        )

    def forward(self, input):
        if self.fix_dimension:
            converted_input = self.conv_1x1(input)
            return self.model(input) + converted_input
        return self.model(input) + input


class Block(nn.Module):
    def __init__(self, input_size, output_size, stride=1, fix_dimension=False):
        super().__init__()
        self.conv = nn.Sequential(Conv2(input_size=input_size, output_size=output_size, stride=stride,
                                        fix_dimension=fix_dimension),
                                  nn.ReLU(inplace=True),

                                  Conv2(input_size=output_size, output_size=output_size),
                                  nn.ReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class ResNet(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.conv1 = Conv1()
        self.conv2 = Block(input_size=64, output_size=64)
        self.conv3 = Block(input_size=64, output_size=128, stride=2, fix_dimension=True)
        self.conv4 = Block(input_size=128, output_size=256, stride=2, fix_dimension=True)
        self.conv5 = Block(input_size=256, output_size=512, stride=2, fix_dimension=True)
        self.features = nn.Sequential(nn.AvgPool2d(7), Flatten())
        self.multi_class = nn.Linear(512, num_outputs)

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        features = self.features(conv5)
        final_layer = self.multi_class(features)
        return final_layer

    def get_features(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        features = self.features(conv5)
        return features
