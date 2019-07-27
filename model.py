import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


# class get_model(nn.Module):
#     model = nn.Sequential(
#         nn.Conv2d(3, 16, 3, 1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(16, 32, 3, 1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(2, 2),
#         nn.Conv2d(32, 64, 3, 1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(64, 64, 3, 1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(2, 2),
#         nn.Conv2d(64, 128, 3, 1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(128, 128, 3, 1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(2, 2),
#         nn.Conv2d(128, 256, 3, 1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(256, 256, 3, 1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(2, 2),
#         nn.Conv2d(256, 256, 3, 1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(256, 256, 3, 1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(2, 2),
#         Flatten(),
#         nn.Linear(8 * 8 * 256, 512),
#         nn.ReLU(inplace=True),
#         nn.Linear(512, 2)
#     )


def get_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(256, 256, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
        Flatten(),
        nn.Linear(8 * 8 * 256, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 2)
    )

    return model


