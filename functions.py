import torch

from network import Outputs


def predict_label(outputs, classifier_type):
    if classifier_type == Outputs.BINARY or classifier_type == Outputs.MULTI_CLASS:
        return outputs.argmax(1)
    elif classifier_type == Outputs.MULTI_BINARY:
        idx = 0
        label = torch.zeros([outputs.size()[0], 4]).long()
        for score in range(0, outputs.size()[1], 2):
            label[:, idx] = outputs[:, score:score + 2].argmax(1).cpu()
            idx += 1
        return torch.sum(label, axis=1)[:, None]
    elif classifier_type == Outputs.REGRESSOR:
        return torch.round(outputs)
    else:
        raise NotImplementedError
