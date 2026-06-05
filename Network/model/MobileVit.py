import torch
from torch import nn
from transformers import MobileViTModel
from torchsummary import summary

def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


class MobileVit(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = MobileViTModel.from_pretrained("apple/mobilevit-xx-small")

        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(320, 128, 3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.UpsamplingBilinear2d([400, 400]),
            nn.Conv2d(64, self.num_classes, 1, 1, padding=0)
        )

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = x.last_hidden_state
        x = self.seg_head(x)
        return x


if __name__ == '__main__':
    model = MobileVit(num_classes=21).to('cuda')
    summary(model, (3, 400, 400))
