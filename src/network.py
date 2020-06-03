import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class ResBlock(nn.Module):
    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.filters = filters

        self.conv1 = nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(self.filters, affine=True)

        self.conv2 = nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(self.filters, affine=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.in1(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = self.in2(y)
        y += x
        return F.relu(y)


class LossNet(nn.Module):
    STYLE_LAYER = {3, 8, 15, 22}
    FEAT_LAYER = {15}

    def __init__(self):
        super(LossNet, self).__init__()
        mymo = vgg16(pretrained=True)
        for p in mymo.parameters():
            p.requires_grad_(False)

        features = list(mymo.features)
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        style_results, feature_results = [], []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in LossNet.STYLE_LAYER:
                style_results.append(x)
            if ii in LossNet.FEAT_LAYER:
                feature_results.append(x)

        return style_results, feature_results


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 9, stride=1, padding=4),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 3, 9, 1, padding=4)
            #nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)