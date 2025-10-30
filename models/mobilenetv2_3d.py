import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial


# ======================================================
# Base Model Interface
# ======================================================

class Base3DModel(nn.Module):
    def __init__(self, num_classes=1, in_channels=1):
        super(Base3DModel, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

    def load_pretrained(self, pretrained_path):
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            filtered = {k: v for k, v in state_dict.items() if 'fc' not in k and 'classifier' not in k}
            self.load_state_dict(filtered, strict=False)
            print(f"✅ Loaded pretrained weights (except classifier) from {pretrained_path}")
            

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = (stride == (1, 1, 1)) and (inp == oup)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True)
            ]
        layers += [
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm3d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class MobileNetV2_3D(Base3DModel):
    def __init__(self, num_classes=1, in_channels=1):
        super(MobileNetV2_3D, self).__init__(num_classes, in_channels)
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (2, 2, 2)],
            [6, 32, 3, (2, 2, 2)],
            [6, 64, 4, (2, 2, 2)],
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 3, (2, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]

        features = [conv_bn(in_channels, input_channel, (1, 2, 2))]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(conv_1x1x1_bn(input_channel, last_channel))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool3d(x, 1).view(x.size(0), -1)
        return torch.sigmoid(self.classifier(x))
