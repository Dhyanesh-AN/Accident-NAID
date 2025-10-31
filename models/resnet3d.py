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
        
            # Handle first conv layer
            conv1_weight = state_dict['conv1.weight']  # shape: [64, 3, 7, 7, 7]
            # Average across channel dimension to make it compatible with 1 input channel
            conv1_weight_gray = conv1_weight.mean(dim=1, keepdim=True)  # shape: [64, 1, 7, 7, 7]
            state_dict['conv1.weight'] = conv1_weight_gray
        
            # Remove classifier weights if shapes mismatch
            filtered = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(f"✅ Loaded pretrained weights from {pretrained_path}")
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")
            

def get_inplanes():
    return [64, 128, 256, 512]

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet3D(Base3DModel):
    def __init__(self, block=Bottleneck, layers=[3, 4, 23, 3], num_classes=1, in_channels=1):
        super(ResNet3D, self).__init__(num_classes, in_channels)
        planes = get_inplanes()
        self.in_planes = planes[0]
        self.conv1 = nn.Conv3d(in_channels, planes[0], kernel_size=(7, 7, 7),
                               stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, planes[0], layers[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(planes[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        layers += [block(self.in_planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.fc(x))
