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
            
            
class C3D_Modified(Base3DModel):
    def __init__(self, num_classes=1, in_channels=1):
        super(C3D_Modified, self).__init__(num_classes, in_channels)

        self.conv1 = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))

        self.conv4a = nn.Conv3d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))

        self.pool5 = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)
        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.pool5(x)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.fc(x))
