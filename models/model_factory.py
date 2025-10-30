import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

from .c3d import C3D_Modified
from .mobilenetv2_3d import MobileNetV2_3D
from .resnet3d import ResNet3D


def get_3d_model(model_name, num_classes=1, in_channels=1, pretrained_path=None):
    model_name = model_name.lower()
    if model_name == "mobilenet3d":
        model = MobileNetV2_3D(num_classes, in_channels)
    elif model_name == "c3d":
        model = C3D_Modified(num_classes, in_channels)
    elif model_name == "resnet3d":
        model = ResNet3D(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if pretrained_path:
        model.load_pretrained(pretrained_path)

    return model

if __name__ == "__main__":
    model_name = "mobilenet3d"
    model = get_3d_model(model_name, num_classes=1, in_channels=1, pretrained_path=None)
    x = torch.randn(1, 1, 30, 10, 4)
    y = model(x)
    print(f"{model_name} output:", y.shape)
