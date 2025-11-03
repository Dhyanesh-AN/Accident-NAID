import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

# ---------- Basic Conv Blocks ----------

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

# ---------- Inverted Residual Block ----------

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = (self.stride == (1, 1, 1)) and (inp == oup)

        layers = []
        if expand_ratio != 1:
            # pw
            layers.extend([
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        # dw
        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm3d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# ---------- Main MobileNetV2 3D Model ----------

class MobileNetV2_3D_Binary(nn.Module):
    def __init__(self, num_classes=1, n_input_channels=1, sample_size=10, width_mult=1.):
        super(MobileNetV2_3D_Binary, self).__init__()
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

        # first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(n_input_channels, input_channel, (1, 2, 2))]

        # inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # last conv layer
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        # classifier (binary)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])  # global avg pool
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = torch.sigmoid(x)  # binary output in [0,1]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ---------- Utility to load pretrained weights ----------

def load_pretrained_mobilenet3d(model, pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    filtered_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
    model_dict = model.state_dict()
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)
    print("✅ Loaded pretrained weights (except classifier)")
    return model


# ---------- Example Usage ----------

if __name__ == "__main__":
    model = MobileNetV2_3D_Binary(num_classes=1, n_input_channels=1, sample_size=10)
    model = model.cuda()
    model = load_pretrained_mobilenet3d(model, "/content/kinetics_mobilenetv2_1.0x_RGB_16_best.pth")

    input_var = Variable(torch.randn(1, 1, 30, 10, 4)).cuda()  # (B, C, D, H, W)
    output = model(input_var)
    print("Output shape:", output.shape)
    print("Output:", output)
