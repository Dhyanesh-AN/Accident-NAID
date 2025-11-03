import torch
import torch.nn as nn

class C3D_Modified(nn.Module):
    """
    Modified C3D for binary classification and grayscale input (1x30x10x4).
    Works with small spatial dimensions by reducing pooling.
    """
    def __init__(self, num_classes=1, pretrained=False):
        super(C3D_Modified, self).__init__()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # keep temporal resolution

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        # Skip spatial pooling here (too small input)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.AdaptiveMaxPool3d((1, 1, 1))  # ensures nonzero output

        self.fc6 = nn.Linear(512 * 1 * 1 * 1, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        logits = self.fc8(x)
        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def __load_pretrained_weights(self):
        """
        Load pretrained Sports1M weights (3-channel → 1-channel adaptation).
        """
        pretrained_dict = torch.load("c3d-pretrained.pth")
        model_dict = self.state_dict()

        # Convert RGB conv1 to grayscale
        if 'conv1.weight' in pretrained_dict:
            old_weight = pretrained_dict['conv1.weight']
            new_weight = old_weight.mean(dim=1, keepdim=True)
            pretrained_dict['conv1.weight'] = new_weight

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print("✅ Loaded pretrained Sports1M weights (adapted for 1-channel input).")


if __name__ == "__main__":
    x = torch.randn(1, 1, 30, 10, 4)
    model = C3D_Modified(num_classes=1, pretrained=True)
    y = model(x)
    print("Output shape:", y.shape)
