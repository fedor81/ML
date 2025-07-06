import torch
import torch.nn as nn
import torch.nn.functional as F
from .custom_layers import ResidualBlock


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ShallowCNN(nn.Module):
    """Неглубокая CNN (2 conv слоя)"""

    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Для визуализации feature maps
        self.feature_maps = []

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        self.feature_maps.append(x.detach().cpu())  # Сохраняем для визуализации
        x = self.pool(F.relu(self.conv2(x)))
        self.feature_maps.append(x.detach().cpu())
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MediumCNN(nn.Module):
    """Средняя CNN (4 conv слоя)"""

    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.feature_maps = []

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        self.feature_maps.append(x.detach().cpu())
        x = self.pool(F.relu(self.conv2(x)))
        self.feature_maps.append(x.detach().cpu())
        x = self.pool(F.relu(self.conv3(x)))
        self.feature_maps.append(x.detach().cpu())
        x = self.pool(F.relu(self.conv4(x)))
        self.feature_maps.append(x.detach().cpu())
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DeepCNN(nn.Module):
    """Глубокая CNN (6 conv слоев)"""

    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.feature_maps = []

    def forward(self, x):
        x = F.relu(self.conv1(x))
        self.feature_maps.append(x.detach().cpu())
        x = self.pool(F.relu(self.conv2(x)))
        self.feature_maps.append(x.detach().cpu())
        x = F.relu(self.conv3(x))
        self.feature_maps.append(x.detach().cpu())
        x = self.pool(F.relu(self.conv4(x)))
        self.feature_maps.append(x.detach().cpu())
        x = F.relu(self.conv5(x))
        self.feature_maps.append(x.detach().cpu())
        x = self.pool(F.relu(self.conv6(x)))
        self.feature_maps.append(x.detach().cpu())
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CNNWithResidual(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, 2)
        self.res3 = ResidualBlock(64, 64)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RegularizedResidual(CNNWithResidual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.res1(x)
        x = self.dropout(x)
        x = self.res2(x)
        x = self.dropout(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CIFARCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class KernelizableCNN(nn.Module):
    def __init__(self, layers: list[dict], in_channels=3, out_channels=10):
        super().__init__()
        __layers = []

        for layer in layers:
            __layers += [
                nn.Conv2d(
                    in_channels,
                    layer["out"],
                    kernel_size=layer["kernel"],
                    padding=layer["kernel"] // 2,
                ),
                nn.ReLU(),
            ]
            in_channels = layer["out"]

        __layers.append(torch.nn.AdaptiveAvgPool2d(1))
        self.features = torch.nn.Sequential(*__layers)
        self.classifier = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ResidualConfigurable(nn.Module):
    def __init__(self, block_type, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block_type, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_type, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_type, 256, num_blocks[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, block_type, out_channels, num_blocks, stride):
        layers = []
        layers.append(block_type(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block_type(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
