import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=False),
        ) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, in_channels, growth_rate, block_config):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate * 2, 7, padding=3, bias=False),
            nn.BatchNorm2d(growth_rate * 2),
            nn.ReLU(inplace=True),
        )
        num_features = growth_rate * 2
        self.dense_blocks = nn.ModuleList([])
        self.transitions = nn.ModuleList([])
        for i, num_layers in enumerate(block_config):
            dense_block = DenseBlock(num_features, growth_rate, num_layers)
            self.dense_blocks.append(dense_block)
            num_features = num_features + growth_rate * num_layers
            if i != len(block_config) - 1:
                transition = Transition(num_features, num_features // 2)
                self.transitions.append(transition)
                num_features = num_features // 2
        self.fc = nn.Linear(num_features, 10)
    def forward(self, x):
        x = self.init_conv(x)
        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transitions[i](x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.fc(x)
        return x

def DenseNet121():
    return DenseNet(3, 32, [6, 12, 24, 16])

if __name__ == "__main__":
    model = DenseNet121()
    print(model)
