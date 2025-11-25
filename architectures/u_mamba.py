import torch
import torch.nn as nn

class SimpleUMambaBackbone(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, img_size=(64, 64)):
        super().__init__()
        # Basit bir örnek backbone (gerçek U-Mamba kodunu buraya ekleyeceksin)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_size = img_size
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # [B, hidden_dim, 1, 1]
        x = x.view(x.size(0), -1)  # [B, hidden_dim]
        return x

class UMambaForClassification(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, hidden_dim=64, img_size=(64, 64)):
        super().__init__()
        self.backbone = SimpleUMambaBackbone(in_channels, hidden_dim, img_size)
        self.fc = nn.Linear(hidden_dim, num_classes)  # buy/hold/sell

    def forward(self, x):
        features = self.backbone(x)
        out = self.fc(features)
        return out