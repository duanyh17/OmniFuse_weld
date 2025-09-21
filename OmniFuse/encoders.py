# encoders.py
import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=512, pretrained=False):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        f = self.backbone(x)  # [B, 512]
        return self.proj(f)

class SoundEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc(h)

class CurrentEncoder(nn.Module):
    def __init__(self, in_len=100, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_len, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.net(x)
