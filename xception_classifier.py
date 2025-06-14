import torch
import torch.nn as nn
from torchvision import models

class XceptionClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionClassifier, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=True).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x
