import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeDetectorResNet(nn.Module):
    """
    A ResNet-based model for 5-Class Audio Deepfake Classification.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super(DeepfakeDetectorResNet, self).__init__()
        
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        
        # Change input channels from 3 to 1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Update output layer for 5 classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
