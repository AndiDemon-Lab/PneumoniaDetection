import torch.nn as nn
from torchvision import models

class CustomVGG16(nn.Module):
    def __init__(self, dropout=0.5, device='cuda'):
        super(CustomVGG16, self).__init__()
        vgg16 = models.vgg16(pretrained=True).to(device)

        # Freeze the features
        for param in vgg16.features.parameters():
            param.requires_grad = False

        self.features = vgg16.features.to(device)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
            nn.Softmax()
        ).to(device)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x