import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=29):
    model = models.segmentation.fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    return model
