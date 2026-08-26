"""
model.py
--------
Transfer-learning model built on EfficientNet-B0 (ImageNet-pretrained).
A custom classification head is attached on top of the pretrained backbone.

EfficientNet-B0 was chosen because:
  - It performs well on medical/dermatoscopic image tasks
  - It is lighter than ResNet50, resulting in faster training
  - Pretrained weights are readily available via torchvision
"""

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(num_classes=7, freeze_backbone=True):
    """
    Args:
        num_classes (int): Number of output classes (7 for multi-class,
                            2 for binary classification)
        freeze_backbone (bool): If True, only the classifier head is trained
                                 (faster training, works well with limited data).
                                 If False, the entire network is fine-tuned
                                 (requires more data and training time, but
                                 can yield higher accuracy).
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the default EfficientNet-B0 classifier (Linear: 1280 -> 1000)
    # with a custom head sized for this task
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(256, num_classes),
    )

    return model