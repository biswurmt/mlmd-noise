"""ResNet-18 wrapper for MedMNIST experiments."""

import timm
from torch import nn


def build_resnet18(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """Build ResNet-18 with ImageNet-1K pretraining and custom head.

    Uses timm default: resnet18.a1_in1k (ImageNet-1K). No ResNet-18 21K weights
    exist in timm; the paper's "21K" claim likely applies to ViT variants only.
    Returns raw logits (no softmax).

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained weights.

    Returns:
        ResNet-18 model outputting logits of shape (batch, num_classes).
    """
    model = timm.create_model(
        "resnet18",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
