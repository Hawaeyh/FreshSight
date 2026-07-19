import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


SUPPORTED_PRETRAINED_WEIGHTS = "IMAGENET1K_V1"


def create_model(
    num_classes: int,
    dropout: float = 0.3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    if num_classes != 3:
        raise ValueError("FreshSight MobileNetV2 requires exactly three output classes.")

    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.last_channel, num_classes),
    )
    return model


def count_parameters(model: nn.Module):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
