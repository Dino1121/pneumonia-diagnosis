import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_resnet50(model_type="scratch", num_classes=2):
    if model_type == "scratch":
        model = resnet50(weights=None)

    elif model_type == "transfer":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model