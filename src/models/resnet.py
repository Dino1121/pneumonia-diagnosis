import torch.nn as nn
from torchvision.models import resnet50


def create_resnet50_scratch(num_classes=2):
    """
    ResNet-50 Scratch 모델 생성

    - ImageNet pretrained weight 사용 X
    - 마지막 FC layer를 num_classes에 맞게 변경
    """

    model = resnet50(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model