from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# 1. Aspect Ratio 유지 Resize + Padding
class ResizeWithPadding:
    def __init__(self, target_size=224, fill=0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size

        # target_size 안에 들어가도록 scale 계산
        scale = min(
            self.target_size / w,
            self.target_size / h
        )

        new_w = round(w * scale)
        new_h = round(h * scale)

        # 비율 유지 Resize
        img = img.resize(
            (new_w, new_h),
            resample=Image.Resampling.BILINEAR
        )

        # 남는 공간 계산
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h

        left = pad_w // 2
        right = pad_w - left

        top = pad_h // 2
        bottom = pad_h - top

        # Zero Padding
        img = ImageOps.expand(
            img,
            border=(left, top, right, bottom),
            fill=self.fill
        )

        return img


# 2. ImageNet Normalization 값
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# 3. Train Transform
train_transform = transforms.Compose([
    # 무조건 1채널 grayscale
    transforms.Grayscale(num_output_channels=1),

    # 비율 유지 Resize + Zero Padding
    ResizeWithPadding(
        target_size=224,
        fill=0
    ),

    # Train에만 augmentation
    transforms.RandomRotation(
        degrees=10,
        interpolation=InterpolationMode.BILINEAR,
        fill=0
    ),

    # 1채널 → 3채널
    transforms.Grayscale(num_output_channels=3),

    # PIL Image → Tensor
    # 0~255 → 0~1
    transforms.ToTensor(),

    # ImageNet Normalization
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# 4. Validation / Test Transform
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),

    ResizeWithPadding(
        target_size=224,
        fill=0
    ),

    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])