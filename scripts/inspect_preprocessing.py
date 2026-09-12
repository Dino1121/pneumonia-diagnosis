from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.transforms import (
    ResizeWithPadding,
    train_transform,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ==========================================
# 1. 확인할 이미지 경로
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
    / "train"
    / "NORMAL"
    / "IM-0115-0001.jpeg"
)


# ==========================================
# 2. 이미지 불러오기
# ==========================================
original = Image.open(IMAGE_PATH)

# 무조건 1채널 Grayscale
gray = original.convert("L")


# ==========================================
# 3. Resize + Padding
# ==========================================
resize_padding = ResizeWithPadding(
    target_size=224,
    fill=0
)

resized = resize_padding(gray)


# ==========================================
# 4. Random Rotation
# ==========================================
rotation = transforms.RandomRotation(
    degrees=10,
    interpolation=InterpolationMode.BILINEAR,
    fill=0
)

rotated = rotation(resized)


# ==========================================
# 5. 실제 train_transform 전체 적용
# ==========================================
final_tensor = train_transform(original)


# ==========================================
# 6. Tensor 정보 확인
# ==========================================
print("Original mode :", original.mode)
print("Original size :", original.size)

print("\nFinal tensor")
print("Shape :", final_tensor.shape)
print("Dtype :", final_tensor.dtype)
print("Min   :", final_tensor.min().item())
print("Max   :", final_tensor.max().item())


# ==========================================
# 7. Normalize 되돌리기
# ==========================================
mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

denormalized = final_tensor * std + mean

# [C, H, W] → [H, W, C]
final_image = denormalized.permute(1, 2, 0)

# matplotlib 표시 범위로 제한
final_image = torch.clamp(final_image, 0, 1)


# ==========================================
# 8. Matplotlib 시각화
# ==========================================
plt.figure(figsize=(16, 4))

# Original
plt.subplot(1, 4, 1)
plt.imshow(gray, cmap="gray")
plt.title(f"Original\n{gray.size}")
plt.axis("off")

# Resize + Padding
plt.subplot(1, 4, 2)
plt.imshow(resized, cmap="gray")
plt.title(f"Resize + Padding\n{resized.size}")
plt.axis("off")

# Rotation
plt.subplot(1, 4, 3)
plt.imshow(rotated, cmap="gray")
plt.title("Random Rotation")
plt.axis("off")

# Final Transform
plt.subplot(1, 4, 4)
plt.imshow(final_image)
plt.title("Final Train Transform")
plt.axis("off")

plt.tight_layout()
plt.show()