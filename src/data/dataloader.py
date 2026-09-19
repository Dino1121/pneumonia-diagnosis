from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.data.transforms import (
    train_transform,
    eval_transform,
)


def create_dataloaders(
    csv_path,
    project_root,
    batch_size=32,
    num_workers=1
):
    """
    Train / Validation / Test DataLoader 생성
    """

    csv_path = Path(csv_path)
    project_root = Path(project_root)

  
    # Dataset
    train_dataset = ChestXrayDataset(
        csv_path=csv_path,
        split="train",
        transform=train_transform,
        project_root=project_root
    )

    val_dataset = ChestXrayDataset(
        csv_path=csv_path,
        split="val",
        transform=eval_transform,
        project_root=project_root,
        return_filename=True
    )

    test_dataset = ChestXrayDataset(
        csv_path=csv_path,
        split="test",
        transform=eval_transform,
        project_root=project_root
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers = True
    )   

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers = True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers = True
    )

    return train_loader, val_loader, test_loader