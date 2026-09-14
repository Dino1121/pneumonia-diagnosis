import argparse
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.data.dataloader import create_dataloaders
from src.models.resnet import create_resnet50_scratch
from src.training.optimizer import create_optimizer


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "splits" / "group_split.csv"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "adamw"]
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    return parser.parse_args()


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):
    model.train()

    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)

    return epoch_loss


def validate(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probabilities = torch.softmax(outputs, dim=1)[:, 1]

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    recall = recall_score(
        all_labels,
        all_preds,
        pos_label=1
    )

    precision = precision_score(
        all_labels,
        all_preds,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        pos_label=1
    )

    auroc = roc_auc_score(
        all_labels,
        all_probs
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    return {
        "loss": val_loss,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auroc": auroc,
        "confusion_matrix": cm,
    }


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=CSV_PATH,
        project_root=PROJECT_ROOT,
        batch_size=args.batch_size
    )

    # Scratch ResNet-50
    model = create_resnet50_scratch(
        num_classes=2
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print("=" * 50)
    print("Scratch ResNet-50 Training")
    print("=" * 50)

    print(f"Device       : {device}")
    print(f"Optimizer    : {args.optimizer}")
    print(f"Learning Rate: {args.lr}")
    print(f"Weight Decay : {args.weight_decay}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Epochs       : {args.epochs}")

    print()

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples  : {len(val_loader.dataset)}")
    print(f"Test samples : {len(test_loader.dataset)}")

    print("=" * 50)

    for epoch in range(args.epochs):

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        print(
            f"\nEpoch [{epoch + 1}/{args.epochs}]"
        )

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Val Loss   : {val_metrics['loss']:.4f}"
        )

        print(
            f"Accuracy   : {val_metrics['accuracy']:.4f}"
        )

        print(
            f"Recall     : {val_metrics['recall']:.4f}"
        )

        print(
            f"Precision  : {val_metrics['precision']:.4f}"
        )

        print(
            f"F1-score   : {val_metrics['f1']:.4f}"
        )

        print(
            f"AUROC      : {val_metrics['auroc']:.4f}"
        )

        print("Confusion Matrix:")
        print(val_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()