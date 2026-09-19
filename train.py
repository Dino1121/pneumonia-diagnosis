import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np
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
from src.models.resnet import create_resnet50
from src.training.optimizer import create_optimizer


PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "splits" / "group_split.csv"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_backbone(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def unfreeze_all(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="scratch",
        choices=["scratch", "transfer"]
    )

    parser.add_argument(
        "--transfer_pilot",
        action="store_true"
    )

    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--freeze_lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=1e-4
    )

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
        default=1e-4
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

    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "plateau"]
    )

    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=10
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6
    )

    return parser.parse_args()


def create_experiment_dir(model_type, transfer_pilot=False):
    if model_type == "transfer" and transfer_pilot:
        experiment_root = PROJECT_ROOT / "experiments" / "transfer" / "pilot_fc_freeze"
    else:
        experiment_root = PROJECT_ROOT / "experiments" / model_type

    experiment_root.mkdir(
        parents=True,
        exist_ok=True
    )

    existing_numbers = []

    for path in experiment_root.iterdir():
        if path.is_dir() and path.name.startswith("exp_"):
            try:
                number = int(path.name.split("_")[1])
                existing_numbers.append(number)
            except ValueError:
                pass

    if existing_numbers:
        experiment_number = max(existing_numbers) + 1
    else:
        experiment_number = 1

    experiment_dir = (
        experiment_root
        / f"exp_{experiment_number:03d}"
    )

    experiment_dir.mkdir()

    return experiment_dir


def save_config(
    experiment_dir,
    args
):
    config = {
        "architecture": "resnet50",
        "model_type": args.model,
        "pretrained_weights": (
            "IMAGENET1K_V2"
            if args.model == "transfer"
            else None
        ),
        "transfer_pilot": args.transfer_pilot,
        "training_phase": (
            "fc_only_pilot"
            if args.model == "transfer" and args.transfer_pilot
            else "freeze_then_finetune"
            if args.model == "transfer"
            else "full_model"
        ),
        "backbone_frozen": (
            True
            if args.model == "transfer"
            else False
        ),
        "freeze_epochs": (
            args.freeze_epochs
            if args.model == "transfer" and not args.transfer_pilot
            else None
        ),
        "freeze_learning_rate": (
            args.freeze_lr
            if args.model == "transfer"
            else None
        ),
        "finetune_learning_rate": (
            args.finetune_lr
            if args.model == "transfer" and not args.transfer_pilot
            else None
        ),
        "optimizer": args.optimizer,
        "learning_rate": (
            args.freeze_lr
            if args.model == "transfer"
            else args.lr
        ),
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "loss": "CrossEntropyLoss",
        "num_classes": 2,
        "positive_class": "PNEUMONIA",
        "seed": 42,
        "scheduler": args.scheduler,
        "created_at": datetime.now().isoformat()
    }

    if args.scheduler == "plateau":
        config.update({
            "scheduler_monitor": "val_loss",
            "scheduler_mode": "min",
            "scheduler_factor": args.scheduler_factor,
            "scheduler_patience": args.scheduler_patience,
            "scheduler_min_lr": args.min_lr
        })

    config_path = experiment_dir / "config.json"

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            config,
            file,
            indent=4,
            ensure_ascii=False
        )


def initialize_history_file(
    experiment_dir
):
    history_path = experiment_dir / "history.csv"

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "phase",
            "train_loss",
            "val_loss",
            "accuracy",
            "recall",
            "precision",
            "f1",
            "auroc",
            "lr"
        ])

    return history_path


def initialize_val_predictions_file(
    experiment_dir
):
    predictions_path = experiment_dir / "val_predictions.csv"

    with open(
        predictions_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "filename",
            "true_label",
            "pred_label",
            "p_pneumonia",
            "loss"
        ])

    return predictions_path


def save_epoch_result(
    history_path,
    epoch,
    phase,
    train_loss,
    val_metrics,
    learning_rate
):
    with open(
        history_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            epoch,
            phase,
            train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["recall"],
            val_metrics["precision"],
            val_metrics["f1"],
            val_metrics["auroc"],
            learning_rate
        ])


def save_val_predictions(
    predictions_path,
    epoch,
    sample_results
):
    with open(
        predictions_path,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        for result in sample_results:
            writer.writerow([
                epoch,
                result["filename"],
                result["true_label"],
                result["pred_label"],
                result["p_pneumonia"],
                result["loss"]
            ])


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
        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

    epoch_loss = (
        running_loss
        / len(dataloader.dataset)
    )

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

    all_filenames = []
    all_sample_losses = []

    sample_criterion = nn.CrossEntropyLoss(
        reduction="none"
    )

    with torch.no_grad():

        for images, labels, filenames in dataloader:
            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            sample_losses = sample_criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_preds.extend(
                predictions.cpu().numpy()
            )

            all_probs.extend(
                probabilities.cpu().numpy()
            )

            all_filenames.extend(
                filenames
            )

            all_sample_losses.extend(
                sample_losses.cpu().numpy()
            )

    val_loss = (
        running_loss
        / len(dataloader.dataset)
    )

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

    sample_results = []

    for (
        filename,
        true_label,
        pred_label,
        probability,
        sample_loss
    ) in zip(
        all_filenames,
        all_labels,
        all_preds,
        all_probs,
        all_sample_losses
    ):
        sample_results.append({
            "filename": filename,
            "true_label": int(true_label),
            "pred_label": int(pred_label),
            "p_pneumonia": float(probability),
            "loss": float(sample_loss)
        })

    return {
        "loss": val_loss,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auroc": auroc,
        "confusion_matrix": cm,
        "sample_results": sample_results
    }


def main():
    args = parse_args()

    set_seed(42)

    if args.transfer_pilot and args.model != "transfer":
        raise ValueError(
            "--transfer_pilot can only be used with --model transfer."
        )

    if args.freeze_epochs < 0:
        raise ValueError(
            "--freeze_epochs must be 0 or greater."
        )

    if (
        args.model == "transfer"
        and not args.transfer_pilot
        and args.freeze_epochs >= args.epochs
    ):
        raise ValueError(
            "--freeze_epochs must be smaller than --epochs for full transfer training."
        )

    experiment_dir = create_experiment_dir(
        model_type=args.model,
        transfer_pilot=args.transfer_pilot
    )

    save_config(
        experiment_dir=experiment_dir,
        args=args
    )

    history_path = initialize_history_file(
        experiment_dir=experiment_dir
    )

    predictions_path = initialize_val_predictions_file(
        experiment_dir=experiment_dir
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=CSV_PATH,
        project_root=PROJECT_ROOT,
        batch_size=args.batch_size
    )

    model = create_resnet50(
        model_type=args.model,
        num_classes=2
    )

    if args.model == "transfer":
        if args.transfer_pilot or args.freeze_epochs > 0:
            freeze_backbone(model)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if args.model == "transfer":
        initial_lr = args.freeze_lr
    else:
        initial_lr = args.lr

    optimizer = create_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        lr=initial_lr,
        weight_decay=args.weight_decay
    )

    scheduler = None

    if args.model == "scratch" and args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_lr
        )

    if (
        args.model == "transfer"
        and args.transfer_pilot
        and args.scheduler == "plateau"
    ):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_lr
        )

    print("=" * 50)
    print(f"{args.model.capitalize()} ResNet-50 Training")

    if args.model == "transfer" and args.transfer_pilot:
        print("Mode         : FC-only Freeze Pilot")
    elif args.model == "transfer":
        print("Mode         : Freeze -> Full Fine-tuning")

    print("=" * 50)

    print(f"Experiment   : {experiment_dir.name}")
    print(f"Save Path    : {experiment_dir}")
    print(f"Device       : {device}")
    print(f"Optimizer    : {args.optimizer}")
    if args.model == "transfer":
        print(f"Freeze LR    : {args.freeze_lr}")
        if not args.transfer_pilot:
            print(f"Fine-tune LR : {args.finetune_lr}")
            print(f"Freeze Epochs: {args.freeze_epochs}")
    else:
        print(f"Initial LR   : {args.lr}")

    print(f"Weight Decay : {args.weight_decay}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Epochs       : {args.epochs}")
    print(f"Scheduler    : {args.scheduler}")

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Trainable Params: "
        f"{trainable_params:,} / {total_params:,}"
    )

    if args.scheduler == "plateau":
        print(f"LR Factor    : {args.scheduler_factor}")
        print(f"LR Patience  : {args.scheduler_patience}")
        print(f"Minimum LR   : {args.min_lr}")

    print()

    print(
        f"Train samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Val samples  : "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test samples : "
        f"{len(test_loader.dataset)}"
    )

    print("=" * 50)

    best_f1 = -1.0

    for epoch in range(args.epochs):

        epoch_number = epoch + 1

        if args.model == "scratch":
            phase = "scratch"
        elif args.transfer_pilot:
            phase = "fc_only_pilot"
        elif epoch_number <= args.freeze_epochs:
            phase = "fc_only"
        else:
            phase = "fine_tuning"

        if (
            args.model == "transfer"
            and not args.transfer_pilot
            and epoch_number == args.freeze_epochs + 1
        ):
            unfreeze_all(model)

            optimizer = create_optimizer(
                model=model,
                optimizer_name=args.optimizer,
                lr=args.finetune_lr,
                weight_decay=args.weight_decay
            )

            scheduler = None

            if args.scheduler == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=args.scheduler_factor,
                    patience=args.scheduler_patience,
                    min_lr=args.min_lr
                )

            print("\n" + "=" * 50)
            print(f"Epoch {epoch_number}: Backbone Unfrozen")
            print("Phase        : Full Fine-tuning")
            print(f"Fine-tune LR : {args.finetune_lr}")
            print("=" * 50)

        current_lr = optimizer.param_groups[0]["lr"]

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

        save_epoch_result(
            history_path=history_path,
            epoch=epoch_number,
            phase=phase,
            train_loss=train_loss,
            val_metrics=val_metrics,
            learning_rate=current_lr
        )

        save_val_predictions(
            predictions_path=predictions_path,
            epoch=epoch_number,
            sample_results=val_metrics["sample_results"]
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]

            torch.save(
                model.state_dict(),
                experiment_dir / "best_model.pth"
            )

        next_lr = current_lr

        if scheduler is not None:
            scheduler.step(
                val_metrics["loss"]
            )

            next_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch "
            f"[{epoch_number}/{args.epochs}]"
        )

        print(
            f"Phase      : "
            f"{phase}"
        )

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"Accuracy   : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Recall     : "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"Precision  : "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"F1-score   : "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"AUROC      : "
            f"{val_metrics['auroc']:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.2e}"
        )

        if next_lr != current_lr:
            print(
                f"LR Reduced   : "
                f"{current_lr:.2e} -> {next_lr:.2e}"
            )

        print("Confusion Matrix:")
        print(
            val_metrics[
                "confusion_matrix"
            ]
        )

        print(
            f"Best Val F1: "
            f"{best_f1:.4f}"
        )


if __name__ == "__main__":
    main()