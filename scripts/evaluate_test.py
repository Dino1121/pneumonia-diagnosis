import argparse
import csv
import json
import sys
from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "splits" / "group_split.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloaders
from src.models.resnet import create_resnet50


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["scratch", "transfer"]
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    return parser.parse_args()


def evaluate(
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

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            running_loss += (
                loss.item()
                * images.size(0)
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

    test_loss = (
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
        "loss": float(test_loss),
        "accuracy": float(accuracy),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "auroc": float(auroc),
        "confusion_matrix": cm.tolist(),
        "sample_results": sample_results
    }


def save_results(
    experiment_dir,
    model_type,
    experiment_name,
    metrics
):
    result_path = (
        experiment_dir
        / "test_results.json"
    )

    result = {
        "model": model_type,
        "experiment": experiment_name,
        "test_samples": len(
            metrics["sample_results"]
        ),
        "positive_class": "PNEUMONIA",
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "auroc": metrics["auroc"],
        "confusion_matrix": metrics[
            "confusion_matrix"
        ]
    }

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )

    predictions_path = (
        experiment_dir
        / "test_predictions.csv"
    )

    with open(
        predictions_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "true_label",
            "pred_label",
            "p_pneumonia",
            "loss"
        ])

        for result in metrics["sample_results"]:
            writer.writerow([
                result["filename"],
                result["true_label"],
                result["pred_label"],
                result["p_pneumonia"],
                result["loss"]
            ])

    return result_path, predictions_path


def main():
    args = parse_args()

    experiment_dir = (
        PROJECT_ROOT
        / "experiments"
        / args.model
        / args.experiment
    )

    checkpoint_path = (
        experiment_dir
        / "best_model.pth"
    )

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: "
            f"{experiment_dir}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    _, _, test_loader = create_dataloaders(
        csv_path=CSV_PATH,
        project_root=PROJECT_ROOT,
        batch_size=args.batch_size
    )

    model = create_resnet50(
        model_type=args.model,
        num_classes=2
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    print("=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    print(f"Model       : {args.model}")
    print(f"Experiment  : {args.experiment}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Device      : {device}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("=" * 60)

    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    result_path, predictions_path = save_results(
        experiment_dir=experiment_dir,
        model_type=args.model,
        experiment_name=args.experiment,
        metrics=metrics
    )

    cm = np.array(
        metrics["confusion_matrix"]
    )

    print()
    print(f"Test Loss : {metrics['loss']:.6f}")
    print(f"Accuracy  : {metrics['accuracy']:.6f}")
    print(f"Recall    : {metrics['recall']:.6f}")
    print(f"Precision : {metrics['precision']:.6f}")
    print(f"F1-score  : {metrics['f1']:.6f}")
    print(f"AUROC     : {metrics['auroc']:.6f}")

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print(
        f"TN={cm[0, 0]}, "
        f"FP={cm[0, 1]}, "
        f"FN={cm[1, 0]}, "
        f"TP={cm[1, 1]}"
    )

    print()
    print(f"Saved: {result_path}")
    print(f"Saved: {predictions_path}")


if __name__ == "__main__":
    main()