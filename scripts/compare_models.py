import argparse
import csv
import json
from pathlib import Path


def load_convergence(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {
        "experiment",
        "best_validation_f1",
        "best_f1_epoch",
        "convergence",
    }

    missing_keys = required_keys - set(data.keys())

    if missing_keys:
        raise ValueError(
            f"{path} is missing required keys: "
            f"{', '.join(sorted(missing_keys))}"
        )

    for threshold in ["T95", "T97", "T99"]:
        if threshold not in data["convergence"]:
            raise ValueError(
                f"{path} is missing convergence result: {threshold}"
            )

    return data


def build_row(model, data):
    return {
        "model": model,
        "experiment": data["experiment"],
        "best_val_f1": data["best_validation_f1"],
        "best_f1_epoch": data["best_f1_epoch"],
        "t95_epoch": data["convergence"]["T95"]["epoch"],
        "t97_epoch": data["convergence"]["T97"]["epoch"],
        "t99_epoch": data["convergence"]["T99"]["epoch"],
    }


def save_comparison(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "experiment",
        "best_val_f1",
        "best_f1_epoch",
        "t95_epoch",
        "t97_epoch",
        "t99_epoch",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_comparison(rows):
    print("=" * 85)
    print(
        f"{'Model':<12}"
        f"{'Experiment':<22}"
        f"{'Best F1':<14}"
        f"{'Best Epoch':<14}"
        f"{'T95':<8}"
        f"{'T97':<8}"
        f"{'T99':<8}"
    )
    print("=" * 85)

    for row in rows:
        print(
            f"{row['model']:<12}"
            f"{row['experiment']:<22}"
            f"{row['best_val_f1']:<14.10f}"
            f"{row['best_f1_epoch']:<14}"
            f"{str(row['t95_epoch']):<8}"
            f"{str(row['t97_epoch']):<8}"
            f"{str(row['t99_epoch']):<8}"
        )

    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Scratch and Transfer Learning convergence results"
    )

    parser.add_argument(
        "--scratch",
        type=Path,
        required=True,
        help="Path to Scratch convergence.json",
    )

    parser.add_argument(
        "--transfer",
        type=Path,
        required=True,
        help="Path to Transfer Learning convergence.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/comparison.csv"),
        help="Path to save comparison.csv",
    )

    args = parser.parse_args()

    if not args.scratch.exists():
        raise FileNotFoundError(
            f"Scratch convergence file not found: {args.scratch}"
        )

    if not args.transfer.exists():
        raise FileNotFoundError(
            f"Transfer convergence file not found: {args.transfer}"
        )

    scratch_data = load_convergence(args.scratch)
    transfer_data = load_convergence(args.transfer)

    rows = [
        build_row("Scratch", scratch_data),
        build_row("Transfer", transfer_data),
    ]

    print_comparison(rows)

    save_comparison(
        rows=rows,
        output_path=args.output,
    )

    print()
    print(f"Saved comparison result: {args.output}")


if __name__ == "__main__":
    main()