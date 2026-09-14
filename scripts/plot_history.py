from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--history",
        type=str,
        required=True,
        help="Path to history.csv"
    )

    return parser.parse_args()


def plot_loss(df, save_dir):
    plt.figure(figsize=(9, 5))

    plt.plot(
        df["epoch"],
        df["train_loss"],
        marker="o",
        label="Train Loss"
    )

    plt.plot(
        df["epoch"],
        df["val_loss"],
        marker="o",
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = save_dir / "loss_curve.png"

    plt.savefig(
        save_path,
        dpi=180
    )

    plt.close()

    print(f"Saved: {save_path}")


def plot_metrics(df, save_dir):
    plt.figure(figsize=(9, 5))

    plt.plot(
        df["epoch"],
        df["f1"],
        marker="o",
        label="F1"
    )

    plt.plot(
        df["epoch"],
        df["recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        df["epoch"],
        df["auroc"],
        marker="o",
        label="AUROC"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")

    plt.ylim(0, 1.05)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = save_dir / "validation_metrics.png"

    plt.savefig(
        save_path,
        dpi=180
    )

    plt.close()

    print(f"Saved: {save_path}")


def main():
    args = parse_args()

    history_path = Path(args.history)

    if not history_path.exists():
        raise FileNotFoundError(
            f"History file not found: {history_path}"
        )

    df = pd.read_csv(history_path)

    save_dir = history_path.parent

    plot_loss(
        df=df,
        save_dir=save_dir
    )

    plot_metrics(
        df=df,
        save_dir=save_dir
    )


if __name__ == "__main__":
    main()