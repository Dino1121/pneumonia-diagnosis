import argparse
import csv
import json
from pathlib import Path


THRESHOLD_RATIOS = [0.95, 0.97, 0.99]
WINDOW_SIZE = 10
REQUIRED_COUNT = 8


def load_history(history_path):
    rows = []

    with open(history_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required_columns = {"epoch", "f1"}
        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "f1": float(row["f1"]),
                }
            )

    if not rows:
        raise ValueError("history.csv is empty.")

    rows.sort(key=lambda x: x["epoch"])

    return rows


def find_best_f1(rows):
    best_row = max(rows, key=lambda x: x["f1"])

    return best_row["f1"], best_row["epoch"]


def find_convergence_epoch(
    rows,
    threshold,
    window_size=WINDOW_SIZE,
    required_count=REQUIRED_COUNT,
):
    for i in range(len(rows) - window_size + 1):
        current = rows[i]

        if current["f1"] < threshold:
            continue

        window = rows[i:i + window_size]

        maintained_count = sum(
            row["f1"] >= threshold
            for row in window
        )

        if maintained_count >= required_count:
            return {
                "epoch": current["epoch"],
                "window_start": window[0]["epoch"],
                "window_end": window[-1]["epoch"],
                "maintained_count": maintained_count,
            }

    return None


def analyze(history_path, model_name):
    rows = load_history(history_path)

    best_f1, best_epoch = find_best_f1(rows)

    result = {
        "experiment": model_name,
        "history": str(history_path),
        "total_epochs": len(rows),
        "best_validation_f1": best_f1,
        "best_f1_epoch": best_epoch,
        "convergence_rule": {
            "window_size": WINDOW_SIZE,
            "required_count": REQUIRED_COUNT,
        },
        "convergence": {},
    }

    print("=" * 60)
    print(f"Model              : {model_name}")
    print(f"History            : {history_path}")
    print(f"Total Epochs       : {len(rows)}")
    print(f"Best Validation F1 : {best_f1:.10f}")
    print(f"Best F1 Epoch      : {best_epoch}")
    print("=" * 60)

    for ratio in THRESHOLD_RATIOS:
        label = f"T{int(ratio * 100)}"
        threshold = best_f1 * ratio

        convergence = find_convergence_epoch(
            rows=rows,
            threshold=threshold,
        )

        print()
        print(f"[{label}]")
        print(f"Threshold          : {threshold:.10f}")

        if convergence is None:
            print("Convergence Epoch  : Not reached")
            print(
                f"Stability Rule     : >= {REQUIRED_COUNT}/{WINDOW_SIZE}"
            )

            result["convergence"][label] = {
                "ratio": ratio,
                "threshold": threshold,
                "epoch": None,
                "window_start": None,
                "window_end": None,
                "maintained_count": None,
            }

        else:
            print(
                f"Convergence Epoch  : {convergence['epoch']}"
            )
            print(
                f"Stability Window   : "
                f"{convergence['window_start']}"
                f"-{convergence['window_end']}"
            )
            print(
                f"Threshold Maintained: "
                f"{convergence['maintained_count']}/{WINDOW_SIZE}"
            )

            result["convergence"][label] = {
                "ratio": ratio,
                "threshold": threshold,
                "epoch": convergence["epoch"],
                "window_start": convergence["window_start"],
                "window_end": convergence["window_end"],
                "maintained_count": convergence["maintained_count"],
            }

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Best F1            : {best_f1:.10f}")
    print(f"Best F1 Epoch      : {best_epoch}")

    for label in ["T95", "T97", "T99"]:
        epoch = result["convergence"][label]["epoch"]

        if epoch is None:
            print(f"{label} Convergence     : Not reached")
        else:
            print(f"{label} Convergence     : Epoch {epoch}")

    return result


def save_result(result, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Saved convergence result: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze validation F1 convergence from history.csv"
    )

    parser.add_argument(
        "--history",
        type=Path,
        required=True,
        help="Path to history.csv",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="Model",
        help="Model or experiment name",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save convergence result as JSON",
    )

    args = parser.parse_args()

    if not args.history.exists():
        raise FileNotFoundError(
            f"History file not found: {args.history}"
        )

    result = analyze(
        history_path=args.history,
        model_name=args.name,
    )

    if args.output is not None:
        save_result(
            result=result,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()