from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SEARCH_ROOT = PROJECT_ROOT / "data" / "raw" / "chest_xray"
PREDICTIONS_CSV = PROJECT_ROOT / "experiments" / "scratch" / "exp_006" / "val_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "audit" / "hard_samples_5plus"

SPIKE_EPOCHS = [25, 40, 55, 61, 78, 91, 97]
MIN_WRONG_COUNT = 5

df = pd.read_csv(PREDICTIONS_CSV)

spike_df = df[df["epoch"].isin(SPIKE_EPOCHS)].copy()
spike_df["is_wrong"] = spike_df["true_label"] != spike_df["pred_label"]

summary = (
    spike_df.groupby("filename")
    .agg(
        true_label=("true_label", "first"),
        wrong_count=("is_wrong", "sum"),
        mean_loss=("loss", "mean"),
        max_loss=("loss", "max"),
    )
    .reset_index()
)

hard_samples = (
    summary[summary["wrong_count"] >= MIN_WRONG_COUNT]
    .sort_values(["wrong_count", "max_loss"], ascending=[False, False])
    .reset_index(drop=True)
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Spike epochs: {SPIKE_EPOCHS}")
print(f"Hard sample criterion: >= {MIN_WRONG_COUNT}/{len(SPIKE_EPOCHS)} misclassified")
print(f"Found: {len(hard_samples)} images")
print()

for _, row in hard_samples.iterrows():
    filename = row["filename"]
    matches = list(SEARCH_ROOT.rglob(filename))

    if not matches:
        print(f"[NOT FOUND] {filename}")
        continue

    src = matches[0]
    dst = OUTPUT_DIR / filename

    shutil.copy2(src, dst)

    print(
        f"[COPIED] {filename} | "
        f"wrong={int(row['wrong_count'])}/7 | "
        f"max_loss={row['max_loss']:.4f}"
    )

summary_path = OUTPUT_DIR / "hard_samples_summary.csv"
hard_samples.to_csv(summary_path, index=False)

print()
print(f"Images saved to: {OUTPUT_DIR}")
print(f"Summary saved to: {summary_path}")