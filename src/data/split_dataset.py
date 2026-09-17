from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_PATH = OUTPUT_DIR / "group_split.csv"

RANDOM_STATE = 42

EXCLUDED_DUPLICATES = {
    "NORMAL2-IM-0095-0001.jpeg",
}


def get_group_id(filename: str, label: str):

    if label == "PNEUMONIA":
        match = re.search(r"(person\d+)", filename, re.IGNORECASE)

    elif label == "NORMAL":
        match = re.search(r"(IM-\d+)", filename, re.IGNORECASE)

    else:
        raise ValueError(f"Unknown label: {label}")

    if match is None:
        raise ValueError(f"Group ID를 추출할 수 없습니다: {filename}")

    return f"{label}_{match.group(1).upper()}"


def build_dataframe():

    records = []
    excluded = []

    for original_split in ["train", "val", "test"]:

        for label in ["NORMAL", "PNEUMONIA"]:

            folder = DATA_DIR / original_split / label

            for filepath in folder.glob("*"):

                if filepath.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                if filepath.name in EXCLUDED_DUPLICATES:
                    excluded.append(filepath)
                    continue

                group_id = get_group_id(filepath.name, label)

                records.append({
                    "filepath": filepath.relative_to(PROJECT_ROOT).as_posix(),
                    "label": label,
                    "group_id": group_id,
                    "original_split": original_split
                })

    print(f"Excluded duplicate images: {len(excluded)}")

    for filepath in excluded:
        print(f"  - {filepath.relative_to(PROJECT_ROOT).as_posix()}")

    return pd.DataFrame(records)


def create_group_split(df):

    sgkf = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    df = df.copy()
    df["fold"] = -1

    for fold, (_, fold_idx) in enumerate(
        sgkf.split(
            X=df["filepath"],
            y=df["label"],
            groups=df["group_id"]
        )
    ):
        df.loc[fold_idx, "fold"] = fold

    df["split"] = "train"
    df.loc[df["fold"] == 8, "split"] = "val"
    df.loc[df["fold"] == 9, "split"] = "test"

    return df


def validate_split(df):

    print("\nDataset Summary")

    print(f"Total images: {len(df)}")
    print(f"Total groups: {df['group_id'].nunique()}")

    print("\nOverall class count")
    print(df["label"].value_counts())

    print("\nOverall class ratio")
    print(df["label"].value_counts(normalize=True))

    for split in ["train", "val", "test"]:

        split_df = df[df["split"] == split]

        print(f"\n[{split.upper()}]")
        print(f"Images: {len(split_df)}")
        print(f"Groups: {split_df['group_id'].nunique()}")

        print("Class count")
        print(split_df["label"].value_counts())

        print("Class ratio")
        print(split_df["label"].value_counts(normalize=True))

    train_groups = set(df[df["split"] == "train"]["group_id"])
    val_groups = set(df[df["split"] == "val"]["group_id"])
    test_groups = set(df[df["split"] == "test"]["group_id"])

    train_val_overlap = train_groups & val_groups
    train_test_overlap = train_groups & test_groups
    val_test_overlap = val_groups & test_groups

    print("\nLeakage Check")

    print("Train-Val overlap:", len(train_val_overlap))
    print("Train-Test overlap:", len(train_test_overlap))
    print("Val-Test overlap:", len(val_test_overlap))

    assert len(train_val_overlap) == 0
    assert len(train_test_overlap) == 0
    assert len(val_test_overlap) == 0

    print("Group leakage 없음.")


def main():

    print("Dataset scanning...")

    df = build_dataframe()

    print(f"Found {len(df)} images.")

    print("\nCreating group-level split...")

    df = create_group_split(df)

    validate_split(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_df = df[
        [
            "filepath",
            "label",
            "group_id",
            "original_split",
            "split"
        ]
    ]

    output_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()