from pathlib import Path
import hashlib

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "splits" / "group_split.csv"


def calculate_sha256(filepath):
    sha256 = hashlib.sha256()

    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    df = pd.read_csv(CSV_PATH)

    hashes = []

    print(f"Checking {len(df)} images...")

    for idx, row in df.iterrows():
        filepath = PROJECT_ROOT / row["filepath"]

        file_hash = calculate_sha256(filepath)
        hashes.append(file_hash)

        if (idx + 1) % 500 == 0:
            print(f"{idx + 1}/{len(df)}")

    df["sha256"] = hashes

    duplicate_groups = df[
        df.duplicated("sha256", keep=False)
    ].sort_values("sha256")

    print("\nExact Duplicate Check")
    print("Total images:", len(df))
    print("Unique hashes:", df["sha256"].nunique())
    print("Duplicate image rows:", len(duplicate_groups))

    leakage_count = 0

    for file_hash, group in duplicate_groups.groupby("sha256"):
        splits = set(group["split"])

        if len(splits) > 1:
            leakage_count += 1

            print("\nCross-split duplicate found")
            print(
                group[
                    [
                        "filepath",
                        "label",
                        "group_id",
                        "split"
                    ]
                ].to_string(index=False)
            )

    print("\nCross-split duplicate hash groups:", leakage_count)

    if leakage_count == 0:
        print("Exact duplicate leakage 없음.")
    else:
        print("WARNING: Exact duplicate leakage 발견.")


if __name__ == "__main__":
    main()