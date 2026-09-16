from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import imagehash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "splits" / "group_split.csv"

HAMMING_THRESHOLD = 4


def calculate_phash(filepath):
    with Image.open(filepath) as image:
        return imagehash.phash(image)


def hash_to_array(image_hash):
    return np.array(image_hash.hash, dtype=np.uint8).flatten()


def main():
    df = pd.read_csv(CSV_PATH)

    print(f"Calculating pHash for {len(df)} images...")

    hashes = []

    for idx, row in df.iterrows():
        filepath = PROJECT_ROOT / row["filepath"]

        image_hash = calculate_phash(filepath)
        hashes.append(hash_to_array(image_hash))

        if (idx + 1) % 500 == 0:
            print(f"{idx + 1}/{len(df)}")

    hashes = np.stack(hashes)

    print("\nSearching cross-split near duplicates...")

    split_pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    found = []

    for split_1, split_2 in split_pairs:
        indices_1 = np.where(df["split"].to_numpy() == split_1)[0]
        indices_2 = np.where(df["split"].to_numpy() == split_2)[0]

        hashes_1 = hashes[indices_1]
        hashes_2 = hashes[indices_2]

        print(f"Checking {split_1} <-> {split_2}...")

        for local_i, hash_1 in enumerate(hashes_1):
            distances = np.count_nonzero(
                hashes_2 != hash_1,
                axis=1
            )

            matches = np.where(
                distances <= HAMMING_THRESHOLD
            )[0]

            i = indices_1[local_i]

            for local_j in matches:
                j = indices_2[local_j]

                found.append({
                    "distance": int(distances[local_j]),
                    "split_1": split_1,
                    "filepath_1": df.iloc[i]["filepath"],
                    "group_id_1": df.iloc[i]["group_id"],
                    "split_2": split_2,
                    "filepath_2": df.iloc[j]["filepath"],
                    "group_id_2": df.iloc[j]["group_id"],
                })

    print("\nNear Duplicate Check")
    print(f"Hamming threshold: {HAMMING_THRESHOLD}")
    print(f"Cross-split candidate pairs: {len(found)}")

    if len(found) == 0:
        print("Cross-split near duplicate candidate 없음.")
        return

    result_df = pd.DataFrame(found)
    result_df = result_df.sort_values(
        ["distance", "split_1", "split_2"]
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "splits"
        / "near_duplicates.csv"
    )

    result_df.to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved: {output_path}")

    print("\nCandidate count by distance:")
    print(
        result_df["distance"]
        .value_counts()
        .sort_index()
    )

    print("\nTop candidates:")
    print(
        result_df.head(30).to_string(index=False)
    )


if __name__ == "__main__":
    main()