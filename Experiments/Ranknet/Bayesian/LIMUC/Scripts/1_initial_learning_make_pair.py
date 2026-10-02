import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from data_paths import add_data_root_argument, generated_path

# ============================================================
# Create the initial Active Learning dataset (AL_0)
#
# 1. Read the train / valid image lists from:
#      <data-root>/generated/training_dataset/
#
# 2. Randomly select 20% of images from each split.
#
# 3. Create one random non-self pair for each selected image.
#
# 4. Save:
#      - selected image CSVs
#      - pair CSVs for RankNet training
#
# Output:
# <data-root>/generated/Add_dataset/
#   LIMUC_AL_001/
#     AL_0/
#       fold_<fold>/
#         RBS/
#           <fold>_<datatype>_RBS.csv
#           <fold>_<datatype>_pair_RBS.csv
#
# In this implementation, AL_0 uses random sampling (RBS).
#
# Example:
#   python 1_initial_learning_make_pair.py
# ============================================================

SEED = 220428
INITIAL_RATE = 0.20
SELECTION = "RBS"
AL_ID = "LIMUC_AL_001"


def make_derangement(n, rng):
    """Return a shuffled index array with no self-pairs."""
    if n < 2:
        raise ValueError("At least two selected images are required to make pairs.")

    indices = list(range(n))

    while True:
        paired = indices.copy()
        rng.shuffle(paired)

        if all(i != j for i, j in zip(indices, paired)):
            return np.asarray(paired, dtype=int)


def select_initial_images(data, foldnum, datatype):
    """Randomly select 20% of images from one split."""
    select_num = round(len(data) * INITIAL_RATE)
    select_num = max(2, select_num)
    select_num = min(select_num, len(data))

    # Deterministic but different sampling for each fold / split.
    split_offset = 0 if datatype == "train" else 10000
    random_state = SEED + foldnum + split_offset

    selected = data.sample(
        n=select_num, replace=False, random_state=random_state
    ).copy()

    selected = selected.reset_index(drop=True)

    # Keep the same columns expected by later AL scripts.
    selected["selected_iteration"] = 0
    selected["selected_result_date"] = "AL_0"
    selected["selection"] = SELECTION

    return selected


def make_pair(selected, foldnum, datatype):
    """Create one self-pair-free random pair per selected image."""
    x1 = selected[["filename", "MayoLabel"]].reset_index(drop=True)

    rng_seed = SEED + foldnum + (0 if datatype == "train" else 10000)
    rng = random.Random(rng_seed)

    pair_idx = make_derangement(len(x1), rng)
    x2 = x1.iloc[pair_idx].reset_index(drop=True)

    x1_label = x1["MayoLabel"].to_numpy(dtype=float)
    x2_label = x2["MayoLabel"].to_numpy(dtype=float)

    pre_relative_label = x1_label - x2_label

    relative_label = copy.deepcopy(pre_relative_label)
    relative_label[pre_relative_label > 0] = 1.0
    relative_label[pre_relative_label == 0] = 0.5
    relative_label[pre_relative_label < 0] = 0.0

    pair_df = pd.DataFrame()
    pair_df["x1_image"] = x1["filename"].to_numpy()
    pair_df["x2_image"] = x2["filename"].to_numpy()
    pair_df["x1_label"] = x1_label
    pair_df["x2_label"] = x2_label
    pair_df["relative_label"] = relative_label

    return pair_df


def main():
    parser = argparse.ArgumentParser(
        description="Create the initial Active Learning image and pair datasets."
    )
    add_data_root_argument(parser)
    parser.add_argument(
        "--training-dataset-dir",
        type=Path,
        default=None,
        help="Input directory. Default: <data-root>/generated/training_dataset",
    )
    parser.add_argument(
        "--add-dataset-root",
        type=Path,
        default=None,
        help="Output directory. Default: <data-root>/generated/Add_dataset",
    )
    parser.add_argument(
        "--al-id",
        default=AL_ID,
        help=f"Active Learning experiment ID. Default: {AL_ID}",
    )
    parser.add_argument(
        "--folds",
        default="1,2,3,4,5",
        help="Comma-separated folds. Default: 1,2,3,4,5",
    )
    args = parser.parse_args()

    dataset_path = (
        args.training_dataset_dir.expanduser()
        if args.training_dataset_dir is not None
        else generated_path(args.data_root, "training_dataset")
    )
    add_dataset_root = (
        args.add_dataset_root.expanduser()
        if args.add_dataset_root is not None
        else generated_path(args.data_root, "Add_dataset")
    )
    folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]

    for foldnum in folds:
        print()
        print(f"fold {foldnum}")

        for datatype in ["train", "valid"]:
            print(datatype)

            input_csv = dataset_path / f"{foldnum}_{datatype}.csv"

            if not os.path.exists(input_csv):
                raise FileNotFoundError(f"Training dataset CSV not found: {input_csv}")

            data = pd.read_csv(input_csv)

            required = {"filename", "MayoLabel"}
            missing = required - set(data.columns)
            if missing:
                raise ValueError(
                    f"{input_csv} is missing required columns: {sorted(missing)}"
                )

            selected = select_initial_images(
                data=data, foldnum=foldnum, datatype=datatype
            )

            pair_df = make_pair(selected=selected, foldnum=foldnum, datatype=datatype)

            output_dir = os.path.join(
                add_dataset_root, args.al_id, "AL_0", f"fold_{foldnum}", SELECTION
            )
            os.makedirs(output_dir, exist_ok=True)

            selected_csv = os.path.join(
                output_dir, f"{foldnum}_{datatype}_{SELECTION}.csv"
            )

            pair_csv = os.path.join(
                output_dir, f"{foldnum}_{datatype}_pair_{SELECTION}.csv"
            )

            selected.to_csv(selected_csv, index=False)
            pair_df.to_csv(pair_csv, index=False)

            print(f"  full images     : {len(data)}")
            print(
                f"  selected AL_0   : {len(selected)} ({len(selected) / len(data):.3f})"
            )
            print(f"  pairs           : {len(pair_df)}")
            print(
                f"  self pairs      : "
                f"{int((pair_df['x1_image'] == pair_df['x2_image']).sum())}"
            )
            print(f"  selected csv    : {selected_csv}")
            print(f"  pair csv        : {pair_csv}")


if __name__ == "__main__":
    main()
