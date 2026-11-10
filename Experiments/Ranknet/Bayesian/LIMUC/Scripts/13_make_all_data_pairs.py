"""Create N one-shot random comparisons from all N training images per fold."""

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from data_paths import add_data_root_argument, generated_path
from pair_acquisition import (
    apply_mayo_oracle,
    canonical_pair,
    image_appearance_counts,
    summarize_pairs,
)

METHOD = "all_data"
ROUND = 1
OUTPUT_FILES = {
    "selected_pairs_without_oracle.csv",
    "new_pairs.csv",
    "cumulative_pairs.csv",
    "new_anchors.csv",
    "pair_summary.csv",
    "image_appearance_counts.csv",
    "mayo_grade_distribution.csv",
    "experiment_config.json",
}


def make_random_pairs(image_ids, seed: int, fold: int) -> pd.DataFrame:
    """Select one uniform valid partner per anchor, without reading labels.

    Rejection sampling avoids an N-by-N candidate list. A complete pass is
    retried if a greedy choice exhausts a later anchor's partners, which can
    happen in tiny pools. Partners are not constrained to form a permutation.

    Args:
        image_ids: Distinct training image IDs, without labels or uncertainty.
        seed (int): Experiment seed.
        fold (int): Fold identity used to derive a separate random stream.

    Returns:
        pd.DataFrame: N comparisons with every image used once as an anchor.

    Raises:
        ValueError: If IDs are invalid or N distinct unordered pairs are impossible.
        RuntimeError: If 100 complete attempts cannot satisfy the constraints.
    """
    image_ids = list(image_ids)
    if any(pd.isna(image_id) for image_id in image_ids):
        raise ValueError("Training image IDs must not be missing.")
    image_ids = sorted(str(image_id) for image_id in image_ids)
    if any(not image_id.strip() for image_id in image_ids):
        raise ValueError("Training image IDs must not be empty.")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Training image IDs must be unique.")
    count = len(image_ids)
    if count < 3:
        raise ValueError(
            "At least three images are needed for N unique non-self pairs."
        )

    rng = random.Random(f"{seed}:{fold}")
    for _ in range(100):
        used = set()
        degrees = Counter()
        records = []
        for index, anchor in enumerate(image_ids):
            if degrees[anchor] == count - 1:
                break
            while True:
                partner_index = rng.randrange(count - 1)
                partner_index += partner_index >= index
                partner = image_ids[partner_index]
                key = canonical_pair(anchor, partner)
                if key not in used:
                    break
            used.add(key)
            degrees[anchor] += 1
            degrees[partner] += 1
            records.append((anchor, partner))
        if len(records) == count:
            return pd.DataFrame(records, columns=["x1_image", "x2_image"])
    raise RuntimeError("Could not create N unique pairs after 100 complete attempts.")


def file_digest(path: Path) -> str:
    """Identify a small input or output CSV without depending on its timestamp."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_directory(args, fold: int) -> Path:
    """Use a separate method and experiment, compatible with the existing trainer."""
    return (
        args.experiment_root
        / args.experiment_id
        / "rounds"
        / METHOD
        / f"seed_{args.seed}"
        / f"round_{ROUND}"
        / f"fold_{fold}"
    )


def write_fold(args, fold: int) -> None:
    """Validate one training split, generate pairs, then attach oracle labels.

    Args:
        args: CLI settings for input and isolated output directories.
        fold (int): Training fold to process.

    Raises:
        FileExistsError: If output exists without an explicit resume request.
        RuntimeError: If a resumed output is incomplete or differs from its inputs.
        ValueError: If training metadata or labels are invalid.
    """
    train_csv = args.training_dataset_dir / f"{fold}_train.csv"
    train = pd.read_csv(train_csv, dtype={"filename": str}, keep_default_na=False)
    if not {"filename", "MayoLabel"} <= set(train.columns):
        raise ValueError(f"{train_csv}: filename and MayoLabel columns are required.")
    train["MayoLabel"] = pd.to_numeric(train["MayoLabel"], errors="raise")
    if not train["MayoLabel"].isin([0, 1, 2, 3]).all():
        raise ValueError(f"{train_csv}: MayoLabel must contain only grades 0–3.")
    identity = {
        "experiment_id": args.experiment_id,
        "method": METHOD,
        "fold": fold,
        "seed": args.seed,
        "round": ROUND,
        "train_csv": str(train_csv),
        "train_csv_sha256": file_digest(train_csv),
        "pair_budget": len(train),
        "pairing_policy": "one_uniform_valid_partner_per_training_anchor_v1",
    }
    directory = output_directory(args, fold)
    manifest_path = directory / "selection_manifest.json"
    if directory.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists: {directory}. Use --resume.")
        if not manifest_path.is_file():
            raise RuntimeError(f"Incomplete all-data pair output: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or any(
            manifest.get(key) != value for key, value in identity.items()
        ):
            raise RuntimeError(f"All-data inputs or conditions differ: {directory}")
        hashes = manifest.get("output_sha256", {})
        if set(hashes) != OUTPUT_FILES or any(
            not (directory / name).is_file() or file_digest(directory / name) != digest
            for name, digest in hashes.items()
        ):
            raise RuntimeError(f"Incomplete or modified all-data output: {directory}")
        print(f"All-data pairs are already complete: {directory}")
        return

    start = time.perf_counter()
    pairs = make_random_pairs(train["filename"].tolist(), args.seed, fold)
    selection_seconds = time.perf_counter() - start
    pairs["round"] = ROUND
    pairs["method"] = METHOD
    labeled = apply_mayo_oracle(pairs, dict(zip(train["filename"], train["MayoLabel"])))
    summary = summarize_pairs(labeled)
    summary.update(
        new_pair_count=len(pairs), anchor_count=len(train), candidate_count=len(train)
    )
    counts = train["MayoLabel"].value_counts().reindex(range(4), fill_value=0)
    grades = pd.DataFrame(
        {
            "MayoLabel": counts.index,
            "unique_image_count": counts.values,
            "unique_image_fraction": counts.values / len(train),
        }
    )
    configuration = {
        **identity,
        "training_image_count": len(train),
        "nominal_labeling_percent": 100,
        "selection_seconds": selection_seconds,
        "candidate_scope": "full_train",
        "initial_pairs_appended": False,
        "partner_cap": 0,
        "initialization_policy": "not_set_by_pair_generator",
    }
    directory.mkdir(parents=True, exist_ok=False)
    tables = {
        "selected_pairs_without_oracle.csv": pairs,
        "new_pairs.csv": labeled,
        "cumulative_pairs.csv": labeled,
        "new_anchors.csv": pairs[["x1_image"]].rename(columns={"x1_image": "filename"}),
        "pair_summary.csv": pd.DataFrame([summary]),
        "image_appearance_counts.csv": image_appearance_counts(labeled),
        "mayo_grade_distribution.csv": grades,
    }
    for name, table in tables.items():
        table.to_csv(directory / name, index=False)
    (directory / "experiment_config.json").write_text(
        json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        **identity,
        "status": "complete",
        "output_sha256": {
            name: file_digest(directory / name)
            for name in [*tables, "experiment_config.json"]
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {len(pairs)} one-shot pairs from {len(train)} training images: {directory}"
    )


def parse_args(argv=None):
    """Parse a CPU-only one-shot pairing job; never modify shared AL_0 data."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_root_argument(parser)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=240121)
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--training-dataset-dir", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    folds = [value.strip() for value in args.folds.split(",")]
    if not folds or any(value not in {"1", "2", "3", "4", "5"} for value in folds):
        parser.error("--folds must contain fold numbers from 1 to 5.")
    args.folds = [int(value) for value in folds]
    if len(args.folds) != len(set(args.folds)):
        parser.error("--folds must not contain duplicates.")
    if (
        not args.experiment_id
        or Path(args.experiment_id).name != args.experiment_id
        or args.experiment_id in {".", ".."}
    ):
        parser.error("--experiment-id must be a single directory name.")
    args.training_dataset_dir = (
        (
            args.training_dataset_dir
            or generated_path(args.data_root, "training_dataset")
        )
        .expanduser()
        .resolve()
    )
    args.experiment_root = (
        (args.experiment_root or generated_path(args.data_root, "PairBALDExperiments"))
        .expanduser()
        .resolve()
    )
    return args


if __name__ == "__main__":
    arguments = parse_args()
    for fold_number in arguments.folds:
        write_fold(arguments, fold_number)
