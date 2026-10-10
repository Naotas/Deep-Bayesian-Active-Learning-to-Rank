"""Select comparison partners for RankNet active-learning experiments."""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from data_paths import add_data_root_argument, generated_path
from pair_acquisition import (
    SUPPORTED_METHODS,
    acquired_pair_keys,
    apply_mayo_oracle,
    build_candidate_rankings,
    image_appearance_counts,
    metrics_frame,
    select_ranked_pairs,
    selected_pairs_frame,
    summarize_pairs,
)

DEFAULT_SEED = 240119
DEFAULT_MC_SAMPLES = 30
DEFAULT_ANCHOR_RATE = 0.05


def read_csv(path, required_columns, description):
    """Read and validate one experiment CSV.

    Args:
        path (Path): CSV path.
        required_columns (set): Required column names.
        description (str): Human-readable input name.

    Returns:
        pd.DataFrame: Validated table.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    data = pd.read_csv(path)
    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return data


def select_anchor_images(
    train_ids,
    uncertainty_data,
    anchor_history,
    anchor_count,
    anchor_rate,
):
    """Apply the existing image-level uncertainty rule to new anchors.

    Args:
        train_ids: Fold-specific training image IDs.
        uncertainty_data (pd.DataFrame): Image IDs and ``var_score`` values.
        anchor_history (pd.DataFrame): Images selected as anchors in earlier rounds.
        anchor_count: Explicit selection count, or ``None`` for a rate.
        anchor_rate (float): Fraction of the full training split selected per round.

    Returns:
        pd.DataFrame: Newly selected anchors in deterministic acquisition order.
    """
    train_id_set = set(train_ids)
    uncertainty = uncertainty_data[["filename", "var_score"]].copy()
    uncertainty["filename"] = uncertainty["filename"].astype(str)
    uncertainty["var_score"] = uncertainty["var_score"].astype(float)
    if not np.all(np.isfinite(uncertainty["var_score"])):
        raise ValueError("Image-level uncertainty contains NaN or infinity.")
    if set(uncertainty["filename"]) != train_id_set:
        raise ValueError(
            "The uncertainty CSV image IDs do not exactly match the fold training CSV."
        )

    history_ids = set(anchor_history["filename"].astype(str))
    remaining = uncertainty[~uncertainty["filename"].isin(history_ids)].copy()
    requested = (
        int(anchor_count)
        if anchor_count is not None
        else max(1, round(len(train_ids) * anchor_rate))
    )
    if requested > len(remaining):
        raise RuntimeError(
            f"Requested {requested} new anchors, but only {len(remaining)} remain."
        )
    return (
        remaining.sort_values(["var_score", "filename"], ascending=[False, True])
        .head(requested)
        .reset_index(drop=True)
    )


def explicit_anchor_images(train_ids, anchor_data):
    """Validate an explicit diagnostic anchor set without reading labels."""
    anchors = anchor_data[["filename"]].copy()
    anchors["filename"] = anchors["filename"].astype(str)
    anchors = anchors.drop_duplicates("filename").reset_index(drop=True)
    outside_train = sorted(set(anchors["filename"]) - set(train_ids))
    if outside_train:
        raise ValueError(
            f"Explicit anchors are outside the training split: {outside_train}"
        )
    if anchors.empty:
        raise ValueError("Explicit anchor CSV is empty.")
    return anchors


def load_score_mapping(args, train_ids, cache_path):
    """Load or create shared-mask MC scores when the method requires them."""
    needs_scores = args.method in {"predictive_entropy", "pair_bald"}
    if not needs_scores and args.checkpoint is None:
        return None, 0.0, False
    if args.checkpoint is None:
        raise ValueError(f"--checkpoint is required for method={args.method}.")

    # TensorFlow is imported only after CUDA visibility is configured. Importing it
    # earlier can make a requested GPU selection ineffective.
    from shared_mc_dropout import create_or_load_cache

    image_ids, scores, seconds, reused = create_or_load_cache(
        cache_path=cache_path,
        image_ids=train_ids,
        image_dir=args.image_dir,
        checkpoint_path=args.checkpoint,
        fold=args.fold,
        mc_samples=args.mc_samples,
        seed=args.seed,
        image_batch_size=args.image_batch_size,
        allow_cpu_smoke_test=args.allow_cpu_smoke_test,
    )
    if list(image_ids.astype(str)) != list(train_ids):
        raise RuntimeError("Validated MC cache returned an unexpected image order.")
    return (
        {
            image_id: scores[:, index]
            for index, image_id in enumerate(image_ids.astype(str))
        },
        seconds,
        reused,
    )


def grade_distribution(pair_data):
    """Count Mayo grades among unique images appearing in acquired pairs."""
    labels = {}
    for image_column, label_column in [
        ("x1_image", "x1_label"),
        ("x2_image", "x2_label"),
    ]:
        for image_id, label in pair_data[[image_column, label_column]].itertuples(
            index=False, name=None
        ):
            image_id = str(image_id)
            label = float(label)
            if image_id in labels and labels[image_id] != label:
                raise ValueError(f"Conflicting oracle labels for image {image_id!r}.")
            labels[image_id] = label
    counts = Counter(labels.values())
    total = max(len(labels), 1)
    return pd.DataFrame(
        [
            {
                "MayoLabel": grade,
                "unique_image_count": counts.get(float(grade), 0),
                "unique_image_fraction": counts.get(float(grade), 0) / total,
            }
            for grade in range(4)
        ]
    )


def output_directory(args):
    """Return the isolated destination for one method, fold, seed, and round."""
    category = "diagnostics" if args.diagnostic else "rounds"
    return (
        args.experiment_root
        / args.experiment_id
        / category
        / args.method
        / f"seed_{args.seed}"
        / f"round_{args.round_number}"
        / f"fold_{args.fold}"
    )


def completed_run(output_dir, args):
    """Return whether ``--resume`` can safely treat the selection as complete."""
    manifest_path = output_dir / "selection_manifest.json"
    if not output_dir.exists():
        return False
    if not args.resume:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --resume only "
            "to verify and reuse a completed run."
        )
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Refusing to resume an incomplete output directory: {output_dir}"
        )
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    expected = {
        "experiment_id": args.experiment_id,
        "method": args.method,
        "seed": args.seed,
        "round": args.round_number,
        "fold": args.fold,
        "status": "complete",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Completed output manifest does not match this request.")
    print(f"Selection is already complete: {output_dir}")
    return True


def run_selection(args):
    """Run one isolated comparison-partner acquisition."""
    total_start = time.perf_counter()
    output_dir = output_directory(args)
    if completed_run(output_dir, args):
        return

    train_data = read_csv(
        args.train_csv,
        {"filename", "MayoLabel"},
        "Fold training CSV",
    )
    if train_data["filename"].astype(str).duplicated().any():
        raise ValueError("Fold training CSV contains duplicate image IDs.")
    train_ids = train_data["filename"].astype(str).tolist()

    previous_pairs = read_csv(
        args.previous_pair_csv,
        {"x1_image", "x2_image", "x1_label", "x2_label", "relative_label"},
        "Previous cumulative pair CSV",
    )
    anchor_history = read_csv(
        args.anchor_history_csv,
        {"filename"},
        "Anchor history CSV",
    )
    train_id_set = set(train_ids)
    history_outside_train = sorted(
        set(anchor_history["filename"].astype(str)) - train_id_set
    )
    if history_outside_train:
        raise ValueError(
            "Anchor history contains validation/test or unknown images: "
            f"{history_outside_train}"
        )
    previous_pair_ids = set(previous_pairs["x1_image"].astype(str)) | set(
        previous_pairs["x2_image"].astype(str)
    )
    previous_outside_train = sorted(previous_pair_ids - train_id_set)
    if previous_outside_train:
        raise ValueError(
            "Previous training pairs contain validation/test or unknown images: "
            f"{previous_outside_train}"
        )

    if args.anchor_csv is not None:
        anchor_data = read_csv(args.anchor_csv, {"filename"}, "Explicit anchor CSV")
        anchors = explicit_anchor_images(train_ids, anchor_data)
    else:
        uncertainty_data = read_csv(
            args.uncertainty_csv,
            {"filename", "var_score"},
            "Image uncertainty CSV",
        )
        anchors = select_anchor_images(
            train_ids=train_ids,
            uncertainty_data=uncertainty_data,
            anchor_history=anchor_history,
            anchor_count=args.anchor_count,
            anchor_rate=args.anchor_rate,
        )
    anchor_ids = anchors["filename"].astype(str).tolist()

    candidate_scope = args.candidate_scope
    if candidate_scope is None:
        candidate_scope = (
            "cumulative_selected" if args.method == "original" else "full_train"
        )
    if args.method == "original" and candidate_scope != "cumulative_selected":
        raise ValueError(
            "method=original requires --candidate-scope cumulative_selected."
        )
    if candidate_scope == "selected":
        candidate_ids = anchor_ids
    elif candidate_scope == "cumulative_selected":
        candidate_ids = list(
            dict.fromkeys(anchor_history["filename"].astype(str).tolist() + anchor_ids)
        )
    else:
        candidate_ids = train_ids

    pair_budget = args.pair_budget if args.pair_budget is not None else len(anchor_ids)
    if args.allow_cpu_smoke_test and len(train_ids) > args.cpu_smoke_max_images:
        raise RuntimeError(
            "CPU smoke mode is limited to "
            f"{args.cpu_smoke_max_images} training images, but {len(train_ids)} were supplied."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    cache_path = args.mc_cache or (
        args.experiment_root
        / args.experiment_id
        / "mc_caches"
        / args.method
        / f"seed_{args.seed}_round_{args.round_number}_fold_{args.fold}.npz"
    )
    score_mapping, prediction_seconds, cache_reused = load_score_mapping(
        args, train_ids, cache_path
    )

    acquisition_start = time.perf_counter()
    acquired = acquired_pair_keys(previous_pairs)
    rankings = build_candidate_rankings(
        anchors=anchor_ids,
        candidates=candidate_ids,
        method=args.method,
        acquired_pairs=acquired,
        score_by_image=score_mapping,
        seed=args.seed,
        chunk_size=args.candidate_chunk_size,
        negative_tolerance=args.negative_bald_tolerance,
    )
    selected = select_ranked_pairs(
        rankings=rankings,
        budget=pair_budget,
        acquired_pairs=acquired,
        previous_partner_counts=Counter(previous_pairs["x2_image"].astype(str)),
        partner_cap=args.partner_cap,
    )
    acquisition_seconds = time.perf_counter() - acquisition_start

    output_dir.mkdir(parents=True, exist_ok=False)
    anchors.assign(selected_round=args.round_number).to_csv(
        output_dir / "new_anchors.csv", index=False
    )
    cumulative_anchors = pd.concat(
        [
            anchor_history[["filename"]].assign(selected_round=np.nan),
            anchors[["filename"]].assign(selected_round=args.round_number),
        ],
        ignore_index=True,
    ).drop_duplicates("filename", keep="last")
    cumulative_anchors.to_csv(output_dir / "cumulative_anchors.csv", index=False)
    pd.DataFrame(
        {"filename": candidate_ids, "candidate_scope": candidate_scope}
    ).to_csv(output_dir / "candidate_scope.csv", index=False)
    metrics_frame(rankings, selected).to_csv(
        output_dir / "candidate_metrics.csv", index=False
    )

    unlabeled_pairs = selected_pairs_frame(selected)
    unlabeled_pairs.insert(2, "round", args.round_number)
    unlabeled_pairs.insert(3, "method", args.method)
    unlabeled_pairs.to_csv(
        output_dir / "selected_pairs_without_oracle.csv", index=False
    )

    oracle_start = time.perf_counter()
    labels_by_image = dict(
        zip(
            train_data["filename"].astype(str),
            train_data["MayoLabel"].astype(float),
        )
    )
    new_pairs = apply_mayo_oracle(unlabeled_pairs, labels_by_image)
    oracle_seconds = time.perf_counter() - oracle_start
    new_pairs.to_csv(output_dir / "new_pairs.csv", index=False)

    if not args.diagnostic:
        cumulative_pairs = pd.concat([previous_pairs, new_pairs], ignore_index=True)
        cumulative_pairs.to_csv(output_dir / "cumulative_pairs.csv", index=False)
        summary = summarize_pairs(cumulative_pairs)
        summary.update(
            {
                "new_pair_count": len(new_pairs),
                "anchor_count": len(anchor_ids),
                "candidate_count": len(candidate_ids),
            }
        )
        pd.DataFrame([summary]).to_csv(output_dir / "pair_summary.csv", index=False)
        image_appearance_counts(cumulative_pairs).to_csv(
            output_dir / "image_appearance_counts.csv", index=False
        )
        grade_distribution(cumulative_pairs).to_csv(
            output_dir / "mayo_grade_distribution.csv", index=False
        )

    total_seconds = time.perf_counter() - total_start
    configuration = {
        "experiment_id": args.experiment_id,
        "method": args.method,
        "fold": args.fold,
        "seed": args.seed,
        "round": args.round_number,
        "diagnostic": args.diagnostic,
        "train_csv": str(args.train_csv),
        "previous_pair_csv": str(args.previous_pair_csv),
        "anchor_history_csv": str(args.anchor_history_csv),
        "uncertainty_csv": (
            str(args.uncertainty_csv) if args.uncertainty_csv is not None else None
        ),
        "anchor_csv": str(args.anchor_csv) if args.anchor_csv is not None else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "mc_cache": str(cache_path),
        "mc_samples": args.mc_samples,
        "candidate_scope": candidate_scope,
        "candidate_chunk_size": args.candidate_chunk_size,
        "anchor_rate": args.anchor_rate,
        "anchor_count": len(anchor_ids),
        "pair_budget": pair_budget,
        "partner_cap": args.partner_cap,
        "prediction_seconds": prediction_seconds,
        "acquisition_seconds": acquisition_seconds,
        "oracle_seconds": oracle_seconds,
        "total_seconds": total_seconds,
        "cache_reused": cache_reused,
    }
    with (output_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as config_file:
        json.dump(configuration, config_file, indent=2, sort_keys=True)
        config_file.write("\n")
    pd.DataFrame([configuration]).to_csv(
        output_dir / "experiment_config.csv", index=False
    )

    manifest = {
        "experiment_id": args.experiment_id,
        "method": args.method,
        "seed": args.seed,
        "round": args.round_number,
        "fold": args.fold,
        "status": "complete",
    }
    with (output_dir / "selection_manifest.json").open(
        "w", encoding="utf-8"
    ) as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    print(f"Saved comparison-pair selection: {output_dir}")


def parse_args():
    """Parse command-line options for one pair-acquisition run."""
    parser = argparse.ArgumentParser(
        description="Select comparison partners using Random, entropy, or Pair-BALD."
    )
    add_data_root_argument(parser)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--method", required=True, choices=sorted(SUPPORTED_METHODS))
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--round", dest="round_number", required=True, type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--previous-pair-csv", type=Path, required=True)
    parser.add_argument("--anchor-history-csv", type=Path, required=True)
    anchor_source = parser.add_mutually_exclusive_group(required=True)
    anchor_source.add_argument("--uncertainty-csv", type=Path)
    anchor_source.add_argument("--anchor-csv", type=Path)
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--mc-cache", type=Path)
    parser.add_argument(
        "--candidate-scope",
        choices=["full_train", "selected", "cumulative_selected"],
        default=None,
    )
    parser.add_argument("--anchor-rate", type=float, default=DEFAULT_ANCHOR_RATE)
    parser.add_argument("--anchor-count", type=int)
    parser.add_argument("--pair-budget", type=int)
    parser.add_argument("--partner-cap", type=int, default=0)
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES)
    parser.add_argument("--candidate-chunk-size", type=int, default=2048)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--negative-bald-tolerance", type=float, default=1e-12)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu-smoke-test", action="store_true")
    parser.add_argument("--cpu-smoke-max-images", type=int, default=16)
    args = parser.parse_args()

    if args.round_number < 1:
        parser.error("--round must be at least 1.")
    if args.fold < 1:
        parser.error("--fold must be at least 1.")
    if not 0.0 < args.anchor_rate <= 1.0:
        parser.error("--anchor-rate must be within (0, 1].")
    if args.anchor_count is not None and args.anchor_count < 1:
        parser.error("--anchor-count must be at least 1.")
    if args.pair_budget is not None and args.pair_budget < 1:
        parser.error("--pair-budget must be at least 1.")
    if args.partner_cap < 0:
        parser.error("--partner-cap must be non-negative.")
    if args.mc_samples < 1:
        parser.error("--mc-samples must be at least 1.")

    args.train_csv = (
        args.train_csv.expanduser()
        if args.train_csv is not None
        else generated_path(
            args.data_root, "training_dataset", f"{args.fold}_train.csv"
        )
    )
    args.image_dir = (
        args.image_dir.expanduser()
        if args.image_dir is not None
        else generated_path(args.data_root, "all_public_UC_images")
    )
    args.experiment_root = (
        args.experiment_root.expanduser()
        if args.experiment_root is not None
        else generated_path(args.data_root, "PairBALDExperiments")
    )
    return args


if __name__ == "__main__":
    run_selection(parse_args())
