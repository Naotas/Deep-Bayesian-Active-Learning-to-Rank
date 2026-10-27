"""Plot saved five-fold pair-acquisition results without loading TensorFlow."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_paths import add_data_root_argument, generated_path

METHOD_COLORS = {
    "original": "#1f77b4",
    "random": "#d62728",
    "predictive_entropy": "#e6b800",
    "pair_bald": "#2ca02c",
}
FOLDS = (1, 2, 3, 4, 5)
METRICS = {"overall": "all_accuracy", "neighboring": "neighbor_accuracy"}


def read_table(path: Path, columns: set) -> pd.DataFrame:
    """Read a required CSV and reject incomplete column sets."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")
    data = pd.read_csv(path, dtype={"x1_image": str, "x2_image": str})
    missing = columns - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return data


def validate_numbers(data: pd.DataFrame, columns: set, description: str) -> None:
    """Reject nonnumeric, NaN, or infinite values before aggregation."""
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{description}: {column} contains NaN or infinity.")


def read_performance(path: Path, args, initial: bool = False) -> pd.DataFrame:
    """Require exactly one result per requested method, round, and fold.

    Args:
        path (Path): Saved evaluation CSV produced by script 10 or 11.
        args: Plotting arguments specifying one experiment and one seed.
        initial (bool): Read the shared initial-model rows with method ``initial``.

    Returns:
        pd.DataFrame: Complete, validated results for the requested experiment.

    Raises:
        ValueError: If results are missing, duplicated, or invalid.
    """
    columns = {
        "experiment_id",
        "method",
        "seed",
        "fold",
        "round",
        "cumulative_pair_count",
        *METRICS.values(),
    }
    data = read_table(path, columns)
    validate_numbers(data, {"seed", "fold", "round"}, str(path))
    methods = ["initial"] if initial else list(METHOD_COLORS)
    rounds = [0] if initial else list(range(1, args.max_round + 1))
    data = data.loc[
        (data["experiment_id"] == args.experiment_dir.name)
        & (data["seed"] == args.seed)
        & data["method"].isin(methods)
        & data["round"].isin(rounds)
    ].copy()
    identity = ["method", "round", "fold"]
    if data.duplicated(identity).any():
        raise ValueError(f"{path}: duplicate method/round/fold results.")
    expected = pd.MultiIndex.from_product([methods, rounds, FOLDS], names=identity)
    actual = pd.MultiIndex.from_frame(data[identity])
    missing = expected.difference(actual)
    if len(missing):
        raise ValueError(
            f"{path}: missing {len(missing)} results (method, round, fold): "
            f"{list(missing[:12])}. All five folds are required; no interpolation."
        )
    if not data["fold"].isin(FOLDS).all():
        raise ValueError(f"{path}: expected folds 1 through 5 only.")
    validate_numbers(data, {"cumulative_pair_count", *METRICS.values()}, str(path))
    budgets = data["cumulative_pair_count"]
    if ((budgets < 1) | (budgets % 1 != 0)).any():
        raise ValueError(f"{path}: pair counts must be positive integers.")
    for column in METRICS.values():
        if not data[column].between(0, 1).all():
            raise ValueError(f"{path}: {column} must be a fraction between 0 and 1.")
    if (data.groupby(["fold", "round"])["cumulative_pair_count"].nunique() != 1).any():
        raise ValueError(f"{path}: methods have different annotation budgets.")
    return data.sort_values(identity).reset_index(drop=True)


def initial_grade_distribution(pair_path: Path) -> tuple:
    """Count both pair endpoints once per image, as in the saved round CSVs.

    Args:
        pair_path (Path): A fold's saved initial training pair CSV.

    Returns:
        tuple: Four grade counts and the number of initial pairs.

    Raises:
        ValueError: If image IDs or oracle grades are invalid or inconsistent.
    """
    pairs = read_table(pair_path, {"x1_image", "x2_image", "x1_label", "x2_label"})
    images = pd.concat(
        [
            pairs[[f"x{side}_image", f"x{side}_label"]].rename(
                columns={f"x{side}_image": "image_id", f"x{side}_label": "MayoLabel"}
            )
            for side in (1, 2)
        ],
        ignore_index=True,
    )
    if images.empty or images["image_id"].isna().any():
        raise ValueError(
            f"{pair_path}: initial pairs contain no images or missing IDs."
        )
    if images["image_id"].str.strip().eq("").any():
        raise ValueError(f"{pair_path}: empty image IDs.")
    validate_numbers(images, {"MayoLabel"}, str(pair_path))
    if not images["MayoLabel"].isin(range(4)).all():
        raise ValueError(f"{pair_path}: Mayo grades must be 0, 1, 2, or 3.")
    if (images.groupby("image_id")["MayoLabel"].nunique() > 1).any():
        raise ValueError(f"{pair_path}: conflicting Mayo grades for the same image.")
    counts = (
        images.drop_duplicates("image_id")["MayoLabel"]
        .value_counts()
        .reindex(range(4), fill_value=0)
    )
    return counts, len(pairs)


def read_grade_distribution(path: Path) -> pd.Series:
    """Read four saved cumulative unique-image counts, including zero counts."""
    manifest_path = path.parent / "selection_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Selection is not complete: missing {manifest_path}")
    if (
        json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
        != "complete"
    ):
        raise ValueError(f"Selection is not complete: {manifest_path}")
    data = read_table(path, {"MayoLabel", "unique_image_count"})
    validate_numbers(data, {"MayoLabel", "unique_image_count"}, str(path))
    if len(data) != 4 or set(data["MayoLabel"]) != set(range(4)):
        raise ValueError(f"{path}: expected one row for each Mayo grade 0 through 3.")
    counts = data.set_index("MayoLabel")["unique_image_count"].reindex(range(4))
    if ((counts < 0) | (counts % 1 != 0)).any() or counts.sum() == 0:
        raise ValueError(
            f"{path}: unique-image counts must be nonnegative integers with a positive total."
        )
    return counts


def load_report_data(args) -> tuple:
    """Load only saved CSVs; never predict, train, or infer missing results."""
    performance = read_performance(args.performance_csv, args)
    initial = read_performance(args.initial_performance_csv, args, initial=True)
    shared_initial = pd.concat(
        [initial.assign(method=method) for method in METHOD_COLORS], ignore_index=True
    )
    performance = pd.concat([shared_initial, performance], ignore_index=True)
    for metric, source in METRICS.items():
        performance[f"{metric}_accuracy_percent"] = performance[source] * 100
    performance["nominal_labeling_percent"] = (
        args.initial_labeling_percent
        + args.round_labeling_percent * performance["round"]
    )
    if (
        performance.sort_values("round")
        .groupby(["method", "fold"])["cumulative_pair_count"]
        .diff()
        .dropna()
        <= 0
    ).any():
        raise ValueError("Cumulative pair budgets must increase at every round.")

    grade_rows = []
    for fold in FOLDS:
        pair_path = (
            args.initial_pairs_root / f"fold_{fold}/RBS/{fold}_train_pair_RBS.csv"
        )
        initial_counts, initial_budget = initial_grade_distribution(pair_path)
        saved_budget = initial.loc[
            initial["fold"] == fold, "cumulative_pair_count"
        ].iloc[0]
        if initial_budget != saved_budget:
            raise ValueError(
                f"{pair_path}: pair count differs from initial performance CSV."
            )
        for method in METHOD_COLORS:
            for round_number in args.bar_rounds:
                path = (
                    pair_path
                    if round_number == 0
                    else (
                        args.experiment_dir
                        / "rounds"
                        / method
                        / f"seed_{args.seed}"
                        / f"round_{round_number}"
                        / f"fold_{fold}"
                        / "mayo_grade_distribution.csv"
                    )
                )
                counts = (
                    initial_counts
                    if round_number == 0
                    else read_grade_distribution(path)
                )
                for grade, count in counts.items():
                    grade_rows.append(
                        {
                            "experiment_id": args.experiment_dir.name,
                            "seed": args.seed,
                            "method": method,
                            "fold": fold,
                            "round": round_number,
                            "MayoLabel": int(grade),
                            "nominal_labeling_percent": args.initial_labeling_percent
                            + args.round_labeling_percent * round_number,
                            "unique_image_count": int(count),
                            "unique_image_fraction": float(count / counts.sum()),
                            "source_csv": str(path),
                        }
                    )
    return performance, pd.DataFrame(grade_rows)


def summarize_performance(data: pd.DataFrame) -> pd.DataFrame:
    """Compute equally weighted fold means and sample SD (ddof=1), not SEM."""
    return (
        data.groupby(["method", "round", "nominal_labeling_percent"])
        .agg(
            fold_count=("fold", "nunique"),
            cumulative_pair_count_mean=("cumulative_pair_count", "mean"),
            cumulative_pair_count_min=("cumulative_pair_count", "min"),
            cumulative_pair_count_max=("cumulative_pair_count", "max"),
            overall_accuracy_mean_percent=("overall_accuracy_percent", "mean"),
            overall_accuracy_sd_percent=("overall_accuracy_percent", "std"),
            neighboring_accuracy_mean_percent=("neighboring_accuracy_percent", "mean"),
            neighboring_accuracy_sd_percent=("neighboring_accuracy_percent", "std"),
        )
        .reset_index()
    )


def summarize_grades(data: pd.DataFrame) -> pd.DataFrame:
    """Average per-fold unique-image counts and fractions, without pooling folds."""
    return (
        data.groupby(["method", "round", "nominal_labeling_percent", "MayoLabel"])
        .agg(
            fold_count=("fold", "nunique"),
            unique_image_count_mean=("unique_image_count", "mean"),
            unique_image_count_sd=("unique_image_count", "std"),
            unique_image_fraction_mean=("unique_image_fraction", "mean"),
            unique_image_fraction_sd=("unique_image_fraction", "std"),
        )
        .reset_index()
    )


def plot_accuracy(summary: pd.DataFrame, metric: str):
    """Draw mean accuracy and a translucent mean ± one sample-SD band."""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    lower_limits = []
    upper_limits = []
    for method, color in METHOD_COLORS.items():
        values = summary.loc[summary["method"] == method].sort_values("round")
        x = values["nominal_labeling_percent"].to_numpy(dtype=float)
        mean = values[f"{metric}_accuracy_mean_percent"].to_numpy(dtype=float)
        sd = values[f"{metric}_accuracy_sd_percent"].to_numpy(dtype=float)
        axis.plot(x, mean, marker="o", color=color, label=method, linewidth=1.8)
        axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0)
        lower_limits.extend(mean - sd)
        upper_limits.extend(mean + sd)
    axis.set_xticks(sorted(summary["nominal_labeling_percent"].unique()))
    axis.set_ylim(max(0, min(lower_limits) - 2), min(100, max(upper_limits) + 2))
    axis.set_xlabel("Labeling ratio (%)")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title(metric.capitalize())
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(fontsize=9)
    figure.tight_layout()
    return figure


def plot_grades(summary: pd.DataFrame, round_number: int):
    """Draw mean per-fold grade percentages on a shared 0–100% scale."""
    figure, axis = plt.subplots(figsize=(7, 4.5))
    width = 0.19
    x = np.arange(4)
    for index, (method, color) in enumerate(METHOD_COLORS.items()):
        values = summary.loc[
            (summary["method"] == method) & (summary["round"] == round_number)
        ].sort_values("MayoLabel")
        axis.bar(
            x + (index - 1.5) * width,
            values["unique_image_fraction_mean"] * 100,
            width=width,
            color=color,
            label=method,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(["Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3"])
    axis.set_xlabel("Mayo grade")
    axis.set_ylabel("Mean proportion of unique images (%)")
    axis.set_title(f"Iteration {round_number}")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(fontsize=9)
    figure.tight_layout()
    return figure


def save_report(args) -> None:
    """Validate all inputs before creating a new, non-overwriting report directory."""
    if args.output_dir.exists():
        raise FileExistsError(
            f"Report output already exists: {args.output_dir}. Choose a new directory."
        )
    performance, grades = load_report_data(args)
    performance_summary = summarize_performance(performance)
    grade_summary = summarize_grades(grades)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in (
        ("figure4_fold_data", performance),
        ("figure4_summary", performance_summary),
        ("figure6_fold_data", grades),
        ("figure6_summary", grade_summary),
    ):
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    for metric in METRICS:
        figure = plot_accuracy(performance_summary, metric)
        for extension in ("png", "pdf"):
            figure.savefig(args.output_dir / f"figure4_{metric}.{extension}", dpi=300)
        plt.close(figure)
    for round_number in args.bar_rounds:
        figure = plot_grades(grade_summary, round_number)
        for extension in ("png", "pdf"):
            figure.savefig(
                args.output_dir / f"figure6_iteration_{round_number}.{extension}",
                dpi=300,
            )
        plt.close(figure)
    metadata = {
        "arguments": vars(args),
        "folds": FOLDS,
        "method_colors": METHOD_COLORS,
        "accuracy_band": "unweighted five-fold mean +/- sample SD (ddof=1), percentage points; not a confidence interval",
        "labeling_ratio": "nominal initial_labeling_percent + round_labeling_percent * round; not unique-image coverage",
        "grade_population": "unique images in either endpoint of cumulative training pairs; each image counted once per fold",
        "grade_axis": "unweighted five-fold mean of within-fold grade fractions, multiplied by 100; not pooled counts",
        "grade_axis_limits_percent": [0, 100],
        "initial_model": "shared initial performance and training pairs copied across methods at round 0",
        "tie_policy": "same-grade pairs excluded from overall and neighboring directional accuracies by the existing evaluator",
        "paper_reference": "https://arxiv.org/html/2409.04952v1 (Sections 5.1.5 and 5.1.6; Figures 4 and 6)",
    }
    (args.output_dir / "plot_config.json").write_text(
        json.dumps(metadata, default=str, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved Figure 4/6 plots and tables: {args.output_dir}")


def parse_args(argv=None):
    """Parse saved-result paths and nominal labeling ratios, without GPU options."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_root_argument(parser)
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--performance-csv", type=Path)
    parser.add_argument("--initial-performance-csv", type=Path)
    parser.add_argument("--initial-pairs-root", type=Path)
    parser.add_argument("--seed", type=int, default=240119)
    parser.add_argument("--max-round", type=int, default=8)
    parser.add_argument(
        "--bar-rounds",
        default="0,3,6,8",
        help="Comma-separated iterations for Figure 6.",
    )
    parser.add_argument("--initial-labeling-percent", type=float, default=20)
    parser.add_argument("--round-labeling-percent", type=float, default=5)
    args = parser.parse_args(argv)
    parts = args.bar_rounds.split(",")
    if not all(part.strip().isdigit() for part in parts):
        parser.error("--bar-rounds must contain comma-separated nonnegative integers.")
    args.bar_rounds = [int(part.strip()) for part in parts]
    if (
        args.max_round < 1
        or len(set(args.bar_rounds)) != len(args.bar_rounds)
        or any(value > args.max_round for value in args.bar_rounds)
    ):
        parser.error(
            "Require --max-round >= 1 and unique --bar-rounds between 0 and --max-round."
        )
    percentages = [args.initial_labeling_percent, args.round_labeling_percent]
    if (
        not np.isfinite(percentages).all()
        or min(percentages) <= 0
        or args.initial_labeling_percent + args.max_round * args.round_labeling_percent
        > 100
    ):
        parser.error(
            "Nominal labeling percentages must be positive, finite, and at most 100% at the final round."
        )
    for name in (
        "data_root",
        "experiment_dir",
        "output_dir",
        "performance_csv",
        "initial_performance_csv",
        "initial_pairs_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    args.performance_csv = (
        args.performance_csv or args.experiment_dir / "performance.csv"
    )
    args.initial_performance_csv = (
        args.initial_performance_csv or args.experiment_dir / "initial_performance.csv"
    )
    args.initial_pairs_root = args.initial_pairs_root or generated_path(
        args.data_root, "Add_dataset", "LIMUC_AL_001", "AL_0"
    )
    return args


if __name__ == "__main__":
    save_report(parse_args())
