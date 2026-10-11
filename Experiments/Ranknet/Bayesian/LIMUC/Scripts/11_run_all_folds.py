"""Run saved-initial-model experiments across folds on separate GPUs."""

import argparse
import csv
import fcntl
import json
import math
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
from data_paths import add_data_root_argument, generated_path

SCRIPT_DIR = Path(__file__).resolve().parent
METHODS = ("original", "random", "predictive_entropy", "pair_bald")
STAGES = ("select", "train", "predict", "evaluate")
CHECKPOINT_PATTERN = re.compile(r"epoch(\d+)-[-+\d.eE]+\.h5")
IDENTITY_COLUMNS = ["experiment_id", "method", "fold", "seed", "round"]


def comma_separated(value):
    """Parse a nonempty list without allowing duplicate entries."""
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Provide a nonempty list of unique values.")
    return values


def best_checkpoint(weights_dir, dry_run=False):
    """Find the last saved improvement from the existing save-best-only callback.

    Args:
        weights_dir (Path): Directory of epoch-named HDF5 weight files.
        dry_run (bool): Allow a placeholder for not-yet-trained round models.

    Returns:
        Path: Checkpoint with the highest saved epoch number.

    Raises:
        FileNotFoundError: If no matching checkpoint exists.
        ValueError: If the highest epoch has ambiguous weight files.
    """
    candidates = []
    for path in weights_dir.glob("epoch*.h5"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if path.is_file() and match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        if dry_run:
            return weights_dir / "<best-checkpoint>.h5"
        raise FileNotFoundError(f"No saved best checkpoint in {weights_dir}")
    last_epoch = max(epoch for epoch, _ in candidates)
    matches = [path for epoch, path in candidates if epoch == last_epoch]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous checkpoints at epoch {last_epoch}: {matches}")
    return matches[0].resolve()


def initial_inputs(args):
    """Resolve each fold's saved initial weights and fixed pair inputs."""
    checkpoints = {}
    if args.initial_checkpoints_csv is not None:
        with args.initial_checkpoints_csv.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if not {"fold", "checkpoint"} <= set(reader.fieldnames or []):
                raise ValueError(
                    "Initial checkpoint CSV needs fold,checkpoint columns."
                )
            for row in reader:
                fold = int(row["fold"])
                if fold in checkpoints:
                    raise ValueError(f"Duplicate fold {fold} in checkpoint CSV.")
                checkpoint = Path(row["checkpoint"]).expanduser()
                if not checkpoint.is_absolute():
                    checkpoint = args.initial_checkpoints_csv.parent / checkpoint
                checkpoints[fold] = checkpoint.resolve()
    else:
        checkpoints = {
            fold: best_checkpoint(args.initial_model_root / f"fold_{fold}" / "weights")
            for fold in args.folds
        }

    inputs = {}
    for fold in args.folds:
        if fold not in checkpoints:
            raise ValueError(f"No initial checkpoint specified for fold {fold}.")
        pair_dir = args.initial_pairs_root / f"fold_{fold}" / "RBS"
        inputs[fold] = {
            "checkpoint": checkpoints[fold],
            "train_pairs": pair_dir / f"{fold}_train_pair_RBS.csv",
            "valid_pairs": pair_dir / f"{fold}_valid_pair_RBS.csv",
            "anchors": pair_dir / f"{fold}_train_RBS.csv",
        }
        for name, path in inputs[fold].items():
            if not path.is_file():
                raise FileNotFoundError(f"Fold {fold} {name} not found: {path}")
        if args.initial_uncertainty_root is not None:
            uncertainty = (
                args.initial_uncertainty_root
                / f"fold_{fold}"
                / "train_mean_score_and_uncertainty.csv"
            )
            if not uncertainty.is_file():
                raise FileNotFoundError(
                    f"Fold {fold} initial uncertainty missing: {uncertainty}"
                )
            inputs[fold]["uncertainty"] = uncertainty
        else:
            inputs[fold]["uncertainty"] = (
                args.experiment_dir
                / "initial_predictions"
                / f"seed_{args.seed}"
                / f"fold_{fold}"
                / "train_mean_score_and_uncertainty.csv"
            )
    return inputs


def round_directory(args, category, method, fold, round_number):
    """Return a method-specific artifact directory shared with scripts 7–9."""
    return (
        args.experiment_dir
        / category
        / method
        / f"seed_{args.seed}"
        / f"round_{round_number}"
        / f"fold_{fold}"
    )


def require_completed(directory, manifest_name):
    """Reject missing or interrupted upstream runs before reading their outputs."""
    manifest_path = directory / manifest_name
    if not manifest_path.is_file():
        raise RuntimeError(f"Upstream run is not complete: {directory}")
    with manifest_path.open(encoding="utf-8") as source:
        if json.load(source).get("status") != "complete":
            raise RuntimeError(f"Upstream run is not complete: {directory}")


def validate_resume(args, script, options):
    """Verify saved conditions before asking an existing script to skip its output."""
    if script == "10_evaluate_pair_experiment.py":
        return
    if script == "9_predict_pair_experiment.py":
        directory = Path(options["output-dir"])
        manifest = "prediction_manifest.json"
        conditions_file = manifest
        keys = ["checkpoint", "fold", "seed", "mc_samples"]
    else:
        category = "rounds" if script == "7_select_comparison_pairs.py" else "training"
        directory = round_directory(
            args, category, options["method"], options["fold"], options["round"]
        )
        keys = ["experiment_id", "method", "fold", "round", "seed"]
        if category == "rounds":
            manifest = "selection_manifest.json"
            conditions_file = "experiment_config.json"
            keys += [
                "checkpoint",
                "previous_pair_csv",
                "anchor_history_csv",
                "uncertainty_csv",
                "candidate_scope",
                "mc_samples",
                "partner_cap",
                "anchor_rate",
            ]
            keys += [
                key
                for key in ("pair_budget", "anchor_count")
                if options.get(key.replace("_", "-")) is not None
            ]
        else:
            manifest = "training_manifest.json"
            conditions_file = "experimental_condition.json"
            keys += [
                "initial_checkpoint",
                "train_pair_csv",
                "valid_pair_csv",
                "batch_size",
                "epochs",
                "patience",
                "learning_rate",
            ]
    if not directory.exists():
        return
    if not args.resume:
        raise FileExistsError(
            f"Output already exists: {directory}. Use --resume for completed matching runs."
        )
    require_completed(directory, manifest)
    with (directory / conditions_file).open(encoding="utf-8") as source:
        conditions = json.load(source)
    for key in keys:
        expected = options[key.replace("_", "-")]
        actual = conditions.get(key)
        if isinstance(expected, Path):
            matches = (
                actual is not None
                and Path(actual).expanduser().resolve() == expected.resolve()
            )
        else:
            matches = actual == expected
        if not matches:
            raise RuntimeError(
                f"Cannot reuse {directory}: saved {key}={actual!r}, requested {expected!r}."
            )
    if script == "9_predict_pair_experiment.py":
        requested = set(options["datatypes"].split(","))
        if requested != set(conditions.get("datatypes", [])):
            raise RuntimeError(
                f"Prediction splits do not match completed output: {directory}"
            )


def run_command(args, fold, gpu, label, script, options, log_dir):
    """Execute one existing script in a fresh process on its assigned GPU.

    Args:
        args: Parsed experiment options.
        fold (int): Fold being processed.
        gpu (str): Assigned GPU index or UUID.
        label (str): Unique step label within the fold.
        script (str): Existing pipeline script filename.
        options (dict): CLI flag names mapped to values.
        log_dir (Path): Unique invocation log directory, or None for a dry run.

    Raises:
        RuntimeError: If the child process exits unsuccessfully.
    """
    command = [sys.executable, str(SCRIPT_DIR / script)]
    for flag, value in options.items():
        if value is True:
            command.append(f"--{flag}")
        elif value is not None and value is not False:
            command.extend([f"--{flag}", str(value)])
    if args.resume and script != "10_evaluate_pair_experiment.py":
        command.append("--resume")
    if args.dry_run:
        print(f"[fold {fold}, GPU {gpu}, {label}] {shlex.join(command)}", flush=True)
        return
    validate_resume(args, script, options)
    log_path = log_dir / f"fold_{fold}_{label}.log"
    print(f"[fold {fold}, GPU {gpu}] {label}: {log_path}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False
        )
    if result.returncode:
        raise RuntimeError(
            f"Fold {fold} {label} failed (exit {result.returncode}); see {log_path}"
        )


def run_fold(args, fold, gpu, inputs, log_dir):
    """Process methods sequentially while retaining each fold's GPU assignment."""
    stages = STAGES if args.stage == "all" else (args.stage,)
    if (
        "select" in stages
        and args.start_round == 1
        and args.initial_uncertainty_root is None
    ):
        run_command(
            args,
            fold,
            gpu,
            "initial_predict",
            "9_predict_pair_experiment.py",
            {
                "data-root": args.data_root,
                "checkpoint": inputs["checkpoint"],
                "output-dir": inputs["uncertainty"].parent,
                "fold": fold,
                "seed": args.seed,
                "datatypes": "train",
                "mc-samples": args.mc_samples,
                "cuda-visible-devices": gpu,
            },
            log_dir,
        )

    for round_number in range(args.start_round, args.rounds + 1):
        for method in args.methods:
            selection = round_directory(args, "rounds", method, fold, round_number)
            training = round_directory(args, "training", method, fold, round_number)
            prediction = round_directory(
                args, "predictions", method, fold, round_number
            )
            shared = {
                "experiment-id": args.experiment_id,
                "method": method,
                "fold": fold,
                "round": round_number,
                "seed": args.seed,
            }
            gpu_options = {
                **shared,
                "data-root": args.data_root,
                "experiment-root": args.experiment_root,
                "cuda-visible-devices": gpu,
            }
            if "select" in stages:
                if round_number == 1:
                    checkpoint = inputs["checkpoint"]
                    previous_pairs = inputs["train_pairs"]
                    history = inputs["anchors"]
                    uncertainty = inputs["uncertainty"]
                    cache = (
                        args.experiment_dir / "shared_cache" / f"fold_{fold}_al0.npz"
                    )
                else:
                    previous_selection = round_directory(
                        args, "rounds", method, fold, round_number - 1
                    )
                    previous_training = round_directory(
                        args, "training", method, fold, round_number - 1
                    )
                    previous_prediction = round_directory(
                        args, "predictions", method, fold, round_number - 1
                    )
                    if not args.dry_run:
                        require_completed(previous_selection, "selection_manifest.json")
                        require_completed(previous_training, "training_manifest.json")
                        require_completed(
                            previous_prediction, "prediction_manifest.json"
                        )
                    checkpoint = best_checkpoint(
                        previous_training / "weights", args.dry_run
                    )
                    previous_pairs = previous_selection / "cumulative_pairs.csv"
                    history = previous_selection / "cumulative_anchors.csv"
                    uncertainty = (
                        previous_prediction / "train_mean_score_and_uncertainty.csv"
                    )
                    cache = (
                        args.experiment_dir
                        / "mc_caches"
                        / method
                        / f"seed_{args.seed}_round_{round_number}_fold_{fold}.npz"
                    )
                run_command(
                    args,
                    fold,
                    gpu,
                    f"{method}_round_{round_number}_select",
                    "7_select_comparison_pairs.py",
                    {
                        **gpu_options,
                        "candidate-scope": "cumulative_selected"
                        if method == "original"
                        else "full_train",
                        "checkpoint": checkpoint,
                        "mc-cache": cache,
                        "previous-pair-csv": previous_pairs,
                        "anchor-history-csv": history,
                        "uncertainty-csv": uncertainty,
                        "anchor-rate": args.anchor_rate,
                        "anchor-count": args.anchor_count,
                        "pair-budget": args.pair_budget,
                        "partner-cap": args.partner_cap,
                        "mc-samples": args.mc_samples,
                        "image-batch-size": args.image_batch_size,
                        "candidate-chunk-size": args.candidate_chunk_size,
                    },
                    log_dir,
                )
            if "train" in stages:
                if not args.dry_run:
                    require_completed(selection, "selection_manifest.json")
                run_command(
                    args,
                    fold,
                    gpu,
                    f"{method}_round_{round_number}_train",
                    "8_train_pair_acquisition.py",
                    {
                        **gpu_options,
                        "train-pair-csv": selection / "cumulative_pairs.csv",
                        "valid-pair-csv": inputs["valid_pairs"],
                        "initial-checkpoint": inputs["checkpoint"],
                        "batch-size": args.batch_size,
                        "epochs": args.epochs,
                        "patience": args.patience,
                        "learning-rate": args.learning_rate,
                    },
                    log_dir,
                )
            if "predict" in stages:
                if not args.dry_run:
                    require_completed(training, "training_manifest.json")
                run_command(
                    args,
                    fold,
                    gpu,
                    f"{method}_round_{round_number}_predict",
                    "9_predict_pair_experiment.py",
                    {
                        "data-root": args.data_root,
                        "checkpoint": best_checkpoint(
                            training / "weights", args.dry_run
                        ),
                        "output-dir": prediction,
                        "fold": fold,
                        "seed": args.seed,
                        "datatypes": "train,test",
                        "mc-samples": args.mc_samples,
                        "cuda-visible-devices": gpu,
                    },
                    log_dir,
                )
            if "evaluate" in stages:
                if not args.dry_run:
                    require_completed(selection, "selection_manifest.json")
                    require_completed(prediction, "prediction_manifest.json")
                run_command(
                    args,
                    fold,
                    gpu,
                    f"{method}_round_{round_number}_evaluate",
                    "10_evaluate_pair_experiment.py",
                    {
                        **shared,
                        "prediction-summary-csv": prediction
                        / "test_mean_score_and_uncertainty.csv",
                        "cumulative-pair-csv": selection / "cumulative_pairs.csv",
                        "output-csv": args.experiment_dir
                        / "fold_performance"
                        / f"fold_{fold}.csv",
                    },
                    log_dir,
                )


def merge_performance(experiment_dir):
    """Merge fold-local evaluation rows without concurrent shared CSV writes."""
    paths = sorted((experiment_dir / "fold_performance").glob("fold_*.csv"))
    if not paths:
        return
    destination = experiment_dir / "performance.csv"
    tables = [pd.read_csv(path) for path in paths]
    if destination.is_file():
        tables.insert(0, pd.read_csv(destination))
    output = pd.concat(tables, ignore_index=True)
    output = output.drop_duplicates(IDENTITY_COLUMNS, keep="last").sort_values(
        ["method", "seed", "fold", "cumulative_pair_count"]
    )
    temporary = destination.with_suffix(f".{uuid4().hex}.tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(destination)


def check_gpus(gpus):
    """Resolve GPU indexes to unique UUIDs before starting GPU subprocesses."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    available = {}
    for line in result.stdout.splitlines():
        index, uuid = [part.strip() for part in line.split(",")]
        available[index] = uuid
        available[uuid] = uuid
    missing = set(gpus) - set(available)
    if missing:
        raise RuntimeError(f"Requested GPUs are not available: {sorted(missing)}")
    if len({available[gpu] for gpu in gpus}) != len(gpus):
        raise ValueError(
            "GPU indexes and UUIDs must not refer to the same device twice."
        )
    return [available[gpu] for gpu in gpus]


def run_experiment(args):
    """Schedule one sequential worker per GPU and report failures after workers exit."""
    inputs = initial_inputs(args)
    if args.dry_run:
        for index, fold in enumerate(args.folds):
            run_fold(args, fold, args.gpus[index % len(args.gpus)], inputs[fold], None)
        return
    gpu_ids = args.gpus if args.stage == "evaluate" else check_gpus(args.gpus)
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    with (args.experiment_dir / "all_folds.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another all-fold runner is active for this experiment."
            ) from error
        log_dir = (
            args.experiment_dir
            / "runner_logs"
            / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
        )
        log_dir.mkdir(parents=True, exist_ok=False)
        config = {
            "arguments": vars(args),
            "initial_inputs": inputs,
            "gpu_assignment": {
                fold: gpu_ids[index % len(gpu_ids)]
                for index, fold in enumerate(args.folds)
            },
        }
        with (log_dir / "run_config.json").open("x", encoding="utf-8") as destination:
            json.dump(config, destination, default=str, indent=2)

        def run_lane(gpu, folds):
            errors = []
            for fold in folds:
                try:
                    run_fold(args, fold, gpu, inputs[fold], log_dir)
                except (RuntimeError, OSError, ValueError) as error:
                    errors.append(f"Fold {fold}: {error}")
                    print(errors[-1], file=sys.stderr, flush=True)
            return errors

        with ThreadPoolExecutor(
            max_workers=min(len(gpu_ids), len(args.folds))
        ) as workers:
            futures = [
                workers.submit(run_lane, gpu, args.folds[index :: len(gpu_ids)])
                for index, gpu in enumerate(gpu_ids[: len(args.folds)])
            ]
            errors = [error for future in futures for error in future.result()]
        if args.stage in {"all", "evaluate"}:
            merge_performance(args.experiment_dir)
        if errors:
            raise RuntimeError("Some folds failed:\n" + "\n".join(errors))
        print(f"All requested folds completed. Logs: {log_dir}")


def parse_args(argv=None):
    """Parse batch execution settings without importing TensorFlow."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_root_argument(parser)
    parser.add_argument("--experiment-id", required=True)
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument("--initial-model-root", type=Path)
    initial.add_argument("--initial-checkpoints-csv", type=Path)
    parser.add_argument("--initial-pairs-root", type=Path)
    parser.add_argument("--initial-uncertainty-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--gpus", required=True, type=comma_separated)
    parser.add_argument(
        "--folds", type=comma_separated, default=["1", "2", "3", "4", "5"]
    )
    parser.add_argument("--methods", type=comma_separated, default=list(METHODS))
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument(
        "--rounds", type=int, default=1, help="Final round number (inclusive)."
    )
    parser.add_argument("--seed", type=int, default=240119)
    parser.add_argument("--anchor-rate", type=float, default=0.05)
    parser.add_argument("--anchor-count", type=int)
    parser.add_argument("--pair-budget", type=int)
    parser.add_argument("--partner-cap", type=int, default=0)
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--candidate-chunk-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.gpus = [str(int(gpu)) if gpu.isdigit() else gpu for gpu in args.gpus]
    if len(args.gpus) != len(set(args.gpus)):
        parser.error("--gpus contains duplicate numeric values.")
    if not all(value.isdigit() and 1 <= int(value) <= 5 for value in args.folds):
        parser.error("--folds must contain unique fold numbers from 1 to 5.")
    args.folds = [int(value) for value in args.folds]
    if len(set(args.folds)) != len(args.folds):
        parser.error("--folds contains duplicate numeric values.")
    if not set(args.methods) <= set(METHODS):
        parser.error(f"--methods must be chosen from {METHODS}.")
    if not 1 <= args.start_round <= args.rounds:
        parser.error("Require 1 <= --start-round <= --rounds.")
    if not 0 < args.anchor_rate <= 1 or args.partner_cap < 0 or args.patience < 0:
        parser.error("Invalid acquisition rate, partner cap, or patience.")
    for name in (
        "mc_samples",
        "image_batch_size",
        "candidate_chunk_size",
        "batch_size",
        "epochs",
        "anchor_count",
        "pair_budget",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if (
        not args.experiment_id
        or Path(args.experiment_id).name != args.experiment_id
        or args.experiment_id
        in {
            ".",
            "..",
        }
    ):
        parser.error("--experiment-id must be one directory name.")
    for name in (
        "data_root",
        "initial_model_root",
        "initial_checkpoints_csv",
        "initial_pairs_root",
        "initial_uncertainty_root",
        "experiment_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.initial_pairs_root is None:
        args.initial_pairs_root = generated_path(
            args.data_root, "Add_dataset", "LIMUC_AL_001", "AL_0"
        )
    if args.experiment_root is None:
        args.experiment_root = generated_path(args.data_root, "PairBALDExperiments")
    args.experiment_dir = args.experiment_root / args.experiment_id
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
