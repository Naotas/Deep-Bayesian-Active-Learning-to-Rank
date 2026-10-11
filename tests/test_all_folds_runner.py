"""Test fold scheduling and round dependencies using generated dummy artifacts."""

import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "Experiments/Ranknet/Bayesian/LIMUC/Scripts"
)


@pytest.fixture
def runner():
    """Import the orchestrator without loading TensorFlow or real data."""
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "all_folds_runner", SCRIPT_DIR / "11_run_all_folds.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def input_paths(tmp_path):
    """Create saved initial checkpoints and fold-specific CSVs for five dummy folds."""
    model_root = tmp_path / "initial_models"
    pair_root = tmp_path / "initial_pairs"
    uncertainty_root = tmp_path / "initial_uncertainty"
    for fold in range(1, 6):
        weights = model_root / f"fold_{fold}" / "weights"
        weights.mkdir(parents=True)
        (weights / "epoch0001-0.6000.h5").write_text("dummy old weights")
        (weights / "epoch0009-0.5000.h5").write_text("dummy best weights")
        pair_dir = pair_root / f"fold_{fold}" / "RBS"
        pair_dir.mkdir(parents=True)
        pair_csv = (
            "x1_image,x2_image,x1_label,x2_label,relative_label\n"
            f"dummy_{fold}_a.png,dummy_{fold}_b.png,0,1,0\n"
        )
        for split in ("train", "valid"):
            (pair_dir / f"{fold}_{split}_pair_RBS.csv").write_text(pair_csv)
        (pair_dir / f"{fold}_train_RBS.csv").write_text(
            f"filename\ndummy_{fold}_a.png\n"
        )
        uncertainty_dir = uncertainty_root / f"fold_{fold}"
        uncertainty_dir.mkdir(parents=True)
        (uncertainty_dir / "train_mean_score_and_uncertainty.csv").write_text(
            f"filename,var_score\ndummy_{fold}_b.png,0.3\n"
        )
    return model_root, pair_root, uncertainty_root


def arguments(runner, tmp_path, input_paths, *extra):
    """Construct a batch request with explicit paths to generated test inputs."""
    model_root, pair_root, uncertainty_root = input_paths
    return runner.parse_args(
        [
            "--experiment-id",
            "dummy",
            "--data-root",
            str(tmp_path / "data"),
            "--experiment-root",
            str(tmp_path / "experiments"),
            "--initial-model-root",
            str(model_root),
            "--initial-pairs-root",
            str(pair_root),
            "--initial-uncertainty-root",
            str(uncertainty_root),
            "--gpus",
            "0,1,2,3,4",
            *extra,
        ]
    )


def complete(directory, manifest):
    """Create a completion marker for a synthetic successful child process."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / manifest).write_text(json.dumps({"status": "complete"}))


def fake_child(script, options):
    """Produce the artifacts needed by the next stage without training a model."""
    if script == "7_select_comparison_pairs.py":
        directory = (
            Path(options["experiment-root"])
            / options["experiment-id"]
            / "rounds"
            / options["method"]
            / f"seed_{options['seed']}"
            / f"round_{options['round']}"
            / f"fold_{options['fold']}"
        )
        complete(directory, "selection_manifest.json")
        (directory / "cumulative_pairs.csv").write_text("dummy pairs")
        (directory / "cumulative_anchors.csv").write_text("dummy anchors")
    elif script == "8_train_pair_acquisition.py":
        directory = (
            Path(options["experiment-root"])
            / options["experiment-id"]
            / "training"
            / options["method"]
            / f"seed_{options['seed']}"
            / f"round_{options['round']}"
            / f"fold_{options['fold']}"
        )
        complete(directory, "training_manifest.json")
        (directory / "weights").mkdir()
        (directory / "weights/epoch0002-0.4000.h5").write_text("dummy round weights")
    elif script == "9_predict_pair_experiment.py":
        directory = Path(options["output-dir"])
        complete(directory, "prediction_manifest.json")
        for split in options["datatypes"].split(","):
            (directory / f"{split}_mean_score_and_uncertainty.csv").write_text(
                "dummy predictions"
            )
    else:
        path = Path(options["output-csv"])
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            name.replace("-", "_"): options[name]
            for name in ("experiment-id", "method", "fold", "seed", "round")
        }
        row.update(cumulative_pair_count=10 + options["round"], all_accuracy=0.75)
        output = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([output, pd.DataFrame([row])], ignore_index=True).to_csv(
            path, index=False
        )


def test_five_folds_start_concurrently_and_follow_their_own_models(
    runner, tmp_path, input_paths, monkeypatch
):
    """All folds should start together and preserve per-method round dependencies."""
    args = arguments(runner, tmp_path, input_paths, "--rounds", "2")
    barrier = threading.Barrier(5)
    calls = []
    lock = threading.Lock()
    active_gpus = set()
    first_calls = set()

    def execute(args, fold, gpu, label, script, options, log_dir):
        with lock:
            assert gpu not in active_gpus
            active_gpus.add(gpu)
            first = fold not in first_calls
            first_calls.add(fold)
            calls.append((fold, gpu, script, options.copy()))
        if first:
            barrier.wait(timeout=10)
        fake_child(script, options)
        with lock:
            active_gpus.remove(gpu)

    monkeypatch.setattr(runner, "run_command", execute)
    monkeypatch.setattr(runner, "check_gpus", lambda gpus: gpus)
    runner.run_experiment(args)

    assert len(first_calls) == 5
    assert len(calls) == 5 * 4 * 2 * 4
    for fold, gpu, script, options in calls:
        assert gpu == str(fold - 1)
        if script == "8_train_pair_acquisition.py":
            assert options["initial-checkpoint"] == (
                input_paths[0] / f"fold_{fold}/weights/epoch0009-0.5000.h5"
            )
            assert f"fold_{fold}" in str(options["valid-pair-csv"])
        if script == "7_select_comparison_pairs.py":
            if options["round"] == 1:
                assert options["mc-cache"] == (
                    args.experiment_dir / "shared_cache" / f"fold_{fold}_al0.npz"
                )
            else:
                assert options["checkpoint"] == (
                    runner.round_directory(args, "training", options["method"], fold, 1)
                    / "weights/epoch0002-0.4000.h5"
                )
                assert options["previous-pair-csv"] == (
                    runner.round_directory(args, "rounds", options["method"], fold, 1)
                    / "cumulative_pairs.csv"
                )
                assert options["uncertainty-csv"] == (
                    runner.round_directory(
                        args, "predictions", options["method"], fold, 1
                    )
                    / "train_mean_score_and_uncertainty.csv"
                )
    performance = pd.read_csv(args.experiment_dir / "performance.csv")
    assert len(performance) == 5 * 4 * 2
    assert not performance.duplicated(runner.IDENTITY_COLUMNS).any()
    for fold in range(1, 6):
        assert (args.experiment_dir / f"fold_performance/fold_{fold}.csv").is_file()


def test_initial_predictions_are_created_once_per_fold(
    runner, tmp_path, input_paths, monkeypatch
):
    """Missing initial uncertainty should trigger prediction, never initial training."""
    args = arguments(runner, tmp_path, input_paths, "--stage", "select")
    args.initial_uncertainty_root = None
    calls = []

    def execute(args, fold, gpu, label, script, options, log_dir):
        calls.append((script, options))
        fake_child(script, options)

    monkeypatch.setattr(runner, "run_command", execute)
    runner.run_fold(args, 1, "3", runner.initial_inputs(args)[1], None)

    assert [script for script, _ in calls] == [
        "9_predict_pair_experiment.py",
        *["7_select_comparison_pairs.py"] * 4,
    ]
    assert calls[0][1]["datatypes"] == "train"
    assert calls[0][1]["checkpoint"].parent.parent.name == "fold_1"


def test_dry_run_does_not_create_results_or_request_gpus(
    runner, tmp_path, input_paths, monkeypatch, capsys
):
    """Preview all commands without running GPU diagnostics or writing artifacts."""
    args = arguments(runner, tmp_path, input_paths, "--dry-run", "--rounds", "2")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("A dry run must not launch subprocesses."),
    )
    runner.run_experiment(args)

    output = capsys.readouterr().out
    assert "[fold 5, GPU 4" in output
    assert "<best-checkpoint>.h5" in output
    assert not args.experiment_dir.exists()


def test_incomplete_training_blocks_prediction(runner, tmp_path, input_paths):
    """A partially saved checkpoint must not be mistaken for a completed model."""
    args = arguments(runner, tmp_path, input_paths, "--stage", "predict")
    training = runner.round_directory(args, "training", "original", 1, 1)
    (training / "weights").mkdir(parents=True)
    (training / "weights/epoch0002-0.5000.h5").write_text("dummy partial weights")

    with pytest.raises(RuntimeError, match="not complete"):
        runner.run_fold(args, 1, "0", runner.initial_inputs(args)[1], None)


def test_failed_fold_stops_its_later_stages_and_other_folds_finish(
    runner, tmp_path, input_paths, monkeypatch
):
    """A child failure should stop its fold, preserve other folds, and fail the batch."""
    args = arguments(runner, tmp_path, input_paths, "--methods", "pair_bald")
    calls = []

    def execute(args, fold, gpu, label, script, options, log_dir):
        calls.append((fold, script))
        if fold == 1 and script == "8_train_pair_acquisition.py":
            raise RuntimeError("dummy training interruption")
        fake_child(script, options)

    monkeypatch.setattr(runner, "run_command", execute)
    monkeypatch.setattr(runner, "check_gpus", lambda gpus: gpus)
    with pytest.raises(RuntimeError, match="Some folds failed"):
        runner.run_experiment(args)

    assert (1, "9_predict_pair_experiment.py") not in calls
    assert (5, "10_evaluate_pair_experiment.py") in calls
    assert set(pd.read_csv(args.experiment_dir / "performance.csv")["fold"]) == {
        2,
        3,
        4,
        5,
    }


def test_explicit_checkpoint_csv_preserves_previously_selected_weights(
    runner, tmp_path, input_paths
):
    """An explicit mapping should override automatic last-best checkpoint discovery."""
    source = tmp_path / "checkpoints.csv"
    source.write_text(
        "fold,checkpoint\n"
        + "\n".join(
            f"{fold},initial_models/fold_{fold}/weights/epoch0001-0.6000.h5"
            for fold in range(1, 6)
        )
    )
    args = arguments(runner, tmp_path, input_paths)
    args.initial_model_root = None
    args.initial_checkpoints_csv = source

    inputs = runner.initial_inputs(args)
    assert all(
        row["checkpoint"].name == "epoch0001-0.6000.h5" for row in inputs.values()
    )


def test_missing_initial_fold_stops_before_launching_any_process(
    runner, tmp_path, input_paths, monkeypatch
):
    """All five folds should be validated before any worker writes results."""
    args = arguments(runner, tmp_path, input_paths)
    args.initial_model_root = tmp_path / "missing"
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("Missing inputs must be found before launch."),
    )

    with pytest.raises(FileNotFoundError):
        runner.run_experiment(args)
    assert not args.experiment_dir.exists()


def test_gpu_aliases_cannot_schedule_the_same_device_twice(runner, monkeypatch):
    """Index and UUID aliases for one GPU must not create overlapping workers."""
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout="0, GPU-dummy-a\n1, GPU-dummy-b\n"),
    )
    assert runner.check_gpus(["0", "1"]) == ["GPU-dummy-a", "GPU-dummy-b"]
    with pytest.raises(ValueError, match="same device twice"):
        runner.check_gpus(["0", "GPU-dummy-a"])
    with pytest.raises(RuntimeError, match="not available"):
        runner.check_gpus(["2"])


def test_latest_saved_improvement_is_selected_not_rounded_loss(runner, tmp_path):
    """Equal rounded validation losses should resolve to the last saved epoch."""
    (tmp_path / "epoch0009-0.5000.h5").write_text("dummy weights")
    (tmp_path / "epoch0010-0.5000.h5").write_text("dummy better weights")

    assert runner.best_checkpoint(tmp_path).name == "epoch0010-0.5000.h5"


def test_merge_is_idempotent_and_preserves_existing_manual_rows(runner, tmp_path):
    """Aggregation should update matching identities and retain other experiment rows."""
    existing = {
        "experiment_id": "dummy",
        "method": "random",
        "fold": 1,
        "seed": 5,
        "round": 1,
        "cumulative_pair_count": 10,
        "all_accuracy": 0.5,
    }
    pd.DataFrame([existing, {**existing, "fold": 5}]).to_csv(
        tmp_path / "performance.csv", index=False
    )
    (tmp_path / "fold_performance").mkdir()
    pd.DataFrame([{**existing, "all_accuracy": 0.9}]).to_csv(
        tmp_path / "fold_performance/fold_1.csv", index=False
    )
    runner.merge_performance(tmp_path)
    runner.merge_performance(tmp_path)

    result = pd.read_csv(tmp_path / "performance.csv")
    assert len(result) == 2
    assert result.loc[result["fold"] == 1, "all_accuracy"].item() == 0.9


def test_child_failure_reports_log_and_nonzero_exit(
    runner, tmp_path, input_paths, monkeypatch
):
    """The runner should retain child output and expose failures rather than continue."""
    args = arguments(runner, tmp_path, input_paths)

    def fail(command, stdout, **kwargs):
        stdout.write("dummy GPU failure")
        return subprocess.CompletedProcess(command, returncode=3)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="exit 3"):
        runner.run_command(
            args,
            1,
            "0",
            "dummy_train",
            "8_train_pair_acquisition.py",
            {"method": "random", "fold": 1, "round": 1},
            tmp_path,
        )

    assert (tmp_path / "fold_1_dummy_train.log").read_text() == "dummy GPU failure"


@pytest.mark.parametrize("stage", ["select", "train", "predict"])
def test_resume_is_forwarded_to_existing_stage_scripts(
    runner, tmp_path, input_paths, capsys, stage
):
    """Completed runs should use existing idempotent resume rather than redo budgets."""
    args = arguments(
        runner, tmp_path, input_paths, "--dry-run", "--resume", "--stage", stage
    )
    runner.run_experiment(args)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 20
    assert all(line.endswith("--resume") for line in lines)


def test_completed_selection_resume_keeps_pairs_and_budget(
    runner, tmp_path, input_paths
):
    """Run the actual completed-selection path without GPU inference or reacquisition."""
    args = arguments(runner, tmp_path, input_paths, "--resume")
    inputs = runner.initial_inputs(args)[1]
    options = {
        "experiment-id": args.experiment_id,
        "experiment-root": args.experiment_root,
        "method": "pair_bald",
        "fold": 1,
        "round": 1,
        "seed": args.seed,
        "checkpoint": inputs["checkpoint"],
        "previous-pair-csv": inputs["train_pairs"],
        "anchor-history-csv": inputs["anchors"],
        "uncertainty-csv": inputs["uncertainty"],
        "candidate-scope": "full_train",
        "mc-samples": 30,
        "partner-cap": 0,
        "anchor-rate": 0.05,
    }
    directory = runner.round_directory(args, "rounds", "pair_bald", 1, 1)
    complete(directory, "selection_manifest.json")
    manifest = {
        "experiment_id": "dummy",
        "method": "pair_bald",
        "fold": 1,
        "round": 1,
        "seed": args.seed,
        "status": "complete",
    }
    (directory / "selection_manifest.json").write_text(json.dumps(manifest))
    conditions = {key.replace("-", "_"): value for key, value in options.items()}
    (directory / "experiment_config.json").write_text(
        json.dumps(conditions, default=str)
    )
    pairs = directory / "cumulative_pairs.csv"
    pairs.write_text("x1_image,x2_image\ndummy_a,dummy_b\n")
    original = pairs.read_bytes()

    runner.run_command(
        args,
        1,
        "0",
        "completed_select",
        "7_select_comparison_pairs.py",
        options,
        tmp_path,
    )

    assert pairs.read_bytes() == original
    assert (
        "Selection is already complete"
        in (tmp_path / "fold_1_completed_select.log").read_text()
    )
    conditions["checkpoint"] = "/dummy/different_initial.h5"
    (directory / "experiment_config.json").write_text(
        json.dumps(conditions, default=str)
    )
    with pytest.raises(RuntimeError, match="saved checkpoint"):
        runner.run_command(
            args,
            1,
            "0",
            "mismatched_select",
            "7_select_comparison_pairs.py",
            options,
            tmp_path,
        )


def test_actual_evaluation_child_writes_fold_local_csv(runner, tmp_path, input_paths):
    """Evaluate generated scores through the real subprocess boundary."""
    args = arguments(runner, tmp_path, input_paths)
    scores = tmp_path / "dummy_predictions.csv"
    pairs = tmp_path / "dummy_pairs.csv"
    pd.DataFrame(
        [
            {
                "filename": f"dummy_{grade}_{index}.png",
                "label": grade,
                "mean_score": grade + index * 0.01,
            }
            for grade in range(4)
            for index in range(3)
        ]
    ).to_csv(scores, index=False)
    pairs.write_text("x1_image,x2_image\ndummy_0_0.png,dummy_1_0.png\n")
    output = tmp_path / "fold_performance/fold_1.csv"
    options = {
        "experiment-id": "dummy",
        "method": "pair_bald",
        "fold": 1,
        "seed": 5,
        "round": 1,
        "prediction-summary-csv": scores,
        "cumulative-pair-csv": pairs,
        "output-csv": output,
    }

    runner.run_command(
        args,
        1,
        "0",
        "dummy_evaluate",
        "10_evaluate_pair_experiment.py",
        options,
        tmp_path,
    )

    result = pd.read_csv(output)
    assert result["all_accuracy"].item() == 1.0
    assert result["cumulative_pair_count"].item() == 1


def test_two_gpu_lanes_queue_extra_folds_without_sharing_devices(
    runner, tmp_path, input_paths, monkeypatch
):
    """With fewer GPUs, folds must queue behind a single worker on each device."""
    args = arguments(
        runner,
        tmp_path,
        input_paths,
        "--gpus",
        "2,3",
        "--stage",
        "select",
        "--methods",
        "random",
    )
    calls = []
    active_gpus = set()
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def execute(args, fold, gpu, label, script, options, log_dir):
        with lock:
            assert gpu not in active_gpus
            active_gpus.add(gpu)
            calls.append((fold, gpu))
        if fold <= 2:
            barrier.wait(timeout=10)
        fake_child(script, options)
        with lock:
            active_gpus.remove(gpu)

    monkeypatch.setattr(runner, "run_command", execute)
    monkeypatch.setattr(runner, "check_gpus", lambda gpus: gpus)
    runner.run_experiment(args)

    assert sorted(calls) == [(1, "2"), (2, "3"), (3, "2"), (4, "3"), (5, "2")]


def test_second_batch_runner_cannot_enter_the_same_experiment(
    runner, tmp_path, input_paths, monkeypatch
):
    """A second runner must fail before creating logs or starting worker processes."""
    args = arguments(runner, tmp_path, input_paths)
    args.experiment_dir.mkdir(parents=True)
    monkeypatch.setattr(runner, "check_gpus", lambda gpus: gpus)
    with (args.experiment_dir / "all_folds.lock").open("a") as lock:
        runner.fcntl.flock(lock, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another all-fold runner"):
            runner.run_experiment(args)

    assert not (args.experiment_dir / "runner_logs").exists()


def test_resume_does_not_accept_an_interrupted_prediction(
    runner, tmp_path, input_paths
):
    """A partial prediction directory must never become a completed resumed step."""
    args = arguments(runner, tmp_path, input_paths, "--resume")
    output = tmp_path / "partial_predictions"
    output.mkdir()
    (output / "train_prediction.csv").write_text("dummy partial output")

    with pytest.raises(RuntimeError, match="not complete"):
        runner.validate_resume(
            args, "9_predict_pair_experiment.py", {"output-dir": output}
        )
    assert (output / "train_prediction.csv").read_text() == "dummy partial output"
