"""Check one-shot random pairing using synthetic IDs and labels only."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "Experiments/Ranknet/Bayesian/LIMUC/Scripts"
)


@pytest.fixture
def maker():
    """Load the CPU-only generator without any training imports."""
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "all_data_pairs", SCRIPT_DIR / "13_make_all_data_pairs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def saved_input(tmp_path, maker):
    """Create a small fold containing all four grades, with repeated grade zero."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for fold in range(1, 6):
        pd.DataFrame(
            {
                "filename": [f"dummy_{index:03}" for index in range(12)],
                "MayoLabel": [index % 4 for index in range(12)],
            }
        ).to_csv(dataset / f"{fold}_train.csv", index=False)
    return maker.parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--training-dataset-dir",
            str(dataset),
            "--experiment-root",
            str(tmp_path / "experiments"),
            "--experiment-id",
            "dummy_all_data",
        ]
    )


@pytest.mark.parametrize("size", [3, 4, 12, 500])
@pytest.mark.parametrize("seed", [1, 7, 240121])
def test_exact_n_pairs_cover_all_anchors_without_reverse_duplicates(maker, size, seed):
    """Generate N unordered-distinct edges without a quadratic candidate table."""
    image_ids = [f"dummy_{index:04}" for index in range(size)]
    pairs = maker.make_random_pairs(image_ids, seed, 1)
    assert len(pairs) == size
    assert pairs.x1_image.tolist() == image_ids
    assert set(pairs.x2_image) <= set(image_ids)
    assert (pairs.x1_image != pairs.x2_image).all()
    keys = {tuple(sorted(pair)) for pair in pairs.itertuples(index=False, name=None)}
    assert len(keys) == size


def test_pairing_is_reproducible_order_independent_and_not_a_permutation(maker):
    """Do not force each partner to appear once or use the input row order."""
    image_ids = [f"dummy_{index:03}" for index in range(50)]
    pairs = maker.make_random_pairs(image_ids, 240121, 1)
    pd.testing.assert_frame_equal(
        pairs, maker.make_random_pairs(image_ids[::-1], 240121, 1)
    )
    assert not pairs.equals(maker.make_random_pairs(image_ids, 240120, 1))
    assert not pairs.equals(maker.make_random_pairs(image_ids, 240121, 2))
    assert pairs.x2_image.nunique() < len(image_ids)


@pytest.mark.parametrize(
    "image_ids",
    [[], ["a"], ["a", "b"], ["a", "a", "b"], ["", "a", "b"], [None, "a", "b"]],
)
def test_invalid_or_infeasible_ids_are_rejected(maker, image_ids):
    """Never duplicate comparisons merely to obtain N pairs in a tiny pool."""
    with pytest.raises(ValueError):
        maker.make_random_pairs(image_ids, 1, 1)


def test_selection_receives_only_training_ids_before_oracle(
    maker, saved_input, monkeypatch
):
    """Mayo affects targets, not pair selection, and evaluation images never enter."""
    args = saved_input
    pd.DataFrame({"filename": ["validation_only"], "MayoLabel": [0]}).to_csv(
        args.training_dataset_dir / "1_valid.csv", index=False
    )
    original = maker.make_random_pairs
    received = []

    def record_selection(image_ids, seed, fold):
        received.append(list(image_ids))
        assert "validation_only" not in image_ids
        return original(image_ids, seed, fold)

    monkeypatch.setattr(maker, "make_random_pairs", record_selection)
    maker.write_fold(args, 1)
    first = pd.read_csv(maker.output_directory(args, 1) / "new_pairs.csv")
    train_path = args.training_dataset_dir / "1_train.csv"
    train = pd.read_csv(train_path)
    train["MayoLabel"] = 3 - train["MayoLabel"]
    train.to_csv(train_path, index=False)
    args.experiment_id = "dummy_other_labels"
    maker.write_fold(args, 1)
    second = pd.read_csv(maker.output_directory(args, 1) / "new_pairs.csv")
    pd.testing.assert_frame_equal(
        first[["x1_image", "x2_image"]], second[["x1_image", "x2_image"]]
    )
    assert first.relative_label.tolist() == pytest.approx(
        (1 - second.relative_label).tolist()
    )
    assert len(received) == 2


def test_csv_outputs_counts_labels_hashes_and_resume(maker, saved_input):
    """Persist light auditable outputs without copying images or checkpoints."""
    args = saved_input
    maker.write_fold(args, 1)
    directory = maker.output_directory(args, 1)
    pairs = pd.read_csv(directory / "cumulative_pairs.csv")
    assert len(pairs) == 12
    assert set(pairs.relative_label) <= {0, 0.5, 1}
    assert (pairs.loc[pairs.x1_label == pairs.x2_label, "relative_label"] == 0.5).all()
    assert set(pairs.method) == {"all_data"}
    counts = pd.read_csv(directory / "image_appearance_counts.csv")
    assert (counts.anchor_appearances == 1).all()
    assert counts.total_appearances.sum() == 24
    grades = pd.read_csv(directory / "mayo_grade_distribution.csv")
    assert grades.unique_image_count.tolist() == [3] * 4
    assert grades.unique_image_fraction.sum() == pytest.approx(1)
    summary = pd.read_csv(directory / "pair_summary.csv").iloc[0]
    assert summary.cumulative_pair_count == summary.unique_image_count == 12
    assert not (directory / "candidate_metrics.csv").exists()
    assert not list(directory.rglob("*.h5"))
    before = (directory / "cumulative_pairs.csv").read_bytes()
    args.resume = True
    maker.write_fold(args, 1)
    assert (directory / "cumulative_pairs.csv").read_bytes() == before
    config = json.loads((directory / "experiment_config.json").read_text())
    assert config["pair_budget"] == config["training_image_count"] == 12
    assert config["initial_pairs_appended"] is False


@pytest.mark.parametrize(
    "problem",
    [
        "no_resume",
        "source_changed",
        "output_changed",
        "missing_manifest",
        "missing_output",
    ],
)
def test_resume_rejects_different_or_incomplete_runs(maker, saved_input, problem):
    """Reject partial or changed files instead of silently adding another budget."""
    args = saved_input
    maker.write_fold(args, 1)
    directory = maker.output_directory(args, 1)
    args.resume = problem != "no_resume"
    if problem == "source_changed":
        source = args.training_dataset_dir / "1_train.csv"
        data = pd.read_csv(source)
        data.loc[0, "MayoLabel"] = 1
        data.to_csv(source, index=False)
    elif problem == "output_changed":
        (directory / "new_pairs.csv").write_text("dummy modified output\n")
    elif problem == "missing_manifest":
        manifest = directory / "selection_manifest.json"
        manifest.rename(directory / "saved_manifest.json")
    elif problem == "missing_output":
        pairs = directory / "new_pairs.csv"
        pairs.rename(directory / "saved_pairs.csv")
    with pytest.raises((RuntimeError, FileExistsError)):
        maker.write_fold(args, 1)


@pytest.mark.parametrize(
    "problem", ["duplicate_id", "missing_id", "invalid_label", "missing_label"]
)
def test_invalid_csv_does_not_create_fold_output(maker, saved_input, problem):
    """Reject malformed input before saving a partial selection."""
    source = saved_input.training_dataset_dir / "1_train.csv"
    data = pd.read_csv(source)
    if problem == "duplicate_id":
        data.loc[0, "filename"] = data.loc[1, "filename"]
    elif problem == "missing_id":
        data.loc[0, "filename"] = ""
    elif problem == "invalid_label":
        data.loc[0, "MayoLabel"] = 4
    else:
        data = data.drop(columns="MayoLabel")
    data.to_csv(source, index=False)
    with pytest.raises(ValueError):
        maker.write_fold(saved_input, 1)
    assert not maker.output_directory(saved_input, 1).exists()


def test_cli_creates_all_five_folds_without_tensorflow(saved_input):
    """Run the documented generation entry point in a CPU-only environment."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "13_make_all_data_pairs.py"),
            "--experiment-id",
            saved_input.experiment_id,
            "--training-dataset-dir",
            str(saved_input.training_dataset_dir),
            "--experiment-root",
            str(saved_input.experiment_root),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.count("Saved 12 one-shot pairs") == 5


def test_documented_training_pipeline_reuses_source_conditions(
    maker, saved_input, tmp_path, monkeypatch
):
    """Check documented commands with dummy checkpoints and mocked GPU subprocesses."""
    args = saved_input
    baseline = args.experiment_root / args.experiment_id
    source = tmp_path / "source_experiment"
    checkpoint = tmp_path / "dummy_initial.h5"
    checkpoint.write_text("dummy checkpoint, not model weights")
    validation = tmp_path / "dummy_validation.csv"
    validation.write_text("dummy validation, not real data")
    for fold in range(1, 6):
        maker.write_fold(args, fold)
        run_path = Path("random") / f"seed_{args.seed}" / "round_16" / f"fold_{fold}"
        train_dir = source / "training" / run_path
        pred_dir = source / "predictions" / run_path
        train_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        (train_dir / "training_manifest.json").write_text('{"status":"complete"}')
        (train_dir / "experimental_condition.json").write_text(
            json.dumps(
                {
                    "method": "random",
                    "seed": args.seed,
                    "fold": fold,
                    "round": 16,
                    "initial_checkpoint": str(checkpoint),
                    "valid_pair_csv": str(validation),
                    "batch_size": 8,
                    "epochs": 10 + fold,
                    "patience": 3,
                    "learning_rate": fold * 1e-5,
                }
            )
        )
        (pred_dir / "prediction_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "mc_samples": 7,
                }
            )
        )

    calls = []

    def fake_run(command, check):
        assert check is True
        script = command[1]
        values = command[2:]
        assert values.count("--resume") == int(
            script != "10_evaluate_pair_experiment.py"
        )
        values = [value for value in values if value != "--resume"]
        options = dict(zip(values[::2], values[1::2]))
        calls.append((script, options))
        fold = int(options["--fold"])
        run_path = Path("all_data") / f"seed_{args.seed}" / "round_1" / f"fold_{fold}"
        if script == "8_train_pair_acquisition.py":
            assert options["--initial-checkpoint"] == str(checkpoint)
            assert options["--valid-pair-csv"] == str(validation)
            assert float(options["--learning-rate"]) == fold * 1e-5
            assert int(options["--epochs"]) == 10 + fold
            assert options["--cuda-visible-devices"] == "GPU-dummy"
            assert len(pd.read_csv(options["--train-pair-csv"])) == 12
            weights = baseline / "training" / run_path / "weights"
            weights.mkdir(parents=True)
            (weights / "epoch0001-0.5000.h5").write_text("dummy saved checkpoint")
        elif script == "9_predict_pair_experiment.py":
            assert options["--mc-samples"] == "7"
            assert options["--datatypes"] == "test"
            assert options["--cuda-visible-devices"] == "GPU-dummy"
            assert Path(options["--checkpoint"]).is_file()
        else:
            assert script == "10_evaluate_pair_experiment.py"
            assert options["--method"] == "all_data"
            output = Path(options["--output-csv"])
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "experiment_id": baseline.name,
                        "method": "all_data",
                        "seed": args.seed,
                        "fold": fold,
                        "round": 1,
                        "cumulative_pair_count": 12,
                        "all_accuracy": 0.9,
                        "neighbor_accuracy": 0.8,
                    }
                ]
            ).to_csv(output, index=False)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.chdir(SCRIPT_DIR)
    monkeypatch.setattr(
        sys,
        "argv",
        ["-", str(source), str(baseline), "GPU-dummy", str(args.seed), "16"],
    )
    readme = (SCRIPT_DIR.parents[4] / "README.md").read_text()
    section = readme.split("### One-shot random all-data control", 1)[1]
    program = section.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    exec(compile(program, "README all-data workflow", "exec"), {})
    assert len(calls) == 15
    performance = pd.read_csv(baseline / "performance.csv")
    assert len(performance) == 5
    assert set(performance.fold) == {1, 2, 3, 4, 5}
    assert set(performance.method) == {"all_data"}


@pytest.mark.parametrize(
    "extra", [["--folds", "1,1"], ["--folds", "0"], ["--experiment-id", "../existing"]]
)
def test_invalid_cli_arguments(maker, tmp_path, extra):
    """Do not allow ambiguous folds or an output outside its named experiment."""
    with pytest.raises(SystemExit):
        maker.parse_args(
            ["--data-root", str(tmp_path), "--experiment-id", "dummy", *extra]
        )
