"""Check a fresh ten-percent AL_0 setup using generated image metadata."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "Experiments/Ranknet/Bayesian/LIMUC/Scripts"
)


def write_splits(root: Path, folds: tuple[int, ...] = (1, 2)) -> None:
    """Create small fold-specific train and validation CSVs without images."""
    directory = root / "generated" / "training_dataset"
    directory.mkdir(parents=True)
    for fold in folds:
        for split, count in (("train", 100), ("valid", 40)):
            pd.DataFrame(
                {
                    "filename": [f"dummy_{fold}_{split}_{i}.jpg" for i in range(count)],
                    "MayoLabel": [i % 4 for i in range(count)],
                }
            ).to_csv(directory / f"{fold}_{split}.csv", index=False)


def generate_initial_pairs(
    root: Path, rate: str, al_id: str
) -> subprocess.CompletedProcess:
    """Run the CPU-only initial pair generator as a user would."""
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "1_initial_learning_make_pair.py"),
            "--data-root",
            str(root),
            "--al-id",
            al_id,
            "--initial-rate",
            rate,
            "--folds",
            "1,2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def load_runner():
    """Load the all-fold runner without TensorFlow."""
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "ten_percent_runner", SCRIPT_DIR / "11_run_all_folds.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ten_percent_initial_pairs_and_eight_rounds_reach_nominal_fifty(tmp_path):
    """Each split uses 10%, then eight new 5% anchor groups reach 50%."""
    write_splits(tmp_path)
    result = generate_initial_pairs(tmp_path, "0.10", "LIMUC_AL_10pct")
    assert result.returncode == 0, result.stderr

    root = tmp_path / "generated/Add_dataset/LIMUC_AL_10pct/AL_0"
    for fold in (1, 2):
        for split, expected in (("train", 10), ("valid", 4)):
            directory = root / f"fold_{fold}/RBS"
            selected = pd.read_csv(directory / f"{fold}_{split}_RBS.csv")
            pairs = pd.read_csv(directory / f"{fold}_{split}_pair_RBS.csv")
            assert len(selected) == len(pairs) == expected
            assert set(pairs.x1_image) == set(selected.filename)
            assert set(pairs.x2_image) == set(selected.filename)
            assert not (pairs.x1_image == pairs.x2_image).any()

    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "ten_percent_selector", SCRIPT_DIR / "7_select_comparison_pairs.py"
    )
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)

    train = pd.read_csv(tmp_path / "generated/training_dataset/1_train.csv")
    selected = pd.read_csv(root / "fold_1/RBS/1_train_RBS.csv")
    history = selected[["filename"]].copy()
    uncertainty = train[["filename"]].assign(var_score=1.0)
    for _ in range(8):
        new = selector.select_anchor_images(
            train_ids=train.filename.tolist(),
            uncertainty_data=uncertainty,
            anchor_history=history,
            anchor_count=None,
            anchor_rate=0.05,
        )
        assert len(new) == 5
        history = pd.concat([history, new[["filename"]]], ignore_index=True)
    assert len(set(history.filename)) == 50


def test_ten_percent_inputs_are_checked_before_all_fold_run(tmp_path):
    """The runner rejects 20% pairs or an incompatible initial checkpoint."""
    write_splits(tmp_path)
    assert generate_initial_pairs(tmp_path, "0.10", "LIMUC_AL_10pct").returncode == 0
    model_root = tmp_path / "initial_models"
    pair_root = tmp_path / "generated/Add_dataset/LIMUC_AL_10pct/AL_0"
    for fold in (1, 2):
        directory = model_root / f"fold_{fold}"
        (directory / "weights").mkdir(parents=True)
        (directory / "weights/epoch0001-0.5000.h5").write_text("dummy weights")
        pd.DataFrame([{"initial_rate": 0.10}]).to_csv(
            directory / "experimental_condition.csv", index=False
        )

    runner = load_runner()
    args = runner.parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--experiment-id",
            "ten_percent_test",
            "--initial-model-root",
            str(model_root),
            "--initial-pairs-root",
            str(pair_root),
            "--initial-rate",
            "0.10",
            "--gpus",
            "0",
            "--folds",
            "1,2",
            "--rounds",
            "8",
            "--dry-run",
        ]
    )
    assert len(runner.initial_inputs(args)) == 2
    assert args.rounds == 8

    pd.DataFrame([{"initial_rate": 0.20}]).to_csv(
        model_root / "fold_2/experimental_condition.csv", index=False
    )
    with pytest.raises(ValueError, match="checkpoint records initial_rate"):
        runner.initial_inputs(args)


def test_existing_initial_pair_output_is_preserved(tmp_path):
    """A rerun fails before overwriting either fold's saved initial CSVs."""
    write_splits(tmp_path)
    assert generate_initial_pairs(tmp_path, "0.10", "LIMUC_AL_10pct").returncode == 0
    first = (
        tmp_path
        / "generated/Add_dataset/LIMUC_AL_10pct/AL_0/fold_1/RBS/1_train_RBS.csv"
    )
    before = first.read_bytes()
    result = generate_initial_pairs(tmp_path, "0.20", "LIMUC_AL_10pct")
    assert result.returncode != 0
    assert "Initial pair output already exists" in result.stderr
    assert first.read_bytes() == before


@pytest.mark.parametrize("rate", ["0", "1.1", "nan", "inf"])
def test_invalid_initial_rates_stop_without_outputs(tmp_path, rate):
    """Invalid rates never create initial pair directories."""
    write_splits(tmp_path)
    result = generate_initial_pairs(tmp_path, rate, "invalid_rate")
    assert result.returncode != 0
    assert not (tmp_path / "generated/Add_dataset/invalid_rate").exists()
