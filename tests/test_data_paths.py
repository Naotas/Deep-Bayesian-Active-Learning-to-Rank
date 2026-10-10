"""Tests for consistent external LIMUC data paths."""

import argparse
import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DEFAULT_DATA_ROOT = Path("/data/umeiro0/patient_based_classified_images")
PATH_MODULES = [
    PROJECT_ROOT / "Data/UC/LIMUC/Preprocessing/data_paths.py",
    PROJECT_ROOT / "Experiments/Ranknet/Bayesian/LIMUC/Scripts/data_paths.py",
]
PIPELINE_SCRIPTS = [
    PROJECT_ROOT / f"Data/UC/LIMUC/Preprocessing/{step}"
    for step in [
        "1_prepare_uc_image_dataset.py",
        "2_create_patient_level_splits.py",
        "3_format_fold_datasets.py",
        "4_create_training_only_datasets.py",
    ]
] + [
    PROJECT_ROOT / f"Experiments/Ranknet/Bayesian/LIMUC/Scripts/{step}"
    for step in [
        "1_initial_learning_make_pair.py",
        "2_train_bayesian_ranknet.py",
        "3_predict_score.py",
        "4_calculate_uncertainty.py",
        "5_active_learning_make_pair.py",
        "6_train_bayesian_ranknet_AL.py",
        "7_select_comparison_pairs.py",
        "8_train_pair_acquisition.py",
        "9_predict_pair_experiment.py",
    ]
]


def load_path_module(module_path: Path):
    """Load a path helper module from a specific directory.

    Args:
        module_path (Path): Module file to load.

    Returns:
        module: Loaded Python module.
    """
    spec = importlib.util.spec_from_file_location(module_path.parent.name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("module_path", PATH_MODULES)
def test_data_root_controls_generated_paths(module_path, tmp_path):
    """A dummy data root should determine every generated-data location."""
    module = load_path_module(module_path)
    dummy_root = tmp_path / "patient-images"
    parser = argparse.ArgumentParser()
    module.add_data_root_argument(parser)

    args = parser.parse_args(["--data-root", str(dummy_root)])

    assert module.generated_path(args.data_root) == dummy_root / "generated"
    assert module.generated_path(args.data_root, "dataset") == (
        dummy_root / "generated/dataset"
    )
    assert module.generated_path(args.data_root, "Results") == (
        dummy_root / "generated/Results"
    )


@pytest.mark.parametrize("module_path", PATH_MODULES)
def test_environment_configures_default_data_root(module_path, monkeypatch, tmp_path):
    """The environment variable should configure the default root."""
    dummy_root = tmp_path / "environment-data"
    monkeypatch.setenv("LIMUC_DATA_ROOT", str(dummy_root))
    module = load_path_module(module_path)

    assert module.default_data_root() == dummy_root


@pytest.mark.parametrize("module_path", PATH_MODULES)
def test_laboratory_default_data_root(module_path, monkeypatch):
    """The fallback should use the laboratory data-server path."""
    monkeypatch.delenv("LIMUC_DATA_ROOT", raising=False)
    module = load_path_module(module_path)

    assert module.default_data_root() == EXPECTED_DEFAULT_DATA_ROOT


@pytest.mark.parametrize("script_path", PIPELINE_SCRIPTS)
def test_pipeline_script_uses_common_data_root(script_path):
    """Every preprocessing and experiment stage should expose the common root."""
    source = script_path.read_text(encoding="utf-8")

    assert "add_data_root_argument(parser)" in source
    assert "../../../../../Data/UC" not in source
