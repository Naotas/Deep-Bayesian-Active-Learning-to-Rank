"""Check the paper's LIMUC learning rate without loading a GPU model."""

import ast
import csv
import importlib.util
from pathlib import Path

import pytest

SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "Experiments/Ranknet/Bayesian/LIMUC/Scripts"
)
spec = importlib.util.spec_from_file_location(
    "limuc_training_settings", SCRIPT_DIR / "limuc_training_settings.py"
)
settings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(settings)
PAPER_LIMUC_LEARNING_RATE = settings.PAPER_LIMUC_LEARNING_RATE
validate_initial_learning_rate = settings.validate_initial_learning_rate


@pytest.mark.parametrize(
    "script_name",
    ["2_train_bayesian_ranknet.py", "6_train_bayesian_ranknet_AL.py"],
)
def test_training_scripts_default_to_paper_limuc_rate(script_name):
    """Initial and later training use the same public-data default."""
    tree = ast.parse((SCRIPT_DIR / script_name).read_text())
    options = (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--learning-rate"
    )
    option = next(options)
    default = next(item.value for item in option.keywords if item.arg == "default")
    assert isinstance(default, ast.Name)
    assert default.id == "PAPER_LIMUC_LEARNING_RATE"
    assert PAPER_LIMUC_LEARNING_RATE == 2e-5


def write_conditions(path: Path, rates: list[str]) -> None:
    """Write dummy AL_0 condition rows for a fold."""
    with path.open("w", newline="") as condition_file:
        writer = csv.DictWriter(condition_file, fieldnames=["learning_rate"])
        writer.writeheader()
        writer.writerows({"learning_rate": rate} for rate in rates)


def test_matching_initial_rate_is_accepted(tmp_path):
    """AL_1 may start from an AL_0 model trained at the paper rate."""
    condition_path = tmp_path / "experimental_condition.csv"
    write_conditions(condition_path, ["0.00002"])
    validate_initial_learning_rate(condition_path, PAPER_LIMUC_LEARNING_RATE)


@pytest.mark.parametrize("saved_rate", ["0.00005", "nan", "inf"])
def test_mismatched_initial_rate_stops(tmp_path, saved_rate):
    """A prior 5e-5 checkpoint cannot silently enter a 2e-5 run."""
    condition_path = tmp_path / "experimental_condition.csv"
    write_conditions(condition_path, [saved_rate])
    with pytest.raises(ValueError, match="AL_0 checkpoint used learning_rate"):
        validate_initial_learning_rate(condition_path, PAPER_LIMUC_LEARNING_RATE)


@pytest.mark.parametrize("rates", [[], ["0.00002", "0.00002"]])
def test_missing_or_ambiguous_initial_rate_stops(tmp_path, rates):
    """An absent or ambiguous training record cannot certify AL_0."""
    condition_path = tmp_path / "experimental_condition.csv"
    write_conditions(condition_path, rates)
    with pytest.raises(ValueError, match="must record one learning_rate"):
        validate_initial_learning_rate(condition_path, PAPER_LIMUC_LEARNING_RATE)
