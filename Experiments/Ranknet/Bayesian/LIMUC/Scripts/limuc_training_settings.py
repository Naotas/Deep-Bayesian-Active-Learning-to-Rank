import csv
import math
from pathlib import Path

PAPER_LIMUC_LEARNING_RATE = 2e-5


def validate_initial_learning_rate(condition_path: Path, expected_rate: float) -> None:
    """Check that the saved initial model used the requested learning rate.

    Args:
        condition_path (Path): Saved AL_0 training conditions.
        expected_rate (float): Learning rate requested for the new iteration.

    Raises:
        FileNotFoundError: The initial training record does not exist.
        ValueError: The saved rate is missing or differs from the requested rate.
    """
    with condition_path.open(newline="") as condition_file:
        rows = list(csv.DictReader(condition_file))

    if len(rows) != 1 or not rows[0].get("learning_rate"):
        raise ValueError(
            f"AL_0 training conditions must record one learning_rate: {condition_path}"
        )

    initial_rate = float(rows[0]["learning_rate"])
    if not math.isfinite(initial_rate) or not math.isclose(
        initial_rate, expected_rate, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(
            f"AL_0 checkpoint used learning_rate={initial_rate}, "
            f"but this run uses --learning-rate={expected_rate}: {condition_path}"
        )
