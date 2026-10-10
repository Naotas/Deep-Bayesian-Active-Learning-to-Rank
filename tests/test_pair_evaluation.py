"""Tests for unseen-patient pairwise evaluation."""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "Experiments/Ranknet/Bayesian/LIMUC/Scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from pair_evaluation import DIRECTIONAL_MODES, evaluate_predictions  # noqa: E402


def test_evaluation_reports_directional_modes_and_ties_separately() -> None:
    """Same-grade pairs must not be counted as binary direction errors."""
    prediction_data = pd.DataFrame(
        {
            "filename": [
                f"grade_{grade}_{index}" for grade in range(4) for index in range(3)
            ],
            "label": [grade for grade in range(4) for _ in range(3)],
            "mean_score": [
                grade + index * 0.01 for grade in range(4) for index in range(3)
            ],
        }
    )

    result = evaluate_predictions(prediction_data)

    for mode in DIRECTIONAL_MODES:
        assert f"{mode}_accuracy" in result
        assert f"{mode}_pair_count" in result
    assert result["tie_pair_count"] > 0
    assert "excluded_from_directional_accuracy" in result["tie_evaluation_policy"]
