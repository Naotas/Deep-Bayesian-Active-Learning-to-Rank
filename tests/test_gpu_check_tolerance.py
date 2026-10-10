"""Tests for the numerical diagnostics used by the Pair-BALD GPU check."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECK_PATH = (
    PROJECT_ROOT / "Experiments/Ranknet/Bayesian/LIMUC/Scripts/check_pair_bald_gpu.py"
)


def load_check_module():
    """Load the GPU checker without importing TensorFlow."""
    spec = importlib.util.spec_from_file_location("check_pair_bald_gpu", CHECK_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_float32_roundoff_is_accepted() -> None:
    """Tiny batch-dependent float32 differences should not fail the check."""
    module = load_check_module()

    difference = module.assert_numerically_consistent(
        "dummy comparison",
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([5e-7, 1.0 + 2e-6], dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )

    assert difference > 0.0


def test_material_difference_is_rejected() -> None:
    """A mask-scale discrepancy must still fail with a labeled explanation."""
    module = load_check_module()

    with pytest.raises(AssertionError, match="shared mask check failed"):
        module.assert_numerically_consistent(
            "shared mask check",
            np.array([0.0, 1.0]),
            np.array([0.1, 1.0]),
            rtol=1e-5,
            atol=1e-6,
        )
