"""Run a small GPU check for shared-mask Pair-BALD inference."""

import argparse
import os
from pathlib import Path

import numpy as np


def assert_numerically_consistent(label, actual, expected, rtol, atol):
    """Check float32 GPU results while reporting the observed discrepancy.

    Args:
        label (str): Description shown when the comparison fails.
        actual: First numeric result.
        expected: Reference numeric result.
        rtol (float): Relative tolerance for float32 GPU arithmetic.
        atol (float): Absolute tolerance for values close to zero.

    Returns:
        float: Maximum absolute difference.

    Raises:
        AssertionError: If the discrepancy exceeds the configured tolerances.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    maximum_difference = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        err_msg=(
            f"{label} failed; max_abs_difference={maximum_difference:.9g}. "
            "This can indicate that masks are not shared, unless the difference "
            "is only float32 GPU roundoff."
        ),
    )
    print(f"{label}: max_abs_difference={maximum_difference:.9g}")
    return maximum_difference


def main():
    """Validate GPU discovery, checkpoint loading, shared masks, and frozen BN."""
    parser = argparse.ArgumentParser(
        description="Check Pair-BALD inference with generated dummy tensors."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--mc-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=240119)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.mc_samples < 2:
        parser.error("--mc-samples must be at least 2 for this check.")
    if args.rtol < 0 or args.atol < 0:
        parser.error("--rtol and --atol must be non-negative.")
    args.checkpoint = args.checkpoint.expanduser()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    from shared_mc_dropout import (
        _assert_batch_normalization_unchanged,
        _batch_normalization_state,
        build_shared_mc_score_model,
        require_gpu,
    )

    require_gpu()
    model = build_shared_mc_score_model(
        checkpoint_path=args.checkpoint,
        mc_samples=args.mc_samples,
        seed=args.seed,
    )
    bn_before = _batch_normalization_state(model)
    chunk = np.ones((args.mc_samples * 2, 224, 224, 3), dtype=np.float32)
    first_scores = model.predict(chunk, batch_size=len(chunk), verbose=0).reshape(
        args.mc_samples, 2
    )
    second_scores = model.predict(chunk, batch_size=len(chunk), verbose=0).reshape(
        args.mc_samples, 2
    )
    assert_numerically_consistent(
        "shared mask within the first fixed-size chunk",
        first_scores[:, 0],
        first_scores[:, 1],
        rtol=args.rtol,
        atol=args.atol,
    )
    assert_numerically_consistent(
        "shared mask within the second fixed-size chunk",
        second_scores[:, 0],
        second_scores[:, 1],
        rtol=args.rtol,
        atol=args.atol,
    )
    assert_numerically_consistent(
        "shared mask across equal-size chunks",
        first_scores[:, 0],
        second_scores[:, 0],
        rtol=args.rtol,
        atol=args.atol,
    )
    _assert_batch_normalization_unchanged(bn_before, model)
    print(
        "GPU, checkpoint, shared-mask, fixed-shape chunk consistency, and frozen-BN "
        "checks passed."
    )


if __name__ == "__main__":
    main()
