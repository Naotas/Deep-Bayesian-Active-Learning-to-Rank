"""Run a small GPU check for shared-mask Pair-BALD inference."""

import argparse
import os
from pathlib import Path


def main():
    """Validate GPU discovery, checkpoint loading, shared masks, and frozen BN."""
    parser = argparse.ArgumentParser(
        description="Check Pair-BALD inference with generated dummy tensors."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--mc-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=240119)
    args = parser.parse_args()
    if args.mc_samples < 2:
        parser.error("--mc-samples must be at least 2 for this check.")
    args.checkpoint = args.checkpoint.expanduser()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import numpy as np
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
    two_images = np.ones((args.mc_samples * 2, 224, 224, 3), dtype=np.float32)
    four_images = np.ones((args.mc_samples * 4, 224, 224, 3), dtype=np.float32)
    first_scores = model.predict(two_images, verbose=0).reshape(args.mc_samples, 2)
    second_scores = model.predict(four_images, verbose=0).reshape(args.mc_samples, 4)
    np.testing.assert_allclose(first_scores[:, 0], first_scores[:, 1])
    np.testing.assert_allclose(second_scores[:, 0], second_scores[:, 3])
    np.testing.assert_allclose(first_scores[:, 0], second_scores[:, 0])
    _assert_batch_normalization_unchanged(bn_before, model)
    print(
        "GPU, checkpoint, shared-mask, chunk-consistency, and frozen-BN checks passed."
    )


if __name__ == "__main__":
    main()
