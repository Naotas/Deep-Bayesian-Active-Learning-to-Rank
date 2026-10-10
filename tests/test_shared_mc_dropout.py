"""Small TensorFlow checks for shared-mask acquisition inference."""

import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "Experiments/Ranknet/Bayesian/LIMUC/Scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from shared_mc_dropout import SharedMCDropout  # noqa: E402


def test_same_mc_index_shares_mask_across_images_and_chunks() -> None:
    """Every image and chunk should use the same mask for a fixed MC index."""
    layer = SharedMCDropout(rate=0.5, mc_samples=3, seed=17, layer_index=2)

    first = layer(tf.ones((3 * 2, 32))).numpy().reshape(3, 2, 32)
    second = layer(tf.ones((3 * 4, 32))).numpy().reshape(3, 4, 32)

    np.testing.assert_array_equal(first[:, :1], first[:, 1:2])
    np.testing.assert_array_equal(second[:, :1], second[:, 1:2])
    np.testing.assert_array_equal(first[:, 0], second[:, 0])


def test_batch_normalization_statistics_stay_fixed_in_inference_mode() -> None:
    """Active Dropout must not force BatchNormalization into training mode."""
    inputs = tf.keras.Input(shape=(4,))
    normalized = tf.keras.layers.BatchNormalization()(inputs)
    outputs = SharedMCDropout(rate=0.5, mc_samples=2, seed=19, layer_index=0)(
        normalized
    )
    model = tf.keras.Model(inputs, outputs)
    batch_normalization = model.layers[1]
    mean_before = batch_normalization.moving_mean.numpy().copy()
    variance_before = batch_normalization.moving_variance.numpy().copy()

    model(tf.ones((2 * 3, 4)), training=False)

    np.testing.assert_array_equal(mean_before, batch_normalization.moving_mean.numpy())
    np.testing.assert_array_equal(
        variance_before, batch_normalization.moving_variance.numpy()
    )
