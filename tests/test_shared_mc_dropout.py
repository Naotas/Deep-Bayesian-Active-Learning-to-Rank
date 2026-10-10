"""Small TensorFlow checks for shared-mask acquisition inference."""

import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "Experiments/Ranknet/Bayesian/LIMUC/Scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import shared_mc_dropout  # noqa: E402
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


def test_partial_image_chunk_uses_fixed_packed_batch_size(monkeypatch) -> None:
    """The final partial chunk should be padded without retaining padded scores."""

    class RecordingModel:
        """Record packed inputs and return deterministic dummy scores."""

        layers = []

        def __init__(self):
            self.packed_batch_sizes = []

        def predict(self, packed_batch, batch_size, verbose):
            assert batch_size == len(packed_batch)
            assert verbose == 0
            self.packed_batch_sizes.append(len(packed_batch))
            return np.arange(len(packed_batch), dtype=np.float32)[:, None]

    monkeypatch.setattr(
        shared_mc_dropout.cv2,
        "imread",
        lambda _path, _mode: np.ones((2, 2, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        shared_mc_dropout.cv2,
        "resize",
        lambda image, dsize: image,
    )
    model = RecordingModel()

    scores = shared_mc_dropout.predict_aligned_scores(
        model=model,
        image_ids=["one.png", "two.png", "three.png"],
        image_dir="/dummy",
        mc_samples=3,
        image_batch_size=2,
        image_shape=(2, 2, 3),
    )

    assert model.packed_batch_sizes == [6, 6]
    np.testing.assert_array_equal(
        scores,
        np.array(
            [
                [0.0, 1.0, 0.0],
                [2.0, 3.0, 2.0],
                [4.0, 5.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )
