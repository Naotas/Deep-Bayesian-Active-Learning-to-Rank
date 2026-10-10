"""Shared-mask MC Dropout inference for Pair-BALD acquisition."""

import hashlib
import json
import os
import time
from pathlib import Path

import bayesian_densenet
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from tensorflow.python.keras.engine.base_layer import Layer


class SharedMCDropout(Layer):
    """Apply deterministic MC masks shared across the image-batch dimension.

    The incoming batch is laid out as ``(mc_samples * images, ...)``. Each MC
    sample receives a different dropout mask, while all images belonging to the
    same sample receive the same mask. Stateless generation makes that mask
    identical across successive image chunks.
    """

    def __init__(self, rate, mc_samples, seed, layer_index, **kwargs):
        """Initialize a shared-mask dropout layer.

        Args:
            rate (float): Dropout probability.
            mc_samples (int): Number of posterior samples packed into a batch.
            seed (int): Experiment seed.
            layer_index (int): Stable per-layer seed component.
            **kwargs: Base Keras layer options.
        """
        super().__init__(**kwargs)
        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be within [0, 1).")
        if mc_samples < 1:
            raise ValueError("mc_samples must be at least 1.")
        self.rate = float(rate)
        self.mc_samples = int(mc_samples)
        self.seed = int(seed)
        self.layer_index = int(layer_index)

    def call(self, inputs, training=None):
        """Apply shared stateless masks regardless of the learning phase."""
        del training
        if self.rate == 0.0:
            return inputs

        input_shape = tf.shape(inputs)
        packed_batch = input_shape[0]
        divisibility_check = tf.debugging.assert_equal(
            tf.math.floormod(packed_batch, self.mc_samples),
            0,
            message="Packed MC batch must be divisible by mc_samples.",
        )
        dependencies = [divisibility_check] if divisibility_check is not None else []
        with tf.control_dependencies(dependencies):
            images_per_sample = packed_batch // self.mc_samples
            expanded_shape = tf.concat(
                [
                    tf.stack(
                        [
                            tf.constant(self.mc_samples, dtype=input_shape.dtype),
                            images_per_sample,
                        ]
                    ),
                    input_shape[1:],
                ],
                axis=0,
            )
            expanded = tf.reshape(inputs, expanded_shape)

        mask_shape = tf.concat(
            [
                tf.constant([self.mc_samples, 1], dtype=input_shape.dtype),
                input_shape[1:],
            ],
            axis=0,
        )
        keep_probability = 1.0 - self.rate
        random_values = tf.random.stateless_uniform(
            mask_shape,
            seed=[self.seed, self.layer_index],
            dtype=inputs.dtype,
        )
        mask = tf.floor(random_values + keep_probability) / keep_probability
        return tf.reshape(expanded * mask, input_shape)

    def get_config(self):
        """Return serializable layer settings."""
        config = super().get_config()
        config.update(
            {
                "rate": self.rate,
                "mc_samples": self.mc_samples,
                "seed": self.seed,
                "layer_index": self.layer_index,
            }
        )
        return config


class SharedDropoutFactory:
    """Create shared-mask layers with stable, distinct per-layer seeds."""

    def __init__(self, mc_samples, seed):
        """Initialize the factory.

        Args:
            mc_samples (int): Number of aligned MC samples.
            seed (int): Experiment seed.
        """
        self.mc_samples = mc_samples
        self.seed = seed
        self.layer_index = 0

    def __call__(self, rate):
        """Create the next dropout layer."""
        layer = SharedMCDropout(
            rate=rate,
            mc_samples=self.mc_samples,
            seed=self.seed,
            layer_index=self.layer_index,
        )
        self.layer_index += 1
        return layer


def require_gpu(allow_cpu_smoke_test=False):
    """Stop production inference when TensorFlow cannot see a GPU.

    Args:
        allow_cpu_smoke_test (bool): Permit an explicitly requested small CPU
            validation run.

    Raises:
        RuntimeError: If no GPU is visible and CPU smoke mode is disabled.
    """
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    if not gpu_devices and not allow_cpu_smoke_test:
        raise RuntimeError(
            "TensorFlow cannot see a GPU. Pair-acquisition inference was not "
            "started. Use --allow-cpu-smoke-test only for a deliberately small "
            "validation run."
        )


def build_shared_mc_score_model(
    checkpoint_path,
    mc_samples,
    seed,
    image_shape=(224, 224, 3),
    dropout_rate=0.2,
    weight_decay=1e-4,
):
    """Build the checkpoint-compatible RankNet scoring model.

    Args:
        checkpoint_path (Path): Existing RankNet ``.h5`` weights.
        mc_samples (int): Number of aligned MC Dropout samples.
        seed (int): Dropout seed.
        image_shape (tuple): Input image shape.
        dropout_rate (float): Existing model dropout rate.
        weight_decay (float): Existing model L2 coefficient.

    Returns:
        keras.Model: Single-image scoring model with shared MC masks.
    """
    dropout_factory = SharedDropoutFactory(mc_samples, seed)
    original_dropout = bayesian_densenet.Dropout
    bayesian_densenet.Dropout = dropout_factory
    try:
        model_features = bayesian_densenet.DenseNet169(
            include_top=True,
            weights=None,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
        )
    finally:
        bayesian_densenet.Dropout = original_dropout

    model_features = Model(
        inputs=model_features.input,
        outputs=model_features.get_layer("avg_pool").output,
    )
    dropped_features = dropout_factory(dropout_rate)(
        model_features.output, training=True
    )
    scores = Dense(
        1,
        kernel_regularizer=regularizers.l2(weight_decay),
        name="fc",
    )(dropped_features)
    scorer = Model(inputs=model_features.input, outputs=scores)
    inputs = Input(shape=image_shape, name="data_x")
    scores = scorer(inputs)
    model = Model(inputs=inputs, outputs=scores)
    model.load_weights(str(checkpoint_path))
    return model


def _batch_normalization_state(model):
    state = {}

    def visit(layer, prefix, seen):
        if id(layer) in seen:
            return
        seen.add(id(layer))
        layer_path = f"{prefix}/{layer.name}" if prefix else layer.name
        if hasattr(layer, "moving_mean") and hasattr(layer, "moving_variance"):
            state[f"{layer_path}/mean"] = K.get_value(layer.moving_mean).copy()
            state[f"{layer_path}/variance"] = K.get_value(layer.moving_variance).copy()
        for child in getattr(layer, "layers", []):
            visit(child, layer_path, seen)

    visit(model, "", set())
    return state


def _assert_batch_normalization_unchanged(before, model):
    after = _batch_normalization_state(model)
    if before.keys() != after.keys():
        raise RuntimeError(
            "BatchNormalization layer inventory changed during inference."
        )
    changed = [name for name in before if not np.array_equal(before[name], after[name])]
    if changed:
        raise RuntimeError(
            "BatchNormalization statistics changed during acquisition inference: "
            f"{changed}"
        )


def predict_aligned_scores(
    model,
    image_ids,
    image_dir,
    mc_samples,
    image_batch_size=1,
    image_shape=(224, 224, 3),
):
    """Predict a ``T×N`` score cache with masks aligned across all images.

    Args:
        model (keras.Model): Model built by ``build_shared_mc_score_model``.
        image_ids: Ordered image IDs.
        image_dir (Path): Directory containing the images.
        mc_samples (int): Number of aligned samples.
        image_batch_size (int): Number of distinct images per model call.
        image_shape (tuple): Resized image shape.

    Returns:
        np.ndarray: Scores with shape ``(T, N)``.

    Raises:
        FileNotFoundError: If an image cannot be read.
        RuntimeError: If BatchNormalization statistics change.
    """
    if image_batch_size < 1:
        raise ValueError("image_batch_size must be at least 1.")
    image_ids = [str(image_id) for image_id in image_ids]
    scores = np.empty((mc_samples, len(image_ids)), dtype=np.float32)
    bn_before = _batch_normalization_state(model)

    for start in range(0, len(image_ids), image_batch_size):
        chunk_ids = image_ids[start : start + image_batch_size]
        valid_image_count = len(chunk_ids)
        images = []
        for image_id in chunk_ids:
            image_path = os.path.join(str(image_dir), image_id)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            image = cv2.resize(image, dsize=(image_shape[0], image_shape[1]))
            images.append(image)
        batch = np.asarray(images, dtype=np.float32) / 255.0
        if valid_image_count < image_batch_size:
            padding = np.repeat(
                batch[-1:], image_batch_size - valid_image_count, axis=0
            )
            batch = np.concatenate([batch, padding], axis=0)
        packed_batch = np.tile(batch, (mc_samples, 1, 1, 1))
        packed_scores = model.predict(
            packed_batch, batch_size=len(packed_batch), verbose=0
        )
        chunk_scores = np.asarray(packed_scores, dtype=np.float32).reshape(
            mc_samples, image_batch_size
        )
        scores[:, start : start + valid_image_count] = chunk_scores[
            :, :valid_image_count
        ]

    _assert_batch_normalization_unchanged(bn_before, model)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Aligned MC score cache contains NaN or infinity.")
    return scores


def checkpoint_identifier(checkpoint_path):
    """Return a content-based identifier for a RankNet checkpoint.

    Args:
        checkpoint_path (Path): Checkpoint file.

    Returns:
        dict: Resolved path, size, modification time, and SHA-256 digest.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    stat = checkpoint_path.stat()
    return {
        "path": str(checkpoint_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def image_order_digest(image_ids):
    """Hash the exact ordered image-ID sequence used by a score cache."""
    payload = "\n".join(str(image_id) for image_id in image_ids).encode()
    return hashlib.sha256(payload).hexdigest()


def cache_metadata(
    image_ids,
    checkpoint_path,
    fold,
    mc_samples,
    seed,
    image_batch_size,
    image_shape=(224, 224, 3),
    dropout_rate=0.2,
):
    """Create the conditions that uniquely identify an aligned MC cache."""
    return {
        "format_version": 2,
        "fold": int(fold),
        "mc_samples": int(mc_samples),
        "seed": int(seed),
        "image_batch_size": int(image_batch_size),
        "image_shape": list(image_shape),
        "dropout_rate": float(dropout_rate),
        "dropout_mode": "stateless_masks_shared_across_images_and_chunks",
        "partial_chunk_policy": "repeat_last_image_to_fixed_batch_size",
        "batch_normalization_mode": "inference",
        "image_count": len(image_ids),
        "image_order_sha256": image_order_digest(image_ids),
        "checkpoint": checkpoint_identifier(checkpoint_path),
    }


def load_validated_cache(cache_path, expected_metadata):
    """Load an aligned score cache only when every condition matches.

    Args:
        cache_path (Path): ``.npz`` cache path.
        expected_metadata (dict): Conditions from ``cache_metadata``.

    Returns:
        tuple: Ordered image IDs and score array, or ``None`` when absent.

    Raises:
        RuntimeError: If a partial or incompatible cache exists.
    """
    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(".json")
    if not cache_path.exists() and not metadata_path.exists():
        return None
    if not cache_path.exists() or not metadata_path.exists():
        raise RuntimeError(f"Incomplete MC cache: {cache_path}")

    with metadata_path.open(encoding="utf-8") as metadata_file:
        actual_metadata = json.load(metadata_file)
    if actual_metadata != expected_metadata:
        raise RuntimeError(
            "MC cache conditions do not match the requested fold, checkpoint, "
            "image order, or dropout settings. Refusing cache reuse."
        )
    with np.load(cache_path, allow_pickle=False) as cache:
        image_ids = cache["image_ids"].astype(str)
        scores = cache["scores"].astype(np.float32)
    expected_shape = (expected_metadata["mc_samples"], len(image_ids))
    if scores.shape != expected_shape:
        raise RuntimeError(
            f"MC cache shape is {scores.shape}, expected {expected_shape}."
        )
    return image_ids, scores


def create_or_load_cache(
    cache_path,
    image_ids,
    image_dir,
    checkpoint_path,
    fold,
    mc_samples=30,
    seed=240119,
    image_batch_size=1,
    allow_cpu_smoke_test=False,
):
    """Create or reuse a validated, aligned ``T×N`` image-score cache.

    Args:
        cache_path (Path): Output ``.npz`` path outside the repository.
        image_ids: Ordered training image IDs.
        image_dir (Path): Image directory.
        checkpoint_path (Path): Explicit RankNet weight file.
        fold (int): Fold identifier.
        mc_samples (int): Number of MC Dropout samples.
        seed (int): Reproducibility seed.
        image_batch_size (int): Distinct images per model call.
        allow_cpu_smoke_test (bool): Explicitly allow a small CPU-only run.

    Returns:
        tuple: Ordered IDs, scores, inference seconds, and a cache-reuse flag.
    """
    image_ids = [str(image_id) for image_id in image_ids]
    metadata = cache_metadata(
        image_ids=image_ids,
        checkpoint_path=checkpoint_path,
        fold=fold,
        mc_samples=mc_samples,
        seed=seed,
        image_batch_size=image_batch_size,
    )
    cached = load_validated_cache(cache_path, metadata)
    if cached is not None:
        return cached[0], cached[1], 0.0, True

    require_gpu(allow_cpu_smoke_test=allow_cpu_smoke_test)
    model = build_shared_mc_score_model(
        checkpoint_path=checkpoint_path,
        mc_samples=mc_samples,
        seed=seed,
    )
    start = time.perf_counter()
    scores = predict_aligned_scores(
        model=model,
        image_ids=image_ids,
        image_dir=image_dir,
        mc_samples=mc_samples,
        image_batch_size=image_batch_size,
    )
    inference_seconds = time.perf_counter() - start

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, image_ids=np.asarray(image_ids), scores=scores)
    with cache_path.with_suffix(".json").open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return np.asarray(image_ids), scores, inference_seconds, False
