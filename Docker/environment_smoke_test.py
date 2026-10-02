"""Validate the legacy training runtime with generated dummy data."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from tempfile import NamedTemporaryFile

import cv2
import h5py
import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend, regularizers
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.layers.core import Activation, Dense, Dropout
from tensorflow.python.keras.utils import data_utils, layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import keras_export

EXPECTED_VERSIONS = {
    "h5py": "2.10.0",
    "keras": "2.4.3",
    "numpy": "1.19.5",
    "opencv-python-headless": "4.5.1.48",
    "scipy": "1.5.4",
    "tensorflow": "2.4.0",
}


def parse_args() -> argparse.Namespace:
    """Parse smoke-test options.

    Returns:
        argparse.Namespace: Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail unless TensorFlow detects at least one GPU.",
    )
    return parser.parse_args()


def validate_versions() -> None:
    """Check the packages whose exact versions define the legacy runtime."""
    for distribution, expected in EXPECTED_VERSIONS.items():
        actual = version(distribution)
        if actual != expected:
            raise RuntimeError(
                f"Unexpected {distribution} version: {actual}; expected {expected}"
            )

    try:
        version("scikit-learn")
    except PackageNotFoundError:
        pass
    else:
        raise RuntimeError("scikit-learn is unused and must not be installed")


def validate_opencv_and_h5py() -> None:
    """Exercise image and HDF5 operations used by preprocessing and training."""
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[2:10, 4:12] = (10, 80, 160)
    resized = cv2.resize(image, dsize=(8, 8))
    encoded, buffer = cv2.imencode(".jpg", resized)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if not encoded or decoded.shape != (8, 8, 3):
        raise RuntimeError("OpenCV image I/O smoke test failed")

    with NamedTemporaryFile(suffix=".h5") as temporary_file:
        with h5py.File(temporary_file.name, "w") as h5_file:
            h5_file.create_dataset("dummy_image", data=decoded)
        with h5py.File(temporary_file.name, "r") as h5_file:
            if h5_file["dummy_image"].shape != (8, 8, 3):
                raise RuntimeError("h5py round-trip smoke test failed")


def validate_tensorflow(require_gpu: bool) -> list[str]:
    """Exercise TensorFlow, standalone Keras, and private APIs used by the code.

    Args:
        require_gpu (bool): Whether absence of a visible GPU is an error.

    Returns:
        list[str]: Names of GPUs visible to TensorFlow.
    """
    private_api_symbols = (
        backend,
        regularizers,
        imagenet_utils,
        training,
        VersionAwareLayers,
        Activation,
        Dense,
        Dropout,
        data_utils,
        layer_utils,
        file_io,
        keras_export,
    )
    if any(symbol is None for symbol in private_api_symbols):
        raise RuntimeError(
            "A TensorFlow private API used by bayesian_densenet is missing"
        )

    tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    product = tf.matmul(tensor, tensor)
    if not np.array_equal(product.numpy(), np.array([[7.0, 10.0], [15.0, 22.0]])):
        raise RuntimeError("TensorFlow tensor operation smoke test failed")

    model = keras.Sequential([keras.layers.Dense(1, input_shape=(2,))])
    prediction = model.predict(np.zeros((1, 2), dtype=np.float32))
    if prediction.shape != (1, 1):
        raise RuntimeError("Standalone Keras smoke test failed")

    with NamedTemporaryFile(suffix=".h5") as temporary_file:
        model.save_weights(temporary_file.name)
        model.load_weights(temporary_file.name)

    session_config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    with tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(),
        config=session_config,
    ) as session:
        tf.compat.v1.keras.backend.set_session(session)

    gpu_names = [device.name for device in tf.config.list_physical_devices("GPU")]
    if require_gpu and not gpu_names:
        raise RuntimeError("TensorFlow did not detect a GPU")
    return gpu_names


def main() -> None:
    """Run dependency and runtime checks."""
    args = parse_args()
    validate_versions()
    validate_opencv_and_h5py()
    gpu_names = validate_tensorflow(args.require_gpu)
    gpu_summary = ", ".join(gpu_names) if gpu_names else "none (CPU-only check)"
    print("Environment smoke test passed.")
    print(f"Visible GPUs: {gpu_summary}")


if __name__ == "__main__":
    main()
