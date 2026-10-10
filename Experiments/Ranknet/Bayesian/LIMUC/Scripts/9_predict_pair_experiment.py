"""Generate legacy-format MC predictions for an isolated pair experiment."""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from data_paths import add_data_root_argument, generated_path


def predict(args):
    """Predict requested splits with the existing independent-mask MC scheme."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    import cv2
    import keras
    import numpy as np
    import tensorflow as tf
    from bayesian_densenet import DenseNet169
    from keras import regularizers
    from keras.layers import Input
    from keras.layers.core import Dense, Dropout
    from keras.models import Model

    if not tf.config.experimental.list_physical_devices("GPU"):
        raise RuntimeError(
            "TensorFlow cannot see a GPU. Experiment prediction was not started."
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    class ImageGenerator(keras.utils.Sequence):
        """Load one deterministic fold split for repeated MC prediction."""

        def __init__(self, image_ids):
            self.image_ids = list(image_ids)

        def __len__(self):
            return int(np.ceil(len(self.image_ids) / args.batch_size))

        def __getitem__(self, index):
            ids = self.image_ids[
                index * args.batch_size : (index + 1) * args.batch_size
            ]
            images = []
            for image_id in ids:
                image_path = args.image_dir / str(image_id)
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Could not read image: {image_path}")
                images.append(cv2.resize(image, dsize=(224, 224)))
            return np.asarray(images, dtype=np.float32) / 255.0

    def build_model():
        features = DenseNet169(
            include_top=True,
            weights=None,
            dropout_rate=0.2,
            weight_decay=1e-4,
        )
        features = Model(
            inputs=features.input,
            outputs=features.get_layer("avg_pool").output,
        )
        dropped = Dropout(0.2)(features.output, training=True)
        scores = Dense(
            1,
            kernel_regularizer=regularizers.l2(1e-4),
            name="fc",
        )(dropped)
        scorer = Model(inputs=features.input, outputs=scores)
        inputs = Input(shape=(224, 224, 3), name="data_x")
        scores = scorer(inputs)
        model = Model(inputs=inputs, outputs=scores)
        model.load_weights(str(args.checkpoint))
        return model

    if args.output_dir.exists():
        if args.resume and (args.output_dir / "prediction_manifest.json").is_file():
            print(f"Prediction is already complete: {args.output_dir}")
            return
        raise FileExistsError(f"Prediction output already exists: {args.output_dir}")

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    model = build_model()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    timing_rows = []
    for datatype in args.datatypes:
        dataset_path = args.dataset_dir / f"{args.fold}_{datatype}.csv"
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Fold dataset CSV not found: {dataset_path}")
        data = pd.read_csv(dataset_path)
        required = {"filename", "MayoLabel"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{dataset_path} is missing columns: {sorted(missing)}")
        image_ids = data["filename"].astype(str).tolist()
        generator = ImageGenerator(image_ids)

        start = time.perf_counter()
        samples = []
        for _ in range(args.mc_samples):
            predicted = model.predict(generator, verbose=0)
            samples.append(np.asarray(predicted, dtype=np.float64).reshape(-1))
        prediction_seconds = time.perf_counter() - start
        sample_matrix = np.stack(samples, axis=1)
        if not np.all(np.isfinite(sample_matrix)):
            raise RuntimeError("MC predictions contain NaN or infinity.")

        prediction = pd.DataFrame(
            {
                "filename": image_ids,
                "label": data["MayoLabel"].astype(float),
            }
        )
        for sample_index in range(args.mc_samples):
            prediction[f"sampling_{sample_index}"] = sample_matrix[:, sample_index]
        prediction.to_csv(args.output_dir / f"{datatype}_prediction.csv", index=False)
        pd.DataFrame(
            {
                "filename": image_ids,
                "label": data["MayoLabel"].astype(float),
                "var_score": np.var(sample_matrix, axis=1),
                "mean_score": np.mean(sample_matrix, axis=1),
            }
        ).to_csv(
            args.output_dir / f"{datatype}_mean_score_and_uncertainty.csv",
            index=False,
        )
        timing_rows.append(
            {
                "datatype": datatype,
                "image_count": len(image_ids),
                "prediction_seconds": prediction_seconds,
            }
        )

    pd.DataFrame(timing_rows).to_csv(
        args.output_dir / "prediction_timing.csv", index=False
    )
    manifest = {
        "checkpoint": str(args.checkpoint),
        "fold": args.fold,
        "seed": args.seed,
        "mc_samples": args.mc_samples,
        "datatypes": args.datatypes,
        "dropout_mode": "legacy_independent_activation_masks",
        "batch_normalization_mode": "inference",
        "status": "complete",
    }
    with (args.output_dir / "prediction_manifest.json").open(
        "w", encoding="utf-8"
    ) as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    print(f"Saved experiment predictions: {args.output_dir}")


def parse_args():
    """Parse explicit prediction paths and MC settings."""
    parser = argparse.ArgumentParser(
        description="Predict train/test scores for one pair-acquisition checkpoint."
    )
    add_data_root_argument(parser)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", type=int, default=240119)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--datatypes", default="train,test")
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.dataset_dir = (
        args.dataset_dir.expanduser()
        if args.dataset_dir is not None
        else generated_path(args.data_root, "dataset")
    )
    args.image_dir = (
        args.image_dir.expanduser()
        if args.image_dir is not None
        else generated_path(args.data_root, "all_public_UC_images")
    )
    args.datatypes = [
        value.strip() for value in args.datatypes.split(",") if value.strip()
    ]
    if not set(args.datatypes) <= {"train", "valid", "test"}:
        parser.error("--datatypes contains an unknown split.")
    if args.fold < 1 or args.mc_samples < 1 or args.batch_size < 1:
        parser.error("--fold, --mc-samples, and --batch-size must be positive.")
    return args


if __name__ == "__main__":
    predict(parse_args())
