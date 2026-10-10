"""Train RankNet on cumulative pairs from one isolated acquisition method."""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from data_paths import add_data_root_argument, generated_path


def read_pair_csv(path):
    """Read a RankNet pair CSV and validate its training columns.

    Args:
        path (Path): Pair CSV path.

    Returns:
        pd.DataFrame: Validated pair table.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Pair CSV not found: {path}")
    data = pd.read_csv(path)
    required = {"x1_image", "x2_image", "relative_label"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError(f"Pair CSV is empty: {path}")
    if not data["relative_label"].isin([0.0, 0.5, 1.0]).all():
        raise ValueError(f"Unexpected relative labels in: {path}")
    return data


def output_directory(args):
    """Return the isolated training destination."""
    return (
        args.experiment_root
        / args.experiment_id
        / "training"
        / args.method
        / f"seed_{args.seed}"
        / f"round_{args.round_number}"
        / f"fold_{args.fold}"
    )


def completed_training(output_dir, args):
    """Return whether a completed training run can be reused safely."""
    manifest_path = output_dir / "training_manifest.json"
    if not output_dir.exists():
        return False
    if not args.resume:
        raise FileExistsError(f"Training output already exists: {output_dir}")
    if not manifest_path.is_file():
        raise RuntimeError(f"Refusing to resume incomplete training: {output_dir}")
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    expected = {
        "experiment_id": args.experiment_id,
        "method": args.method,
        "seed": args.seed,
        "round": args.round_number,
        "fold": args.fold,
        "status": "complete",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Completed training manifest does not match this request.")
    print(f"Training is already complete: {output_dir}")
    return True


def train(args):
    """Train one fold from the same fixed initial checkpoint.

    TensorFlow and Keras are imported after ``CUDA_VISIBLE_DEVICES`` is set so
    the requested accelerator selection takes effect before runtime discovery.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    import cv2
    import keras
    import numpy as np
    import tensorflow as tf
    from bayesian_densenet import DenseNet169
    from callbacks import makecallbacks
    from keras import backend as K
    from keras import regularizers
    from keras.layers import Input
    from keras.layers.core import Dense, Dropout
    from keras.models import Model
    from keras.optimizers import Adam

    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    if not gpu_devices:
        raise RuntimeError(
            "TensorFlow cannot see a GPU. Full RankNet training was not started."
        )

    class BatchGenerator(keras.utils.Sequence):
        """Load RankNet image pairs without retaining real images in memory."""

        def __init__(self, pair_data, shuffle):
            self.pair_data = pair_data.reset_index(drop=True)
            self.shuffle = shuffle
            self.indexes = np.arange(len(pair_data))

        def __len__(self):
            return int(np.ceil(len(self.pair_data) / args.batch_size))

        def __getitem__(self, index):
            indexes = self.indexes[
                index * args.batch_size : (index + 1) * args.batch_size
            ]
            x1_batch = []
            x2_batch = []
            labels = []
            for row_index in indexes:
                row = self.pair_data.iloc[row_index]
                images = []
                for column in ["x1_image", "x2_image"]:
                    image_path = args.image_dir / str(row[column])
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        raise FileNotFoundError(f"Could not read image: {image_path}")
                    images.append(cv2.resize(image, dsize=(224, 224)))
                x1_batch.append(images[0])
                x2_batch.append(images[1])
                labels.append(row["relative_label"])
            return [
                np.asarray(x1_batch, dtype=np.float32) / 255.0,
                np.asarray(x2_batch, dtype=np.float32) / 255.0,
                np.asarray(labels, dtype=np.float32),
            ]

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indexes)

    def pairwise_loss(x1_score, x2_score, relative_label):
        difference = x1_score - x2_score
        return K.mean((1.0 - relative_label) * difference + K.softplus(-difference))

    def build_ranknet():
        x1_inputs = Input(shape=(224, 224, 3), name="data_x1")
        x2_inputs = Input(shape=(224, 224, 3), name="data_x2")
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
        score = Dense(1, kernel_regularizer=regularizers.l2(1e-4), name="fc")(dropped)
        scorer = Model(inputs=features.input, outputs=score)
        x1_score = scorer(x1_inputs)
        x2_score = scorer(x2_inputs)
        relative_label = Input(shape=(1,), name="rel_label")
        model = Model(
            inputs=[x1_inputs, x2_inputs, relative_label],
            outputs=[x1_score, x2_score],
        )
        model.add_loss(pairwise_loss(x1_score, x2_score, relative_label))
        model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss=None)
        return model

    output_dir = output_directory(args)
    if completed_training(output_dir, args):
        return
    train_pairs = read_pair_csv(args.train_pair_csv)
    valid_pairs = read_pair_csv(args.valid_pair_csv)
    if not args.initial_checkpoint.is_file():
        raise FileNotFoundError(
            f"Initial RankNet checkpoint not found: {args.initial_checkpoint}"
        )

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    model = build_ranknet()
    model.load_weights(str(args.initial_checkpoint))

    callbacks = makecallbacks(
        weight_name=str(output_dir / "weights/epoch{epoch:04d}-{val_loss:.4f}.h5"),
        tsv_name=str(output_dir / "history.csv"),
        isEarlyStop=True,
        patience=args.patience,
        isTensorBoard=False,
        tensorboard_path=str(output_dir / "logdir"),
    )
    (output_dir / "weights").mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model.fit(
        x=BatchGenerator(train_pairs, shuffle=True),
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=BatchGenerator(valid_pairs, shuffle=False),
        shuffle=True,
    )
    training_seconds = time.perf_counter() - start

    conditions = {
        "experiment_id": args.experiment_id,
        "method": args.method,
        "seed": args.seed,
        "round": args.round_number,
        "fold": args.fold,
        "train_pair_csv": str(args.train_pair_csv),
        "valid_pair_csv": str(args.valid_pair_csv),
        "initial_checkpoint": str(args.initial_checkpoint),
        "initialization_policy": "fixed_AL_0_checkpoint_every_round",
        "train_pair_count": len(train_pairs),
        "valid_pair_count": len(valid_pairs),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "dropout_rate": 0.2,
        "weight_decay": 1e-4,
        "training_seconds": training_seconds,
    }
    pd.DataFrame([conditions]).to_csv(
        output_dir / "experimental_condition.csv", index=False
    )
    with (output_dir / "experimental_condition.json").open(
        "w", encoding="utf-8"
    ) as condition_file:
        json.dump(conditions, condition_file, indent=2, sort_keys=True)
        condition_file.write("\n")

    manifest = {
        "experiment_id": args.experiment_id,
        "method": args.method,
        "seed": args.seed,
        "round": args.round_number,
        "fold": args.fold,
        "status": "complete",
    }
    with (output_dir / "training_manifest.json").open(
        "w", encoding="utf-8"
    ) as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    print(f"Saved trained method model: {output_dir}")


def parse_args():
    """Parse explicit, reproducible training inputs."""
    parser = argparse.ArgumentParser(
        description="Train RankNet from a fixed AL_0 checkpoint and cumulative pairs."
    )
    add_data_root_argument(parser)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--round", dest="round_number", required=True, type=int)
    parser.add_argument("--seed", type=int, default=240119)
    parser.add_argument("--train-pair-csv", required=True, type=Path)
    parser.add_argument("--valid-pair-csv", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.round_number < 1 or args.fold < 1:
        parser.error("--round and --fold must be at least 1.")
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 0:
        parser.error("Invalid training count option.")
    args.experiment_root = (
        args.experiment_root.expanduser()
        if args.experiment_root is not None
        else generated_path(args.data_root, "PairBALDExperiments")
    )
    args.image_dir = (
        args.image_dir.expanduser()
        if args.image_dir is not None
        else generated_path(args.data_root, "all_public_UC_images")
    )
    args.initial_checkpoint = args.initial_checkpoint.expanduser()
    return args


if __name__ == "__main__":
    train(parse_args())
