# Deep Bayesian Active Learning-to-Rank

Code for the paper:

**Deep Bayesian active learning-to-rank with relative annotation for estimation of ulcerative colitis severity**

- Medical Image Analysis, 2024
- DOI: https://doi.org/10.1016/j.media.2024.103262
- arXiv: https://arxiv.org/abs/2409.04952

## Dataset

This repository uses the public **LIMUC (Labeled Images for Ulcerative Colitis) dataset**:

https://doi.org/10.5281/zenodo.5827695

Use the patient-classified image directory included in the LIMUC dataset. On the
laboratory server, its path is:

```text
/data/umeiro0/patient_based_classified_images
```

The dataset itself is not included in this repository.

## Environment

The original experiments reported in the paper were conducted using:

- Ubuntu 18.04
- TensorFlow 1.13.1
- Keras 2.2.4

For this public code release, the reproducible environment uses Python 3.8 and
the standard PyPI `tensorflow==2.4.0` package. GPU execution uses CUDA 11.0 and
cuDNN 8 from the Docker base image. The main compatibility pins are:

- NumPy 1.19.5
- SciPy 1.5.4
- Keras 2.4.3
- h5py 2.10.0
- OpenCV headless 4.5.1.48

The code does not import scikit-learn, so it is intentionally absent. OpenCV is
used only for file I/O and resizing; no GUI API is required.

`pyproject.toml` records the direct dependencies and `uv.lock` records the
complete resolved environment. This environment targets Linux x86_64 because
TensorFlow 2.4.0 does not provide a macOS arm64 wheel.

The released code has been reorganized and updated for public use and is therefore not an exact archival copy of the original experimental code.

For the original experimental settings, please refer to the paper. Hyperparameters in this released implementation can be adjusted as needed for your environment and experimental settings.

## Directory Structure

```text
Deep-Bayesian-Active-Learning-to-Rank/
├── Docker/
│   ├── Dockerfile
│   └── run_build.sh
│
├── run_docker.sh
│
├── Data/
│   └── UC/
│       └── LIMUC/
│           ├── Images/
│           └── Preprocessing/
│               ├── 1_prepare_uc_image_dataset.py
│               ├── 2_create_patient_level_splits.py
│               ├── 3_format_fold_datasets.py
│               └── 4_create_training_only_datasets.py
│
└── Experiments/
    └── Ranknet/
        └── Bayesian/
            └── LIMUC/
                └── Scripts/
                    ├── 1_initial_learning_make_pair.py
                    ├── 2_train_bayesian_ranknet.py
                    ├── 3_predict_score.py
                    ├── 4_calculate_uncertainty.py
                    ├── 5_active_learning_make_pair.py
                    ├── 6_train_bayesian_ranknet_AL.py
                    ├── 7_select_comparison_pairs.py
                    ├── 8_train_pair_acquisition.py
                    ├── 9_predict_pair_experiment.py
                    ├── 10_evaluate_pair_experiment.py
                    ├── check_pair_bald_gpu.py
                    ├── pair_acquisition.py
                    ├── pair_evaluation.py
                    ├── shared_mc_dropout.py
                    ├── bayesian_densenet.py
                    ├── callbacks.py
                    ├── test_balanced_pair_accuracy.py
                    ├── bar_graph.py
                    ├── box_plot.py
                    └── scatter_plot.py
```

Dataset files, active-learning datasets, and experiment results are stored on the
data server rather than in this repository.

## Usage

### 1. Prepare the LIMUC dataset

Download the LIMUC dataset from:

https://doi.org/10.5281/zenodo.5827695

The data root must directly contain the patient directories, each of which contains
the `Mayo 0` through `Mayo 3` directories. The laboratory server already provides
this directory at:

```text
/data/umeiro0/patient_based_classified_images/
```

All scripts accept `--data-root`. They also read `LIMUC_DATA_ROOT`, and finally
fall back to the laboratory path above. Set the environment variable once before
running the pipeline:

```bash
export LIMUC_DATA_ROOT=/data/umeiro0/patient_based_classified_images
```

Generated files are kept below `$LIMUC_DATA_ROOT/generated/`:

```text
generated/
├── all_public_UC_images_original_bmp/
├── all_public_UC_images/
├── all_public_UC_data.csv
├── original_splits/
├── dataset/
├── training_dataset/
├── Add_dataset/
└── Results/
```

This keeps both source and generated data outside the project checkout. You can
override an individual input or output with the more specific options shown by
`--help`.

### 2. Create the uv environment directly on the server

On a Linux x86_64 server, create the exact locked environment from the project
root:

```bash
uv sync --frozen --no-dev
```

GPU execution outside Docker additionally requires CUDA 11.0 and cuDNN 8 to be
available on the server. Without those libraries, this environment can only use
the CPU.

Use `uv run --frozen` for commands in this environment. Do not run `uv sync` on
an Apple Silicon Mac for this legacy environment; build and run the Docker image
instead.

### 3. Build and verify the Docker image

From the project root, run:

```bash
bash Docker/run_build.sh
```

The image build performs a CPU smoke test using generated dummy data. On the
NVIDIA server, also verify that TensorFlow sees the GPU:

```bash
bash Docker/run_smoke_test.sh
```

This GPU check must report at least one device. The host needs an NVIDIA driver
that supports CUDA 11.0 and the NVIDIA Container Toolkit.

### 4. Start the Docker container

From the project root, run:

```bash
bash run_docker.sh
```

The project directory is mounted to `/workdir`. `run_docker.sh` also mounts
`$LIMUC_DATA_ROOT` at the same absolute path inside the container and passes the
environment variable through. Set `LIMUC_DATA_ROOT` before starting the container
when using a different data-server path.

### 5. Prepare the dataset

Inside the Docker container, move to:

```bash
cd /workdir/Data/UC/LIMUC/Preprocessing
```

Run the preprocessing scripts sequentially:

```bash
uv run --frozen --no-sync 1_prepare_uc_image_dataset.py
uv run --frozen --no-sync 2_create_patient_level_splits.py
uv run --frozen --no-sync 3_format_fold_datasets.py
uv run --frozen --no-sync 4_create_training_only_datasets.py
```

To specify the root without an environment variable, pass the same option to every
command, for example:

```bash
uv run --frozen --no-sync 1_prepare_uc_image_dataset.py \
    --data-root /data/umeiro0/patient_based_classified_images
```

### 6. Run Bayesian active learning-to-rank

Move to:

```bash
cd /workdir/Experiments/Ranknet/Bayesian/LIMUC/Scripts
```

To start the active-learning procedure, run scripts 1–6 sequentially:

```bash
uv run --frozen --no-sync 1_initial_learning_make_pair.py
uv run --frozen --no-sync 2_train_bayesian_ranknet.py
uv run --frozen --no-sync 3_predict_score.py <AL_0_result_date>
uv run --frozen --no-sync 4_calculate_uncertainty.py <AL_0_result_date>
uv run --frozen --no-sync 5_active_learning_make_pair.py <AL_0_result_date> --iteration 1
uv run --frozen --no-sync 6_train_bayesian_ranknet_AL.py --iteration 1 --initial-result-date <AL_0_result_date>
```

After completing script 6, repeat scripts 3–6 for each subsequent active-learning iteration. For example, for iteration 2:

```bash
uv run --frozen --no-sync 3_predict_score.py <AL_1_result_date>
uv run --frozen --no-sync 4_calculate_uncertainty.py <AL_1_result_date>
uv run --frozen --no-sync 5_active_learning_make_pair.py <AL_1_result_date> --iteration 2
uv run --frozen --no-sync 6_train_bayesian_ranknet_AL.py --iteration 2 --initial-result-date <AL_0_result_date>
```

Here, `<AL_0_result_date>` refers to the result directory from the initial training, and `<AL_1_result_date>` refers to the result directory from the first active-learning iteration.

For later iterations, replace `<AL_1_result_date>` with the result directory from the immediately preceding iteration and increment `--iteration` accordingly.

Continue this cycle until the desired active-learning iteration is reached.

Every script in steps 1–6 accepts the same `--data-root` option. When
`LIMUC_DATA_ROOT` is exported, the commands above automatically read preprocessed
data, Active Learning CSV files, and results from the shared `generated/`
directory. No current-working-directory-dependent data path is used.

In the experiments reported in the paper, the number of active-learning iterations was set to `K = 6`.

## Pair-BALD comparison-partner experiments

The additional pipeline keeps the existing image-level UBS selection and compares
four partner-selection methods under the same new-pair budget:

- `original`: random partners from all cumulatively selected images
- `random`: random partners from all fold-specific training images
- `predictive_entropy`: maximum Bernoulli predictive entropy over the same full pool
- `pair_bald`: maximum MC estimate of mutual information over the same full pool

The theory, the exact correspondence to Houlsby et al., the confirmed behavior of
the released scripts, and the approximation limits are described in
[`docs/pair_bald_experiments.md`](docs/pair_bald_experiments.md).

All inputs below are explicit so every method can share the same saved AL_0 model,
initial pairs, validation pairs, and initial anchors. Set them to existing files on
the server; do not copy those artifacts into this repository.

```bash
cd /workdir/Experiments/Ranknet/Bayesian/LIMUC/Scripts

export LIMUC_DATA_ROOT=/data/umeiro0/patient_based_classified_images
EXPERIMENT_ID=pair_bald_main
FOLD=1
SEED=240119
INITIAL_CHECKPOINT=/absolute/path/to/fold_1/weights/epochXXXX-X.XXXX.h5
INITIAL_TRAIN_PAIRS=/absolute/path/to/1_train_pair_RBS.csv
INITIAL_VALID_PAIRS=/absolute/path/to/1_valid_pair_RBS.csv
INITIAL_ANCHORS=/absolute/path/to/1_train_RBS.csv
AL0_TRAIN_UNCERTAINTY=/absolute/path/to/train_mean_score_and_uncertainty.csv
SHARED_AL0_CACHE=/data/umeiro0/patient_based_classified_images/generated/PairBALDExperiments/${EXPERIMENT_ID}/shared_cache/fold_${FOLD}_al0.npz
```

First run the small A100 check. It uses generated tensors, loads the existing HDF5
checkpoint, verifies masks across image chunks, and confirms that BatchNormalization
statistics do not change. The check uses equal-sized chunks, matching acquisition
inference, which pads the final partial chunk and discards its padded scores.

```bash
uv run --frozen --no-sync check_pair_bald_gpu.py \
    --checkpoint "${INITIAL_CHECKPOINT}" \
    --cuda-visible-devices 0
```

Run round 1 separately for each method. Use `cumulative_selected` for `original`;
use `full_train` for the other three methods. Supplying the same checkpoint and
cache also records entropy/BALD diagnostics for the random methods.

```bash
METHOD=pair_bald
CANDIDATE_SCOPE=full_train

uv run --frozen --no-sync 7_select_comparison_pairs.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --method "${METHOD}" \
    --candidate-scope "${CANDIDATE_SCOPE}" \
    --fold "${FOLD}" \
    --round 1 \
    --seed "${SEED}" \
    --checkpoint "${INITIAL_CHECKPOINT}" \
    --mc-cache "${SHARED_AL0_CACHE}" \
    --previous-pair-csv "${INITIAL_TRAIN_PAIRS}" \
    --anchor-history-csv "${INITIAL_ANCHORS}" \
    --uncertainty-csv "${AL0_TRAIN_UNCERTAINTY}" \
    --partner-cap 0 \
    --cuda-visible-devices 0
```

Replace `METHOD` with `original`, `random`, or `predictive_entropy`; set
`CANDIDATE_SCOPE=cumulative_selected` for `original`. Use the same `--pair-budget`
for every method. The command stops if that budget cannot give every selected
anchor at least one new valid pair. Omitting `--pair-budget` sets it to the number
of selected anchors. The default `--mc-samples` is 30. The additional `selected`
scope restricts partners to the current round's anchors for diagnostics.

Train every method from the same saved AL_0 checkpoint. The validation pair CSV is
fixed rather than actively expanded.

```bash
SELECTION_DIR=${LIMUC_DATA_ROOT}/generated/PairBALDExperiments/${EXPERIMENT_ID}/rounds/${METHOD}/seed_${SEED}/round_1/fold_${FOLD}

uv run --frozen --no-sync 8_train_pair_acquisition.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --method "${METHOD}" \
    --fold "${FOLD}" \
    --round 1 \
    --seed "${SEED}" \
    --train-pair-csv "${SELECTION_DIR}/cumulative_pairs.csv" \
    --valid-pair-csv "${INITIAL_VALID_PAIRS}" \
    --initial-checkpoint "${INITIAL_CHECKPOINT}" \
    --cuda-visible-devices 0
```

Choose the actual saved best `.h5` file from that training directory explicitly,
then create the legacy-format 30-sample prediction CSV and the mean/variance summary.
The `test` split contains the fold's unseen patients; the `train` summary supplies
the next round's unchanged UBS anchor rule.

```bash
ROUND_CHECKPOINT=/absolute/path/to/the/method/round_1/fold_1/weights/epochXXXX-X.XXXX.h5
PREDICTION_DIR=${LIMUC_DATA_ROOT}/generated/PairBALDExperiments/${EXPERIMENT_ID}/predictions/${METHOD}/seed_${SEED}/round_1/fold_${FOLD}

uv run --frozen --no-sync 9_predict_pair_experiment.py \
    --checkpoint "${ROUND_CHECKPOINT}" \
    --output-dir "${PREDICTION_DIR}" \
    --fold "${FOLD}" \
    --seed "${SEED}" \
    --datatypes train,test \
    --cuda-visible-devices 0
```

For round 2, use that method's own round-1 checkpoint and training uncertainty,
plus its prior cumulative pairs and anchors. Training still starts from the fixed
AL_0 checkpoint, matching the released script 6.

```bash
uv run --frozen --no-sync 7_select_comparison_pairs.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --method "${METHOD}" \
    --candidate-scope "${CANDIDATE_SCOPE}" \
    --fold "${FOLD}" \
    --round 2 \
    --seed "${SEED}" \
    --checkpoint "${ROUND_CHECKPOINT}" \
    --previous-pair-csv "${SELECTION_DIR}/cumulative_pairs.csv" \
    --anchor-history-csv "${SELECTION_DIR}/cumulative_anchors.csv" \
    --uncertainty-csv "${PREDICTION_DIR}/train_mean_score_and_uncertainty.csv" \
    --partner-cap 0 \
    --cuda-visible-devices 0
```

To compare selection only, prepare one CSV with a `filename` column and invoke each
method with the same checkpoint, explicit anchors, prior pairs, and cache. No model
training is performed in diagnostic mode.

```bash
DIAGNOSTIC_ANCHORS=/absolute/path/to/shared_anchor_images.csv

uv run --frozen --no-sync 7_select_comparison_pairs.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --method pair_bald \
    --candidate-scope full_train \
    --fold "${FOLD}" \
    --round 1 \
    --seed "${SEED}" \
    --checkpoint "${INITIAL_CHECKPOINT}" \
    --mc-cache "${SHARED_AL0_CACHE}" \
    --previous-pair-csv "${INITIAL_TRAIN_PAIRS}" \
    --anchor-history-csv "${INITIAL_ANCHORS}" \
    --anchor-csv "${DIAGNOSTIC_ANCHORS}" \
    --diagnostic \
    --cuda-visible-devices 0
```

Evaluate a test prediction summary and update one comparison CSV indexed by method,
fold, seed, round, and cumulative pair count:

```bash
COMPARISON_CSV=${LIMUC_DATA_ROOT}/generated/PairBALDExperiments/${EXPERIMENT_ID}/performance.csv

uv run --frozen --no-sync 10_evaluate_pair_experiment.py \
    --prediction-summary-csv "${PREDICTION_DIR}/test_mean_score_and_uncertainty.csv" \
    --cumulative-pair-csv "${SELECTION_DIR}/cumulative_pairs.csv" \
    --output-csv "${COMPARISON_CSV}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --method "${METHOD}" \
    --fold "${FOLD}" \
    --seed "${SEED}" \
    --round 1
```

Selection, training, and prediction refuse to overwrite existing destinations. Add
`--resume` to the same command to accept an already completed matching run. An
incomplete directory is rejected so a restart cannot silently double-count pairs or
budget; correct the partial destination explicitly before retrying.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{KADOTA2024103262,
  title   = {Deep Bayesian active learning-to-rank with relative annotation for estimation of ulcerative colitis severity},
  journal = {Medical Image Analysis},
  volume  = {97},
  pages   = {103262},
  year    = {2024},
  issn    = {1361-8415},
  doi     = {10.1016/j.media.2024.103262},
  url     = {https://www.sciencedirect.com/science/article/pii/S1361841524001877},
  author  = {Takeaki Kadota and Hideaki Hayashi and Ryoma Bise and Kiyohito Tanaka and Seiichi Uchida}
}
```
