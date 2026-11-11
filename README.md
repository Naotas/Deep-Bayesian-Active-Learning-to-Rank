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
                    ├── 11_run_all_folds.py
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
anchor at least one new valid pair. Omitting `--pair-budget` sets it to the
requested anchor count, before any final-round adjustment described below.
The default `--mc-samples` is 30. The additional `selected`
scope restricts partners to the current round's anchors for diagnostics.

When using the automatic `--anchor-rate`, integer rounding can leave fewer new
anchors than the usual per-round count near the end of acquisition. That last
round selects all remaining anchors, still ordered by image uncertainty, but
**keeps the comparison budget unchanged**. For example, if the rate requests 339
anchors and only 333 remain, select those 333 and acquire 339 new pairs. Every
anchor receives a pair in the first pass; the remaining six comparisons follow
the existing partner-selection rule. Partners still come from the configured
candidate scope, and self-pairs, acquired pairs, reverse duplicates, and partner
caps remain enforced. Earlier-round anchors are not added to the new-anchor
set, and acquired pairs are not reacquired to fill the budget.

Explicit `--anchor-count` requests remain strict and are never shortened.
With `--anchor-csv`, the default budget is the number of distinct supplied
anchors. An explicit `--pair-budget` always remains unchanged. The selector
stops if no new anchors remain or if the exact pair budget and coverage
constraints cannot be satisfied. The rate-based adjustment is printed and
recorded in `experiment_config.json` and `.csv` as `requested_anchor_count`,
`anchor_count`, `remaining_anchor_count_before_selection`, and
`anchor_count_adjusted`, alongside the actual `pair_budget`.

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

### Start a separate experiment at 10% and acquire through 50%

The existing AL_0 checkpoint was trained on 20% of each split, so a 10% run
needs new initial pairs and new initial weights. The initial pair generator now
accepts `--initial-rate`; the initial trainer checks its input pair counts and
records the actual configured rate. Keep a new `--al-id`, results root, and
experiment ID so the earlier 20% run remains intact. The commands below sample
10% of both training and validation images in each fold, train all five initial
models, then acquire 5% of each fold's training images per round for eight
rounds. This reaches a **nominal** 50% budget (`10 + 8 × 5`). Integer rounding
can make actual counts differ slightly, and comparison partners can appear in
the pairs before being selected as anchors.

Run these commands on the GPU server from the scripts directory. Use a GPU
assigned to you. The initial trainer prints the exact result directory after
training; copy that path into `INITIAL_MODEL_ROOT` before running script 11.

```bash
cd /home/mil/n-inoue/Deep-Bayesian-Active-Learning-to-Rank/Experiments/Ranknet/Bayesian/LIMUC/Scripts

export LIMUC_DATA_ROOT=/data/umeiro0/patient_based_classified_images
AL_ID=LIMUC_AL_10pct
INITIAL_RESULTS_ROOT=${LIMUC_DATA_ROOT}/generated/Results_AL0_10pct
INITIAL_PAIRS_ROOT=${LIMUC_DATA_ROOT}/generated/Add_dataset/${AL_ID}/AL_0
EXPERIMENT_ID=pair_bald_10_to_50_seed_240122
SEED=240122
GPU_LIST=0

uv run --frozen --no-sync 1_initial_learning_make_pair.py \
    --data-root "${LIMUC_DATA_ROOT}" \
    --al-id "${AL_ID}" --initial-rate 0.10 --folds 1,2,3,4,5

df -h "${LIMUC_DATA_ROOT}"
nvidia-smi
uv run --frozen --no-sync 2_train_bayesian_ranknet.py \
    --data-root "${LIMUC_DATA_ROOT}" \
    --al-id "${AL_ID}" --initial-rate 0.10 --folds 1,2,3,4,5 \
    --results-root "${INITIAL_RESULTS_ROOT}" \
    --cuda-visible-devices "${GPU_LIST}"
```

After initial training, set the printed directory containing `fold_1/weights`
through `fold_5/weights` and check the paths before starting acquisition:

```bash
INITIAL_MODEL_ROOT=/absolute/path/printed/by/initial/training

uv run --frozen --no-sync 11_run_all_folds.py \
    --data-root "${LIMUC_DATA_ROOT}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --initial-model-root "${INITIAL_MODEL_ROOT}" \
    --initial-pairs-root "${INITIAL_PAIRS_ROOT}" \
    --initial-rate 0.10 --anchor-rate 0.05 \
    --gpus "${GPU_LIST}" --folds 1,2,3,4,5 \
    --methods original,random,predictive_entropy,pair_bald \
    --seed "${SEED}" --rounds 8 --dry-run

uv run --frozen --no-sync 11_run_all_folds.py \
    --data-root "${LIMUC_DATA_ROOT}" \
    --experiment-id "${EXPERIMENT_ID}" \
    --initial-model-root "${INITIAL_MODEL_ROOT}" \
    --initial-pairs-root "${INITIAL_PAIRS_ROOT}" \
    --initial-rate 0.10 --anchor-rate 0.05 \
    --gpus "${GPU_LIST}" --folds 1,2,3,4,5 \
    --methods original,random,predictive_entropy,pair_bald \
    --seed "${SEED}" --rounds 8 --resume
```

The dry run checks that the initial pair files, checkpoint conditions, and
training split sizes agree with 10%; it performs no GPU work. The production
run writes to `generated/PairBALDExperiments/${EXPERIMENT_ID}`. On a later
invocation, keep the same arguments and `--resume`; completed stages are
reused. To plot this experiment, provide its initial test evaluation data to
script 12 and set `--initial-labeling-percent 10 --round-labeling-percent 5
--max-round 8`. Its round-8 point is the nominal 50% point.

### Run all five folds together

After completing initial learning for folds 1–5, use `11_run_all_folds.py` to
run pair selection, training, prediction, and evaluation from one terminal or
tmux session. The default is all five folds and all four acquisition methods.
With five GPUs, each fold runs on its own GPU; the four methods run sequentially
inside each fold. This starts five GPU processes at a time rather than twenty.
Supplying fewer GPUs queues the remaining folds, without sharing a GPU between
workers. Use GPU indexes or full GPU UUIDs allocated to your job.

If other users are running jobs, list only the GPUs you may use; `--gpus` specifies
device IDs, not a GPU count. For example, if GPUs 2 and 7 are available, use
`--gpus 2,7`. GPU 2 processes folds 1, 3, and 5 in order, while GPU 7 processes
folds 2 and 4. All five folds still run. The runner does not automatically select
unoccupied GPUs or add newly available GPUs during a run.

| Available GPUs | Example `--gpus` | Simultaneous fold workers |
|---|---|---|
| 4 | `0,2,5,7` | 4; fold 5 waits behind fold 1 |
| 3 | `1,4,6` | 3; folds 4 and 5 wait |
| 2 | `2,7` | 2; folds 3–5 wait |
| 1 | `6` | 1; folds 1–5 run sequentially |

Check GPU usage with `nvidia-smi` and follow the server's allocation rules before
choosing IDs. If no GPU is available, start the GPU stages later; the runner does
not switch them to CPU. GPU assignments can be changed between invocations,
including when using `--resume` to skip completed matching steps.

Set `INITIAL_MODEL_ROOT` to the existing AL_0 result directory that contains
`fold_1/weights` through `fold_5/weights`. The runner selects the highest saved
epoch in each fold. The existing callback saves only validation-loss improvements,
so this is the last saved best model, including when filenames have equal rounded
losses. Initial models and initial pairs are never retrained or overwritten.

```bash
cd /workdir/Experiments/Ranknet/Bayesian/LIMUC/Scripts

export LIMUC_DATA_ROOT=/data/umeiro0/patient_based_classified_images
INITIAL_MODEL_ROOT=/absolute/path/to/the/saved_AL_0_result_directory
INITIAL_PAIRS_ROOT=${LIMUC_DATA_ROOT}/generated/Add_dataset/LIMUC_AL_001/AL_0
EXPERIMENT_ID=pair_bald_all_folds
GPU_LIST=0,1,2,3,4
```

Replace `INITIAL_MODEL_ROOT` with your actual saved result directory. For standard
initial training it is under `${LIMUC_DATA_ROOT}/generated/Results/` and ends in
`_AL_0`. Do not use a single fold's `weights` directory for this argument.

First preview the commands. This verifies the saved initial files without loading
TensorFlow, requesting GPUs, or creating results. Future checkpoint paths appear
as `<best-checkpoint>.h5` until those models have been trained.

```bash
uv run --frozen --no-sync 11_run_all_folds.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --initial-model-root "${INITIAL_MODEL_ROOT}" \
    --initial-pairs-root "${INITIAL_PAIRS_ROOT}" \
    --gpus "${GPU_LIST}" \
    --rounds 1 \
    --dry-run
```

Run the complete first round for all folds and methods:

```bash
uv run --frozen --no-sync 11_run_all_folds.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --initial-model-root "${INITIAL_MODEL_ROOT}" \
    --initial-pairs-root "${INITIAL_PAIRS_ROOT}" \
    --gpus "${GPU_LIST}" \
    --rounds 1
```

If initial uncertainty CSVs are not supplied, the runner predicts the training
images once per fold from the saved AL_0 checkpoint. These predictions are saved
under the new experiment's `initial_predictions/seed_<seed>/fold_<fold>` directory
and shared across methods. To reuse existing uncertainty CSVs instead, add
`--initial-uncertainty-root /absolute/path/to/the/prediction_result_directory`;
that directory must contain `fold_1/train_mean_score_and_uncertainty.csv` through
`fold_5/train_mean_score_and_uncertainty.csv`. Keep this option the same when
resuming the experiment.

You can run each stage separately across all five folds. Each command completes
before the next one should be started. `--stage train` is useful when pair
selection for all folds is already complete:

```bash
uv run --frozen --no-sync 11_run_all_folds.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --initial-model-root "${INITIAL_MODEL_ROOT}" \
    --initial-pairs-root "${INITIAL_PAIRS_ROOT}" \
    --gpus "${GPU_LIST}" --stage select

uv run --frozen --no-sync 11_run_all_folds.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --initial-model-root "${INITIAL_MODEL_ROOT}" \
    --initial-pairs-root "${INITIAL_PAIRS_ROOT}" \
    --gpus "${GPU_LIST}" --stage train

uv run --frozen --no-sync 11_run_all_folds.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --initial-model-root "${INITIAL_MODEL_ROOT}" \
    --initial-pairs-root "${INITIAL_PAIRS_ROOT}" \
    --gpus "${GPU_LIST}" --stage predict

uv run --frozen --no-sync 11_run_all_folds.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --initial-model-root "${INITIAL_MODEL_ROOT}" \
    --initial-pairs-root "${INITIAL_PAIRS_ROOT}" \
    --gpus "${GPU_LIST}" --stage evaluate
```

Evaluation uses fold-local files under `fold_performance/` while the workers are
running. After they finish, the runner merges the rows into `performance.csv`,
preserving existing rows and updating matching method/fold/seed/round identities.
Do not run manual evaluations that write the same `performance.csv` while this
runner is active. Evaluation-only execution does not require a GPU.

Set `--rounds 3` to execute rounds 1–3, or use the following command to continue
from completed round 1 through rounds 2–3:

```bash
uv run --frozen --no-sync 11_run_all_folds.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --initial-model-root "${INITIAL_MODEL_ROOT}" \
    --initial-pairs-root "${INITIAL_PAIRS_ROOT}" \
    --gpus "${GPU_LIST}" \
    --start-round 2 --rounds 3
```

Later rounds use each method's own previous model and uncertainty for selection.
Training always loads the same saved AL_0 checkpoint, as in single-fold execution.
`--pair-budget`, `--anchor-count`, `--anchor-rate`, `--partner-cap`, `--mc-samples`,
`--image-batch-size`, `--candidate-chunk-size`, `--batch-size`, `--epochs`,
`--patience`, and `--learning-rate` are applied identically to every fold/method.
The default budgets remain fold-dependent because the default anchor count is 5%
of that fold's training images; methods within a fold share the same budget.
The final-round anchor adjustment does not reduce this budget, so the same
rule applies to all four methods. Completed rounds are preserved by `--resume`.
Use `--methods pair_bald` for only that method, or `--folds 1,2` for a subset.

For completed outputs, add `--resume` to the same command:

```bash
uv run --frozen --no-sync 11_run_all_folds.py \
    --experiment-id "${EXPERIMENT_ID}" \
    --initial-model-root "${INITIAL_MODEL_ROOT}" \
    --initial-pairs-root "${INITIAL_PAIRS_ROOT}" \
    --gpus "${GPU_LIST}" \
    --rounds 1 --resume
```

The runner checks saved conditions before skipping completed work. It rejects
incomplete outputs, and it does not resume training midway through an epoch.
Preserve interrupted output directories by moving them aside before retrying.
A failed stage stops that fold; other folds finish, and the runner exits with a
failure status. Separate step logs and the resolved initial checkpoint paths are
saved under `runner_logs/<invocation>/`. The runner prevents a second batch runner
from starting in the same experiment while the first is active.

If you already chose specific AL_0 checkpoints for earlier comparisons, use those
same files. Replace `--initial-model-root` with
`--initial-checkpoints-csv /absolute/path/to/initial_checkpoints.csv`. This CSV has
the following format; use the actual existing paths, not these placeholders:

```csv
fold,checkpoint
1,/absolute/path/to/fold_1/weights/epochXXXX-X.XXXX.h5
2,/absolute/path/to/fold_2/weights/epochXXXX-X.XXXX.h5
3,/absolute/path/to/fold_3/weights/epochXXXX-X.XXXX.h5
4,/absolute/path/to/fold_4/weights/epochXXXX-X.XXXX.h5
5,/absolute/path/to/fold_5/weights/epochXXXX-X.XXXX.h5
```

For a small server GPU check, first run `check_pair_bald_gpu.py` with one existing
AL_0 checkpoint as described above. To check the batch workflow in a separate
experiment, use `--experiment-id pair_bald_all_folds_smoke --methods pair_bald
--anchor-count 2 --pair-budget 2 --mc-samples 3 --epochs 1`. This still predicts
the full fold pools, so it is not a small-data benchmark. Keep the production
defaults for the main experiment.

### Plot Figure 4/6-style comparisons from saved results

`12_plot_pair_experiments.py` creates tables and figures for the four partner
selection methods. It reads CSVs only: no TensorFlow import, GPU, prediction,
training, or changes to saved experiment results are needed. Run it after the
requested rounds have finished, not while their CSVs are being written.

The comparison uses five-fold averaging as in Sections 5.1.5 and 5.1.6 of the
[paper](https://arxiv.org/html/2409.04952v1), with the following explicit
conventions. Figure 6 uses percentages instead of the paper's raw image counts:

- Figure 4: two line charts, `overall` and `neighboring`. Each point is the
  equally weighted mean test accuracy of folds 1–5. The shaded band is mean ±
  one **sample standard deviation** across folds (`ddof=1`), in percentage
  points. This added band is not MC Dropout uncertainty, variance, standard
  error, or a confidence interval. Same-grade pairs remain excluded from these
  directional accuracies by the existing evaluator.
- The horizontal axis uses the **configured nominal** labeling ratio,
  `20 + 5 * iteration` percent: iterations 0, 3, 6, and 8 correspond to 20%,
  35%, 50%, and 60%. It is not computed from unique-image coverage. Actual
  cumulative pair counts are preserved in the exported tables. Custom
  experiments can explicitly set `--initial-labeling-percent` and
  `--round-labeling-percent`; these options change labels, not the experiment.
- Figure 6: four grouped bar charts for iterations 0, 3, 6, and 8. Each Mayo
  grade has four method bars. Heights are the equally weighted five-fold mean
  **proportion of unique images (%)**. For each fold, divide a grade's unique
  image count by the total unique images across all four grades, then average
  these fractions across folds and multiply by 100. Do not pool counts before
  normalization: folds with more images must not receive greater weight. Later
  iterations read the saved `mayo_grade_distribution.csv` in each
  selection directory. These files count both endpoints of cumulative training
  pairs, deduplicated within each fold; partner-only images are included. This
  distinction matters when partners can come from the entire training split.
  All four charts use a 0–100% y-axis scale, and the four grade bars sum to 100%
  for each method in each chart. The tables retain image counts as well as
  fractions on a 0–1 scale and their means and sample SDs.
- Iteration 0 shares one initial model and initial training pair set across all
  four methods. Initial grade counts are reconstructed from the saved initial
  pair CSVs using the same endpoint-deduplication rule.
- Colors are blue (`original`), red (`random`), yellow (`predictive_entropy`),
  and green (`pair_bald`). An optional black dotted horizontal reference uses
  saved `random` results at the configured nominal 100% budget; it is never
  estimated from earlier rounds or invented when results are missing.

The required inputs are:

1. `performance.csv` from the all-fold runner, containing every requested round,
   all four methods, and all five folds for the chosen seed.
2. `initial_performance.csv` from evaluation of the initial models on **test**
   images, with `method=initial` and `round=0` for all five folds. The automatic
   `initial_predictions/` directory created by script 11 contains training-only
   predictions and is not a substitute for these test results.
3. The saved initial training pair CSVs and the requested iterations' saved
   `mayo_grade_distribution.csv` files with completed selection manifests.

If the initial test predictions already exist under
`initial_test_predictions/seed_240119/fold_<fold>/`, evaluate them with the existing
script 10 as follows. This step does not predict again or require a GPU. If they
were saved elsewhere, set `INITIAL_TEST_ROOT` to that actual directory first.
If test predictions do not exist yet, generate them from the saved initial
checkpoints using script 9 before this step; do not substitute a later model.

```bash
cd /home/mil/n-inoue/Deep-Bayesian-Active-Learning-to-Rank/Experiments/Ranknet/Bayesian/LIMUC/Scripts

export LIMUC_DATA_ROOT=/data/umeiro0/patient_based_classified_images
EXPERIMENT_DIR=${LIMUC_DATA_ROOT}/generated/PairBALDExperiments/pair_bald_all_folds
INITIAL_PAIRS_ROOT=${LIMUC_DATA_ROOT}/generated/Add_dataset/LIMUC_AL_001/AL_0
SEED=240119
INITIAL_TEST_ROOT=${EXPERIMENT_DIR}/initial_test_predictions/seed_${SEED}

for FOLD in 1 2 3 4 5; do
    uv run --frozen --no-sync 10_evaluate_pair_experiment.py \
        --prediction-summary-csv "${INITIAL_TEST_ROOT}/fold_${FOLD}/test_mean_score_and_uncertainty.csv" \
        --cumulative-pair-csv "${INITIAL_PAIRS_ROOT}/fold_${FOLD}/RBS/${FOLD}_train_pair_RBS.csv" \
        --output-csv "${EXPERIMENT_DIR}/initial_performance.csv" \
        --experiment-id pair_bald_all_folds \
        --method initial --fold "${FOLD}" --seed "${SEED}" --round 0 || break
done
```

If the initial evaluation CSV is already complete, skip the preceding loop.
Create the figures and tables with one command:

```bash
uv run --frozen --no-sync 12_plot_pair_experiments.py \
    --data-root "${LIMUC_DATA_ROOT}" \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --initial-pairs-root "${INITIAL_PAIRS_ROOT}" \
    --initial-performance-csv "${EXPERIMENT_DIR}/initial_performance.csv" \
    --seed "${SEED}" \
    --max-round 8 \
    --bar-rounds 0,3,6,8 \
    --output-dir "${EXPERIMENT_DIR}/plots_fig4_fig6_v1"
```

The command requires all five folds, rejects duplicate evaluation rows and
unequal method budgets within a fold/round, and never interpolates missing
results. If round 8 is not complete yet, wait for it or explicitly request only
available rounds, for example `--max-round 6 --bar-rounds 0,3,6`. A missing-input
error names the CSV or the missing method/round/fold combinations. The report
directory is created only after all input tables have been validated. Existing
report directories are never overwritten; use a new name such as
`plots_fig4_fig6_v2` for another run.

#### Add the seed 240121 random-method 100% reference

If `random` has completed through nominal 100% for all five folds with seed
240121, pass its saved `performance.csv` to the same plotter. With the default
20% initial budget and 5% per round, the reference is **round 16**, independently
of `--max-round 8` for the four method curves. The other methods do not need
rounds 9–16. For example, plot the seed 240121 curves with its own reference:

```bash
export LIMUC_DATA_ROOT=/data/umeiro0/patient_based_classified_images
EXPERIMENT_DIR=${LIMUC_DATA_ROOT}/generated/PairBALDExperiments/pair_bald_all_folds_seed_240121
REFERENCE_EXPERIMENT_DIR=${LIMUC_DATA_ROOT}/generated/PairBALDExperiments/pair_bald_all_folds_seed_240121
INITIAL_PAIRS_ROOT=${LIMUC_DATA_ROOT}/generated/Add_dataset/LIMUC_AL_001/AL_0
SEED=240121

uv run --frozen --no-sync 12_plot_pair_experiments.py \
    --data-root "${LIMUC_DATA_ROOT}" \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --initial-pairs-root "${INITIAL_PAIRS_ROOT}" \
    --seed "${SEED}" \
    --max-round 8 \
    --bar-rounds 0,3,6,8 \
    --reference-performance-csv "${REFERENCE_EXPERIMENT_DIR}/performance.csv" \
    --reference-experiment-id pair_bald_all_folds_seed_240121 \
    --reference-seed 240121 \
    --output-dir "${EXPERIMENT_DIR}/plots_fig4_fig6_random100_v1"
```

The `overall` line is the equally weighted mean of `all_accuracy` across the
five reference folds; the `neighboring` line uses `neighbor_accuracy` instead.
Both are converted to percentages. The legend reads `all data`; the exact seed,
method, round, and accuracies are retained in the reference CSVs and metadata.
The reference has no shaded band and is included in the y-axis limits
even if it lies above or below the displayed curves. Figure 6 is unchanged.
This is an observed random-method reference at nominal 100%, **not a theoretical
upper bound, an all-possible-pairs result, or proof that all training images
have appeared**.

To reuse this same reference for seed 240119 or 240120 curves, change
`EXPERIMENT_DIR` and `SEED` to the desired curve experiment but keep the reference
path and reference identity unchanged. A copy of the reference CSV can also be
used, provided its rows retain the matching `experiment_id` and `seed`.
The defaults for the reference identity are
`--reference-experiment-id pair_bald_all_folds_seed_240121 --reference-seed 240121`.
Different experiment names or reference seeds require explicit overrides.
The plotter requires exactly one valid result for each reference fold 1–5;
missing or duplicate folds and invalid accuracies stop report creation.
No training, prediction, or changes to the source CSVs occur. Omitting
`--reference-performance-csv` keeps the original plots without a reference.

The output directory contains:

| File | Contents |
| --- | --- |
| `figure4_overall.png`, `.pdf` | Overall accuracy curves with SD bands |
| `figure4_neighboring.png`, `.pdf` | Neighboring accuracy curves with SD bands |
| `figure6_iteration_0.png`, `.pdf` | Shared initial grade proportions (%) |
| `figure6_iteration_3.png`, `.pdf` | Cumulative grade proportions (%) at iteration 3 |
| `figure6_iteration_6.png`, `.pdf` | Cumulative grade proportions (%) at iteration 6 |
| `figure6_iteration_8.png`, `.pdf` | Cumulative grade proportions (%) at iteration 8 |
| `figure4_fold_data.csv` | Per-fold results, nominal ratios, actual pair budgets, and accuracies |
| `figure4_summary.csv` | Five-fold means and sample SDs; accuracy values are percentages |
| `figure4_reference_fold_data.csv` | Optional: five reference folds, actual pair counts, accuracies, and source CSV |
| `figure4_reference_summary.csv` | Optional: reference means and sample SDs; only means are drawn as dotted lines |
| `figure6_fold_data.csv` | Per-fold counts, fractions, and source CSV paths |
| `figure6_summary.csv` | Five-fold count/fraction means and sample SDs |
| `plot_config.json` | Input paths, seed, colors, and aggregation definitions |

Check that `fold_count` is 5 in both summary CSVs. At iteration 0 all methods
must have the same accuracy and grade counts; at later iterations they may
differ. The plotter does not pool results from different seeds or experiments.
Override `--performance-csv` or `--initial-performance-csv` if those tables are
stored outside the default experiment directory.

### One-shot random all-data control (N images, N comparisons)

`13_make_all_data_pairs.py` creates a separate `all_data` control without
running active learning. For each fold, all N training images are used exactly
once as anchors; a partner is sampled uniformly from valid training images.
Self-pairs and reverse duplicates are excluded. Partner repetitions across
different comparisons are allowed: partners do not form a permutation. The
result has exactly N unordered-distinct comparisons. Mayo labels are attached
only after selection. At least three images are required. Tiny-pool dead ends
restart the whole pass, with a maximum of 100 attempts; infeasible runs stop
rather than duplicate pairs. Memory is linear in N, with expected linear
sampling work; no N-by-N rankings or MC cache are generated.

The following commands compare pair construction with the completed random
round-16 experiment: they reuse its saved AL_0 checkpoint, fixed validation
pairs, seed, training hyperparameters, and MC sample count. They do **not**
change the learning rate to the paper's setting while changing the pairs.
This is a controlled diagnostic, not a claim of exact reproduction of the
paper's baseline. N one-shot pairs also need not equal the rounded AL budget
(for example, N+6). Moreover, an AL_0 warm start has already used initial
annotations that may be absent from the new N pairs; the unique annotation
cost of the entire warm-start experiment is the union of these pair sets,
not automatically N. Do not present this as a budget-matched AL comparison.

#### 1. Set separate paths and generate all five folds

Run on the GPU server from the scripts directory. These are new output names;
existing AL_0 data and active-learning results are not overwritten. No shell
script file is required.

```bash
cd /home/mil/n-inoue/Deep-Bayesian-Active-Learning-to-Rank/Experiments/Ranknet/Bayesian/LIMUC/Scripts

export LIMUC_DATA_ROOT=/data/umeiro0/patient_based_classified_images
SEED=240121
SOURCE_EXPERIMENT_DIR=${LIMUC_DATA_ROOT}/generated/PairBALDExperiments/pair_bald_all_folds_seed_240121
SOURCE_ROUND=16
ALL_DATA_ID=all_data_one_shot_seed_240121
ALL_DATA_DIR=${LIMUC_DATA_ROOT}/generated/PairBALDExperiments/${ALL_DATA_ID}

uv run --frozen --no-sync 13_make_all_data_pairs.py \
    --data-root "${LIMUC_DATA_ROOT}" \
    --experiment-id "${ALL_DATA_ID}" \
    --seed "${SEED}" --folds 1,2,3,4,5 --resume
```

The generator is CPU-only. The tables are under
`rounds/all_data/seed_240121/round_1/fold_<fold>/`. Here `round_1` is a single
full-budget run identifier, not AL's 25% point. `cumulative_pairs.csv` contains
only the new N one-shot pairs; it does not append initial pairs.
`pair_summary.csv` must report `cumulative_pair_count == unique_image_count == N`.
`--resume` checks source and output hashes, skips matching completed folds, and
rejects incomplete or modified outputs. It never silently repairs partial data.

#### 2. Train, predict, and evaluate sequentially on one allocated GPU

Stop other writers to these experiment directories first. Check `df -h` for
sufficient free space for five training runs and `nvidia-smi` for a GPU allocated
to you. Replace the prompted GPU value with its index or UUID. The source
experiment must contain completed random round-16 training and prediction
manifests for all five folds, plus their `experimental_condition.json` files.
No new initial-model path is guessed: it is read from each source condition.
If round 16 is not complete, explicitly set `SOURCE_ROUND=8` to reuse completed
round-8 conditions instead; this does not create AL round 16. Check that those
conditions are the ones intended for the comparison. There is no silent fallback.

```bash
nvidia-smi
df -h "${ALL_DATA_DIR}"
printf 'GPU index or UUID: '
read -r GPU_ID

uv run --frozen --no-sync - \
    "${SOURCE_EXPERIMENT_DIR}" "${ALL_DATA_DIR}" "${GPU_ID}" "${SEED}" "${SOURCE_ROUND}" <<'PY'
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

source, baseline = map(Path, sys.argv[1:3])
gpu, seed = sys.argv[3].strip(), int(sys.argv[4])
source_round = int(sys.argv[5])
if not gpu.strip():
    raise ValueError("Specify an allocated GPU; CPU training is not allowed")
runner = runpy.run_path("11_run_all_folds.py")
resume_args = SimpleNamespace(experiment_dir=baseline, seed=seed, resume=True)
inputs = {}
for fold in range(1, 6):
    previous = Path("random") / f"seed_{seed}" / f"round_{source_round}" / f"fold_{fold}"
    train_dir = source / "training" / previous
    pred_dir = source / "predictions" / previous
    runner["require_completed"](train_dir, "training_manifest.json")
    runner["require_completed"](pred_dir, "prediction_manifest.json")
    condition = json.loads((train_dir / "experimental_condition.json").read_text())
    prediction = json.loads((pred_dir / "prediction_manifest.json").read_text())
    if (condition["method"], condition["fold"], condition["seed"], condition["round"]) != ("random", fold, seed, source_round):
        raise ValueError(f"Unexpected source conditions for fold {fold}")
    for key in ("initial_checkpoint", "valid_pair_csv"):
        if not Path(condition[key]).is_file():
            raise FileNotFoundError(condition[key])
    run_path = Path("all_data") / f"seed_{seed}" / "round_1" / f"fold_{fold}"
    runner["require_completed"](baseline / "rounds" / run_path, "selection_manifest.json")
    inputs[fold] = (condition, prediction, run_path)

def run(script, options):
    runner["validate_resume"](resume_args, script, options)
    command = [sys.executable, script]
    for key, value in options.items():
        command.extend(["--" + key, str(value)])
    if script != "10_evaluate_pair_experiment.py":
        command.append("--resume")
    subprocess.run(command, check=True)

for fold, (condition, prediction, run_path) in inputs.items():
    identity = {"experiment-id": baseline.name, "method": "all_data", "seed": seed, "fold": fold, "round": 1}
    pairs = baseline / "rounds" / run_path / "cumulative_pairs.csv"
    run("8_train_pair_acquisition.py", {
        **identity, "experiment-root": baseline.parent,
        "train-pair-csv": pairs,
        "valid-pair-csv": Path(condition["valid_pair_csv"]),
        "initial-checkpoint": Path(condition["initial_checkpoint"]),
        **{key.replace("_", "-"): condition[key] for key in ("batch_size", "epochs", "patience", "learning_rate")},
        "cuda-visible-devices": gpu,
    })
    checkpoint = runner["best_checkpoint"](baseline / "training" / run_path / "weights")
    run("9_predict_pair_experiment.py", {
        "checkpoint": checkpoint, "output-dir": baseline / "predictions" / run_path,
        "fold": fold, "seed": seed, "datatypes": "test",
        "mc-samples": prediction["mc_samples"], "cuda-visible-devices": gpu,
    })
    run("10_evaluate_pair_experiment.py", {
        **identity,
        "prediction-summary-csv": baseline / "predictions" / run_path / "test_mean_score_and_uncertainty.csv",
        "cumulative-pair-csv": pairs,
        "output-csv": baseline / "fold_performance" / f"fold_{fold}.csv",
    })
runner["merge_performance"](baseline)
print("All five all-data folds completed:", baseline / "performance.csv")
PY
```

This runs existing scripts 8, 9, and 10 in the current uv environment. Training
and prediction refuse to start without a GPU. Matching completed training and
prediction stages are reused on another execution; evaluation is refreshed.
An interrupted stage directory must be inspected before retrying: it is never
deleted or overwritten automatically. Completion produces five `all_data`
rows with `seed=240121` and `round=1` in the new `performance.csv`.

#### 3. Plot the one-shot reference as the black dotted `all data` line

```bash
uv run --frozen --no-sync 12_plot_pair_experiments.py \
    --data-root "${LIMUC_DATA_ROOT}" \
    --experiment-dir "${SOURCE_EXPERIMENT_DIR}" \
    --seed "${SEED}" --max-round 8 --bar-rounds 0,3,6,8 \
    --reference-performance-csv "${ALL_DATA_DIR}/performance.csv" \
    --reference-experiment-id "${ALL_DATA_ID}" \
    --reference-seed "${SEED}" --reference-method all_data --reference-round 1 \
    --output-dir "${SOURCE_EXPERIMENT_DIR}/plots_fig4_fig6_one_shot_v1"
```

The reference is now the five-fold mean of the one-shot control, **not** the
old random round-16 result. Overall and neighboring use their own accuracies.
Four method curves and Figure 6 remain unchanged. Choose a new output suffix
if that report directory already exists. To draw the reference on another
seed's curves, change only the curve experiment/seed and output directory;
keep the reference identity pointing to the one-shot experiment.

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
