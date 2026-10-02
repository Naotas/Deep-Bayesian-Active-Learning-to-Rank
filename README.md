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

For this public code release, we provide a Docker environment based on NVIDIA TensorFlow 21.05 (`nvcr.io/nvidia/tensorflow:21.05-tf2-py3`) with Keras 2.4.3.

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

### 2. Build the Docker image

From the project root, run:

```bash
cd Docker
bash run_build.sh
cd ..
```

### 3. Start the Docker container

From the project root, run:

```bash
bash run_docker.sh
```

The project directory is mounted to `/workdir`. `run_docker.sh` also mounts
`$LIMUC_DATA_ROOT` at the same absolute path inside the container and passes the
environment variable through. Set `LIMUC_DATA_ROOT` before starting the container
when using a different data-server path.

### 4. Prepare the dataset

Inside the Docker container, move to:

```bash
cd /workdir/Data/UC/LIMUC/Preprocessing
```

Run the preprocessing scripts sequentially:

```bash
python 1_prepare_uc_image_dataset.py
python 2_create_patient_level_splits.py
python 3_format_fold_datasets.py
python 4_create_training_only_datasets.py
```

To specify the root without an environment variable, pass the same option to every
command, for example:

```bash
python 1_prepare_uc_image_dataset.py \
    --data-root /data/umeiro0/patient_based_classified_images
```

### 5. Run Bayesian active learning-to-rank

Move to:

```bash
cd /workdir/Experiments/Ranknet/Bayesian/LIMUC/Scripts
```

To start the active-learning procedure, run scripts 1–6 sequentially:

```bash
python 1_initial_learning_make_pair.py
python 2_train_bayesian_ranknet.py
python 3_predict_score.py <AL_0_result_date>
python 4_calculate_uncertainty.py <AL_0_result_date>
python 5_active_learning_make_pair.py <AL_0_result_date> --iteration 1
python 6_train_bayesian_ranknet_AL.py --iteration 1 --initial-result-date <AL_0_result_date>
```

After completing script 6, repeat scripts 3–6 for each subsequent active-learning iteration. For example, for iteration 2:

```bash
python 3_predict_score.py <AL_1_result_date>
python 4_calculate_uncertainty.py <AL_1_result_date>
python 5_active_learning_make_pair.py <AL_1_result_date> --iteration 2
python 6_train_bayesian_ranknet_AL.py --iteration 2 --initial-result-date <AL_0_result_date>
```

Here, `<AL_0_result_date>` refers to the result directory from the initial training, and `<AL_1_result_date>` refers to the result directory from the first active-learning iteration.

For later iterations, replace `<AL_1_result_date>` with the result directory from the immediately preceding iteration and increment `--iteration` accordingly.

Continue this cycle until the desired active-learning iteration is reached.

Every script in steps 1–6 accepts the same `--data-root` option. When
`LIMUC_DATA_ROOT` is exported, the commands above automatically read preprocessed
data, Active Learning CSV files, and results from the shared `generated/`
directory. No current-working-directory-dependent data path is used.

In the experiments reported in the paper, the number of active-learning iterations was set to `K = 6`.

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
