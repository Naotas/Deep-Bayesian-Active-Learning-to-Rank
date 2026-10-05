#!/usr/bin/env bash
#SBATCH --job-name=limuc-initial
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=dgx-a100-80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --time=2-00:00:00

set -euo pipefail

# プロジェクトルートからsbatchで投入する。
project_root="${SLURM_SUBMIT_DIR}"
venv_path="${project_root}/.venv"
python_path="${venv_path}/bin/python"

if [[ ! -x "${python_path}" ]]; then
    echo "Python environment not found: ${python_path}" >&2
    exit 1
fi

cd "${project_root}/Experiments/Ranknet/Bayesian/LIMUC/Scripts"

# 使用するPythonとGPUをログに記録する。
"${python_path}" --version
nvidia-smi

# TensorFlowからGPUを認識できなければ終了する。
"${python_path}" -c '
import tensorflow as tf
import keras
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if not gpus:
    raise SystemExit("TensorFlow could not detect a GPU.")
'

# 初期学習を実行する。
"${python_path}" -u 2_train_bayesian_ranknet.py