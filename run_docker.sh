#!/bin/bash

set -euo pipefail

readonly DEFAULT_LIMUC_DATA_ROOT="/data/umeiro0/patient_based_classified_images"
readonly DATA_ROOT="${LIMUC_DATA_ROOT:-${DEFAULT_LIMUC_DATA_ROOT}}"

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "LIMUC data root not found: ${DATA_ROOT}" >&2
    exit 1
fi

docker run --rm -it \
    --gpus all \
    -p 8888:8888 \
    -v "$(pwd):/workdir" \
    -v "${DATA_ROOT}:${DATA_ROOT}" \
    -e "LIMUC_DATA_ROOT=${DATA_ROOT}" \
    -w /workdir \
    tensorflow2.4 \
    bash
