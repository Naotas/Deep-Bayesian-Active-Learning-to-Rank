#!/bin/bash

set -euo pipefail

docker run --rm \
    --gpus all \
    tensorflow2.4 \
    uv run \
    --project /opt/project \
    --frozen \
    --no-sync \
    /opt/environment_smoke_test.py \
    --require-gpu
