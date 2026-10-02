#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
readonly PROJECT_ROOT

docker build \
    --tag=tensorflow2.4 \
    --platform=linux/amd64 \
    --build-arg USER_ID="$(id -u)" \
    --build-arg GROUP_ID="$(id -g)" \
    --rm=true \
    --force-rm=true \
    --file "${SCRIPT_DIR}/Dockerfile" \
    "${PROJECT_ROOT}"
