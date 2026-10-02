FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Asia/Tokyo \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/bin/python3.8 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:${PATH}"

RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        python3.8 \
        python3.8-venv \
        tzdata \
    && apt-get clean

RUN curl -LsSf https://astral.sh/uv/0.11.15/install.sh \
        --output /tmp/install-uv.sh \
    && UV_UNMANAGED_INSTALL=/usr/local/bin sh /tmp/install-uv.sh

WORKDIR /opt/project
COPY pyproject.toml uv.lock ./
COPY Docker/environment_smoke_test.py /opt/environment_smoke_test.py

RUN uv sync --frozen --no-dev --no-install-project \
    && uv run --frozen --no-sync /opt/environment_smoke_test.py

# Create a non-root user matching the host UID/GID
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd --gid ${GROUP_ID} dockeruser \
    && useradd --create-home --uid ${USER_ID} --gid ${GROUP_ID} \
        --shell /bin/bash dockeruser

USER dockeruser
WORKDIR /workdir
