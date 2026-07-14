# htsglang runtime image — uneven-TP sglang fork.
#
# Features baked in:
#   - uneven Tensor Parallelism (--rank-tp-ratio auto, --rank-gpu-memory-mib)
#   - multi-rank-per-GPU co-location (--rank-gpu-id with duplicates)
#     -> REQUIRES NCCL >= 2.30, so nvidia-nccl-cu13 is pinned to 2.30.7
#        (torch 2.11 ships 2.28.9, which REJECTS co-located communicators).
#   - HiCache 3-tier with the file storage backend
#     -> needs libssl-dev at runtime: the native page-hash C++ extension is
#        JIT-compiled against <openssl/sha.h>; without it the server aborts
#        with "Failed to load HiCache native hash extension".
#
# GPU coverage: sm75..sm120 (Turing .. Blackwell/RTX 5090). flashinfer is
# JIT-based so it covers every listed arch at runtime; TORCH_CUDA_ARCH_LIST
# only matters for any source build that may happen.
#
# Strategy: reproduce the validated host venv via pip (torch 2.11.0,
# sgl-kernel 0.3.21, flashinfer-python 0.6.14, triton 3.6.0, py3.12, cu13),
# then install the fork editable. NOT the upstream multi-stage Dockerfile.
ARG CUDA_VERSION=13.0.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 12.0" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PATH="/root/.cargo/bin:${PATH}"

# System dependencies. libssl-dev is REQUIRED for the HiCache file backend
# (native hash ext compiled against openssl/sha.h). libnuma for NUMA-aware
# host memory (HiCache L1). Rust for the fork's setuptools-rust build step.
RUN --mount=type=cache,target=/var/cache/apt,id=htsglang-apt \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-dev python3.12-venv \
      build-essential ninja-build cmake git curl wget ca-certificates \
      protobuf-compiler \
      libssl-dev libnuma1 libnuma-dev \
      libgl1 libglib2.0-0 \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3.12 get-pip.py --break-system-packages && rm get-pip.py \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && python3 -m pip config set global.break-system-packages true \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sgl-workspace

# Pinned dependency set captured from the validated host venv. nvidia-nccl-cu13
# stays at torch's pinned 2.28.9 HERE (so every `-c constraints` step resolves),
# and is force-upgraded to 2.30.7 in the final install step below — torch pins
# 2.28.9, which pip's resolver refuses to override under a constraint, so the
# bump must be a separate --no-deps --force-reinstall (as validated on the host).
COPY docker/htsglang-constraints.txt /sgl-workspace/constraints.txt

# 1) PyTorch 2.11.0 for CUDA 13.0 (matches the host venv).
RUN --mount=type=cache,target=/root/.cache/pip,id=htsglang-pip \
    python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu130 \
       torch==2.11.0 torchvision torchaudio

# 2) Core inference kernels — exact host versions. sgl-kernel from PyPI
#    (cp312 abi3 wheel), flashinfer JIT package (covers sm75-120 at runtime).
RUN --mount=type=cache,target=/root/.cache/pip,id=htsglang-pip \
    python3 -m pip install -c /sgl-workspace/constraints.txt \
       sgl-kernel==0.3.21 flashinfer-python==0.6.14 triton==3.6.0

# 3) The htsglang fork source (feature/uneven-tp: flashinfer per-rank head
#    fix f7ff51435 + co-located KV/Mamba memory fix 2c2af4f09). Editable so
#    the package resolves to /sgl-workspace/sglang/python.
COPY python /sgl-workspace/sglang/python
COPY rust /sgl-workspace/sglang/rust
COPY proto /sgl-workspace/sglang/proto
RUN --mount=type=cache,target=/root/.cache/pip,id=htsglang-pip \
    --mount=type=cache,target=/root/.cargo/registry,id=htsglang-cargo \
    cd /sgl-workspace/sglang/python \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev15138 \
       python3 -m pip install -c /sgl-workspace/constraints.txt -e "." \
    && python3 -m pip install --no-deps --force-reinstall nvidia-nccl-cu13==2.30.7

# 4) Build-time verify (NO GPU driver here): confirm the bundled NCCL is
#    2.30.7 and the fork source is present. A real `import torch`/`sgl_kernel`
#    needs libcuda.so.1 from the host driver, so that is deferred to the host
#    smoke test after `docker run --gpus all`.
RUN set -eu; \
    NCCL_LIB="$(find / -path '*nvidia/nccl/lib/libnccl.so.2' 2>/dev/null | head -1)"; \
    echo "bundled libnccl: ${NCCL_LIB}"; \
    strings "${NCCL_LIB}" | grep -qE '^2\.30\.7' \
      && echo "NCCL 2.30.7 confirmed" \
      || { echo "FATAL: bundled NCCL is not 2.30.7"; exit 1; }; \
    test -f /sgl-workspace/sglang/python/sglang/srt/entrypoints/engine.py \
      && echo "fork source present" \
      || { echo "FATAL: fork source missing"; exit 1; }

# `python` alias -> python3.12 (Ubuntu ships only python3). The entrypoint
# uses python3, but this keeps `docker run ... python ...` and any tool that
# calls bare `python` working.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# 5) Runtime assets: entrypoint (ENV-driven launch_server flags with prod
#    defaults) and the froggeric v21.3 chat template.
RUN mkdir -p /etc/htsglang
COPY --chmod=0644 docker/htsglang-chat_template.jinja /etc/htsglang/chat_template.jinja
COPY --chmod=0755 docker/htsglang-entrypoint.sh /usr/local/bin/htsglang-entrypoint.sh

# cu13 shared libs (nvrtc etc.) on the loader path, as on the host.
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}" \
    CHAT_TEMPLATE=/etc/htsglang/chat_template.jinja

EXPOSE 30000
ENTRYPOINT ["/usr/local/bin/htsglang-entrypoint.sh"]
