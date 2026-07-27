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
#   - HTCCL host-staged collectives for a second node over RoCE/UCX
#     -> needs libucx0 at runtime (ctypes dlopen of libucp.so.0).
#   - the planner / rigmon GUI (MODE=planner in the entrypoint). The planner's
#     web UI is stdlib http.server — no fastapi/uvicorn/jinja needed — but its
#     quality-benchmark tab pulls the `sglang[planner]` extra, which needs
#     libcairo2 at runtime.
#
# Strategy: reproduce the validated host venv via pip (torch 2.11.0,
# sgl-kernel 0.3.21, flashinfer-python 0.6.14, triton 3.6.0, py3.12, cu13),
# then install the fork editable. NOT the upstream multi-stage Dockerfile.
#
# ---------------------------------------------------------------------------
# GPU architecture coverage — what actually determines it
# ---------------------------------------------------------------------------
# This image installs *prebuilt wheels*; it compiles no CUDA source. So
# TORCH_CUDA_ARCH_LIST does NOT drive build time or image size here. It is set
# only so that a runtime JIT step (flashinfer, the HiCache hash extension,
# torch.utils.cpp_extension) picks a sane target set. What actually determines
# coverage is which cubins the wheels carry:
#
#   torch 2.11.0+cu130      sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
#                           (measured: torch._C._cuda_getArchFlags(); note the
#                           absence of any compute_XX PTX -> no JIT fallback to
#                           an arch not in that list. sm_87/sm_89/sm_121 are
#                           reached by CUDA minor-version compatibility from
#                           sm_86 / sm_120.)
#   sgl-kernel 0.3.21       sm_80 sm_86 sm_90 sm_120 (+ a separate sm100/
#                           subpackage). Measured with cuobjdump --list-elf.
#                           Cubin-only, no PTX -> hard floor sm_80.
#   flashinfer-python       JIT at runtime from source, so it follows the
#                           driver's actual card. Documented floor sm_75.
#   Marlin / native fp8     sm_80+ / sm_89+ respectively, model-side gates.
#
# sm_75 (Turing, RTX 2080 Ti) therefore runs WITHOUT sgl-kernel. That is a
# supported configuration of this fork, not an accident: `sgl_kernel_runnable()`
# in python/sglang/srt/utils/common.py compares the live device capability
# against SGL_KERNEL_MIN_CUDA_CC = (8, 0) and routes to `forward_native` /
# Triton / the torch-native sampler when the card is below it. So a Turing rank
# works whether the wheel is installed or not; installing it merely wastes
# ~2 GB of image. Use INSTALL_SGL_KERNEL=0 for a Turing-only image.
#
# CUDA 13 dropped Maxwell/Pascal/Volta. sm_70 and below need a cu12 base image;
# that is out of scope for this Dockerfile.
# ---------------------------------------------------------------------------

ARG CUDA_VERSION=13.0.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

# Version surface. Defaults are exactly the validated host venv — changing one
# leaves the "reproduce the validated venv" guarantee behind, which is fine for
# an experiment but must not be done silently.
ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG SGL_KERNEL_VERSION=0.3.21
ARG FLASHINFER_VERSION=0.6.14
ARG TRITON_VERSION=3.6.0
ARG NCCL_VERSION=2.30.7
ARG NCCL_PACKAGE=nvidia-nccl-cu13
ARG SGLANG_SCM_VERSION=0.0.0.dev15138

# Feature toggles.
#   INSTALL_SGL_KERNEL=0 -> Turing-only slim image (see the arch block above).
#   INSTALL_UCX=0        -> drop HTCCL/UCX; single-node images do not need it.
#   INSTALL_PLANNER=0    -> no planner extra; MODE=planner still serves the UI,
#                           only the quality-benchmark tab breaks.
ARG INSTALL_SGL_KERNEL=1
ARG INSTALL_UCX=1
ARG INSTALL_PLANNER=1

# Only consumed by runtime JIT steps — see the arch block above.
ARG TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 12.0"

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PATH="/root/.cargo/bin:${PATH}"

# System dependencies. libssl-dev is REQUIRED for the HiCache file backend
# (native hash ext compiled against openssl/sha.h). libnuma for NUMA-aware
# host memory (HiCache L1). Rust for the fork's setuptools-rust build step.
# ffmpeg for GGUF repos that ship a video_preprocessor (torchcodec at
# config-load time).
RUN --mount=type=cache,target=/var/cache/apt,id=htsglang-apt \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-dev python3.12-venv \
      build-essential ninja-build cmake git curl wget ca-certificates \
      protobuf-compiler \
      libssl-dev libnuma1 libnuma-dev \
      libgl1 libglib2.0-0 libcairo2 \
      ffmpeg \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3.12 get-pip.py --break-system-packages && rm get-pip.py \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && python3 -m pip config set global.break-system-packages true \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && rm -rf /var/lib/apt/lists/*

# UCX runtime for the HTCCL cross-rig transport. htccl_ucx_bindings.py dlopens
# libucp.so.0 by name via ctypes — no headers, no build step, so the runtime
# package is enough. Ubuntu 24.04 ships UCX 1.16.0, which is the release the
# binding's struct offsets were dumped from.
#
# UCX refuses to interoperate across releases with a different UCP wire address
# format. Running the SAME image on both nodes is what guarantees parity; if
# the second node runs UCX from the host instead, point SGLANG_HTCCL_UCX_LIB at
# a matching build on both sides.
ARG INSTALL_UCX
RUN --mount=type=cache,target=/var/cache/apt,id=htsglang-apt \
    if [ "${INSTALL_UCX}" = "1" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        libucx0 ibverbs-providers libibverbs1 librdmacm1 \
      && rm -rf /var/lib/apt/lists/*; \
    else echo "INSTALL_UCX=0 -> no UCX, HTCCL ucx transport unavailable"; fi

WORKDIR /sgl-workspace

# Pinned dependency set captured from the validated host venv. nvidia-nccl-cu13
# stays at torch's pinned 2.28.9 HERE (so every `-c constraints` step resolves),
# and is force-upgraded to 2.30.7 in the final install step below — torch pins
# 2.28.9, which pip's resolver refuses to override under a constraint, so the
# bump must be a separate --no-deps --force-reinstall (as validated on the host).
COPY docker/htsglang-constraints.txt /sgl-workspace/constraints.txt

# 1) PyTorch for CUDA 13.0 (matches the host venv).
ARG TORCH_VERSION
ARG TORCH_INDEX
RUN --mount=type=cache,target=/root/.cache/pip,id=htsglang-pip \
    python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --index-url "${TORCH_INDEX}" \
       "torch==${TORCH_VERSION}" torchvision torchaudio

# 2) Core inference kernels — exact host versions. sgl-kernel from PyPI
#    (cp312 abi3 wheel), flashinfer JIT package (covers sm75+ at runtime).
#    sgl-kernel is skippable: see the arch block at the top of this file.
ARG INSTALL_SGL_KERNEL
ARG SGL_KERNEL_VERSION
ARG FLASHINFER_VERSION
ARG TRITON_VERSION
RUN --mount=type=cache,target=/root/.cache/pip,id=htsglang-pip \
    python3 -m pip install -c /sgl-workspace/constraints.txt \
       "flashinfer-python==${FLASHINFER_VERSION}" "triton==${TRITON_VERSION}" \
    && if [ "${INSTALL_SGL_KERNEL}" = "1" ]; then \
         python3 -m pip install -c /sgl-workspace/constraints.txt \
           "sgl-kernel==${SGL_KERNEL_VERSION}"; \
       else \
         echo "INSTALL_SGL_KERNEL=0 -> sm75/Turing slim image, native+Triton paths"; \
       fi

# 3) The htsglang fork source. Editable so the package resolves to
#    /sgl-workspace/sglang/python.
COPY python /sgl-workspace/sglang/python
COPY rust /sgl-workspace/sglang/rust
COPY proto /sgl-workspace/sglang/proto
ARG SGLANG_SCM_VERSION
ARG NCCL_PACKAGE
ARG NCCL_VERSION
RUN --mount=type=cache,target=/root/.cache/pip,id=htsglang-pip \
    --mount=type=cache,target=/root/.cargo/registry,id=htsglang-cargo \
    cd /sgl-workspace/sglang/python \
    && SETUPTOOLS_SCM_PRETEND_VERSION="${SGLANG_SCM_VERSION}" \
       python3 -m pip install -c /sgl-workspace/constraints.txt -e "." \
    && python3 -m pip install --no-deps --force-reinstall \
       "${NCCL_PACKAGE}==${NCCL_VERSION}"

# 3b) Planner extra: chess + cairosvg + defusedxml (pyproject's `planner`
#     extra). Only the quality-benchmark tab imports them, and it imports them
#     lazily, so the UI boots without them — but the tab then hard-fails.
ARG INSTALL_PLANNER
RUN --mount=type=cache,target=/root/.cache/pip,id=htsglang-pip \
    if [ "${INSTALL_PLANNER}" = "1" ]; then \
      python3 -m pip install -c /sgl-workspace/constraints.txt \
        "chess>=1.11.0" "cairosvg>=2.7.0" "defusedxml>=0.7.1"; \
    else echo "INSTALL_PLANNER=0 -> planner quality-benchmark tab unavailable"; fi

# 4) Build-time verify (NO GPU driver here): confirm the bundled NCCL matches
#    NCCL_VERSION and the fork source is present. A real `import torch` /
#    `sgl_kernel` needs libcuda.so.1 from the host driver, so that is deferred
#    to the host smoke test after `docker run --gpus all`.
RUN set -eu; \
    NCCL_LIB="$(find / -path '*nvidia/nccl/lib/libnccl.so.2' 2>/dev/null | head -1)"; \
    echo "bundled libnccl: ${NCCL_LIB}"; \
    strings "${NCCL_LIB}" | grep -qF "${NCCL_VERSION}" \
      && echo "NCCL ${NCCL_VERSION} confirmed" \
      || { echo "FATAL: bundled NCCL is not ${NCCL_VERSION}"; exit 1; }; \
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

# 6) Mount points for the state that must survive a container restart. Created
#    here so a bind mount lands on an existing path with the right owner, and
#    so an UNMOUNTED run still works (it just loses the cache on exit).
#    See docker/htsglang.env.example for what each one holds.
RUN mkdir -p \
      /root/.cache/sglang \
      /root/.cache/sglang/rigmon \
      /root/.cache/flashinfer \
      /root/.cache/torch_extensions \
      /root/.triton \
      /var/lib/htsglang/hicache \
      /var/lib/htsglang/hibernate

# cu13 shared libs (nvrtc etc.) on the loader path, as on the host.
#
# CHAT_TEMPLATE is deliberately NOT set here: the froggeric v21.3 template is
# Qwen-specific, and a baked-in default would silently apply it to every model.
# The compose files set it explicitly.
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}" \
    HICACHE_STORAGE_DIR=/var/lib/htsglang/hicache \
    TRITON_CACHE_DIR=/root/.triton \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
    SGLANG_PLANNER_PROFILES=/root/.cache/sglang/planner_profiles.json \
    SGLANG_PLANNER_GRAPH_ANCHORS=/root/.cache/sglang/graph_mem_anchors.json

# 30000 = OpenAI-compatible server; 8780 = planner web UI (MODE=planner);
# 8770 = rigmon aggregator, if it is started manually.
EXPOSE 30000 8780 8770
ENTRYPOINT ["/usr/local/bin/htsglang-entrypoint.sh"]
