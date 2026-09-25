# htsglang runtime image -- UPGRADE des August-Containers. ENTWURF (27B-Sitz R, 24.09.2026). NICHT GEBAUT.
# Bauen nur nach Go des Operators. Zielort bei Freigabe: docker/htsglang.Dockerfile im Repo.
#
# ABSTAMMUNG
#   veroeffentlicht:  ghcr.io/efschu/htsglang:cu130-nccl2307 (14.07., oeffentlich; laut #384-Probe
#                     vom 14.08. "SHADOWED": zwei Distributionen liefern sgl_kernel)
#   August-Stand:     docker/htsglang.Dockerfile auf chore/release-chain-prep-r3 @ ab4a42d392
#                     (Builds htsglang:r2-99a4b0a4[-gated], 14.08., Gate "verdict=ARMED", nie gepusht;
#                     RELEASE_CHECKLIST §0.0/§4.0/§7.0)
#   Dieses Upgrade behaelt: Schritt-Aufbau, #384-Provenienz-Gate (unbedingt), INSTALL_UCX,
#   INSTALL_PLANNER, Server-/Planner-Modus unveraendert (August-Entrypoint), Chat-Template-Asset,
#   Zustands-Mountpunkte, SGLANG_BARLINK_LAUNCH_DUMP=0.
#   Es aendert:
#     * Basis CUDA 12.9.1-devel statt 13.0.1-cudnn-devel (Nutzer-Entscheid F5; JIT-nvcc des Rigs = 12.9.86,
#       alle Rig-JIT-Kerne linken libcudart.so.12, torch laeuft auf der cu13-Laufzeit aus dem venv)
#     * venv /opt/venv statt System-pip (der weg2-Launcher adressiert {venv}/lib/python3.12/site-packages/nvidia/cu13)
#     * Pakete exakt aus dem Referenz-venv (lock/requirements.lock) statt Constraints + Einzelpins
#     * NCCL (F10, Beleg RELEASE_PLAN.md §4 "F10"): die Release-Linie laeuft am Rig auf "NCCL version 2.28.9+cuda13.0"
#       aus nvidia-nccl-cu13==2.28.9. Im Referenz-venv beansprucht AUCH nvidia-nccl-cu12==2.29.7 dieselbe
#       Datei nvidia/nccl/lib/libnccl.so.2 (Schatten wie #384); am Rig gewann cu13, weil es zuletzt kam.
#       Hier wird das nicht der pip-Reihenfolge ueberlassen: cu13 2.28.9 wird nach dem Lock erzwungen und
#       per Banner + sha256 gegen die Rig-Datei geprueft. NCCL_PIN=2.30.7 (+ NCCL_BANNER/NCCL_SHA256
#       passend) stellt den August-Stand fuer Co-Location (--rank-gpu-id mit Duplikaten) her.
#     * sgl-kernel: das Fork-Wheel (INT8-Arm) ist Pflicht, REQUIRE_INT8_ARM=1 Default
#     * NEU MODE=weg2: P/D-Flip-Launcher mit Profilen je Linie und Modellformat, Transport bar1|nccl
#     * NEU optionale NV-Header fuer barlink BAR1 (WITH_NV_HEADERS=0 Default, Nutzer-Entscheid F6)
#     * NEU JIT-Vorbau (FlashInfer in Server-Reihenfolge, barlink-Exts, TMS) + tvm-ffi-Saat aus dem Rig
#     * NEU Stufe-B-Pfade (SGLANG_WEG2_*, Commit bae049a3b4) statt fest verdrahteter Rig-Pfade
#
# KONTEXT = das von prepare_context.sh erzeugte Verzeichnis ctx/<linie>-<sha10>/ (nicht der Repo-Baum):
#   src/ (flacher git-Klon, sauber, mit .git), lock/, assets/{wheels,nvidia-open-595,devtools,arb-seed,
#   profiles,tvm-ffi,flashinfer_modules.txt}, tools/{entrypoint.sh,healthcheck.sh,prebuild_jit.py,profiles/},
#   BUILD_INFO.json, .dockerignore.
#
# Bau (Beispiel; auf dem Proxmox-Host aus dem LXC-Pfad, siehe host_acceptance.sh):
#   docker build -f <ctx>/Dockerfile --build-arg CUDA_BASE=nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:<Pull> \
#     --build-arg HTSGLANG_REVISION=<sha40> --build-arg HTSGLANG_LINE=27b \
#     --build-arg SGLANG_SCM_VERSION=0.0.0.dev0+27b.<release>.<sha10> [--build-arg WITH_NV_HEADERS=1] \
#     --build-arg NCCL_SHA256=$(jq -r .nccl.sha256 <ctx>/BUILD_INFO.json) \
#     -t htsglang:cu129-weg2-27b-<release>-<sha10> <ctx>    # F14: neuer versionierter Tag, nie ueberschreiben

ARG CUDA_BASE=nvidia/cuda:12.9.1-devel-ubuntu24.04
FROM ${CUDA_BASE}

# SCHICHTFOLGE UND CACHE (RELEASE_PLAN §7.1): jede ARG-Deklaration geht als Umgebung in ALLE folgenden RUN-Schritte
# und damit in deren Cache-Schluessel ein. Oben stehen darum nur Argumente, die sich NICHT je SHA aendern.
# HTSGLANG_REVISION und SGLANG_SCM_VERSION werden erst in Schritt 4 (Quellbaum) deklariert -- hier oben invalidierte
# jede neue SHA schon apt, pip und den FlashInfer-Vorbau. Ebenso HTSGLANG_LINE (27b|nf) und PUSH_STATE erst ganz am Ende:
# so teilen sich das 27B- und das NF-Image alle Schichten bis einschliesslich FlashInfer-Vorbau (gleiches Lock,
# gleiche Modulliste, gleiches prebuild_jit.py vorausgesetzt).
ARG PIP_VERSION=26.1.2
ARG NCCL_PIN=2.28.9
ARG NCCL_PACKAGE=nvidia-nccl-cu13
ARG NCCL_BANNER="NCCL version 2.28.9+cuda13.0"
# sha256 der Rig-Datei (BUILD_INFO.json .nccl.sha256; host_acceptance.sh reicht es durch). Leer = nur Banner.
ARG NCCL_SHA256=
ARG KERNEL_WHEEL=sglang_kernel-0.4.4-cp310-abi3-linux_x86_64.whl
ARG KERNEL_WHEEL_SHA256=67f03cfa755efa01498c7732bd6ae015ec5673feffe9a51452fefdbe0dcd4664
ARG REQUIRE_INT8_ARM=1
ARG INSTALL_UCX=1
ARG INSTALL_PLANNER=1
# F6: die Header binden das Image an den Treiberstand 595.58.03 (+smallbar). Default AUS.
ARG WITH_NV_HEADERS=0
ARG SEED_TVM_FFI=1
ARG PREBUILD_ARCHS=8.6,12.0
# Parallele nvcc-Schritte im JIT-Vorbau. 4 reichten am 25.09. nicht unter 24 GiB (gemm_sm120, CUTLASS: >23 GiB, OOM);
# host_build.sh setzt den Wert zusammen mit dem Speicher-Deckel des Builders.
ARG MAX_JOBS=3
# 1 = jeder Modul-Fehler im Vorbau bricht den Bau ab; 0 (Default) = laut im Log und in JIT_PREBUILD.json
# (verdict INCOMPLETE), das Modul baut dann beim ersten Boot per JIT.
ARG PREBUILD_STRICT=0

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 TZ=Etc/UTC

# --- 1. System (August-Liste + weg2-Werkzeuge) -----------------------------------
# procps/psmisc: pgrep/ps/fuser (Launcher #1217, Deadman, Teardown). protobuf-compiler + rustup:
# Rust-Ext sglang.srt.grpc._core (python/pyproject.toml, nicht optional). libssl-dev: HiCache-Hash-Ext.
# KEIN apt ninja-build: Vorbau und Laufzeit muessen DASSELBE ninja nehmen (das aus dem venv).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-dev python3.12-venv \
      build-essential git curl wget ca-certificates jq \
      protobuf-compiler libssl-dev libnuma1 libnuma-dev \
      procps psmisc iproute2 util-linux \
      libgl1 libglib2.0-0 libcairo2 ffmpeg
ARG INSTALL_UCX
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    if [ "${INSTALL_UCX}" = "1" ]; then \
      apt-get install -y --no-install-recommends libucx0 ibverbs-providers libibverbs1 librdmacm1; \
    else echo "INSTALL_UCX=0 -> no UCX, barlink ucx transport unavailable"; fi
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable

# --- 2. venv mit dem Paketstand des Referenz-venv ---------------------------------
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/root/.cargo/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CUDA_HOME=/usr/local/cuda
COPY lock/requirements.lock /opt/htsglang/lock/requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip,id=htsglang-pip \
    pip install "pip==${PIP_VERSION}" \
    && pip install --no-deps -r /opt/htsglang/lock/requirements.lock \
    && python -c "import torch; assert torch.__version__.startswith('2.11.0'), torch.__version__; assert torch.version.cuda == '13.0', torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
ARG NCCL_PIN
ARG NCCL_PACKAGE
ARG NCCL_BANNER
ARG NCCL_SHA256
# Eine Datei, zwei RECORDs: nie `pip uninstall nvidia-nccl-cu12` im Image -- das loescht die geteilte
# libnccl.so.2 mit (RECORD listet sie). Der erzwungene cu13-Einbau zuletzt legt den Gewinner fest.
RUN --mount=type=cache,target=/root/.cache/pip,id=htsglang-pip \
    set -eu; \
    [ -n "${NCCL_PIN}" ] || { echo "FATAL: NCCL_PIN leer -- F10 verlangt eine benannte NCCL-Version"; exit 1; }; \
    pip install --no-deps --force-reinstall "${NCCL_PACKAGE}==${NCCL_PIN}"; \
    L=/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2; \
    V=$(strings "$L" | grep -m1 -oE 'NCCL version [0-9.]+\+cuda[0-9.]+' || true); \
    H=$(sha256sum "$L" | cut -d' ' -f1); \
    echo "bundled libnccl: $L -> ${V:-?} sha256=$H"; \
    [ "$V" = "${NCCL_BANNER}" ] || { echo "FATAL: libnccl.so.2 meldet '${V:-?}', erwartet '${NCCL_BANNER}'"; exit 1; }; \
    if [ -n "${NCCL_SHA256}" ]; then [ "$H" = "${NCCL_SHA256}" ] || { echo "FATAL: libnccl.so.2 sha256 $H != Rig ${NCCL_SHA256}"; exit 1; }; fi; \
    for r in /opt/venv/lib/python3.12/site-packages/nvidia_nccl_cu1*.dist-info/RECORD; do \
      echo "RECORD-Anspruch $(basename "$(dirname "$r")"): $(grep -E '^nvidia/nccl/lib/libnccl\.so\.2,' "$r" | cut -d, -f2)"; \
    done

# --- 3. sgl-kernel: genau EIN Anbieter, Arm verlangt (#384, August-Gate unbedingt) --
COPY assets/wheels/ /tmp/htsglang-wheels/
ARG KERNEL_WHEEL
ARG KERNEL_WHEEL_SHA256
RUN set -eu; W="/tmp/htsglang-wheels/${KERNEL_WHEEL}"; \
    echo "${KERNEL_WHEEL_SHA256}  ${W}" | sha256sum -c -; \
    pip uninstall -y sgl-kernel sglang-kernel >/dev/null 2>&1 || true; \
    pip install --no-deps "${W}"; rm -rf /tmp/htsglang-wheels

# --- 3b. FlashInfer-Vorbau VOR dem Quellbaum (RELEASE_PLAN §7.1) -------------------
# Der teure Teil des Vorbaus (FlashInfer 86 + 120f, ~36 von 38,7 min am 25.09.) braucht nur das venv (flashinfer, torch
# aus dem Lock), die CUDA-Toolchain der Basis, prebuild_jit.py und die Modulliste -- KEIN src/: fi_child importiert nur
# flashinfer und bildet den Flag-Kontext (Import- vs. Server-Kontext) selbst nach. Darum steht er hier: eine neue SHA
# trifft diese Schicht im Builder-Cache. Neu gebaut wird sie nur, wenn sich Lock, Modulliste, prebuild_jit.py oder
# PREBUILD_ARCHS/MAX_JOBS/PREBUILD_STRICT aendern. Alles, was in eine build.ninja eingeht, steht wie zur Laufzeit:
# CUDA_HOME, PATH (nvcc/ninja), HOME, kein CC, FLASHINFER_NVCC_THREADS ungesetzt, FLASHINFER_JIT_DEBUG/VERBOSE ungesetzt.
COPY tools/prebuild_jit.py /opt/htsglang/tools/prebuild_jit.py
COPY assets/flashinfer_modules.txt /opt/htsglang/tools/flashinfer_modules.txt
ARG PREBUILD_ARCHS
ARG MAX_JOBS
ARG PREBUILD_STRICT
RUN set -eu; cd /; \
    STRICT=""; if [ "${PREBUILD_STRICT}" = "1" ]; then STRICT="--strict"; fi; \
    MAX_JOBS=${MAX_JOBS} python /opt/htsglang/tools/prebuild_jit.py --only flashinfer --archs ${PREBUILD_ARCHS} \
      --flashinfer-manifest /opt/htsglang/tools/flashinfer_modules.txt \
      --report /opt/htsglang/JIT_PREBUILD.json ${STRICT}

# --- 4. Quellbaum der Linie (sauber, mit Historie) + Paket aus Wegwerf-Kopie ------
COPY src/ /opt/htsglang/src/
ARG HTSGLANG_REVISION
ARG SGLANG_SCM_VERSION=0.0.0.dev0
ARG REQUIRE_INT8_ARM
RUN set -eu; cd /opt/htsglang/src; \
    test -n "${HTSGLANG_REVISION}" || { echo "FATAL: HTSGLANG_REVISION fehlt"; exit 1; }; \
    test "$(git rev-parse HEAD)" = "${HTSGLANG_REVISION}" || { echo "FATAL: HEAD != HTSGLANG_REVISION"; exit 1; }; \
    test -z "$(git status --porcelain)" || { echo "FATAL: Baum nicht sauber"; git status --porcelain | head; exit 1; }
RUN --mount=type=cache,target=/root/.cache/pip,id=htsglang-pip \
    --mount=type=cache,target=/root/.cargo/registry,id=htsglang-cargo \
    set -eu; cp -a /opt/htsglang/src /tmp/pkgsrc; cd /tmp/pkgsrc/python; \
    SETUPTOOLS_SCM_PRETEND_VERSION="${SGLANG_SCM_VERSION}" pip install --no-deps . ; \
    cd /; rm -rf /tmp/pkgsrc; \
    ARM=""; if [ "${REQUIRE_INT8_ARM}" = "1" ]; then ARM="--require-arm"; fi; \
    python -I /opt/htsglang/src/python/sglang/srt/utils/kernel_dist_guard.py \
      --site-packages /opt/venv/lib/python3.12/site-packages ${ARM} --expect-pinned-sha256; \
    mkdir -p /sgl-workspace && ln -sfn /opt/htsglang/src /sgl-workspace/sglang; \
    ln -sf /opt/venv/bin/python /usr/local/bin/python

# 4b) Planner-Extra wie im August (nur der Quality-Benchmark-Reiter braucht es).
ARG INSTALL_PLANNER
RUN --mount=type=cache,target=/root/.cache/pip,id=htsglang-pip \
    if [ "${INSTALL_PLANNER}" = "1" ]; then \
      pip install "chess>=1.11.0" "cairosvg>=2.7.0" "defusedxml>=0.7.1"; \
    else echo "INSTALL_PLANNER=0 -> planner quality-benchmark tab unavailable"; fi

# --- 5. Optionale NV-Header (barlink BAR1) ------------------------------------------
# assets/nvidia-open-595/ existiert immer (leer ohne --with-nv-headers in prepare_context.sh).
# Ohne Header lehnt barlink_bar1_ext.load_dmabuf_ext() mit Grund ab; der Entrypoint verweigert
# dann HTSGLANG_TRANSPORT=bar1 mit diesem Grund (oder die Header werden zur Laufzeit gemountet:
# -v <treiberbaum>:/opt/nvidia-open-595:ro, passend zum LAUFENDEN Host-Treiber).
COPY assets/nvidia-open-595/ /opt/nvidia-open-595/
ARG WITH_NV_HEADERS
RUN set -eu; H=/opt/nvidia-open-595/src/common/sdk/nvidia/inc/nvos.h; \
    if [ "${WITH_NV_HEADERS}" = "1" ]; then test -f "$H" || { echo "FATAL: WITH_NV_HEADERS=1, aber $H fehlt im Kontext"; exit 1; }; \
    else rm -rf /opt/nvidia-open-595/* ; echo "WITH_NV_HEADERS=0 -> keine NV-Header im Image (bar1 nur mit gemounteten Headern)"; fi
ENV SGLANG_BARLINK_BAR1_NV_SOURCE=/opt/nvidia-open-595
# --- 6. Vorbau Teil B: was src/ braucht (barlink, HiCache-Hash, TMS-Hook) -----------
# Stufe-B-Pfade schon hier, damit der TMS-Hook an seinem Laufzeitort landet.
ENV SGLANG_WEG2_VENV=/opt/venv \
    SGLANG_WEG2_TMS_OUT_DIR=/opt/htsglang/tms \
    SGLANG_WEG2_DEVTOOLS_DIR=/opt/htsglang/devtools
# Zuerst die Rig-Paritaet der cu13-Links: der Referenz-venv traegt seit 17.07. zehn VON HAND angelegte Links unter
# nvidia/cu13/lib, die in keinem RECORD stehen (unversionierte .so fuer cudart, curand, nvrtc, cufft, cusolver, cublas,
# nvJitLink, cusparse, cublasLt, dazu stubs/libcuda.so -> /usr/local/cuda-12.9/lib64/stubs/libcuda.so). Ohne
# libcudart.so scheitert der Link des TMS-Preload-Hooks (Bau 25.09. 02:44Z: "ld: cannot find -lcudart") -- und der
# Launcher baut den Hook bei JEDEM Boot (launcher.py build_tms_preload) und verweigert dann (Weg2LaunchRefused).
# Danach barlink/cpu_ext/tms; --continue-report ergaenzt den Bericht aus Schritt 3b und rechnet das verdict ueber alles.
RUN set -eu; L=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib; \
    for n in cudart.so.13 curand.so.10 nvrtc.so.13 cufft.so.12 cusolver.so.12 cublas.so.13 nvJitLink.so.13 cusparse.so.12 cublasLt.so.13; do \
      test -f "$L/lib$n" || { echo "FATAL: $L/lib$n fehlt"; exit 1; }; \
      ln -sfn "lib$n" "$L/lib${n%%.so.*}.so"; \
    done; \
    test -f /usr/local/cuda-12.9/lib64/stubs/libcuda.so || { echo "FATAL: /usr/local/cuda-12.9/lib64/stubs/libcuda.so fehlt"; exit 1; }; \
    mkdir -p "$L/stubs"; ln -sfn /usr/local/cuda-12.9/lib64/stubs/libcuda.so "$L/stubs/libcuda.so"; \
    cd /opt/htsglang/src; \
    STRICT=""; if [ "${PREBUILD_STRICT}" = "1" ]; then STRICT="--strict"; fi; \
    MAX_JOBS=${MAX_JOBS} PYTHONPATH=/opt/htsglang/src/python \
    python /opt/htsglang/tools/prebuild_jit.py --only barlink,cpu_ext,tms --continue-report \
      --tree /opt/htsglang/src --venv /opt/venv --archs ${PREBUILD_ARCHS} \
      --report /opt/htsglang/JIT_PREBUILD.json ${STRICT}; \
    test -z "$(git -C /opt/htsglang/src status --porcelain)" || { echo "FATAL: Vorbau hat den Baum veraendert"; exit 1; }

# --- 7. Werkzeuge, Saat, Profile, Einstieg, Zustands-Mountpunkte ----------------------
# Alles hier aendert sich oft (Profile beim Freischalten, tvm-ffi-Saat nach neuen Rig-Boots, BUILD_INFO) und steht
# darum HINTER dem Vorbau: ein Profilwechsel kostet nur diese Schichten und den Export.
COPY assets/devtools/ /opt/htsglang/devtools/
COPY assets/arb-seed/ /opt/htsglang/arb-seed/
COPY tools/profiles/ /opt/htsglang/profiles/
COPY assets/profiles/ /opt/htsglang/profiles/
# Der py-spy-Symlink bedient boot_deadman.sh:484 (Rig-Werkzeug, ruft py-spy ueber den Rig-venv-Pfad).
RUN chmod 0755 /opt/htsglang/devtools/*.sh \
    && mkdir -p /spinning/htsglang-gpu/.venv/bin && ln -sf /opt/venv/bin/py-spy /spinning/htsglang-gpu/.venv/bin/py-spy

# tvm-ffi (sglang jit_kernel, u.a. Marlin FP8/NVFP4/INT4 fuer sm_86) ist inhaltsadressiert und
# pfadunabhaengig (jit_kernel/utils.py:196-345, 600-700): die Rig-Saat wird zur LAUFZEIT gelesen, soweit
# Quellen und Flags zum Baum passen -- der Vorbau braucht sie nicht, darum steht sie hinter ihm.
# assets/tvm-ffi/ ist leer, wenn prepare_context.sh ohne Saat lief.
ARG SEED_TVM_FFI
COPY assets/tvm-ffi/ /root/.cache/tvm-ffi/
RUN if [ "${SEED_TVM_FFI}" != "1" ]; then rm -rf /root/.cache/tvm-ffi/*; fi; \
    echo "tvm-ffi Saat: $(ls /root/.cache/tvm-ffi | wc -l) Eintraege"

RUN mkdir -p /etc/htsglang \
    && cp /opt/htsglang/src/docker/htsglang-chat_template.jinja /etc/htsglang/chat_template.jinja \
    && install -m 0755 /opt/htsglang/src/docker/htsglang-entrypoint.sh /usr/local/bin/htsglang-entrypoint.sh
COPY tools/entrypoint.sh /opt/htsglang/entrypoint.sh
COPY tools/healthcheck.sh /opt/htsglang/healthcheck.sh
COPY BUILD_INFO.json /opt/htsglang/BUILD_INFO.json
RUN chmod 0755 /opt/htsglang/entrypoint.sh /opt/htsglang/healthcheck.sh \
    && mkdir -p /root/.cache/sglang/rigmon /root/.cache/flashinfer /root/.cache/torch_extensions \
                /root/.cache/tvm-ffi /root/.triton \
                /var/lib/htsglang/hicache /var/lib/htsglang/hibernate \
                /var/lib/htsglang/evidence /var/lib/htsglang/arb /var/lib/htsglang/hicache-weg2

# August-ENV unveraendert (Server-/Planner-Modus), dazu der Loader-Pfad des venv und Stufe B.
# TORCH_CUDA_ARCH_LIST bleibt fuer den Server-Modus wie im August; der weg2-Modus entfernt es
# (sonst passten die vorgebauten barlink-Namen *_cuda_86_120 nicht). Bewusst NICHT gesetzt:
# FLASHINFER_CUDA_ARCH_LIST, PYTORCH_CUDA_ALLOC_CONF.
ARG HTSGLANG_LINE=27b
# Push-Stand der Revision aus BUILD_INFO.json (prepare_context.sh): "pushed ..." oder
# "UNPUSHED, Veroeffentlichung erst nach Push durch den Nutzer" -- host_publish.sh liest das Label und sperrt.
ARG PUSH_STATE=unbekannt
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 12.0" \
    SGLANG_BARLINK_LAUNCH_DUMP=0 \
    HICACHE_STORAGE_DIR=/var/lib/htsglang/hicache \
    TRITON_CACHE_DIR=/root/.triton \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
    SGLANG_PLANNER_PROFILES=/root/.cache/sglang/planner_profiles.json \
    SGLANG_PLANNER_GRAPH_ANCHORS=/root/.cache/sglang/graph_mem_anchors.json \
    SGLANG_WEG2_EVIDENCE_DIR=/var/lib/htsglang/evidence \
    SGLANG_WEG2_GPU_ARB=/var/lib/htsglang/arb \
    SGLANG_WEG2_STORE_ROOT=/var/lib/htsglang/hicache-weg2 \
    HTSGLANG_TREE=/opt/htsglang/src \
    HTSGLANG_REVISION=${HTSGLANG_REVISION} \
    HTSGLANG_LINE=${HTSGLANG_LINE}

LABEL org.opencontainers.image.title="htsglang" \
      org.opencontainers.image.revision="${HTSGLANG_REVISION}" \
      org.opencontainers.image.source="https://github.com/efschu/htsglang" \
      org.opencontainers.image.description="htsglang (uneven-TP sglang fork): MODE=server|planner wie bisher, MODE=weg2 P/D-Flip (${HTSGLANG_LINE}), barlink BAR1 oder NCCL" \
      htsglang.line="${HTSGLANG_LINE}" \
      htsglang.push_state="${PUSH_STATE}" \
      htsglang.driver.expected="595.58.03" \
      htsglang.flashinfer="0.6.14"

# 30000 Server (MODE=server), 8780 Planner, 8770 rigmon, 30030 weg2-Front (MODE=weg2).
EXPOSE 30000 8780 8770 30030
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=10s --start-period=1200s --retries=3 CMD ["/opt/htsglang/healthcheck.sh"]
ENTRYPOINT ["/opt/htsglang/entrypoint.sh"]
