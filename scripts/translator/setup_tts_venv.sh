#!/usr/bin/env bash
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
#
# Build the SEPARATE virtual environment that serves Qwen3-TTS.
#
# Why separate, restated here because it is the first thing a reader will
# question: every candidate TTS package pins a `transformers` version that
# conflicts with the one the sglang venv carries (`qwen-tts` wants
# transformers==4.57.3 against 5.12.1 here), and vLLM-Omni additionally pulls
# vLLM's torch against sglang's. One environment cannot hold both. Making the
# boundary a process boundary turns an unresolvable pin conflict into an HTTP
# hop -- which we wanted anyway, because the translator talks to its services
# the way any client would.
#
# This script NEVER touches /spinning/htsglang-gpu/.venv.
#
#   scripts/translator/setup_tts_venv.sh
#   scripts/translator/serve_tts.sh
#
set -euo pipefail

VENV="${TRANSLATOR_TTS_VENV:-/spinning/llm_stuff/translator-models/tts-venv}"
MODEL_DIR="${TRANSLATOR_TTS_MODEL_DIR:-/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base}"
SGLANG_VENV="/spinning/htsglang-gpu/.venv"

die() { echo "error: $*" >&2; exit 1; }

[ "$VENV" != "$SGLANG_VENV" ] || die "refusing to install into the sglang venv"

if [ ! -d "$VENV" ]; then
  echo "creating $VENV"
  python3 -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip wheel

# vLLM-Omni serves the model behind the stock OpenAI /v1/audio/speech surface
# and adds the /v1/audio/voices registry that zero-shot cloning needs (OpenAI's
# schema has nowhere to put a reference clip). Pin the vLLM version: an
# unpinned install silently changes the torch it drags in.
echo "installing the serving stack (this pulls its own torch -- expected)"
"$VENV/bin/pip" install \
  "vllm-omni${TRANSLATOR_VLLM_OMNI_PIN:-}" \
  "soundfile" \
  "numpy"

# vllm-omni does NOT depend on vllm (observed 2026-08-03: `vllm` appears
# nowhere in its Requires-Dist). It is a plugin, so the core has to be
# installed explicitly or `vllm-omni serve` dies with ModuleNotFoundError on
# the first import. The two projects version in lockstep -- vllm-omni X.Y is
# released "aligned with vLLM X.Y" -- so the core pin is derived from the
# installed plugin version rather than hardcoded, and installing it correctly
# pins torch back down (vllm-omni's own deps float torch upward past what the
# core supports: 2.13.0 vs the 2.11.0 vLLM 0.24 wants).
OMNI_VERSION="$("$VENV/bin/python" - <<'PY'
import importlib.metadata as md
print(".".join(md.version("vllm-omni").split(".")[:2]))
PY
)"
[ -n "$OMNI_VERSION" ] || die "could not read the installed vllm-omni version"
echo "installing the matching vLLM core: vllm==${OMNI_VERSION}.*"
"$VENV/bin/pip" install "vllm==${OMNI_VERSION}.*"

if [ ! -d "$MODEL_DIR" ]; then
  cat >&2 <<EOF
warning: no checkpoint at $MODEL_DIR
  Download it with (Apache-2.0, freely re-downloadable -- deliberately NOT in
  the private vendor backup, which covers only restrictive or irreplaceable
  artifacts):

    huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
      --revision ${TRANSLATOR_TTS_REVISION:-5d83992436eae1d760afd27aff78a71d676296fc} \\
      --local-dir $MODEL_DIR
EOF
fi

echo
echo "done. venv: $VENV"
"$VENV/bin/python" -c "import sys; print('python', sys.version.split()[0])"
echo "the sglang venv was not modified:"
"$SGLANG_VENV/bin/python" -c "import transformers; print('  sglang transformers', transformers.__version__)" 2>/dev/null \
  || echo "  (could not read the sglang venv, which is fine -- it was not touched)"
