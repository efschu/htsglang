#!/bin/bash
# Task #651: stage Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf + its tokenizer on the rig.
#
# The checkpoint was present on 2026-08-08, was used for one boot attempt, and
# was removed in the bulk models-cache cleanup around 2026-08-21. Nothing on the
# rig or the laptop carries a copy, so it is re-fetched from source.
#
# Two repos are involved and this is not incidental:
#   weights   unsloth/Qwen3.6-35B-A3B-MTP-GGUF   (the MTP-preserved GGUF; blk.40
#                                                 is the NEXTN draft block, which
#                                                 is what makes speculation
#                                                 possible at all on this file)
#   tokenizer Qwen/Qwen3.6-35B-A3B               (the GGUF repo ships NO tokenizer
#                                                 files -- only .gguf, README and
#                                                 an imatrix)
#
# Both are public and ungated. Resumable: hf_hub_download skips complete files
# and continues partial ones, so re-running after an interruption is cheap.
set -euo pipefail

VENV=${VENV:-/spinning/htsglang-gpu/.venv}
DEST=${DEST:-/spinning/llm_stuff/club-3090/models-cache/unsloth/Qwen3.6-35B-A3B-MTP-GGUF}
GGUF_REPO=unsloth/Qwen3.6-35B-A3B-MTP-GGUF
BASE_REPO=Qwen/Qwen3.6-35B-A3B
GGUF_FILE=Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf

mkdir -p "$DEST"

# Refuse early rather than half-way: 21.28 GiB of weights plus headroom.
AVAIL_GIB=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
if [ "${AVAIL_GIB:-0}" -lt 30 ]; then
  echo "REFUSE: only ${AVAIL_GIB} GiB free at $DEST; need ~30 GiB." >&2
  exit 2
fi

exec "$VENV/bin/python" - "$DEST" "$GGUF_REPO" "$BASE_REPO" "$GGUF_FILE" <<'PY'
import os
import sys
import time

from huggingface_hub import hf_hub_download

dest, gguf_repo, base_repo, gguf_file = sys.argv[1:5]

# Tokenizer first: small, and a failure here is worth learning in 5 seconds
# rather than after a 21 GiB transfer. chat_template.jinja matters for a served
# chat model; the rest is the standard BPE pair plus configs.
tok_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
]
for name in tok_files:
    try:
        p = hf_hub_download(repo_id=base_repo, filename=name, local_dir=dest)
        print(f"[tok] {name} -> {p}", flush=True)
    except Exception as exc:  # a missing optional file must not stop the run
        print(f"[tok] {name} SKIPPED: {type(exc).__name__}: {exc}", flush=True)

t0 = time.time()
print(f"[gguf] fetching {gguf_file} (21.28 GiB) from {gguf_repo}", flush=True)
path = hf_hub_download(repo_id=gguf_repo, filename=gguf_file, local_dir=dest)
size = os.path.getsize(path)
dt = time.time() - t0
print(
    f"[gguf] DONE {path} {size} bytes ({size / 2**30:.2f} GiB) "
    f"in {dt:.0f}s ({size / 2**20 / max(dt, 1):.1f} MiB/s)",
    flush=True,
)
PY
