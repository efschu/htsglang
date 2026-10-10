#!/usr/bin/env bash
# NF-GGUF G5 (09.10.2026): the SIBLING directory the nf-gguf profile names (PROFILE_SIBLING).
#
# The unsloth Qwen3.8-Flash-Next UD-IQ4_XS export is three .gguf parts and NO config.json / tokenizer: the server reads the
# geometry of a bespoke-arch GGUF only from a sibling config.json (W172) and its tokenizer only from sibling files or
# --tokenizer-path (W163). This builds ONE directory that is both: symlinks to every .gguf part (the split-set reader resolves
# the siblings of part 1 next to it) + the config.json and tokenizer files of the safetensors original of the same model
# (Qwen3.8-Flash-Next-NVFP4-nvidia, whose geometry the GGUF header is checked against by W173).
#
# usage: make_nf_gguf_sibling.sh [DEST]      (default: $MC/Qwen3.8-Flash-Next-GGUF-unsloth-sibling)
# Symlinks and small files only; nothing is read of the 87 GiB; idempotent.
set -euo pipefail
MC=${MODELS_CACHE:-/spinning/llm_stuff/club-3090/models-cache}
GGUF_DIR=${NF_GGUF_PARTS_DIR:-$MC/Qwen3.8-Flash-Next-GGUF-unsloth/UD-IQ4_XS}
ORIG=${NF_GGUF_CONFIG_SRC:-$MC/Qwen3.8-Flash-Next-NVFP4-nvidia}
DEST=${1:-$MC/Qwen3.8-Flash-Next-GGUF-unsloth-sibling}
shopt -s nullglob
parts=("$GGUF_DIR"/*.gguf)
[ ${#parts[@]} -gt 0 ] || { echo "no .gguf part in $GGUF_DIR" >&2; exit 1; }
[ -f "$ORIG/config.json" ] || { echo "no config.json in $ORIG" >&2; exit 1; }
mkdir -p "$DEST"
for p in "${parts[@]}"; do ln -sfn "$p" "$DEST/$(basename "$p")"; done
for f in config.json tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja generation_config.json; do
  [ -f "$ORIG/$f" ] && cp -f "$ORIG/$f" "$DEST/$f"
done
echo "sibling dir $DEST: ${#parts[@]} gguf part(s) + $(ls "$DEST" | grep -vc '\.gguf$') file(s) of $ORIG"
