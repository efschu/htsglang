#!/usr/bin/env bash
# NF-GGUF G5 (09.10.2026): the SIBLING directory the nf-gguf profile names (PROFILE_SIBLING).
#
# The unsloth Qwen3.8-Flash-Next UD-IQ4_XS export is three .gguf parts and NO config.json / tokenizer: the server reads the
# geometry of a bespoke-arch GGUF only from a sibling config.json (W172) and its tokenizer only from sibling files or
# --tokenizer-path (W163). This builds ONE directory that is both: symlinks to every .gguf part (the split-set reader resolves
# the siblings of part 1 next to it) + the config.json and tokenizer files of the safetensors original of the same model
# (Qwen3.8-Flash-Next-NVFP4-nvidia, whose geometry the GGUF header is checked against by W173).
#
# usage: make_nf_gguf_sibling.sh [--check] [DEST]      (default DEST: $MC/Qwen3.8-Flash-Next-GGUF-unsloth-sibling)
# Symlinks and small files only; nothing is read of the 87 GiB; idempotent.
#
# --check (G8, 10.10.2026): ONLY VERIFY, write nothing (exit 0 = the directory is what the nf-gguf profiles name, 1 = it is not, each
# failed check named). Checked: every part of the split set exists next to part 1 as a symlink/file that resolves, carries the GGUF
# magic and the -0000K-of-0000N numbering is complete; config.json is valid JSON and names a model_type; tokenizer.json and
# tokenizer_config.json are present and non-empty. Reads 4 bytes per part and the small files, never a tensor.
set -euo pipefail
CHECK=0
if [ "${1:-}" = "--check" ]; then CHECK=1; shift; fi
MC=${MODELS_CACHE:-/spinning/llm_stuff/club-3090/models-cache}
GGUF_DIR=${NF_GGUF_PARTS_DIR:-$MC/Qwen3.8-Flash-Next-GGUF-unsloth/UD-IQ4_XS}
ORIG=${NF_GGUF_CONFIG_SRC:-$MC/Qwen3.8-Flash-Next-NVFP4-nvidia}
DEST=${1:-$MC/Qwen3.8-Flash-Next-GGUF-unsloth-sibling}
if [ "$CHECK" = 1 ]; then
  bad=0
  fail() { echo "CHECK FAIL: $*" >&2; bad=1; }
  [ -d "$DEST" ] || { echo "CHECK FAIL: $DEST is not a directory (run $0 without --check)" >&2; exit 1; }
  shopt -s nullglob
  have=("$DEST"/*.gguf)
  [ ${#have[@]} -gt 0 ] || fail "no .gguf part in $DEST"
  n_of=""
  for f in "${have[@]}"; do
    [ -e "$f" ] || { fail "$(basename "$f") does not resolve (dead symlink)"; continue; }
    [ "$(head -c 4 "$f" 2>/dev/null)" = "GGUF" ] || fail "$(basename "$f") has no GGUF magic"
    case "$f" in *-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].gguf) n_of=${f##*-of-}; n_of=${n_of%.gguf} ;; esac
  done
  if [ -n "$n_of" ]; then
    want=$((10#$n_of))
    [ ${#have[@]} -eq "$want" ] || fail "split set says $want parts, $DEST holds ${#have[@]}"
    for ((i = 1; i <= want; i++)); do
      pat=$(printf '%s/*-%05d-of-%s.gguf' "$DEST" "$i" "$n_of")
      m=($pat)
      [ ${#m[@]} -eq 1 ] || fail "part $i of $want missing ($pat)"
    done
  fi
  [ -s "$DEST/config.json" ] || fail "config.json missing or empty"
  if [ -s "$DEST/config.json" ]; then
    python3 - "$DEST/config.json" <<'PY' || fail "config.json is not valid JSON naming a model_type"
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
t = c.get("text_config", c)
assert c.get("model_type") or t.get("model_type")
PY
  fi
  for f in tokenizer.json tokenizer_config.json; do [ -s "$DEST/$f" ] || fail "$f missing or empty"; done
  [ "$bad" = 0 ] && echo "sibling dir $DEST: OK (${#have[@]} gguf part(s), config.json, tokenizer files)"
  exit "$bad"
fi
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
