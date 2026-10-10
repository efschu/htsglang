#!/usr/bin/env bash
# NF-GGUF G8 (10.10.2026): the DRAFT directory the nf-gguf-d profile names (PROFILE_DRAFT_DIR / PROFILE_DRAFT).
#
# The unsloth MTP head is one .gguf FILE (MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf, 2.59 GiB, no token_embd/output: the draft shares
# the vocabulary modules of the target). The server reads the draft GEOMETRY (hidden size, hc_count, ...) from a config.json NEXT TO
# that file -- exactly as for the target (W172 missing / W173 foreign) -- and the MTP folder of the download holds none. This builds
# ONE directory with: a symlink to the MTP file + the config.json of the safetensors original of the same model (the same file the
# sibling directory of the TARGET carries; the NVFP4-nvidia export, whose geometry the GGUF header is checked against by W173).
# Fallback when that original is not on the box: the config.json of the TARGET sibling directory (make_nf_gguf_sibling.sh).
#
# THE PATH THE SERVER GETS is the symlink INSIDE this directory (<DEST>/<file>.gguf), NOT the directory: a GGUF draft is
# recognised by ``check_gguf_file`` (a FILE, suffix .gguf; load_config.resolve_draft_load_format, model_config.py draft branch) and
# a directory is read by DefaultModelLoader as a safetensors checkpoint -- it would load ``auto`` and find no weights.
#
# usage: make_nf_gguf_draft_dir.sh [--check] [DEST]      (default DEST: $MC/Qwen3.8-Flash-Next-GGUF-unsloth-draft)
#   env: NF_GGUF_MTP_FILE (default $MC/Qwen3.8-Flash-Next-GGUF-unsloth/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf; the fallback
#        variant mtp-Qwen3.8-Flash-Next-Q8_0.gguf works the same, name it here and in HTSGLANG_DRAFT),
#        NF_GGUF_CONFIG_SRC (default $MC/Qwen3.8-Flash-Next-NVFP4-nvidia), NF_GGUF_SIBLING (target sibling dir, config fallback).
# Symlink and one small file; idempotent.
#
# --check: ONLY VERIFY, write nothing; exit 0 = the directory is what the profile names, 1 = not (each failed check named). Checked: the
# symlink exists and resolves to a file with the GGUF magic, config.json is valid JSON naming a model_type with a hidden_size,
# and that hidden_size/hc_count agree with the TARGET sibling's config.json when that one is present (a draft config of another
# model is what W173 would refuse at boot). Reads 4 bytes of the MTP file and the two small configs.
set -euo pipefail
CHECK=0
if [ "${1:-}" = "--check" ]; then CHECK=1; shift; fi
MC=${MODELS_CACHE:-/spinning/llm_stuff/club-3090/models-cache}
MTP=${NF_GGUF_MTP_FILE:-$MC/Qwen3.8-Flash-Next-GGUF-unsloth/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
ORIG=${NF_GGUF_CONFIG_SRC:-$MC/Qwen3.8-Flash-Next-NVFP4-nvidia}
SIB=${NF_GGUF_SIBLING:-$MC/Qwen3.8-Flash-Next-GGUF-unsloth-sibling}
DEST=${1:-$MC/Qwen3.8-Flash-Next-GGUF-unsloth-draft}
LINK="$DEST/$(basename "$MTP")"

if [ "$CHECK" = 1 ]; then
  bad=0
  fail() { echo "CHECK FAIL: $*" >&2; bad=1; }
  [ -d "$DEST" ] || { echo "CHECK FAIL: $DEST is not a directory (run $0 without --check)" >&2; exit 1; }
  shopt -s nullglob
  have=("$DEST"/*.gguf)
  [ ${#have[@]} -eq 1 ] || fail "$DEST must hold exactly one .gguf (the MTP head), holds ${#have[@]}"
  for f in "${have[@]}"; do
    [ -L "$f" ] || fail "$(basename "$f") is not a symlink (the download stays the single copy)"
    [ -e "$f" ] || { fail "$(basename "$f") does not resolve (dead symlink)"; continue; }
    [ "$(head -c 4 "$f" 2>/dev/null)" = "GGUF" ] || fail "$(basename "$f") has no GGUF magic"
  done
  [ -s "$DEST/config.json" ] || fail "config.json missing or empty (W172 at boot)"
  if [ -s "$DEST/config.json" ]; then
    python3 - "$DEST/config.json" "$SIB/config.json" <<'PY' || fail "config.json: not valid JSON naming a model_type/hidden_size, or foreign to the target sibling"
import json, os, sys
def geom(path):
    c = json.load(open(path, encoding="utf-8"))
    t = c.get("text_config", c)
    assert c.get("model_type") or t.get("model_type"), "no model_type"
    assert t.get("hidden_size"), "no hidden_size"
    return (t.get("hidden_size"), t.get("hc_count"), t.get("num_hidden_layers"), t.get("vocab_size"))
d = geom(sys.argv[1])
if os.path.isfile(sys.argv[2]):
    s = geom(sys.argv[2])
    assert d == s, "draft config %r != target sibling config %r" % (d, s)
PY
  fi
  [ "$bad" = 0 ] && echo "draft dir $DEST: OK (${have[*]##*/} -> config.json beside it); pass <$DEST>/$(basename "${have[0]:-x}") to the server, not the directory"
  exit "$bad"
fi

[ -f "$MTP" ] || { echo "no MTP file $MTP" >&2; exit 1; }
if [ -f "$ORIG/config.json" ]; then CFG="$ORIG/config.json"
elif [ -f "$SIB/config.json" ]; then CFG="$SIB/config.json"
else echo "no config.json in $ORIG nor in the target sibling $SIB" >&2; exit 1; fi
mkdir -p "$DEST"
ln -sfn "$MTP" "$LINK"
cp -f "$CFG" "$DEST/config.json"
echo "draft dir $DEST: $(basename "$LINK") -> $MTP + config.json of $(dirname "$CFG")"
