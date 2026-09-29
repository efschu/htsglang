#!/bin/bash
# Weg-2 (#1233) boot wrapper: two groups + the front, sequenced by
# python/flliper/srt/pdflip/launcher.py.  The environment lines are inherited
# from /spinning/gpu-arb/boot_855_train0901.sh (:152-160 cu13 loader path,
# :251-255 FLLIPER_* pins) and the launcher owns the rest (presence sweep,
# host-ledger preflight, ledger, store, sequencing, deadmen, symlink).
#
# Usage:
#   TREE_ARG=/spinning/wt-weg2-s3s4 TAG_ARG=weg2ls1b1 scripts/pdflip/boot_pdflip.sh [--dry-run] [--debug-hold P|D|both]
#   scripts/pdflip/boot_pdflip.sh --teardown /spinning/gpu-arb/weg2/boot_<tag>.json
set -u
TREE=${TREE_ARG:-/spinning/wt-weg2-s3s4}
VENV=${VENV_ARG:-/spinning/htsglang-gpu/.venv}
TAG=${TAG_ARG:-weg2ls}
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$TREE/python"
if [ "${1:-}" = "--teardown" ]; then
  exec "$VENV/bin/python" -m flliper.srt.pdflip.launcher --tree "$TREE" --tag "$TAG" --teardown "$2"
fi
exec "$VENV/bin/python" -m flliper.srt.pdflip.launcher --tree "$TREE" --tag "$TAG" "$@"
