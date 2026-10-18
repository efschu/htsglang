#!/usr/bin/env bash
# S37 MIXED-D x QSA mutation check: one plausible WRONG edit per danger direction; each MUST turn
# test_pdflip_s37_mixed_qsa_grain_1018.py red. Runs LOCALLY in a 4 GiB cgroup (the rig RAM guard's capped form).
# ONLY=<name> runs one. Restores every file on exit.
#   T1  "Tail als Extend": no tail is recognised (the rc10 death: QSA-PREFIX-GRAIN)
#   T2  the tail closes the PREVIOUS group always (start = end - 1, not the decode rule)
#   T3  the tail token is dumped like an extend token when it closes the group (ring member lost)
#   T4  the indexer never compresses the tail groups
#   T5  the tail group's ring members in the wrong order (first member's rope / pooling order)
#   T6  the tails stay in the extend plan too (members read from packed rows of other requests)
#   T7  the K11 exemption covers every row of a MIXED forward (a long prefill row off the grain slips through)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
SRT=$ROOT/python/flliper/srt
TEST=test/registered/unit/pdflip/test_pdflip_s37_mixed_qsa_grain_1018.py
PY=${PY:-/spinning/shvllm/.venv/bin/python}
B=$SRT/layers/attention/qwen_sparse_attn_backend.py
M=$SRT/layers/attention/qsa/metadata.py
I=$SRT/layers/attention/qsa/qsa_indexer.py
FILES=("$B" "$M" "$I")
BAKD=$(mktemp -d); for f in "${FILES[@]}"; do cp "$f" "$BAKD/$(echo "$f" | tr / _)"; done
restore() { for f in "${FILES[@]}"; do cp "$BAKD/$(echo "$f" | tr / _)" "$f"; done; }
trap restore EXIT
mutate() { python3 - "$1" "$2" "$3" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read(); n = s.count(old)
if n != 1: sys.exit(f"mutation anchor found {n}x: {old!r}")
open(path, "w").write(s.replace(old, new))
PY
}
declare -A F OLD NEW
F[T1]=$B; OLD[T1]='    if not _qsa_is_mixed(forward_batch):
        return None
    return extend_lens.long() == 1'; NEW[T1]='    return None'
F[T2]=$B; OLD[T2]='start_blocks = torch.where(tail, end_blocks - (lengths % ratio == 0).long(), end_blocks)'; NEW[T2]='start_blocks = torch.where(tail, end_blocks - 1, end_blocks)'
F[T3]=$M; OLD[T3]='            pending = pending | always_rows.to(device=pending.device, dtype=torch.bool)[rows]'; NEW[T3]='            pass'
F[T4]=$I; OLD[T4]='            self._compress_from_ring(pool, metadata.tail_group_ring_locs, tail_locs)'; NEW[T4]='            pass'
F[T5]=$I; OLD[T5]='            self._compress_from_ring(pool, metadata.tail_group_ring_locs, tail_locs)'; NEW[T5]='            self._compress_from_ring(pool, metadata.tail_group_ring_locs.flip(1), tail_locs)'
F[T6]=$B; OLD[T6]='            start_blocks = torch.where(tail, end_blocks, start_blocks)'; NEW[T6]='            pass'
F[T7]=$B; OLD[T7]='    return [e == 1 for e in ext]'; NEW[T7]='    return [True for e in ext]'
ORDER=(T1 T2 T3 T4 T5 T6 T7)
[ -n "${ONLY:-}" ] && ORDER=("$ONLY")
killed=0; survived=0
for m in "${ORDER[@]}"; do
  restore
  if ! mutate "${F[$m]}" "${OLD[$m]}" "${NEW[$m]}"; then echo "MUTANT $m: ANCHOR MISSING"; survived=$((survived+1)); continue; fi
  out=$(cd "$ROOT/test" && systemd-run --scope -p MemoryMax=4G --quiet env PYTHONPATH="$ROOT/python" timeout 600 "$PY" -m pytest -q -p no:cacheprovider "-opythonpath=registered/unit/pdflip" "${TEST#test/}" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -aE "[0-9]+ (passed|failed)" | tail -1)
  case "$out" in *failed*) r=red;; *passed*) r=green;; *) r="?($out)";; esac
  if [ "$r" = red ]; then killed=$((killed+1)); echo "MUTANT $m: KILLED ($out)"; else survived=$((survived+1)); echo "MUTANT $m: SURVIVED $r ($out)"; fi
done
restore
echo "MUTANTS: killed=$killed survived=$survived of ${#ORDER[@]}"
