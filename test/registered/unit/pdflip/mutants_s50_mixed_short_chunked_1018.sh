#!/usr/bin/env bash
# S50 MIXED-D short-chunked mutation check: one plausible WRONG edit per danger direction; each MUST turn
# test_s50_mixed_short_chunked_1018.py red. Local, 4-GiB cgroup. ONLY=<name> runs one. Restores every file on exit.
#   C1 chunked = big again (total ignored)            C2 boundary: total == halbX reads short (>= -> >)
#   C3 a big chunk may mix (never big)                C4 stamp re-taken every chunk (the last chunk of a big one reads short)
#   C5 a chunked request without a total reads short  C6 the chunked_req behind the batch is not looked at
#   C7 the verdict is the rank-local flag (no group MIN, R11)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
SRT=$ROOT/python/flliper/srt
PY=${PY:-/spinning/shvllm/.venv/bin/python}
TEST=test/registered/unit/pdflip/test_s50_mixed_short_chunked_1018.py
FILES=("$SRT/pdflip/mixed_d_short.py" "$SRT/managers/scheduler.py")
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
M=$SRT/pdflip/mixed_d_short.py
F[C1]=$M; OLD[C1]='    if not chunked_enabled():
        return True'; NEW[C1]='    return True'
F[C2]=$M; OLD[C2]='if total is None or hx <= 0 or int(total) >= hx:'; NEW[C2]='if total is None or hx <= 0 or int(total) > hx:'
F[C3]=$M; OLD[C3]='    if not cands:
        return False'; NEW[C3]='    return False'
F[C4]=$M; OLD[C4]='    if getattr(req, ATTR_TOTAL, None) is not None:
        return
'; NEW[C4]=''
F[C5]=$M; OLD[C5]='if total is None or hx <= 0 or int(total) >= hx:'; NEW[C5]='if total is not None and (hx <= 0 or int(total) >= hx):'
F[C6]=$M; OLD[C6]='    if chunked_req is not None:
        cands.append(chunked_req)'; NEW[C6]=''
F[C7]=$SRT/managers/scheduler.py; OLD[C7]='agreed = bool(self._pdflip_group_min_flags([mine])[0])'; NEW[C7]='agreed = bool(mine)'
ORDER=(C1 C2 C3 C4 C5 C6 C7)
[ -n "${ONLY:-}" ] && ORDER=("$ONLY")
killed=0; survived=0
for m in "${ORDER[@]}"; do
  restore
  mutate "${F[$m]}" "${OLD[$m]}" "${NEW[$m]}" || { echo "$m ANCHOR-FAIL"; survived=$((survived+1)); continue; }
  if (cd "$ROOT" && systemd-run --scope -q -p MemoryMax=4G -E PYTHONPATH=python:test/registered/unit/pdflip "$PY" -m pytest -q -x -p no:cacheprovider "$TEST" >/dev/null 2>&1); then
    echo "$m SURVIVED"; survived=$((survived+1))
  else
    echo "$m killed"; killed=$((killed+1))
  fi
done
echo "killed=$killed survived=$survived"
[ "$survived" -eq 0 ]
