#!/usr/bin/env bash
# MIXED-D S32 mutation check: one plausible WRONG edit per danger direction; each MUST turn test_weg2_mixed_d_short_s32_1018.py
# red. A survivor = that direction is asserted nowhere. Runs through remote_desk.sh (CPU desk, --allow-dirty snapshots).
# ONLY=<name> runs one. Restores every file on exit.
#   G1  gate: u == halbX counts as short (< -> <=)                         G2  gate: a chunked request may mix (R6 rest)
#   (G3 'X == 0 may mix' is an EQUIVALENT mutant: halbX = 0 so u < 0 never holds -- dropped)   A1  verdict is the rank-local flag (no group MIN, R11)
#   K1  chunk stamp never written (last chunk of a big prefill reads short)
#   S1  tail seq_len = base (attention drops the row)                      S2  tail prefix_lens delta 0 instead of -1
#   S3  late binding: seq_lens = fresh (not fresh + 1)                     S4  late binding: bonus slot gathered at fresh + 1
#   S5  late binding never called under overlap+spec                       S6  running spec_info not dropped (one-sided merge)
#   M1  mamba track of the tail rows set True                              R1  decode tail stamped as a finished prefill (prefill clock)
#   R2  tail's bonus token not committed (kv_committed_len += 1 gone)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
SRT=$ROOT/python/sglang/srt
RD=${RD:-/spinning/gpu-arb/devtools/remote_desk.sh}
TEST=test/registered/unit/weg2/test_weg2_mixed_d_short_s32_1018.py
FILES=("$SRT/weg2/mixed_d_short.py" "$SRT/managers/schedule_batch.py" "$SRT/managers/overlap_utils.py" "$SRT/managers/scheduler.py" "$SRT/managers/scheduler_components/batch_result_processor.py")
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
F[G1]=$SRT/weg2/mixed_d_short.py; OLD[G1]='return int(u_tokens) < half_x_of'; NEW[G1]='return int(u_tokens) <= half_x_of'
F[G2]=$SRT/weg2/mixed_d_short.py; OLD[G2]='if int(x_tokens) <= 0 or chunked:'; NEW[G2]='if int(x_tokens) <= 0:'
F[A1]=$SRT/managers/scheduler.py; OLD[A1]='agreed = bool(self._weg2_group_min_flags([mine])[0])'; NEW[A1]='agreed = bool(mine)'
F[K1]=$SRT/managers/scheduler.py; OLD[K1]='self.chunked_req._mixed_d_was_chunked = True'; NEW[K1]='pass'
F[S1]=$SRT/managers/schedule_batch.py; OLD[S1]='merged[-running_bs:] = tail_base + 1'; NEW[S1]='merged[-running_bs:] = tail_base'
F[S2]=$SRT/managers/schedule_batch.py; OLD[S2]='        if spec_mix:
            delta = -1'; NEW[S2]='        if spec_mix:
            delta = 0'
F[S3]=$SRT/managers/overlap_utils.py; OLD[S3]='seq_lens[-n:] = fresh + 1'; NEW[S3]='seq_lens[-n:] = fresh'
F[S4]=$SRT/managers/overlap_utils.py; OLD[S4]='self.req_to_token[idx.long(), fresh.long()]'; NEW[S4]='self.req_to_token[idx.long(), (fresh + 1).long()]'
F[S5]=$SRT/managers/overlap_utils.py; OLD[S5]='                future_map.resolve_mixed_spec_tails(batch)'; NEW[S5]='                pass'
F[S6]=$SRT/managers/schedule_batch.py; OLD[S6]='            running_batch.spec_info = None
'; NEW[S6]=''
F[M1]=$SRT/managers/schedule_batch.py; OLD[M1]='torch.zeros(running_bs, dtype=_mt[1].dtype, device=_mt[1].device)'; NEW[M1]='torch.ones(running_bs, dtype=_mt[1].dtype, device=_mt[1].device)'
F[R1]=$SRT/managers/scheduler_components/batch_result_processor.py; OLD[R1]='                    if not _is_mixed_tail:'; NEW[R1]='                    if True:'
F[R2]=$SRT/managers/scheduler_components/batch_result_processor.py; OLD[R2]='                        req.kv_committed_len += 1
'; NEW[R2]='                        pass
'
run_suite() { (cd "$ROOT" && "$RD" run --allow-dirty "$ROOT" -q -p no:cacheprovider "-opythonpath=registered/unit/weg2" "$TEST" 2>&1 | grep -E "^VERDICT" ); }
base=$(run_suite); echo "BASELINE: $base"
case "$base" in *"VERDICT: OK"*) ;; *) echo "baseline not green -- abort"; exit 2;; esac
surv=0
for m in ${ONLY:-G1 G2 A1 K1 S1 S2 S3 S4 S5 S6 M1 R1 R2}; do
  restore; mutate "${F[$m]}" "${OLD[$m]}" "${NEW[$m]}" || { echo "$m: ANCHOR FAIL"; surv=1; continue; }
  v=$(run_suite)
  case "$v" in *"VERDICT: OK"*) echo "$m: SURVIVED  <<<"; surv=1;; *) echo "$m: killed ($v)";; esac
done
restore
exit $surv
