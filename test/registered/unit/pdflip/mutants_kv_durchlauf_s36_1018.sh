#!/usr/bin/env bash
# S36 KV-DURCHLAUF / write-behind / ledger mutation check: one plausible WRONG edit per danger direction; each MUST turn
# test_pdflip_s36_kv_durchlauf_1018.py red. A survivor = that direction is asserted nowhere. Runs LOCALLY in a 4 GiB
# cgroup (systemd-run --scope -p MemoryMax=4G: the rig RAM guard's capped form) -- or RD=1 through remote_desk.sh.
# ONLY=<name> runs one. Restores every file on exit.
#   Direction "Durchlauf nur auf einem Rang" (R11, raenge-nie-uneins):
#     R1  the decision (kv_durchlauf.durchlauf_stems) only on the attention host -- the workers wait out the room bound
#     R2  the read site (_arena_page_get) asks the durchlauf only on the attention host
#   Direction "falsche Seite / Offset":
#     O1  resolve: a disk page's ids name the NEXT disk page              O2  load: lane off by one inside the page
#     O3  load: the K rows take the V block                               O4  read: the L3 file read at a wrong offset
#     O5  rows -> disk page index off by one                              O6  QSA: every layer takes layer 0's block
#     C1  chain proof: a disk page proven for ANY key (not only its own held file)
#   Direction "Write-Behind haeuft wieder an":
#     W1  the old quiet gate is back (legs / sleep / dormancy / loads stop the pass)
#     W2  read priority gone (a chunk never waits for a read in flight)    W3  the pass takes no read turn per chunk
#   Leak / ledger:
#     F1  ArenaMHAHostPool.free keeps the disk page (held fd leaks)       F2  HostPoolGroup.free keeps it
#     L1  #810 books --hicache-size again under the arena host pool
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
SRT=$ROOT/python/flliper/srt
TEST=test/registered/unit/pdflip/test_pdflip_s36_kv_durchlauf_1018.py
PY=${PY:-/spinning/shvllm/.venv/bin/python}
FILES=("$SRT/pdflip/kv_durchlauf.py" "$SRT/managers/cache_controller.py" "$SRT/mem_cache/pool_host/arena_pool.py"
       "$SRT/mem_cache/qsa_pool_host.py" "$SRT/mem_cache/l3_write_behind.py" "$SRT/mem_cache/hicache_storage.py"
       "$SRT/mem_cache/memory_pool_host.py" "$SRT/server_args.py")
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
RR='__import__("flliper.srt.rank_role", fromlist=["x"]).this_rank_is_form_a_worker()'
F[R1]=$SRT/pdflip/kv_durchlauf.py; OLD[R1]='    if not on():
        return []
    return [int(i) for i, st in still_full if st in on_disk]'; NEW[R1]="    if not on() or $RR:
        return []
    return [int(i) for i, st in still_full if st in on_disk]"
F[R2]=$SRT/managers/cache_controller.py; OLD[R2]='                    if _kv_dl.on():'; NEW[R2]="                    if _kv_dl.on() and not $RR:"
F[O1]=$SRT/mem_cache/pool_host/arena_pool.py; OLD[O1]='first = torch.where(is_d, num * P + _kv_dl.disk_base(P), num * P + int(self.staging_rows))'; NEW[O1]='first = torch.where(is_d, (num + 1) * P + _kv_dl.disk_base(P), num * P + int(self.staging_rows))'
F[O2]=$SRT/mem_cache/pool_host/arena_pool.py; OLD[O2]='            ln = lane[sel].to(dev)'; NEW[O2]='            ln = ((lane[sel] + 1) % P).to(dev)'
F[O3]=$SRT/mem_cache/pool_host/arena_pool.py; OLD[O3]='                for off, buf in ((self._k_offs_b[l], device_pool.k_buffer[l]),'; NEW[O3]='                for off, buf in ((self._v_offs_b[l], device_pool.k_buffer[l]),'
F[O4]=$SRT/pdflip/kv_durchlauf.py; OLD[O4]='return pio.read_fds([entries[i].fd for i in idx], [0] * len(idx),'; NEW[O4]='return pio.read_fds([entries[i].fd for i in idx], [8] * len(idx),'
F[O5]=$SRT/pdflip/kv_durchlauf.py; OLD[O5]='    return r // P, r % P'; NEW[O5]='    return r // P + 1, r % P'
F[O6]=$SRT/mem_cache/qsa_pool_host.py; OLD[O6]='blk = stage.view(len(ds), self.layer_num, self.item_bytes)[:, layer_id]'; NEW[O6]='blk = stage.view(len(ds), self.layer_num, self.item_bytes)[:, 0]'
F[C1]=$SRT/mem_cache/pool_host/arena_pool.py; OLD[C1]='ok[j] = (int(hf[j]) - base) % P == 0 and _kv_dl.REGISTRY.stem_of(d) == str(stems[j])'; NEW[C1]='ok[j] = True'
F[W1]=$SRT/mem_cache/l3_write_behind.py; OLD[W1]='    censuses and the tests that pin that they no longer gate."""
    return None'; NEW[W1]='    censuses and the tests that pin that they no longer gate."""
    return former_quiet_reason()'
F[W2]=$SRT/mem_cache/l3_write_behind.py; OLD[W2]='    if _reads["inflight"] <= 0 or max_wait_s <= 0:'; NEW[W2]='    if True:'
F[W3]=$SRT/mem_cache/hicache_storage.py; OLD[W3]='                    tot["read_yield_ms"] += _gate.write_turn() * 1e3'; NEW[W3]='                    pass'
F[F1]=$SRT/mem_cache/pool_host/arena_pool.py; OLD[F1]='            _kv_dl.REGISTRY.release_rows(idx, _psz(self))   # S36: a freed node'"'"'s disk pages'; NEW[F1]='            pass'
F[F2]=$SRT/mem_cache/memory_pool_host.py; OLD[F2]='                _rel(_idx)'; NEW[F2]='                pass'
F[L1]=$SRT/server_args.py; OLD[L1]='            per_rank_bytes = int(_stg_bytes)'; NEW[L1]='            pass'
ORDER=(R1 R2 O1 O2 O3 O4 O5 O6 C1 W1 W2 W3 F1 F2 L1)
[ -n "${ONLY:-}" ] && ORDER=("$ONLY")
killed=0; survived=0
for m in "${ORDER[@]}"; do
  restore
  if ! mutate "${F[$m]}" "${OLD[$m]}" "${NEW[$m]}"; then echo "MUTANT $m: ANCHOR MISSING"; survived=$((survived+1)); continue; fi
  if [ -n "${RD:-}" ]; then
    out=$(/spinning/gpu-arb/devtools/remote_desk.sh run "$ROOT" --allow-dirty -q -p no:cacheprovider "-opythonpath=registered/unit/pdflip" "$TEST" 2>&1 | grep -E "VERDICT")
    case "$out" in *TESTS-FAILED*) r=red;; *OK*) r=green;; *) r="?($out)";; esac
  else
    out=$(cd "$ROOT/test" && systemd-run --scope -p MemoryMax=4G --quiet env PYTHONPATH="$ROOT/python" timeout 600 "$PY" -m pytest -q -p no:cacheprovider "-opythonpath=registered/unit/pdflip" "${TEST#test/}" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -aE "[0-9]+ (passed|failed)" | tail -1)
    case "$out" in *failed*) r=red;; *passed*) r=green;; *) r="?($out)";; esac
  fi
  if [ "$r" = red ]; then killed=$((killed+1)); echo "MUTANT $m: KILLED ($out)"; else survived=$((survived+1)); echo "MUTANT $m: SURVIVED $r ($out)"; fi
done
restore
echo "MUTANTS: killed=$killed survived=$survived of ${#ORDER[@]}"
