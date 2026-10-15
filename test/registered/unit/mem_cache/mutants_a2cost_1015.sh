#!/usr/bin/env bash
# A2COST-1015 mutation check (nf-chainkey-a2-review-1015 B-C, A2 VOTE-CACHE). One plausible WRONG implementation per
# new decision; each MUST turn test_weg2_a2cost_1015.py red. A survivor = that decision is asserted nowhere.
# ONLY=<name> runs one mutant. Runs every suite through /spinning/gpu-arb/pytest_gedeckelt.sh (CPU only). Restores every file on exit.
#
#   MA  a hit ignores the remembered generations (state-only): a recycled slot is "proven"
#   MB  a hit ignores the slot states (generation-only): an EVICTING slot (key unlinked, generation kept) is "proven"
#   MC  a hit ignores the key list (first/last/len signature only)
#   MD  a hit ignores the rows
#   ME  the fill is not bracketed by the generations (a slot moving under the proof is remembered)
#   MF  a failing chain is remembered too
#   MG  verify_load_chain (the load-time proof) consults the vote cache
#   MH  a cache error propagates into the vote
#   MI  the switch is ignored
#   MJ  the default is OFF
#   MK  the cache is unbounded
#   ML  a hit is "proven" without re-reading the arena at all
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
AP=$ROOT/python/sglang/srt/mem_cache/pool_host/arena_pool.py
TESTS=test/registered/unit/mem_cache/test_weg2_a2cost_1015.py
GED=${GED:-/spinning/gpu-arb/pytest_gedeckelt.sh}

BAK_DIR=$(mktemp -d)
cp "$AP" "$BAK_DIR/ap.py"
restore() { cp "$BAK_DIR/ap.py" "$AP"; }
trap restore EXIT

run_suite() { (cd "$ROOT" && "$GED" "$TESTS" -x -p no:warnings >/dev/null 2>&1); }

mutate() {
  python3 - "$1" "$2" "$3" <<'EOF'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
n = s.count(old)
if n != 1:
    sys.exit(f"mutation anchor found {n}x in {path}: {old!r}")
open(path, "w").write(s.replace(old, new))
EOF
}

survivors=0
check() {
  local name=$1 file=$2 old=$3 new=$4
  if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$name" ]; then return; fi
  restore
  if ! mutate "$file" "$old" "$new"; then echo "$name ANCHOR-MISSING"; survivors=$((survivors + 1)); return; fi
  if run_suite; then echo "$name SURVIVED"; survivors=$((survivors + 1)); else echo "$name killed"; fi
}

check MA "$AP" '        if not np.array_equal(gens, egens):' '        if False:'
check MB "$AP" '        if not bool((states == 2).all()):' '        if False:'
check MC "$AP" '        if ekeys != keys or not torch.equal(ear, ar):' '        if not torch.equal(ear, ar):'
check MD "$AP" '        if ekeys != keys or not torch.equal(ear, ar):' '        if ekeys != keys:'
check ME "$AP" '                if np.array_equal(g0, g1) and bool((s1 == 2).all()):' '                if True:'
check MF "$AP" '        elif vote is not None and g0 is not None:' '        if vote is not None and g0 is not None:'
check MG "$AP" '    if not _chain_shifted(nodes, keys_of, p_chain, page0) and _chain_clean(P, ar) \
            and _chain_keys_hold(pool, nodes, ar, stems_of, keys_of):   # A2 CHAIN-KEY
        return host_indices' '    _vc = _vote_cache_of(pool)
    if _vc is not None and _vc.hit(pool.arena, ar, _flat_keys(nodes, keys_of)):
        return host_indices
    if not _chain_shifted(nodes, keys_of, p_chain, page0) and _chain_clean(P, ar) \
            and _chain_keys_hold(pool, nodes, ar, stems_of, keys_of):   # A2 CHAIN-KEY
        return host_indices'
check MH "$AP" '                _vote_cache_log("error")
                vote = None' '                _vote_cache_log("error")
                raise'
check MI "$AP" '    return os.environ.get(ENV_VOTE_CACHE, "1").strip().lower() not in ("0", "false", "off", "no", "")' '    return True'
check MJ "$AP" '    return os.environ.get(ENV_VOTE_CACHE, "1").strip().lower() not in' '    return os.environ.get(ENV_VOTE_CACHE, "0").strip().lower() not in'
check MK "$AP" '        while len(self.d) > VOTE_CACHE_MAX:' '        while False:'
check ML "$AP" '        states, gens = arena.slot_state_gens_np(eslots)     # states first, generations second' '        return True
        states, gens = arena.slot_state_gens_np(eslots)     # states first, generations second'

restore
echo "survivors=$survivors"
exit $survivors
