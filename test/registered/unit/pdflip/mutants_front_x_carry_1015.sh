#!/usr/bin/env bash
# FRONT-X-MIT 1015 mutation check. One plausible WRONG implementation per decision; each MUST turn
# test_pdflip_front_x_carry_1015.py red. A survivor = that decision is asserted nowhere.
# ONLY=<name> runs one mutant. Runs through /spinning/gpu-arb/pytest_gedeckelt.sh (CPU only). Restores every file on exit.
#
#   MA  D's gate ignores the front X (limit = static X)                        -> the silent compute between X and 12288 returns
#   MB  tolerance 0 (limit = front_x)                                          -> a request priced AT its X is bounced by page rounding
#   MC  tolerance = one chunk (2880) instead of one page                       -> the very mispricings this exists for are swallowed
#   MD  the static ceiling is dropped (limit = front_x + page)                 -> a front X above 12288 raises D's own bound
#   ME  a resumed request (it decoded) is bounded by the front X again          -> D-park resumes bounce D -> P
#   MF  the front sends no header (request_headers returns {})                 -> D never sees the X
#   MG  leg 2's D post drops the headers (no **_post_kw)                       -> the X is computed and thrown away
#   MH  QUEUED-SHORT does not record its X                                     -> the queued SHORT is bounded by the arrival X_k
#   MI  the first decision wins instead of the last                            -> a queued SHORT keeps its arrival X_k
#   MJ  a leg after P (pending, not d_direct) also carries the X               -> D's read-back of P's KV is checked against X
#   MK  the W50 body loses the sentence the front parses ("extent after prefix") -> the front cannot price the re-route
#   ML  the serving layer does not stamp the headers on the request            -> headers parsed, then dropped
#   MM  a malformed / zero X header is read as X=0 (refuse everything)         -> one bad header refuses every prefill
# review dx-r2 (A1..A5), the four that survived the first test set + the /generate route:
#   MR3 the scheduler resets req.weg2_front_x after the copy                   -> the gate falls back to the static X, no line
#   MR4 the tokenizer manager drops the field (``obj.weg2_front_x if False``)   -> same, one hop earlier
#   MR5 the capacity-park vote runs on the static X                            -> park / re-read / re-gate loop on D until the park bound
#   MR7 D-SHORT-DRAIN stamps the X in force instead of its own (busy) X        -> a busy-D drain may compute up to the idle X
#   MS  the native /generate route does not stamp the headers (A1)             -> a D-direct /generate is bounded by the static X alone
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../.." && pwd)}
FX=$ROOT/python/flliper/srt/pdflip/front_x.py
FR=$ROOT/python/flliper/srt/pdflip/front.py
SC=$ROOT/python/flliper/srt/managers/scheduler.py
CH=$ROOT/python/flliper/srt/entrypoints/openai/serving_chat.py
TM=$ROOT/python/flliper/srt/managers/tokenizer_manager.py
HS=$ROOT/python/flliper/srt/entrypoints/http_server.py
TESTS=test/registered/unit/pdflip/test_pdflip_front_x_carry_1015.py
GED=${GED:-/spinning/gpu-arb/pytest_gedeckelt.sh}

BAK_DIR=$(mktemp -d)
cp "$FX" "$BAK_DIR/fx.py"; cp "$FR" "$BAK_DIR/fr.py"; cp "$SC" "$BAK_DIR/sc.py"; cp "$CH" "$BAK_DIR/ch.py"
cp "$TM" "$BAK_DIR/tm.py"; cp "$HS" "$BAK_DIR/hs.py"
restore() {
  cp "$BAK_DIR/fx.py" "$FX"; cp "$BAK_DIR/fr.py" "$FR"; cp "$BAK_DIR/sc.py" "$SC"; cp "$BAK_DIR/ch.py" "$CH"
  cp "$BAK_DIR/tm.py" "$TM"; cp "$BAK_DIR/hs.py" "$HS"
}
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

check MA "$SC" '        limit, _front_x, _front_priced = _pdflip_front_x.resolve_limit(req, x, getattr(self, "page_size", None))
        verdict = "W31" if uncached > limit else "admit"' '        limit, _front_x, _front_priced = _pdflip_front_x.resolve_limit(req, x, getattr(self, "page_size", None))
        verdict = "W31" if uncached > x else "admit"'
check MB "$FX" '    return min(sx, int(front_x) + tolerance_tokens(page_size))' '    return min(sx, int(front_x) + 0)'
check MC "$FX" '    return min(sx, int(front_x) + tolerance_tokens(page_size))' '    return min(sx, int(front_x) + 2880)'
check MD "$FX" '    return min(sx, int(front_x) + tolerance_tokens(page_size))' '    return int(front_x) + tolerance_tokens(page_size)'
check ME "$FX" '        if getattr(req, "output_ids", None):
            return None, None
        x = getattr(req, ATTR_X, None)' '        x = getattr(req, ATTR_X, None)'
check MF "$FX" '    if xi <= 0 or not enabled():
        return {}' '    if True:
        return {}'
check MG "$FR" '                                         **_post_kw) as r:' '                                         ) as r:'
check MH "$FR" '            Front._stamp_x_decision(p, x)  # FRONT-X-MIT: QUEUED-SHORT'"'"'s X (the line'"'"'s X=)
' ''
check MI "$FR" '            d[str(rid)] = (int(x), None if priced is None else int(priced))' '            d.setdefault(str(rid), (int(x), None if priced is None else int(priced)))'
check MN "$FR" '            Front._note_x_decision(self, rid, *p.x_decision)  # FRONT-X-MIT: the hand-over'"'"'s X rides with leg 2' '            pass'
check MJ "$FR" '        if single_prefill or (pending is not None and not getattr(pending, "d_direct", False)):
            return {}' '        if single_prefill:
            return {}'
check MK "$SC" 'f"tokens itself (--tp-prefill-max-tokens); this request'"'"'s extent after "
                f"prefix matching is {uncached}. Refused by name so the caller re-routes it "' 'f"tokens itself (--tp-prefill-max-tokens); this request'"'"'s extent is "
                f"{uncached}. Refused by name so the caller re-routes it "'
check ML "$CH" '        weg2_front_x.stamp_request(adapted_request, raw_request)
' '        pass
'
check MM "$FX" '    if not x:
        return None, None
    return x, priced' '    if x is None:
        return 0, priced
    return x, priced'

check MR3 "$SC" '            req.weg2_front_priced = getattr(recv_req, "weg2_front_priced", None)
' '            req.weg2_front_priced = getattr(recv_req, "weg2_front_priced", None)
            req.weg2_front_x = None
'
check MR4 "$TM" '                weg2_front_x=obj.weg2_front_x,
' '                weg2_front_x=obj.weg2_front_x if False else None,
'
check MR5 "$SC" '                    r, _pdflip_front_x.resolve_limit(r, x, getattr(self, "page_size", None))[0], sched=self' '                    r, x, sched=self'
check MR7 "$FR" "            Front._stamp_x_decision(p, x)  # FRONT-X-MIT: the drain's X" "            Front._stamp_x_decision(p, self.tp_prefill_max_tokens)  # FRONT-X-MIT: the drain's X"
check MS "$HS" '    weg2_front_x.stamp_request(obj, request)
' '    pass
'

echo "survivors=$survivors"
exit $survivors
