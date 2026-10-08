"""HANDBACK (01.10., user via 27B: "den anker fix koennen alle brauchen"): what
group D holds and computes after a P hand-off -- the NF side of 27B fe5c55041b.

NF's contract is not 27B's N-1. P computes all N prompt tokens, publishes the
END section (rows [page_prefix, N) + the state after N + its sampled token,
H24 E2 / H63 fold) and D takes it without a target forward: d_prefix = N,
d_compute = 0, the END key checked against the prompt (``tail_key``) and the
rows by digest. Measured on the last 12 NF boots (D TP0, exact rid pairing):
492 of 565 hand-offs (87 %) ran exactly that (TAIL-READY adopt=done, SKIP-EXTEND
prefix=N). The rest fell back to an extend from the store's page anchor
(floor_page(N-2), the CLAIM ANCHOR P files its recurrent state at): 36 with no
tail at all (park/resume shapes), 25 ``no_parts``, 10 ``end_only:batch_not_empty``
(the END state dropped in a batch that already runs a forward), 4 park windows.

So 27B's other half does NOT come to NF:
  * the D claim stays the upstream N-1 RAW tokens (= N-2 bigram units) --
    P's CLAIM ANCHOR (floor_page(N-2)) and D's claim are one geometry, and the
    exact keying + E2 already deliver the state after N. Widening the claim on
    an exact-bigram D (27B ``handback_bigram_claim``) would move D's match
    without moving P's anchor;
  * the L3 store identity stays byte for byte (NF has run the exact keying
    since fe031f2c80; its directory holds exact anchors already).

What does come: a hand-off read is read whatever its length (below).

27B line (fe5c55041b, HANDBACK N-1) on the same module -- its claim half is
gated to the 27B profile (``_claim_n_on``), NF keeps the upstream claim:
HANDBACK N-1 (01.10.; user: "den anker fix koennen alle brauchen"): the hand-back
claim of a P-prefilled prompt is N-1 tokens -- every P->D hand-back, the flip
form (P=PP3 -> D=TP3) and the dual layout alike. First found in the dual layout:

Metal dual1m (...10011503_ef9b3a899c): P cut pdflip-0-1 to N-1 = 41187 tokens
(P-TRIM-END-ANCHOR) but published 41186 page keys (#1442 HANDOFF page_keys=41186,
END-ANCHOR units=41186/41186), D read 41186 (PROBE-HOLD held=41186, EXTENT
anchor_depth=41186) and kept uncached=2 -> with ef9b3a899c's X=1 every hand-back
was refused (W50, W35 re-queue loop, W53). Not new: dual1m before showed
'#new-token: 2, #cached-token: 4059' on every hand-back; X >= 4096 hid it.

ROOT: the 27B tree keys BIGRAMS (DFlash/EAGLE keys: r raw tokens = r-1 units) with
the UPSTREAM keying. (1) A reader claims at most N-1 raw tokens
(Req._compute_max_prefix_len) = N-2 units. (2) P's trimmed finish insert keyed its
N-1 committed tokens with N-2 units (the exact key's next token, the held-back
prompt token, was not in the ids). So both sides stopped one token short.

FIX, every 27B P->D hand-back:
  * the EXACT bigram keying (the qwen27b profile's bigram_anchor_exact, the NF
    keying, metal-proven there): a node of k units holds the KV of k tokens AND
    the state after k tokens; the L3 store identity names the keying
    (launcher.l3_persist_identity anchor_keying), so a store written under the
    upstream keying -- whose anchors carry the state after k+1 tokens -- is a
    new directory, never a wrong hit;
  * P's trimmed finish insert takes the held-back prompt token as the key's next
    token -> N-1 units = the N-1 tokens P computed, state after N-1 tokens;
  * D's claim (this module) is N raw tokens = N-1 units: a bigram key of the whole
    prompt still leaves token N-1 to forward (its KV is in no unit), which is the
    invariant the upstream N-1 raw limit protects for NON-bigram keys.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

HANDBACK_MARK = "PDFLIP-HANDBACK"
_N = [0]

#: set by UnifiedRadixCache when this process's tree resolves to bigram keys
#: with the exact keying (bigram_anchor_exact True) -- AT CONSTRUCTION of the tree:
#: the claim reads it before the request's own match_prefix, so a note left to the
#: first match read it cold on the first hand-back after a warmup-less boot
BIGRAM_EXACT_TREE = [False]


def note_tree(is_bigram_exact: bool) -> None:
    if is_bigram_exact:
        BIGRAM_EXACT_TREE[0] = True


def handback_bigram_claim(env=None) -> bool:
    """True on a group-D rank whose tree keys exact bigrams (flip form and dual
    layout alike): a reader may claim N raw tokens (N-1 units)."""
    e = os.environ if env is None else env
    return (BIGRAM_EXACT_TREE[0]
            and (e.get("FLLIPER_PDFLIP_GROUP", "") or "").strip().upper() == "D"
            and _claim_n_on(e))


def _claim_n_on(e) -> bool:
    """RELEASE-HEAD 1002: the N-claim is the 27B contract, not NF's (NF D keeps
    the upstream N-1 raw claim: P's CLAIM ANCHOR floor_page(N-2) is filed for it,
    NF desk 1160d65e1d). Both profiles key exact bigrams now, so the tree note
    alone cannot tell them apart -- the profile switch
    FLLIPER_PDFLIP_HANDBACK_CLAIM_N does (pdflip/form.py ``handback_claim_n``:
    qwen27b on, nextflash off; an explicit env value wins)."""
    raw = (e.get("FLLIPER_PDFLIP_HANDBACK_CLAIM_N", "") or "").strip().lower()
    if raw:
        return raw in ("1", "true", "yes", "on")
    if e is os.environ:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_HANDBACK_CLAIM_N.get())
    from flliper.srt.pdflip.form import profile_switch_default

    return bool(profile_switch_default("FLLIPER_PDFLIP_HANDBACK_CLAIM_N", True, environ=e))


def handback_min_tokens(has_handoff: bool = False, env=None):
    """1 for a P hand-off read on a group-D rank -- a request carrying P's
    #1442 hand-off chain, or any D request of the dual layout -- whatever its
    length; else None (the tree's #915 threshold, unchanged for D's own reads).

    Metal NF (12 boots, D TP0): 33x '#915 PREFETCH REFUSED reason=vote_negative
    need=64..255 keys=handoff' -- pages P had written and handed over, refused as
    "too short" by the 256-token threshold, and D prefilled them again. The
    threshold prices whether opening a FRESH read pays; a hand-off is not fresh,
    P's pages are the request's own prefix (27B fe5c55041b, NF y4a TS for the
    told read: same reasoning, ``pdflip_store_told.told_read_min_tokens``)."""
    e = os.environ if env is None else env
    if (e.get("FLLIPER_PDFLIP_GROUP", "") or "").strip().upper() != "D":
        return None
    if has_handoff or (e.get("FLLIPER_PDFLIP_DUAL_LAYOUT", "") or "").strip() == "1":
        return 1
    return None


def handback_line(rid, n_tokens: int, d_prefix: int, d_compute: int, path: str) -> str:
    """One line per hand-off at D's admission: what D holds (``d_prefix``) and
    what its target forward still computes (``d_compute``), by path:
    ``skip`` (E2: P's END state + token, contract d_prefix = N, d_compute = 0),
    ``e1`` (rows to c, extend [c, N)), ``extend:<why>`` (the tail refused:
    today's extend from the page anchor). Any d_compute > 0 also writes a
    ``PDFLIP-HANDBACK-DEFECT`` line (below)."""
    _N[0] += 1
    line = (
        f"{HANDBACK_MARK} rid={str(rid)[:24]} N={int(n_tokens)} d_prefix={int(d_prefix)} "
        f"d_compute={int(d_compute)} path={path} (contract NF: E2 d_prefix = N, d_compute = 0; "
        f"else d_compute = N - d_prefix) n={_N[0]}"
    )
    logger.info(line)
    rec = take_origin(rid) or {}
    if int(d_compute) > 0:
        # a tail was agreed (P's hand-off or D's park part) and still not taken
        # whole: every token past d_prefix is computed again
        defect(rid, path, path.split(":", 1)[-1], n_tokens=n_tokens, d_prefix=d_prefix,
               d_compute=d_compute, kind=rec.get("kind") or "tail")
    return line


# -- ZR (01.10., user: "warum koennen wir bei nf das nicht so bauen, dass es auch
# 0 token neu berechnet? ... nur die software, nicht die physik"; then: "below
# 100 % path=skip is a bug") -----------------------------------------------------
#
# Every token a P hand-off or a D park/resume makes group D compute AGAIN is a
# software defect: P computed all N (E2 hands the state after N over), a D park
# keeps its END state (F4). A request D never computed -- a fresh turn routed
# straight to D (X route) -- computes its new tokens there for the first time;
# it has no origin below and never writes a DEFECT line. The metal reads the
# rate directly: target 0 lines.
DEFECT_MARK = "PDFLIP-HANDBACK-DEFECT"
ORIGIN_HANDOFF = "handoff"  # P handed the request over (#1442 chain resolved on D)
ORIGIN_PARK = "park"  # D parked it (park_running)
_ORIGIN_CAP = 4096
_ORIGIN: dict = {}  # rid -> {"kind": ..., "why": ...}
_DEFECT_N = [0]


def note_origin(rid, kind: str) -> None:
    """``rid`` must come back with 0 tokens computed again: ``kind`` =
    ``handoff`` or ``park`` (the latest wins: a hand-off parked later is a park)."""
    if not rid:
        return
    key = str(rid)
    prev = _ORIGIN.pop(key, None) or {}
    _ORIGIN[key] = {"kind": str(kind), "why": prev.get("why", "")}
    while len(_ORIGIN) > _ORIGIN_CAP:
        _ORIGIN.pop(next(iter(_ORIGIN)))


def origin(rid):
    """The origin record of ``rid`` (None = D computes it for the first time)."""
    return _ORIGIN.get(str(rid))


def note_why(rid, why: str) -> None:
    """Name WHY the tail of an origin request cannot be taken (``no_parts`` ...);
    the admission's DEFECT line carries it."""
    rec = _ORIGIN.get(str(rid))
    if rec is not None:
        rec["why"] = str(why)


def take_origin(rid):
    """The admission decided this request: one verdict per resume."""
    return _ORIGIN.pop(str(rid), None)


def defect(rid, path: str, why: str, *, n_tokens=None, d_prefix=None, d_compute=None, kind=None) -> str:
    """One ``PDFLIP-HANDBACK-DEFECT`` line: a hand-off or a resume that computes
    tokens already computed (by P, or by D before the park). Log only."""
    _DEFECT_N[0] += 1
    line = (
        f"{DEFECT_MARK} rid={str(rid)[:24]} origin={kind or '?'} path={path} why={why} "
        f"N={'-' if n_tokens is None else int(n_tokens)} "
        f"d_prefix={'-' if d_prefix is None else int(d_prefix)} "
        f"d_compute={'-' if d_compute is None else int(d_compute)} "
        f"(contract: a hand-off or a resume computes 0 tokens again; target 0 lines) n={_DEFECT_N[0]}"
    )
    logger.warning(line)
    return line


def admission_without_tail(rid, fill_len: int, prefix_len: int):
    """The admission of a request with NO agreed tail. For an origin request
    every token from ``prefix_len`` on is computed again -- a DEFECT, named by
    the reason the staging left (``note_why``) or ``no_staging``. Silent for a
    request without an origin (first compute) and when nothing is computed."""
    rec = take_origin(rid)
    if rec is None:
        return None
    d_compute = max(0, int(fill_len) - int(prefix_len))
    if d_compute <= 0:
        return None
    why = rec.get("why") or "no_staging"
    return defect(rid, f"extend:{why}", why, n_tokens=fill_len, d_prefix=prefix_len,
                  d_compute=d_compute, kind=rec.get("kind"))
