"""ZR-4 D-TWIN-PASS (01.10.): two requests in ONE D admission pass do not
compute the same tokens twice.

Metal y6h (D log ...dauer10011531, TP0, 15:42:50): ``#969 EXTENT n=51 reqs=[
('weg2-10-', 42880, 44402, 42880, 1522), ..., ('weg2-10-', 42880, 44231,
42880, 1351)]`` -- two turns of one agent, both matched D's tree to 42880,
both extended from there in the same forward. Whatever their ids share
beyond 42880 was computed twice. P has TW (``p_twin_defer``: the later twin
waits at its intake until the first one published); D's admission had no
equivalent, its X-route and resume turns go straight into the adder.

THE RULE, inside ``PrefillAdder.add_one_req`` (D group only): a request whose
own extend would compute at least ``min_tokens`` tokens that a request
already in this pass computes too waits ONE pass (``AddReqResult.OTHER``).
The earlier one's forward inserts its pages into the tree; the next pass
matches the waiter deeper and it extends only what is its own. Nothing
waits on a decode, a flip or another group; the waiter loses at most one
forward and saves the shared span's compute.

Every term is a pure function of the requests' token ids, their matched
prefixes and the adder's order -- replicated identically on every rank of
the group, so the verdict is group-uniform without a collective.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

from sglang.srt.weg2.p_twin_defer import shared_prefix_len

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_D_TWIN_PASS"
ENV_MIN_TOKENS = "SGLANG_WEG2_D_TWIN_PASS_MIN_TOKENS"
#: one KV page: below that the deeper match cannot take a page anyway
DEFAULT_MIN_TOKENS = 64
MARK = "WEG2-D-TWIN-PASS"

_N = [0]


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    if (e.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() != "D":
        return False
    return (e.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def min_tokens(env=None) -> int:
    e = os.environ if env is None else env
    try:
        return max(1, int(e.get(ENV_MIN_TOKENS, "") or DEFAULT_MIN_TOKENS))
    except ValueError:
        return DEFAULT_MIN_TOKENS


def _ids(req):
    ids = getattr(req, "full_untruncated_fill_ids", None)
    if ids is None:
        ids = getattr(req, "origin_input_ids", None)
    return ids if ids is not None else ()


def overlap(req, prefix_len: int, other) -> int:
    """Tokens ``req``'s extend from ``prefix_len`` shares with what ``other``
    computes in this pass (its admitted ``extend_range``)."""
    if getattr(req, "extra_key", None) != getattr(other, "extra_key", None):
        return 0
    rng = getattr(other, "extend_range", None)
    if rng is None:
        return 0
    o_start, o_end = int(rng.start), int(rng.end)
    lo = max(int(prefix_len), o_start)
    if o_end <= lo:
        return 0
    ia, ib = _ids(req), _ids(other)
    if len(ia) <= lo or len(ib) <= lo or ia[lo] != ib[lo]:
        return 0  # O(1): the two diverge at the first token both would compute
    shared = shared_prefix_len(ia, ib)
    return max(0, min(shared, o_end) - lo)


def waits(req, prefix_len: int, admitted: Iterable, *, env=None) -> bool:
    if not enabled(env):
        return False
    need = min_tokens(env)
    for other in admitted:
        if other is req:
            continue
        n = overlap(req, prefix_len, other)
        if n >= need:
            _N[0] += 1
            k = _N[0]
            if k <= 20 or k % 200 == 0:
                logger.info(
                    "%s rid=%s twin=%s prefix=%d shared=%d n=%d -- the twin computes these tokens in "
                    "this pass; this one waits one pass and matches them from the tree",
                    MARK, str(getattr(req, "rid", "?"))[:16], str(getattr(other, "rid", "?"))[:16],
                    int(prefix_len), n, k)
            return True
    return False
