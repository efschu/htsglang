"""#1580 ABORT-EXACT-RID (27B port 24.10., #DPJ-S6): one rid-match rule for every abort path.

Upstream matches an AbortReq with ``req.rid.startswith(abort.rid)`` because the sub-requests of a
parallel-sampling / batch request share the parent rid as a prefix (``io_struct._normalize_rid``:
``f"{rid}_{i}"``). The front names its rids ``pdflip-<epoch>-<n>`` with a counter, so the abort of
``pdflip-0-2`` also hit the LIVE ``pdflip-0-20`` .. ``pdflip-0-29`` and ``pdflip-0-200`` .. on every
path that used the prefix rule (scheduler waiting queue / chunked / in-flight, #1180-W hold, K14
stamp, PP-SKIPOUT ``mark_aborts``, D park, tail adopt/handoff, intake stall, OOM retreat, told maps).

THE RULE. For an abort rid of exactly the front's shape ``pdflip-<digits>-<digits>`` a request rid
is hit when it IS that rid, or when it continues it with a NON-digit (a defined child rid such as
``pdflip-0-2_1`` of ``n>1``). ``pdflip-0-2`` therefore no longer hits ``pdflip-0-20``. Every other
rid shape (uuid hex, user rids) keeps the upstream prefix rule byte for byte, and so does an empty
abort rid. Pure string function of the two rids and a boot-constant switch: rank-uniform (SOLL v2
R11 -- every rank decides the same from the same wire bytes).

Switch ``FLLIPER_PDFLIP_ABORT_EXACT_RID`` (default ON, user order 14.10.; ``0`` = the prefix rule).

NOT covered (named): two objects of ONE rid (rid reuse after P-PAUSE/RESUME) -- an exact comparison
cannot tell them apart; that is the Q-693/Q-697/#DPJ-W/#1430q family.
"""

from __future__ import annotations

import os
import re

ENV = "FLLIPER_PDFLIP_ABORT_EXACT_RID"
_FRONT_RID = re.compile(r"^pdflip-\d+-\d+$")
_ARMED = None


def exact_armed() -> bool:
    """Boot constant (read once per process): default on, ``0``/``false``/``off`` = prefix rule."""
    global _ARMED
    if _ARMED is None:
        v = str(os.environ.get(ENV, "1")).strip().lower()
        _ARMED = v not in ("0", "false", "off", "no")
    return _ARMED


def _reset_for_tests() -> None:
    global _ARMED
    _ARMED = None


def rid_hit(req_rid, abort_rid) -> bool:
    """Does an abort naming ``abort_rid`` hit the request (or key) ``req_rid``?"""
    a = str(abort_rid if abort_rid is not None else "")
    r = str(req_rid if req_rid is not None else "")
    if not r.startswith(a):
        return False
    if len(r) == len(a) or not a:
        return True
    if r[len(a)].isdigit() and _FRONT_RID.match(a) and exact_armed():
        return False  # pdflip-0-2 vs pdflip-0-2x: a different request
    return True
