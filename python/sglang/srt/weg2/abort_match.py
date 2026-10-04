"""#1580 ABORT-EXACT-RID: one rid-match rule for every abort path of the Weg 2 dual layout.

The scheduler matches an AbortReq against requests with ``req.rid.startswith(abort.rid)`` (upstream:
a batch request's sub-rids share a prefix). The front names its rids ``weg2-<epoch>-<n>`` with a
counter, so the abort of ``weg2-0-1`` also hits the LIVE requests ``weg2-0-10`` ... ``weg2-0-19`` and
``weg2-0-100`` ... (and ``weg2-0-8`` hits 80-89, ``weg2-0-9`` hits 90-99). The front never sends a
sub-rid of a weg2 rid, so for a rid of exactly that shape an exact comparison is the only correct one.

Armed only when ``SGLANG_WEG2_ABORT_EXACT_RID`` is on AND the process is a dual-layout one
(``SGLANG_WEG2_DUAL_LAYOUT=1``): flip / INT8 / NF forms keep the prefix rule byte for byte. Every
rank reads the same environment and compares the same strings (no clock, no per-rank state), so the
decision is rank-uniform.

NOT fixed here (named in the report): an abort that names a rid the front has already re-issued for a
NEW instance (same string) still hits both -- that is the rid-reuse family (Q-698 / POP-KEEPS-TWIN),
an exact comparison cannot tell two objects of one rid apart.
"""

from __future__ import annotations

import os
import re

_WEG2_RID = re.compile(r"^weg2-\d+-\d+$")


def exact_armed() -> bool:
    """True when abort matching of weg2-shaped rids is exact (dual layout + switch)."""
    if str(os.environ.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() != "1":
        return False
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_ABORT_EXACT_RID.get())


def rid_hit(req_rid, abort_rid) -> bool:
    """Does an abort naming ``abort_rid`` hit the request ``req_rid``? Default (off, other forms,
    non-weg2 rid shapes): the upstream prefix rule."""
    a = str(abort_rid)
    r = str(req_rid)
    if _WEG2_RID.match(a) and exact_armed():
        return r == a
    return r.startswith(a)
