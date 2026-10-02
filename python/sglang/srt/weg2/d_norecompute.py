"""D-NORECOMPUTE (user law 02.10.: after a flip D computes nothing -- E2,
HANDBACK d_compute=0).

A request the #1471 settle releases after the wake with a SHORT store read
used to go to D's extend, which computed ``[delivered, N)`` on D:

* y6z weg2-12-64 (08:24:58): the wake read delivered 45568 of 51520 (the sleep
  flush had lost 13 anchors, see unified_radix_cache ``_weg2_flush_spill_room``),
  the settle lapsed, D computed 5952 tokens in 3 chunks (08:25:50-08:26:02),
  every chunk 1.2-1.8 s in pool.host_fetch;
* y6z weg2-16-79 (08:27:29): delivered 54144 of 54336, '#x38 SETTLE-TAIL'
  released the 192-token tail to the extend: 201 tokens took 1.56 s and held
  five woken seats from decoding.

Now the release asks :func:`d_would_compute`: 0 when the read is whole or an
agreed F4 PARK-END / E2 tail covers it (``delivered >= page_prefix``), else the
tokens D would compute. Such a request goes through RESUME-VIA-P instead
(``resume_via_p.keep_on_d`` reason ``store_short_after_flip``): P prefills its
context from the shared store, D resumes it after the flip back under E2. A
request RESUME-VIA-P cannot take (non-stream, multimodal, attempts spent) is
released as before and NAMED (``D-NORECOMPUTE-FALLBACK``), never silently.

Switch ``SGLANG_WEG2_D_NORECOMPUTE`` (default on; 0 = the release as before).
Rank-uniform: the release list is the group's MIN verdict, ``delivered`` the
synced prefix, the agreed tail the group's answer.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_D_NORECOMPUTE"
MARK = "WEG2 D-NORECOMPUTE"
REASON = "store_short_after_flip"


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def d_would_compute(req: Any) -> int:
    """Tokens D's extend would compute for ``req`` after its short read
    (0 = whole read, no short-read stamp, or an agreed tail that adopts)."""
    delivered = getattr(req, "_weg2_store_delivered", None)
    ids = getattr(req, "full_untruncated_fill_ids", None)
    if delivered is None or ids is None:
        return 0
    rem = max(0, len(ids) - int(delivered))
    if rem <= 0:
        return 0
    try:
        from sglang.srt.weg2 import tail_adopt as _ta

        entry = _ta._AGREED.get(str(getattr(req, "rid", "")))
        if entry is not None and entry.agreed and int(delivered) >= int(entry.staged.spec.page_prefix):
            return 0  # the F4 / E2 tail adopts [page_prefix, N): no target forward
    except Exception:  # noqa: BLE001 -- no tail module: the remainder stands
        pass
    return rem


def divert(sched: Any, req: Any, rem: int) -> bool:
    """RESUME-VIA-P for a released short read. True = kept parked for P (the
    caller does not queue it); False = not eligible (released, named)."""
    from sglang.srt.weg2 import resume_via_p as _rvp

    x = int(getattr(getattr(sched, "server_args", None), "tp_prefill_max_tokens", 0) or 0)
    if not _rvp.eligible(req, sched=sched):
        logger.warning("%s-FALLBACK rid=%s would_compute=%d (RESUME-VIA-P not eligible: non-stream, "
                       "multimodal or attempts spent) -- released to D's extend, named",
                       MARK, str(getattr(req, "rid", "?"))[:24], int(rem))
        return False
    _rvp.keep_on_d(sched, req, int(rem), x, reason=REASON)
    try:
        from sglang.srt.weg2 import d_park_runtime as _dpr

        # an awake re-queue gives it back to the settle, never to the X gate
        setattr(req, _dpr.FROM_SETTLE_ATTR, True)
    except Exception:  # noqa: BLE001
        pass
    logger.info("%s rid=%s would_compute=%d delivered=%s -> RESUME-VIA-P (P prefills from the store, "
                "D resumes under E2 after the flip back; no D compute after a flip)",
                MARK, str(getattr(req, "rid", "?"))[:24], int(rem),
                getattr(req, "_weg2_store_delivered", None))
    return True
