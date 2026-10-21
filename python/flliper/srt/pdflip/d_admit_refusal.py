"""D-ADMIT-REFUSED: every request group D's admission loop does NOT take is named.

Metal rc27 (D log boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwflbar1dauer10210206_893d09fa3f
_1021_020616.D.log, 02:15:44-02:15:56): pdflip-2-30 (kurz, u=181, prefix on D's device
at 176960) was not taken in 11 passes over 12.3 s. Each pass printed ``X-GATE ...
verdict=admit`` and then only ``#924D station=give_back ... site=admission_revert``;
the refusal itself had no line:

* the H105 group verdict's WAIT line (``tp_match_floor._note_admission_wait``) was
  throttled on a PROCESS-GLOBAL counter (n<=20 or n%64==0) that stood between 65 and
  127 for the whole window (n=64 at 02:15:05 for pdflip-2-27, n=128 at 02:16:27 for
  pdflip-2-38) -- the only trace was ``AFTER-WAIT refusals=11 waited_s=12.3
  host_price=4341 host_budget=81672`` once the request was in;
* the host's own gate (TP0, the decider: ``WORKER-GATE`` 0 lines in the boot) wrote
  nothing at all: the pre-lock lifetime refusal (``total_tokens >= rem_total_tokens``)
  only stores a tuple (``_note_lifetime_refusal``); ``GATE-BUDGET-PARTS`` fires only
  on the in-lock re-check (0 lines in the boot);
* the pass's skip census (``loop_skips(add_result_NO_TOKEN=...)``) reached the log
  only through PDFLIP-POST-WAKE-PASS lines.

SOLL v2 R7: a request under X that D does not take is a defect, never a flip reason
-- so the refusal must carry its gate and its numbers, every time it matters.

THE RULE. ``PrefillAdder.add_one_req`` stamps the gate that refused (``stamp``,
one name per return site, with that gate's own numbers) and the admission loop
calls ``note_not_added`` for every request it does not take (the #991 give-back
site), and ``note_admitted`` when a request it refused before is taken. The line
carries this rank's local gate, the GROUP verdict of the H105 exchange with the
decider's (host's / gather MIN's) price and budget -- identical on every rank by
construction -- the budget's terms, the X-gate's uncached/limit and ``soll=R7``
for a request under X.

THROTTLE PER REQUEST, NOT PER PROCESS: first 3 refusals of a rid, every change of
its (gate, verdict), refusal counts that are a power of two, and every 64th. One
``D-ADMIT-AFTER-REFUSALS`` line per refused rid once it is taken (refusals,
waited_s, per-gate counts). Counters in ``STATS`` (per gate, total).

Observation only: no decision reads anything written here, nothing is exchanged,
no collective; a failing read prints ``unreadable`` instead of raising. Group D
only (``FLLIPER_PDFLIP_GROUP=D``); ``FLLIPER_PDFLIP_D_ADMIT_REFUSAL=0`` switches the
lines off (default ON)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_D_ADMIT_REFUSAL"
MARK = "PDFLIP D-ADMIT-REFUSED"
MARK_AFTER = "PDFLIP D-ADMIT-AFTER-REFUSALS"
#: the adder attribute the gate stamp lives on (rebuilt every pass)
STAMP_ATTR = "_pdflip_d_refusal_stamp"
#: the scheduler attribute of the per-rid refusal ledger
STATE_ATTR = "_pdflip_d_refusal_state"
STATE_CAP = 512
#: the req attribute the X-gate leaves (uncached, limit) on
XGATE_ATTR = "_pdflip_xgate_priced"

STATS: Dict[str, int] = {}


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    if (e.get("FLLIPER_PDFLIP_GROUP", "") or "").strip().upper() != "D":
        return False
    return (e.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def _res_name(res: Any) -> str:
    n = getattr(res, "name", None)
    return str(n) if n is not None else str(res)


def stamp(adder: Any, req: Any, gate: str, res: Any = None, nums: Optional[dict] = None) -> None:
    """The gate that refuses ``req`` in this ``add_one_req`` call, with its numbers.
    Never raises (an instrument never stops the admission)."""
    try:
        setattr(adder, STAMP_ATTR, (str(getattr(req, "rid", "")), str(gate),
                                    _res_name(res) if res is not None else "-", dict(nums or {})))
    except Exception:  # noqa: BLE001
        pass


def clear(adder: Any) -> None:
    try:
        setattr(adder, STAMP_ATTR, None)
    except Exception:  # noqa: BLE001
        pass


def budget_parts(adder: Any) -> dict:
    """The terms ``PrefillAdder.rem_total_tokens`` is made of (the PW-W set of
    ``_pdflip_note_budget_parts``), read now. ``unreadable`` on any failure."""
    try:
        alloc = adder.token_to_kv_pool_allocator
        tc = adder.tree_cache
        ledger = getattr(adder, "commitment_ledger", None)
        parts = {
            "available": int(alloc.available_size()),
            "evictable": int(tc.evictable_size()),
            "offset": int(adder.rem_total_token_offset),
            "dcp_deficit": int(getattr(adder, "dcp_avail_deficit", 0) or 0),
            "floor": getattr(adder, "fundable_extend_floor", None),
            "ledger": (int(ledger.outstanding_tokens()) if ledger is not None else None),
        }
        try:
            from flliper.srt.mem_cache.common import deliverable_evictable_cap_aware_or

            parts["deliverable"] = int(deliverable_evictable_cap_aware_or(
                tc, getattr(tc, "full_evictable_size", tc.evictable_size)))
        except Exception:  # noqa: BLE001 -- optional term
            pass
        try:
            parts["rem_total"] = adder.rem_total_tokens
        except Exception:  # noqa: BLE001 -- optional term
            pass
        return parts
    except Exception as e:  # noqa: BLE001
        return {"unreadable": type(e).__name__}


def _fmt(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items()) if d else "-"


def _group_verdict(adder: Any, rid: str):
    """(code, price, budget) of the H105 exchange for ``rid`` in this pass, else None.
    The tuple is the decider's (host or gather MIN) -- the same on every rank."""
    follow = getattr(adder, "form_a_admission_follow", None)
    lv = getattr(follow, "last_verdict", None) if follow is not None else None
    if isinstance(lv, tuple) and len(lv) == 4 and str(lv[0]) == rid:
        return lv[1], lv[2], lv[3]
    return None


def _due(k: int, changed: bool) -> bool:
    return changed or k <= 3 or (k & (k - 1)) == 0 or k % 64 == 0


def note_not_added(sched: Any, adder: Any, req: Any, res: Any, *, site: str,
                   gave_back: Any = None, now: Optional[float] = None, env=None) -> Optional[str]:
    """The admission loop did not take ``req`` (``res`` = add_one_req's result).
    Returns the line it printed (None when throttled or off)."""
    if not enabled(env):
        return None
    try:
        rid = str(getattr(req, "rid", ""))
        st = getattr(adder, STAMP_ATTR, None)
        if isinstance(st, tuple) and len(st) == 4 and st[0] == rid:
            gate, local_code, nums = st[1], st[2], st[3]
        else:
            gate, local_code, nums = "unlabeled", "-", {}
        gv = _group_verdict(adder, rid)
        if gv is not None and gate == "unlabeled":
            # the group refused although this rank's gate let it through: the
            # decider's (host's / gather MIN's) gate, numbers below
            gate = "h105_group_verdict"
        code = _res_name(res)
        t = time.monotonic() if now is None else float(now)
        book = sched.__dict__.setdefault(STATE_ATTR, {})
        ent = book.get(rid)
        if ent is None:
            if len(book) >= STATE_CAP:
                book.pop(next(iter(book)))
            ent = book[rid] = {"t0": t, "k": 0, "gates": {}, "last": None}
        ent["k"] += 1
        ent["gates"][gate] = ent["gates"].get(gate, 0) + 1
        key = (gate, code)
        changed = ent["last"] != key
        ent["last"] = key
        STATS[gate] = STATS.get(gate, 0) + 1
        STATS["total"] = STATS.get("total", 0) + 1
        k = ent["k"]
        if not _due(k, changed):
            return None
        xg = getattr(req, XGATE_ATTR, None)
        if isinstance(xg, tuple) and len(xg) == 2:
            uncached, limit = int(xg[0]), int(xg[1])
            x_part = f"uncached={uncached} limit={limit} under_x={int(uncached <= limit)}"
            soll = " soll=R7" if uncached <= limit else ""
        else:
            x_part, soll = "uncached=? limit=?", ""
        gv_part = (
            "group=-" if gv is None
            else f"group={gv[0]} decider_price={gv[1]} decider_budget={gv[2]}"
        )
        line = (
            f"{MARK} rid={rid} res={code} gate={gate} local={local_code} {gv_part} "
            f"{_fmt(nums)} {x_part} parts=[{_fmt(budget_parts(adder))}] "
            f"site={site} gave_back={gave_back if gave_back is not None else '-'} "
            f"refusals={k} waited_s={t - ent['t0']:.1f} gate_n={STATS[gate]} "
            f"total={STATS['total']}{soll} -- D's admission did not take this request "
            f"this pass; the gate and its numbers are this rank's, group= is the "
            f"decider's (identical on every rank)"
        )
        logger.info(line)
        return line
    except Exception:  # noqa: BLE001 -- an instrument never stops the admission
        return None


def note_admitted(sched: Any, req: Any, *, now: Optional[float] = None, env=None) -> Optional[str]:
    """``req`` was taken: one line when it had been refused before."""
    if not enabled(env):
        return None
    try:
        book = sched.__dict__.get(STATE_ATTR)
        if not book:
            return None
        rid = str(getattr(req, "rid", ""))
        ent = book.pop(rid, None)
        if ent is None:
            return None
        t = time.monotonic() if now is None else float(now)
        gates = ",".join(f"{g}:{n}" for g, n in sorted(ent["gates"].items()))
        line = (f"{MARK_AFTER} rid={rid} refusals={ent['k']} waited_s={t - ent['t0']:.1f} "
                f"gates={{{gates}}}")
        logger.info(line)
        return line
    except Exception:  # noqa: BLE001
        return None
