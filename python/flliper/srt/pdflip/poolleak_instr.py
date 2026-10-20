"""POOLLEAK-INSTR: log-only instruments for the D pool leak after a park.

NF y9nf6 boot 3 (191228abc5, D log ...10040650_191228abc5_1004_065037, 07:03:50Z):
right after ``PDFLIP-D-PARK park_running epoch=24`` (3 running retracted, span
retained) and a non-blocking flush, TP1/TP2 died in ``on_idle``:
``pool memory leak detected! [full] total=524288, available=0,
evictable=416640, withheld=92672 ... deficit of 128 row(s)`` and
``[mamba] total=38, available=18, evictable=16`` with ``leaked_mamba_pages=
{33, 34, 31}``. The enumerated ``leaked_full_pages`` held 92800 ids = the
92672 KvRowCap-withheld ids + the 128 that nobody owns -- which 128 cannot be
read off the line. Three measurements make the next boot name them:

1. :func:`park_snapshot` -- per running request (pool row, mamba slot,
   prefix / committed / allocated lengths) and the pool ledger, BEFORE and
   AFTER ``retract_all`` in ``d_park_runtime.park_running``;
2. :func:`ledger_line` -- the idle checker's equation, read-only, one line
   after every park (never raises);
3. :func:`unwithheld_leak_ids` -- in the leak report: expected ids minus the
   free lists, the tree's values AND the KvRowCap-withheld ids, as ranges.

Switch ``FLLIPER_PDFLIP_POOLLEAK_INSTR`` (default OFF). Off: every entry point
returns before reading anything. On: reads and logs only -- no allocator,
tree or request state is written. Every reader is guarded: an instrument
never takes a rank down.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

TAG = "POOLLEAK-INSTR"


def enabled() -> bool:
    try:
        from flliper.srt.environ import envs

        return bool(envs.FLLIPER_PDFLIP_POOLLEAK_INSTR.get())
    except Exception:  # noqa: BLE001 -- an instrument never takes a rank down
        return False


def _int(fn, default=None):
    try:
        v = fn()
        return None if v is None else int(v)
    except Exception:  # noqa: BLE001
        return default


def ranges(ids: Iterable[int], limit: int = 64) -> str:
    """``[a-b(n), c(1), ...]`` over sorted ids; at most ``limit`` runs."""
    xs = sorted(int(x) for x in ids)
    if not xs:
        return "[]"
    runs: List[Tuple[int, int]] = []
    a = b = xs[0]
    for x in xs[1:]:
        if x == b + 1:
            b = x
            continue
        runs.append((a, b))
        a = b = x
    runs.append((a, b))
    parts = [f"{a}-{b}({b - a + 1})" if b > a else f"{a}(1)" for a, b in runs[:limit]]
    more = f" +{len(runs) - limit} runs" if len(runs) > limit else ""
    return "[" + ", ".join(parts) + "]" + more


def _allocator(sched):
    return getattr(sched, "token_to_kv_pool_allocator", None)


def ledger(sched) -> dict:
    """The idle checker's terms (available + evictable + protected + withheld
    vs total, full and mamba), read only."""
    alloc = _allocator(sched)
    tc = getattr(sched, "tree_cache", None)
    out = {}
    if alloc is not None:
        out["full_total"] = _int(lambda: alloc.size)
        out["full_available"] = _int(alloc.available_size)
        out["full_withheld"] = _int(lambda: getattr(alloc, "residency_withheld_slots", 0) or 0, 0)
    if tc is not None:
        mamba_tree = _int(tc.supports_mamba, 0) if hasattr(tc, "supports_mamba") else 0
        if mamba_tree:
            out["full_evictable"] = _int(tc.full_evictable_size)
            out["full_protected"] = _int(tc.full_protected_size)
            out["mamba_evictable"] = _int(tc.mamba_evictable_size)
            out["mamba_protected"] = _int(tc.mamba_protected_size)
        else:
            out["full_evictable"] = _int(tc.evictable_size)
            out["full_protected"] = _int(tc.protected_size)
    terms = ("full_available", "full_evictable", "full_protected", "full_withheld")
    if out.get("full_total") is not None and all(out.get(t) is not None for t in terms):
        out["full_unowned"] = out["full_total"] - sum(out[t] for t in terms)
    rtp = getattr(sched, "req_to_token_pool", None)
    m_alloc = getattr(rtp, "mamba_allocator", None)
    m_pool = getattr(rtp, "mamba_pool", None)
    if m_alloc is not None:
        out["mamba_total"] = _int(lambda: m_pool.size) if m_pool is not None else None
        out["mamba_available"] = _int(m_alloc.available_size)
        withheld = getattr(m_alloc, "phase_withheld_slots", None)
        out["mamba_withheld"] = _int(withheld, 0) if callable(withheld) else 0
        used = getattr(m_alloc, "slot_used", None)
        if used is not None:
            out["mamba_slot_used"] = _int(lambda: used.sum().item())
        mt = ("mamba_available", "mamba_evictable", "mamba_protected", "mamba_withheld")
        if out.get("mamba_total") is not None and all(out.get(t) is not None for t in mt):
            out["mamba_unowned"] = out["mamba_total"] - sum(out[t] for t in mt)
    return out


def _fmt(d: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in d.items())


def req_holdings(req) -> dict:
    """One request's pool holdings, read only."""
    def g(name):
        v = getattr(req, name, None)
        if v is None:
            return None
        try:
            return int(v)
        except Exception:  # noqa: BLE001 -- a tensor index
            try:
                return int(v.item())
            except Exception:  # noqa: BLE001
                return str(v)[:24]

    pi = getattr(req, "prefix_indices", None)
    return {
        "rid": str(getattr(req, "rid", "?"))[:24],
        "req_pool_idx": g("req_pool_idx"),
        "mamba_pool_idx": g("mamba_pool_idx"),
        "prefix": None if pi is None else _int(lambda: len(pi)),
        "committed": g("kv_committed_len"),
        "allocated": g("kv_allocated_len"),
        "fill": _int(lambda: len(req.fill_ids)) if getattr(req, "fill_ids", None) is not None else None,
    }


def park_snapshot(sched, reqs, *, phase: str, epoch: int) -> Optional[dict]:
    """Measurement 1: one line per request + one pool line, at ``phase``
    (``before-retract`` / ``after-retract``). Returns the ledger read (for a
    delta), or None when off."""
    if not enabled():
        return None
    try:
        rank = getattr(getattr(sched, "tp_group", None), "rank", "?")
        for req in reqs or ():
            logger.info("%s PARK-REQ epoch=%s phase=%s tp=%s %s", TAG, epoch, phase, rank,
                        _fmt(req_holdings(req)))
        led = ledger(sched)
        logger.info("%s PARK-POOL epoch=%s phase=%s tp=%s n=%d %s", TAG, epoch, phase, rank,
                    len(list(reqs or ())), _fmt(led))
        return led
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s PARK snapshot unreadable phase=%s: %r", TAG, phase, exc)
        return None


def ledger_line(sched, *, epoch: int, before: Optional[dict] = None) -> Optional[str]:
    """Measurement 2: the idle equation right after the park, as a LINE --
    ``balanced`` or ``DEFICIT``/``SURPLUS`` with the unowned count and the
    delta against ``before``. Never raises."""
    if not enabled():
        return None
    try:
        rank = getattr(getattr(sched, "tp_group", None), "rank", "?")
        led = ledger(sched)
        fu, mu = led.get("full_unowned"), led.get("mamba_unowned")

        def verdict(u):
            if u is None:
                return "unreadable"
            return "balanced" if u == 0 else ("DEFICIT" if u > 0 else "SURPLUS")

        delta = ""
        if before:
            keys = [k for k in led if isinstance(led.get(k), int) and isinstance(before.get(k), int)]
            delta = " delta[" + " ".join(f"{k}={led[k] - before[k]:+d}" for k in keys
                                          if led[k] != before[k]) + "]"
        line = (f"{TAG} PARK-LEDGER epoch={epoch} tp={rank} full={verdict(fu)}({fu}) "
                f"mamba={verdict(mu)}({mu}) {_fmt(led)}{delta} (read-only; the idle check "
                f"raises on a non-zero unowned term, this line only names it)")
        (logger.warning if (fu or mu) else logger.info)("%s", line)
        return line
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s PARK-LEDGER unreadable: %r", TAG, exc)
        return None


def withheld_ids(allocator) -> set:
    """Every id a KvRowCap on ``allocator`` currently withholds (the caps the
    allocator carries as attributes, e.g. ``_pdflip_kv_stage_cap``)."""
    from flliper.srt.managers.kv_backing_relief import KvRowCap

    out: set = set()
    for v in list(vars(allocator).values()):
        if isinstance(v, KvRowCap):
            held = getattr(v, "_withheld", None)
            if held is not None:
                out |= set(int(x) for x in held.tolist())
    return out


def unwithheld_leak_ids(allocator, tree_cache) -> Optional[str]:
    """Measurement 3: the leaked full ids NOT explained by a KvRowCap --
    ``range(1, size+1)`` minus both free lists, the tree's values and the
    withheld ids. None when off."""
    if not enabled():
        return None
    try:
        free = set(allocator.free_pages.tolist()) | set(allocator.release_pages.tolist())
        cached = set(tree_cache.all_values_flatten().tolist())
        leaked = set(range(1, int(allocator.size) + 1)) - free - cached
        held = withheld_ids(allocator)
        rest = leaked - held
        return (f"{TAG} LEAK-IDS leaked={len(leaked)} withheld_ids={len(held)} "
                f"withheld_published={getattr(allocator, 'residency_withheld_slots', None)} "
                f"withheld_not_leaked={len(held - leaked)} unexplained={len(rest)} "
                f"ids={ranges(rest)}")
    except Exception as exc:  # noqa: BLE001
        return f"{TAG} LEAK-IDS unreadable: {exc!r}"
