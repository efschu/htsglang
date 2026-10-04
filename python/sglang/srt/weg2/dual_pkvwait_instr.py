"""#1500i PKVWAIT-INSTR: log-only census for "why does a full-arena D hand no VRAM to a waiting P"
(boot y9d4d, desk analysis deskq/done/1390-p-kv-wait-stall-y9d4d.md, Fix 1).

WHAT THE y9d4d LOGS COULD NOT SAY (1390 B1/B2):
  * ``Q-1190 DUAL D-ARENA-YIELD ... candidates=0`` -- the spill counted the host-only leaves it MAY
    take, never the ones it refused and why;
  * ``D-KV CACHE-YIELD evicted=N`` is ``evictable_size()`` BEFORE the evict; whether the evict freed
    anything was invisible (hypothesis: un-backed write_back leaves stay on the card because
    ``unbacked_drop_allowed`` is False on a TP group);
  * ``SHRINK-BLOCKED live_floor=N`` is the HIGHEST live row id, not a token count -- who sits on that
    row was invisible.

THIS MODULE changes no behaviour. Every function here only READS local counters (no collective: a rank
that did not call it cannot make another rank wait), and emits one line per marker with the fixed prefix
``#1500i PKVWAIT-INSTR`` and ``key=value`` fields. Gate: the dual layout (either group,
``dual_arena_spill.armed_any``) AND ``SGLANG_WEG2_DUAL_PKVWAIT_INSTR`` (default on). Outside the dual
layout nothing is evaluated: ``begin`` returns None before it touches a tree. Rate limit: at most one
line per ``MIN_GAP_S`` per marker per process (= per rank); the calls skipped in between are counted
and printed as ``suppressed=n`` on the next line.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

MARK = "#1500i PKVWAIT-INSTR"
ENV_NAME = "SGLANG_WEG2_DUAL_PKVWAIT_INSTR"
#: at most one line per this many seconds per marker (per rank)
MIN_GAP_S = 5.0
#: bounds of the (rare, rate-limited) scans of the tree / the requests for the owner of the top row
OWNER_SCAN_CAP = 2048
LEAF_SCAN_CAP = 4096

_S: Dict[str, Dict[str, Any]] = {"last": {}, "suppressed": {}, "emitted": {}}


def _reset_for_tests() -> None:
    for d in _S.values():
        d.clear()


def switch_on(env=None) -> bool:
    """The switch alone (default on)."""
    if env is None:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_DUAL_PKVWAIT_INSTR.get())
    return str(env.get(ENV_NAME, "1")).strip().lower() not in ("0", "false", "no", "n", "off")


def enabled(env=None) -> bool:
    """The dual layout (either group) AND the switch. Cheap: env reads only."""
    try:
        from sglang.srt.weg2 import dual_arena_spill as _das

        if not _das.armed_any(env):
            return False
        return switch_on(env)
    except Exception:  # noqa: BLE001 -- an instrument never breaks the caller
        return False


def begin(marker: str, now: Optional[float] = None, env=None) -> Optional[int]:
    """None = do nothing (off the gate / switch off / rate-limited -- the skip is counted). Otherwise the
    number of calls suppressed since the last line of this marker; the caller then computes its census
    and calls ``emit``. The slot is taken here, so a census that raises costs its marker 5 s."""
    if not enabled(env):
        return None
    t = time.monotonic() if now is None else float(now)
    last = _S["last"].get(marker)
    if last is not None and t - last < MIN_GAP_S:
        _S["suppressed"][marker] = _S["suppressed"].get(marker, 0) + 1
        return None
    _S["last"][marker] = t
    return _S["suppressed"].pop(marker, 0)


def count(census: Dict[str, int], reason: str, pages: int = 0) -> None:
    """``census[n.<reason>]`` += 1, ``census[pg.<reason>]`` += pages."""
    census["n." + reason] = census.get("n." + reason, 0) + 1
    census["pg." + reason] = census.get("pg." + reason, 0) + int(pages)


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        return "%.3f" % v
    s = str(v)
    return s.replace(" ", "_") if s else "-"


def format_line(marker: str, fields: Iterable, suppressed: int = 0) -> str:
    """``#1500i PKVWAIT-INSTR marker=<m> k=v ... suppressed=<n>`` (fields: pairs, kept in order; zero
    ``n.*`` / ``pg.*`` census entries are the caller's to omit)."""
    parts = ["marker=" + marker]
    for k, v in fields:
        parts.append("%s=%s" % (k, _fmt_value(v)))
    parts.append("suppressed=%d" % int(suppressed))
    return "%s %s" % (MARK, " ".join(parts))


def emit(marker: str, fields: Iterable, suppressed: int = 0) -> str:
    line = format_line(marker, fields, suppressed)
    logger.info(line)
    _S["emitted"][marker] = _S["emitted"].get(marker, 0) + 1
    return line


def census_fields(census: Dict[str, int]):
    """The census entries in a stable order: n.<r> then pg.<r>, reasons by descending node count."""
    reasons = sorted({k[2:] for k in census if k.startswith("n.")}, key=lambda r: (-census["n." + r], r))
    for r in reasons:
        yield "n." + r, census["n." + r]
        yield "pg." + r, census.get("pg." + r, 0)


# ---------------------------------------------------------------- D tree census (cache yield)
def _lock_of(node: Any) -> int:
    try:
        return max(int(getattr(cd, "lock_ref", 0) or 0) for cd in node.component_data)
    except Exception:  # noqa: BLE001
        return 0


def _dev_tokens(node: Any) -> int:
    try:
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

        v = node.component_data[BASE_COMPONENT_TYPE].value
        return int(v.numel()) if v is not None else 0
    except Exception:  # noqa: BLE001
        return 0


def device_leaf_census(tree: Any, cap: int = LEAF_SCAN_CAP) -> Dict[str, int]:
    """The device leaves a cache yield may evict, by what keeps them on the card: un-backed (write_back
    leaf without a host copy: eviction needs ``write_backup`` first), backed (evictable at once),
    locked (a running seat). Read from ``evictable_device_leaves``; bounded by ``cap``."""
    out = {"dev_leaves": 0, "dev_unbacked": 0, "dev_unbacked_tok": 0, "dev_backed": 0, "dev_backed_tok": 0,
           "dev_locked": 0, "dev_scan_cut": 0}
    leaves = getattr(tree, "evictable_device_leaves", None)
    if leaves is None:
        return out
    for i, n in enumerate(leaves):
        if i >= cap:
            out["dev_scan_cut"] = 1
            break
        out["dev_leaves"] += 1
        tok = _dev_tokens(n)
        if _lock_of(n) > 0:
            out["dev_locked"] += 1
        if getattr(n, "backuped", False):
            out["dev_backed"] += 1
            out["dev_backed_tok"] += tok
        else:
            out["dev_unbacked"] += 1
            out["dev_unbacked_tok"] += tok
    return out


def tree_sizes(tree: Any) -> Dict[str, int]:
    """evictable / protected token counts (each -1 when the tree has no such counter)."""
    out = {}
    for key, name in (("evictable", "evictable_size"), ("protected", "protected_size")):
        try:
            out[key] = int(getattr(tree, name)() or 0)
        except Exception:  # noqa: BLE001
            out[key] = -1
    return out


# ---------------------------------------------------------------- owner of the topmost live row
def _describe_node(node: Any) -> Dict[str, Any]:
    ch = getattr(node, "children", None)
    return {
        "node_id": getattr(node, "id", "?"),
        "lock": _lock_of(node),
        "backuped": int(bool(getattr(node, "backuped", False))),
        "children": len(ch) if ch is not None else -1,
        "tok": _dev_tokens(node),
    }


def _holds_rows(tensor: Any, lo: int, hi: int) -> bool:
    """Does the integer tensor name a row in [lo, hi)? (one device sync; callers are rate-limited)"""
    if tensor is None or not getattr(tensor, "numel", None) or not tensor.numel():
        return False
    return bool(((tensor >= lo) & (tensor < hi)).any().item())


def top_row_owner(sched: Any, top_row: int, page: int = 1, cap: int = OWNER_SCAN_CAP) -> Dict[str, Any]:
    """Who sits on D's topmost live row ``[top_row, top_row + page)`` (this rank's own view): a request of
    the running batch / the chunked one / the waiting queue / the parked list (``owner=req``), else a
    device-resident tree node (``owner=tree`` with its lock / backed / children state), else
    ``none_found`` (a row held by neither: hand-off, W50 hold, the allocator's own bookkeeping). Reads
    only; a failure is reported as ``owner=err:<ExceptionName>``."""
    lo, hi = int(top_row), int(top_row) + max(1, int(page))
    out: Dict[str, Any] = {"owner": "none_found", "top_row": lo}
    try:
        r2t_pool = getattr(sched, "req_to_token_pool", None)
        r2t = getattr(r2t_pool, "req_to_token", None)
        if r2t is not None:
            groups = (
                ("running", list(getattr(getattr(sched, "running_batch", None), "reqs", None) or ())),
                ("chunked", [getattr(sched, "chunked_req", None)]),
                ("waiting", list(getattr(sched, "waiting_queue", None) or ())),
                ("parked", list(getattr(sched, "weg2_d_parked", None) or ())),
            )
            seen = 0
            for state, reqs in groups:
                for req in reqs:
                    idx = getattr(req, "req_pool_idx", None) if req is not None else None
                    if idx is None:
                        continue
                    seen += 1
                    n = len(getattr(req, "origin_input_ids", None) or ()) + len(getattr(req, "output_ids", None) or ())
                    if n > 0 and _holds_rows(r2t[int(idx), :n], lo, hi):
                        out.update(owner="req", req_state=state, rid=getattr(req, "rid", "?"), req_tokens=n)
                        return out
            out["reqs_scanned"] = seen
        tree = getattr(sched, "tree_cache", None)
        collect = getattr(tree, "_collect_all_nodes", None)
        if callable(collect):
            from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

            scanned = 0
            for n in collect():
                if n is getattr(tree, "root_node", None) or getattr(n, "evicted", True):
                    continue                      # no device value: cannot own a device row
                if scanned >= cap:
                    out["scan_cut"] = 1
                    break
                scanned += 1
                v = n.component_data[BASE_COMPONENT_TYPE].value
                if _holds_rows(v, lo, hi):
                    out.update(owner="tree", **{"top_" + k: v2 for k, v2 in _describe_node(n).items()})
                    out["nodes_scanned"] = scanned
                    return out
            out["nodes_scanned"] = scanned
    except Exception as exc:  # noqa: BLE001 -- an instrument never breaks the tick
        out["owner"] = "err:" + type(exc).__name__
    return out
