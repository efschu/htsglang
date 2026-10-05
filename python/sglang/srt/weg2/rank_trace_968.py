"""#968 RANK-TRACE (desk 1951): pure instrumentation for the
PREFIX MATERIALISATION SHORTFALL death class (STOP W17 Weg2GroupDead, rank
divergence PP0 resident=22528 / PP1 0; y9d4 09:34Z, y9d4c 10:32Z, hb boot
04:33:13Z 05.10.).

WHY. The root of #968 is not proven: PP0 and a follower reach DIFFERENT
prefix decisions for the same rid, and the existing lines (#988 LOADBACK,
#1040 STATE-ALIGN, #1400 TOLD-ANCHOR-HOLD TAKE, PF TOLD-FALLBACK) are sampled,
cut to 8 characters, carry different fields on each site and do not name the
rank that decided. This module gives every decision site ONE line shape, with
the full rid, so that after the next death ``grep '#968-RT' P.log`` lines of
one rid can be laid side by side per rank and the first differing field is
the divergence.

LINE SHAPE (one line, ``key=value``, always these keys in this order)::

    #968-RT site=<site> pp=<rank> rid=<full rid> told=<n|-> matched_prefix_len=<n|->
    resident_rows=<n|-> src=<source|-> decision=<verdict|-> [extra k=v ...]

* ``site``   which decision point wrote the line (see SITES).
* ``pp``     the writer's pipeline rank (``scheduler.ps.pp_rank`` when a
             scheduler is at hand, else the pp group's ``rank_in_group``,
             else ``?``).
* ``told``   the group-uniform told of the rid as this rank knows it.
* ``matched_prefix_len``  what this rank's match / read / decision yielded.
* ``resident_rows``  what this rank holds (device prefix + host hit).
* ``src``    where the number comes from (``l3_index``, ``radix``,
             ``store_read``, ``host``, ``pp0_verdict`` ...).
* ``decision`` the verdict taken at the site.

WHAT IT NEVER DOES (the divergence must not be sharpened by the instrument):

* No behaviour change: every call is a read of fields, a ``logger.info`` and
  a swallowed exception. Nothing is returned to the caller's logic.
* No collective, no ``torch.distributed`` call, no tensor sync, no GPU call:
  the rank label is read from the scheduler's own ``ps`` or from the pp group
  object's attribute, never by communication.
* No wall-clock and no rank-dependent decision: nothing here decides
  anything, there is no sampling (``n <= 32 or n % 64``): every call logs, a
  sampled instrument would hide exactly the rank that diverged. The sites sit
  on admission / verdict / take events, never in the decode round path, and
  never in a poll loop.
* OFF by default. Armed only with ``SGLANG_WEG2_968_RANK_TRACE=1`` AND in
  group P of the dual layout (``SGLANG_WEG2_DUAL_LAYOUT=1`` and
  ``SGLANG_WEG2_GROUP=P``, the gate every dual fix lives behind). Off costs
  two ``os.environ.get`` calls per site event.

SITES
  told_verdict      PP0's told verdict (TOLD-ACKED / TOLD-FALLBACK), weg2_told_fallback.pp0_note_verdict
  told_release      a follower absorbs the fallback and drops its read, weg2_told_fallback.follower_release
  told_admission    the admission comparison own-prefix vs told, weg2_store_told.admission
  state_align       the rank's own match decision (host hit, anchor depth, key match depth ->
                    extent), pp_admission_congruence.state_aligned_load_back_len
  loadback          the applied load-back (#988), schedule_policy._note_988_loadback
  anchor_hold_take  an anchor given up (TOLD-ANCHOR-HOLD TAKE, with host/device flags of its
                    recurrent state), unified_radix_cache._weg2_told_note_take
  prefix_exec       the follower's execution of PP0's decision, entry / load-back / outcome,
                    pp_admission_congruence.execute_scheduled_prefix

HOW TO READ AFTER A DEATH: see deskq/done/1951-968-rank-trace.md.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

ENV = "SGLANG_WEG2_968_RANK_TRACE"
MARKER = "#968-RT"

SITES = (
    "told_verdict",
    "told_release",
    "told_admission",
    "state_align",
    "loadback",
    "anchor_hold_take",
    "prefix_exec",
)


def _dual_p() -> bool:
    return (
        (os.environ.get("SGLANG_WEG2_DUAL_LAYOUT", "") or "").strip() == "1"
        and (os.environ.get("SGLANG_WEG2_GROUP", "") or "").strip().upper() == "P"
    )


def armed() -> bool:
    """Switch on AND dual group P. Read on every call (two env lookups): the
    sites are admission/verdict events, not round steps."""
    if (os.environ.get(ENV, "") or "").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    return _dual_p()


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    return str(v)


def pp_label(scheduler: Any = None) -> str:
    """The writer's pp rank, WITHOUT communication. Never raises."""
    try:
        ps = getattr(scheduler, "ps", None)
        r = getattr(ps, "pp_rank", None)
        if r is not None:
            return str(int(r))
    except Exception:  # noqa: BLE001 - an instrument
        pass
    try:
        from sglang.srt.distributed.parallel_state import get_pp_group

        return str(int(get_pp_group().rank_in_group))
    except Exception:  # noqa: BLE001 - group not initialised (unit doubles)
        return "?"


#: bound of the de-duplication table of :func:`once` (FIFO eviction).
ONCE_MAX = 8192
_once_seen: dict = {}


def once(*key: Any) -> bool:
    """True the first time ``key`` is seen in this process (bounded FIFO),
    False when off. For sites a request passes more
    than once with the SAME facts (a re-matched queued request): one line per
    distinct fact, never a sample -- a changed fact is a new key and logs."""
    try:
        if not armed():
            return False
        if key in _once_seen:
            return False
        _once_seen[key] = None
        if len(_once_seen) > ONCE_MAX:
            _once_seen.pop(next(iter(_once_seen)))
        return True
    except Exception:  # noqa: BLE001 - an instrument
        return False


def resident_rows_of(req: Any) -> Optional[int]:
    """device prefix + host hit of ``req``: the number a rank 'holds'.
    ``len()`` of the object itself (no truthiness of a tensor). None when the
    request does not carry the fields."""
    try:
        pi = getattr(req, "prefix_indices", None)
        n = 0 if pi is None else int(len(pi))
        return n + int(getattr(req, "host_hit_length", 0) or 0)
    except Exception:  # noqa: BLE001 - an instrument
        return None


def emit(
    site: str,
    rid: Any,
    *,
    told: Any = None,
    matched_prefix_len: Any = None,
    resident_rows: Any = None,
    src: Any = None,
    decision: Any = None,
    scheduler: Any = None,
    **extra: Any,
) -> bool:
    """One ``#968-RT`` line when armed. Returns True when a line was written.
    Never raises, never returns data the caller may act on."""
    try:
        if not armed():
            return False
        tail = "".join(" %s=%s" % (k, _fmt(v)) for k, v in extra.items())
        logger.info(
            "%s site=%s pp=%s rid=%s told=%s matched_prefix_len=%s resident_rows=%s "
            "src=%s decision=%s%s",
            MARKER,
            site,
            pp_label(scheduler),
            _fmt(rid),
            _fmt(told),
            _fmt(matched_prefix_len),
            _fmt(resident_rows),
            _fmt(src),
            _fmt(decision),
            tail,
        )
        return True
    except Exception:  # noqa: BLE001 - an instrument never breaks the station
        return False


def node_depth_rel(best: Any, stop: Any) -> Optional[int]:
    """Tokens from ``stop`` (req.last_node) down to ``best``, None when
    ``stop`` is not an ancestor of ``best``. Read-only tree walk."""
    try:
        node, tot = best, 0
        while node is not None and node is not stop:
            key = getattr(node, "key", None)
            tot += 0 if key is None else len(key)
            node = getattr(node, "parent", None)
        return tot if node is stop else None
    except Exception:  # noqa: BLE001 - an instrument
        return None


def mamba_flags(node: Any) -> str:
    """``host=<0|1>,device=<0|1>`` of the node's recurrent state, ``na`` when
    the node has none / cannot be read. Read-only."""
    try:
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            ComponentType,
        )

        comp = node.component_data[ComponentType.MAMBA]
        if comp is None:
            return "none"
        return "host=%d,device=%d" % (
            int(getattr(comp, "host_value", None) is not None),
            int(getattr(comp, "value", None) is not None),
        )
    except Exception:  # noqa: BLE001 - an instrument
        return "na"
