# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#631: PP0 geometry authority over a /dev/shm bulletin (single node only).

WHY THIS EXISTS (DESIGN_631_PP0_GEOMETRY_AUTHORITY.md, boot-proven chain).
Six boots of 2026-08-31 died at the #631 width guard because two PP ranks
BUILT different batches for the same slot: the batch geometry consumed
rank-local, wall-time-dependent facts (a storage prefetch that had completed
on one rank and not yet on another -- boot 1065sets: PP0 prefix 32768 /
extend 902 vs PP1 prefix 33148 / extend 522, delta 380 = exactly the
prefetch gain). The discriminator boot (disc631) removed the async feed and
survived 285 readmit laps where the death boots died within ~20.

Every cheaper closure was checked and leaks (see the design doc): the
prefetch termination sync covers only the TP group; a blocking collective at
the consult sits behind capacity gates that themselves diverge (the
2026-08-17 deadlock family); the admission row on the proxy frame arrives
one pass after the receiver planned (#1059 SITE 4), so it can never cover
the intake pass -- and all six deaths WERE intake passes.

THE MECHANISM. PP ranks on this rig are processes on ONE host (the barlink
build windows already coordinate through /dev/shm). PP0 -- the one deciding
place, per the standing PP0-authoritative order -- publishes the geometry of
every batch it plans, per (epoch, slot), latest-wins, via atomic rename.
A downstream rank reads the bulletin BEFORE its own plan of that slot
(pipeline order runs PP0 first; a short bounded poll covers the boundary)
and enforces, at the one site that writes match geometry onto a request
(schedule_batch.init_next_round_input, the old #1059 SITE 5 position):

  * device prefix above PP0's told  -> truncate down (#930-paired helper;
    always materializable, surplus stays in the tree),
  * async host adoption (host_hit_length, the #988 raise input) capped to
    max(0, told - device_prefix); with NO bulletin in hand the cap is 0 --
    a rank may never raise its geometry on a fact PP0 has not published.

NEVER BLOCKING, NEVER A COLLECTIVE, NEVER A REFUSAL LOOP: a missing or
stale bulletin degrades to "no async adoption this pass" (bounded loss of
at most the prefetch gain, the class the Kein-Doppel-Prefill law allows),
and every degradation is counted with its denominator. Divergence that
still occurs (the under-coverage direction: PP0 raised, this rank's own
prefetch lagging) stays DETECTED by the #631 guard -- this module narrows
construction, it does not smooth detection (RAENGE-NIE-UNEINS).

Single-node is a hard precondition (server_args.nnodes == 1); on multi-node
forms the module disables itself and says so once.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_DIR = "/dev/shm/sglang-pp-admission"

#: Cross-boot staleness fence: a bulletin older than this process's import
#: is a PREVIOUS boot's file on the same port (epoch 0 == epoch 0 at fill
#: time would otherwise read as fresh). Both PP processes import minutes
#: before serving publishes, so 60 s of slack cannot reject a live file.
_BOOT_FENCE_TS = time.time() - 60.0
#: How long a downstream rank will wait for the slot's bulletin before
#: degrading to cap=0. Pipeline order makes the file present in steady
#: state; this only covers the fill/boundary laps.
_POLL_S = 0.05
_POLL_STEP_S = 0.005

#: Tree-cache attribute the apply site reads. Planted only on downstream
#: ranks, only for the duration of one plan call -- PP0 and non-PP forms
#: never see it and the apply site no-ops on absence.
GEOMETRY_ATTR = "_pp_bulletin_geometry"

#: Cumulative counters, printed on every emitted line (#1047: an acceptance
#: metric may not be sampled without its totals).
_STATS = {
    "published": 0,
    "publish_failed": 0,
    "loaded": 0,
    "load_absent": 0,
    "load_stale_epoch": 0,
    "capped_device": 0,
    "capped_host": 0,
    "no_bulletin_host_zeroed": 0,
    "under_coverage": 0,
}


def _slot_path(port: int, slot: int) -> str:
    return f"{_DIR}/p{int(port)}_s{int(slot)}.json"


def _epoch_of(scheduler) -> int:
    try:
        from sglang.srt.managers.scheduler_pp_mixin import pp_flip_epoch_of

        return int(pp_flip_epoch_of(scheduler))
    except Exception:  # noqa: BLE001 - epoch source missing means epoch 0 form
        return 0


def bulletin_enabled(scheduler) -> bool:
    if os.environ.get("SGLANG_PP0_BULLETIN", "1") in ("0", "false", "False"):
        return False
    try:
        if int(getattr(scheduler.server_args, "nnodes", 1)) != 1:
            if not getattr(scheduler, "_pp_bulletin_multinode_warned", False):
                scheduler._pp_bulletin_multinode_warned = True
                logger.warning(
                    "#631 BULLETIN DISABLED: nnodes != 1. The /dev/shm channel "
                    "is single-node by construction; multi-node PP keeps the "
                    "pre-bulletin behaviour (rank-local geometry, guarded by "
                    "the #631 width check)."
                )
            return False
    except Exception:  # noqa: BLE001
        return False
    return True


def publish(scheduler, mb_id: int, batch) -> None:
    """PP0: publish the geometry of the batch just planned for this slot.

    Called right after ``self.mbs[mb_id] = plan.batch_to_run`` -- the
    earliest point at which the decision exists, and strictly before any
    downstream rank can plan the same slot (it is still waiting on this
    pass's frame). An EMPTY plan publishes an empty entry list on purpose:
    "PP0 planned nothing" is a fact, and it keeps the file's epoch fresh.
    """
    if not bulletin_enabled(scheduler):
        return
    entries = {}
    try:
        reqs = getattr(batch, "reqs", None) or ()
        fm = getattr(batch, "forward_mode", None)
        is_extend = bool(getattr(fm, "is_extend", lambda: False)()) if fm else False
        if is_extend:
            for req in reqs:
                rid = getattr(req, "rid", None)
                if rid is None:
                    continue
                pi = getattr(req, "prefix_indices", None)
                entries[str(rid)] = 0 if pi is None else int(len(pi))
        payload = {
            "epoch": _epoch_of(scheduler),
            "slot": int(mb_id),
            "seq": int(getattr(scheduler, "_pp_bulletin_seq", 0)) + 1,
            "ts": time.time(),
            "prefix": entries,
        }
        scheduler._pp_bulletin_seq = payload["seq"]
        os.makedirs(_DIR, exist_ok=True)
        path = _slot_path(scheduler.server_args.port, mb_id)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
        _STATS["published"] += 1
    except Exception:  # noqa: BLE001 - a publish failure must never kill PP0
        _STATS["publish_failed"] += 1
        n = _STATS["publish_failed"]
        if n <= 5 or n % 256 == 0:
            logger.warning(
                "#631 BULLETIN PUBLISH FAILED (n=%d, published=%d). Downstream "
                "degrades to cap=0 for this slot (no async adoption), which is "
                "bounded loss, not divergence.",
                n,
                _STATS["published"],
                exc_info=True,
            )


def load_for_plan(scheduler, mb_id: int) -> None:
    """Downstream: read this slot's bulletin and plant it on the tree cache.

    Bounded poll (<= _POLL_S) ONLY while the file is absent or carries a
    stale epoch; pipeline order makes both rare outside fill/cutover laps.
    On timeout the map is planted as None -- the apply site then caps async
    adoption at 0 (counted), never at "whatever this rank happens to see".
    """
    tree_cache = getattr(scheduler, "tree_cache", None)
    if tree_cache is None:
        return
    if not bulletin_enabled(scheduler):
        setattr(tree_cache, GEOMETRY_ATTR, None)
        return
    epoch = _epoch_of(scheduler)
    path = _slot_path(scheduler.server_args.port, mb_id)
    deadline = time.monotonic() + _POLL_S
    payload = None
    stale_seen = False
    while True:
        try:
            with open(path) as f:
                candidate = json.load(f)
            if (
                int(candidate.get("epoch", -1)) == epoch
                and float(candidate.get("ts", 0)) >= _BOOT_FENCE_TS
            ):
                payload = candidate
                break
            stale_seen = True
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001 - a torn read retries within budget
            pass
        if time.monotonic() >= deadline:
            break
        time.sleep(_POLL_STEP_S)
    if payload is None:
        if stale_seen:
            _STATS["load_stale_epoch"] += 1
        else:
            _STATS["load_absent"] += 1
        n = _STATS["load_stale_epoch"] + _STATS["load_absent"]
        if n <= 5 or n % 256 == 0:
            logger.info(
                "#631 BULLETIN ABSENT for slot %d (stale_epoch=%d absent=%d "
                "loaded=%d): async host adoption capped at 0 this pass -- "
                "bounded recovery loss, never a rank-local raise.",
                mb_id,
                _STATS["load_stale_epoch"],
                _STATS["load_absent"],
                _STATS["loaded"],
            )
        setattr(tree_cache, GEOMETRY_ATTR, {})
        return
    _STATS["loaded"] += 1
    setattr(tree_cache, GEOMETRY_ATTR, dict(payload.get("prefix") or {}))


def clear_after_plan(scheduler) -> None:
    """Remove the planted map so no other path can read a stale bulletin."""
    tree_cache = getattr(scheduler, "tree_cache", None)
    if tree_cache is not None and hasattr(tree_cache, GEOMETRY_ATTR):
        setattr(tree_cache, GEOMETRY_ATTR, None)


def cap_req_geometry(req, tree_cache) -> None:
    """Enforce PP0's published ceiling on one request, at the match writer.

    Runs at the old #1059 SITE 5 position in ``init_next_round_input``:
    after the match wrote ``prefix_indices``/``host_hit_length`` and after
    ``cache_protected_len`` was set (so ``truncate_prefix_to``'s #930
    pairing operates on fresh state), and strictly before ``add_one_req``
    derives ``extend_range`` from the current prefix -- the same ordering
    contract the #791 truncate site documents.

    The map is ONLY planted on downstream PP ranks during their own plan
    call; everywhere else this is a getattr miss and a return.
    """
    geometry = getattr(tree_cache, GEOMETRY_ATTR, None) if tree_cache else None
    if geometry is None:
        return  # PP0, non-PP, bulletin disabled, or outside a plan call.
    rid = str(getattr(req, "rid", ""))
    told = geometry.get(rid) if geometry else None
    local = 0 if req.prefix_indices is None else len(req.prefix_indices)
    if told is None:
        # No published fact for this rid (or no bulletin at all): a rank may
        # not raise its geometry on an async fact PP0 has not published.
        hh = int(getattr(req, "host_hit_length", 0) or 0)
        if hh > 0:
            req.host_hit_length = 0
            _STATS["no_bulletin_host_zeroed"] += 1
            n = _STATS["no_bulletin_host_zeroed"]
            if n <= 5 or n % 256 == 0:
                logger.info(
                    "#631 BULLETIN CAP rid=%s: host adoption %d -> 0, no "
                    "published fact (n=%d, capped_host=%d, under=%d). Loss is "
                    "bounded by the prefetch gain; the content stays in the "
                    "host tier for a later, published lap.",
                    rid[:8],
                    hh,
                    n,
                    _STATS["capped_host"],
                    _STATS["under_coverage"],
                )
        return
    told = int(told)
    if local > told:
        # Device prefix above PP0's ceiling (eviction skew, or an earlier
        # uncapped raise): truncate down. Always materializable; the #930
        # pairing keeps cache_protected_len consistent, and add_one_req
        # re-derives extend_range from the truncated prefix afterwards.
        req.truncate_prefix_to(told)
        local = told
        _STATS["capped_device"] += 1
        n = _STATS["capped_device"]
        if n <= 20 or n % 256 == 0:
            logger.warning(
                "#631 BULLETIN CAP rid=%s: device prefix truncated to told=%d "
                "(n=%d, capped_host=%d, under=%d). This is the divergence "
                "direction that killed boots 26-35; the truncate makes this "
                "rank run PP0's geometry instead of its own.",
                rid[:8],
                told,
                n,
                _STATS["capped_host"],
                _STATS["under_coverage"],
            )
    allowed_host = max(0, told - local)
    hh = int(getattr(req, "host_hit_length", 0) or 0)
    if hh > allowed_host:
        req.host_hit_length = allowed_host
        _STATS["capped_host"] += 1
    elif local + hh < told:
        # Under-coverage: PP0 published more than this rank can currently
        # materialize. Counted loudly; the #631 width guard remains the
        # detector if the pass diverges (construction narrowed, detection
        # untouched).
        _STATS["under_coverage"] += 1
        n = _STATS["under_coverage"]
        if n <= 20 or n % 64 == 0:
            logger.warning(
                "#631 BULLETIN UNDER-COVERAGE rid=%s: told=%d, local=%d, "
                "host_hit=%d (n=%d). PP0 raised on a fact this rank has not "
                "materialized yet; if this pass runs divergent the #631 guard "
                "will name it. A rising counter here orders the next slice "
                "(PP0 defers its raise to the group floor).",
                rid[:8],
                told,
                local,
                hh,
                n,
            )


def stats() -> Dict[str, int]:
    return dict(_STATS)
