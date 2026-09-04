"""#703: push warm prefixes to the canonical store BEFORE a phase flip.

THE GAP THIS CLOSES, traced rather than assumed.

Nothing in the flip path touches HiCache. The cutover
(``phase_flip_runtime.py``) rebuilds topology, groups, loop state and the
worker stack; it never backs up, flushes, detaches or resets the cache. That
is deliberate -- device rows survive the flip, because the live row set (radix
tree values UNION parked requests' rows) is relocated between the two phase
pools BY ROW ID. So the device tier rides through.

The HOST tier does not, and the reason is structural rather than a missing
call. There are two device KV pools: the boot PP stack's and the flip's TP
stack's (``phase_flip_runtime`` builds both, and the TP stack builds with
``pp_size = 1``, so it spans ALL attention layers while a PP stage spans only
its own 7/5/4). ``HiRadixCache`` binds its host pool to whichever device pool
existed at construction -- the boot one -- and the scheduler's allocator is
likewise assigned once and never reassigned. In the phase that did not build
it, therefore, the host tier mirrors a pool that is not the live one.

So a prefix's only way across the flip is the geometry-free STORE (#706): the
disk tier, whose keys carry content alone and whose pages are cut at read time
for whichever geometry asks. Getting a prefix there is what this module does.

WHY A HOOK IS NEEDED AT ALL -- the write-back timing gap:

* ``write_through``            device->host on the FIRST insert (eager);
* ``write_through_selective``  on the second hit;
* ``write_back``               ONLY under eviction pressure.

and host->storage is a separate asynchronous stage: the backup thread picks
entries off ``backup_queue`` and acknowledges on ``ack_backup_queue``. Under
``write_back`` -- and under either write-through policy for anything below
threshold or still in flight -- a warm prefix can sit in device memory,
never staged, indefinitely. A flip does not change that: it neither forces
the staging nor waits for the queue to drain.

There is no existing API to lean on. ``writing_check`` only completes writes
already issued. ``detach_storage_backend`` DROPS pending backups rather than
finishing them. ``flush_cache`` is a wipe, not a write-back, despite the name.
``HiCacheController.reset`` clears every queue outright, so pending backup work
is lost, not drained.

This hook therefore does the one thing none of them does: it stages the tree's
un-backed prefixes to host, lets the storage stage run, and WAITS for the
acknowledgements -- under a deadline, because an unbounded wait at the flip
seam is the #630 wedge shape and this runs with requests parked.

It refuses loudly without the canonical store. Writing pages named by this
phase's geometry, at the seam of a phase change, would buy exactly nothing:
the other phase cannot name them. Better to say so than to spend the IO.

#872, MEASURED 2026-08-26 (R7, ``boot_accept0826r7fix_0826_1817.log``): the
ticket's symptom -- "retention fires at the seam but nothing reaches the
store" -- is REFUTED for the shipped configuration, and the paragraph above is
the reason the wrong conclusion was easy to reach. It describes the
``write_back`` policy; the live boot runs ``hicache_write_policy='write_through'``,
under which the ORDINARY insert path already carries a node all the way to the
store: ``_inc_hit_count`` -> ``write_backup`` -> ack -> ``_finish_write_through_ack``
-> ``write_backup_storage`` (``unified_radix_cache.py:2328``, the sole caller
of that method inside the tree). So by the time the fence runs, most eligible
nodes are ``backuped`` AND already persisted, and the fence's ``already_staged``
skip below is a correct idempotent no-op rather than a lost prefix.

``already_staged`` implies "was offered to the store" because ``backuped`` --
``component_data[FULL].host_value is not None`` -- has only three writers:
``full_component.py:340`` (the BACKUP_HOST commit, i.e. ``write_backup``,
which acks into ``write_backup_storage``), ``full_component.py:100-103`` (a
node SPLIT, which re-slices host indices whose per-page keys are already in
the store at ``page_size == 1``), and ``unified_radix_cache.py:1764`` (the
host-only insert of a prefetch, whose bytes came OUT of the store). None of
the three can produce a host copy the store never saw.

The three measurements that close it, all from the R7 window (18:17-18:28):
24277 canonical pages written under ``/tmp/hicache_783``; all 16 attention
slots non-zero in 400 of 400 sampled pages, so the pages are COMPLETE rather
than PP-stage slices; and 126 of 264 storage prefetches returned
``completed_local=4096``, so the keys written are the keys the read side asks
for. What remains of the fence's own failure surface is the silent zero
counted as ``refused_silently`` below -- 3 node visits of 408 in that same
window (96 fence lines: 210 staged, 195 already staged, 3 unaccounted).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#703 flip-writeback]"

# Poll interval while waiting for storage acknowledgements.
_POLL_S = 0.005
# #1028: hard total = this many times the no-progress bound. See
# `maybe_flip_writeback` for why it is derived rather than separately settable.
_CEILING_MULTIPLE = 12.0


class FlipWritebackRefused(RuntimeError):
    """The writeback cannot do anything useful in this configuration."""


@dataclasses.dataclass(frozen=True)
class FlipWritebackReport:
    """What the hook actually achieved. Counts, never a verdict."""

    eligible: int  # tree nodes carrying a hash, i.e. persistable
    staged: int  # nodes this call staged device -> host
    already_staged: int  # nodes the normal policy had already staged
    acknowledged: int  # storage backups confirmed before the deadline
    outstanding: int  # storage backups still in flight when time ran out
    elapsed_s: float
    deadline_s: float
    # #872: nodes the loop actually HANDED to `write_backup` and that came back
    # falsy. Appended with a default so every existing constructor keeps
    # working; see the counter's own comment at the call site for why a plain
    # `if written:` made this population indistinguishable from "nothing to do".
    refused_silently: int = 0
    # Of those, the share attributable to the #581/#773 mamba write-through pin
    # budget, read from the tree's own monotone counter. -1 when the tree does
    # not carry that counter, which is NOT the same as zero and must not be
    # printed as zero.
    refused_mamba_pin: int = -1
    # #1028: the HARD total bound, distinct from `deadline_s` which is now the
    # NO-PROGRESS bound. Printed beside elapsed so a fence that ran long
    # because acks kept landing is distinguishable from one that hung.
    ceiling_s: float = 0.0
    # #1205: WRITE-direction device-tier operations the FLIP SEAM refused
    # during this fence run, as a delta over
    # `hicache_phase_guard.seam_refusals("write")`. -1 when the guard carries
    # no counter, which is NOT the same fact as zero and is not printed as one.
    #
    # IT IS NOT A SUBSET OF `refused_silently` AND MUST NOT BE READ AS ONE.
    # The counter is a process-wide module global covering every "write"
    # caller of `device_tier_disarmed` -- the per-node enqueue at
    # `cache_controller.py:1596`, `consume_gate` (`:435`), which refuses a
    # whole queue and counts ONE, and `hybrid_cache_controller.py:486` -- and
    # those run on the controller's own threads alongside this fence. So it can
    # exceed `refused_silently`, and it can be non-zero while `refused_silently`
    # is 0. It answers "did the seam refuse device-tier writes while this fence
    # ran", which is the seam-arm-ordering question; it does not attribute
    # individual nodes.
    seam_refused: int = -1

    @property
    def complete(self) -> bool:
        return self.outstanding == 0

    @property
    def persisted_nothing(self) -> bool:
        """#783: nothing of this fence is retrievable after the tree is dropped.

        NOT the same question as `complete`, which is `outstanding == 0` and is
        therefore TRUE for a fence that had nothing to do. In W37-G 33 of 39
        cutovers fenced `eligible=0 staged=0 already_staged=0 acked=0
        outstanding=0 elapsed=0.000s` -- complete, cheap, and empty -- and every
        instrument in the tree read them as healthy. `#cached-token: 0` on all
        209 prefill batch lines is the consequence.

        The seam's own guard could not see it either: it warns when
        `_writeback_fence_ms(...) is None`, and an empty fence returns 0.0. A
        cheap fence and a healthy fence are indistinguishable BY COST, because
        the healthy case is also cheap. So this asks about retrievability, which
        is the property the read-through actually depends on, using counts the
        report already carried and nothing read.

        DELIBERATELY NARROW: true only when NEITHER route survives -- no storage
        acknowledgement AND no pre-existing host copy. A fence that acked
        nothing but left `already_staged` nodes behind is NOT claimed here.

        #872 2026-08-26 -- THE REASON GIVEN FOR THAT NARROWING WAS WRONG, while
        the narrowing itself was right. The old wording was "that host copy may
        still serve the read-through", which this module's own opening
        paragraphs refute: the host tier mirrors the pool that built it, so
        after the cutover it is not the live pool and cannot serve anything.
        The correct reason is stronger and is proved in the module docstring:
        under every writer of `backuped`, an `already_staged` node has ALREADY
        been offered to the store, so a fence that finds only such nodes has
        lost nothing and has nothing to do. Widening this to cover it would
        double-report one condition and make a crying-wolf gate out of the
        instrument built to replace one.

        `refused_silently` is deliberately NOT part of this predicate. It names
        a population the fence tried and failed to persist, which is a real
        loss, but a fence can carry a silent refusal and still have persisted
        everything else -- reporting that whole fence as having persisted
        nothing would be the same conflation in the other direction.
        """
        return self.acknowledged == 0 and self.already_staged == 0

    def as_log(self) -> str:
        # `refused_mamba_pin` prints `?` rather than 0 when the tree carries no
        # such counter: an unmeasured share and a measured zero are different
        # facts and a log line that spells them the same way is the #872 probe
        # failure one level up.
        mamba = "?" if self.refused_mamba_pin < 0 else str(self.refused_mamba_pin)
        seam = "?" if self.seam_refused < 0 else str(self.seam_refused)
        return (
            f"eligible={self.eligible} staged={self.staged} "
            f"already_staged={self.already_staged} acked={self.acknowledged} "
            f"outstanding={self.outstanding} "
            f"refused_silently={self.refused_silently} "
            f"refused_mamba_pin={mamba} "
            f"seam_refused={seam} "
            f"elapsed={self.elapsed_s:.3f}s/{self.deadline_s:.3f}s"
            f" ceiling={self.ceiling_s:.3f}s"
        )


def canonical_store_of(tree_cache: Any):
    """The canonical page window a storage backend is running with, or None."""
    controller = getattr(tree_cache, "cache_controller", None)
    backend = getattr(controller, "storage_backend", None)
    return getattr(backend, "canonical_kv_page", None)


def require_canonical_store(tree_cache: Any) -> None:
    """Refuse unless a flip-crossing store is actually configured.

    Three separate ways to have nothing to write to, each named, because each
    has a different fix and a generic "not supported" would hide which one is
    in force.
    """
    if not getattr(tree_cache, "enable_storage", False):
        raise FlipWritebackRefused(
            f"{LOG_PREFIX} no storage tier is attached, so there is nowhere "
            "for a prefix to survive the flip: the host tier mirrors the pool "
            "that built it, which in the other phase is not the live pool. "
            "Attach the file backend (--hicache-storage-backend file)."
        )
    if canonical_store_of(tree_cache) is None:
        raise FlipWritebackRefused(
            f"{LOG_PREFIX} the storage tier is not running the #706 canonical "
            "page format, so its keys carry this phase's geometry "
            "(_{tp_rank}_{tp_size} and _{pp_size}_{pp_rank}). Pages written at "
            "the flip seam would be unreadable in the phase they were written "
            "for. Enable --phase-flip-canonical-kv-page, or leave the "
            "writeback off; spending the IO for keys the other phase cannot "
            "name is the one outcome worth refusing."
        )


def _total_nodes(tree_cache: Any) -> int:
    """EVERY node under the root, hashed or not. #969E discriminator.

    `_hashed_nodes` is the eligibility predicate: a node counts only once it
    carries a `hash_value`. So `eligible=0` has two very different causes and
    the fence cannot tell them apart:

      (a) nothing was INSERTED into the tree at all, or
      (b) it was inserted and is not HASHED YET, because hashing happens on the
          write-through path asynchronously and this fence runs immediately
          after the retraction.

    Rising total with eligible=0 is (b) -- an ordering problem, fixable under
    this fence's existing deadline. Flat total is (a) -- the insert never
    happened. One number decides it; nothing else in the tree reports it.
    """
    root = getattr(tree_cache, "root_node", None)
    if root is None:
        return -1
    n = 0

    def _walk(node) -> None:
        nonlocal n
        for child in list(getattr(node, "children", {}).values()):
            n += 1
            _walk(child)

    try:
        _walk(root)
    except Exception:  # noqa: BLE001 - a probe may never break a fence
        return -1
    return n



#: #1063: THE DECIDER, three states printed side by side rather than one saldo.
#: rid-free, keyed by the store STEM the fence believed it had persisted.
_1063_AT_FENCE: Dict[str, str] = {}
_1063_STATE: Dict[str, int] = {}
_1063_STEM_CAP = 4096


def _1063_bump(key: str, n: int = 1) -> None:
    _1063_STATE[key] = _1063_STATE.get(key, 0) + n


def _1063_backend(tree_cache):
    """The storage backend, or None. Never raises."""
    try:
        return getattr(
            getattr(tree_cache, "cache_controller", None), "storage_backend", None
        )
    except Exception:  # noqa: BLE001
        return None


def _1063_stem_state(backend, stem: str) -> str:
    """What the store holds for ``stem`` RIGHT NOW, as one of three words.

    * ``readable``   -- the final ``.bin`` exists. A reader can serve it.
    * ``assembling`` -- a ``.part706`` and/or its ``.slots706`` marker exist and
      the ``.bin`` does NOT. This is the shape the canonical-page protocol calls
      out itself: *"a writer acting alone leaves nothing readable behind"*
      (`_set_canonical_slice`). A blob whose writer GROUP was torn apart -- which
      is exactly what a cutover does to an in-flight assembly -- stays in this
      state forever and is invisible to every reader.
    * ``absent``     -- neither. Either never written, or reaped/evicted.

    Read-only stat()s, no locks, never raises: an unreadable probe answers
    ``unknown`` rather than inventing one of the three.
    """
    try:
        from sglang.srt.mem_cache.canonical_page_store import (
            marker_path,
            part_path,
        )

        final = backend._sharded_path(stem)
        if os.path.exists(final):
            return "readable"
        try:
            flat = backend._flat_path(stem)
            if os.path.exists(flat):
                return "readable"
        except Exception:  # noqa: BLE001
            pass
        if os.path.exists(part_path(final)) or os.path.exists(marker_path(final)):
            return "assembling"
        return "absent"
    except Exception:  # noqa: BLE001 - a probe may never break the seam
        return "unknown"


def _1063_stems_for_node(backend, node) -> list:
    """The store stems this node's pages would occupy. Never raises."""
    out = []
    try:
        hv = getattr(node, "hash_value", None) or ()
        for h in hv:
            try:
                out.append(backend._get_suffixed_key(str(h)))
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        return []
    return out


def _1063_record_fence(tree_cache, nodes) -> None:
    """Snapshot the store state of every eligible node AT THE FENCE.

    Two points, not one, and that is the whole design: `evicted_since_flip`
    cannot be read from a single observation at re-admission -- "absent now" and
    "absent all along" are the same stat() and have opposite fixes. Recording
    what the fence BELIEVED it had, and re-reading the same stems later, is what
    separates them.
    """
    try:
        backend = _1063_backend(tree_cache)
        if backend is None:
            _1063_bump("fence_no_backend")
            return
        seen = {"readable": 0, "assembling": 0, "absent": 0, "unknown": 0}
        # #1068: HARD TIME BUDGET on a diagnostic that sits on the cutover's
        # no-return path. This scan walks every page stem of every eligible
        # node through 2-4 os.path.exists() against a million-file store.
        # Warm dcache that is ~3 s for 221862 stems (measured 06:47:10,
        # boot_855_1067park); the very NEXT fence hit the re-prefill's fresh
        # stems cold and never returned -- PP0's last line for 25 minutes was
        # the FENCE-NODES header above, no abandon could run (PP0 is the
        # timeout carrier), and the deadman killed the boot at 07:15:37.
        # An instrument may never gate. A scan that cannot finish inside the
        # budget reports itself CAPPED -- the counts become a sample and say
        # so -- instead of holding the seam. The two-point consumer
        # (`_1063_probe_since_fence`) reads only `_1063_AT_FENCE`, which is
        # capped at 4096 stems anyway, so nothing downstream loses coverage
        # it ever had.
        # NOTE: deliberately time.monotonic(), not the fence's `now` -- that
        # is a PARAMETER of the fence function and is not in scope here; a
        # bare now() would NameError into the outer except and silently
        # delete this whole snapshot (the #872 silent-zero shape).
        _scan_deadline = time.monotonic() + 2.0
        _scan_capped = False
        for node in nodes or ():
            for stem in _1063_stems_for_node(backend, node):
                if time.monotonic() > _scan_deadline:
                    _scan_capped = True
                    break
                st = _1063_stem_state(backend, stem)
                seen[st] = seen.get(st, 0) + 1
                if len(_1063_AT_FENCE) < _1063_STEM_CAP:
                    _1063_AT_FENCE[stem] = st
            if _scan_capped:
                break
        _1063_bump("fences")
        if _scan_capped:
            _1063_bump("fence_scan_capped")
        for k, v in seen.items():
            _1063_bump(f"fence_{k}", v)
        logger.warning(
            "#1063 FENCE STORE STATE%s: stems=%d readable=%d assembling=%d "
            "absent=%d unknown=%d (tracked=%d of cap %d). `assembling` is a "
            "blob whose writer group never covered the last byte -- invisible "
            "to every reader, forever. `absent` here is 'not written or already "
            "reaped'; whether it was LOST LATER is only decidable against this "
            "snapshot, which is why it is taken.",
            (
                " (SCAN CAPPED at 2.0s -- #1068: counts are a SAMPLE of the "
                "population, not the population)"
                if _scan_capped
                else ""
            ),
            sum(seen.values()),
            seen.get("readable", 0),
            seen.get("assembling", 0),
            seen.get("absent", 0),
            seen.get("unknown", 0),
            len(_1063_AT_FENCE),
            _1063_STEM_CAP,
        )
    except Exception:  # noqa: BLE001 - a census may never break the seam
        pass


def _1063_probe_since_fence(tree_cache) -> None:
    """Re-read the fence's stems and classify the TRANSITION.

    This is the decider the coordinator asked for, and it prints the three
    states explicitly rather than a saldo:

    * ``present_and_readable``  -- was persisted and still is. If the
      re-admission still misses on these, neither candidate root holds and the
      hunt moves to the lookup.
    * ``complete_marker_absent``-- the blob is stuck mid-assembly (``.part706``
      present, no ``.bin``). The cutover tore a writer group apart.
    * ``evicted_since_flip``    -- it WAS readable at the fence and is gone now.
    """
    try:
        backend = _1063_backend(tree_cache)
        if backend is None or not _1063_AT_FENCE:
            return
        now = {
            "present_and_readable": 0,
            "complete_marker_absent": 0,
            "evicted_since_flip": 0,
            "still_absent": 0,
            "unknown": 0,
        }
        for stem, was in _1063_AT_FENCE.items():
            st = _1063_stem_state(backend, stem)
            if st == "readable":
                now["present_and_readable"] += 1
            elif st == "assembling":
                now["complete_marker_absent"] += 1
            elif st == "absent":
                if was == "readable":
                    now["evicted_since_flip"] += 1
                else:
                    now["still_absent"] += 1
            else:
                now["unknown"] += 1
        logger.warning(
            "#1063 SINCE-FENCE DECIDER (denominator=stems tracked at the last "
            "fence=%d): present_and_readable=%d complete_marker_absent=%d "
            "evicted_since_flip=%d still_absent=%d unknown=%d. "
            "`complete_marker_absent` is the torn-assembly candidate (a cutover "
            "splitting a writer group leaves a permanently unreadable blob); "
            "`evicted_since_flip` is the eviction candidate, and it is only "
            "countable because the fence snapshot exists. If BOTH are ~0 while "
            "the re-admission still misses, both roots are refuted and the "
            "defect is in the LOOKUP, not in the bytes.",
            len(_1063_AT_FENCE),
            now["present_and_readable"],
            now["complete_marker_absent"],
            now["evicted_since_flip"],
            now["still_absent"],
            now["unknown"],
        )
    except Exception:  # noqa: BLE001 - a census may never break admission
        pass


def _hashed_nodes(tree_cache: Any) -> list:
    """Persistable nodes, PARENT FIRST.

    Parent order is not cosmetic: the backup invariant is that backed-up nodes
    form a contiguous prefix from the root, and a child staged before its
    parent is a gap the normal path would have skipped.

    #841: parent order alone was never enough, and the walk below shows why --
    a node WITHOUT a ``hash_value`` is skipped from the list while the walk
    still descends into its children. So an unhashed node's children reach the
    staging loop with their parent absent from it entirely, and the loop's
    ``write_back=True`` also disarms ``write_backup``'s own parent gate. Order
    is preserved here; the law itself is enforced at the call site.
    """
    root = getattr(tree_cache, "root_node", None)
    if root is None:
        return []
    out: list = []

    def _walk(node) -> None:
        for child in list(getattr(node, "children", {}).values()):
            if getattr(child, "hash_value", None):
                out.append(child)
            _walk(child)

    _walk(root)
    return out


def _seam_refusals_or_none() -> Optional[int]:
    """The guard's monotone WRITE-direction seam-refusal total, or None.

    #1205. Same sentinel discipline as ``_mamba_pin_skipped`` directly below:
    an UNMEASURED share must never reach the log line spelled as a zero.

    THE DIRECTION IS THE POPULATION, and asking without one was the #1205
    defect committed inside the #1205 fix. ``seam_refusals()`` with no argument
    sums EVERY direction, and the "load" callers
    (``cache_controller.py:1701``, ``hybrid_cache_controller.py:600``) run on
    the controller's own background threads, concurrently with this fence. A
    delta taken across the fence window therefore picked up refusals the fence
    never caused and reported them as its own -- the label naming a population
    the number does not measure, which is the class this ticket exists for.
    """
    try:
        from sglang.srt.mem_cache.hicache_phase_guard import seam_refusals

        return int(seam_refusals("write"))
    except Exception:  # noqa: BLE001 - an instrument never breaks the fence
        return None


def _mamba_pin_skipped(tree_cache: Any) -> int:
    """The tree's monotone #581/#773 pin-budget refusal counter, or -1.

    -1 means UNMEASURED, and the caller must keep it distinguishable from 0 all
    the way into the log line. A probe whose miss path returns what a healthy
    zero returns is the exact defect #872 was opened for one level down, in
    ``_await_storage_acks``; repeating it here to save a sentinel would be
    building the same blind spot on purpose.
    """
    value = getattr(tree_cache, "_mamba_pin_skipped", None)
    if isinstance(value, bool) or not isinstance(value, int):
        return -1
    return value


def flip_writeback(
    tree_cache: Any,
    *,
    deadline_s: float = 2.0,
    ceiling_s: Optional[float] = None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> FlipWritebackReport:
    """Stage the tree's warm prefixes to the canonical store, bounded.

    Called at the flip's quiescent seam, with requests parked, BEFORE the
    cutover -- while the pool this cache is bound to is still the live one.
    That ordering is the whole point: after the cutover the same copy would
    read a pool the model is no longer writing into.

    #1028: THE BOUND IS ON STALLING, NOT ON DURATION.

    ``deadline_s`` was a flat wall-clock cut, and that is the wrong axis: it
    cuts a fence that is making steady progress at exactly the same moment as
    one that is wedged. MEASURED, boot_855_wt1016 19:22:42 and 19:22:45, the
    same four nodes fenced twice three seconds apart: ``acked=0 outstanding=4``
    then ``acked=1 outstanding=3``, both ``elapsed=2.000s/2.000s``. Acks were
    landing at roughly one per three seconds -- the backups were not stuck,
    they were slow -- so the flat bound threw away a 13179-token prompt's
    persistence that another ~10 s would have completed. The consequence is
    the #1028 churn: ``host_hit=0`` on re-admission, a full recompute of the
    whole prompt, and the decode phase armed away from it mid-recompute.

    So the bound that stays is the one that CONVERGES, the same argument
    ``phase_policy`` makes for its drain stall: ``deadline_s`` is now the
    NO-PROGRESS bound -- give up after this long with no acknowledgement --
    and ``ceiling_s`` is a hard total so a backend that acks forever in small
    increments still cannot hold the seam open without end.

    Behaviour is UNCHANGED for the two cases that dominate: a fence with
    nothing outstanding still returns in milliseconds, and a genuinely stuck
    backend still gives up after ``deadline_s`` exactly as before. Only the
    slow-but-progressing case, which is the defect, behaves differently.

    Nothing here aborts the flip -- this function reports and the CALLER
    decides. Under #1028 the caller does now act on ``outstanding > 0``
    (phase_flip_runtime's unanimous abandon), which is where such a verdict
    belongs: group-agreed, before anything is mutated.
    """
    require_canonical_store(tree_cache)

    started = now()
    if ceiling_s is None:
        ceiling_s = float(deadline_s)
    ceiling_s = max(float(ceiling_s), float(deadline_s))
    deadline = started + float(deadline_s)
    ceiling = started + float(ceiling_s)

    nodes = _hashed_nodes(tree_cache)
    # #969E: the discriminator, logged beside eligible. See _total_nodes.
    _969e_total = _total_nodes(tree_cache)
    logger.warning(
        "#969E FENCE-NODES total=%d eligible=%d", _969e_total, len(nodes)
    )
    # #1063: snapshot what the store ACTUALLY holds for these nodes, before the
    # staging below changes anything. Two-point measurement -- see
    # `_1063_record_fence` for why one observation cannot separate
    # "evicted since" from "never written".
    _1063_record_fence(tree_cache, nodes)
    root = getattr(tree_cache, "root_node", None)
    staged = 0
    already = 0
    skipped_unbacked_parent = 0
    refused_silently = 0
    mamba_pin_before = _mamba_pin_skipped(tree_cache)
    seam_refused_before = _seam_refusals_or_none()
    for node in nodes:
        if getattr(node, "backuped", False):
            already += 1
            continue
        # #841: THE CONTIGUOUS-BACKUP LAW. This loop used to reason that
        # "the parent is in this same list, earlier, so the invariant holds
        # by construction", and staged the child regardless with
        # write_back=True -- which ALSO disarms write_backup's own parent gate
        # at unified_radix_cache.py:2129-2134 (cited as :1943-1948 until
        # 2026-08-26; that range is an LRU helper and always was). Both halves
        # fail: an unhashed parent is never in the list at all (see
        # _hashed_nodes), and a parent that IS in the list can still have had
        # its own write_backup refused a moment earlier -- the mamba pin
        # budget, the rank-uniform host floor and the staging ring each return
        # 0 without raising.
        #
        # A child staged over that gap is not merely a lost page. Under any
        # write policy other than `write_back` the tree's own idle-path
        # sanity check enforces the law and ABORTS every rank on it, and an
        # un-backed parent above a backed child can be deleted by a device
        # eviction, orphaning the subtree in the host ledger. That is the
        # window-5 crash. The store being content-addressed per page does not
        # make the tree's ledger consistent.
        parent = getattr(node, "parent", None)
        if (
            parent is not None
            and parent is not root
            and not getattr(parent, "backuped", False)
        ):
            skipped_unbacked_parent += 1
            continue
        try:
            written = tree_cache.write_backup(node, write_back=True)
        except Exception as e:  # an instrument at a seam, never a gate
            logger.warning("%s staging node failed: %s", LOG_PREFIX, e)
            continue
        if written:
            staged += 1
        else:
            # #872 THE SILENT ZERO. `write_backup` has SEVEN paths that return
            # a bare 0 without raising and without logging -- in
            # `unified_radix_cache.py`: no controller (:2111), the #581/#773
            # mamba write-through pin budget (:2125), the write-through parent
            # recursion (:2131, unreachable from here because write_back=True),
            # the #639/#645 rank-uniform host floor (:2210), an `evict_host`
            # that freed less than `needed` (:2215), the #810 staging-ring
            # refusal (:2226), and a `cache_controller.write` that returned
            # None (:2234). Every one of them means THIS node will not cross
            # the flip: it has no host copy, so there is nothing for
            # `write_backup_storage` to read, and the tree is dropped at the
            # cutover regardless.
            #
            # `if written: staged += 1` alone discarded that verdict, so a node
            # the fence TRIED and FAILED to persist counted exactly like a node
            # the fence never had to touch. That is the #872 probe failure --
            # a miss path that returns what a healthy no-op returns -- one
            # level up from the drain lookup below, and it is why the R7 lines
            # `eligible=9 staged=0 already_staged=8` needed subtraction to be
            # read at all: 3 of 96 fence lines, one node each, out of 408
            # eligible node visits in that boot.
            #
            # Counted, never repaired: there is no fallback to reach for. The
            # store write reads `host_value`, which is exactly what the refusal
            # withheld, so staging this node anyway would mean writing bytes
            # whose host state does not exist. An honest skip with a named line
            # is the whole of the correct behaviour here.
            refused_silently += 1
    if skipped_unbacked_parent:
        logger.info(
            "%s skipped %d node(s) whose parent carries no host copy; staging "
            "them would break the contiguous-backup law the tree asserts on "
            "the idle path.",
            LOG_PREFIX,
            skipped_unbacked_parent,
        )

    mamba_pin_after = _mamba_pin_skipped(tree_cache)
    if mamba_pin_before < 0 or mamba_pin_after < 0:
        refused_mamba_pin = -1
    else:
        refused_mamba_pin = max(0, mamba_pin_after - mamba_pin_before)
    seam_refused_after = _seam_refusals_or_none()
    if seam_refused_before is None or seam_refused_after is None:
        seam_refused = -1
    else:
        seam_refused = max(0, seam_refused_after - seam_refused_before)
    if refused_silently:
        logger.warning(
            "%s #872 %d node(s) were handed to write_backup and refused "
            "without raising; those prefixes have no host copy, so nothing "
            "can be written to the store for them and they are lost when the "
            "tree is dropped at the cutover. Attributable to the mamba "
            "write-through pin budget: %s. The other refusal paths are the "
            "rank-uniform host floor (unified_radix_cache.py:2210), a short "
            "evict_host (:2215), the staging ring (:2226) and a host write "
            "that returned None (:2234). Device-tier WRITES the flip seam "
            "refused process-wide while this fence ran (#1205; every \"write\" "
            "caller of device_tier_disarmed, NOT a share of the count above "
            "and not per-node -- consume_gate refuses a whole queue and counts "
            "one): %s.",
            LOG_PREFIX,
            refused_silently,
            "unmeasured" if refused_mamba_pin < 0 else refused_mamba_pin,
            "unmeasured" if seam_refused < 0 else seam_refused,
        )

    # Complete the device->host copies. This is also what triggers the
    # host->storage stage: the write ack path calls write_backup_storage.
    try:
        tree_cache.writing_check(write_back=True)
    except Exception as e:
        logger.warning("%s writing_check failed: %s", LOG_PREFIX, e)

    acknowledged, outstanding = _await_storage_acks(
        tree_cache, deadline=deadline, ceiling=ceiling, now=now, sleep=sleep
    )

    report = FlipWritebackReport(
        eligible=len(nodes),
        staged=staged,
        already_staged=already,
        acknowledged=acknowledged,
        outstanding=outstanding,
        elapsed_s=now() - started,
        deadline_s=float(deadline_s),
        refused_silently=refused_silently,
        refused_mamba_pin=refused_mamba_pin,
        ceiling_s=float(ceiling_s),
        seam_refused=seam_refused,
    )
    if report.complete:
        logger.info("%s %s", LOG_PREFIX, report.as_log())
    else:
        # Not an error: the bound did its job. Loud enough to be attributable
        # if post-flip hit rates disappoint.
        logger.warning(
            "%s deadline reached with backups still in flight; those prefixes "
            "will miss after the flip. %s",
            LOG_PREFIX,
            report.as_log(),
        )
    return report


def _await_storage_acks(
    tree_cache: Any,
    *,
    deadline: float,
    ceiling: Optional[float] = None,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[int, int]:
    """Drain backup acknowledgements until the tree is quiet or time is up.

    Drains LOCALLY. The cross-rank variant all-reduces the queue sizes, and a
    collective issued inside the flip seam is the #630 wedge shape; every rank
    runs this hook at the same seam anyway, so a rank draining its own acks
    needs no agreement with its peers.

    #872: BOTH LOOKUPS BELOW ARE LOUD ON A MISS, and that is the point of the
    ticket rather than a courtesy. These are duck-typed capability probes, and
    each one's miss path used to return exactly what a healthy no-op returns --
    "nothing was outstanding", "nothing was acknowledged" -- so a cache class
    that simply does not carry the name was INDISTINGUISHABLE from a cache with
    an empty queue. ``UnifiedRadixCache`` was such a class for the whole of its
    life on this path: it has ``_drain_storage_control_queues_impl`` and had no
    ``_local`` wrapper, so the fence skipped its wait entirely and reported
    ``acked=0`` on every fence of every boot while the guards above and the
    #871 streak alarm all read healthy.

    Naming the method on one more class fixes today's cache and leaves the next
    one to fail the same invisible way, so the probe itself now reports. The
    fence still does not raise -- a missed prefix is a later miss, never a
    wrong answer, and a flip may not die on an instrument -- but the condition
    is no longer silent, and it is no longer reported as completeness.
    """
    ongoing = getattr(tree_cache, "ongoing_backup", None)
    if ongoing is None:
        logger.error(
            "%s #872 UNSERVABLE CACHE: %s carries no `ongoing_backup`, so this "
            "fence cannot see whether ANY backup is in flight and reports the "
            "seam complete without having looked. Every prefix it was supposed "
            "to persist may be lost when the tree is dropped at the cutover, "
            "and the loss appears only as a post-flip cache miss.",
            LOG_PREFIX,
            type(tree_cache).__name__,
        )
        return 0, 0
    before = len(ongoing)
    drain = getattr(tree_cache, "_drain_storage_control_queues_local", None)
    if not callable(drain):
        # Reported as fully OUTSTANDING, never as acknowledged: nothing was
        # drained, so claiming otherwise would hand `persisted_nothing` and
        # `complete` a value neither of them earned. This is what makes the
        # #871 streak alarm able to see the condition at all.
        logger.error(
            "%s #872 UNSERVABLE CACHE: %s has no callable "
            "`_drain_storage_control_queues_local`, so the writeback fence "
            "cannot wait for storage acknowledgements and did not wait: %d "
            "backup(s) are left in flight and the tree is dropped at the "
            "cutover regardless. Add the local-drain wrapper to that class "
            "(all it needs is `_drain_storage_control_queues_impl` with None "
            "limits, as UnifiedRadixCache/HiRadixCache/HiMambaRadixCache do) "
            "-- a fence that never waits is retention that never happens.",
            LOG_PREFIX,
            type(tree_cache).__name__,
            before,
        )
        return 0, before
    # #1028: WAIT WHILE PROGRESS IS BEING MADE, not while a clock runs.
    #
    # `deadline` is the NO-PROGRESS bound and `ceiling` the hard total. The
    # stall clock is RESET every time the in-flight set shrinks, so a backend
    # acking one node every second keeps the fence open, while a backend that
    # acks nothing gives up after exactly the same interval as before this
    # change. `best` is monotone by construction (the set only shrinks as acks
    # land), so the reset cannot be triggered by noise.
    if ceiling is None:
        ceiling = deadline
    best = len(ongoing)
    stall_window = max(0.0, deadline - now())
    last_progress = now()
    while True:
        try:
            drain()
        except Exception as e:
            logger.warning("%s draining backup acks failed: %s", LOG_PREFIX, e)
            break
        if not ongoing:
            break
        remaining = len(ongoing)
        if remaining < best:
            best = remaining
            last_progress = now()
        t = now()
        if t - last_progress >= stall_window:
            break
        if t >= ceiling:
            logger.warning(
                "%s #1028 writeback fence hit its HARD CEILING with %d backup(s) "
                "still in flight while acks were still landing -- this is the "
                "bound that exists so a slow backend cannot hold the seam open "
                "without end, not a stall. Those prefixes will miss.",
                LOG_PREFIX,
                len(ongoing),
            )
            break
        sleep(_POLL_S)
    outstanding = len(ongoing)
    return max(0, before - outstanding), outstanding


def maybe_flip_writeback(
    scheduler: Any, *, deadline_s: Optional[float] = None
) -> Optional[FlipWritebackReport]:
    """Flag-gated entry point for the flip seam.

    Returns None when the feature is off -- the default, and byte-identical to
    not calling it. When it is ON and the store cannot serve the flip, the
    refusal propagates: an operator who asked for retention across the flip and
    silently got none is the failure mode worth being loud about.
    """
    server_args = getattr(scheduler, "server_args", None)
    if not getattr(server_args, "phase_flip_writeback", False):
        return None
    tree_cache = getattr(scheduler, "tree_cache", None)
    if tree_cache is None:
        raise FlipWritebackRefused(
            f"{LOG_PREFIX} --phase-flip-writeback is set but this scheduler has "
            "no tree cache to write back."
        )
    if deadline_s is None:
        deadline_s = float(
            getattr(server_args, "phase_flip_writeback_deadline_s", None) or 2.0
        )
    # #1028: the hard total. Deliberately NOT a second CLI knob for now -- it is
    # derived from the one the operator already sets, so there is no new
    # configuration surface and no way for the two to be set inconsistently.
    # The multiple is sized from the measurement that opened #1028 (four nodes
    # acking at ~1 per 3 s against a 2 s bound); 12x2 s = 24 s covers it with
    # room, and costs nothing whenever the fence is already complete.
    ceiling_s = float(
        getattr(server_args, "phase_flip_writeback_ceiling_s", None)
        or (deadline_s * _CEILING_MULTIPLE)
    )
    return flip_writeback(tree_cache, deadline_s=deadline_s, ceiling_s=ceiling_s)
