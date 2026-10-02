"""L15-SLEEP-AGREE: the D ranks decide the hold TOGETHER at the sleep.

N4f (257e0900f4): the cap-0 rank's (empty) keep arm succeeded and it kept
the held chains in its tree, while the capped ranks' arms failed and they
flushed plain -- the trees diverged across the sleep. The wake then dropped
the cap-0 rank's tree alone (L15-FIX-NOHOLD-TREE) and with it the #248 hold
prefetch the early wake read had just inserted: the parked request lost its
head and store credit on one rank only -> W50 after the first byte (6 client
aborts). And every rank paid bind + retain (0.5-1.4 s) for a hold that was
discarded.

Two host gathers over D's group, both at positions every D rank reaches:

* PRE (before bind/retain): every rank votes whether it CAN hold this sleep
  -- a capped rank needs its hold bases split on the saver (the first sleep
  of a boot cannot hold: the split runs in its plain flush), a cap-0 rank
  needs the L2 refill (REFILL on). Any "no" -> no rank binds or retains,
  the plain flush runs everywhere (costs one gather, not a second).
* POST (after the keep arm): every rank votes whether its round armed; any
  "no" -> the ranks that armed undo it (manifest discarded, keep spans
  cleared, sleep refs released) and flush plain -- no rank keeps chains the
  others dropped.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Tuple


def gather_for(sched) -> Callable[[object], List[object]]:
    """One all_gather_object over the scheduler's world group (D's ranks)."""
    import torch

    wg = getattr(sched, "world_group", None)
    grp = getattr(wg, "cpu_group", None) if wg is not None else None
    world = torch.distributed.get_world_size(group=grp) if grp is not None else 1

    def gather(v):
        if grp is None or world <= 1:
            return [v]
        out = [None] * world
        torch.distributed.all_gather_object(out, v, group=grp)
        return out

    return gather


def can_hold(cap_rows: int, env, split_ready: Callable[[], bool]) -> Optional[str]:
    """None when this rank can hold this sleep, else the named reason."""
    from sglang.srt.weg2 import l15_plan

    if int(cap_rows) <= 0:
        if not l15_plan._switch(env, "SGLANG_WEG2_L15_REFILL"):
            return "cap-0 rank without the L2 refill (SGLANG_WEG2_L15_REFILL=0)"
        return None
    if not split_ready():
        return "hold bases not split on the saver yet"
    return None


def agree(why: Optional[str], gather) -> Optional[str]:
    """None when EVERY rank votes ok, else the first named refusal."""
    votes = gather(why)
    bad = [v for v in votes if v is not None]
    return None if not bad else str(bad[0])


def post_vote(res, cap_rows: int, prefix: Optional[Sequence[int]] = None,
              rank: Optional[int] = None) -> Optional[str]:
    """This rank's POST vote: None when its round armed AND it will be able
    to honour the hold at the wake. A cap-0 rank refills from L2, which
    needs every held span's END anchor L2 identity -- without it the wake
    votes no hold (N4f: 'L15-REFILL anchors-missing: votes no hold' -> the
    group fell back and the wake reset dropped the early hold reads). That
    is known HERE, so the whole group flushes plain now instead."""
    if res is None:
        return "round did not arm"
    if int(cap_rows) <= 0:
        spans = getattr(getattr(res, "manifest", None), "spans", ()) or ()
        missing = [str(sp.rid) for sp in spans
                   if int(getattr(sp, "anchor_l2_slot", -1)) < 0]
        if missing:
            return "cap-0 rank: END anchor without L2 identity for %s" % missing[:4]
        # L15-UNBACKED-REFUSE (N5n 14:44:17, dac8b62b8c): a LONG freshly handed
        # over from P sat on D with ~2% of its chain written back to L2
        # (L15-HOSTLOCK slots=2304 for 133k held tokens); the cap-0 rank's wake
        # refill then had 1006 rows to load of ~47k owned, and the hold was
        # paid (480 ms) and dropped at the wake (4.2 s to the first token). A
        # cap-0 rank keeps NOTHING on its card -- every owned held token must
        # come back from L2, so one token without an L2 source means the hold
        # cannot be honoured: refuse it HERE (every rank then flushes plain)
        # rather than wait for the write-through inside the flip.
        if prefix is not None and rank is not None:
            from sglang.srt.weg2 import l15_restore

            m = getattr(res, "manifest", None)
            if m is not None:
                miss = int(l15_restore.count_missing(m, int(rank), list(prefix)))
                if miss:
                    return ("cap-0 rank: %d owned held token(s) without an L2 "
                            "source (unbacked chain, refill impossible)" % miss)
    return None


def rank_prefix(sched) -> Tuple[int, List[int]]:
    """(tp_rank, cumulative CP token-ratio prefix) the restore plans use."""
    tp = int(getattr(sched, "tp_size", 0) or getattr(
        getattr(sched, "server_args", None), "tp_size", 1) or 1)
    rank = int(getattr(getattr(sched, "ps", None), "tp_rank", 0) or 0)
    try:
        from sglang.srt.distributed.utils import get_cp_token_ratios

        ratios = get_cp_token_ratios()
    except Exception:  # noqa: BLE001 -- no CP ratios: even split
        ratios = None
    vw = ([int(x) for x in ratios]
          if ratios is not None and len(ratios) == tp
          and all(int(x) > 0 for x in ratios) else [1] * tp)
    prefix = [0]
    for x in vw:
        prefix.append(prefix[-1] + x)
    return rank, prefix


def undo_armed(sched, manifest_path: str, log) -> None:
    """A rank whose round armed while a peer's did not: give it all back
    (the caller then runs the plain flush on this rank too). Never raises."""
    from sglang.srt.weg2 import l15_keep_arm

    try:
        l15_keep_arm.discard_manifest(manifest_path)
    except Exception as exc:  # noqa: BLE001
        log("L15-SLEEP-AGREE undo: manifest discard failed (%s)" % (exc,))
    wu = getattr(sched, "weight_updater", None)
    for name in ("_l15_clear_tms_keep_spans", "_l15_release_host_hold_refs"):
        fn = getattr(wu, name, None) if wu is not None else None
        if fn is None:
            continue
        try:
            fn(sched)
        except Exception as exc:  # noqa: BLE001
            log("L15-SLEEP-AGREE undo: %s failed (%s)" % (name, exc))
    try:
        from sglang.srt.weg2 import l15_share_publish  # noqa: F401
        pub = getattr(sched, "_l15_share_pub", None)
        if pub is not None:
            pub.close()
            sched._l15_share_pub = None
    except Exception:  # noqa: BLE001
        pass


def split_ready_native() -> bool:
    """The capped rank's hold bases exist on the saver as span extents
    covering their hold regions (L15-EXTENTS truth; an old saver without
    tms_list_extents answers from the split record alone)."""
    from sglang.srt.weg2 import l15_keep_split
    from sglang.srt.weg2.l15_hold_share import list_extents, native_cover

    holds = dict(l15_keep_split._HOLD)
    if not holds:
        return False
    for ptr, regions in holds.items():
        native = list_extents(int(ptr))
        if native is None:
            continue
        if native_cover(native, list(regions)) is None:
            return False
    return True


def env_on(env=None) -> bool:
    """Default on; =0 restores the old per-rank decision (tests/diagnosis)."""
    env = os.environ if env is None else env
    return str(env.get("SGLANG_WEG2_L15_SLEEP_AGREE", "1")).strip() != "0"
