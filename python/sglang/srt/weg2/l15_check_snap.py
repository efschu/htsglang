"""L15-CHECK-SNAP: the wake check's sample rows, snapshotted at the sleep.

N6e (0831cdc4d3) 17:34:38: the hold was fully L2-backed (HOSTLOCK-COVER
unbacked=0, TP0 refill 58263 rows) and still failed on ONE sampled row per
capped rank ('L15-CHECK-DIAG bad=1 foreign=1 row=49507 l2_slot=136825 gen=1'),
a prompt row of P's arena page -- the card and L2 disagreed at an unchanged
generation. Who changed what: the card during the pause, the card before the
sleep (a decode writing a held slot), or the L2 page in place?

At the sleep (after retain, capped rank, hold armed) this module copies the
SAME deterministic sample the wake will draw (l15_restore.sample_rows over the
manifest's owned_l2_rows) twice to the host: the device rows, and their L2
source loaded into a scratch. At the wake the extended diag compares, per bad
row: token index / rid / compacted slot / pre-move slot, the number of layers
that differ (all = a foreign token, some = a partial write), and

  dev_moved  = device(wake) != device(sleep)   -> the card changed during the pause
  l2_moved   = L2(wake)     != L2(sleep)       -> the L2 page was rewritten in place
  at_sleep   = device(sleep) != L2(sleep)      -> they already disagreed at the sleep

Process-local (sleep and wake run in the same scheduler process); one snapshot,
replaced at every held sleep. SGLANG_WEG2_L15_CHECK_SNAP=0 turns it off.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

#: the last held sleep's snapshot: {"fp": fingerprint, "rows": {row: dict}}
LAST: Optional[dict] = None
#: retain's last compact moves, NEW global slot -> OLD global slot
LAST_PRE_SLOT: Dict[int, int] = {}
#: the running wake check's (rank, manifest, prefix, device_pool), or None
_WAKE: Optional[tuple] = None


def set_wake_context(rank, m, prefix, device_pool) -> None:
    """The wake sample check's context for :func:`explain_current`; rank None clears."""
    global _WAKE
    _WAKE = None if rank is None else (int(rank), m, list(prefix), device_pool)


def explain_current(bad_rows) -> List[str]:
    """:func:`explain` under the context the wake set; none set -> no lines."""
    if _WAKE is None or not enabled():
        return []
    rank, m, prefix, pool = _WAKE
    return explain(rank, m, prefix, pool, bad_rows)


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("SGLANG_WEG2_L15_CHECK_SNAP", "1")).strip() != "0"


def note_moves(moves) -> None:
    """retain's compact plan moves (old, new): remembered new -> old."""
    global LAST_PRE_SLOT
    try:
        LAST_PRE_SLOT = {int(new): int(old) for old, new in moves}
    except Exception:  # noqa: BLE001 -- diagnostics only
        LAST_PRE_SLOT = {}


def token_of_rows(m, rank: int, prefix: Sequence[int]) -> Dict[int, Tuple[str, int, int]]:
    """compact row -> (rid, token index, compacted global slot), first visit."""
    from sglang.srt.weg2 import l15_restore

    out: Dict[int, Tuple[str, int, int]] = {}
    for span, i, slot in l15_restore._owned_tokens(m, rank, list(prefix)):
        row = l15_restore._compact_row(list(prefix), rank, int(slot))
        out.setdefault(int(row), (str(span.rid), int(i), int(slot)))
    return out


def layer_bounds(pool) -> List[Tuple[int, int]]:
    """[start, end) of each k then v layer inside a read_rows flat row."""
    p = getattr(pool, "full_kv_pool", pool)
    out, pos = [], 0
    for b in list(p.k_buffer) + list(p.v_buffer):
        n = int(b[0].numel())
        out.append((pos, pos + n))
        pos += n
    return out


def _sample(m, rank, prefix, k):
    from sglang.srt.weg2 import l15_restore, l15_sample

    plan = [(str(rids[0]), row, slot, gen)
            for row, slot, gen, _lane, rids in l15_restore.owned_l2_rows(m, rank, list(prefix))]
    return l15_sample.sample_plan(plan, k)


def snap_at_sleep(m, rank: int, prefix: Sequence[int], device_pool, host_pool,
                  k: int = 64) -> int:
    """Copy the wake's sample rows (device + L2) to the host. Never raises."""
    global LAST
    if not enabled() or m is None:
        return 0
    try:
        from sglang.srt.weg2 import l15_manifest, l15_sample, l15_scratch

        sampled = _sample(m, rank, prefix, k)
        if not sampled:
            return 0
        page_tokens = max(1, int(getattr(host_pool, "_arena_page_tokens", 1)))
        dev = l15_sample.read_rows(device_pool, [int(t[1]) for t in sampled])
        scratch = l15_scratch.make_scratch_pool(device_pool, len(sampled))
        try:
            srows = l15_sample.load_into_scratch(sampled, host_pool, scratch, page_tokens)
            l2 = l15_sample.read_rows(scratch, srows)
        finally:
            scratch.free()
        rows = {}
        for t, d, s in zip(sampled, dev, l2):
            rows[int(t[1])] = {"dev": d.detach().to("cpu", copy=True),
                               "l2": s.detach().to("cpu", copy=True)}
        LAST = {"fp": int(l15_manifest.fingerprint(m)), "rows": rows}
        same = sum(1 for r in rows.values() if torch.equal(r["dev"], r["l2"]))
        logger.info("L15-CHECK-SNAP rank=%d rows=%d equal_at_sleep=%d (device vs L2 of the wake's "
                    "sample, copied at the sleep)", int(rank), len(rows), same)
        return len(rows)
    except Exception as exc:  # noqa: BLE001 -- diagnostics only
        logger.info("L15-CHECK-SNAP failed (%s: %s)", type(exc).__name__, exc)
        LAST = None
        return 0


def explain(rank: int, m, prefix: Sequence[int], device_pool, bad_rows) -> List[str]:
    """One 'L15-CHECK-WHO' line per bad row ``(row, dev_now, l2_now)``."""
    lines: List[str] = []
    try:
        from sglang.srt.weg2 import l15_manifest

        tok = token_of_rows(m, rank, prefix) if m is not None else {}
        bounds = layer_bounds(device_pool)
        snap = LAST if (LAST is not None and m is not None
                        and LAST.get("fp") == int(l15_manifest.fingerprint(m))) else None
        for row, dev_now, l2_now in bad_rows:
            rid, i, slot = tok.get(int(row), ("?", -1, -1))
            pre = LAST_PRE_SLOT.get(int(slot), int(slot)) if slot >= 0 else -1
            diff = [j for j, (a, b) in enumerate(bounds)
                    if not torch.equal(dev_now[a:b].cpu(), l2_now[a:b].cpu())]
            s = snap["rows"].get(int(row)) if snap is not None else None
            if s is None:
                tail = "snap=none"
            else:
                tail = "dev_moved=%d l2_moved=%d at_sleep=%d" % (
                    int(not torch.equal(dev_now.cpu(), s["dev"])),
                    int(not torch.equal(l2_now.cpu(), s["l2"])),
                    int(not torch.equal(s["dev"], s["l2"])))
            lines.append("L15-CHECK-WHO rank=%d row=%d rid=%s token=%d slot=%d pre_slot=%d "
                         "layers_diff=%d/%d first_layers=%s %s"
                         % (int(rank), int(row), rid, i, slot, pre, len(diff), len(bounds),
                            diff[:6], tail))
    except Exception as exc:  # noqa: BLE001 -- diagnostics only
        lines.append("L15-CHECK-WHO failed (%s: %s)" % (type(exc).__name__, exc))
    return lines
