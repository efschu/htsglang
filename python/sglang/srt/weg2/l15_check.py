"""L15-12c-E1S: the wake sample CHECK as one pure composition
(plan L15-12-PART3-PLAN sec 4).

Device rows vs their L2 source, giving the (ok, bad, missing) triple the
wake folds into l15_restore.check_vote. Composition only -- the pieces
are l15_sample (plan/subset/load/read) and l15_wake_check
(sample_rows_equal); this module adds the missing count and the
load-failure fold, and nothing else. Pure: no collective, no logging,
no scheduler import.
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

import torch

from sglang.srt.weg2 import l15_restore, l15_sample, l15_wake_check

logger = logging.getLogger(__name__)


def sample_check(
    plan: Sequence[l15_sample.PlanRow],
    host_pool,
    live_pool,
    scratch_pool,
    page_tokens: int,
    k: int = 64,
) -> Tuple[int, int, int]:
    """(ok, bad, missing) over the k-sample of ``plan`` (4-tuples
    (rid, compact_row, l2_slot, l2_gen), the refill_plan + rid-tag shape).

    missing is counted BEFORE sample_plan's exclusion: the deterministic
    k-sample is l15_restore.sample_rows over ALL of the plan's compact
    rows, and every sampled row with l2_slot < 0 has no L2 source and is
    never byte-compared -- each one counts as missing. Rows the sample
    did not pick are irrelevant to this wake.

    ok/bad: the sampled rows' L2 source is loaded into scratch rows
    0..n-1 (l15_sample.load_into_scratch, ONE call, all-or-nothing) and
    the flat device rows of live_pool are compared positionwise with the
    scratch rows (l15_wake_check.sample_rows_equal). An L15SampleError
    (P>1 refused, a l2_slot -1 slipped through, any load failure)
    returns (0, len(sampled), missing) -- the whole group must refuse,
    never keep an unverified hold; no log here, the caller logs.

    Empty sample (empty plan, or every sampled row missing) ->
    (0, 0, missing)."""
    chosen = set(l15_restore.sample_rows([int(t[1]) for t in plan], k))
    missing = sum(1 for t in plan if int(t[1]) in chosen and int(t[2]) < 0)
    sampled = l15_sample.sample_plan(plan, k)
    if not sampled:
        return (0, 0, missing)
    try:
        scratch = l15_sample.load_into_scratch(
            sampled, host_pool, scratch_pool, page_tokens
        )
    except l15_sample.L15SampleError:
        return (0, len(sampled), missing)
    device = l15_sample.read_rows(live_pool, [int(t[1]) for t in sampled])
    source = l15_sample.read_rows(scratch_pool, scratch)
    ok, bad = l15_wake_check.sample_rows_equal(device, source)
    if bad:
        try:
            logger.warning("%s", check_diag(device, source, sampled))
        except Exception as exc:  # noqa: BLE001 -- diagnostics only
            logger.warning("L15-CHECK-DIAG failed: %r", exc)
    return (ok, bad, missing)


def check_diag(device_rows, l2_rows, sampled) -> str:
    """L15-CHECK-DIAG: classify the bad sampled rows. For each L2 row that
    differs from its own live row, look for an EQUAL live row elsewhere in
    the sample: found -> misaligned (offset j - i recorded), an all-zero L2
    row -> zero, else -> foreign. First bad row's identity is printed."""
    n = min(len(device_rows), len(l2_rows))
    misaligned, zero, foreign = 0, 0, 0
    offsets = set()
    first = None
    for i in range(n):
        if torch.equal(device_rows[i], l2_rows[i]):
            continue
        if first is None:
            first = i
        hit = next((j for j in range(n) if j != i
                    and torch.equal(device_rows[j], l2_rows[i])), None)
        if hit is not None:
            misaligned += 1
            offsets.add(hit - i)
        elif not bool(torch.any(l2_rows[i] != 0)):
            zero += 1
        else:
            foreign += 1
    bad = misaligned + zero + foreign
    who = ""
    if first is not None:
        t = sampled[first]
        dv, lv = device_rows[first].float(), l2_rows[first].float()
        who = (" first=(rid=%s row=%s l2_slot=%s gen=%s live_absmax=%.4g "
               "l2_absmax=%.4g)" % (t[0], t[1], t[2], t[3],
                                    float(dv.abs().max()) if dv.numel() else 0.0,
                                    float(lv.abs().max()) if lv.numel() else 0.0))
    return ("L15-CHECK-DIAG bad=%d misaligned=%d offsets=%s zero=%d foreign=%d%s"
            % (bad, misaligned, sorted(offsets)[:6], zero, foreign, who))
