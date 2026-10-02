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

from typing import Sequence, Tuple

from sglang.srt.weg2 import l15_restore, l15_sample, l15_wake_check


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
    return (ok, bad, missing)
