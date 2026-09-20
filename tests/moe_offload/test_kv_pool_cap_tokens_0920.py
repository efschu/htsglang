# SPDX-License-Identifier: Apache-2.0
"""Tasks #14/#48 (20.09., fn8am): --max-total-tokens is a GLOBAL context budget
under weighted uneven DCP, not a per-rank pool size.

WHAT fn8am MEASURED
(``/spinning/evidence-665-f1/boot_fn_fn8am_20260920T103851Z.server.log``).
The boot carried ``--max-total-tokens 90816``, chosen as rank 0's physical pool
need (``unit 8256 x ratio 11``).  The log then says, verbatim::

    Uneven-DCP token sizing: rank 0 local capacity 90816 tokens / ratio 11 =
    unit 8256; min-reduced unit 8256 -> projected 264192 -> EFFECTIVE
    max_total_num_tokens 90816 (bound by --max-total-tokens user limit 90816;
    vector [11, 11, 10], hybrid mamba cap 262151).

and the 259415-token needle came back as::

    Input length (259415 tokens) exceeds the maximum allowed length (90810 ...

90810 is ``(90816 // 10) * 10`` -- rank 2's reconstructed local capacity.  The
POOLS were right (``KV Cache is allocated ... #tokens: 31229`` on the ratio-11
ranks, ``28390`` on the ratio-10 one, i.e. ``C * ratio_r / S``); the CEILING was
not.

ROOT, in ``ModelRunnerKVCacheMixin._apply_token_constraints``: the flag is
applied TWICE to two different quantities -- first to ``P_r``, this rank's
physical token capacity, then thirty lines later to
``C = min_r(P_r // ratio_r) * S``, the global context budget.  The second clamp
is what ``max_req_input_len`` is built from (``scheduler.py:5051`` ->
``managers/utils.py:202``).

FIX: ``SGLANG_KV_POOL_CAP_TOKENS`` caps only ``P_r``.  ``--max-total-tokens``
keeps both of its meanings byte-identically, so no existing recipe moves.
Hermetic: no CUDA, no GPU.
"""

import pytest

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    KV_POOL_CAP_ENV,
    kv_pool_cap_tokens_for_rank,
)

#: fn8am, verbatim from the log.
FN8AM_RATIOS = (11, 11, 10)
FN8AM_PROFILED = (358784, 359424, 329472)  # KV pool sizing, per rank
FN8AM_CAP = 90816  # --max-total-tokens as the boot carried it
FN8AM_NEEDLE = 259415
FN8AM_REFUSED_AT = 90810  # "maximum allowed length (90810 tokens)"


def _two_clamps(profiled, ratios, user_limit):
    """The arithmetic of _apply_token_constraints, both clamps, in order."""
    p_r = [min(p, user_limit) if user_limit is not None else p for p in profiled]
    unit = min(p // r for p, r in zip(p_r, ratios))
    c = unit * sum(ratios)
    if user_limit is not None:
        c = min(c, user_limit)
    return p_r, unit, c


def test_the_fn8am_refusal_is_reproduced_by_the_two_clamps():
    p_r, unit, c = _two_clamps(FN8AM_PROFILED, FN8AM_RATIOS, FN8AM_CAP)
    assert p_r == [FN8AM_CAP] * 3  # first clamp: every rank's pool
    assert unit == 8256 and unit * sum(FN8AM_RATIOS) == 264192  # the projection
    assert c == FN8AM_CAP  # second clamp: the GLOBAL budget, and this is the bug
    # rank 2 reconstructs its share from the clamped C and that is the number
    # the needle was measured against.
    assert (c // 10) * 10 == FN8AM_REFUSED_AT < FN8AM_NEEDLE


def test_without_the_flag_the_same_pools_serve_the_whole_context():
    """The pool cap reaches the same per-rank pools through the unit relation,
    while C stays the projection -- so the needle fits."""
    capped = [min(p, 90112) for p in FN8AM_PROFILED]
    _, unit, c = _two_clamps(capped, FN8AM_RATIOS, None)
    assert unit == 8192 and c == 262144 > FN8AM_NEEDLE
    # and each rank still physically stores only its share.
    assert [c * r // sum(FN8AM_RATIOS) for r in FN8AM_RATIOS] == [90112, 90112, 81920]


# --- the env ---------------------------------------------------------------


def test_unset_is_none_so_sizing_is_byte_identical():
    assert kv_pool_cap_tokens_for_rank("", 0) is None
    assert kv_pool_cap_tokens_for_rank("   ", 2) is None
    assert kv_pool_cap_tokens_for_rank(None, 0) is None


def test_a_scalar_applies_to_every_rank():
    assert [kv_pool_cap_tokens_for_rank("90112", r) for r in range(3)] == [90112] * 3


def test_a_vector_is_read_per_rank():
    text = "90112,90112,81920"
    assert [kv_pool_cap_tokens_for_rank(text, r) for r in range(3)] == [
        90112,
        90112,
        81920,
    ]


def test_an_unparsable_value_is_refused_by_name_not_ignored():
    """A pool cap that silently does nothing costs a GPU window measuring the
    configuration it was supposed to replace."""
    with pytest.raises(ValueError, match=KV_POOL_CAP_ENV):
        kv_pool_cap_tokens_for_rank("90112,oops", 1)
    with pytest.raises(ValueError, match="not an integer"):
        kv_pool_cap_tokens_for_rank("0.9", 0)


def test_a_non_positive_cap_is_refused():
    with pytest.raises(ValueError, match="cannot serve"):
        kv_pool_cap_tokens_for_rank("0", 0)
    with pytest.raises(ValueError, match="cannot serve"):
        kv_pool_cap_tokens_for_rank("90112,-1,81920", 1)


def test_a_vector_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="one entry per rank"):
        kv_pool_cap_tokens_for_rank("90112,81920", 2)


def test_the_planner_emits_the_cap_and_the_ceiling_as_two_numbers():
    from sglang.srt.planner import expert_pool_budget as epb

    from test_expert_pool_budget_0920 import FN8AJ_LOG

    env = epb.plan_expert_pool(
        epb.parse_boot_log(FN8AJ_LOG), ctx_tokens=262144, strict=False
    ).env()
    assert env["SGLANG_KV_POOL_CAP_TOKENS"] == "90112,90112,81920"
    assert env["MAX_TOTAL_TOKENS"] == "262144"
    # the two are related by the unit relation, and neither is the other.
    caps = [int(x) for x in env["SGLANG_KV_POOL_CAP_TOKENS"].split(",")]
    assert min(c // r for c, r in zip(caps, FN8AM_RATIOS)) * sum(FN8AM_RATIOS) == int(
        env["MAX_TOTAL_TOKENS"]
    )
