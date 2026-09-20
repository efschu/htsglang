# SPDX-License-Identifier: Apache-2.0
"""Slice 1 of the Next-Flash P/D flip: one launcher line, two groups.

Hermetic: no torch, no NVML, no device, no live environment. Every number is
either injected or carries the boot log line it was read from.
"""
from __future__ import annotations

import pytest

from sglang.srt.flip_nextflash_groups import (
    D_GROUP,
    FLIP_CONTEXT_TOKENS,
    GROUP_ENV_VALUES,
    GROUP_OWNED_ENV,
    P_GROUP,
    REQ_TO_TOKEN_EXTRA_MEASURED,
    Weg2FlipGroupEnvInherited,
    build_flip_groups,
    build_group_env,
    solve_group_context,
)
from sglang.srt.flip_nextflash_plan import Weg2FlipKvRelayInfeasible


# --------------------------------------------------------------------------
# The law: 262144, both groups
# --------------------------------------------------------------------------
def test_the_mandatory_context_is_262144():
    """Memory KONTEXT-262K-PFLICHT: not a default, a law."""
    assert FLIP_CONTEXT_TOKENS == 262144


def test_the_measured_hybrid_cap_reproduces_fnFA19_line_1027():
    """fnFA19 printed ``Hybrid mamba/attention KV cap: 270000 -> 262151``.

    262151 = 1 x (262144 + 7). If this test ever fails, the per-request
    headroom changed and every pool number in the design moved with it.
    """
    plan = solve_group_context(D_GROUP, 262144, 1, REQ_TO_TOKEN_EXTRA_MEASURED, 270000)
    assert plan.hybrid_cap_tokens == 262151
    assert plan.reachable_tokens == 262151


# --------------------------------------------------------------------------
# The --max-total-tokens trap -- the actual slice-1 finding
# --------------------------------------------------------------------------
def test_the_run_fn7s2_default_max_total_tokens_refuses_at_262k():
    """``run_fn7s2.sh`` default ``--max-total-tokens 40000`` at CTX=262144.

    This is the trap: the runtime does NOT refuse it (a user limit below the
    profiled capacity is the documented use of the flag), the boot comes up
    healthy, and the 259k needle fails. W113 moves it to the desk.
    """
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        solve_group_context(P_GROUP, 262144, 1, REQ_TO_TOKEN_EXTRA_MEASURED, 40000)
    msg = str(exc.value)
    assert "W113 Weg2FlipKvRelayInfeasible" in msg
    assert "40000" in msg
    assert "262144" in msg
    # short by 262144 - 40000
    assert "222144" in msg
    # the refusal must say WHY nothing else catches it
    assert "does NOT" in msg and "refuse at boot" in msg


def test_the_fnFA19_max_total_tokens_passes():
    """``launch_fnFA19.sh`` passes ``--max-total-tokens 270000``: above the
    hybrid cap, so the cap binds and the context is reachable."""
    plan = solve_group_context(D_GROUP, 262144, 1, REQ_TO_TOKEN_EXTRA_MEASURED, 270000)
    assert plan.reachable_tokens >= 262144


def test_exactly_the_hybrid_cap_is_enough_and_one_below_is_not():
    """The boundary is the context, not the cap: a pool of exactly
    ``context_tokens`` is sufficient; one token less is W113."""
    ok = solve_group_context(P_GROUP, 1000, 1, 0, 1000)
    assert ok.reachable_tokens == 1000
    with pytest.raises(Weg2FlipKvRelayInfeasible):
        solve_group_context(P_GROUP, 1000, 1, 0, 999)


def test_no_user_limit_means_the_hybrid_cap_alone_decides():
    plan = solve_group_context(P_GROUP, 262144, 1, 7, None)
    assert plan.user_limit_tokens is None
    assert plan.reachable_tokens == 262151


def test_the_refusal_names_the_hybrid_cap_when_the_cap_is_what_binds():
    """max_running_requests=0 is rejected; but a cap that binds BELOW the
    context (possible only with a negative-headroom-free arithmetic error)
    must name the cap, not a user limit that is not there."""
    # A concurrency of 1 with a context of 100 and a user limit of 1000:
    # the cap (100) binds, the limit does not. Reachable == context -> ok.
    plan = solve_group_context(P_GROUP, 100, 1, 0, 1000)
    assert plan.hybrid_cap_tokens == 100
    assert plan.reachable_tokens == 100


@pytest.mark.parametrize("bad", [0, -1, -262144])
def test_a_non_context_is_refused(bad):
    with pytest.raises(Weg2FlipKvRelayInfeasible):
        solve_group_context(P_GROUP, bad)


def test_a_non_positive_concurrency_is_refused_not_silently_uncapped():
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        solve_group_context(P_GROUP, 262144, 0)
    assert "max_running_requests=0" in str(exc.value)


def test_a_zero_pool_is_refused():
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        solve_group_context(P_GROUP, 262144, 1, 7, 0)
    assert "not a pool" in str(exc.value)


# --------------------------------------------------------------------------
# Seam D: the DCP axis must not be inherited
# --------------------------------------------------------------------------
def test_p_and_d_get_opposite_dcp_values():
    p = build_group_env(P_GROUP, {"PYTHONPATH": "/x"})
    d = build_group_env(D_GROUP, {"PYTHONPATH": "/x"})
    assert p.env["SGLANG_UNEVEN_DCP"] == "1"
    assert p.env["SGLANG_UNEVEN_DCP_WEIGHTED"] == "1"
    assert d.env["SGLANG_UNEVEN_DCP"] == "0"
    assert d.env["SGLANG_UNEVEN_DCP_WEIGHTED"] == "0"
    # the common part survives into both
    assert p.env["PYTHONPATH"] == d.env["PYTHONPATH"] == "/x"


def test_neither_group_unsets_the_axis():
    """Memory ``Uneven nie ab``: an empty env override is forbidden. Both
    sides state a value; neither states an empty string."""
    for group, values in GROUP_ENV_VALUES.items():
        for key in GROUP_OWNED_ENV:
            assert key in values, f"{group} does not state {key}"
            assert values[key] != "", f"{group} unsets {key} -- forbidden"


def test_an_inherited_dcp_in_the_common_env_is_refused_at_the_desk():
    """This is the bug the design's seam D describes: a re-exec into the
    Form-A group with SGLANG_UNEVEN_DCP=1 inherited lands in RankRoleError
    (rank_role.py:925-938) -- inside the GPU window. W116 is the same
    refusal, one layer earlier and free."""
    base = {"PYTHONPATH": "/x", "SGLANG_UNEVEN_DCP": "1"}
    with pytest.raises(Weg2FlipGroupEnvInherited) as exc:
        build_group_env(D_GROUP, base)
    msg = str(exc.value)
    assert "W116 Weg2FlipGroupEnvInherited" in msg
    assert "SGLANG_UNEVEN_DCP" in msg
    assert "rank_role.py:925-938" in msg
    assert "Uneven nie ab" in msg


def test_the_p_group_is_refused_the_same_inheritance():
    """Not an asymmetric rule: a value in the COMMON prefix is wrong for
    either group, even when it happens to match what P wants."""
    with pytest.raises(Weg2FlipGroupEnvInherited):
        build_group_env(P_GROUP, {"SGLANG_UNEVEN_DCP": "1"})


def test_extra_env_may_not_smuggle_a_group_owned_axis():
    with pytest.raises(Weg2FlipGroupEnvInherited) as exc:
        build_group_env(D_GROUP, {}, {"SGLANG_UNEVEN_DCP_WEIGHTED": "0"})
    assert "same bug one layer up" in str(exc.value)


def test_an_unknown_group_is_refused():
    with pytest.raises(Weg2FlipGroupEnvInherited):
        build_group_env("X", {})


# --------------------------------------------------------------------------
# Both groups from one line
# --------------------------------------------------------------------------
def test_one_launcher_line_yields_two_groups_with_their_own_values():
    plan = build_flip_groups(
        base_env={"PYTHONPATH": "/spinning/wt-fn-boot/python"},
        context_tokens=262144,
        max_running_requests=1,
        p_max_total_tokens=270000,
        d_max_total_tokens=270000,
        p_argv=("--tp-size", "1", "--pp-size", "3"),
        d_argv=("--tp-size", "3", "--pp-size", "1", "--rank-role", "host,worker,worker"),
    )
    assert plan.p_env.env["SGLANG_UNEVEN_DCP"] == "1"
    assert plan.d_env.env["SGLANG_UNEVEN_DCP"] == "0"
    assert plan.p_context.reachable_tokens >= 262144
    assert plan.d_context.reachable_tokens >= 262144
    assert "--pp-size" in plan.p_argv and "--rank-role" in plan.d_argv
    text = plan.report()
    assert "FLIP GROUPS @ 262144" in text


def test_the_flip_refuses_when_only_the_p_side_carries_the_40000_default():
    """The realistic failure: the D template was fixed (270000) and the P
    template still carries run_fn7s2.sh's default. One group short is the
    whole flip short."""
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        build_flip_groups(
            base_env={},
            p_max_total_tokens=40000,
            d_max_total_tokens=270000,
        )
    assert "group 'P'" in str(exc.value)
