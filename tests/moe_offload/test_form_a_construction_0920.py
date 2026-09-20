# SPDX-License-Identifier: Apache-2.0
"""Hermetic falsifier for the Form A construction skip (slice 4a).

No CUDA, no checkpoint, no model. The decision is a pure predicate and the
placeholder is thirty lines of nn.Module, so the ACCEPTANCE CRITERION of the
slice -- "a worker's [vram-census] shows 'experts' and nothing else" -- is
checkable here rather than only in a boot log.

Why the slice exists at all, in one measurement: slice 5's loader veto stops
a worker READING the dense tensors and saves no VRAM, because create_weights
allocated them at construction. fn8ah measured 1.84 GiB of non-expert model
tensors per 3080. This is the half that gets them back.
"""

import pytest
import torch.nn as nn

from sglang.srt.form_a_construction import (
    HOST_ONLY_KINDS,
    FormAHostOnlyModuleUsed,
    HostOnlyModule,
    MODULE_KINDS,
    WORKER_KINDS,
    expected_census_categories,
    skip_on_worker,
    worker_builds,
)
from sglang.srt.rank_role import (
    RankRoleError,
    RankRolePlan,
    set_form_a_role_plan,
)

FORM_A = RankRolePlan(("host", "worker", "worker"))


@pytest.fixture(autouse=True)
def _no_plan_leaks():
    """The role plan is a PROCESS global; a test that installs one and does
    not remove it changes every later test in the session."""
    try:
        yield
    finally:
        set_form_a_role_plan(None)


# ==========================================================================
# 1. The decision
# ==========================================================================
def test_a_worker_builds_exactly_one_kind_of_module():
    assert WORKER_KINDS == ("experts",)
    assert worker_builds("experts") is True
    for kind in HOST_ONLY_KINDS:
        assert worker_builds(kind) is False, kind


def test_the_host_only_list_names_every_post_the_census_measured():
    """fn8ah's census categories, each one accounted for. A post that is in
    the census but not in this list is a post nobody decided about."""
    measured = {
        "hyper_connection",
        "linear_attn",
        "embed_tokens",
        "lm_head",
        "moe_gate",
        "shared_expert",
        "ple",
        "norm",
    }
    assert measured <= set(HOST_ONLY_KINDS)
    # ... and the two that are not census keys but are real modules
    assert {"self_attn", "o_proj", "draft", "vision"} <= set(HOST_ONLY_KINDS)


def test_an_unknown_kind_is_refused_rather_than_defaulted():
    """A typo that silently answered 'host-only' would delete a module from
    the host; one that silently answered 'build' would put it back on a
    worker. Neither announces itself, so neither is allowed to happen."""
    with pytest.raises(RankRoleError, match="unknown module kind"):
        worker_builds("attention")  # the real kind is 'self_attn'
    with pytest.raises(RankRoleError, match="unknown module kind"):
        worker_builds("")


# ==========================================================================
# 2. The acceptance criterion, as code
# ==========================================================================
def test_a_worker_census_may_show_experts_and_nothing_else():
    assert expected_census_categories("worker") == ("experts",)
    host = expected_census_categories("host")
    assert "experts" in host and "hyper_connection" in host
    assert len(host) == len(MODULE_KINDS)
    with pytest.raises(RankRoleError, match="unknown role"):
        expected_census_categories("driver")


def test_the_census_contract_would_fail_on_todays_boot():
    """The pin that makes the criterion meaningful: fn8ah's worker line
    carried seven categories beyond 'experts'. If this ever passes, the
    criterion has been weakened rather than met."""
    fn8ah_worker = {
        "experts",
        "hyper_connection",
        "linear_attn",
        "embed_tokens",
        "lm_head",
        "moe_gate",
        "shared_expert",
        "ple",
    }
    assert not fn8ah_worker <= set(expected_census_categories("worker"))


# ==========================================================================
# 3. The placeholder
# ==========================================================================
def test_the_placeholder_holds_no_parameters_which_is_the_whole_point():
    ph = HostOnlyModule("hyper_connection", "layers.0.attn")
    assert isinstance(ph, nn.Module)
    assert list(ph.parameters()) == []
    assert list(ph.buffers()) == []
    assert sum(p.numel() for p in ph.parameters()) == 0


def test_using_the_placeholder_refuses_by_name_and_names_the_next_slice():
    """A worker still runs the same forward today, so it WILL reach a dense
    module it does not have. With None that is an AttributeError naming
    nothing; this names the module, the role and the slice that has to
    land."""
    ph = HostOnlyModule("linear_attn", "layers.3.linear_attn")
    with pytest.raises(FormAHostOnlyModuleUsed) as e:
        ph(1, 2)
    msg = str(e.value)
    assert "linear_attn" in msg and "layers.3" in msg
    assert "host-centric decode path" in msg  # says what comes next
    with pytest.raises(FormAHostOnlyModuleUsed, match="attribute 'weight'"):
        ph.weight


# ==========================================================================
# 4. skip_on_worker -- inert unless Form A, and only on a worker
# ==========================================================================
def test_skip_is_inert_without_a_role_plan():
    set_form_a_role_plan(None)
    for kind in MODULE_KINDS:
        assert skip_on_worker(kind) is None, kind


def test_the_host_builds_everything_even_under_form_a():
    set_form_a_role_plan(FORM_A, 0)
    for kind in MODULE_KINDS:
        assert skip_on_worker(kind) is None, kind


def test_a_worker_skips_every_host_only_kind_and_builds_its_experts():
    set_form_a_role_plan(FORM_A, 2)
    assert skip_on_worker("experts") is None
    for kind in HOST_ONLY_KINDS:
        ph = skip_on_worker(kind, f"layers.0.{kind}")
        assert isinstance(ph, HostOnlyModule), kind
        assert ph.kind == kind
        assert list(ph.parameters()) == []


def test_the_model_wires_the_skip_at_the_posts_that_cost_the_most():
    """The two hyper-connection mixers are not sharded at all
    (layers/hyperconnection.py builds plain nn.Linear), so every rank holds
    them in full -- 0.63 GiB per rank in INT8. They and the PLE are the
    sites wired in this slice, and the wiring is checked in the source so a
    later refactor that drops it fails here."""
    import inspect

    from sglang.srt.models import qwen4_exp

    src = inspect.getsource(qwen4_exp)
    assert src.count("skip_on_worker(") >= 3
    assert 'skip_on_worker("ple"' in src
    assert 'skip_on_worker("hyper_connection"' in src
