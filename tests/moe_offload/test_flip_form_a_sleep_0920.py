# SPDX-License-Identifier: Apache-2.0
"""Slice 5a: the sleep/wake tag table Form A does not have yet (risk R7).

Hermetic: no torch allocation, no device, no memory saver. The only imports
that touch the runtime are the tag CONSTANTS, which is the point -- the table
must name the same strings the runtime pauses.
"""
from __future__ import annotations

import pytest

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
    GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
)
from sglang.srt.flip_form_a_sleep import (
    FORM_A_ALLOCATIONS,
    ROLE_HOST,
    ROLE_WORKER,
    Weg2FlipFormAUntagged,
    plan_form_a_sleep,
    sleep_tag_order,
    tags_for_role,
    wake_tag_order,
)

_HOST_ALLOCS = [
    "host_kv_pool",
    "mamba_gdn_state",
    "dense_weights",
    "solo_draft_weights",
    "role_graphs",
]
_WORKER_ALLOCS = ["expert_weights_resident", "role_graphs"]


# --------------------------------------------------------------------------
# R7: the gap this closes actually exists
# --------------------------------------------------------------------------
def test_form_a_still_has_no_runtime_sleep_hook():
    """The design's risk R7, kept honest as a test rather than as prose: if
    a pause/resume ever appears in the Form-A modules, this test fails and
    somebody must decide whether THIS table is still the authority."""
    import inspect
    import pathlib

    import sglang.srt.form_a_plan as fa

    root = pathlib.Path(inspect.getfile(fa)).parent
    hits = []
    for name in (
        "form_a_boot_gate.py",
        "form_a_construction.py",
        "form_a_plan.py",
        "form_a_symmetry.py",
        "form_a_worker_forward.py",
        "rank_role.py",
    ):
        text = (root / name).read_text()
        for needle in ("memory_saver", "def pause", "def resume"):
            if needle in text:
                hits.append(f"{name}:{needle}")
    assert hits == [], f"Form A grew a sleep hook: {hits}"


# --------------------------------------------------------------------------
# The role asymmetry
# --------------------------------------------------------------------------
def test_the_host_wears_all_three_device_tags():
    assert tags_for_role(ROLE_HOST) == set(GPU_MEMORY_ALL_TYPES)


def test_a_worker_wears_no_kv_tag():
    """fnFA19: TP1/TP2 hold a 768 B stub cell and 0.00 GB of mamba state.
    Pausing an empty region grades a sleep against a denominator that is not
    there (SleepAcceptanceCensus)."""
    tags = tags_for_role(ROLE_WORKER)
    assert GPU_MEMORY_TYPE_KV_CACHE not in tags
    assert tags == {GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_CUDA_GRAPH}


def test_the_draft_tag_is_in_neither_role_because_it_stays_resident():
    """constants.py:16-21 -- weights_draft is outside GPU_MEMORY_ALL_TYPES on
    purpose: 'Adding this tag there would pause the drafter -- the exact
    opposite of the purpose'."""
    assert GPU_MEMORY_TYPE_WEIGHTS_DRAFT not in tags_for_role(ROLE_HOST)
    assert GPU_MEMORY_TYPE_WEIGHTS_DRAFT not in tags_for_role(ROLE_WORKER)
    assert GPU_MEMORY_TYPE_WEIGHTS_DRAFT not in GPU_MEMORY_ALL_TYPES


def test_an_unknown_role_is_refused():
    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        tags_for_role("stage")
    assert "not a Form-A role" in str(exc.value)


# --------------------------------------------------------------------------
# The orders: mirrors of the 27B path, not reverses of each other by accident
# --------------------------------------------------------------------------
def test_sleep_pauses_kv_first_and_the_weights_base_tag_closes_the_family():
    order = sleep_tag_order(ROLE_HOST, ["weights_chunk_0", "weights_chunk_1"])
    assert order[0] == GPU_MEMORY_TYPE_KV_CACHE
    # chunks first, base tag last of the weights family
    wi = order.index(GPU_MEMORY_TYPE_WEIGHTS)
    assert order.index("weights_chunk_0") < wi
    assert order.index("weights_chunk_1") < wi
    assert order[-1] == GPU_MEMORY_TYPE_CUDA_GRAPH


def test_wake_resumes_the_graph_tag_first_and_kv_last():
    """weight_updater.resume_memory_occupation: graph :6408, weights :6503,
    KV :6893."""
    order = wake_tag_order(ROLE_HOST, ["weights_chunk_0"])
    assert order[0] == GPU_MEMORY_TYPE_CUDA_GRAPH
    assert order[-1] == GPU_MEMORY_TYPE_KV_CACHE


def test_wake_is_not_the_reverse_of_sleep():
    """Both put the weights chunks BEFORE the base tag; a naive reverse would
    put the base tag first and make derive_waves read the wrong tail."""
    chunks = ["weights_chunk_0", "weights_chunk_1"]
    s = sleep_tag_order(ROLE_HOST, chunks)
    w = wake_tag_order(ROLE_HOST, chunks)
    assert w != list(reversed(s))
    for order in (s, w):
        assert order.index("weights_chunk_0") < order.index(GPU_MEMORY_TYPE_WEIGHTS)


def test_a_worker_order_has_no_kv_step_at_either_end():
    for order in (sleep_tag_order(ROLE_WORKER), wake_tag_order(ROLE_WORKER)):
        assert GPU_MEMORY_TYPE_KV_CACHE not in order
        assert GPU_MEMORY_TYPE_WEIGHTS in order
        assert GPU_MEMORY_TYPE_CUDA_GRAPH in order


# --------------------------------------------------------------------------
# The table: an allocation without a tag is a refusal, not a survivor
# --------------------------------------------------------------------------
def test_the_measured_host_inventory_plans_cleanly():
    plan = plan_form_a_sleep(ROLE_HOST, _HOST_ALLOCS, ["weights_chunk_0"])
    assert plan.role == ROLE_HOST
    names = dict(plan.tagged)
    assert names["host_kv_pool"] == GPU_MEMORY_TYPE_KV_CACHE
    assert names["mamba_gdn_state"] == GPU_MEMORY_TYPE_KV_CACHE
    assert names["dense_weights"] == GPU_MEMORY_TYPE_WEIGHTS
    assert names["solo_draft_weights"] == GPU_MEMORY_TYPE_WEIGHTS_DRAFT
    # the draft is tagged but outside the pause population
    assert plan.resident == ("solo_draft_weights",)
    assert "FORM-A SLEEP HOOK role=host" in plan.report()


def test_the_measured_worker_inventory_plans_cleanly():
    plan = plan_form_a_sleep(ROLE_WORKER, _WORKER_ALLOCS)
    assert dict(plan.tagged)["expert_weights_resident"] == GPU_MEMORY_TYPE_WEIGHTS
    assert plan.resident == ()


def test_an_untagged_allocation_is_refused_with_the_tight_card_named():
    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        plan_form_a_sleep(ROLE_HOST, _HOST_ALLOCS + ["hc_mixer_int8_buffer"])
    msg = str(exc.value)
    assert "W118 Weg2FlipFormAUntagged" in msg
    assert "hc_mixer_int8_buffer" in msg
    assert "0.08 GiB" in msg
    assert "WAKING group" in msg


def test_the_refusal_names_the_one_legitimate_untagged_case():
    """The shared host expert pool wears no device tag on purpose --
    torch_memory_saver manages DEVICE memory only."""
    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        plan_form_a_sleep(ROLE_WORKER, ["spilled_expert_host_pool"])
    msg = str(exc.value)
    assert "DEVICE memory only" in msg
    assert "constants.py:2-4" in msg


def test_a_host_allocation_on_a_worker_is_a_layout_bug_not_a_tagging_one():
    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        plan_form_a_sleep(ROLE_WORKER, ["host_kv_pool", "role_graphs"])
    assert "layout bug" in str(exc.value)


def test_every_table_entry_carries_its_evidence():
    """A table whose rows cannot be checked is a table nobody will check."""
    for alloc in FORM_A_ALLOCATIONS:
        assert alloc.why, alloc.name
        assert alloc.roles <= {ROLE_HOST, ROLE_WORKER}
        assert alloc.tag in set(GPU_MEMORY_ALL_TYPES) | {GPU_MEMORY_TYPE_WEIGHTS_DRAFT}


# --------------------------------------------------------------------------
# The barlink rule (today's fix): pause/resume runs under pause_polling
# --------------------------------------------------------------------------
def test_the_plan_declares_that_it_runs_under_pause_polling():
    """The watchdog READS device memory from ANOTHER thread, and between
    pause and resume those pages are unmapped -- the same hazard a CUDA-graph
    capture has (parallel_state.py:3404)."""
    plan = plan_form_a_sleep(ROLE_HOST, _HOST_ALLOCS)
    assert plan.requires_pause_polling is True
    assert "pause_polling" in plan.report()


def test_pause_polling_exists_and_is_the_thing_being_referred_to():
    from sglang.srt.distributed.device_communicators import barlink_abort_gate

    assert callable(barlink_abort_gate.pause_polling)
