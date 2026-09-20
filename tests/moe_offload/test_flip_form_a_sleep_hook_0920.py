# SPDX-License-Identifier: Apache-2.0
"""Slice 5a runtime half: the hook, against a stub saver.

Hermetic by construction -- the adapter, the free-memory probe, the flush and
the clock are all injected. The hazards being tested are ORDERING hazards,
which a stub can prove and a GPU cannot prove cheaply.
"""
from __future__ import annotations

import contextlib

import pytest

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.flip_form_a_sleep import ROLE_HOST, ROLE_WORKER, Weg2FlipFormAUntagged
from sglang.srt.flip_form_a_sleep_hook import FormASleepHook

_GIB = 1024**3

_HOST_ALLOCS = [
    "host_kv_pool",
    "mamba_gdn_state",
    "dense_weights",
    "solo_draft_weights",
    "role_graphs",
]
_WORKER_ALLOCS = [
    "expert_weights_resident",
    "expert_scratch_slots",
    "expert_lru_pool_rows",
    "role_graphs",
]


class StubSaver:
    """Records the call order; that order IS the assertion."""

    __module__ = "tests.stub"  # not sglang.* -> skips the adapter-shape check

    def __init__(self, bytes_by_tag=None, resume_fails=()):
        self.calls = []
        self.bytes_by_tag = bytes_by_tag or {}
        self.resume_fails = set(resume_fails)
        self.paused = set()

    @contextlib.contextmanager
    def region(self, tag, enable_cpu_backup=False):
        self.calls.append(("region", tag))
        yield

    def pause(self, tag):
        self.calls.append(("pause", tag))
        self.paused.add(tag)

    def resume(self, tag):
        self.calls.append(("resume", tag))
        if tag in self.resume_fails:
            raise RuntimeError(f"W119 Weg2TmsResumeRefused tag={tag}")
        self.paused.discard(tag)

    def tag_bytes(self, tag):
        return self.bytes_by_tag.get(tag)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.001
        return self.t


def _host_hook(**kw):
    saver = kw.pop("saver", None) or StubSaver(
        {GPU_MEMORY_TYPE_KV_CACHE: 4 * _GIB, GPU_MEMORY_TYPE_WEIGHTS: 13 * _GIB}
    )
    hook = FormASleepHook(saver, ROLE_HOST, 0, clock=FakeClock(), **kw)
    for name in _HOST_ALLOCS:
        with hook.region(name):
            pass
    return hook, saver


# --------------------------------------------------------------------------
# The one door
# --------------------------------------------------------------------------
def test_every_region_puts_the_allocation_under_its_tag():
    hook, saver = _host_hook()
    regions = [t for c, t in saver.calls if c == "region"]
    assert regions.count(GPU_MEMORY_TYPE_KV_CACHE) == 2  # kv pool + mamba state
    assert GPU_MEMORY_TYPE_WEIGHTS in regions
    assert GPU_MEMORY_TYPE_CUDA_GRAPH in regions
    assert hook.registered() == tuple(_HOST_ALLOCS)


def test_an_unnamed_allocation_cannot_slip_through_untagged():
    """The failure this module exists to close: an allocation with no tag is
    invisible to a call-site review, because nothing is written where nothing
    was done."""
    hook, _ = _host_hook()
    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        with hook.region("hc_mixer_int8_buffer"):
            pass
    msg = str(exc.value)
    assert "W118 Weg2FlipFormAUntagged" in msg
    assert "invisible to review" in msg


def test_a_host_region_on_a_worker_is_refused():
    saver = StubSaver()
    hook = FormASleepHook(saver, ROLE_WORKER, 1, clock=FakeClock())
    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        with hook.region("host_kv_pool"):
            pass
    assert "layout bug" in str(exc.value)


def test_an_unknown_role_fails_at_construction_not_at_the_first_flip():
    with pytest.raises(Weg2FlipFormAUntagged):
        FormASleepHook(StubSaver(), "stage", 0)


# --------------------------------------------------------------------------
# Hazard 1: flush BEFORE the kv pause
# --------------------------------------------------------------------------
def test_the_flush_runs_before_the_kv_pause_not_after():
    order = []
    saver = StubSaver({GPU_MEMORY_TYPE_KV_CACHE: 4 * _GIB})

    def flush():
        order.append("flush")
        return True

    hook, saver = _host_hook(saver=saver, flush_fn=flush)
    saver.calls.clear()
    hook.sleep()
    first_pause = next(i for i, (c, _) in enumerate(saver.calls) if c == "pause")
    assert order == ["flush"]
    assert saver.calls[first_pause] == ("pause", GPU_MEMORY_TYPE_KV_CACHE)


def test_a_refused_flush_stops_the_sleep_before_any_pause():
    saver = StubSaver({GPU_MEMORY_TYPE_KV_CACHE: 4 * _GIB})
    hook, saver = _host_hook(saver=saver, flush_fn=lambda: False)
    saver.calls.clear()
    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        hook.sleep()
    assert "flush_cache() refused" in str(exc.value)
    assert "memory_pool.py:1017" in str(exc.value)
    assert [c for c, _ in saver.calls if c == "pause"] == []
    assert hook.asleep is False


def test_a_worker_sleep_needs_no_flush_because_it_has_no_kv_tag():
    saver = StubSaver({GPU_MEMORY_TYPE_WEIGHTS: 13 * _GIB})
    hook = FormASleepHook(saver, ROLE_WORKER, 1, clock=FakeClock())
    for n in _WORKER_ALLOCS:
        with hook.region(n):
            pass
    rep = hook.sleep()
    assert GPU_MEMORY_TYPE_KV_CACHE not in [t for t, _, _ in rep.steps]


# --------------------------------------------------------------------------
# The orders
# --------------------------------------------------------------------------
def test_sleep_pauses_kv_first_and_graph_last():
    hook, saver = _host_hook(flush_fn=lambda: True)
    saver.calls.clear()
    hook.sleep()
    paused = [t for c, t in saver.calls if c == "pause"]
    assert paused[0] == GPU_MEMORY_TYPE_KV_CACHE
    assert paused[-1] == GPU_MEMORY_TYPE_CUDA_GRAPH


def test_wake_resumes_graph_first_and_kv_last():
    hook, saver = _host_hook(flush_fn=lambda: True, free_bytes_fn=lambda: 30 * _GIB)
    hook.sleep()
    saver.calls.clear()
    hook.wake()
    resumed = [t for c, t in saver.calls if c == "resume"]
    assert resumed[0] == GPU_MEMORY_TYPE_CUDA_GRAPH
    assert resumed[-1] == GPU_MEMORY_TYPE_KV_CACHE
    assert hook.asleep is False


def test_a_double_sleep_is_refused():
    hook, _ = _host_hook(flush_fn=lambda: True)
    hook.sleep()
    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        hook.sleep()
    assert "already" in str(exc.value)


def test_a_wake_without_a_sleep_is_refused():
    hook, _ = _host_hook()
    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        hook.wake()
    assert "not \nasleep" in str(exc.value) or "not asleep" in str(exc.value).replace(
        "\n", " "
    )


# --------------------------------------------------------------------------
# Hazard 2: ask the card BEFORE the void-ABI resume
# --------------------------------------------------------------------------
def test_a_card_that_cannot_fund_the_kv_refuses_before_resuming():
    """The Form-A host maps its KV onto the card P released one RPC earlier.
    'The release has not landed yet' is the NORMAL case."""
    saver = StubSaver({GPU_MEMORY_TYPE_KV_CACHE: 4 * _GIB})
    hook, saver = _host_hook(
        saver=saver, flush_fn=lambda: True, free_bytes_fn=lambda: 1 * _GIB
    )
    hook.sleep()
    saver.calls.clear()
    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        hook.wake()
    msg = str(exc.value)
    assert "VOID" in msg
    assert "weg2xsn408" in msg
    # graph + weights resumed, kv did NOT
    assert ("resume", GPU_MEMORY_TYPE_KV_CACHE) not in saver.calls
    assert ("resume", GPU_MEMORY_TYPE_CUDA_GRAPH) in saver.calls


def test_the_corridor_floor_is_reported_and_never_subtracted():
    """Memory KEINE-KORRIDOR-RESERVE-NIE: a reserve may shape a plan, it may
    never be the reason a wake is refused. free == need must pass even with a
    large floor declared."""
    saver = StubSaver({GPU_MEMORY_TYPE_KV_CACHE: 4 * _GIB})
    hook, saver = _host_hook(
        saver=saver,
        flush_fn=lambda: True,
        free_bytes_fn=lambda: 4 * _GIB,
        corridor_floor_bytes=2 * _GIB,
    )
    hook.sleep()
    hook.wake()  # no raise: exact fit is not a refusal
    assert ("resume", GPU_MEMORY_TYPE_KV_CACHE) in saver.calls


def test_an_absent_free_probe_is_not_a_refusal():
    """An absent probe is an ABSENCE, never a False."""
    hook, saver = _host_hook(flush_fn=lambda: True, free_bytes_fn=None)
    hook.sleep()
    hook.wake()
    assert ("resume", GPU_MEMORY_TYPE_KV_CACHE) in saver.calls


def test_the_adapters_own_landing_refusal_propagates():
    """W119 from the adapter must not be swallowed by the hook."""
    saver = StubSaver(
        {GPU_MEMORY_TYPE_KV_CACHE: 4 * _GIB}, resume_fails=(GPU_MEMORY_TYPE_CUDA_GRAPH,)
    )
    hook, saver = _host_hook(
        saver=saver, flush_fn=lambda: True, free_bytes_fn=lambda: 30 * _GIB
    )
    hook.sleep()
    with pytest.raises(RuntimeError) as exc:
        hook.wake()
    assert "W119" in str(exc.value)


# --------------------------------------------------------------------------
# Hazard 3: the exclusion is the adapter's, and it must be there
# --------------------------------------------------------------------------
def test_the_real_adapter_still_carries_the_abort_poll_exclusion():
    from sglang.srt.utils import torch_memory_saver_adapter as tms

    assert hasattr(tms, "_abort_poll_excluded")


def test_a_downgraded_sglang_adapter_is_refused(monkeypatch):
    from sglang.srt.utils import torch_memory_saver_adapter as tms

    class SglangShapedSaver(StubSaver):
        __module__ = "sglang.srt.utils.torch_memory_saver_adapter"

    saver = SglangShapedSaver({GPU_MEMORY_TYPE_KV_CACHE: 1 * _GIB})
    hook = FormASleepHook(saver, ROLE_HOST, 0, clock=FakeClock())
    for n in _HOST_ALLOCS:
        with hook.region(n):
            pass
    monkeypatch.delattr(tms, "_abort_poll_excluded")
    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        hook.sleep()
    assert "_abort_poll_excluded" in str(exc.value)
    assert "Do not wrap it" in str(exc.value)


# --------------------------------------------------------------------------
# The absent deposit is a decision, not a gap
# --------------------------------------------------------------------------
def test_the_missing_deposit_step_is_stated():
    hook, _ = _host_hook()
    note = hook.deposit_note()
    assert "shared page-locked HOST pool" in note
    assert "lie still" in note


def test_the_report_line_carries_tag_bytes_and_ms():
    hook, _ = _host_hook(flush_fn=lambda: True)
    rep = hook.sleep()
    line = rep.line()
    assert line.startswith("FORM-A-SLEEP role=host rank=0")
    assert "kv_cache=" in line and "MiB/" in line
    assert rep.total_bytes > 0


# --------------------------------------------------------------------------
# The Sleep/Wake RPC for the Form-A group
# --------------------------------------------------------------------------
def test_a_tagless_request_resolves_per_role_not_to_all_types():
    """One group-wide request, two different populations."""
    from sglang.srt.flip_form_a_sleep_hook import resolve_rpc_tags

    host = resolve_rpc_tags(ROLE_HOST, None)
    worker = resolve_rpc_tags(ROLE_WORKER, None)
    assert GPU_MEMORY_TYPE_KV_CACHE in host
    assert GPU_MEMORY_TYPE_KV_CACHE not in worker
    assert set(worker) == {GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_CUDA_GRAPH}


def test_a_kv_request_to_a_worker_is_refused_not_filtered():
    """Filtering is the silent form: host pauses three, workers two, both
    answer OK, and the front reads one verdict for two different acts."""
    from sglang.srt.flip_form_a_sleep_hook import resolve_rpc_tags

    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        resolve_rpc_tags(ROLE_WORKER, [GPU_MEMORY_TYPE_KV_CACHE])
    msg = str(exc.value)
    assert "does not own" in msg
    assert "TWO different acts" in msg


def test_a_request_naming_only_owned_tags_passes_through():
    from sglang.srt.flip_form_a_sleep_hook import resolve_rpc_tags

    assert resolve_rpc_tags(ROLE_WORKER, [GPU_MEMORY_TYPE_WEIGHTS]) == [
        GPU_MEMORY_TYPE_WEIGHTS
    ]


def test_the_group_fence_accepts_the_measured_form_a_shape():
    from sglang.srt.flip_form_a_sleep_hook import assert_group_agrees

    assert_group_agrees(
        [
            (0, ROLE_HOST, [GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS,
                            GPU_MEMORY_TYPE_CUDA_GRAPH]),
            (1, ROLE_WORKER, [GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_CUDA_GRAPH]),
            (2, ROLE_WORKER, [GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_CUDA_GRAPH]),
        ]
    )


def test_two_hosts_in_the_gather_is_a_disagreement():
    from sglang.srt.flip_form_a_sleep_hook import assert_group_agrees

    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        assert_group_agrees([(0, ROLE_HOST, []), (1, ROLE_HOST, [])])
    assert "2 attention host(s)" in str(exc.value)


def test_a_worker_that_slept_kv_is_a_disagreement():
    """Memory RAENGE-NIE-UNEINS: disagreement is CRASH/STOP, never a
    per-rank best effort."""
    from sglang.srt.flip_form_a_sleep_hook import assert_group_agrees

    with pytest.raises(Weg2FlipFormAUntagged) as exc:
        assert_group_agrees(
            [
                (0, ROLE_HOST, [GPU_MEMORY_TYPE_KV_CACHE]),
                (1, ROLE_WORKER, [GPU_MEMORY_TYPE_KV_CACHE]),
            ]
        )
    msg = str(exc.value)
    assert "rank 1" in msg
    assert "RAENGE-NIE-UNEINS" in msg


def test_an_empty_gather_is_not_agreement():
    from sglang.srt.flip_form_a_sleep_hook import assert_group_agrees

    with pytest.raises(Weg2FlipFormAUntagged):
        assert_group_agrees([])
