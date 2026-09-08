"""Item `dormant`: the sleeping group's residual VRAM, and the gate on the two
mechanisms that release part of it (record section [1y] of
/spinning/gpu-arb/weg2/WEG2_BUILD_DECISIONS_0906.md).

The measured problem, boot weg2rg6: while D serves, the three SLEEPING P ranks
still hold **1820 / 1368 / 1422 MiB** (NVML per-process bytes on the
``[weg2 sleep-acceptance]`` line), and the 5090 sat at **341 MiB** free -- below
the 819-1229 MiB corridor band -- for ten minutes.  [1y] attributes that image;
two of its rows are releasable with mechanisms that already exist:

* the CUDA-graph **capture pool** (92 / 102 / 133 MiB), which upstream already
  routes through ``memory_saver_adapter.cuda_graph(tag=cuda_graph)`` and which
  nothing ever asked to be paused, and
* the flashinfer **FLOAT workspace** (384 MiB/rank), whose content contract is
  ZERO by ``zero_flashinfer_workspaces`` and #50's own bisection.

What these tests pin is the WIRING and its refusals, because that is where this
class of change fails: an asymmetric tag (paused on one leg, not removed on the
other) is a ``KeyError`` that kills the group, an untagged allocation that still
prints "TAGGED" is a lie in the log, and a size gate below #102's
``MIN_TAGGED_BYTES`` is the illegal-memory-access class that #102 measured live.

Hermetic: no CUDA, no server, no boot.  The one test that needs a device
(captured-graph address stability across pause/resume) skips without one and is
carried on metal by ``arm_dormant_boot.sh`` instead.
"""

import inspect
import os
import re

import pytest

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)


def _saver(armed: bool):
    """Import ``weg2_memory_saver`` with the graph tag armed or disarmed.

    ``weg2_graph_tag_armed`` caches on purpose (the sleep and the wake must not
    be able to disagree), so a test that wants the other answer resets the
    cache explicitly rather than relying on import order.
    """
    from sglang.srt.managers import weg2_memory_saver as ms

    os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = "1" if armed else "0"
    ms._GRAPH_TAG_ARMED = None
    return ms


def _updater():
    from sglang.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager,
    )

    return SchedulerWeightUpdaterManager


# ---------------------------------------------------------------------------
# The coupling: which RPCs carry the graph tag
# ---------------------------------------------------------------------------


def test_graph_tag_rides_the_kv_carrier_and_nothing_else():
    _saver(armed=True)
    add = _updater()._weg2_with_graph_tag

    assert add([GPU_MEMORY_TYPE_KV_CACHE], True) == [
        GPU_MEMORY_TYPE_KV_CACHE,
        GPU_MEMORY_TYPE_CUDA_GRAPH,
    ]
    # The weights family legs must stay untouched: they are gathered with the
    # OTHER group's wake and carry a cpu backup, which the graph tag has not.
    assert add([GPU_MEMORY_TYPE_WEIGHTS, "weights_0"], True) == [
        GPU_MEMORY_TYPE_WEIGHTS,
        "weights_0",
    ]
    # Already present (tags=None -> GPU_MEMORY_ALL_TYPES): not duplicated.
    assert add([GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH], True) == [
        GPU_MEMORY_TYPE_KV_CACHE,
        GPU_MEMORY_TYPE_CUDA_GRAPH,
    ]
    # The stock (non-Weg-2) path is byte-identical to what it always was.
    assert add([GPU_MEMORY_TYPE_KV_CACHE], False) == [GPU_MEMORY_TYPE_KV_CACHE]


def test_graph_tag_absent_when_the_capture_did_not_route_into_it():
    """Disarmed means the CAPTURE did not use the tag either, so pausing it
    would release a workspace whose graph is not released with it."""
    _saver(armed=False)
    add = _updater()._weg2_with_graph_tag
    assert add([GPU_MEMORY_TYPE_KV_CACHE], True) == [GPU_MEMORY_TYPE_KV_CACHE]


def test_sleep_and_wake_are_symmetric_over_the_fronts_real_sequence():
    """THE group-fatal fault this guards: the wake does
    ``offload_tags.remove(tag)``, which raises on a tag the sleep never added.

    The sequence is the front's own (front.py:901 then the gathered family;
    the wake mirrors it), so the set must come back empty.
    """
    _saver(armed=True)
    add = _updater()._weg2_with_graph_tag

    offload = set()
    for tags in ([GPU_MEMORY_TYPE_KV_CACHE], [GPU_MEMORY_TYPE_WEIGHTS, "weights_0"]):
        offload.update(add(tags, True))
    assert GPU_MEMORY_TYPE_CUDA_GRAPH in offload

    for tags in ([GPU_MEMORY_TYPE_WEIGHTS, "weights_0"], [GPU_MEMORY_TYPE_KV_CACHE]):
        for tag in add(tags, True):
            offload.remove(tag)  # KeyError here IS the failure
    assert offload == set()


def test_both_handlers_call_the_coupling_and_the_wake_calls_it_before_removing():
    """The symmetry above is a property of the HELPER; this is the property of
    the WIRING, and it is the half that actually kills the group.  If only the
    sleep leg adds the tag, the wake's ``offload_tags.remove`` raises KeyError
    -- and if the wake calls the coupling AFTER the remove loop, it removes a
    tag it has not added yet, which is the same fault one line later.
    """
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    release = inspect.getsource(wu.SchedulerWeightUpdaterManager.release_memory_occupation)
    resume = inspect.getsource(wu.SchedulerWeightUpdaterManager.resume_memory_occupation)

    assert "_weg2_with_graph_tag" in release, "the sleep leg does not add the tag"
    assert "_weg2_with_graph_tag" in resume, (
        "the wake leg does not add the tag -- offload_tags.remove will raise "
        "KeyError on the tag the sleep paused"
    )
    assert resume.index("_weg2_with_graph_tag") < resume.index(
        "self.offload_tags.remove(tag)"
    ), "the wake couples the tag AFTER the remove loop, which is too late"


# ---------------------------------------------------------------------------
# The region and its four refusals
# ---------------------------------------------------------------------------


def test_size_gate_is_102s_number_and_its_reason_is_carried():
    """#102 measured the fault this gate closes (a ~1.5 MiB allocation served
    from a paused tag's segment tail -> illegal access on first touch).  The
    constant must be #102's, not a second one invented here."""
    from sglang.srt.speculative.adaptive_graph_memory import MIN_TAGGED_BYTES

    ms = _saver(armed=True)
    assert ms.WEG2_GRAPH_SCRATCH_MIN_BYTES == MIN_TAGGED_BYTES == 2 * 1024 * 1024


def test_below_min_bytes_refuses_at_the_gate_not_downstream(caplog):
    """The size gate must decide BEFORE the pool is touched.

    Checking only ``tagged is False`` cannot tell the gate from a failed pool
    entry -- on a CPU box both are False -- so the discriminator is the log:
    the gate is a silent decision, a failed entry is a loud degrade.  Remove
    the gate and this goes red on a box with no CUDA, which is the box the
    change is written on.
    """
    from sglang.srt.speculative.adaptive_graph_memory import MIN_TAGGED_BYTES

    ms = _saver(armed=True)
    with caplog.at_level("WARNING"):
        with ms.weg2_graph_scratch_region(MIN_TAGGED_BYTES - 1) as tagged:
            assert tagged is False
    assert "NOT tagged" not in caplog.text, (
        "a sub-MIN_TAGGED_BYTES allocation reached the pool -- the #102 size "
        "gate is gone and a later small allocation can land in a paused tag's "
        "segment tail"
    )


def test_region_is_a_no_op_when_disarmed():
    ms = _saver(armed=False)
    with ms.weg2_graph_scratch_region(384 * 1024 * 1024) as tagged:
        assert tagged is False


def test_region_degrades_loudly_and_says_the_bytes_stay_resident(caplog):
    """A failed pool/region entry must NOT raise -- that would turn a VRAM
    optimisation into a boot killer at attention-backend build time -- but it
    must also not be silent, which is the warn-then-continue defect class."""
    ms = _saver(armed=True)

    class _Boom:
        def region_config(self, **_kw):
            raise RuntimeError("saver not initialised")

    with caplog.at_level("WARNING"):
        with ms.weg2_graph_scratch_region(384 * 1024 * 1024, adapter=_Boom()) as tagged:
            assert tagged is False
    text = caplog.text
    assert "NOT tagged" in text
    assert "RESIDENT across the sleep" in text or "default pool" in text


def test_callers_own_exception_is_not_swallowed_by_the_degrade_path():
    ms = _saver(armed=True)
    with pytest.raises(ValueError, match="caller fault"):
        with ms.weg2_graph_scratch_region(384 * 1024 * 1024):
            raise ValueError("caller fault")


# ---------------------------------------------------------------------------
# Instrument-text law (Klasse A): a line must say what it measures
# ---------------------------------------------------------------------------


def test_allocator_cache_line_names_its_instrument_and_prints_n_a_not_zero(caplog):
    M = _updater()
    probe = M.__new__(M)
    mib = 1024 * 1024

    with caplog.at_level("INFO"):
        M._weg2_log_allocator_cache_released(probe, None, None)
    assert "allocator_cache_released_mib=n/a" in caplog.text
    # A reading that was not taken must never render as an empty cache.
    assert "allocator_cache_released_mib=0.0" not in caplog.text

    caplog.clear()
    with caplog.at_level("INFO"):
        M._weg2_log_allocator_cache_released(probe, (900 * mib, 400 * mib), (512 * mib, 400 * mib))
    line = caplog.text
    assert "allocator_cache_released_mib=388.0" in line
    assert "memory_reserved" in line          # the instrument
    assert "UNTAGGED" in line                 # the population
    assert "NOT NVML" in line                 # what it must not be added to


@pytest.mark.parametrize(
    "marker, must_carry",
    [
        ("WEG2-SLEEP released tags=", ("tms_tag_bytes", "population", "not an empty tag")),
        ("WEG2-RESUME remapped mib=", ("AFTER the resume", "cpu backup", "n/a")),
    ],
)
def test_graph_tag_lines_carry_their_denominator(marker, must_carry):
    """Both new flip-path lines report a NUMBER; each must state the instrument
    that produced it and the population it covers, or a reader will add it to
    ``proc_used``, which is a different instrument over a different population.
    """
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu)
    assert marker in src, f"{marker} is not emitted anywhere"
    start = src.index(marker)
    window = src[start : start + 1400]
    for phrase in must_carry:
        assert phrase in window, f"{marker} does not state {phrase!r}"


# ---------------------------------------------------------------------------
# The two wiring sites
# ---------------------------------------------------------------------------


def test_float_workspace_is_wrapped_and_the_int_workspace_is_not():
    """The FLOAT workspace has the zero contract (#50 bisection) and may be
    remapped; the INT workspace carries the plan the `full` backend captured
    INSIDE the graph and may not."""
    from sglang.srt.layers.attention import flashinfer_backend as fb

    src = inspect.getsource(fb)
    m = re.search(
        r"with weg2_graph_scratch_region\((.{0,4000}?)\n        weg2_tagged =",
        src,
        re.S,
    )
    assert m, "the float workspace allocation is not inside weg2_graph_scratch_region"
    block = m.group(1)
    assert 'get_buffer("flashinfer_workspace"' in block
    assert "_int_workspace" not in block, (
        "an int-workspace allocation moved inside the tagged region -- it holds "
        "the captured plan and a remap would destroy it"
    )


def test_tagged_line_is_not_printed_when_the_buffer_cache_already_had_it():
    """``get_buffer`` is a cache: on a second backend instance the factory does
    not run and nothing lands in the region.  Announcing TAGGED then would be a
    log that lies about what it did."""
    from sglang.srt.layers.attention import flashinfer_backend as fb

    src = inspect.getsource(fb)
    assert "weg2_tagged = weg2_region_open and bool(weg2_alloc_ran)" in src, (
        "the TAGGED line is gated on the region alone, not on the allocation "
        "having actually run inside it"
    )


def test_wake_restores_the_zero_contract_after_resuming_the_graph_tag():
    from sglang.srt.managers.scheduler_components import weight_updater as wu

    src = inspect.getsource(wu)
    resume_at = src.index("self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)")
    window = src[resume_at : resume_at + 2000]
    assert "_weg2_zero_graph_scratch()" in window, (
        "resume(cuda_graph) maps FRESH pages with no cpu backup; without "
        "zero_flashinfer_workspaces the first forward after a wake reads the "
        "other group's residue where NOTE(#50) promises zeros"
    )


def test_launcher_arms_the_graph_tag_and_lets_an_operator_override_it():
    from sglang.srt.weg2.launcher import build_env

    saved = os.environ.pop("SGLANG_MEMORY_SAVER_CUDA_GRAPH", None)
    try:
        env = build_env(tree="/tmp/t", venv="/tmp/v", cvd="0", store_dir="/tmp/s",
                        debug_hold=False, tag="probe")
        assert env["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] == "1"

        os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = "0"
        env = build_env(tree="/tmp/t", venv="/tmp/v", cvd="0", store_dir="/tmp/s",
                        debug_hold=False, tag="probe")
        assert env["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] == "0"
    finally:
        os.environ.pop("SGLANG_MEMORY_SAVER_CUDA_GRAPH", None)
        if saved is not None:
            os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = saved


# ---------------------------------------------------------------------------
# The one claim that needs a device
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("WEG2_DORMANT_GPU_GATE") != "1",
    reason=(
        "needs a real CUDA device, a preload-mode torch_memory_saver hook and a "
        "gpuq window; run it from arm_dormant_boot.sh with WEG2_DORMANT_GPU_GATE=1"
    ),
)
def test_captured_graph_static_buffers_keep_their_addresses_across_pause_resume():
    """RESTORE-NEVER-REBUILD, checked on the thing that would break first.

    A graph's replay reads its buffers by ADDRESS.  torch_memory_saver's whole
    claim is that a pause unmaps PHYSICAL pages while the virtual addresses
    stay, so the graph stays replayable and no recapture is needed.  If that is
    false on this driver, the capture pool must not ride the tag -- and the
    boot would find out as an illegal memory access in the first decode after a
    wake, which is a far worse instrument than this.
    """
    import torch

    from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

    from sglang.srt.managers import weg2_memory_saver as ms

    adapter = TorchMemorySaverAdapter.create(enable=True)
    static_in = torch.zeros(1024, device="cuda")
    static_out = torch.zeros(1024, device="cuda")
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with adapter.cuda_graph(cuda_graph=graph, tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
                            stream=stream):
        static_out.copy_(static_in * 2 + 1)

    static_in.fill_(3.0)
    graph.replay()
    torch.cuda.synchronize()
    before_ptr = (static_in.data_ptr(), static_out.data_ptr())
    before_val = static_out.clone()

    adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)
    adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)
    ms.weg2_graph_tag_armed()  # the resolver must not raise on this path

    assert (static_in.data_ptr(), static_out.data_ptr()) == before_ptr, (
        "the pause/resume MOVED a static buffer -- the captured graph now reads "
        "somebody else's memory; the capture pool must not ride this tag"
    )
    static_in.fill_(3.0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, before_val), (
        "replay after pause/resume produced a different result"
    )
