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

import contextlib
import inspect
import os
import re

import pytest

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)


def _saver(armed: bool, group: str = "P", raw: str = None):
    """Import ``weg2_memory_saver`` with the graph tag armed or disarmed.

    Both cached resolvers are reset explicitly rather than relying on import
    order: they cache ON PURPOSE (the sleep and the wake must not be able to
    disagree), so a test that wants the other answer has to say so.

    ``group`` is the FIX 2 discriminator -- ``""`` means "this engine is not a
    Weg-2 group at all", which is the stock path.  ``raw`` overrides the env
    string when the test is about the PARSER rather than the decision.
    """
    from sglang.srt.managers import weg2_memory_saver as ms

    os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = (
        raw if raw is not None else ("1" if armed else "0")
    )
    if group:
        os.environ[ms.WEG2_GROUP_ENV] = group
    else:
        os.environ.pop(ms.WEG2_GROUP_ENV, None)
    ms._GRAPH_TAG_ARMED = None
    ms._WEG2_GROUP_NAME = None
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
    # An engine WITHOUT the memory saver: nothing to pause, nothing added.
    # This is NOT the whole stock path -- see
    # test_a_stock_memory_saver_engine_is_not_widened_by_this_item, which is
    # the case FIX 2 finding 1 was about and which this assertion cannot see.
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
# FIX 2: the three conjuncts of the arming gate.  Each of these went red on
# 3d507dfab8 -- the gate was ONE env read there, and neither the Weg-2 term
# nor the memory-saver term existed.
# ---------------------------------------------------------------------------


def test_a_stock_memory_saver_engine_is_not_widened_by_this_item():
    """THE finding: `--enable-memory-saver` is an UPSTREAM flag on an UPSTREAM
    endpoint, so it cannot be the gate for a Weg-2-only change of that
    endpoint's meaning.

    The configuration below is documented upstream, not exotic: the memory
    saver on and `SGLANG_MEMORY_SAVER_CUDA_GRAPH` set (environ.py registers it;
    full_cuda_graph_backend.py:78-81 reads it).  A caller that posts
    `{"tags":["kv_cache"]}` on such an engine asked for ONE tag.  Before this
    fix it also got `cuda_graph` paused -- the capture pool plus the 384 MiB
    flashinfer FLOAT workspace -- and a `zero_flashinfer_workspaces()` memset
    on the paired resume.  release_memory_occupation's own rule says a stock
    call "must stay byte-for-byte the upstream path".
    """
    _saver(armed=True, group="")          # memory saver ON, env ON, NOT Weg 2
    add = _updater()._weg2_with_graph_tag
    assert add([GPU_MEMORY_TYPE_KV_CACHE], True) == [GPU_MEMORY_TYPE_KV_CACHE], (
        "a stock --enable-memory-saver engine had its kv_cache RPC silently "
        "widened to kv_cache + cuda_graph"
    )
    # ...and the same engine's attention backend must not route its float
    # workspace into a region no sleep of that engine will ever pause.
    ms = _saver(armed=True, group="")
    with ms.weg2_graph_scratch_region(384 * 1024 * 1024, True) as tagged:
        assert tagged is False


def test_the_weg2_conjunct_is_the_group_env_and_the_launcher_publishes_it():
    """The discriminator has to EXIST, which is the half that was missing.

    `server_args.weg2_group` was believed to be it: it is read once
    (weight_updater._weg2_group_name) and assigned nowhere, so it answered "?"
    on every rank of every boot.  The env below is published by build_env for
    both groups and by nothing else.
    """
    from sglang.srt import server_args as sa
    from sglang.srt.managers import weg2_memory_saver as ms
    from sglang.srt.weg2 import launcher

    # Assigned nowhere in the tree -- the reason a new fact was needed at all.
    assert not hasattr(sa.ServerArgs, "weg2_group"), (
        "server_args grew a weg2_group field -- then IT is the discriminator "
        "and SGLANG_WEG2_GROUP is second bookkeeping; collapse them"
    )

    saved = os.environ.pop(ms.WEG2_GROUP_ENV, None)
    try:
        for group in ("P", "D"):
            env = launcher.build_env(tree="/tmp/t", venv="/tmp/v", cvd="0",
                                     store_dir="/tmp/s", debug_hold=False,
                                     tag="probe", group=group)
            assert env[ms.WEG2_GROUP_ENV] == group
        # An inherited value must be SCRUBBED, not carried: the same discipline
        # the ring family already follows, and here it decides whether a
        # coupling nobody asked for is armed.
        os.environ[ms.WEG2_GROUP_ENV] = "leftover-from-the-operators-shell"
        env = launcher.build_env(tree="/tmp/t", venv="/tmp/v", cvd="0",
                                 store_dir="/tmp/s", debug_hold=False, tag="probe")
        assert ms.WEG2_GROUP_ENV not in env
    finally:
        os.environ.pop(ms.WEG2_GROUP_ENV, None)
        if saved is not None:
            os.environ[ms.WEG2_GROUP_ENV] = saved


def test_every_launcher_call_site_names_its_group():
    """The helper being right is not the wiring being right (the commit-3 M1
    lesson).  build_env defaults `group=""` so a probe caller stays stock; that
    default is exactly what would silently disarm the item if a launch site
    forgot it, and the boot's only symptom would be a missing log line.
    """
    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher)
    calls = [ln for ln in src.splitlines() if "= build_env(" in ln]
    assert len(calls) == 3, f"launcher has {len(calls)} build_env call sites, expected 3"
    for ln in calls:
        assert 'group="' in ln, f"a launch site does not name its group: {ln.strip()}"


def test_the_sleep_gate_reads_the_capture_sites_own_reader():
    """The invariant is agreement with the CAPTURE, not with the registry.

    Three parsers exist for this one variable and they disagree; measured, one
    process per value:

        value    hand-rolled(v1)   get_bool_env_var   envs.EnvBool
        'yes'    True              False              True
        'on'     True              False              False
        'y'      False             False              True

    An armed sleep tag whose capture did NOT route into that tag releases a
    workspace whose graph is not released with it -- what
    flashinfer_backend.py:1060-1061 means by "released together or not at all".
    The launcher honours an operator override of this variable
    (launcher.py:1383-1385), so a non-canonical value is a reachable input.
    """
    from sglang.srt.utils.common import get_bool_env_var

    for raw in ("1", "true", "TRUE", "yes", "on", "y", "0", "false", "", "banana"):
        ms = _saver(armed=True, group="P", raw=raw)
        capture_routes = get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        assert ms.weg2_graph_tag_armed(True) is bool(capture_routes), (
            f"value {raw!r}: the sleep gate and the capture site disagree"
        )

    # And the capture site really is the reader mirrored above: if it is ever
    # migrated to `envs`, this goes red instead of the boot.
    from sglang.srt.model_executor.runner_backend import full_cuda_graph_backend as fg

    cap = inspect.getsource(fg)
    assert 'get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")' in cap, (
        "the capture site changed its reader -- weg2_graph_tag_armed mirrors "
        "get_bool_env_var and the two can now disagree on a value"
    )


def test_the_memory_saver_conjunct_is_read_from_the_callers_server_args():
    """Finding 2(a): with the saver OFF but the env ON, the capture site builds
    a NOOP adapter and routes nothing, while the region used to build a REAL
    TorchMemorySaverAdapter and route 384 MiB into a tag no sleep would pause.
    """
    ms = _saver(armed=True, group="P")
    assert ms.weg2_graph_tag_armed(False) is False
    with ms.weg2_graph_scratch_region(384 * 1024 * 1024, False) as tagged:
        assert tagged is False

    # The flashinfer call site must pass the real fact, not a literal.
    from sglang.srt.layers.attention import flashinfer_backend as fb

    src = inspect.getsource(fb)
    start = src.index("with weg2_graph_scratch_region(")
    window = src[start : start + 900]
    assert "enable_memory_saver" in window, (
        "the region's memory-saver conjunct is not read from the caller's "
        "server_args -- a hardcoded value re-opens the split it closes"
    )


def test_a_missing_torch_memory_saver_wheel_degrades_instead_of_killing_the_boot(caplog):
    """Finding 4: `TorchMemorySaverAdapter.create(enable=True)` RE-RAISES the
    import error (torch_memory_saver_adapter.py:36-45), and in the first
    version it sat ABOVE the try that exists to stop exactly that -- so it
    escaped into FlashInferAttnBackend.__init__, which is the boot killer the
    degrade path names as its reason to exist.
    """
    from sglang.srt.utils import torch_memory_saver_adapter as tmsa

    ms = _saver(armed=True, group="P")
    saved = tmsa.import_error
    tmsa.import_error = ImportError("No module named 'torch_memory_saver'")
    try:
        with caplog.at_level("WARNING"):
            with ms.weg2_graph_scratch_region(384 * 1024 * 1024, True) as tagged:
                assert tagged is False
    finally:
        tmsa.import_error = saved
    assert "NOT tagged" in caplog.text
    assert "RESIDENT across the sleep" in caplog.text


def test_the_group_name_instrument_stops_printing_a_question_mark():
    """Klasse A, carried: WEG2-FLIP-TAG's `group=` read an attribute nobody
    assigns, so it printed "?" on every rank of every boot while claiming to
    name the group.  ring_table._TAG_RE reads the token as a non-space run and does
    not capture it, so a real name breaks no parser.
    """
    M = _updater()
    probe = M.__new__(M)

    _saver(armed=True, group="D")
    assert M._weg2_group_name(probe) == "D"
    _saver(armed=True, group="")
    assert M._weg2_group_name(probe) == "?"


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
        with ms.weg2_graph_scratch_region(MIN_TAGGED_BYTES - 1, True) as tagged:
            assert tagged is False
    assert "NOT tagged" not in caplog.text, (
        "a sub-MIN_TAGGED_BYTES allocation reached the pool -- the #102 size "
        "gate is gone and a later small allocation can land in a paused tag's "
        "segment tail"
    )


def test_region_is_a_no_op_when_disarmed():
    ms = _saver(armed=False)
    with ms.weg2_graph_scratch_region(384 * 1024 * 1024, True) as tagged:
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
        with ms.weg2_graph_scratch_region(384 * 1024 * 1024, True, adapter=_Boom()) as tagged:
            assert tagged is False
    text = caplog.text
    assert "NOT tagged" in text
    assert "RESIDENT across the sleep" in text or "default pool" in text


@contextlib.contextmanager
def _tagged_region(ms):
    """Drive the region down its TAGGED path on a box with no CUDA.

    FIX 2 mutant M14 SURVIVED the first version of the caller-exception test,
    and the reason is the whole point of writing mutants: on this box the
    region degrades long before it reaches `yield True`, so a test that merely
    raises inside `with region(...)` proves nothing about the block that
    actually wraps the caller -- it exercised an early `yield False` return
    instead.  The pool and region API are therefore faked here so the caller's
    body really does run inside the guarded block.
    """
    import torch

    entered = []

    class _Pool:
        pass

    @contextlib.contextmanager
    def _use_mem_pool(pool):
        entered.append("pool")
        yield

    class _Adapter:
        @contextlib.contextmanager
        def region_config(self, **_kw):
            entered.append("region")
            yield

    saved = (getattr(torch.cuda, "MemPool", None), getattr(torch.cuda, "use_mem_pool", None))
    saved_pool = ms._GRAPH_SCRATCH_POOL
    torch.cuda.MemPool = _Pool
    torch.cuda.use_mem_pool = _use_mem_pool
    ms._GRAPH_SCRATCH_POOL = None
    try:
        yield _Adapter(), entered
    finally:
        if saved[0] is None:
            delattr(torch.cuda, "MemPool")
        else:
            torch.cuda.MemPool = saved[0]
        if saved[1] is None:
            delattr(torch.cuda, "use_mem_pool")
        else:
            torch.cuda.use_mem_pool = saved[1]
        ms._GRAPH_SCRATCH_POOL = saved_pool


def test_the_region_really_tags_when_the_pool_and_the_region_are_available():
    """The precondition of the test below: without this, a green
    caller-exception test can be green because the region never opened."""
    ms = _saver(armed=True)
    with _tagged_region(ms) as (adapter, entered):
        with ms.weg2_graph_scratch_region(384 * 1024 * 1024, True, adapter=adapter) as t:
            assert t is True
        assert entered == ["pool", "region"], (
            "the pool and the region_config were not both entered, so the "
            "allocation did not land in the cuda_graph tag"
        )


def test_callers_own_exception_is_not_swallowed_by_the_degrade_path():
    """Both paths: the early degrade AND the guarded block the caller's body
    actually runs inside (mutant M14)."""
    ms = _saver(armed=True)
    with pytest.raises(ValueError, match="caller fault"):
        with ms.weg2_graph_scratch_region(384 * 1024 * 1024, True):
            raise ValueError("caller fault")

    ms = _saver(armed=True)
    with _tagged_region(ms) as (adapter, _entered):
        with pytest.raises(ValueError, match="caller fault inside the region"):
            with ms.weg2_graph_scratch_region(
                384 * 1024 * 1024, True, adapter=adapter
            ):
                raise ValueError("caller fault inside the region")


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
                        debug_hold=False, tag="probe", group="P")
        assert env["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] == "1"

        os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = "0"
        env = build_env(tree="/tmp/t", venv="/tmp/v", cvd="0", store_dir="/tmp/s",
                        debug_hold=False, tag="probe", group="P")
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
    ms.weg2_graph_tag_armed(True)  # the resolver must not raise on this path

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


# ---------------------------------------------------------------------------
# FIX 3, FINDING 1: the boot arm's two instants, and the criterion that is
# satisfiable by a perfect run.
#
# Commit 3 sampled NVML free at both ends of the cycle IN THE SAME PHASE and
# then graded `free_after - free_before >= 384` as "did the item deliver".  A
# perfect run reads ~0, so it scored every card UNDER FLOOR and exited 4 -- it
# failed a working boot by 384 MiB per card.  Moving one sample to the other
# phase does not repair it: the cross-phase delta is `P_release - D_release`,
# in which this item's contribution CANCELS, because both groups release the
# same tag when they sleep.  So the same-phase delta grades CONSERVATION and
# delivery is graded on the ranks' own released-tag lines.
# ---------------------------------------------------------------------------

_FREE_STEADY = "0, RTX 3080, 1515\n1, RTX 5090, 816\n2, RTX 3080, 1368\n"
_I_BEFORE = "phase=D(awake) epoch=41 state=serving"
_I_AFTER = "phase=D(awake) epoch=45 state=serving"
_RELEASED_OK = (
    "WEG2-SLEEP released tags=['cuda_graph'] mib=486.0 ms=44 (instrument: ...)\n"
    "WEG2-SLEEP released tags=['cuda_graph'] mib=476.0 ms=41 (instrument: ...)\n"
    "WEG2-SLEEP released tags=['cuda_graph'] mib=517.0 ms=48 (instrument: ...)\n"
)


def _arm():
    from sglang.srt.weg2 import dormant_arm

    return dormant_arm


def _free(text):
    return {
        int(ln.split(",")[0]): (ln.split(",")[1].strip(), int(ln.split(",")[2]))
        for ln in text.strip().splitlines()
    }


def _grade(before=None, after=None, released=_RELEASED_OK, ranks=3,
           instant_before=_I_BEFORE, instant_after=_I_AFTER, degrades=(), noitem=None):
    arm = _arm()
    return arm.grade(
        before=_free(before or _FREE_STEADY),
        after=_free(after or _FREE_STEADY),
        instant_before=instant_before,
        instant_after=instant_after,
        capture={0: 102, 1: 92, 2: 133},
        capture_prov="boot weg2rg6",
        released=None if released is None else arm.parse_released(released),
        degrades=list(degrades),
        ranks=ranks,
        noitem=noitem if noitem is not None else {0: 1029, 1: 340, 2: 851},
        noitem_prov="boot weg2rg6 (no item)",
    )


def test_a_perfect_run_is_not_graded_as_a_failure_by_a_same_phase_delta():
    """THE round-2 blocker, as an executable regression.

    Both readings in the same phase, a 2 MiB jitter between them, and all three
    ranks releasing their full workspace + pool.  Commit 3's block returned
    EXIT 4 ("the item did NOT deliver") on exactly this input -- measured, with
    the heredoc extracted: `delta +0 / +0 / +0 -> UNDER FLOOR` on all three
    cards.  The only honest failure left on this input is the 5090's residual.
    """
    rc, lines = _grade(after="0, RTX 3080, 1517\n1, RTX 5090, 814\n2, RTX 3080, 1368\n")
    text = "\n".join(lines)
    assert rc != 4, (
        "a perfect run is graded as 'the item did not deliver' -- the same-phase "
        "delta is being used as a delivery test again:\n" + text
    )
    assert rc == 5, "the 5090 sits at 814 MiB, which is the named residual (exit 5)"
    assert "UNDER FLOOR" not in text


def test_the_delivery_criterion_is_the_released_tag_lines_not_the_free_delta():
    """Delivery has exactly one instrument, and it is not NVML free."""
    rc_missing, lines = _grade(released="")
    assert rc_missing == 4 and "0 released-tag line" in "\n".join(lines)
    rc_short, _ = _grade(released=_RELEASED_OK.replace("486.0", "383.0"))
    assert rc_short == 4, "a rank that released less than the workspace alone passed"
    rc_two, _ = _grade(released="\n".join(_RELEASED_OK.splitlines()[:2]))
    assert rc_two == 4, "two lines for three sleeping ranks passed"


def test_the_savers_could_not_answer_sentinel_is_not_read_as_an_empty_tag():
    """`mib=0.0` is the saver failing to answer; the line's own text says so.

    Not enough that the exit code is 4 -- "the saver could not answer" and "the
    rank released 380 of 384 MiB" are different faults with different next
    steps, so the VERDICT (not only the per-line table) has to say which one
    happened.
    """
    rc, lines = _grade(released=_RELEASED_OK.replace("486.0", "0.0"))
    assert rc == 4
    verdict = [ln for ln in lines if ln.startswith("VERDICT:")]
    assert verdict and "sentinel" in verdict[0], (
        "the verdict blamed the wrong thing for a mib=0.0 line: %s" % verdict
    )


def test_two_instants_in_different_phases_are_refused_rather_than_graded():
    rc, lines = _grade(instant_after="phase=P(awake) epoch=44 state=serving")
    assert rc == 2
    text = "\n".join(lines)
    assert "DIFFERENT front phases" in text
    assert "cancels" in text, "the refusal must say WHY, not just that it refuses"


def test_a_reading_without_its_instant_is_refused():
    """A free number without its phase is a comparison of two unknown states."""
    rc, lines = _grade(instant_before="epoch=41")
    assert rc == 2 and "does not name its phase" in "\n".join(lines)


def test_the_one_free_delta_that_does_fail_is_a_lost_workspace():
    """The hazard this item can introduce: a pause whose resume did not hand
    the pages back.  That is a LOSS of one whole workspace across a cycle that
    ends in the phase it started in."""
    rc, lines = _grade(after="0, RTX 3080, 1131\n1, RTX 5090, 816\n2, RTX 3080, 1368\n")
    assert rc == 4 and "LOST A WORKSPACE" in "\n".join(lines)


def test_a_gain_is_named_and_handed_to_the_answer_probe_not_silently_dropped():
    rc, lines = _grade(after="0, RTX 3080, 1915\n1, RTX 5090, 816\n2, RTX 3080, 1368\n")
    text = "\n".join(lines)
    assert "gained >= one workspace" in text
    assert rc != 4, "a gain is the answer probe's finding, not this table's"


def test_an_unreadable_window_prints_n_a_and_exits_2_rather_than_passing():
    rc, lines = _grade(released=None)
    text = "\n".join(lines)
    assert rc == 2 and "n/a" in text and "does not pass" in text


def test_the_cross_boot_baseline_is_an_indicator_and_never_an_exit_code():
    """It pairs THIS boot's reading with ANOTHER boot's, so it may inform and
    must not grade -- the class the record calls out for cross-boot numbers."""
    in_band = "0, RTX 3080, 1000\n1, RTX 5090, 900\n2, RTX 3080, 1100\n"
    rc_a, lines = _grade(before=in_band, after=in_band)
    assert rc_a == 0, "the sample is in band on every card and delivered"
    rc_b, _ = _grade(before=in_band, after=in_band, noitem={0: 1, 1: 1, 2: 1})
    assert rc_a == rc_b, (
        "a cross-boot baseline below the floor changed THIS boot's verdict from "
        "%d to %d -- another boot's number is now an exit code" % (rc_a, rc_b)
    )
    assert "INDICATOR, never an exit code" in "\n".join(lines)


def test_named_degrades_are_searched_over_the_whole_log_not_only_the_window():
    """The group-gate refusal is rate-limited and fires at the launcher's
    STARTUP sleep -- before this arm attaches.  A window-only search would
    report 'no degrade' on exactly the boot that degraded."""
    arm = _arm()
    whole = (
        "[boot] WEG2-SLEEP graph tag NOT armed: no_group -- SGLANG_WEG2_GROUP is empty\n"
        "[boot] ... a thousand lines ...\n"
        "[boot] WEG2-SLEEP released tags=['cuda_graph'] mib=486.0 ms=44\n"
    )
    found = arm.find_degrades(whole)
    assert len(found) == 1 and "no_group" in found[0]
    assert arm.find_degrades("nothing here") == []
    rc, lines = _grade(degrades=found)
    assert "NAMED DEGRADES FOUND" in "\n".join(lines)


def test_the_module_names_the_cancellation_that_forbids_a_cross_phase_grade():
    """Instrument-text law: the reason a cross-phase delta cannot grade
    delivery is arithmetic, and it has to be written where the grading is, not
    only in a record file."""
    arm = _arm()
    doc = arm.__doc__
    assert "P_release - D_release" in doc
    assert "CANCELS" in doc
    assert "before the front process exists" in doc


@pytest.mark.skipif(
    not os.path.exists("/spinning/gpu-arb/weg2/arm_dormant_boot.sh"),
    reason="evidence-tree-bound: the boot arm lives in /spinning/gpu-arb, not in the repo",
)
def test_the_boot_arm_stamps_both_readings_with_the_fronts_own_instant():
    src = open("/spinning/gpu-arb/weg2/arm_dormant_boot.sh").read()
    assert "/weg2/state" in src, "the instants must come from the front, not from a comment"
    assert "--instant-before" in src and "--instant-after" in src
    assert "dormant_arm.py" in src, "the grading must be the tested module"
    assert "PYCORRIDOR" not in src, (
        "the untested heredoc is back; it is the artifact that shipped the "
        "same-phase delivery test"
    )
    assert "plog-window" in src, "delivery needs the released-tag window"


# ---------------------------------------------------------------------------
# FIX 3, FINDING 2: the group gate refuses BY NAME.
#
# FIX 2 made the whole item depend on a NEW necessary condition --
# `weg2_group_name()` non-empty, i.e. SGLANG_WEG2_GROUP published by
# launcher.build_env -- and gave it no instrument.  When it was absent the item
# degraded SILENTLY: no line at the gate, none at the region (`yield False`),
# none at the sleep (the tag list came back unchanged).  A rank that had lost
# the variable was byte-identical in the logs to a rank running the pre-item
# tree, and the boot arm's degrade grep could not fire for it.  That is the
# #1246 shape: silent capability loss, exit 0, no W-code.
# ---------------------------------------------------------------------------


@pytest.fixture
def weg2_env():
    """Snapshot/restore the variables these tests write directly."""
    keys = ("SGLANG_WEG2_GROUP", "SGLANG_WEG2_WEIGHT_CHUNKS",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH")
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _rank(group, armed=True, family=True):
    """A rank with/without its group name, with/without the rest of the family."""
    ms = _saver(armed=armed, group=group)
    if family:
        os.environ["SGLANG_WEG2_WEIGHT_CHUNKS"] = "4"
    else:
        for key in list(os.environ):
            if key.startswith("SGLANG_WEG2_") and key != ms.WEG2_GROUP_ENV:
                os.environ.pop(key, None)
    ms._GRAPH_TAG_REFUSALS.clear()
    return ms


def test_a_lost_group_name_refuses_by_name_instead_of_disarming_silently(caplog, weg2_env):
    ms = _rank(group="", family=True)
    with caplog.at_level("WARNING"):
        assert ms.weg2_graph_tag_armed(True) is False
    text = caplog.text
    assert "WEG2-SLEEP graph tag NOT armed" in text, (
        "the item disarmed with no line at all -- the silent-capability-loss "
        "shape this fix exists to close"
    )
    assert "no_group" in text and ms.WEG2_GROUP_ENV in text, (
        "the line must name WHICH conjunct failed, not merely that one did"
    )
    assert ms.weg2_graph_tag_refusals() == {"no_group": 1}


def test_the_refusal_is_silent_when_the_group_name_is_there(caplog, weg2_env):
    ms = _rank(group="P", family=True)
    with caplog.at_level("WARNING"):
        assert ms.weg2_graph_tag_armed(True) is True
    assert "NOT armed" not in caplog.text
    assert ms.weg2_graph_tag_refusals() == {}


def test_a_stock_engine_gains_no_line_and_no_counter(caplog, weg2_env):
    """The stock path must stay byte-identical -- including its log.  A stock
    engine never had SGLANG_WEG2_GROUP, so its absence there is not a degrade
    and must not be reported as one."""
    ms = _rank(group="", family=False)
    with caplog.at_level("WARNING"):
        assert ms.weg2_graph_tag_armed(True) is False
    assert "NOT armed" not in caplog.text, (
        "a stock --enable-memory-saver engine now prints a Weg-2 warning it can "
        "do nothing about"
    )
    assert ms.weg2_graph_tag_refusals() == {}


def test_the_rate_limited_refusal_prints_its_own_denominator(caplog, weg2_env):
    """DENOMINATOR LAW: a rate-limited emitter that does not print its
    suppressed count turns 'refused on every leg' into 'refused once'."""
    ms = _rank(group="", family=True)
    with caplog.at_level("WARNING"):
        for _ in range(5):
            ms.weg2_graph_tag_armed(True)
    fired = [r for r in caplog.records if "NOT armed" in r.getMessage()]
    assert len(fired) == 3, "expected occurrences 1, 2 and 4 to fire, got %d" % len(fired)
    assert "occurrence=1" in fired[0].getMessage()
    assert "occurrence=4" in fired[-1].getMessage()
    assert ms.weg2_graph_tag_refusals() == {"no_group": 5}, (
        "the count is the denominator the line's occurrence= is drawn from"
    )


def test_the_env_conjunct_also_refuses_by_name(caplog, weg2_env):
    ms = _rank(group="P", armed=False, family=True)
    with caplog.at_level("WARNING"):
        assert ms.weg2_graph_tag_armed(True) is False
    assert "env_off" in caplog.text and "SGLANG_MEMORY_SAVER_CUDA_GRAPH" in caplog.text


def test_the_memory_saver_conjunct_also_refuses_by_name(caplog, weg2_env):
    ms = _rank(group="P", family=True)
    with caplog.at_level("WARNING"):
        assert ms.weg2_graph_tag_armed(False) is False
    assert "no_memory_saver" in caplog.text


def test_the_regions_disarmed_path_is_no_longer_a_silent_yield(caplog, weg2_env):
    """weg2_memory_saver.py's region took `yield False; return` with no log
    when the tag was not armed -- one of the three sites that made a lost group
    name invisible.  It calls the gate, so the gate's line is its line."""
    ms = _rank(group="", family=True)
    with caplog.at_level("WARNING"):
        with ms.weg2_graph_scratch_region(384 * 1024 * 1024, True) as tagged:
            assert tagged is False
    assert "WEG2-SLEEP graph tag NOT armed" in caplog.text
    assert "no_group" in caplog.text


def test_the_size_gate_stays_a_silent_decision(caplog, weg2_env):
    """Carried from FIX 2: the #102 size gate is a DECISION, not a degrade, and
    must not start printing now that its neighbour does."""
    from sglang.srt.speculative.adaptive_graph_memory import MIN_TAGGED_BYTES

    ms = _rank(group="P", family=True)
    with caplog.at_level("WARNING"):
        with ms.weg2_graph_scratch_region(MIN_TAGGED_BYTES - 1, True) as tagged:
            assert tagged is False
    assert "NOT armed" not in caplog.text and "NOT tagged" not in caplog.text


def test_the_stock_discriminator_is_the_rest_of_the_weg2_family(weg2_env):
    """`weg2_env_present` is what tells a stock engine from a Weg-2 rank that
    lost its name, and its limit is stated rather than hidden: it is sufficient
    for the loud case, not necessary."""
    ms = _rank(group="", family=False)
    assert ms.weg2_env_present() is False
    os.environ["SGLANG_WEG2_WEIGHT_CHUNKS"] = "4"
    assert ms.weg2_env_present() is True
    os.environ[ms.WEG2_GROUP_ENV] = "P"
    os.environ.pop("SGLANG_WEG2_WEIGHT_CHUNKS")
    assert ms.weg2_env_present() is False, (
        "the group variable must not count as its own corroboration"
    )
    assert "HONEST LIMIT" in ms.weg2_env_present.__doc__


def test_launcher_publishes_the_group_beside_that_family_on_both_groups(weg2_env):
    """The refusal above is only correct if the launcher really does publish
    both halves on both groups: the group name, and at least one other
    SGLANG_WEG2_* variable to corroborate it."""
    from sglang.srt.managers import weg2_memory_saver as ms
    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher)
    family = set(re.findall(r'"(SGLANG_WEG2_[A-Z_]+)"', src)) - {ms.WEG2_GROUP_ENV}
    assert family, "no corroborating SGLANG_WEG2_* variable exists in the launcher"

    saved = os.environ.pop(ms.WEG2_GROUP_ENV, None)
    try:
        for group in ("P", "D"):
            env = launcher.build_env(tree="/tmp/t", venv="/tmp/v", cvd="0",
                                     store_dir="/tmp/s", debug_hold=False,
                                     tag="probe", chunk_layers=4, chunk_count=2,
                                     group=group)
            assert env[ms.WEG2_GROUP_ENV] == group
            assert family & set(env), (
                "group %s is published without any corroborating variable, so a "
                "rank that lost the group name cannot be told from a stock "
                "engine" % group
            )
    finally:
        if saved is not None:
            os.environ[ms.WEG2_GROUP_ENV] = saved
