# SPDX-License-Identifier: Apache-2.0
"""#462 -- the BREAKABLE MoE expert-offload graph route, hermetic gates.

The route: the eager phase fetches the routed experts into a FIXED slot arena
and publishes the slot vector before replay; the captured compute addresses
slots. Fetching inside the graph stays refuted (#452) and gate 1 is asserted
intact here rather than assumed.

Gates:
  * IDENTITY -- the breakable step and the eager step remap every element the
    same way and leave the slot arena holding the same bytes. Proven on a real
    ``MoEExpertOffloadCache`` over CPU tensors, so it is a fixture, not a mock.
  * PAD CONTRACT -- graph-PADDED batches carry ``-1`` rows, and padded rows that
    carry REAL ids still route real experts. Both are pinned: ``-1`` survives to
    ``-1``, and the fetch plan covers the padded rows' experts.
  * SHARED BUFFER -- two captured shapes of one layer get disjoint (bridge,
    stage) pairs, and preparing one never disturbs the other. This is the family
    that keeps recurring (htccl ``_get_out_buf``, ``GraphSharedOutput``,
    ``_DEQUANT_WS``), so the ordering assumption is a test, not a comment.
  * CAPTURE REFUSAL -- the arena consults #286's register gate and refuses a
    park while a capture is active.
  * GATE -- mode selection, both spellings of the refuted path still refused,
    and every boot precondition of the breakable route.
  * SYNC BUDGET -- exactly ONE ``tolist`` per layer per step, and no other host
    read of a device-side tensor.

Run:  python -m pytest tests/moe_offload/test_breakable_route_462.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe import offload_capture_gate as gate  # noqa: E402
from sglang.srt.layers.moe.breakable_offload import (  # noqa: E402
    EAGER_HOST_BLOCKING_CROSSINGS_PER_LAYER_PER_STEP,
    HOST_BLOCKING_CROSSINGS_PER_LAYER_PER_STEP,
    HOST_SYNCS_PER_LAYER_PER_STEP,
    BreakableOffloadArena,
    BreakableScratchOverflow,
    breakable_opt_in,
)
from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    MoEExpertOffloadCache,
    remap_ids_host,
)
from sglang.srt.model_executor import short_term_offload_register as reg  # noqa: E402

E, R, C = 32, 8, 6
HID, INTER = 4, 6


class _StubLayer:
    """Minimal expert-major layer with a LOAD-TIME PRESPLIT already stashed.

    The presplit branch of ``install()`` is the one that needs no CUDA (the
    split-here branch calls ``pin_memory()``), and it is also the branch
    production takes, so the fixture exercises the real path rather than a
    CPU-only detour. Expert ``e``'s rows are filled with the value ``e``, so a
    slot's contents identify their occupant by inspection.
    """

    def __init__(self, layer_id=11, scratch=C):
        self.layer_id = layer_id
        self.num_local_experts = E
        self._moe_offload_full_experts = E
        slots = R + scratch
        presplit = {}
        for attr, shape in (
            ("w13_weight", (INTER, HID)),
            ("w2_weight", (HID, INTER)),
        ):
            resident = torch.zeros((slots,) + shape)
            for e in range(R):
                resident[e].fill_(float(e))
            spill = torch.zeros((E - R,) + shape)
            for j, e in enumerate(range(R, E)):
                spill[j].fill_(float(e))
            presplit[attr] = (resident, spill)
        self._moe_offload_presplit = presplit


@pytest.fixture(autouse=True)
def _no_scratch_env_leak():
    """``_cache`` sets SGLANG_MOE_SCRATCH_SLOTS, which ``scratch_slot_count``
    reads from the raw environment. Restore it after every test so this module
    cannot change what a later one measures (it did: test_planner's default-C
    assertion went red)."""
    key = "SGLANG_MOE_SCRATCH_SLOTS"
    before = os.environ.get(key)
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = before


def _cache(layer=None, scratch=C):
    os.environ["SGLANG_MOE_SCRATCH_SLOTS"] = str(scratch)
    cache = MoEExpertOffloadCache(layer or _StubLayer(scratch=scratch), R / E)
    assert cache.resident_count == R and cache.scratch == scratch
    cache.install()
    return cache


def _eager_remap(cache, topk_ids):
    """The eager single-wave step's remap, exactly as ``_run_single_wave`` does
    it: host read -> resolve -> fetch -> device LUT -> device gather."""
    ids_list = topk_ids.tolist()
    needed = sorted({e for row in ids_list for e in row if e >= 0})
    slot_of_needed, fetch_plan = cache.planner.resolve(needed)
    cache._fetch(fetch_plan)
    lut = cache._build_lut(slot_of_needed, topk_ids.dtype, topk_ids.device)
    return cache._remap(topk_ids, lut)


def _ids(rows):
    return torch.tensor(rows, dtype=torch.int64)


# --------------------------------------------------------------------------
# IDENTITY
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows",
    [
        [[0, 9, 3]],  # decode bs=1, mixed resident/spill
        [[9, 10, 11], [12, 9, 3]],  # two tokens sharing a spill expert
        [[0, 1, 2]],  # fully resident: empty fetch plan
        [[31, 30, 29], [28, 27, 9]],  # spill only, exactly C distinct
    ],
)
def test_breakable_and_eager_remap_every_element_the_same(rows):
    topk = _ids(rows)

    eager_cache = _cache()
    want = _eager_remap(eager_cache, topk)

    brk_cache = _cache()
    bridge = torch.empty_like(topk)
    brk_cache.prepare_breakable(topk, bridge, stage=None)

    assert torch.equal(bridge, want), f"{bridge.tolist()} != {want.tolist()}"


@pytest.mark.parametrize("rows", [[[0, 9, 3]], [[9, 10, 11], [12, 9, 3]]])
def test_breakable_leaves_the_slot_arena_byte_identical_to_eager(rows):
    """The remap agreeing is not enough: the SLOTS the graph will read must
    hold the same expert bytes. This is the half a mocked fetch would miss."""
    topk = _ids(rows)

    eager_cache = _cache()
    _eager_remap(eager_cache, topk)

    brk_cache = _cache()
    brk_cache.prepare_breakable(topk, torch.empty_like(topk), stage=None)

    for attr in ("w13_weight", "w2_weight"):
        assert torch.equal(
            brk_cache._resident[attr], eager_cache._resident[attr]
        ), f"slot arena diverged on {attr}"


def test_remap_ids_host_is_the_lut_gather_element_for_element():
    """Exhaustive over the whole id range including the pad marker, so the
    equivalence is proven rather than sampled."""
    cache = _cache()
    rows = [[e for e in range(-1, E)]]
    ids_list = rows
    needed = sorted({e for r in ids_list for e in r if e >= 0})
    # Only the first C spill experts fit; take a resolvable subset instead.
    needed = [e for e in needed if e < R] + [e for e in needed if e >= R][:C]
    slot_of_needed, _ = cache.planner.resolve(needed)

    topk = _ids(rows)
    lut = cache._build_lut(slot_of_needed, topk.dtype, topk.device)
    want = cache._remap(topk, lut).reshape(-1).tolist()

    assert remap_ids_host(ids_list, slot_of_needed) == want


def test_falsifier_a_wrong_host_remap_is_caught():
    """Can-fail arm for the identity gate: perturb one entry and the element
    comparison must go red, i.e. the gate is not vacuous."""
    cache = _cache()
    slot_of_needed, _ = cache.planner.resolve([0, 9])
    good = remap_ids_host([[0, 9]], slot_of_needed)
    bad = dict(slot_of_needed)
    bad[9] = bad[9] + 1
    assert remap_ids_host([[0, 9]], bad) != good


# --------------------------------------------------------------------------
# PAD CONTRACT (#444 pad-slot family)
# --------------------------------------------------------------------------


def test_pad_rows_survive_as_pad():
    cache = _cache()
    topk = _ids([[0, -1, 9], [-1, -1, -1]])
    bridge = torch.empty_like(topk)
    cache.prepare_breakable(topk, bridge, stage=None)
    assert bridge[0, 1].item() == -1
    assert bridge[1].tolist() == [-1, -1, -1]
    assert bridge[0, 0].item() >= 0 and bridge[0, 2].item() >= 0


def test_padded_rows_carrying_real_ids_are_still_fetched():
    """A graph-padded batch's tail rows are not blank -- the router ran on
    padded hidden states and produced REAL ids. Those experts must be resolved
    and fetched, or the padded rows would read a stale slot; and they must be
    counted against the scratch bound, which is why the boot bound uses the full
    padded ``tokens x top_k``."""
    cache = _cache()
    real, padded = [0, 9, 3], [17, 18, 19]  # 'padded' row routes real spill ids
    topk = _ids([real, padded])
    bridge = torch.empty_like(topk)
    cache.prepare_breakable(topk, bridge, stage=None)

    assert (bridge >= 0).all()
    slots = set(bridge.reshape(-1).tolist())
    assert len(slots) == 6, "each distinct expert must land in its own slot"
    for expert in (17, 18, 19):
        slot = bridge[1, [17, 18, 19].index(expert)].item()
        assert torch.equal(
            cache._resident["w13_weight"][slot],
            torch.full((INTER, HID), float(expert)),
        ), f"slot {slot} does not hold expert {expert}"


def test_pad_only_batch_fetches_nothing_and_stays_pad():
    cache = _cache()
    topk = _ids([[-1, -1], [-1, -1]])
    bridge = torch.empty_like(topk)
    before = cache.planner.stats.fetches
    cache.prepare_breakable(topk, bridge, stage=None)
    assert bridge.tolist() == [[-1, -1], [-1, -1]]
    assert cache.planner.stats.fetches == before


# --------------------------------------------------------------------------
# SHARED-BUFFER FAMILY
# --------------------------------------------------------------------------


def test_two_capture_shapes_get_disjoint_bridges():
    arena = BreakableOffloadArena(layer_id=1)
    a = arena.bridge_for(_ids([[0, 1, 2]]))
    b = arena.bridge_for(_ids([[0, 1, 2], [3, 4, 5]]))
    assert a is not b
    assert a.buf.data_ptr() != b.buf.data_ptr()
    assert arena.bridge_for(_ids([[0, 1, 2]])) is a  # stable across replays
    assert len(arena.shapes) == 2


def test_preparing_one_bucket_does_not_disturb_another():
    """The aliasing falsifier. If the two buckets ever shared one buffer -- the
    recurring shared-buffer bug -- bucket A's published slot vector would be
    overwritten by bucket B's and this goes red."""
    cache = _cache()
    arena = BreakableOffloadArena(layer_id=1)

    small = _ids([[0, 9, 3]])
    large = _ids([[0, 9, 3], [10, 11, 12]])

    bridge_small = arena.bridge_for(small)
    cache.prepare_breakable(small, bridge_small.buf, bridge_small.stage)
    snapshot = bridge_small.buf.clone()

    bridge_large = arena.bridge_for(large)
    cache.prepare_breakable(large, bridge_large.buf, bridge_large.stage)

    assert torch.equal(bridge_small.buf, snapshot), (
        "bucket A's bridge changed while bucket B was prepared -- the bridges "
        "alias, which is the shared-buffer family"
    )


def test_every_bridge_owns_its_stage_exclusively():
    """On CUDA each bridge carries its own pinned mirror; the pairing is what
    removes the reuse-ordering question. Without a device the stage is None for
    every bridge, and the pairing is asserted structurally."""
    arena = BreakableOffloadArena(layer_id=1)
    bridges = [
        arena.bridge_for(_ids([[0] * 3] * n)) for n in (1, 2, 4)
    ]
    stages = [b.stage for b in bridges if b.stage is not None]
    ptrs = {s.data_ptr() for s in stages}
    assert len(ptrs) == len(stages), "two bridges share one staging buffer"
    for bridge in bridges:
        if bridge.stage is not None:
            assert bridge.stage.numel() == bridge.buf.numel()


# --------------------------------------------------------------------------
# CAPTURE REFUSAL (#286 consumption)
# --------------------------------------------------------------------------


def test_arena_refuses_to_park_while_a_capture_is_active():
    arena = BreakableOffloadArena(layer_id=4)
    reg.set_capture_probe(lambda: True)
    try:
        with pytest.raises(reg.OffloadUnderCaptureRefused) as excinfo:
            arena.park(target="host_ram")
    finally:
        reg.set_capture_probe(None)
    assert "park" in str(excinfo.value)
    assert "layer 4" in str(excinfo.value)


def test_arena_park_refuses_by_name_even_between_replays():
    """Outside a capture the register's gate passes, and the arena still
    refuses -- a graph that baked in these addresses may exist regardless of
    whether a capture is running right now. Falsifier for 'the #286 gate alone
    is enough'."""
    arena = BreakableOffloadArena(layer_id=4)
    reg.set_capture_probe(lambda: False)
    try:
        with pytest.raises(gate.BreakableModeRefused):
            arena.park()
    finally:
        reg.set_capture_probe(None)


def test_arena_consumes_the_286_descriptor_rather_than_restating_it():
    arena = BreakableOffloadArena(layer_id=4)
    assert arena.asset_class.offload_class == "experts"
    assert arena.asset_class is reg.describe_class("experts")


# --------------------------------------------------------------------------
# GATE
# --------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch):
    for name in (
        gate.ENV_GRAPH_MODE,
        "SGLANG_MOE_OFFLOAD_CUDA_GRAPH",
        gate.ENV_GRAPH_REFUTED_OVERRIDE,
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_default_is_eager_and_nothing_else(clean_env):
    assert gate.env_graph_mode() == gate.MODE_EAGER
    assert gate.resolve_offload_graph_mode(0.5, False, 0) == gate.MODE_EAGER
    assert gate.resolve_graph_mode(0.5, False, 0) is False
    assert breakable_opt_in() is False


def test_breakable_is_selected_only_by_its_own_name(clean_env):
    clean_env.setenv(gate.ENV_GRAPH_MODE, "breakable")
    assert gate.resolve_offload_graph_mode(0.5, False, 0) == gate.MODE_BREAKABLE
    assert breakable_opt_in() is True
    # ... and it does NOT turn on the refuted capturable flag.
    assert gate.resolve_graph_mode(0.5, False, 0) is False


def test_unknown_mode_is_refused_by_name(clean_env):
    clean_env.setenv(gate.ENV_GRAPH_MODE, "captureable")
    with pytest.raises(gate.BreakableModeRefused) as excinfo:
        gate.env_graph_mode()
    assert "not a known mode" in str(excinfo.value)


@pytest.mark.parametrize("spelling", ["env", "mode"])
def test_both_spellings_of_the_refuted_path_still_refuse(clean_env, spelling):
    """#452's refusal must not be walkable around by using the new vocabulary."""
    opt_in = False
    if spelling == "env":
        opt_in = True
    else:
        clean_env.setenv(gate.ENV_GRAPH_MODE, "capturable")
    with pytest.raises(gate.CapturableOffloadRefuted):
        gate.resolve_offload_graph_mode(0.5, opt_in, 0)


def test_capturable_override_still_opens_the_refuted_path(clean_env):
    clean_env.setenv(gate.ENV_GRAPH_REFUTED_OVERRIDE, "1")
    assert gate.resolve_offload_graph_mode(0.5, True, 0) == gate.MODE_CAPTURABLE
    assert gate.resolve_graph_mode(0.5, True, 0) is True


def test_breakable_boot_refuses_without_an_offload(clean_env):
    with pytest.raises(gate.BreakableModeRefused) as excinfo:
        gate.validate_breakable_boot(1.0, layer_id=2)
    assert "not active" in str(excinfo.value)


def test_breakable_boot_refuses_alongside_the_capturable_opt_in(clean_env):
    clean_env.setenv("SGLANG_MOE_OFFLOAD_CUDA_GRAPH", "1")
    with pytest.raises(gate.BreakableModeRefused) as excinfo:
        gate.validate_breakable_boot(0.5, layer_id=2)
    assert "mutually exclusive" in str(excinfo.value)


class _Phase:
    def __init__(self, backend):
        self.backend = backend


class _Cfg:
    def __init__(self, decode, prefill):
        self.decode = _Phase(decode)
        self.prefill = _Phase(prefill)


class _Args:
    disable_cuda_graph = False

    def __init__(self, decode, prefill):
        self.cuda_graph_config = _Cfg(decode, prefill)


def _with_args(monkeypatch, decode, prefill):
    import sglang.srt.runtime_context as rc

    monkeypatch.setattr(rc, "get_server_args", lambda: _Args(decode, prefill))


def test_breakable_boot_requires_the_breakable_decode_backend(clean_env):
    _with_args(clean_env, "full", "disabled")
    with pytest.raises(gate.BreakableModeRefused) as excinfo:
        gate.validate_breakable_boot(0.5, layer_id=2)
    assert "NO-OP under any other backend" in str(excinfo.value)


def test_breakable_boot_requires_eager_prefill(clean_env):
    _with_args(clean_env, "breakable", "breakable")
    with pytest.raises(gate.BreakableModeRefused) as excinfo:
        gate.validate_breakable_boot(0.5, layer_id=2)
    assert "wave-splitting" in str(excinfo.value)


def test_breakable_boot_accepts_the_one_working_shape(clean_env):
    _with_args(clean_env, "breakable", "disabled")
    gate.validate_breakable_boot(0.5, layer_id=2)  # must not raise


def test_legacy_disable_cuda_graph_reads_as_disabled(clean_env):
    import sglang.srt.runtime_context as rc

    args = _Args("breakable", "disabled")
    args.disable_cuda_graph = True
    clean_env.setattr(rc, "get_server_args", lambda: args)
    assert gate.resolved_backend("decode") == "disabled"


# --------------------------------------------------------------------------
# SCRATCH BOUND
# --------------------------------------------------------------------------


def test_scratch_overflow_refuses_by_name_with_the_numbers():
    cache = _cache(scratch=2)
    topk = _ids([[9, 10, 11, 12]])
    with pytest.raises(BreakableScratchOverflow) as excinfo:
        cache.prepare_breakable(topk, torch.empty_like(topk), stage=None)
    message = str(excinfo.value)
    assert excinfo.value.spill == 4 and excinfo.value.scratch == 2
    assert "cannot wave-split" in message
    assert "SGLANG_MOE_SCRATCH_SLOTS" in message


def test_the_overflow_check_runs_before_any_counter_moves():
    """A refusal must not leave the residency stats claiming a forward that
    never happened -- that is what makes the #390 dump trustworthy."""
    cache = _cache(scratch=2)
    before = (cache.planner.stats.forwards, cache.planner.stats.fetches)
    topk = _ids([[9, 10, 11, 12]])
    with pytest.raises(BreakableScratchOverflow):
        cache.prepare_breakable(topk, torch.empty_like(topk), stage=None)
    assert (cache.planner.stats.forwards, cache.planner.stats.fetches) == before


# --------------------------------------------------------------------------
# SYNC BUDGET
# --------------------------------------------------------------------------


def test_exactly_one_host_read_per_layer_per_step(monkeypatch):
    """The route's cost claim, pinned. One ``tolist`` -- the irreducible
    rendezvous -- and no ``item``/``cpu``/``numpy``/``nonzero`` on top of it."""
    cache = _cache()
    topk = _ids([[0, 9, 3], [10, 11, 12]])

    counts = {}
    for name in ("tolist", "item", "cpu", "numpy", "nonzero"):
        original = getattr(torch.Tensor, name)
        counts[name] = 0

        def make(name=name, original=original):
            def probe(self, *a, **kw):
                counts[name] += 1
                return original(self, *a, **kw)

            return probe

        monkeypatch.setattr(torch.Tensor, name, make())

    cache.prepare_breakable(topk, torch.empty_like(topk), stage=None)

    assert counts["tolist"] == HOST_SYNCS_PER_LAYER_PER_STEP == 1
    assert counts["item"] == 0
    assert counts["cpu"] == 0
    assert counts["nonzero"] == 0


def test_the_eager_path_pays_more_host_blocking_than_the_breakable_one():
    """The documented crossing counts are a claim about the two paths, so state
    the direction as a test rather than only in a docstring."""
    assert (
        HOST_BLOCKING_CROSSINGS_PER_LAYER_PER_STEP
        < EAGER_HOST_BLOCKING_CROSSINGS_PER_LAYER_PER_STEP
    )
    assert HOST_SYNCS_PER_LAYER_PER_STEP == 1


def test_the_eager_lut_build_really_does_issue_the_two_extra_transfers(monkeypatch):
    """Positive control for the crossing count above.

    ``_build_lut`` ships TWO numpy-backed host tensors to the device with
    ``non_blocking=True``. Those two are the crossings the breakable route
    removes, and the flag is the point: ``non_blocking`` is honoured only for
    PINNED memory, and ``torch.from_numpy`` gives pageable memory, so on a real
    device both copies block the host despite asking not to.

    The transfer count -- not the source/destination devices -- is what this
    asserts, because the hermetic fixture has no device to copy to. If the eager
    path ever stops issuing them, the documented 3-vs-2 delta is stale and this
    goes red.
    """
    cache = _cache()
    transfers = []
    original = torch.Tensor.to

    def probe(self, *a, **kw):
        if kw.get("non_blocking") is True:
            transfers.append(tuple(str(x) for x in a))
        return original(self, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "to", probe)
    slot_of_needed, _ = cache.planner.resolve([0, 9, 3])
    cache._build_lut(slot_of_needed, torch.int64, torch.device("cpu"))

    assert len(transfers) == (
        EAGER_HOST_BLOCKING_CROSSINGS_PER_LAYER_PER_STEP
        - HOST_SYNCS_PER_LAYER_PER_STEP
    ) == 2, f"expected two non_blocking transfers, saw {transfers}"


def test_the_breakable_step_issues_no_non_blocking_transfer_at_all(monkeypatch):
    """The other half of the same claim: the breakable publish is ONE copy and
    it is deliberately blocking, so a reused pinned stage cannot be overwritten
    before its DMA lands (the shared-buffer family). Nothing on this path asks
    for ``non_blocking`` except the expert fetch itself, which copies out of the
    pinned pool and is joined on the copy stream."""
    cache = _cache()
    topk = _ids([[0, 9, 3]])
    asks = []
    original = torch.Tensor.to

    def probe(self, *a, **kw):
        if kw.get("non_blocking") is True:
            asks.append(tuple(str(x) for x in a))
        return original(self, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "to", probe)
    cache.prepare_breakable(topk, torch.empty_like(topk), stage=None)
    assert asks == [], f"breakable publish issued a non_blocking transfer: {asks}"
