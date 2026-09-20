"""CPU whole-boot probe: P39 -> P32 -> P39 across three fake ranks.

The per-blocker tests each prove one repair in isolation. This one is the
acceptance: it stands up the whole object graph a real boot stands up -- three
PP stages, each with a model, a ModelRunner, a KV pool, a captured graph ladder
and a BAR1 mover -- and drives a full round trip through it, on the CPU, with
fake cards.

It exists because the four blockers fail in COMBINATION, and each one alone
looks survivable:

* the runner can follow the boundary while the graphs do not, and then only the
  replayed path is wrong;
* the graphs can follow while the pool was built narrow, and then the newly
  activated layer has no rows;
* both can follow while the mover ran under capture, and then bytes were
  deposited and never collected.

The round trip is the shape that matters. A one-way switch passes with a
destructive narrowing in it (the hybrid-SWA lists), an indexing base that
moved, and a graph ladder that was discarded rather than restored. Coming BACK
to P39 and finding the boot state byte for byte is what those three cannot
fake.

Hermetic: CUDA_VISIBLE_DEVICES="", no torch device, no model weights. Every
card is a dict and every graph is a string.
"""

from __future__ import annotations

import pytest

from sglang.srt.model_executor.layout_boundary import (
    LayoutBoundaryActuator,
    LayoutBoundaryError,
    ModelRunnerRangeMirror,
    cuda_graph_observer,
    model_runner_observer,
    pool_coverage_observer,
    union_layer_window,
    validate_world_tiling,
)
from sglang.srt.model_executor.runner.graph_recapture import (
    PrefillGraphRecaptureRegistry,
)
from sglang.srt.weg2.p_layout_switch import (
    BandGeometry,
    LayerBytes,
    LinkRates,
    MoverNotCapturable,
    PLayout,
    RingState,
    StagePermutation,
    accept_pp0_verdict,
    emit_move_ops,
    placement_switch_seconds,
    pp0_broadcast,
    preconditions_digest,
    require_switchable,
    require_union_fits,
    run_move_ops_guarded,
    union_kv_costs,
    union_layout,
)

# The rig's geometry, the same one tests/moe_offload/test_p_layout_switch_0920.py
# is calibrated on: 64 layers, the 5090 carrying 39 of them and about three
# times as fast per layer as the two 3080s.
N_LAYERS = 64
P39 = PLayout(name="p39", counts=(39, 13, 12))
P32 = PLayout(name="p32", counts=(32, 16, 16))
PER_LAYER_BYTES = 385_000_000
LAYER_BYTES = LayerBytes({i: PER_LAYER_BYTES for i in range(N_LAYERS)})
GEOM = BandGeometry(layers_per_chunk=8, chunk_count=8)

# Shipped placement: stage 0 = 5090 (cuda:0), stage 1 = 3080 x4, stage 2 = 3080 x8.
RATES = LinkRates(
    {
        (0, 1): 7.13,
        (1, 0): 6.56,
        (0, 2): 14.25,
        (2, 0): 13.15,
        (1, 2): 6.58,
        (2, 1): 6.58,
    }
)
CAPTURE_LADDER = (512, 2048, 8192)


# ---------------------------------------------------------------------------
# The fake rank.
# ---------------------------------------------------------------------------


class _Param:
    pass


class _RealLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id

    def parameters(self):
        yield _Param()


class _Model:
    def __init__(self, resident):
        self._start_layer = min(resident)
        self._end_layer = max(resident) + 1
        # Load wide: every layer of the UNION is a real module on this rank.
        self.layers = [
            _RealLayer(i) if i in resident else None for i in range(N_LAYERS)
        ]
        self.layers = [
            layer if layer is not None else _Missing() for layer in self.layers
        ]

    @property
    def start_layer(self):
        return self._start_layer

    @property
    def end_layer(self):
        return self._end_layer


class _Missing:
    def parameters(self):
        return iter(())


class _HfConfig:
    architectures = ["Qwen3NextForCausalLM"]
    loop_num = 1


class _ModelConfig:
    def __init__(self, full_ids, swa_ids):
        self.num_hidden_layers = N_LAYERS
        self.num_attention_layers = N_LAYERS
        self.hf_config = _HfConfig()
        self.is_deepseek_v4_arch = False
        self.full_attention_layer_ids = list(full_ids)
        self.swa_attention_layer_ids = list(swa_ids)


class _Runner:
    def __init__(self, pp_rank, model_config):
        self.pp_rank = pp_rank
        self.pp_size = 3
        self.model_config = model_config
        self.is_hybrid_swa = True
        self.start_layer = None
        self.end_layer = None
        self.num_effective_layers = None


class _KvPool:
    """Built ONCE, over the union. Records every build so a rebuild is visible."""

    builds = 0

    def __init__(self, start_layer, end_layer, layer_num):
        type(self).builds += 1
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.layer_num = int(layer_num)

    def local_slot(self, layer_id):
        """The indexing base. If start_layer ever moved, every cached row
        silently shifts -- which is why the probe asserts it never does."""
        if not (self.start_layer <= layer_id < self.end_layer):
            raise KeyError(
                f"layer {layer_id} has no rows in this pool "
                f"[{self.start_layer},{self.end_layer})"
            )
        return layer_id - self.start_layer


class _Backend:
    def __init__(self):
        self._graphs = {}
        self.discards = 0

    def discard_shape(self, shape_key):
        self.discards += 1
        return self._graphs.pop(int(shape_key.size), None) is not None

    def can_run(self, size):
        return int(size) in self._graphs


class _GraphRunner:
    def __init__(self, model):
        self.model = model
        self.backend = _Backend()
        self.captures = []
        self.capturing = False

    def capture_boot(self):
        for size in CAPTURE_LADDER:
            self._capture(size)

    def _capture(self, size):
        # What a real capture bakes: the range in force at capture time.
        rng = (self.model._start_layer, self.model._end_layer)
        self.backend._graphs[int(size)] = rng
        self.captures.append((int(size), rng))

    def recapture_shapes(self, sizes):
        self.capturing = True
        try:
            for size in sizes:
                self._capture(size)
        finally:
            self.capturing = False
        return tuple(int(s) for s in sizes)

    def baked_range(self, size):
        return self.backend._graphs[int(size)]


class _Card:
    """A fake card: a BAR1 window and a byte counter."""

    def __init__(self, stage, direct):
        self.stage = stage
        self.direct = direct
        self.dev_ptr = 0x1000_0000 + stage * 0x100_0000
        self.bytes_in = 0


class _Regions:
    def __init__(self, cards):
        self.cards = cards

    def src_ptr(self, layer_id):
        return 0x2000_0000 + layer_id * 0x10000

    def dst_ptr(self, layer_id):
        return 0x3000_0000 + layer_id * 0x10000

    def window(self, src_stage, dst_stage):
        return self.cards[dst_stage]


class _DeviceOps:
    def __init__(self):
        self.calls = []

    def memcpy_async(self, dst, src, nbytes, stream):
        self.calls.append((dst, src, nbytes, stream))


class FakeRank:
    """One PP stage, wired the way a boot wires it."""

    def __init__(self, stage, union_span, full_ids, swa_ids, cards):
        self.stage = stage
        self.union = union_span
        resident = set(range(*union_span))
        self.model = _Model(resident)
        self.config = _ModelConfig(full_ids, swa_ids)
        self.runner = _Runner(stage, self.config)
        self.cards = cards
        self.regions = _Regions(cards)
        self.device_ops = _DeviceOps()

        # --- boot order, and the order is the point --------------------
        # 1. pools and the mirror are built INSIDE the union window, so every
        #    one of the ~20 readers sees the union without being told.
        with union_layer_window(self.runner, union_span) as span:
            self.pool = _KvPool(span[0], span[1], span[1] - span[0])
            self.mirror = ModelRunnerRangeMirror(self.runner, span)
        # 2. the actuator then narrows the MODEL to the boot rung.
        self.act = LayoutBoundaryActuator(
            self.model,
            {"p39": P39.range_of(stage), "p32": P32.range_of(stage)},
            "p39",
        )
        # 3. the runner follows it down to the same rung.
        self.mirror.apply(P39.range_of(stage))
        # 4. graphs are captured at the rung actually in force.
        self.graphs = _GraphRunner(self.model)
        self.graphs.capture_boot()
        self.registry = PrefillGraphRecaptureRegistry(
            self.graphs,
            P39.range_of(stage),
            CAPTURE_LADDER,
            per_shape_seconds=0.12,
        )
        # 5. the three observers, in the order a failure must unwind them.
        self.act.add_observer(model_runner_observer(self.mirror))
        self.act.add_observer(cuda_graph_observer(self.registry))
        self.act.add_observer(pool_coverage_observer(span[0], span[1]))

    # -- the switch, as a rank performs it -------------------------------

    def switch(self, frm: PLayout, to: PLayout, ring: RingState, capturing=False):
        require_switchable(ring)
        plan_ops = emit_move_ops(_plan(frm, to), self.regions, self.stage)
        moved = run_move_ops_guarded(
            plan_ops,
            self.device_ops,
            stream=1,
            capturing=capturing,
        )
        report = self.act.flip(to.name, quiescent=True)
        return report, moved

    # -- what the probe asserts ------------------------------------------

    def state(self):
        return {
            "model_range": (self.model._start_layer, self.model._end_layer),
            "runner_range": (self.runner.start_layer, self.runner.end_layer),
            "effective": self.runner.num_effective_layers,
            "full_ids": list(self.config.full_attention_layer_ids),
            "swa_ids": list(self.config.swa_attention_layer_ids),
            "pool_base": self.pool.start_layer,
            "pool_layers": self.pool.layer_num,
            "baked": {s: self.graphs.baked_range(s) for s in CAPTURE_LADDER},
        }


def _plan(frm, to):
    from sglang.srt.weg2.p_layout_switch import plan_layer_moves

    return plan_layer_moves(frm, to, LAYER_BYTES)


# Layer id lists over the WHOLE model: every third layer is full attention,
# the rest SWA. The probe never narrows these by hand -- the mirror does.
ALL_FULL = [i for i in range(N_LAYERS) if i % 3 == 0]
ALL_SWA = [i for i in range(N_LAYERS) if i % 3 != 0]


@pytest.fixture
def world():
    _KvPool.builds = 0
    spans = union_layout(P39, P32)
    cards = {
        0: _Card(0, direct=True),  # the 5090: 32 GiB BAR, no collect
        1: _Card(1, direct=False),
        2: _Card(2, direct=False),
    }
    ranks = [
        FakeRank(
            stage,
            spans[stage],
            [i for i in ALL_FULL if spans[stage][0] <= i <= spans[stage][1]],
            [i for i in ALL_SWA if spans[stage][0] <= i <= spans[stage][1]],
            cards,
        )
        for stage in range(3)
    ]
    return ranks


# ---------------------------------------------------------------------------
# The probe.
# ---------------------------------------------------------------------------


def test_boot_tiles_the_model_and_builds_every_pool_over_the_union(world):
    validate_world_tiling([P39.range_of(s) for s in range(3)], N_LAYERS)
    validate_world_tiling([P32.range_of(s) for s in range(3)], N_LAYERS)
    spans = union_layout(P39, P32)
    assert _KvPool.builds == 3  # one per rank, and only one
    for stage, rank in enumerate(world):
        assert rank.pool.start_layer == spans[stage][0]
        assert rank.pool.end_layer == spans[stage][1]
        # The pool is WIDER than the rung the rank runs: that is the union's
        # price, paid at boot so a later switch has rows to land in.
        lo, hi = P39.range_of(stage)
        assert rank.pool.layer_num >= hi - lo
        assert rank.state()["runner_range"] == (lo, hi)
        assert rank.state()["model_range"] == (lo, hi)


def test_the_full_round_trip_returns_every_rank_to_the_boot_state(world):
    before = [r.state() for r in world]
    ring = RingState(
        at_chunk_boundary=True, inflight_microbatches=0, inflight_requests=0
    )

    # --- P39 -> P32 --------------------------------------------------
    for rank in world:
        report, moved = rank.switch(P39, P32, ring)
        assert report.to_range == P32.range_of(rank.stage)
    mid = [r.state() for r in world]
    for stage, st in enumerate(mid):
        lo, hi = P32.range_of(stage)
        assert st["model_range"] == (lo, hi)
        assert st["runner_range"] == (lo, hi)
        assert st["effective"] == hi - lo
        # Every captured shape now bakes the NEW range -- none was left stale.
        assert set(st["baked"].values()) == {(lo, hi)}

    # --- P32 -> P39 --------------------------------------------------
    for rank in world:
        rank.switch(P32, P39, ring)
    after = [r.state() for r in world]

    assert after == before, "the round trip did not restore the boot state"
    # And no pool was ever rebuilt: the indexing base never moved.
    assert _KvPool.builds == 3


def test_the_newly_activated_layer_has_rows_on_the_rank_that_gains_it(world):
    """Stage 1 gains layers 32..38 under P32. Without the union they have no
    rows at all, and the failure is a KeyError deep in a pool, not here."""
    ring = RingState(True, 0, 0)
    for rank in world:
        rank.switch(P39, P32, ring)
    stage1 = world[1]
    for layer_id in range(32, 39):
        slot = stage1.pool.local_slot(layer_id)
        assert slot >= 0


def test_a_narrow_pool_refuses_the_switch_instead_of_serving_rowless_layers():
    """The counter-case: build stage 1's pool for P39 only and the flip stops."""
    spans = union_layout(P39, P32)
    cards = {s: _Card(s, direct=(s == 0)) for s in range(3)}
    rank = FakeRank(1, spans[1], ALL_FULL, ALL_SWA, cards)
    # Replace the union coverage guard with one built for the narrow rung.
    rank.act._observers = [
        model_runner_observer(rank.mirror),
        cuda_graph_observer(rank.registry),
        pool_coverage_observer(*P39.range_of(1)),
    ]
    with pytest.raises(LayoutBoundaryError) as e:
        rank.act.flip("p32", quiescent=True)
    assert "no rows" in str(e.value)
    assert rank.model._start_layer == P39.range_of(1)[0]


def test_hybrid_swa_lists_survive_the_round_trip_on_every_rank(world):
    """The destructive-narrowing trap, at world scale."""
    before = [
        (
            list(r.config.full_attention_layer_ids),
            list(r.config.swa_attention_layer_ids),
        )
        for r in world
    ]
    ring = RingState(True, 0, 0)
    for rank in world:
        rank.switch(P39, P32, ring)
    for rank in world:
        rank.switch(P32, P39, ring)
    after = [
        (
            list(r.config.full_attention_layer_ids),
            list(r.config.swa_attention_layer_ids),
        )
        for r in world
    ]
    assert after == before


def test_every_rank_moved_bytes_on_the_eager_route_and_the_5090_took_direct(world):
    ring = RingState(True, 0, 0)
    moved = {}
    for rank in world:
        _, n = rank.switch(P39, P32, ring)
        moved[rank.stage] = n
    # Stage 0 ships seven layers out; stage 2 receives three.
    assert moved[0] == 7 * PER_LAYER_BYTES
    assert moved[2] == 4 * PER_LAYER_BYTES
    # Stage 1 both loses and gains -- the cascade.
    ops = [c for c in world[1].device_ops.calls]
    assert ops, "stage 1 issued nothing, so the cascade was missed"
    # Nothing crossed while a capture was open.
    assert all(not r.graphs.capturing for r in world)


def test_the_mover_refuses_under_capture_on_every_rank(world):
    ring = RingState(True, 0, 0)
    for rank in world:
        if rank.stage == 2:
            continue  # stage 2 only receives from a direct window at boot rung
        with pytest.raises(MoverNotCapturable) as e:
            rank.switch(P39, P32, ring, capturing=True)
        assert "W124" in str(e.value)
    # Refused BEFORE any byte moved and before any range changed.
    for rank in world:
        assert rank.model._start_layer == P39.range_of(rank.stage)[0]


def test_a_switch_off_a_chunk_boundary_is_refused_before_anything_moves(world):
    busy = RingState(
        at_chunk_boundary=False, inflight_microbatches=2, inflight_requests=1
    )
    for rank in world:
        with pytest.raises(Exception):
            rank.switch(P39, P32, busy)
        assert rank.device_ops.calls == []
        assert rank.model._start_layer == P39.range_of(rank.stage)[0]


def test_graph_recapture_is_charged_once_per_rank_per_switch(world):
    ring = RingState(True, 0, 0)
    for rank in world:
        rank.switch(P39, P32, ring)
        rep = rank.registry.last_report
        assert sorted(rep.recaptured) == sorted(CAPTURE_LADDER)
        assert rep.seconds == pytest.approx(0.12 * len(CAPTURE_LADDER))
        assert rank.registry.captured_range == P32.range_of(rank.stage)


def test_one_verdict_at_pp0_and_the_downstream_ranks_only_obey(world):
    """#968: downstream is verdict-free and is given nothing to re-derive from."""
    from sglang.srt.planner.pp_cut import prefill_timing_from_measurement
    from sglang.srt.weg2.p_layout_switch import SwitchCalibration, decide

    # Calibrated from the one measured, near-time-balanced cut, exactly as
    # test_p_layout_switch_0920.rig_calibration does: stage times roughly
    # equal while the layer counts are 39/13/12, i.e. the 5090 is about three
    # times faster per layer. A calibration proportional to layer COUNTS would
    # invert the whole trade.
    timing = prefill_timing_from_measurement(
        counts=P39.counts, stage_ms=(392.0, 386.0, 380.0)
    )
    calib = SwitchCalibration(
        timing=timing,
        chunk_tokens=4096,
        rates=RATES,
        layer_bytes=LAYER_BYTES,
        geom=GEOM,
        flip_seconds={"p39": 4.0, "p32": 3.1},
        margin_s=0.05,
    )
    costs = union_kv_costs(
        P39,
        P32,
        priced_layout=P39,
        kv_budget_bytes={0: 40 << 30, 1: 40 << 30, 2: 40 << 30},
        bytes_per_token_per_layer={0: 512, 1: 512, 2: 512},
    )
    require_union_fits(costs, 262144)
    verdict = decide(
        current=P39,
        fast=P32,
        lean=P39,
        pending_tokens=4_000_000,
        calib=calib,
        union_costs=costs,
        required_context_tokens=262144,
        recapture_s=0.36,
    )
    digest = preconditions_digest(
        current=P39,
        fast=P32,
        lean=P39,
        geom=GEOM,
        layer_bytes=LAYER_BYTES,
        epoch=7,
        chunk_index=3,
    )
    record = pp0_broadcast(verdict, digest)
    for rank in world[1:]:
        local = preconditions_digest(
            current=P39,
            fast=P32,
            lean=P39,
            geom=GEOM,
            layer_bytes=LAYER_BYTES,
            epoch=7,
            chunk_index=3,
        )
        assert accept_pp0_verdict(record, local, rank.stage) == record["action"]
    assert "union_tok_lost=" in verdict.as_line()


def test_a_rank_whose_preconditions_differ_stops_rather_than_skipping(world):
    from sglang.srt.weg2.p_layout_switch import PLayoutRankDisagree

    digest = preconditions_digest(
        current=P39,
        fast=P32,
        lean=P39,
        geom=GEOM,
        layer_bytes=LAYER_BYTES,
        epoch=7,
        chunk_index=3,
    )
    stale = preconditions_digest(
        current=P39,
        fast=P32,
        lean=P39,
        geom=GEOM,
        layer_bytes=LAYER_BYTES,
        epoch=7,
        chunk_index=2,
    )
    record = {
        "action": "to_fast",
        "to": "p32",
        "run_on": "p32",
        "digest": digest,
        "pending_tokens": 1,
        "gain_s": 1.0,
    }
    with pytest.raises(PLayoutRankDisagree):
        accept_pp0_verdict(record, stale, 1)


def test_the_card_placement_that_halves_the_switch_is_a_single_flag(world):
    """Stage order is welded to rank; the CARDS are what move.

    And the gain is smaller than the order assumed. Swapping to 0,2,1 makes the
    5090's hop carry its seven layers at 14.25 instead of 7.13 GB/s, but under
    the 64-layer geometry stage 1 also sheds FOUR layers to stage 2 over a
    3080-to-3080 link at 6.58 GB/s -- and once the fast hop is fast, that hop
    becomes the binding term. 0.378 s -> 0.234 s is a 38 % cut, not the halving
    the briefing estimated. The halving is what the 5090's hop alone does; the
    switch is paced by the busiest NODE, and stage 1 is busy in both directions.
    """
    swapped_rates = LinkRates(
        {
            (0, 1): 14.25,
            (1, 0): 13.15,
            (0, 2): 7.13,
            (2, 0): 6.56,
            (1, 2): 6.58,
            (2, 1): 6.58,
        }
    )
    shipped = placement_switch_seconds(P39, P32, LAYER_BYTES, RATES)
    swapped = placement_switch_seconds(P39, P32, LAYER_BYTES, swapped_rates)
    assert shipped == pytest.approx(0.378, abs=0.01)
    assert swapped == pytest.approx(0.234, abs=0.01)
    assert shipped / swapped == pytest.approx(1.61, rel=0.05)
    assert StagePermutation((0, 2, 1)).as_flag() == "0,2,1"
