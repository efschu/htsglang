"""The four things that stop a P-layout switch from working IN THE PROCESS.

The morning's module (``weg2/p_layout_switch.py``) decides WHETHER to switch
and what it costs. It could not be booted, because four pieces of the running
process do not follow a boundary change. Each of the four fails QUIETLY -- none
of them raises, all of them produce plausible output from the wrong layer set
-- which is why each gets a test that proves the repair rather than a note that
promises it.

1. ``ModelRunner`` snapshots the range at init and derives three further things
   from the snapshot (``model_runner.py:1185-1212``, ``:1729-1745``).
2. Captured CUDA graphs bake the executed layer set
   (``prefill_cuda_graph_runner.py:512`` -> ``models/qwen3_5.py:1720``).
3. The KV pools are built for one layout's layer count, so a layer a switch
   activates has no rows.
4. The BAR1 mover's recv half is not graph-capturable
   (``barlink_bar1_p2p.py:181``).

Hermetic: no CUDA, no model, no pool. Every collaborator is a stub that records
what was asked of it.
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
)
from sglang.srt.model_executor.runner.graph_recapture import (
    GraphRecaptureError,
    PrefillGraphRecaptureRegistry,
    no_recapture_registry,
    recapture_cost_seconds,
)
from sglang.srt.weg2.p_layout_switch import (
    RANK_INDEXED_VECTORS,
    LayerBytes,
    LinkRates,
    MoverNotCapturable,
    PLayout,
    StagePermutation,
    StagePermutationIncomplete,
    UnionKvCost,
    UnionKvTooCostly,
    placement_switch_seconds,
    plan_layer_moves,
    require_breakable_route,
    require_union_fits,
    run_move_ops_guarded,
    slowest_hop_seconds,
    union_kv_costs,
    union_layout,
)

# ---------------------------------------------------------------------------
# Stubs.
# ---------------------------------------------------------------------------


class _Param:
    pass


class _RealLayer:
    def parameters(self):
        yield _Param()


class _MissingLayer:
    def parameters(self):
        return iter(())


class _Model:
    """qwen3_5's shape: properties over mutable backing fields."""

    def __init__(self, n_layers: int, start: int, end: int, real=None):
        self._start_layer = start
        self._end_layer = end
        real = range(start, end) if real is None else real
        self.layers = [
            _RealLayer() if i in real else _MissingLayer() for i in range(n_layers)
        ]

    @property
    def start_layer(self):
        return self._start_layer

    @property
    def end_layer(self):
        return self._end_layer


class _HfConfig:
    def __init__(self, arch="Qwen3NextForCausalLM", loop_num=1):
        self.architectures = [arch]
        self.loop_num = loop_num


class _ModelConfig:
    def __init__(self, n_layers, full_ids=None, swa_ids=None, loop_num=1):
        self.num_hidden_layers = n_layers
        self.num_attention_layers = n_layers
        self.hf_config = _HfConfig(loop_num=loop_num)
        self.is_deepseek_v4_arch = False
        if full_ids is not None:
            self.full_attention_layer_ids = list(full_ids)
        if swa_ids is not None:
            self.swa_attention_layer_ids = list(swa_ids)


class _Runner:
    """Only what the mirror touches."""

    def __init__(self, model_config, pp_rank=0, pp_size=3, is_hybrid_swa=False):
        self.model_config = model_config
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.is_hybrid_swa = is_hybrid_swa
        self.start_layer = None
        self.end_layer = None
        self.num_effective_layers = None


# ---------------------------------------------------------------------------
# Blocker 1: the ModelRunner's derived state follows the boundary.
# ---------------------------------------------------------------------------


def _mirror_at_union(union=(0, 39), n_layers=48, **kw):
    cfg = _ModelConfig(n_layers, **kw)
    runner = _Runner(cfg, is_hybrid_swa=kw.pop("is_hybrid_swa", False))
    runner.start_layer, runner.end_layer = union
    runner.num_effective_layers = union[1] - union[0]
    return runner, ModelRunnerRangeMirror(runner, union)


def test_mirror_moves_start_end_and_effective_count():
    runner, mirror = _mirror_at_union((0, 39))
    out = mirror.apply((0, 32))
    assert (runner.start_layer, runner.end_layer) == (0, 32)
    assert runner.num_effective_layers == 32
    assert out["num_effective_layers"] == 32
    # ...and back, which is the half a one-way design would pass and a boot
    # would fail on the second switch.
    mirror.apply((0, 39))
    assert (runner.start_layer, runner.end_layer, runner.num_effective_layers) == (
        0,
        39,
        39,
    )


def test_mirror_refuses_construction_outside_the_union():
    """The pristine lists are only pristine at the union."""
    cfg = _ModelConfig(48)
    runner = _Runner(cfg)
    runner.start_layer, runner.end_layer = 0, 32
    runner.num_effective_layers = 32
    with pytest.raises(LayoutBoundaryError) as e:
        ModelRunnerRangeMirror(runner, (0, 39))
    assert "UNION" in str(e.value)
    assert "union_layer_window" in str(e.value)


def test_mirror_refuses_a_range_outside_its_union():
    _, mirror = _mirror_at_union((0, 39))
    with pytest.raises(LayoutBoundaryError) as e:
        mirror.apply((0, 41))
    assert "leaves the union" in str(e.value)


def test_hybrid_swa_lists_rederive_from_pristine_not_from_the_narrowed_value():
    """The bug this test exists for: the init narrowing is DESTRUCTIVE.

    ``adjust_hybrid_swa_layers_for_pp`` assigns the filtered list back over the
    attribute it filtered, so narrowing to [0,32) and then widening to [0,39)
    by re-running it would intersect with the already-narrowed list and never
    get layers 32..38 back. Silently: a full-attention layer quietly demoted to
    SWA produces plausible output.
    """
    full = [0, 5, 10, 20, 33, 36, 38]
    swa = [1, 2, 3, 30, 34, 37]
    cfg = _ModelConfig(48, full_ids=full, swa_ids=swa)
    runner = _Runner(cfg, is_hybrid_swa=True)
    runner.start_layer, runner.end_layer = 0, 39
    runner.num_effective_layers = 39
    mirror = ModelRunnerRangeMirror(runner, (0, 39))

    mirror.apply((0, 32))
    assert cfg.full_attention_layer_ids == [0, 5, 10, 20]
    assert cfg.swa_attention_layer_ids == [1, 2, 3, 30]

    mirror.apply((0, 39))
    # Recovered in full -- the property a re-run of the init helper loses.
    assert cfg.full_attention_layer_ids == full
    assert cfg.swa_attention_layer_ids == swa
    assert mirror.pristine_layer_ids["full_attention_layer_ids"] == full


def test_hybrid_swa_reproduces_the_inclusive_end_bound():
    """``range(start, end + 1)`` at model_runner.py:1734 is reproduced, not fixed.

    A mirror that quietly corrected the off-by-one would make the FIRST flip
    change the layer id lists for a reason unrelated to the boundary.
    """
    cfg = _ModelConfig(48, full_ids=[31, 32, 33], swa_ids=[])
    runner = _Runner(cfg, is_hybrid_swa=True)
    runner.start_layer, runner.end_layer = 0, 39
    runner.num_effective_layers = 39
    mirror = ModelRunnerRangeMirror(runner, (0, 39))
    mirror.apply((0, 32))
    # 32 is included because the bound is end_layer + 1. Exclusive would drop it.
    assert cfg.full_attention_layer_ids == [31, 32]


def test_non_hybrid_model_leaves_the_lists_alone():
    cfg = _ModelConfig(48, full_ids=[1, 40], swa_ids=[2])
    runner = _Runner(cfg, is_hybrid_swa=False)
    runner.start_layer, runner.end_layer = 0, 39
    runner.num_effective_layers = 39
    mirror = ModelRunnerRangeMirror(runner, (0, 39))
    mirror.apply((0, 32))
    assert cfg.full_attention_layer_ids == [1, 40]


def test_loop_num_multiplies_the_effective_count():
    cfg = _ModelConfig(48, loop_num=3)
    runner = _Runner(cfg)
    runner.start_layer, runner.end_layer = 0, 39
    runner.num_effective_layers = 39
    mirror = ModelRunnerRangeMirror(runner, (0, 39))
    mirror.apply((0, 32))
    assert runner.num_effective_layers == 96


def test_pp_layer_set_env_is_a_refusal_not_a_recount(monkeypatch):
    """The env names one fixed set per stage; a moved range contradicts it."""
    monkeypatch.setenv("SGLANG_PP_LAYER_SET", "0-38;39-43;44-47")
    cfg = _ModelConfig(48)
    runner = _Runner(cfg, pp_rank=0, pp_size=3)
    runner.start_layer, runner.end_layer = 0, 39
    runner.num_effective_layers = 39
    mirror = ModelRunnerRangeMirror(runner, (0, 39))
    with pytest.raises(LayoutBoundaryError) as e:
        mirror.apply((0, 32))
    assert "SGLANG_PP_LAYER_SET" in str(e.value)


def test_union_layer_window_widens_every_reader_and_restores():
    cfg = _ModelConfig(48)
    runner = _Runner(cfg)
    runner.start_layer, runner.end_layer, runner.num_effective_layers = 0, 32, 32
    with union_layer_window(runner, (0, 39)) as span:
        assert span == (0, 39)
        assert (runner.start_layer, runner.end_layer) == (0, 39)
        assert runner.num_effective_layers == 39
    assert (runner.start_layer, runner.end_layer, runner.num_effective_layers) == (
        0,
        32,
        32,
    )


def test_union_layer_window_restores_even_when_the_build_raises():
    cfg = _ModelConfig(48)
    runner = _Runner(cfg)
    runner.start_layer, runner.end_layer, runner.num_effective_layers = 0, 32, 32
    with pytest.raises(ValueError):
        with union_layer_window(runner, (0, 39)):
            raise ValueError("pool build blew up")
    assert (runner.start_layer, runner.end_layer) == (0, 32)


def test_observer_form_rolls_the_flip_back_when_the_runner_cannot_follow(monkeypatch):
    """An observer raise must leave model and runner agreeing, not disagreeing."""
    monkeypatch.setenv("SGLANG_PP_LAYER_SET", "0-38;39-43;44-47")
    cfg = _ModelConfig(48)
    runner = _Runner(cfg, pp_rank=0, pp_size=3)
    runner.start_layer, runner.end_layer, runner.num_effective_layers = 0, 39, 39
    mirror = ModelRunnerRangeMirror(runner, (0, 39))
    model = _Model(48, 0, 39)
    act = LayoutBoundaryActuator(model, {"p39": (0, 39), "p32": (0, 32)}, "p39")
    act.add_observer(model_runner_observer(mirror))
    with pytest.raises(LayoutBoundaryError):
        act.flip("p32", quiescent=True)
    assert (model._start_layer, model._end_layer) == (0, 39)
    assert (runner.start_layer, runner.end_layer) == (0, 39)


# ---------------------------------------------------------------------------
# Blocker 2: captured graphs follow, or stop being replayable.
# ---------------------------------------------------------------------------


class _Backend:
    def __init__(self, shapes):
        self._graphs = {int(s): f"graph@{s}" for s in shapes}
        self.discarded = []

    def discard_shape(self, shape_key):
        size = int(getattr(shape_key, "size", shape_key))
        self.discarded.append(size)
        return self._graphs.pop(size, None) is not None

    def can_run(self, size):
        return int(size) in self._graphs


class _GraphRunner:
    def __init__(self, shapes):
        self.backend = _Backend(shapes)
        self.recaptured = []
        self.range_at_capture = []
        self.model = None
        self.fail = False

    def recapture_shapes(self, sizes):
        if self.fail:
            raise RuntimeError("capture OOM")
        for s in sizes:
            self.backend._graphs[int(s)] = f"graph@{s}"
            self.recaptured.append(int(s))
            if self.model is not None:
                self.range_at_capture.append(
                    (self.model._start_layer, self.model._end_layer)
                )
        return tuple(int(s) for s in sizes)


def test_recapture_rerecords_every_captured_shape_by_default():
    runner = _GraphRunner([512, 2048, 8192])
    reg = PrefillGraphRecaptureRegistry(runner, (0, 39), [512, 2048, 8192])
    report = reg.recapture((0, 32))
    assert report.frm_range == (0, 39) and report.to_range == (0, 32)
    assert sorted(report.recaptured) == [512, 2048, 8192]
    assert report.discarded == ()
    assert reg.captured_range == (0, 32)
    assert sorted(runner.backend._graphs) == [512, 2048, 8192]


def test_cold_shapes_are_discarded_not_left_replayable():
    """The whole point: no shape survives under the OLD range."""
    runner = _GraphRunner([512, 2048, 8192])
    reg = PrefillGraphRecaptureRegistry(
        runner, (0, 39), [512, 2048, 8192], hot_shapes=[8192]
    )
    report = reg.recapture((0, 32))
    assert report.recaptured == (8192,)
    assert sorted(report.discarded) == [512, 2048]
    # The cold ones are GONE, so can_run answers False and they go eager.
    assert not runner.backend.can_run(512)
    assert runner.backend.can_run(8192)


def test_recapture_happens_under_the_new_range():
    """A recapture that ran before the range moved would re-bake the old one."""
    runner = _GraphRunner([1024])
    model = _Model(48, 0, 39)
    runner.model = model
    reg = PrefillGraphRecaptureRegistry(runner, (0, 39), [1024])
    act = LayoutBoundaryActuator(model, {"p39": (0, 39), "p32": (0, 32)}, "p39")
    act.add_observer(cuda_graph_observer(reg))
    act.flip("p32", quiescent=True)
    assert runner.range_at_capture == [(0, 32)]


def test_a_failed_recapture_leaves_nothing_replayable():
    runner = _GraphRunner([512, 2048])
    runner.fail = True
    reg = PrefillGraphRecaptureRegistry(runner, (0, 39), [512, 2048])
    with pytest.raises(GraphRecaptureError) as e:
        reg.recapture((0, 32))
    assert "eager" in str(e.value)
    assert runner.backend._graphs == {}


def test_backend_without_discard_shape_is_refused_by_name():
    class _Bare:
        pass

    class _R:
        backend = _Bare()

        def recapture_shapes(self, sizes):
            return tuple(sizes)

    reg = PrefillGraphRecaptureRegistry(_R(), (0, 39), [512])
    with pytest.raises(GraphRecaptureError) as e:
        reg.recapture((0, 32))
    assert "discard_shape" in str(e.value)


def test_recapture_cost_refuses_to_be_guessed():
    runner = _GraphRunner([512, 2048])
    reg = PrefillGraphRecaptureRegistry(runner, (0, 39), [512, 2048])
    with pytest.raises(GraphRecaptureError) as e:
        reg.cost_seconds()
    assert "per_shape_seconds" in str(e.value)
    assert reg.cost_seconds_or_none() is None


def test_recapture_cost_is_per_hot_shape_when_measured():
    runner = _GraphRunner([512, 2048, 8192])
    reg = PrefillGraphRecaptureRegistry(
        runner,
        (0, 39),
        [512, 2048, 8192],
        per_shape_seconds=0.4,
        hot_shapes=[2048, 8192],
    )
    assert reg.cost_seconds() == pytest.approx(0.8)
    assert recapture_cost_seconds(reg, (0, 39)) == 0.0
    assert recapture_cost_seconds(reg, (0, 32)) == pytest.approx(0.8)


def test_hot_shape_that_was_never_captured_is_refused():
    runner = _GraphRunner([512])
    with pytest.raises(GraphRecaptureError) as e:
        PrefillGraphRecaptureRegistry(runner, (0, 39), [512], hot_shapes=[4096])
    assert "never captured" in str(e.value)


def test_no_recapture_registry_makes_the_observer_refuse_the_flip():
    model = _Model(48, 0, 39)
    act = LayoutBoundaryActuator(model, {"p39": (0, 39), "p32": (0, 32)}, "p39")
    act.add_observer(cuda_graph_observer(no_recapture_registry((0, 39))))
    with pytest.raises(LayoutBoundaryError) as e:
        act.flip("p32", quiescent=True)
    assert "recapture()" in str(e.value)
    assert (model._start_layer, model._end_layer) == (0, 39)


# ---------------------------------------------------------------------------
# Blocker 3: the pools cover the union, and the union is priced.
# ---------------------------------------------------------------------------

P39 = PLayout(name="p39", counts=(39, 5, 4))
P32 = PLayout(name="p32", counts=(32, 9, 7))


def test_union_span_per_stage_covers_both_layouts():
    spans = union_layout(P39, P32)
    assert spans == ((0, 39), (32, 44), (41, 48))
    # Stage 1 must hold 12 layers to run either 5 or 9 of them: that is the
    # union's price, and it is why it gets a number below.
    assert spans[1][1] - spans[1][0] == 12


def test_pool_coverage_observer_passes_over_the_union_and_refuses_below_it():
    ok = pool_coverage_observer(0, 39)
    from sglang.srt.model_executor.layout_boundary import BoundaryFlipReport

    rep = BoundaryFlipReport("p39", "p32", (0, 39), (0, 32), (), (), 0, "")
    ok(rep)
    narrow = pool_coverage_observer(0, 32)
    rep_back = BoundaryFlipReport("p32", "p39", (0, 32), (0, 39), (), (), 0, "")
    with pytest.raises(LayoutBoundaryError) as e:
        narrow(rep_back)
    assert "no rows" in str(e.value)


def test_union_kv_cost_counts_the_tokens_the_union_takes():
    cost = UnionKvCost(
        stage=1,
        kv_budget_bytes=12 * 1024**3,
        bytes_per_token_per_layer=8192,
        layout_layers=5,
        union_layers=12,
    )
    assert cost.tokens_with_layout == (12 * 1024**3) // (8192 * 5)
    assert cost.tokens_with_union == (12 * 1024**3) // (8192 * 12)
    assert cost.tokens_lost > 0
    assert "costs" in cost.as_line()


def test_union_kv_costs_refuse_an_unpriced_stage():
    with pytest.raises(UnionKvTooCostly) as e:
        union_kv_costs(
            P39,
            P32,
            priced_layout=P39,
            kv_budget_bytes={0: 1 << 30},
            bytes_per_token_per_layer={0: 4096},
        )
    assert "wrong card" in str(e.value)


def test_union_smaller_than_the_layout_is_a_wrong_union_not_a_bargain():
    with pytest.raises(UnionKvTooCostly):
        UnionKvCost(
            stage=0,
            kv_budget_bytes=1 << 30,
            bytes_per_token_per_layer=4096,
            layout_layers=39,
            union_layers=32,
        )


def test_require_union_fits_refuses_the_PAIR_when_262k_no_longer_fits():
    budgets = {0: 20 << 30, 1: 5 << 30, 2: 5 << 30}
    per_layer = {0: 2048, 1: 2048, 2: 2048}
    costs = union_kv_costs(
        P39,
        P32,
        priced_layout=P39,
        kv_budget_bytes=budgets,
        bytes_per_token_per_layer=per_layer,
    )
    with pytest.raises(UnionKvTooCostly) as e:
        require_union_fits(costs, 262144)
    msg = str(e.value)
    assert "W123" in msg and "layout PAIR" in msg


def test_require_union_fits_passes_with_room_and_returns_the_numbers():
    budgets = {0: 40 << 30, 1: 40 << 30, 2: 40 << 30}
    per_layer = {0: 512, 1: 512, 2: 512}
    costs = union_kv_costs(
        P39,
        P32,
        priced_layout=P39,
        kv_budget_bytes=budgets,
        bytes_per_token_per_layer=per_layer,
    )
    out = require_union_fits(costs, 262144)
    assert len(out) == 3
    assert all(c.tokens_with_union >= 262144 for c in out)


# ---------------------------------------------------------------------------
# Blocker 4: the mover runs eager, and says so by name under capture.
# ---------------------------------------------------------------------------


class _DeviceOps:
    def __init__(self):
        self.calls = []

    def memcpy_async(self, dst, src, nbytes, stream):
        self.calls.append((dst, src, nbytes, stream))


def _op(n=1024):
    from sglang.srt.weg2.p_layout_switch import MoveOp

    return MoveOp(kind="deposit", layer_id=3, dst=2, src=1, nbytes=n, peer_stage=1)


def test_mover_runs_on_the_eager_route():
    ops = (_op(),)
    dev = _DeviceOps()
    moved = run_move_ops_guarded(ops, dev, stream=7, capturing=False)
    assert moved == 1024
    assert len(dev.calls) == 1


def test_mover_refuses_under_capture_by_name_before_issuing_anything():
    """The danger is the HALF that records: deposits without collects."""
    dev = _DeviceOps()
    with pytest.raises(MoverNotCapturable) as e:
        run_move_ops_guarded((_op(), _op()), dev, stream=7, capturing=True)
    msg = str(e.value)
    assert "W124" in msg
    assert "never collects" in msg
    # Not one byte was issued before the refusal.
    assert dev.calls == []


def test_route_gate_quotes_the_live_bar1_capture_safety_reason():
    """The reason is read from barlink, not restated here (and re-stating it is
    how the two drift)."""
    from sglang.srt.distributed.device_communicators.barlink_bar1_p2p import (
        capture_safety,
    )

    live = capture_safety()
    assert live["recv_capturable"] is False
    with pytest.raises(MoverNotCapturable) as e:
        require_breakable_route(capturing=True, route="breakable", n_ops=1)
    assert live["reason"][:40] in str(e.value)


def test_route_gate_passes_the_route_name_through_when_eager():
    assert require_breakable_route(capturing=False, route="breakable") == "breakable"


# ---------------------------------------------------------------------------
# Stage order: the 5090's neighbour.
# ---------------------------------------------------------------------------

# PLAN_BAR1_LANES_0918, ordered pairs. Stage indices under the SHIPPED
# placement 0,1,2: stage 0 = 5090, stage 1 = 3080 on x4, stage 2 = 3080 on x8.
_SHIPPED_RATES = LinkRates(
    {
        (0, 1): 7.13,
        (1, 0): 6.56,
        (0, 2): 14.25,
        (2, 0): 13.15,
        (1, 2): 6.58,
        (2, 1): 6.58,
    }
)
# Under 0,2,1 the cards swap, so stage 1 is now the x8 3080 and stage 2 the x4.
_SWAPPED_RATES = LinkRates(
    {
        (0, 1): 14.25,
        (1, 0): 13.15,
        (0, 2): 7.13,
        (2, 0): 6.56,
        (1, 2): 6.58,
        (2, 1): 6.58,
    }
)


def test_stage_order_is_expressed_as_a_card_permutation():
    assert StagePermutation((0, 1, 2)).is_identity
    swap = StagePermutation((0, 2, 1))
    assert not swap.is_identity
    assert swap.as_flag() == "0,2,1"
    assert swap.neighbour_pairs() == ((0, 1), (1, 2))


def test_a_permutation_that_is_not_a_permutation_is_refused():
    with pytest.raises(StagePermutationIncomplete):
        StagePermutation((0, 1, 1))


def test_half_applied_permutation_is_refused_by_the_missing_vector():
    swap = StagePermutation((0, 2, 1))
    with pytest.raises(StagePermutationIncomplete) as e:
        swap.apply({"rank_gpu_id": (0, 1, 2), "pp_stage_ratio": (39, 5, 4)})
    msg = str(e.value)
    assert "kv_tokvec" in msg and "vram_reserve_mib" in msg


def test_full_permutation_moves_every_rank_indexed_vector_together():
    swap = StagePermutation((0, 2, 1))
    out = swap.apply(
        {
            "rank_gpu_id": (0, 1, 2),
            "rank_gpu_memory_mib": (30000, 15000, 15000),
            "pp_stage_ratio": (39, 5, 4),
            "kv_tokvec": (33, 13, 18),
            "vram_reserve_mib": (3000, 2200, 2200),
        }
    )
    assert set(out) == set(RANK_INDEXED_VECTORS)
    # The 5090's own numbers stay on stage 0; the two 3080s trade places.
    assert out["kv_tokvec"] == (33, 18, 13)
    assert out["vram_reserve_mib"] == (3000, 2200, 2200)
    assert out["rank_gpu_id"] == (0, 2, 1)


def test_identity_permutation_needs_no_vectors_at_all():
    assert StagePermutation((0, 1, 2)).apply({}) == {}


def test_swapping_the_cards_does_NOT_speed_up_the_slowest_link():
    """The correction: the worst LINK is unchanged by the swap.

    0,2,1 makes the 5090's hop fast (7.13 -> 14.25 GB/s) but the pipeline then
    has a 3080-to-3080 hop at 6.58 GB/s, which becomes the worst one. A
    placement decision taken on "the slowest link" would therefore conclude the
    swap buys nothing -- and be wrong, for the reason the next test gives.
    """
    nbytes = 2_700_000_000
    _, shipped_s = slowest_hop_seconds(
        StagePermutation((0, 1, 2)), _SHIPPED_RATES, nbytes
    )
    swapped_pair, swapped_s = slowest_hop_seconds(
        StagePermutation((0, 2, 1)), _SWAPPED_RATES, nbytes
    )
    assert swapped_pair in ((1, 2), (2, 1))
    assert swapped_s == pytest.approx(shipped_s, rel=0.01)


def test_swapping_the_5090s_neighbour_halves_the_SWITCH():
    """The order's claim, priced on the bytes each hop actually carries.

    P39 -> P32 moves seven layers from stage 0 to stage 1 and three from stage
    1 to stage 2. The 5090's hop carries 7/10 of the bytes, so making it twice
    as fast roughly halves the switch even though the slowest link did not
    move: 0.41 s -> ~0.21 s.
    """
    per_layer = 385_000_000
    lb = LayerBytes({i: per_layer for i in range(48)})
    shipped = placement_switch_seconds(P39, P32, lb, _SHIPPED_RATES)
    swapped = placement_switch_seconds(P39, P32, lb, _SWAPPED_RATES)
    assert shipped == pytest.approx(0.378, abs=0.02)
    assert swapped == pytest.approx(0.19, abs=0.02)
    assert shipped / swapped == pytest.approx(2.0, rel=0.1)


def test_the_moves_the_switch_makes_are_the_cascade_not_one_boundary():
    """Stage 1 both GAINS and LOSES, which boundary arithmetic would miss."""
    per_layer = 385_000_000
    lb = LayerBytes({i: per_layer for i in range(48)})
    plan = plan_layer_moves(P39, P32, lb)
    assert sorted(plan.layers_gained_by(1)) == [32, 33, 34, 35, 36, 37, 38]
    assert sorted(plan.layers_lost_by(1)) == [41, 42, 43]
    assert plan.by_pair()[(0, 1)] == 7 * per_layer
    assert plan.by_pair()[(1, 2)] == 3 * per_layer
