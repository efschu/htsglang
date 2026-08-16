"""#704 slice 1a-ii: moving the PP layer boundary at runtime.

The union arena (slice 1a) established that consecutive rungs can share one
byte layout, so no weight moves. This file wires the other half: actually
changing which layers a rank EXECUTES, at runtime, with nothing copied.

The model makes this cheap, and the shape of the solution is dictated by three
facts verified in `models/qwen3_5.py`:

  * `make_layers` (`utils/common.py:1970-2010`) builds a ModuleList of length
    num_hidden_layers with `PPMissingLayer` placeholders outside the owned
    range, so layer indices are GLOBAL on every rank -- a boundary change is
    not an index shift;
  * `start_layer`/`end_layer` are PROPERTIES over mutable `_start_layer` /
    `_end_layer` backing fields (`qwen3_5.py:1452-1457`);
  * the decoder forward iterates `range(self.start_layer, self.end_layer)`
    (`qwen3_5.py:1483`), reading those properties on EVERY pass.

So a real layer module parked outside the active range is simply not executed.
The boundary change is a range mutation plus the dependent per-layer structures
-- no module swapping, no reallocation, and no weight bytes.

The pattern is **load wide, run narrow**: at boot a rank builds and loads real
modules for the UNION of the ranges it may occupy (weight loading is itself
gated on the same range at `qwen3_5.py:1563-1564` and `:1707-1708`, so the
union must be in force during load), and then runs whichever sub-range the
ladder selects.

Hermetic: fake modules on CPU, no CUDA, no server, no real checkpoint.
"""

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.model_executor.layout_boundary import (
    LayoutBoundaryActuator,
    LayoutBoundaryError,
    validate_world_tiling,
)

NUM_LAYERS = 64


class _RecordingLayer(torch.nn.Module):
    """A real layer that records that it ran, and owns a parameter."""

    def __init__(self, idx: int, log: list):
        super().__init__()
        self.idx = idx
        self._log = log
        self.weight = torch.nn.Parameter(torch.zeros(4))

    def forward(self, hidden_states, **_):
        self._log.append(self.idx)
        return hidden_states


class _Missing(torch.nn.Module):
    """Stand-in for PPMissingLayer: pass-through, no parameters."""

    def forward(self, hidden_states, **_):
        return hidden_states


class _FakeModel(torch.nn.Module):
    """Minimal stand-in for qwen3_5's decoder: global indices, range forward."""

    def __init__(self, union_start: int, union_end: int, log: list):
        super().__init__()
        self._log = log
        self.layers = torch.nn.ModuleList(
            [
                _RecordingLayer(i, log) if union_start <= i < union_end else _Missing()
                for i in range(NUM_LAYERS)
            ]
        )
        self._start_layer = union_start
        self._end_layer = union_end

    @property
    def start_layer(self) -> int:
        return self._start_layer

    @property
    def end_layer(self) -> int:
        return self._end_layer

    def forward(self, hidden_states):
        for i in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[i](hidden_states)
        return hidden_states


# Slice 1a pair on rank0: [28,20,16] -> [29,19,16]. Union range is [0,29).
RUNGS_RANK0 = {"[28,20,16]": (0, 28), "[29,19,16]": (0, 29)}


def _rank0(log=None):
    log = [] if log is None else log
    model = _FakeModel(0, 29, log)
    act = LayoutBoundaryActuator(
        model=model, rung_ranges=RUNGS_RANK0, current_rung="[28,20,16]"
    )
    return model, act, log


def test_the_active_range_selects_what_actually_runs():
    """The whole point: a resident-but-unowned layer must NOT execute."""
    model, act, log = _rank0()
    model(torch.zeros(4))
    assert log == list(range(28)), "rung A must run layers 0..27 only"
    assert isinstance(model.layers[28], _RecordingLayer), (
        "layer 28 must be RESIDENT under the union even while unowned"
    )

    log.clear()
    act.flip("[29,19,16]", quiescent=True)
    model(torch.zeros(4))
    assert log == list(range(29)), "rung B must run layers 0..28"


def test_a_flip_copies_no_bytes_and_moves_no_storage():
    """The runtime half of the zero-copy claim.

    Every parameter must occupy the SAME storage before and after. If a flip
    reallocated or copied, these pointers would move.
    """
    model, act, _ = _rank0()
    before = {
        i: model.layers[i].weight.data_ptr()
        for i in range(29)
        if isinstance(model.layers[i], _RecordingLayer)
    }
    report = act.flip("[29,19,16]", quiescent=True)
    after = {
        i: model.layers[i].weight.data_ptr()
        for i in range(29)
        if isinstance(model.layers[i], _RecordingLayer)
    }
    assert before == after
    assert report.bytes_copied == 0
    assert report.activated == (28,)
    assert report.deactivated == ()


def test_flip_back_restores_the_original_range_exactly():
    model, act, log = _rank0()
    act.flip("[29,19,16]", quiescent=True)
    back = act.flip("[28,20,16]", quiescent=True)
    assert (model.start_layer, model.end_layer) == (0, 28)
    assert back.deactivated == (28,)
    assert back.activated == ()
    log.clear()
    model(torch.zeros(4))
    assert log == list(range(28))


def test_a_flip_is_refused_when_not_quiescent():
    """GDN state travels with its layer; slice 1a moves none of it."""
    model, act, _ = _rank0()
    with pytest.raises(LayoutBoundaryError, match="quiescen"):
        act.flip("[29,19,16]", quiescent=False)
    assert (model.start_layer, model.end_layer) == (0, 28), "refusal must not mutate"


def test_a_target_outside_the_resident_union_is_refused():
    """The failure this actuator exists to prevent.

    Entering a range whose weights were never loaded would execute a
    PPMissingLayer as if it were a real layer -- silently wrong output rather
    than a crash, because PPMissingLayer is a pass-through.
    """
    model, act, _ = _rank0()
    act.rung_ranges["[31,17,16]"] = (0, 31)
    with pytest.raises(LayoutBoundaryError, match="not resident"):
        act.flip("[31,17,16]", quiescent=True)
    assert (model.start_layer, model.end_layer) == (0, 28)


def test_a_hole_in_the_resident_union_is_caught_at_construction():
    """A PPMissingLayer inside the claimed union is a boot-time defect."""
    log = []
    model = _FakeModel(0, 29, log)
    model.layers[17] = _Missing()  # simulate a layer that failed to load
    with pytest.raises(LayoutBoundaryError, match="not resident"):
        LayoutBoundaryActuator(
            model=model, rung_ranges=RUNGS_RANK0, current_rung="[28,20,16]"
        )


def test_an_unknown_rung_is_refused():
    _, act, _ = _rank0()
    with pytest.raises(LayoutBoundaryError, match="not a known rung"):
        act.flip("[44,10,10]", quiescent=True)


def test_flipping_to_the_current_rung_is_a_no_op_not_an_error():
    model, act, _ = _rank0()
    report = act.flip("[28,20,16]", quiescent=True)
    assert report.activated == () and report.deactivated == ()
    assert report.bytes_copied == 0
    assert (model.start_layer, model.end_layer) == (0, 28)


def test_the_world_ranges_must_tile_the_model_exactly():
    """A gap silently drops layers; an overlap computes them twice.

    Both produce wrong output rather than an error, because every rank's own
    range looks locally sensible. So the check has to be at world level.
    """
    validate_world_tiling([(0, 28), (28, 48), (48, 64)], NUM_LAYERS)
    with pytest.raises(LayoutBoundaryError, match="gap"):
        validate_world_tiling([(0, 28), (29, 48), (48, 64)], NUM_LAYERS)
    with pytest.raises(LayoutBoundaryError, match="overlap"):
        validate_world_tiling([(0, 29), (28, 48), (48, 64)], NUM_LAYERS)
    with pytest.raises(LayoutBoundaryError, match="cover"):
        validate_world_tiling([(0, 28), (28, 48), (48, 63)], NUM_LAYERS)


def test_the_slice_1a_pair_tiles_on_both_rungs():
    """[28,20,16] and [29,19,16] must each tile the 64 layers."""
    validate_world_tiling([(0, 28), (28, 48), (48, 64)], NUM_LAYERS)
    validate_world_tiling([(0, 29), (29, 48), (48, 64)], NUM_LAYERS)


def test_dependent_structures_are_reported_for_update():
    """A range change is not complete when the range changes.

    KV pool layer filters and GDN state maps are keyed by the owned range
    (model_runner_kv_cache_mixin.py:2466-2470). The actuator must SAY which
    layers changed hands so the caller updates them, rather than leaving a
    silent inconsistency.
    """
    _, act, _ = _rank0()
    report = act.flip("[29,19,16]", quiescent=True)
    assert report.activated == (28,)
    assert "kv" in report.requires_caller_update.lower()


def test_observers_are_notified_and_a_failing_observer_rolls_the_flip_back():
    """If a dependent structure cannot follow, the range must not move.

    A half-applied boundary -- new range, stale KV filter -- is the worst
    outcome available, so the actuator restores the old range and re-raises.
    """
    model, act, _ = _rank0()
    seen = []
    act.add_observer(lambda rep: seen.append(rep.to_rung))
    act.flip("[29,19,16]", quiescent=True)
    assert seen == ["[29,19,16]"]

    def _boom(_rep):
        raise RuntimeError("kv filter rebuild failed")

    act.add_observer(_boom)
    with pytest.raises(LayoutBoundaryError, match="rolled back"):
        act.flip("[28,20,16]", quiescent=True)
    assert (model.start_layer, model.end_layer) == (0, 29), "range must be restored"
    assert act.current_rung == "[29,19,16]"


def test_construction_performs_the_narrowing_step():
    """ "Load wide, run narrow" -- the narrowing must actually happen.

    The model arrives with the UNION range in force, because weight loading is
    gated on the same range it executes (qwen3_5.py:1563-1564, :1707-1708), so
    the union had to be active for the union to load. Constructing the actuator
    is what hands the rank its first rung.

    This was found by failure: the first version assumed the model already sat
    at its current rung, so the actuator's belief and the model's actual range
    disagreed from the very first forward -- a split brain in which the rank
    silently executed a layer it did not own.
    """
    log = []
    model = _FakeModel(0, 29, log)
    assert (model.start_layer, model.end_layer) == (0, 29), "fixture starts wide"
    LayoutBoundaryActuator(
        model=model, rung_ranges=RUNGS_RANK0, current_rung="[28,20,16]"
    )
    assert (model.start_layer, model.end_layer) == (0, 28), "narrowing did not happen"
    model(torch.zeros(4))
    assert log == list(range(28))


def test_the_two_halves_of_slice_1a_compose():
    """End to end: union-bound weights + a boundary flip = nothing copied.

    The union arena fixes byte offsets so no weight moves; the boundary
    actuator changes what executes. Neither is useful alone, so this pins that
    together they deliver a rung change in which every parameter keeps its
    storage and the executed set changes.
    """
    from sglang.srt.model_executor.weights_arena_union import (
        flip_delta,
        plan_union_arena,
    )

    # The arena half, over the same pair of rungs.
    def _named(upto):
        return {
            f"model.layers.{i}.w": torch.zeros(4, dtype=torch.float32)
            for i in range(upto)
        }

    plan = plan_union_arena({"[28,20,16]": _named(28), "[29,19,16]": _named(29)})
    arena_delta = flip_delta(plan, "[28,20,16]", "[29,19,16]")
    assert arena_delta.bytes_to_copy == 0
    assert arena_delta.requires_quiescence is True

    # The boundary half, on the same pair.
    model, act, log = _rank0()
    ptrs_before = {
        i: model.layers[i].weight.data_ptr()
        for i in range(29)
        if isinstance(model.layers[i], _RecordingLayer)
    }
    boundary = act.flip("[29,19,16]", quiescent=True)

    assert boundary.bytes_copied == arena_delta.bytes_to_copy == 0
    # The arena names layer 28's tensors; the boundary names layer 28's index.
    assert boundary.activated == (28,)
    assert all(".28." in n for n in arena_delta.activated)
    ptrs_after = {
        i: model.layers[i].weight.data_ptr()
        for i in range(29)
        if isinstance(model.layers[i], _RecordingLayer)
    }
    assert ptrs_before == ptrs_after
    log.clear()
    model(torch.zeros(4))
    assert log == list(range(29))
