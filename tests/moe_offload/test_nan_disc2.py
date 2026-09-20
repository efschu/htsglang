"""Task #49 (20.09.): the '[nan-disc2]' discriminator's evaluation logic.

Hermetic -- no CUDA, no model. The pieces exercised here are the ones that
decide the verdict a needle boot will be read by: the expert intersection over
the bad rows, the byte fingerprint, the planner-vs-holds consistency tests, and
the class each combination maps to. A stub cache stands in for
``MoEExpertOffloadCache`` (plain CPU tensors, the same attribute names).
"""

import inspect

import pytest
import torch

from sglang.srt.layers.moe import nan_disc2 as nd
from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe.fused_moe_triton import fused_marlin_moe as fmm


@pytest.fixture(autouse=True)
def _fresh():
    nd._reset_for_tests(on=True, armed=True)
    yield
    nd._reset_for_tests(on=True, armed=True)


# --- the switches ---------------------------------------------------------


def test_disc2_default_on_and_env_off(monkeypatch):
    for raw, want in (("", True), ("1", True), ("0", False), ("off", False)):
        nd._STATE["on"] = None
        if raw == "":
            monkeypatch.delenv("SGLANG_NAN_DISC2", raising=False)
        else:
            monkeypatch.setenv("SGLANG_NAN_DISC2", raw)
        assert nd.disc2_on() is want, raw
    nd._STATE["on"] = None


def test_arm_once_fires_exactly_once_per_process():
    assert nd.arm_once() is True
    assert nd.arm_once() is False
    assert nd.arm_once() is False


def test_private_workspace_switch_parses_and_defaults_off(monkeypatch):
    for raw, want in (("", False), ("0", False), ("1", True), ("on", True)):
        fmm._PRIVATE_WS["on"] = None
        if raw == "":
            monkeypatch.delenv("SGLANG_MOE_MARLIN_PRIVATE_WORKSPACE", raising=False)
        else:
            monkeypatch.setenv("SGLANG_MOE_MARLIN_PRIVATE_WORKSPACE", raw)
        assert fmm.marlin_private_workspace_on() is want, raw
    fmm._PRIVATE_WS["on"] = None


def test_private_workspace_drops_the_supplied_buffer_before_the_alloc():
    """The switch must land BEFORE the `workspace is None` branch, or it would
    keep the shared buffer and quietly prove nothing."""
    src = inspect.getsource(fmm)
    i = src.index("    if marlin_private_workspace_on():")
    j = src.index("    if workspace is None:")
    assert i < j
    assert "workspace = None" in src[i:j]
    # and the provenance line is emitted before the switch can hide it
    assert src.index("    _log_workspace_provenance(workspace)") < i


def test_the_moe_path_really_is_handed_a_shared_workspace():
    """Belegt, not assumed: the wNa16 Marlin MoE scheme allocates ONE workspace
    at load and passes that same object to every fused_marlin_moe call."""
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_wNa16_moe as scheme,
    )

    src = inspect.getsource(scheme)
    assert "layer.workspace = marlin_make_workspace(" in src
    assert "workspace=layer.workspace," in src
    # the runner path keeps a process-global one
    from sglang.srt.layers.moe.moe_runner import marlin as runner

    rsrc = inspect.getsource(runner)
    assert "MARLIN_MOE_WORKSPACE" in rsrc and "workspace=MARLIN_MOE_WORKSPACE," in rsrc


# --- fingerprints ---------------------------------------------------------


def test_fingerprint_is_byte_exact_and_stable():
    a = torch.arange(64, dtype=torch.int32)
    b = a.clone()
    assert nd.fingerprint(a) == nd.fingerprint(b)
    b[7] += 1
    assert nd.fingerprint(a) != nd.fingerprint(b)


def test_fingerprint_separates_a_single_flipped_element_of_a_float_row():
    a = torch.ones(128, dtype=torch.float16)
    b = a.clone()
    b[99] = torch.tensor(1.0009765625, dtype=torch.float16)
    assert nd.fingerprint(a) != nd.fingerprint(b)


def test_fingerprint_never_raises():
    assert nd.fingerprint(None) == "none"
    assert isinstance(nd.fingerprint(object()), str)


# --- the expert intersection ---------------------------------------------


def _ids(rows):
    return rows


def test_the_expert_in_every_bad_row_is_named_and_the_shared_ones_are_not():
    # expert 7 is in every bad row; expert 3 is too, but also in a good row
    ids = {
        0: [7, 3, 11],
        1: [7, 3, 12],
        2: [7, 3, 13],
        3: [1, 3, 5],  # good
        4: [2, 4, 6],  # good
    }
    ids_list = [ids[i] for i in range(5)]
    bad = nd.row_expert_sets(ids_list, [0, 1, 2])
    good = nd.row_expert_sets(ids_list, [3, 4])
    common, exclusive = nd.suspect_experts(bad, good)
    assert common == [3, 7]
    assert exclusive == [7]


def test_no_common_expert_yields_an_empty_intersection_not_everything():
    ids_list = [[1, 2], [3, 4], [5, 6]]
    bad = nd.row_expert_sets(ids_list, [0, 1, 2])
    assert nd.common_experts(bad) == []
    # and an intersection over NOTHING is empty, not 'all'
    assert nd.common_experts([]) == []


def test_padding_and_out_of_range_rows_are_dropped():
    ids_list = [[5, -1, -1], [5, 9, -1]]
    sets = nd.row_expert_sets(ids_list, [0, 1, 99])
    assert sets == [{5}, {5, 9}]


# --- the trace tests ------------------------------------------------------


def _wave(w, slot_of, holds=None, needed=None):
    return {
        "wave": w,
        "needed": needed if needed is not None else sorted(slot_of),
        "slot_of_needed": dict(slot_of),
        "holds": dict(holds or {}),
    }


def test_locate_expert_names_wave_and_slot():
    waves = [_wave(0, {4: 0, 7: 5}, holds={5: 7}), _wave(1, {9: 5}, holds={5: 9})]
    hits = nd.locate_expert(waves, 7)
    assert hits == [{"wave": 0, "slot": 5, "held_by_then": 7}]
    assert nd.locate_expert(waves, 42) == []


def test_hold_mismatch_is_only_reported_when_the_two_records_disagree():
    ok = [_wave(0, {7: 5}, holds={5: 7})]
    assert nd.hold_mismatches(ok) == []
    bad = [_wave(0, {7: 5}, holds={5: 8})]
    assert nd.hold_mismatches(bad) == [{"wave": 0, "expert": 7, "slot": 5, "held": 8}]
    # a resident slot is absent from the holds and must not be guessed at
    resident = [_wave(0, {1: 0}, holds={})]
    assert nd.hold_mismatches(resident) == []


def test_two_experts_on_one_slot_is_a_collision():
    assert nd.slot_collisions([_wave(0, {7: 5, 8: 5})]) == [
        {"wave": 0, "slot": 5, "experts": [7, 8]}
    ]
    assert nd.slot_collisions([_wave(0, {7: 5, 8: 6})]) == []


# --- the verdicts ---------------------------------------------------------


D_OK = "0" * 16  # a well-formed 64-bit digest
D_BAD = "f" * 16  # a different one


def _obs(**kw):
    base = dict(
        common=[7],
        located=[{"wave": 1, "slot": 5, "held_by_then": 7}],
        suspect=7,
        fp_before={"q": D_OK, "s": D_OK},
        fp_after={"q": D_OK, "s": D_OK},
        fp_host={"q": D_OK, "s": D_OK},
        fp_valid=True,
        hold_mismatches=[],
        slot_collisions=[],
        ws_nonzero_before=0,
        common_wave=1,
        partials="table",
    )
    base.update(kw)
    return base


def test_class_d_router_id_outside_the_lut():
    code, text = nd.classify(_obs(located=[]))
    assert code == nd.CLASS_ROUTER
    assert "NO wave" in text


def test_class_a_from_a_slot_collision():
    code, _ = nd.classify(_obs(slot_collisions=[{"wave": 1, "slot": 5, "experts": [7, 8]}]))
    assert code == nd.CLASS_LUT


def test_class_a_from_a_planner_holds_disagreement():
    code, _ = nd.classify(
        _obs(hold_mismatches=[{"wave": 1, "expert": 7, "slot": 5, "held": 8}])
    )
    assert code == nd.CLASS_LUT


def test_class_b_transient_slot_bytes_when_the_refetch_repairs_them():
    code, text = nd.classify(_obs(fp_before={"q": D_BAD, "s": D_OK}))
    assert code == nd.CLASS_BYTES
    assert "refetch repaired" in text


def test_class_b_persistent_when_the_refetch_does_not_repair_them():
    code, text = nd.classify(
        _obs(fp_before={"q": D_BAD, "s": D_OK}, fp_after={"q": D_BAD, "s": D_OK})
    )
    assert code == nd.CLASS_BYTES
    assert "pinned pool row itself" in text


def test_class_e_wave_partials_when_no_expert_is_shared():
    code, text = nd.classify(_obs(common=[], located=[], common_wave=3, partials="stream"))
    assert code == nd.CLASS_PARTIALS
    assert "stream" in text


def test_class_c_kernel_when_every_fingerprint_agrees():
    code, text = nd.classify(_obs())
    assert code == nd.CLASS_KERNEL
    assert "identical inputs" in text


def test_class_c_names_a_dirty_lock_workspace_when_there_was_one():
    code, text = nd.classify(_obs(ws_nonzero_before=17))
    assert code == nd.CLASS_KERNEL
    assert "17 non-zero" in text
    assert "SGLANG_MOE_MARLIN_PRIVATE_WORKSPACE" in text


def test_a_reused_slot_refuses_to_pretend_the_bytes_prove_anything():
    code, text = nd.classify(_obs(fp_valid=False, fp_before={"q": D_BAD}, holds_now=8))
    assert code == nd.CLASS_UNRESOLVED
    assert "proves nothing" in text


def test_an_unexplained_combination_is_unresolved_not_a_guess():
    # the slot matched the host original before the recompute (so not B) but
    # the recompute's refetch changed it (so not C), no expert is shared by the
    # bad rows and they span several waves (so not E): nothing fits, and the
    # instrument says so instead of picking the nearest class.
    code, text = nd.classify(
        _obs(
            common=[],
            fp_before={"q": D_OK},
            fp_host={"q": D_OK},
            fp_after={"q": D_BAD},
            common_wave=None,
        )
    )
    assert code == nd.CLASS_UNRESOLVED
    assert "do not guess" in text


def test_an_unreadable_fingerprint_is_never_counted_as_a_corruption():
    """A pool row that could not be read must not become a 'B' verdict."""
    code, _ = nd.classify(
        _obs(fp_before={"q": "a" * 16}, fp_after={"q": "a" * 16}, fp_host={"q": "unreadable(IndexError)"})
    )
    assert code == nd.CLASS_KERNEL


# --- snapshot / finish / render over a stub cache -------------------------


class _StubCache:
    def __init__(self, resident_count=2, holds=None, waves=None, ids_list=None):
        self.resident_count = resident_count
        self._spill_pool_index = None
        self._scratch_holds = dict(holds or {})
        self._resident = {
            "qweight": torch.arange(4 * 8, dtype=torch.int32).reshape(4, 8),
            "scales": torch.ones(4, 8, dtype=torch.float16),
        }
        self._pinned = {
            "qweight": torch.zeros(4, 8, dtype=torch.int32),
            "scales": torch.ones(4, 8, dtype=torch.float16),
        }
        self._nan_trace = {
            "ids_list": ids_list,
            "waves": waves or [],
            "partials": "table",
        }


class _StubExperts:
    def __init__(self, ws=None):
        self.workspace = ws


def test_snapshot_collects_the_whole_group_and_finish_adds_the_after_half():
    ids_list = [[7, 1], [7, 2], [7, 3], [1, 2], [2, 3]]
    cache = _StubCache(
        resident_count=2,
        holds={3: 7},
        waves=[_wave(0, {1: 0, 2: 1}, holds={}), _wave(1, {7: 3}, holds={3: 7})],
        ids_list=ids_list,
    )
    experts = _StubExperts(torch.zeros(8, dtype=torch.int32))
    snap = nd.snapshot(experts, cache, [0, 1, 2], [3, 4])
    assert snap["suspect"] == 7
    assert snap["slot"] == 3
    assert snap["resident_slot"] is False
    assert snap["holds_now"] == 7
    assert snap["fp_valid"] is True
    assert set(snap["fp_before"]) == {"qweight", "scales"}
    # spill row of expert 7 with resident_count=2 and no pool index is 7-2=5,
    # which is out of the stub pool -> reported as unreadable, never invented
    assert snap["fp_host"] and all(
        v.startswith("unreadable") for v in snap["fp_host"].values()
    )
    assert snap["ws_nonzero_before"] == 0

    out = nd.finish(experts, cache, snap, recompute_bad_rows=0)
    assert out["fp_after"] == snap["fp_before"]  # nothing moved in the stub
    assert out["ws_nonzero_after"] == 0
    # nothing moved and the host row is unreadable -> the kernel class, with the
    # missing host comparison named rather than papered over
    assert out["verdict"] == nd.CLASS_KERNEL
    assert "no comparable host fingerprint" in out["verdict_text"]
    text = nd.render(5, out)
    assert text.count("[nan-disc2]") >= 10
    assert "VERDICT" in text and "\n" in text


def test_snapshot_returns_none_without_a_trace():
    cache = _StubCache()
    cache._nan_trace = None
    assert nd.snapshot(_StubExperts(), cache, [0], [1]) is None


def test_find_cache_prefers_the_named_attribute():
    class _M:
        pass

    experts = _M()
    cache = _StubCache()
    experts._expert_offload = cache
    layer = _M()
    layer.mlp = _M()
    layer.mlp.experts = experts
    assert nd.find_cache(layer) == (experts, cache)


# --- the recorder in the offload cache ------------------------------------


class _TraceHost:
    _nan_trace_begin = eo.MoEExpertOffloadCache._nan_trace_begin
    _nan_trace_wave = eo.MoEExpertOffloadCache._nan_trace_wave

    def __init__(self):
        self._nan_trace = None
        self._scratch_holds = {3: 7}


def test_the_recorder_is_a_no_op_while_the_guard_is_off(monkeypatch):
    from sglang.srt.layers import nan_guard

    nan_guard._reset_for_tests(on=False)
    h = _TraceHost()
    h._nan_trace_begin([[1, 2]])
    assert h._nan_trace is None
    h._nan_trace_wave(0, [1], {1: 0})  # must not raise
    nan_guard._reset_for_tests(on=False)


def test_the_recorder_keeps_the_slot_map_and_the_holds_of_each_wave():
    from sglang.srt.layers import nan_guard

    nan_guard._reset_for_tests(on=True)
    try:
        h = _TraceHost()
        h._nan_trace_begin([[1, 7]])
        assert h._nan_trace["ids_list"] == [[1, 7]]
        h._nan_trace_wave(1, [7], {7: 3})
        w = h._nan_trace["waves"][0]
        assert w == {"wave": 1, "needed": [7], "slot_of_needed": {7: 3}, "holds": {3: 7}}
        # the holds are COPIED, not aliased: a later fetch must not rewrite history
        h._scratch_holds[3] = 9
        assert h._nan_trace["waves"][0]["holds"] == {3: 7}
    finally:
        nan_guard._reset_for_tests(on=False)


def test_every_wave_loop_records_the_trace_right_after_its_fetch():
    for fn in (
        eo.MoEExpertOffloadCache._run_single_wave,
        eo.MoEExpertOffloadCache._run_waves_expert_major,
        eo.MoEExpertOffloadCache.run_waves,
    ):
        src = inspect.getsource(fn)
        assert "_nan_trace" in src, fn.__name__
    src = inspect.getsource(eo.MoEExpertOffloadCache._run_waves_expert_major)
    i = src.index("self._fetch(fetch_plan)")
    j = src.index("self._nan_trace_wave(")
    k = src.index("combine_out = apply_fn(sub)")
    assert i < j < k
