"""VISION-WEIGHTS AP3 (plan PLAN-VISION-GEWICHTE-VERDRAENGEN-1009, NF line):
the tower borrows RESIDENT EXPERT ROWS that have a store slot, and the rows
come back from the store through the wake's own ``load_refill_rows``.

Hermetic, CPU. The expert layers are plain ``nn.Module`` s carrying what the
load-time presplit leaves on a FusedMoE layer (``_moe_offload_refill_runs`` +
``_moe_offload_presplit``), so the inventory runs through the REAL
``expert_offload._rearm_targets`` branch; the store is a CPU tensor whose
slot rows are what the wake loaded.

Pinned (plan §9 T6 + the AP3 order):
  * only rows with a store slot are candidates: the Platztausch prefix, a pad
    run and a run past the store's end never are -- even when the tower is
    then short (W105b, nothing moved);
  * the inventory and the plan are deterministic (same names, same plan on a
    rebuilt rank at other addresses);
  * the stage carries the checkpoint on expert rows, and afterwards every row
    is bitwise back; the return calls ``load_refill_rows`` with the rearm's
    runs, clipped to the rows the tower touched; no host image;
  * join order: the R3 joins run before any byte moves AND again before the
    return; ``DeferredRowsFill.land_now`` lands open rows without promoting;
  * a stale store row or a skipped return is W110c (fatal); a plan that no
    longer matches the live rows is W111b;
  * switch off: kvtail/auto never touch the source (no join, no refill), and
    the NF core resolves only the experts source.
"""

import types

import pytest
import torch

from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.weg2 import vision_rank_runner as vrr
from sglang.srt.weg2 import vision_rank_stage as vrs
from sglang.srt.weg2 import vision_victim as vv
from sglang.srt.weg2 import vision_victim_experts as vve
from test_weg2_vision_rank_runner import _Alloc, _build, _Item, _kv, _req, _write_model  # noqa: E402

import torch.nn.functional as F

W13, W2 = "w13_weight_packed", "w2_weight_packed"
#: int32 words per row: w13 512 B, w2 256 B (the 256 B grain of the core)
ROW_WORDS = {W13: 128, W2: 64}
NO_AIR = lambda device: (0, 0)  # noqa: E731


def _aligned(rows, words, gen):
    """[rows, words] int32 starting on a 256 B boundary (a CUDA allocation is;
    a CPU one need not be), carved out of a larger storage -- so the
    storage_offset path is exercised too."""
    base = torch.randint(-2**31, 2**31 - 1, (rows * words + 64,), generator=gen, dtype=torch.int32)
    shift = ((-base.data_ptr()) % 256) // 4
    return base[shift:shift + rows * words].view(rows, words)


class _MoELayer(torch.nn.Module):
    """What the presplit leaves on an offload layer: device buffers [rows]
    (prefix rows first, then the extra rows), the store [slots], the runs.
    The extra rows hold their store slot's bytes (the wake loaded them)."""

    def __init__(self, *, rows=12, slots=8, runs=((4, 0, 2), (6, 2, 2)), seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self._moe_offload_refill_runs = tuple(runs)
        self._moe_offload_presplit = {}
        for attr, words in ROW_WORDS.items():
            store = torch.randint(-2**31, 2**31 - 1, (slots, words), generator=g, dtype=torch.int32)
            buf = _aligned(rows, words, g)
            for z0, p0, n in runs:
                if p0 >= 0 and p0 + n <= slots:
                    buf[z0:z0 + n].copy_(store[p0:p0 + n])
            self._moe_offload_presplit[attr] = (buf, store)

    def bufs(self):
        return {a: b for a, (b, _s) in self._moe_offload_presplit.items()}


class _Model(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def snapshot(self):
        return [{a: b.clone() for a, b in lay.bufs().items()} for lay in self.layers]

    def same_as(self, snap):
        return all(torch.equal(lay.bufs()[a], snap[i][a]) for i, lay in enumerate(self.layers) for a in ROW_WORDS)


def _source(model, log=None, refill=None):
    log = [] if log is None else log
    calls = []

    def _refill(entries, runs, *, layer_id="?"):
        log.append("refill")
        calls.append((entries[0][0], tuple(runs), layer_id))
        return (refill or eo.load_refill_rows)(entries, runs, layer_id=layer_id)

    src = vve.ExpertRowVictims(targets=lambda: vve.model_targets(model), refill=_refill,
                               joins=(lambda: log.append("join"),))
    return src, calls, log


def _sched(alloc=None):
    return types.SimpleNamespace(token_to_kv_pool_allocator=alloc or _Alloc(_kv()))


def _run(s, reqs, tmp_path, **kw):
    kw.setdefault("build", _build())
    kw.setdefault("place", vrs.PLACE_WEIGHTS)
    return vrr.run_rank_stage(s, reqs, model_dir=str(tmp_path), hf_config=None,
                              device=torch.device("cpu"), **kw)


def _ref_rows(ck, px):
    w1, b1 = ck["model.visual.blocks.0.attn.qkv.weight"], ck["model.visual.blocks.0.attn.qkv.bias"]
    w2, b2 = ck["model.visual.merger.linear_fc1.weight"], ck["model.visual.merger.linear_fc1.bias"]
    return F.linear(F.linear(px.to(torch.bfloat16), w1, b1), w2, b2)


# -------------------------------------------------------------- inventory --


def test_only_rows_with_a_store_slot_are_candidates():
    """Prefix rows [0,4) have no slot (the Karte gives the common set none), a
    pad run (slot -1) and a run past the store's 8 slots are refused; the
    candidates are the maximal row ranges of the slot runs, one per
    attribute, and no candidate byte lies on a refused row."""
    lay = _MoELayer(rows=14, slots=8, runs=((4, -1, 1), (5, 0, 2), (7, 2, 2), (9, 6, 3)))
    src, _calls, _log = _source(_Model([lay]))
    inv = {c.name: c for c in src.inventory()}
    assert sorted(inv) == [f"000:layers.0.{W13}[5:9]", f"000:layers.0.{W2}[5:9]"]
    for attr, words in ROW_WORDS.items():
        buf = lay.bufs()[attr]
        c = inv[f"000:layers.0.{attr}[5:9]"]
        row = words * 4
        assert c.key == buf.data_ptr() + 5 * row and c.nbytes == c.storage_nbytes == 4 * row
    census = src.census()
    assert (census.layers, census.candidates, census.refused_rows) == (1, 2, 2 * (1 + 3))
    assert vve.slot_runs([(0, -1, 1), (1, 0, 2), (3, 7, 2)], store_slots=8) == (((1, 0, 2),), 3)


def test_rows_without_a_slot_are_never_taken_even_when_the_tower_is_short(tmp_path):
    """One slot row (512 B w13, 256 B w2) against a 1.5 KiB tower: W105b
    with the numbers, and not one byte of any row moved -- the 4 prefix rows
    are not borrowed to make up the difference."""
    _write_model(tmp_path)
    model = _Model([_MoELayer(rows=12, slots=8, runs=((4, 0, 1),))])
    snap = model.snapshot()
    src, calls, _log = _source(model)
    it = _Item()
    out = _run(_sched(), [_req("r", [it])], tmp_path, victims=src, air=NO_AIR)
    assert not out.ok and out.code == vv.W_VICTIM_SHORT and "victims hold" in out.detail
    assert model.same_as(snap) and calls == [] and it.precomputed_embeddings is None


def test_inventory_and_plan_are_deterministic_across_a_rebuilt_rank():
    """R5: the same layout gives the same candidates and the same plan, also
    on a rank rebuilt at other addresses (names carry the module order)."""
    tower = [("a", 384), ("b", 288), ("c", 48), ("d", 12)]
    plans = []
    for seed in (0, 1):
        model = _Model([_MoELayer(seed=seed), _MoELayer(seed=seed + 10, runs=((4, 0, 4),))])
        src, _c, _l = _source(model)
        inv = src.inventory()
        assert inv == src.inventory()
        plan = vv.plan_victims(inv, tower)
        plans.append(([s.name for s in plan.segments], [(t.seg, t.offset) for t in plan.slots]))
    assert plans[0] == plans[1]
    assert plans[0][0] == [f"000:layers.0.{W13}[4:8]"]


def test_a_plan_that_no_longer_matches_the_live_rows_is_W111b():
    model = _Model([_MoELayer()])
    src, _c, _l = _source(model)
    plan = vv.plan_victims(src.inventory(), [("a", 1024)])
    lay = model.layers[0]
    buf, store = lay._moe_offload_presplit[W13]
    lay._moe_offload_presplit[W13] = (buf.clone(), store)          # the rows moved (new storage)
    with pytest.raises(vv.VisionVictimPlanRefused, match="W111b"):
        src.views(plan.segments)


# ------------------------------------------------------------------ stage --


def test_the_tower_rides_on_expert_rows_and_every_row_comes_back_from_the_store(tmp_path):
    """T4/T6: the encode sees the tower IN the w13 rows, the result equals the
    reference, afterwards every buffer is bitwise what it was, the return
    went through load_refill_rows with the rearm's runs clipped to the 3 rows
    the 1.5 KiB tower touched (not all 4), and no host image was held."""
    ck = _write_model(tmp_path)
    model = _Model([_MoELayer(seed=3), _MoELayer(seed=4)])
    snap = model.snapshot()
    src, calls, log = _source(model)
    seen = {}

    def encode(module, items):
        seen["ptr"] = module.blocks[0].attn.qkv_proj.weight.data_ptr()
        seen["moved"] = not model.same_as(snap)
        return vrr.encode_items(module, items)

    it = _Item()
    px = it.feature.clone()
    out = _run(_sched(), [_req("r", [it])], tmp_path, victims=src, encode=encode, air=NO_AIR)
    assert out.ok, out.detail
    assert torch.equal(it.precomputed_embeddings, _ref_rows(ck, px))
    w13 = model.layers[0].bufs()[W13]
    assert w13.data_ptr() + 4 * 512 <= seen["ptr"] < w13.data_ptr() + 8 * 512 and seen["moved"]
    assert model.same_as(snap) and out.checksum == "ok" and not out.fatal
    assert calls == [(W13, ((4, 0, 2), (6, 2, 1)), "layers.0")]
    assert "victim=experts" in out.victim_fields and "host_image_mib=0.0" in out.victim_fields
    assert log == ["join", "join", "refill"]


def test_the_joins_run_before_any_byte_moves_and_again_before_the_return(tmp_path):
    """R3 order: the first join sees every row untouched (nothing moved yet),
    the second sees the tower in the rows (after the encode) and comes before
    the refill."""
    _write_model(tmp_path)
    model = _Model([_MoELayer(seed=5)])
    snap = model.snapshot()
    states = []
    src = vve.ExpertRowVictims(
        targets=lambda: vve.model_targets(model),
        refill=lambda entries, runs, *, layer_id="?": (states.append("refill"),
                                                        eo.load_refill_rows(entries, runs, layer_id=layer_id))[1],
        joins=(lambda: states.append("join:" + ("intact" if model.same_as(snap) else "moved")),))
    out = _run(_sched(), [_req("r", [_Item()])], tmp_path, victims=src, air=NO_AIR)
    assert out.ok, out.detail
    assert states == ["join:intact", "join:moved", "refill"]


def test_land_now_waits_every_open_row_without_promoting(monkeypatch):
    """DeferredRowsFill.land_now (the R3 join on P, DEFER_HOST): rows not yet
    issued are issued, every event is waited on the host, and the layers stay
    pending -- the next forward's tick promotes them."""
    issued, waited = [], []

    class _Ev:
        def __init__(self, n):
            self.n = n

        def synchronize(self):
            waited.append(self.n)

        def query(self):
            return True

    class _Ops:
        def new_stream(self):
            return "side"

        def stream_ctx(self, stream):
            import contextlib

            return contextlib.nullcontext()

        def after_current(self, stream):
            pass

        def record(self, stream):
            return _Ev(len(issued))

        def current_waits(self, ev):
            pass

    lay = _MoELayer(runs=((4, 0, 4),))
    buf, store = lay._moe_offload_presplit[W13]
    buf[4:8].zero_()                                             # not loaded yet
    cache = types.SimpleNamespace(_deferred_rows=eo.DeferredRows(
        entries=[(W13, buf, store)], runs=((4, 0, 4),), full=None, rows=4, layer_id=0, early=True))
    fill = eo.DeferredRowsFill(stream_ops=_Ops())
    assert fill.land_now(why="test") == 0                         # nothing open: nothing done
    fill.add(cache)
    orig = eo.load_refill_rows
    monkeypatch.setattr(eo, "load_refill_rows",
                        lambda e, r, layer_id="?": (issued.append(1), orig(e, r, layer_id=layer_id))[1])
    assert fill.land_now(why="test") == 1
    assert issued == [1] and waited == [1] and torch.equal(buf[4:8], store[0:4])
    assert fill.pending == [cache] and cache._deferred_rows is not None


# -------------------------------------------------------- W110c (fatal) --


@pytest.mark.parametrize("case,needle", [("stale_store", "checksum MISMATCH"),
                                         ("skip", "checksum MISMATCH"),
                                         ("raise", "restore raised")])
def test_a_stale_store_or_a_failed_return_is_W110c(tmp_path, case, needle):
    """A store slot whose bytes differ from the device row (stale store), a
    skipped refill (mutant) and a failing copy: the checksum or the raise
    makes the outcome W110c -- the pass then stops the group (core test)."""
    _write_model(tmp_path)
    model = _Model([_MoELayer(seed=6)])
    refill = None
    if case == "stale_store":
        model.layers[0]._moe_offload_presplit[W13][1][0] ^= 1    # slot 0 no longer = row 4
    elif case == "skip":
        refill = lambda entries, runs, *, layer_id="?": 0         # noqa: E731
    else:
        def refill(entries, runs, *, layer_id="?"):
            raise RuntimeError("H2D failed")
    src, _calls, _log = _source(model, refill=refill)
    out = _run(_sched(), [_req("r", [_Item()])], tmp_path, victims=src, air=NO_AIR)
    assert not out.ok and out.code == vv.W_VICTIM_NOT_RESTORED and needle in out.fatal


# ------------------------------------------------------- switch off / wiring --


def test_switch_off_never_touches_the_rows_and_nf_resolves_only_experts(tmp_path, monkeypatch):
    """kvtail (and auto/free) never call the source: no join, no refill, every
    row untouched. The NF core lists only the experts module; a rank with
    refill rows resolves to it, a rank without is W111b by name."""
    _write_model(tmp_path)
    model = _Model([_MoELayer(seed=7)])
    snap = model.snapshot()
    src, calls, log = _source(model)
    out = _run(_sched(), [_req("r", [_Item()])], tmp_path, victims=src, place=vrs.PLACE_KVTAIL)
    assert out.ok and out.place == vrs.PLACE_KVTAIL and not out.victim_fields
    assert log == [] and calls == [] and model.same_as(snap)

    assert vv.SOURCE_MODULES == ("sglang.srt.weg2.vision_victim_experts",)
    sched = types.SimpleNamespace(tp_worker=types.SimpleNamespace(
        model_runner=types.SimpleNamespace(model=model)))
    got = vv.resolve_source(sched)
    assert isinstance(got, vve.ExpertRowVictims) and got.kind == "experts" and got.host_bytes == 0
    bare = types.SimpleNamespace(tp_worker=types.SimpleNamespace(
        model_runner=types.SimpleNamespace(model=_Model([]))))
    with pytest.raises(vv.VisionVictimPlanRefused, match="no victim source applies"):
        vv.resolve_source(bare)


def test_arming_line_plans_the_header_tower_on_the_slot_rows():
    """M0: the arming line names victim=experts, the inventory and a plan for
    the checkpoint's tower sizes; a tower tensor longer than every slot run
    is the W105b refusal with both numbers."""
    model = _Model([_MoELayer(seed=8), _MoELayer(seed=9)])
    src, _c, _l = _source(model)
    ck = [vrs.CkptTensor("model.visual.blocks.0.attn.qkv.weight", torch.bfloat16, (24, 8), 0, 384),
          vrs.CkptTensor("model.visual.merger.linear_fc1.weight", torch.bfloat16, (6, 24), 384, 288)]
    line, why = vv.arming_line(src, ck, lambda n: n)
    assert why == "" and "victim=experts" in line and "candidates=4" in line and "planned_segments=1" in line
    big = [vrs.CkptTensor("model.visual.merger.linear_fc2.weight", torch.bfloat16, (2048, 1), 0, 4096)]
    line, why = vv.arming_line(src, big, lambda n: n)
    assert "plan=REFUSED" in line and "W105b" in why and "in one piece" in why
