"""G4 (NF-GGUF, PLAN-GGUF-NF-1009): the GGUF door into the Platztausch / shared-store presplit.

Before G4 the only door into ``presplit_expert_offload_after_repack`` (Karte order, shared store rows, H95c seat
rows, D-store-adopt, the flip's expert buffer) was a Marlin scheme with a real ``[E, ...]`` stack. GGUF staged into a
PRIVATE pinned pool: no store row, no Karte, nothing for the flip to pair. These tests drive the REAL
``FusedMoE.materialize_gguf_weights`` on CPU tensors for a toy NF-shaped boot (12 experts, P = one unsharded PP
stage, D = two expert-dim-sharded ranks with GGUF's TRAILING pad expert, a nested Karte) and check, byte for byte,
what lands where:

* the device bank holds ``plan.resident_ids[i]`` in slot ``i`` (the pad row zero),
* the shared store holds every cold expert at the slot the Karte names, written by P AND by D with the SAME bytes,
* three row classes (the unsloth UD-IQ4_XS types per layer give 2.2217 / 3.0029 / 3.3203 MiB) get three file sizes
  and nothing assumes layer uniformity,
* the flip's join / plan moves those rows as plain bytes, both directions,
* the guards (#323b whitelist, W120 for an unstaged layer, row geometry, expert count) refuse by name,
* the store identity of a GGUF source reads its header (and the INT4 identity does not move).

No CUDA, no checkpoint. The unsloth header test reads the real file when present.
"""

from __future__ import annotations

import hashlib
import json
import os
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.moe import expert_map as em  # noqa: E402
from sglang.srt.layers.moe import expert_offload as eo  # noqa: E402
from sglang.srt.layers.moe import expert_store as es  # noqa: E402
from sglang.srt.layers.moe import gguf_layout as gl  # noqa: E402
from sglang.srt.layers.moe import gguf_presplit as gp  # noqa: E402
from sglang.srt.layers.moe import store_adopt as sa  # noqa: E402
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # noqa: E402
from sglang.srt.layers.quantization.gguf import GGUFUninitializedParameter  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=30, suite="stage-a-weg2-unit")

TOTAL = 12
D_RATIOS = [1, 1]          # two D ranks, 6 owned experts each
FR_D = [0.6, 0.6]          # 4 residents of 6 owned (+ the pad row)
FR_P = [0.7]               # 9 of 12 on the one PP stage
LAYERS = (0, 1, 2)         # one layer per row class
Q8_0 = 8                   # a ggml type with a GGUF MoE kernel (covered)

#: per layer: (w13 row shape, w2 row shape) -- three DIFFERENT row classes, bytes per expert row
#: class A / B / C of the unsloth export differ the same way (gate/up IQ3_S|IQ4_XS, down IQ4_NL|Q8_0)
ROW_SHAPES = {
    0: ((8, 64), (4, 32)),     # 512 + 128   = 640 B
    1: ((8, 96), (4, 48)),     # 768 + 192   = 960 B
    2: ((8, 112), (4, 64)),    # 896 + 256   = 1152 B
}


def _karte():
    k = em.build_nested(total=TOTAL, ratios=D_RATIOS, fr_pp=FR_P, fr_tp=FR_D,
                        p_layer_stage=[0, 0, 0], pad_tp=1)
    return json.loads(json.dumps(k))  # what the ranks read: the JSON file


def _expert_bytes(layer_id, attr, g):
    """The loaded bytes of global expert ``g`` -- recognizable per (layer, tensor, expert)."""
    shape = ROW_SHAPES[layer_id][0 if attr == "w13_qweight" else 1]
    n = shape[0] * shape[1]
    salt = (layer_id * 7 + (1 if attr == "w13_qweight" else 2)) * 31 + g * 13
    return torch.tensor([(salt + i * 5) % 251 + 1 for i in range(n)], dtype=torch.uint8).reshape(shape)


class _CoveredType:
    weight_type = Q8_0


class _FakeGGUFMoEMethod:
    """Stands in for the CUDA ``GGUFMoEMethod`` by class name only."""


_FakeGGUFMoEMethod.__name__ = "GGUFMoEMethod"


class _StubMoe(torch.nn.Module):
    """A loaded GGUF ``FusedMoE`` stand-in carrying the REAL methods under test."""

    materialize_gguf_weights = FusedMoE.materialize_gguf_weights
    _gguf_expert_source = staticmethod(FusedMoE._gguf_expert_source)
    _gguf_moe_offload_eligible = FusedMoE._gguf_moe_offload_eligible
    _gguf_moe_offload_eligible_uncached = FusedMoE._gguf_moe_offload_eligible_uncached
    _drain_gguf_stream_stagers = FusedMoE._drain_gguf_stream_stagers
    _finish_gguf_moe_offload_staging = FusedMoE._finish_gguf_moe_offload_staging
    _gguf_owned_expert_count = FusedMoE._gguf_owned_expert_count
    _gguf_cold_shard_context = FusedMoE._gguf_cold_shard_context
    _gguf_stream_staging_enabled = FusedMoE._gguf_stream_staging_enabled
    _build_expert_shard_topk_remap = FusedMoE._build_expert_shard_topk_remap


def _gguf_layer(layer_id, *, rank=None, poison=(), fraction=None, eligible=True):
    """P layer (``rank is None``: unsharded, all 12 experts) or D rank ``rank`` (owned span + trailing pad)."""
    lay = _StubMoe()
    lay.layer_id = layer_id
    lay.quant_method = _FakeGGUFMoEMethod()
    lay.num_experts = TOTAL
    lay.num_local_experts = TOTAL          # GGUF's own shard does NOT shrink this (the trap G4 closes)
    lay._moe_offload_excluded = False
    lay.w13_qweight_type = _CoveredType() if eligible else types.SimpleNamespace(weight_type=39 + 100)
    lay.w2_qweight_type = _CoveredType()
    if rank is None:
        lay.moe_tp_rank, lay.moe_tp_size = 0, 1
        lay._gguf_expert_shard = False
        lay._gguf_expert_range = (0, TOTAL)
        lay._expert_offload_fraction = FR_P[0] if fraction is None else fraction
        owned = range(TOTAL)
    else:
        spans = em.scaled_spans(D_RATIOS, TOTAL)
        lo = em.bounds(spans)[rank]
        lay.moe_tp_rank, lay.moe_tp_size = rank, len(D_RATIOS)
        lay._gguf_expert_shard = True
        lay._gguf_expert_range = (lo, lo + spans[rank])
        lay._expert_offload_fraction = FR_D[rank] if fraction is None else fraction
        owned = range(lo, lo + spans[rank])
    shape13, shape2 = ROW_SHAPES[layer_id]
    half = shape13[0] // 2
    for attr, shape in (("w13_qweight", shape13), ("w2_qweight", shape2)):
        param = GGUFUninitializedParameter(requires_grad=False)
        param.is_gguf_weight = True
        param.tensor_shape = (TOTAL,) + tuple(shape)
        param.data_container = []
        param.expert_data_map = {}
        for g in owned:
            full = _expert_bytes(layer_id, attr, g)
            if g in poison:
                full = torch.full_like(full, 0xEE)
            shards = (
                [("w1", full[:half].clone()), ("w3", full[half:].clone())]
                if attr == "w13_qweight" else [("w2", full)]
            )
            for sid, t in shards:
                param.expert_data_map[(g, sid)] = t
                param.data_container.append(t)
        lay.register_parameter(attr, param)
    return lay


@pytest.fixture
def boot(monkeypatch, tmp_path):
    """The env a flip boot hands every rank: group, store dir, the Karte, the slot geometry, an identity."""
    karte = _karte()
    store = tmp_path / "store"
    monkeypatch.setattr(es, "expert_map", lambda: karte)
    monkeypatch.setenv(es.EXPERT_MAP_ENV, "/karte-served-by-the-patched-reader.json")
    monkeypatch.setenv(es.STORE_DIR_ENV, str(store))
    monkeypatch.setenv(es.SLOT_FRACTION_ENV, "0.5")
    monkeypatch.setenv(es.STORE_IDENTITY_ENV, "g4-toy")
    es.forget_written_rows()
    sa.reset_for_tests()
    eo.reset_expert_offload_release()

    def group(g):
        monkeypatch.setenv("SGLANG_WEG2_GROUP", g)

    return types.SimpleNamespace(karte=karte, store=str(store), group=group, monkeypatch=monkeypatch)


def _stage(layer):
    layer.materialize_gguf_weights()
    return layer


def _bank(layer, attr):
    return layer._moe_offload_presplit[attr][0]


def _store_row(boot, layer_id, attr, slot):
    path = es.store_path(boot.store, f"L{layer_id}", attr)
    shape13, shape2 = ROW_SHAPES[layer_id]
    shape = shape13 if attr == "w13_qweight" else shape2
    raw = open(path, "rb").read()
    n = shape[0] * shape[1]
    return torch.frombuffer(bytearray(raw[slot * n:(slot + 1) * n]), dtype=torch.uint8).reshape(shape)


# ---------------------------------------------------------------------------------------------------------------
# 1. the pad's POSITION: GGUF's trailing pad is not the generic shard's leading pad
# ---------------------------------------------------------------------------------------------------------------


def test_the_gguf_shard_window_is_trailing_and_the_generic_one_is_unchanged():
    d1 = _gguf_layer(0, rank=1)
    assert eo._layer_expert_window_ex(d1) == (6, eo.PAD_TRAIL)
    # the legacy window calls this layer an unsharded stage with lo == 0 (num_local == num_global): the trap
    assert eo._layer_expert_window(d1) == (0, False)
    p = _gguf_layer(0)
    assert eo._layer_expert_window_ex(p) == (0, None)
    generic = types.SimpleNamespace(num_experts=TOTAL, num_local_experts=7, _expert_shard_generic=True,
                                    _gguf_expert_shard=True, _gguf_expert_range=(6, 12))
    assert eo._layer_expert_window_ex(generic) == (6, eo.PAD_LEAD)
    assert eo._layer_expert_window(generic) == (6, True)
    ep = types.SimpleNamespace(num_experts=TOTAL, num_local_experts=4)
    assert eo._layer_expert_window_ex(ep) is None and eo._layer_expert_window(ep) is None


def test_the_karte_layout_of_a_gguf_d_rank_puts_the_pad_last_of_the_prefix(boot):
    boot.group("D")
    d1 = _gguf_layer(0, rank=1)
    order, n_praefix, refill = eo._karten_layout_lokal(d1, 7)
    # owned 6..11 -> local 0..5, pad = local 6; prefix = residents 6..9 = local 0..3, then the pad, no extra
    assert list(order) == [0, 1, 2, 3, 6] and n_praefix == 4
    assert refill == ((4, -1),)
    assert len(order) == eo.resident_slot_count(7, FR_D[1])
    res = eo._karten_residenz_lokal(d1, 7)
    assert res == (0, 1, 2, 3, 6)           # the pad stays resident wherever it sits


def test_the_generic_leading_pad_layout_is_unchanged(boot):
    """The Marlin / compressed-tensors shard (leading pad) reads what it always read."""
    boot.group("D")
    lay = types.SimpleNamespace(num_experts=TOTAL, num_local_experts=7, layer_id=0, moe_tp_rank=1,
                                _expert_shard_generic=True, _gguf_expert_range=(6, 12))
    order, n_praefix, refill = eo._karten_layout_lokal(lay, 7)
    assert list(order) == [1, 2, 3, 4, 0] and n_praefix == 4 and refill == ((4, -1),)
    assert eo._karten_residenz_lokal(lay, 7) == (0, 1, 2, 3, 4)


def test_the_store_rows_of_a_trailing_pad_shard_never_name_the_pad(boot):
    boot.group("D")
    d1 = _gguf_layer(0, rank=1)
    plan = types.SimpleNamespace(spill_ids=[4, 5])          # local 4,5 = global 10,11 (cold on D)
    sdir, key, lo, slots, index, pad = eo._expert_store_rows_for(d1, plan)
    assert (key, lo, pad) == ("L0", 6, False)
    assert index == {4: boot.karte["phases"]["D"]["slot_of"]["10"], 5: boot.karte["phases"]["D"]["slot_of"]["11"]}
    assert slots == boot.karte["slots"]


# ---------------------------------------------------------------------------------------------------------------
# 2. the door, byte for byte (P unsharded, D sharded, three row classes)
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("layer_id", LAYERS)
def test_p_stage_residents_store_and_flip_buffer(boot, layer_id):
    boot.group("P")
    p = _stage(_gguf_layer(layer_id))
    plan_order = boot.karte["phases"]["P"]["resident"][0]
    for attr in gp.GGUF_EXPERT_ATTRS:
        bank = _bank(p, attr)
        assert bank.shape[1:] == torch.Size(ROW_SHAPES[layer_id][0 if attr == "w13_qweight" else 1])
        for slot, g in enumerate(plan_order):
            assert torch.equal(bank[slot], _expert_bytes(layer_id, attr, g)), (attr, slot, g)
        # the cold experts of P (5, 10, 11) are in the store at the Karte's slots
        slot_of = boot.karte["phases"]["P"]["slot_of"]
        for g in (5, 10, 11):
            assert torch.equal(_store_row(boot, layer_id, attr, slot_of[str(g)]),
                               _expert_bytes(layer_id, attr, g)), (attr, g)
        # the flip sees the PREFIX of the bank under the Karte
        n_praefix = len(boot.karte["phases"]["P"]["common"][0])
        pub = getattr(p, "weg2_experts_" + attr)
        assert pub.shape[0] == n_praefix and pub.data_ptr() == bank.data_ptr()
    assert p._moe_offload_exchange_rows == len(boot.karte["phases"]["P"]["common"][0])
    assert p._moe_offload_gguf_staged is True
    assert list(p._moe_offload_frozen_layout[0]) == plan_order


@pytest.mark.parametrize("layer_id", LAYERS)
@pytest.mark.parametrize("rank", [0, 1])
def test_d_rank_residents_pad_row_and_store(boot, layer_id, rank):
    boot.group("D")
    d = _stage(_gguf_layer(layer_id, rank=rank))
    lo, hi = d._gguf_expert_range
    resident = boot.karte["phases"]["D"]["resident"][rank]
    for attr in gp.GGUF_EXPERT_ATTRS:
        bank = _bank(d, attr)
        # prefix residents, then the pad (zeros), in plan order
        for slot, g in enumerate(resident):
            assert torch.equal(bank[slot], _expert_bytes(layer_id, attr, g)), (attr, slot, g)
        assert not bank[len(resident)].any(), "the trailing pad expert must be a zero row"
        slot_of = boot.karte["phases"]["D"]["slot_of"]
        for g in range(lo, hi):
            if str(g) in slot_of:
                assert torch.equal(_store_row(boot, layer_id, attr, slot_of[str(g)]),
                                   _expert_bytes(layer_id, attr, g)), (attr, g)
    assert d._moe_offload_exchange_rows == len(resident)
    # the sentinel lists exactly the rows this rank wrote, stamped with the boot's identity
    path = es._sentinel(boot.store, f"L{layer_id}", "w13_qweight", rank)
    data = json.load(open(path))
    want = sorted(boot.karte["phases"]["D"]["slot_of"][str(g)] for g in range(lo, hi)
                  if str(g) in boot.karte["phases"]["D"]["slot_of"])
    assert data["rows"] == want and data["identity"] == "g4-toy"


def test_p_and_d_write_the_same_bytes_into_the_same_store_rows(boot):
    """The store is ONE file per (layer, tensor) for both groups: P writes first, D writes the same slots again --
    for every id both hold, identical bytes (so the overlap is harmless), and the union covers the cold set."""
    for layer_id in LAYERS:
        boot.group("P")
        _stage(_gguf_layer(layer_id))
        snapshot = {a: open(es.store_path(boot.store, f"L{layer_id}", a), "rb").read()
                    for a in gp.GGUF_EXPERT_ATTRS}
        boot.group("D")
        for rank in (0, 1):
            _stage(_gguf_layer(layer_id, rank=rank))
        for a in gp.GGUF_EXPERT_ATTRS:
            now = open(es.store_path(boot.store, f"L{layer_id}", a), "rb").read()
            n = len(now) // boot.karte["slots"]
            slot_of = boot.karte["phases"]["D"]["slot_of"]
            for g, slot in slot_of.items():
                got = now[slot * n:(slot + 1) * n]
                assert got == _expert_bytes(layer_id, a, int(g)).numpy().tobytes(), (layer_id, a, g)
                before = snapshot[a][slot * n:(slot + 1) * n]
                assert before == b"\0" * n or before == got  # P's row, if P wrote it, is unchanged by D


def test_three_row_classes_three_store_file_sizes(boot):
    boot.group("P")
    sizes = {}
    for layer_id in LAYERS:
        _stage(_gguf_layer(layer_id))
        for a in gp.GGUF_EXPERT_ATTRS:
            sizes[(layer_id, a)] = os.path.getsize(es.store_path(boot.store, f"L{layer_id}", a))
    slots = boot.karte["slots"]
    for layer_id in LAYERS:
        s13, s2 = ROW_SHAPES[layer_id]
        assert sizes[(layer_id, "w13_qweight")] == slots * s13[0] * s13[1]
        assert sizes[(layer_id, "w2_qweight")] == slots * s2[0] * s2[1]
    assert len({sizes[(i, "w13_qweight")] for i in LAYERS}) == 3
    # a store file of ANOTHER class under the same name is refused loudly, never aliased
    path = es.store_path(boot.store, "L1", "w13_qweight")
    with open(path, "r+b") as fh:
        fh.truncate(sizes[(0, "w13_qweight")])
    boot.group("D")
    with pytest.raises(ValueError, match="has .* bytes, this layout wants"):
        _stage(_gguf_layer(1, rank=0))


def test_the_release_tally_counts_the_rows_this_rank_owns(boot):
    boot.group("P")
    p = _stage(_gguf_layer(0))
    t = eo.expert_offload_release_totals()
    row = 8 * 64 + 4 * 32               # w13 + w2 bytes of ONE expert of layer 0
    bank_rows = _bank(p, "w13_qweight").shape[0]
    assert t.host_bytes == 3 * row      # this rank's own cold rows (5, 10, 11), not the whole shared file
    assert t.device_bytes == (TOTAL - bank_rows) * row
    assert t.layers == 1 and t.tensors == 2


# ---------------------------------------------------------------------------------------------------------------
# 3. the default boot is untouched: no store, no Karte, no seats -> the pre-G4 door, byte for byte
# ---------------------------------------------------------------------------------------------------------------


def test_without_store_karte_or_seats_the_pre_g4_door_runs(monkeypatch, tmp_path):
    monkeypatch.delenv(es.STORE_DIR_ENV, raising=False)
    monkeypatch.delenv(es.EXPERT_MAP_ENV, raising=False)
    monkeypatch.delenv("SGLANG_WEG2_D_SEAT_EXPERT_ROWS", raising=False)
    monkeypatch.setattr(es, "expert_map", lambda: None)
    monkeypatch.delenv("SGLANG_WEG2_GROUP", raising=False)
    eo.reset_expert_offload_release()
    d = _gguf_layer(0, rank=1)
    assert gp.door_wanted(d) is False
    _stage(d)
    # the private pool of the #123 door: published, staged, NO flip buffer, NO store file
    assert d._moe_offload_gguf_staged is True
    assert not hasattr(d, "weg2_experts_w13_qweight")
    assert not hasattr(d, "_gguf_presplit_plan")
    bank, spill = d._moe_offload_presplit["w13_qweight"]
    # trailing pad pinned (the pre-G4 pin), residents sorted by id, cold rows in the spill pool in id order
    ids, cold = d._moe_offload_frozen_layout
    assert d._moe_offload_pinned_experts == [6]
    assert sorted(ids) == list(ids) and 6 in ids
    for slot, local in enumerate(ids):
        want = _expert_bytes(0, "w13_qweight", 6 + local) if local < 6 else torch.zeros(8, 64, dtype=torch.uint8)
        assert torch.equal(bank[slot], want)
    for row, local in enumerate(cold):
        assert torch.equal(spill[row], _expert_bytes(0, "w13_qweight", 6 + local))


def test_the_door_is_wanted_for_each_thing_only_it_serves(boot, monkeypatch):
    d = _gguf_layer(0, rank=0)
    assert gp.door_wanted(d) is True                      # store dir + Karte (the fixture)
    monkeypatch.setattr(es, "expert_map", lambda: None)
    assert gp.door_wanted(d) is True                      # the store alone
    monkeypatch.delenv(es.STORE_DIR_ENV)
    assert gp.door_wanted(d) is False
    monkeypatch.setattr(es, "expert_map", lambda: boot.karte)
    assert gp.door_wanted(d) is True                      # the Karte alone
    monkeypatch.delenv(es.EXPERT_MAP_ENV)
    assert gp.door_wanted(d) is False                     # nothing published: the default boot answers without imports
    # a layer the GGUF half does not cover never takes the door
    assert gp.door_wanted(_gguf_layer(0, rank=0, eligible=False)) is False


def test_the_streaming_door_yields_to_the_platztausch_door(boot):
    """One door per layer: the streaming stager tiers into PRIVATE pools; with a store / Karte the layer is staged
    at materialization instead (the latch is where the two meet)."""
    from sglang.srt.environ import envs

    with envs.SGLANG_MOE_GGUF_STREAM_STAGING.override(True):
        assert _gguf_layer(0, rank=0)._gguf_stream_staging_enabled() is False
        boot.monkeypatch.delenv(es.STORE_DIR_ENV)
        boot.monkeypatch.delenv(es.EXPERT_MAP_ENV)
        boot.monkeypatch.setattr(es, "expert_map", lambda: None)
        on = _gguf_layer(0, rank=0)
        assert on._gguf_stream_staging_enabled() is True
        assert on._gguf_stream_stagers == {}


# ---------------------------------------------------------------------------------------------------------------
# 4. guards
# ---------------------------------------------------------------------------------------------------------------


def test_the_323b_whitelist_names_every_tiered_tensor():
    gp.assert_attrs_whitelisted()
    for attr in gp.GGUF_EXPERT_ATTRS:
        assert attr in eo.MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS
    assert tuple(sa.GGUF_EXPERT_ATTRS) == tuple(gp.GGUF_EXPERT_ATTRS)
    with pytest.raises(gp.GGUFPresplitRefused, match="absent from MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS"):
        gp.assert_attrs_whitelisted(("w13_qweight", "w_bogus_qweight"))


def test_the_offload_guard_admits_a_staged_gguf_layer_only(boot):
    boot.group("P")
    staged = _stage(_gguf_layer(0))
    eo.assert_expert_offload_quant_supported(staged.quant_method, layer_id=0, layer=staged)  # no raise
    plain = _gguf_layer(1)
    with pytest.raises(RuntimeError, match="_moe_offload_gguf_staged"):
        eo.assert_expert_offload_quant_supported(plain.quant_method, layer_id=1, layer=plain)


def test_a_row_that_is_not_whole_ggml_blocks_is_refused_by_name():
    assert gp.assert_row_geometry("w13_qweight", (8, 64), torch.uint8, 3) == 512
    with pytest.raises(gp.GGUFPresplitRefused, match="not a multiple of 16"):
        gp.assert_row_geometry("w2_qweight", (3, 5), torch.uint8, 3)
    with pytest.raises(gp.GGUFPresplitRefused, match="uint8"):
        gp.assert_row_geometry("w2_qweight", (4, 32), torch.bfloat16, 3)


def test_a_missing_expert_shifts_every_local_id_and_is_refused(boot):
    """``_gguf_expert_source`` numbers experts by position among the ones it received: one expert missing from the
    loader shifts every later local id, and the Karte / the store rows would address the wrong experts."""
    boot.group("D")
    d = _gguf_layer(0, rank=0)
    for attr in gp.GGUF_EXPERT_ATTRS:
        p = getattr(d, attr)
        for key in [k for k in p.expert_data_map if k[0] == 2]:
            del p.expert_data_map[key]
    with pytest.raises(gp.GGUFPresplitRefused, match="loader delivered 6 experts, the layer owns 7"):
        _stage(d)


def test_an_even_tp_intermediate_shard_has_no_store_to_share(boot):
    """Rows that are slices of an expert cannot sit in a store keyed by whole global experts."""
    boot.group("D")
    d = _gguf_layer(0, rank=0)
    d._gguf_expert_shard = False
    d.moe_tp_size = 2
    with pytest.raises(gp.GGUFPresplitRefused, match="intermediate-dim tensor-parallel shard"):
        _stage(d)


def test_a_layer_the_half_does_not_stage_is_refused_under_a_karte_not_after_the_stack(boot):
    """W120 for GGUF: the Karte gives layer 0 a Platztausch buffer; a ggml type without a MoE kernel keeps the
    plain stack, which the other group has no counterpart for. Refused before anything is allocated."""
    boot.group("D")
    bad = _gguf_layer(0, rank=0, eligible=False)
    with pytest.raises(RuntimeError, match="W120 Weg2PlatztauschBufferUnbuilt: layer 0"):
        _stage(bad)
    assert all(isinstance(p, GGUFUninitializedParameter) for p in (bad.w13_qweight, bad.w2_qweight))
    # the excluded draft is outside the Karte by design
    draft = _gguf_layer(0, rank=0, eligible=False)
    draft._moe_offload_excluded = True
    _stage(draft)
    # and without a nested Karte nothing changes for a layer the half declines
    boot.monkeypatch.delenv(es.EXPERT_MAP_ENV)
    boot.monkeypatch.setattr(es, "expert_map", lambda: None)
    _stage(_gguf_layer(0, rank=0, eligible=False))


def test_a_karte_that_pins_every_row_of_a_gguf_layer_is_refused(boot):
    boot.group("D")
    d = _gguf_layer(0, rank=0, fraction=1.0 - 1e-9)       # fraction so high that R >= E
    with pytest.raises(RuntimeError, match="W120|Platztausch-Karte nennt"):
        _stage(d)


# ---------------------------------------------------------------------------------------------------------------
# 4b. the real unsloth row shapes: every class passes the geometry guard and the pool copy's word view
# ---------------------------------------------------------------------------------------------------------------

#: (class, w13 row shape = gate and up concatenated on the row axis, w2 row shape), bytes per row from the header:
#: A gate/up IQ3_S (2560 elements -> 1100 B) down IQ4_NL (640 -> 360 B); B down Q8_0 (640 -> 680 B);
#: C gate/up IQ4_XS (2560 -> 1360 B) down Q8_0
REAL_CLASSES = {
    "A": ((1280, 1100), (2560, 360), 2329600),
    "B": ((1280, 1100), (2560, 680), 3148800),
    "C": ((1280, 1360), (2560, 680), 3481600),
}


@pytest.mark.parametrize("cls", sorted(REAL_CLASSES))
def test_the_real_row_classes_pass_the_guard_and_the_pool_copy_view(cls):
    from sglang.srt.layers.moe import expert_pool_device as epd

    s13, s2, total = REAL_CLASSES[cls]
    b13 = gp.assert_row_geometry("w13_qweight", s13, torch.uint8, 0)
    b2 = gp.assert_row_geometry("w2_qweight", s2, torch.uint8, 0)
    assert b13 + b2 == total
    for shape in (s13, s2):
        bank = torch.arange(3 * shape[0] * shape[1], dtype=torch.int64).remainder(251).to(torch.uint8)
        bank = bank.reshape((3,) + shape)
        words = epd._word_rows(bank)
        assert words.dtype == torch.int32 and tuple(words.shape) == (3, shape[0] * shape[1] // 4)
        assert torch.equal(words.view(torch.uint8).reshape(bank.shape), bank)
        # the reference row copy (what the CPU path of ``copy_rows`` runs) moves a row whole
        dst = torch.zeros_like(bank)
        epd.copy_rows_reference([bank], [dst], [(2, 0), (0, 1)])
        assert torch.equal(dst[0], bank[2]) and torch.equal(dst[1], bank[0]) and not dst[2].any()


# ---------------------------------------------------------------------------------------------------------------
# 5. store identity: a GGUF source reads its header; INT4 does not move
# ---------------------------------------------------------------------------------------------------------------


def _legacy_identity(model, map_path):
    """``compute_identity`` as it was before G4 / H88-C (c651892375), verbatim."""
    if not map_path or not os.path.isfile(map_path):
        return ""
    h = hashlib.sha256()
    h.update(b"h2c-v1\0")
    if os.path.isdir(model):
        for name in sorted(os.listdir(model)):
            if name == "config.json" or name.endswith(".safetensors.index.json"):
                with open(os.path.join(model, name), "rb") as fh:
                    h.update(name.encode() + b"\0" + fh.read() + b"\0")
            elif name.endswith(".safetensors"):
                h.update(f"{name}\0{os.path.getsize(os.path.join(model, name))}\0".encode())
    else:
        h.update(b"model-id\0" + str(model).encode() + b"\0")
    with open(map_path, "rb") as fh:
        h.update(b"map\0" + fh.read())
    return h.hexdigest()[:24]


def _int4_dir(root, shard_size=10):
    os.makedirs(root, exist_ok=True)
    open(os.path.join(root, "config.json"), "wb").write(b'{"num_experts": 512}')
    open(os.path.join(root, "model.safetensors.index.json"), "wb").write(b'{"weight_map": {}}')
    open(os.path.join(root, "model-00001.safetensors"), "wb").write(b"\0" * shard_size)
    return root


def _write_gguf(path, layer_types, *, rows=2, experts=2, kv=None):
    """A tiny qwen4exp GGUF: per layer gate/up/down expert tensors of the given ggml types."""
    import gguf
    import numpy as np

    Q = gguf.GGMLQuantizationType
    w = gguf.GGUFWriter(path, "qwen4exp")
    w.add_block_count(len(layer_types))
    for k, v in (kv or {}).items():
        w.add_uint32(k, v)
    for i, (g, u, d) in enumerate(layer_types):
        for proj, t in (("gate", g), ("up", u), ("down", d)):
            qt = getattr(Q, t)
            _blk, tsize = gguf.GGML_QUANT_SIZES[qt]
            arr = np.full((experts, rows, tsize), 1 + i, dtype=np.uint8)
            w.add_tensor(f"blk.{i}.ffn_{proj}_exps.weight", arr, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.fixture
def idmap(tmp_path):
    p = tmp_path / "karte.json"
    p.write_text(json.dumps(_karte()))
    return str(p)


def test_the_int4_identity_does_not_move(tmp_path, idmap):
    model = _int4_dir(str(tmp_path / "int4"))
    assert es.compute_identity(model, idmap) == _legacy_identity(model, idmap)
    assert es.compute_identity(model, idmap, es.DEFAULT_LAYOUT) == _legacy_identity(model, idmap)
    assert es.compute_identity("some/hf-model-id", idmap) == _legacy_identity("some/hf-model-id", idmap)
    assert es.compute_identity(model, "") == ""
    # H88-C's non-default layout tag still enters the hash (same text as origin/desk/nf-h88-ident-1008)
    assert es.compute_identity(model, idmap, "marlin_w4a8") != _legacy_identity(model, idmap)


REAL_INT4 = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist-abl-wxp"


@pytest.mark.skipif(not os.path.isdir(REAL_INT4), reason="the INT4 checkpoint is not on this machine")
def test_the_real_int4_checkpoint_identity_is_the_pre_g4_one(idmap):
    assert es.compute_identity(REAL_INT4, idmap) == _legacy_identity(REAL_INT4, idmap)


def test_a_gguf_identity_reads_the_header_not_the_path(tmp_path, idmap):
    types_a = [("IQ3_S", "IQ3_S", "IQ4_NL"), ("IQ3_S", "IQ3_S", "Q8_0")]
    a = tmp_path / "a"
    a.mkdir()
    _write_gguf(str(a / "m-00001-of-00001.gguf"), types_a)
    ident = es.compute_identity(str(a), idmap)
    assert ident and ident != _legacy_identity(str(a), idmap)
    assert es.compute_identity(str(a), idmap) == ident
    # the same bytes under another path: the same identity (header digest instead of path)
    b = tmp_path / "elsewhere"
    b.mkdir()
    (b / "m-00001-of-00001.gguf").write_bytes((a / "m-00001-of-00001.gguf").read_bytes())
    assert es.compute_identity(str(b), idmap) == ident
    # a .gguf FILE path is read too, not hashed by name
    single = tmp_path / "single.gguf"
    single.write_bytes((a / "m-00001-of-00001.gguf").read_bytes())
    other = tmp_path / "renamed.gguf"
    other.write_bytes(single.read_bytes())
    assert es.compute_identity(str(single), idmap) == es.compute_identity(str(other), idmap)
    assert es.compute_identity(str(single), idmap) != _legacy_identity(str(single), idmap)


def test_a_requantized_tensor_changes_the_identity(tmp_path, idmap):
    def ident(name, layer_types, **kw):
        d = tmp_path / name
        d.mkdir()
        _write_gguf(str(d / "m.gguf"), layer_types, **kw)
        return es.compute_identity(str(d), idmap)

    base = ident("base", [("IQ3_S", "IQ3_S", "IQ4_NL"), ("IQ3_S", "IQ3_S", "Q8_0")])
    # layer 1 down: Q8_0 -> IQ4_NL (the type set moves: layout tag)
    assert ident("t1", [("IQ3_S", "IQ3_S", "IQ4_NL"), ("IQ3_S", "IQ3_S", "IQ4_NL")]) != base
    # layer 0 gate/up: IQ3_S -> IQ4_XS
    assert ident("t2", [("IQ4_XS", "IQ4_XS", "IQ4_NL"), ("IQ3_S", "IQ3_S", "Q8_0")]) != base
    # same types, another shape (the header digest moves, the layout tag does not)
    assert ident("t3", [("IQ3_S", "IQ3_S", "IQ4_NL"), ("IQ3_S", "IQ3_S", "Q8_0")], rows=3) != base
    # same types and shapes, another KEY in the header
    assert ident("t4", [("IQ3_S", "IQ3_S", "IQ4_NL"), ("IQ3_S", "IQ3_S", "Q8_0")], kv={"qwen4exp.x": 1}) != base
    # the identical file again: the identical identity
    assert ident("t5", [("IQ3_S", "IQ3_S", "IQ4_NL"), ("IQ3_S", "IQ3_S", "Q8_0")]) == base


def test_the_layout_tag_is_a_function_of_the_types_only(tmp_path):
    a = _write_gguf(str(tmp_path / "a.gguf"), [("IQ3_S", "IQ3_S", "IQ4_NL"), ("IQ3_S", "IQ3_S", "Q8_0")])
    b = _write_gguf(str(tmp_path / "b.gguf"), [("IQ3_S", "IQ3_S", "IQ4_NL"), ("IQ3_S", "IQ3_S", "Q8_0")], rows=4)
    ta = gl.gguf_layout_tag(gl.expert_types_per_layer([a]))
    assert ta == gl.gguf_layout_tag(gl.expert_types_per_layer([b]))
    assert gl.is_gguf_layout(ta) and ta.startswith("gguf:") and len(ta) == len("gguf:") + 16
    assert not gl.is_gguf_layout("marlin_w4a16") and not gl.is_gguf_layout("gguf:")
    assert gl.expert_types_per_layer([a]) == {
        0: {"gate": "IQ3_S", "up": "IQ3_S", "down": "IQ4_NL"},
        1: {"gate": "IQ3_S", "up": "IQ3_S", "down": "Q8_0"},
    }


def test_an_incomplete_gguf_set_is_a_named_error_not_a_guessed_identity(tmp_path, idmap):
    one = _write_gguf(str(tmp_path / "m-00001-of-00002.gguf"), [("IQ3_S", "IQ3_S", "IQ4_NL")])
    with pytest.raises(OSError, match="split set incomplete"):
        es.compute_identity(one, idmap)


REAL_GGUF = ("/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-GGUF-unsloth/UD-IQ4_XS/"
             "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")


@pytest.mark.skipif(not os.path.isfile(REAL_GGUF), reason="the unsloth export is not on this machine")
def test_the_real_unsloth_header_has_three_row_classes_and_a_stable_identity(idmap):
    files = gl.source_files(REAL_GGUF)
    assert len(files) == 3
    types = gl.expert_types_per_layer(files)
    assert len(types) == 48
    from sglang.srt.weg2 import xchg_gguf_census as xc

    rows = xc.layer_row_bytes(files)
    classes = xc.row_classes(rows)
    # the classes of deskq/gguf-nf/calc.py, recomputed here from the header (not copied)
    assert {b: ls for b, ls in classes.items()} == {
        2329600: [l for l in range(48) if l not in (2, 4, 30, 46, 47)],
        3148800: [4, 30, 46, 47],
        3481600: [2],
    }
    assert [round(b / 2**20, 4) for b in sorted(classes)] == [2.2217, 3.0029, 3.3203]
    t1 = es.compute_identity(REAL_GGUF, idmap)
    assert t1 and t1 == es.compute_identity(os.path.dirname(REAL_GGUF), idmap)


# ---------------------------------------------------------------------------------------------------------------
# 5b. H95c seat rows behind the bank
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
def seats(monkeypatch):
    from sglang.srt.weg2 import d_seat_vram as sv

    calls = []
    monkeypatch.setattr(sv, "presplit_seat_rows", lambda layer: 2)

    def fake_seat(*, rows, extra, tail, dtype, device, in_tag_pool=True, name="", spans=None, granule=None):
        calls.append(dict(rows=rows, extra=extra, tail=tuple(tail), dtype=dtype, name=name))
        return torch.zeros((rows + extra,) + tuple(tail), dtype=dtype)

    monkeypatch.setattr(sv, "seat_expert_buffer", fake_seat)
    return calls


def test_seat_rows_ride_behind_the_bank_of_a_store_boot(boot, seats):
    boot.group("D")
    d = _stage(_gguf_layer(0, rank=0))
    assert [c["extra"] for c in seats] == [2, 2] and seats[0]["name"] == "layer 0 w13_qweight"
    for attr in gp.GGUF_EXPERT_ATTRS:
        bank, _spill = d._moe_offload_presplit[attr]
        plan_rows = seats[0]["rows"]
        assert bank.shape[0] == plan_rows + 2
        for slot, g in enumerate(boot.karte["phases"]["D"]["resident"][0]):
            assert torch.equal(bank[slot], _expert_bytes(0, attr, g))
        assert not bank[plan_rows:].any(), "the seat rows are untouched"
    assert d._weg2_seat_rows == 2
    # the row geometry handed to the seat allocator is THIS layer's class, per tensor
    assert seats[0]["tail"] == ROW_SHAPES[0][0] and seats[1]["tail"] == ROW_SHAPES[0][1]


def test_a_seat_only_boot_takes_the_door_with_a_private_pool(monkeypatch, seats):
    monkeypatch.delenv(es.STORE_DIR_ENV, raising=False)
    monkeypatch.delenv(es.EXPERT_MAP_ENV, raising=False)
    monkeypatch.setenv("SGLANG_WEG2_D_SEAT_EXPERT_ROWS", "2")
    monkeypatch.setattr(es, "expert_map", lambda: None)
    monkeypatch.delenv("SGLANG_WEG2_GROUP", raising=False)
    d = _gguf_layer(1, rank=1)
    assert gp.door_wanted(d) is True
    _stage(d)
    bank, spill = d._moe_offload_presplit["w13_qweight"]
    ids, cold = d._moe_offload_frozen_layout
    for slot, local in enumerate(ids):
        want = _expert_bytes(1, "w13_qweight", 6 + local) if local < 6 else torch.zeros(8, 96, dtype=torch.uint8)
        assert torch.equal(bank[slot], want)
    for row, local in enumerate(cold):
        assert torch.equal(spill[row], _expert_bytes(1, "w13_qweight", 6 + local))
    assert d._weg2_seat_rows == 2
    # no Karte: the whole bank is published to the flip, not a prefix
    assert getattr(d, "weg2_experts_w13_qweight").shape[0] == bank.shape[0]


# ---------------------------------------------------------------------------------------------------------------
# 6. D-store-adopt on a GGUF layer: ``repack_rows`` is a PURE ROW COPY here (nothing to repack)
# ---------------------------------------------------------------------------------------------------------------


def _arm_adopt(boot, monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    ctx = [envs.SGLANG_WEG2_ENABLE_D_STORE_ADOPT.override(True),
           envs.SGLANG_MOE_REPACK_SKIP_VETOED.override(True)]
    for c in ctx:
        c.__enter__()
    return ctx


def _publish_p(boot, layer_id):
    boot.group("P")
    _stage(_gguf_layer(layer_id))
    sa.reset_for_tests()


def test_the_veto_set_of_a_gguf_shard_is_its_cold_published_rows(boot, monkeypatch):
    _publish_p(boot, 0)
    ctx = _arm_adopt(boot, monkeypatch)
    try:
        d = _gguf_layer(0, rank=1)
        d._moe_store_adopt_ok = True          # what the scheme's ``discount_expected`` arms (GGUF: the door's caller)
        # D rank 1 owns 6..11, holds 6..9 resident: its cold ids 10, 11 are P's published rows (slots 2, 3);
        # the pad is nobody's row; P's own extra expert 4 is not in this rank's window
        assert sa.vetoed_global_ids(d) == frozenset({10, 11})
        # the rows the "repack" (a plain row copy for GGUF) must still touch: everything but the vetoed,
        # the pad (the LAST local row of a trailing-pad shard) kept
        assert sa.repack_rows(d, 7) == [0, 1, 2, 3, 6]
        # the generic (leading-pad) shard keeps ITS pad at row 0 -- unchanged
        g = types.SimpleNamespace(num_experts=TOTAL, num_local_experts=7, _expert_shard_generic=True,
                                  _gguf_expert_range=(6, 12), _moe_store_adopt_vetoed_global=frozenset({10, 11}))
        assert sa.repack_rows(g, 7) == [0, 1, 2, 3, 4]
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)


def test_adopted_rows_are_neither_read_nor_rewritten(boot, monkeypatch):
    """D's loader still delivers every expert here (the GGUF stream is not vetoed), but the door never READS a
    vetoed one: poison bytes in experts 10 and 11 must not reach the store, P's rows stay P's."""
    _publish_p(boot, 0)
    good = {g: _store_row(boot, 0, "w13_qweight", s) for g, s in
            ((10, boot.karte["phases"]["D"]["slot_of"]["10"]), (11, boot.karte["phases"]["D"]["slot_of"]["11"]))}
    ctx = _arm_adopt(boot, monkeypatch)
    try:
        d = _gguf_layer(0, rank=1, poison=(10, 11))
        d._moe_store_adopt_ok = True
        assert sa.vetoed_global_ids(d) == frozenset({10, 11})
        _stage(d)
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    for g, row in good.items():
        slot = boot.karte["phases"]["D"]["slot_of"][str(g)]
        assert torch.equal(_store_row(boot, 0, "w13_qweight", slot), row)
        assert torch.equal(row, _expert_bytes(0, "w13_qweight", g))
    # D's sentinel still lists the adopted rows: they are valid store rows (P's bytes)
    data = json.load(open(es._sentinel(boot.store, "L0", "w13_qweight", 1)))
    assert data["rows"] == sorted(boot.karte["phases"]["D"]["slot_of"][str(g)] for g in (10, 11))
    # the residents were read, from the checkpoint, as ever
    bank = _bank(d, "w13_qweight")
    for slot, g in enumerate(boot.karte["phases"]["D"]["resident"][1]):
        assert torch.equal(bank[slot], _expert_bytes(0, "w13_qweight", g))


def test_a_vetoed_resident_is_refused_not_loaded_as_garbage(boot):
    boot.group("D")
    d = _gguf_layer(0, rank=1)
    d._moe_store_adopt_vetoed_global = frozenset({6})     # expert 6 is RESIDENT on rank 1
    with pytest.raises(sa.StoreAdoptBroken, match="RESIDENT"):
        _stage(d)


# ---------------------------------------------------------------------------------------------------------------
# 7. the flip moves BYTES: join, plan and memmove over the buffers the door published, both directions
# ---------------------------------------------------------------------------------------------------------------


def _flip_manifest(group, rank, name, tensor, tag):
    from sglang.srt.weg2 import weight_exchange as wx
    from sglang.srt.weg2 import xchg_manifest as xm

    geom = wx.ParamGeom.of(tensor, name=name, tag=tag, shard_axis=wx.REPLICATED, shard_total=0, stage=rank)
    return xm.RankManifest(group=group, rank=rank, card=rank, region_tag="weights", boot_token="t",
                           pieces=xm.pieces_from_inventory([geom]))


def _apply_plan(plan):
    import ctypes

    from sglang.srt.weg2 import weight_exchange as wx

    for d in plan.descs:
        if d.kind == wx.ZEROFILL:
            ctypes.memset(d.dst_ptr + d.dst_off, 0, d.nbytes)
        elif d.kind == wx.FLAT:
            ctypes.memmove(d.dst_ptr + d.dst_off, d.src_ptr + d.src_off, d.nbytes)
        else:
            assert d.kind == wx.STRIDED2D, d
            for r in range(d.rows):
                ctypes.memmove(d.dst_ptr + d.dst_off + r * d.dpitch,
                               d.src_ptr + d.src_off + r * d.spitch, d.run_bytes)


@pytest.mark.parametrize("layer_id", LAYERS)
@pytest.mark.parametrize("direction", ["pp_to_tp", "tp_to_pp"])
def test_the_flip_reproduces_every_gguf_row_byte(boot, layer_id, direction):
    from sglang.srt.weg2 import weight_exchange as wx
    from sglang.srt.weg2 import xchg_manifest as xm

    attr = "w13_qweight"
    name = f"model.layers.{layer_id}.mlp.experts.weg2_experts_{attr}"
    tag = "weights_0"
    boot.group("P")
    p = _stage(_gguf_layer(layer_id))
    boot.group("D")
    ds = [_stage(_gguf_layer(layer_id, rank=r)) for r in (0, 1)]
    p_buf = getattr(p, "weg2_experts_" + attr)
    d_bufs = [getattr(d, "weg2_experts_" + attr) for d in ds]
    # the published views are the PREFIXES: P's is the union of D's, in id order -- a plain row cut
    assert p_buf.shape[0] == sum(b.shape[0] for b in d_bufs) == 8
    assert p_buf.dtype == torch.uint8 and all(b.dtype == torch.uint8 for b in d_bufs)
    mans = [_flip_manifest("P", 1, name, p_buf, tag)] + [
        _flip_manifest("D", r, name, b, tag) for r, b in enumerate(d_bufs)]
    xm.clear_join_memo()
    join = xm.join_manifests(mans)
    assert join.tensors[0].shard_axis == wx.ROWS
    want = {("P", 1): bytes(p_buf.contiguous().numpy().tobytes()),
            **{("D", r): bytes(b.contiguous().numpy().tobytes()) for r, b in enumerate(d_bufs)}}
    tp_is_dst = direction == "pp_to_tp"
    if tp_is_dst:
        for b in d_bufs:
            b.view(-1).fill_(0xEE)
    else:
        p_buf.view(-1).fill_(0xEE)
    def p_addr(n, rank):
        return p_buf.data_ptr() if rank == 1 else None

    def d_addr(n, rank):
        return d_bufs[rank].data_ptr()

    plan = xm.plan_from_join(
        join,
        direction=wx.LEGS_PP_TO_TP if tp_is_dst else wx.LEGS_TP_TO_PP,
        waves=[[tag]],
        src_addr=(p_addr if tp_is_dst else d_addr),
        dst_addr=(d_addr if tp_is_dst else p_addr),
    )
    assert plan.descs and all(d.kind in (wx.FLAT, wx.STRIDED2D) for d in plan.descs)
    _apply_plan(plan)
    if tp_is_dst:
        for r, b in enumerate(d_bufs):
            assert b.contiguous().numpy().tobytes() == want[("D", r)], r
    else:
        assert p_buf.contiguous().numpy().tobytes() == want[("P", 1)]
    # the exchange moved exactly the prefix: the bytes in flight are rows x row bytes, nothing re-blocked
    row = ROW_SHAPES[layer_id][0][0] * ROW_SHAPES[layer_id][0][1]
    assert sum(d.nbytes for d in plan.descs) == 8 * row


def test_the_gguf_buffer_is_one_contiguous_uint8_block_the_flip_can_name(boot):
    """``StorageGeom`` flattens ``[prefix, rows, bytes]`` to ``rows = prefix * rows-per-expert`` -- one expert is
    ``rows`` storage rows of ``bytes`` each, itemsize 1: the exchange sees bytes, not ggml blocks."""
    from sglang.srt.weg2 import weight_exchange as wx

    boot.group("D")
    d = _stage(_gguf_layer(1, rank=0))
    pub = getattr(d, "weg2_experts_w13_qweight")
    g = wx.StorageGeom.of(pub)
    assert (g.rows, g.cols, g.itemsize, g.contiguous) == (4 * 8, 96, 1, True)
    from sglang.srt.managers import weg2_memory_saver as ms

    assert ms.is_expert_buffer_attr("model.layers.1.mlp.experts.weg2_experts_w13_qweight")


# ---------------------------------------------------------------------------------------------------------------
# 8. W71 census: the expert bank per tag from the GGUF header
# ---------------------------------------------------------------------------------------------------------------


def test_the_int4_row_is_the_calc_formula():
    from sglang.srt.weg2 import xchg_gguf_census as xc

    assert xc.int4_row_bytes() == 2534400
    assert round(xc.int4_row_bytes() / 2**20, 4) == 2.4170


def test_bank_bytes_are_summed_per_layer_not_a_mean_row():
    from sglang.srt.weg2 import xchg_gguf_census as xc

    MiB = 2**20
    rows = {0: 2 * MiB, 1: 2 * MiB, 2: 4 * MiB, 3: 2 * MiB, 4: 2 * MiB, 5: 8 * MiB}
    by_tag = xc.bank_mib_by_tag(rows, range(6), 10, chunk_layers=3)
    assert by_tag == {"weights_0": 80.0, "weights_1": 120.0}
    mean_row = sum(rows.values()) / 6
    assert 6 * 10 * mean_row / MiB == 200.0 and sum(by_tag.values()) == 200.0   # equal in total ...
    assert by_tag["weights_0"] != 3 * 10 * mean_row / MiB                       # ... but not per tag
    assert xc.row_classes(rows) == {2 * MiB: [0, 1, 3, 4], 4 * MiB: [2], 8 * MiB: [5]}
    assert xc.tag_of_layer(47, 3) == "weights_15" and xc.tag_of_layer(0, 3) == "weights_0"


def test_swap_expert_bank_replaces_only_the_bank_term_and_rounds_up():
    from sglang.srt.weg2 import xchg_gguf_census as xc

    measured = {"weights_0": 1000, "weights_1": 900, "weights": 50}
    old = {"weights_0": 600.0, "weights_1": 600.0}
    new = {"weights_0": 580.2, "weights_1": 640.0}
    got = xc.swap_expert_bank(measured, old, new)
    assert got == {"weights_0": 981, "weights_1": 940, "weights": 50}     # ceil(1000-600+580.2), 900-600+640
    with pytest.raises(ValueError, match="not a bank of this census"):
        xc.swap_expert_bank({"weights_0": 100}, {"weights_0": 600.0}, {"weights_0": 1.0})


def test_w71_solves_over_a_census_with_gguf_bank_bytes():
    """The swapped census is a census W71 reads: the peak moves by exactly the bank bytes that changed."""
    from sglang.srt.weg2 import xchg_gguf_census as xc
    from sglang.srt.weg2 import xchg_residency as xr

    card = types.SimpleNamespace(uuid="GPU-g4", nvml_index=0, name="RTX 5090", total_mib=32000)

    def peak(d_tags, p_tags):
        census = xr.XchgCensus(
            cards={"GPU-g4": xr.CardCensus(uuid="GPU-g4", tags={"D": d_tags, "P": p_tags},
                                           dormant_proc_used_mib=500, dormant_source="t")},
            waves=(("weights_0", "weights_1"),), provenance="t")
        res = xr.solve([card], census, floor_mib=1229.0)
        return res.peak_mib("d2p", "GPU-g4"), res.peak_mib("p2d", "GPU-g4")

    int4_d = {"weights_0": 4000, "weights_1": 4000}
    int4_p = {"weights_0": 3000, "weights_1": 3000}
    base = peak(int4_d, int4_p)
    old = {"weights_0": 3000.0, "weights_1": 3000.0}
    d_new = xc.swap_expert_bank(int4_d, old, {"weights_0": 2900.0, "weights_1": 3100.0})
    p_new = xc.swap_expert_bank(int4_p, old, {"weights_0": 3300.0, "weights_1": 2800.0})
    moved = peak(d_new, p_new)
    assert d_new == {"weights_0": 3900, "weights_1": 4100} and p_new == {"weights_0": 3300, "weights_1": 2800}
    # one wave holding both tags: peak = image_S + taken_W (+ overhead), so both directions move by dD + dP
    assert moved[0] - base[0] == (3900 + 4100 + 3300 + 2800) - (4000 + 4000 + 3000 + 3000)
    assert moved[1] == moved[0]


def test_the_census_table_for_the_reference_form_from_the_real_header():
    """The table of the G4 report: slots a budget buys with INT4 rows and with the three GGUF row classes, and the
    bank MiB per chunk tag. Budgets are the x177 measurement quoted in PLAN-GGUF-NF-1009 section 4."""
    if not os.path.isfile(REAL_GGUF):
        pytest.skip("the unsloth export is not on this machine")
    from sglang.srt.weg2 import xchg_gguf_census as xc

    rows = xc.layer_row_bytes(gl.source_files(REAL_GGUF))
    places = [
        xc.Place("P-PP0 5090", tuple(range(0, 29)), 13.83 * 1024),
        xc.Place("P-PP1 3080", tuple(range(29, 40)), 9.35 * 1024),
        xc.Place("P-PP2 3080", tuple(range(40, 48)), 7.70 * 1024),
        xc.Place("D-TP0 5090", tuple(range(48)), 14.73 * 1024),
    ]
    table = xc.compare(places, rows, xc.int4_row_bytes(), chunk_layers=3)
    by = {r["place"]: r for r in table}
    # the INT4 slot counts of the plan document (202 / 360 / 407.8 / 130) and the GGUF ones (213.6 / 379.6 / 407.8 /
    # 136.1) -- recomputed from the header, so a different export moves them
    assert round(by["P-PP0 5090"]["slots_int4"]) == 202 and round(by["P-PP0 5090"]["slots_gguf"], 1) == 213.6
    assert round(by["P-PP1 3080"]["slots_int4"]) == 360 and round(by["P-PP1 3080"]["slots_gguf"], 1) == 379.6
    assert round(by["P-PP2 3080"]["slots_int4"], 1) == 407.8 and round(by["P-PP2 3080"]["slots_gguf"], 1) == 407.8
    assert round(by["D-TP0 5090"]["slots_int4"]) == 130 and round(by["D-TP0 5090"]["slots_gguf"], 1) == 136.1
    # a bank never exceeds its budget (floor of the slots)
    for r in table:
        assert r["bank_gguf_mib"] <= r["budget_mib"] + 1e-6
    md = xc.render_markdown(table)
    assert md.count("\n") == len(table) + 1


def test_the_default_boot_probe_spellings_are_the_real_env_names():
    from sglang.srt.environ import envs

    assert gp._BOOT_ENVS == (es.STORE_DIR_ENV, es.EXPERT_MAP_ENV, envs.SGLANG_WEG2_D_SEAT_EXPERT_ROWS.name)


def test_the_default_boot_imports_nothing_new(monkeypatch):
    """With none of the three envs set the door answers False, and refuses nothing, without touching the Karte /
    store modules (the host-RSS meter of ``test_gguf_host_residency_644`` runs this path)."""
    for k in gp._BOOT_ENVS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(es, "expert_map", lambda: (_ for _ in ()).throw(AssertionError("Karte asked")))
    monkeypatch.setattr(es, "store_enabled", lambda: (_ for _ in ()).throw(AssertionError("store asked")))
    d = _gguf_layer(0, rank=0, eligible=False)
    assert gp.boot_wants_platztausch(d) is False
    assert gp.door_wanted(d) is False
    gp.refuse_unstaged_platztausch(d, why="x")  # no raise
