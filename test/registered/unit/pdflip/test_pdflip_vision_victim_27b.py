"""VISION-WEIGHTS AP2 (plan PLAN-VISION-GEWICHTE-VERDRAENGEN-1009 §2/§4): the
27B victims -- flip ``dense`` (MLP storages of PP0) and dual ``pp_only``
(only the hull parts D does not share) -- and the row split of a tower
tensor larger than every victim run.

Hermetic, CPU (``device_types=("cpu",)`` stands in for the card).

Pinned:
  * T2: the victim inventory never holds a storage the shared part reaches
    (params, buffers, attributes, an alias from another part), a storage in
    an attached union arena, embed_tokens/lm_head, a draft's storage; it is
    deduplicated and deterministic; flip takes only ``.mlp.`` storages;
  * the source per rank: pp_only under FLLIPER_PDFLIP_DUAL_SHARE with hull
    parts, dense otherwise;
  * T3: the row split takes the fewest pieces that fit, computes the unsplit
    numbers (up to summation order), and a whole stage on victims smaller than
    a tower tensor encodes the checkpoint's numbers and returns the victims;
    a tensor that cannot be split is W105b;
  * M0 arithmetic on the real 27B merger sizes: 12 MiB runs (dual) split the
    two merger linears, 170 MiB runs (flip) split nothing.
"""

import types

import pytest
import torch
import torch.nn.functional as F

from flliper.srt.pdflip import vision_rank_runner as vrr
from flliper.srt.pdflip import vision_rank_stage as vrs
from flliper.srt.pdflip import vision_victim as vv
from flliper.srt.pdflip import vision_victim_27b as v27
from test_pdflip_vision_rank_runner import _Alloc, _Item, _kv, _req, _Tower, _write_model  # noqa: E402

CPU = ("cpu",)
MIB = 1 << 20


class _Part(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 8)
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].mlp = torch.nn.Linear(8, 16)
        self.layers[0].self_attn = torch.nn.Linear(8, 8)
        self.register_buffer("workspace", torch.zeros(32))


def _names(src):
    return [n for n, _v in src._named_storages()]


def test_T2_pp_only_never_takes_what_D_shares_and_is_deterministic():
    parts = [_Part() for _ in range(3)]
    local = 1
    # an alias INTO the shared part, held by a non-local part's parameter
    parts[2].layers[0].self_attn.weight = torch.nn.Parameter(
        parts[local].layers[0].mlp.weight.view(8, 16).t(), requires_grad=False)
    # a plain attribute of the shared part (a kernel workspace) aliasing part 0's bias
    parts[local].layers[0].mlp.captured = parts[0].layers[0].self_attn.bias
    arena_t = parts[0].layers[0].mlp.bias          # lies in D's attached arena
    st = arena_t.untyped_storage()
    src = v27.PpOnlyVictims(parts, local, arena_ranges=[(st.data_ptr(), st.data_ptr() + st.nbytes())],
                            device_types=CPU)
    names = _names(src)
    assert names == _names(src)                    # deterministic
    assert names == ["part0.layers.0.mlp.weight", "part0.layers.0.self_attn.weight",
                     "part2.layers.0.mlp.weight", "part2.layers.0.mlp.bias",
                     "part2.layers.0.self_attn.bias"]
    assert not any(n.startswith(f"part{local}.") or "embed_tokens" in n for n in names)
    with pytest.raises(vv.VisionVictimPlanRefused, match="shared part 3"):
        v27.PpOnlyVictims(parts, 3, device_types=CPU)


def test_T2_dense_takes_only_mlp_storages_dedups_aliases_and_spares_the_draft():
    model = torch.nn.Module()
    model.model = _Part()
    model.model.layers.append(torch.nn.Module())
    model.model.layers[1].mlp = torch.nn.Linear(8, 16)
    model.lm_head = torch.nn.Linear(8, 4)
    model.model.layers[0].mlp.alias = torch.nn.Parameter(model.model.layers[0].mlp.weight[:4],
                                                         requires_grad=False)   # same storage
    draft = torch.nn.Module()
    draft.shared = model.model.layers[1].mlp.bias      # the draft reaches this storage
    src = v27.DenseMlpVictims(model, exclude=[draft], device_types=CPU)
    assert _names(src) == ["model.layers.0.mlp.weight", "model.layers.0.mlp.bias",
                           "model.layers.1.mlp.weight"]


def test_the_source_per_rank_is_pp_only_under_dual_share_and_dense_otherwise(monkeypatch):
    runner = types.SimpleNamespace(model=torch.nn.Module(), dual_share_part_models=[_Part(), _Part()],
                                   dual_share_local_part=0)
    s = types.SimpleNamespace(tp_worker=types.SimpleNamespace(model_runner=runner))
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_SHARE", raising=False)
    assert vv.resolve_source(s).kind == v27.KIND_DENSE
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_SHARE", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert vv.resolve_source(s).kind == v27.KIND_PP_ONLY


def test_T3_row_split_takes_the_fewest_fitting_pieces_and_computes_the_same_numbers():
    split = v27.row_split_map([("m.fc.weight", (10, 4), 10 * 4 * 2), ("m.fc.bias", (10,), 20)], 40)
    assert split == {"m.fc.weight": (("m.fc.weight_parts.0", 5), ("m.fc.weight_parts.1", 5))}
    assert v27.row_split_map([("m.fc.weight", (10, 4), 80)], 39)["m.fc.weight"] == (
        ("m.fc.weight_parts.0", 4), ("m.fc.weight_parts.1", 4), ("m.fc.weight_parts.2", 2))
    with pytest.raises(vv.VisionVictimShort, match="cannot be row-split"):
        v27.row_split_map([("m.pos", (3, 4, 4), 96)], 40)
    torch.manual_seed(3)
    for dtype in (torch.float32, torch.bfloat16):
        lin = torch.nn.Linear(24, 10, dtype=dtype)
        x = torch.randn(7, 24, dtype=dtype)
        m = torch.nn.Module()
        m.fc = lin
        ref = lin(x)
        v27.apply_row_split(m, {"fc.weight": (("fc.weight_parts.0", 6), ("fc.weight_parts.1", 4))})
        with torch.no_grad():
            m.fc.weight_parts[0].copy_(lin.weight[:6])
            m.fc.weight_parts[1].copy_(lin.weight[6:])
        # each output element is its own dot product; only the GEMM's summation
        # order may differ with the piece's shape (CPU fp32: <= 1 ulp measured)
        torch.testing.assert_close(m.fc(x), ref)
    # the real tower's class (merger.linear_fc1: ColumnParallelLinear, tp1, unquantized):
    # the split keeps its (out, bias-or-None) return convention
    from flliper.srt.layers.linear import ColumnParallelLinear, RowParallelLinear

    col = ColumnParallelLinear(24, 10, bias=True, tp_size=1, tp_rank=0, params_dtype=torch.float32)
    with torch.no_grad():
        col.weight.normal_()
        col.bias.normal_()
    x = torch.randn(5, 24)
    ref, _ = col(x)
    m = torch.nn.Module()
    m.fc = col
    v27.apply_row_split(m, {"fc.weight": (("fc.weight_parts.0", 6), ("fc.weight_parts.1", 4))})
    with torch.no_grad():
        m.fc.weight_parts[0].copy_(col.weight[:6])
        m.fc.weight_parts[1].copy_(col.weight[6:])
    out, bias = m.fc(x)
    torch.testing.assert_close(out, ref)
    assert bias is None
    assert v27._splittable(RowParallelLinear(24, 10, bias=True, tp_size=1, tp_rank=0)) == ""
    assert "tp_size=2" in v27._splittable(ColumnParallelLinear(24, 10, bias=True, tp_size=2, tp_rank=0))
    m2 = torch.nn.Module()
    m2.conv = torch.nn.Conv1d(2, 2, 1)
    with pytest.raises(vv.VisionVictimShort, match="not a linear layer"):
        v27.apply_row_split(m2, {"conv.weight": (("conv.weight_parts.0", 1), ("conv.weight_parts.1", 1))})


class _SplitTower(_Tower):
    """The runner tests' tower, its dtype read like the real one's (from a
    tensor that is never split -- the real tower reads patch_embed.proj)."""

    @property
    def dtype(self):
        return torch.bfloat16


def _build_split():
    def build(hf_config, device):
        with vrs.params_on_meta():
            return _SplitTower(), None

    return build


class _SmallVictims(v27._Line27BVictims):
    """pp_only-like: many victims, each smaller than the tower's two weights
    (qkv 384 B, fc1 288 B > 280 B), so both are row-split whatever the CPU
    allocator's 64 B base alignment leaves usable."""

    kind = v27.KIND_PP_ONLY

    def __init__(self, n=24, size=280):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        self.tensors = [torch.randint(0, 256, (size,), generator=g, dtype=torch.uint8) for _ in range(n)]
        self.before = [t.clone() for t in self.tensors]

    def _named_storages(self):
        return [(f"v{i}", vv.storage_view(t)) for i, t in enumerate(self.tensors)]


def test_T3_a_stage_on_victims_smaller_than_a_tower_tensor_splits_and_encodes_the_checkpoint(tmp_path):
    ck = _write_model(tmp_path)
    victims = _SmallVictims()
    it = _Item()
    px = it.feature.clone()
    out = vrr.run_rank_stage(types.SimpleNamespace(token_to_kv_pool_allocator=_Alloc(_kv())),
                             [_req("r", [it])], model_dir=str(tmp_path), hf_config=None,
                             device=torch.device("cpu"), build=_build_split(), place=vrs.PLACE_WEIGHTS,
                             victims=victims)
    assert out.ok, out.detail
    w1, b1 = ck["model.visual.blocks.0.attn.qkv.weight"], ck["model.visual.blocks.0.attn.qkv.bias"]
    w2, b2 = ck["model.visual.merger.linear_fc1.weight"], ck["model.visual.merger.linear_fc1.bias"]
    assert torch.equal(it.precomputed_embeddings, F.linear(F.linear(px.to(torch.bfloat16), w1, b1), w2, b2))
    assert "split_tensors=2" in out.victim_fields and out.checksum == "ok"
    assert all(torch.equal(a, b) for a, b in zip(victims.tensors, victims.before))


class _Inventory(v27._Line27BVictims):
    """Header numbers only: the inventory as the arming sees it."""

    def __init__(self, kind, run, count):
        super().__init__()
        self.kind = kind
        self._inv = [vv.VictimCandidate(name=f"v{i}", key=(i + 1) << 32, offset=0, nbytes=run,
                                        storage_nbytes=run) for i in range(count)]

    def inventory(self):
        return list(self._inv)


def _merger_tower():
    """The 27B tower's two largest tensors (safetensors header) and a body of
    27 blocks x (qkv 3456x1152, proj 1152x1152, fc1 4304x1152, fc2 1152x4304)."""
    t = [vrs.CkptTensor("model.visual.merger.linear_fc1.weight", torch.bfloat16, (4608, 4608), 0, 4608 * 4608 * 2),
         vrs.CkptTensor("model.visual.merger.linear_fc2.weight", torch.bfloat16, (5120, 4608), 0, 5120 * 4608 * 2)]
    for b in range(27):
        for name, shape in (("attn.qkv", (3456, 1152)), ("attn.proj", (1152, 1152)),
                            ("mlp.linear_fc1", (4304, 1152)), ("mlp.linear_fc2", (1152, 4304))):
            t.append(vrs.CkptTensor(f"model.visual.blocks.{b}.{name}.weight", torch.bfloat16, shape, 0,
                                    shape[0] * shape[1] * 2))
    return t


@pytest.mark.parametrize("kind,run,count,splits", [
    (v27.KIND_PP_ONLY, 12 * MIB, 340, 2),    # dual: ~12 MiB pp_only tensors, 4060 MiB
    (v27.KIND_DENSE, 170 * MIB, 40, 0),      # flip INT8: fused gate_up per layer
])
def test_M0_arming_arithmetic_on_the_27b_merger(kind, run, count, splits):
    tower = _merger_tower()
    line, why = vv.arming_line(_Inventory(kind, run, count), tower, vrr._tower_name)
    assert not why, why
    assert f"victim={kind}" in line and f"split_tensors={splits}" in line
    total = sum(t.nbytes for t in tower) / MIB
    planned = float(line.split("planned_victim_mib=")[1].split()[0])
    assert total <= planned <= total + 1.0                     # moved bytes = the tower (+ 256 B grain)
