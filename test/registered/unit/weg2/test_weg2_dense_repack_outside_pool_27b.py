"""H39 on the 27B line: the dense Marlin repack leaves no dead block in a weg2
tag pool -- behind SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL.

UNIFY S2 (desk/27b-unified-0926): the switch has the SAME name on both lines; the
27B line defaulted it OFF (byte-identical until booted), the NF line ON.  ONE
environ.py entry, its DEFAULT per profile (weg2/form.py PROFILE_SWITCH_DEFAULTS):
qwen27b OFF, nextflash ON, no published form ON (the NF default); an explicitly
set value always wins (the 27B arms and docker/profiles/27b.env set 1).

Code ported from the NF line (b48c7f07e5 tag-pool tracking, 11cadf67a3
outside/back_into, 4da5c56cc2 load pool + release after the region,
196d43b514 transients untracked by the saver, d6b7d4a1d3 H39 dense scheme +
checkpoint-format pool).  NO NF measurement is used here: section 1 reads the
27B's own boot (literal lines in ``fixtures/d_card_h39_27b/``).

BEFUND 27B (weg2xsn424 f888d6ad59; weg2xsn423 e7745fbf97 byte-identical):
the target Qwen3.8-27B-INT8-gdncov-vocabembed is W8A8-INT8 (channel weights,
dynamic token activations) -- ``CompressedTensorsW8A8Int8`` only transposes a
view, no repack, and reserved minus allocated after its load is 0.04/0.20/0.20
GiB on D and 0.02/0.00/0.08 on P.  The DFlash2-W8 drafter is pack-quantized
8 bit group 128, i.e. ``compressed_tensors_wNa16`` + Marlin repack, loaded in
the private ``weights_draft`` pool: its tms segments exceed its tensors by
431/209/199 MiB on D TP0/TP1/TP2 and 771 MiB on P PP2.  Those bytes are inside
'weights + runtime state' (used_by_me = pre-load free - free), so on D, where
the --rank-gpu-memory-mib budget binds, they come 1:1 out of the KV pool.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import types
from contextlib import contextmanager

import pytest
import torch
from torch.overrides import TorchFunctionMode

from sglang.srt.environ import envs
from sglang.srt.managers import weg2_memory_saver as saver

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "d_card_h39_27b")
MIB = float(1 << 20)
GIB = float(1 << 30)
CELL = 32768  # the 27B D cell (fp8 KV, 16 attention layers), from the log


def _lines(group):
    with open(os.path.join(FIX, f"weg2xsn424.{group}.lines")) as fh:
        return fh.read()


_CENSUS = re.compile(
    r" (?:TP|PP)(\d)\] \[vram-census\] pp(\d)tp(\d)(-draft)? after load: model tensors on "
    r"device ([\d.]+) GiB .*?torch allocated ([\d.]+) GiB, reserved ([\d.]+) GiB"
)


def _census(text):
    """{(log_rank, kind): (tensors, allocated, reserved)} in GiB."""
    out = {}
    for m in _CENSUS.finditer(text):
        kind = "draft" if m.group(4) else "target"
        out[(int(m.group(1)), kind)] = tuple(float(m.group(i)) for i in (5, 6, 7))
    return out


def _tms_resident(text):
    """{log_rank: {tag: MiB}} of the first DC-BREAKDOWN release per rank."""
    out = {}
    for m in re.finditer(
        r" (?:TP|PP)(\d)\] WEG2-DC-BREAKDOWN stage=release .*?tms_resident \d+ (\{[^}]*\})",
        text,
    ):
        tags = {k: int(v) for k, v in re.findall(r"'([^']+)': (\d+)", m.group(2))}
        out.setdefault(int(m.group(1)), tags)
    return out


# ---------------------------------------------------------------------------
# 1. der 27B-Befund, woertlich aus weg2xsn424
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group", ["D", "P"])
def test_the_w8a8_target_leaves_no_repack_residue(group):
    c = _census(_lines(group))
    targets = {r: v for (r, k), v in c.items() if k == "target"}
    assert len(targets) == 3
    for r, (_t, alloc, res) in targets.items():
        assert res - alloc <= 0.20 + 1e-9, (group, r, res - alloc)


def test_the_drafter_pool_holds_the_repack_residue_on_d():
    text = _lines("D")
    c, tms = _census(text), _tms_resident(text)
    dead = {r: tms[r]["weights_draft"] - c[(r, "draft")][0] * 1024 for r in range(3)}
    # census GiB carry 2 decimals: +-5.12 MiB
    assert dead[0] == pytest.approx(431, abs=6)
    assert dead[1] == pytest.approx(209, abs=6)
    assert dead[2] == pytest.approx(199, abs=6)
    # second instrument: the reserved-minus-allocated GROWTH over the draft load
    for r, want in ((0, 0.47), (1, 0.26), (2, 0.25)):
        _t0, a0, r0 = c[(r, "target")]
        _t1, a1, r1 = c[(r, "draft")]
        assert (r1 - a1) - (r0 - a0) == pytest.approx(want, abs=0.011)
        # the two instruments agree within 60 MiB (default-pool cache, slack)
        assert abs(((r1 - a1) - (r0 - a0)) * 1024 - dead[r]) <= 60


def test_the_drafter_pool_holds_the_repack_residue_on_p_pp2():
    text = _lines("P")
    c, tms = _census(text), _tms_resident(text)
    # the drafter is loaded on the last stage only (its census line says pp0tp0-draft)
    (r,) = [r for (r, k) in c if k == "draft"]
    assert r == 2
    assert tms[2]["weights_draft"] - c[(2, "draft")][0] * 1024 == pytest.approx(771, abs=6)
    assert "weights_draft" not in tms[0] and "weights_draft" not in tms[1]


def test_d_kv_is_budget_bound_so_a_freed_mib_is_a_kv_mib():
    """rest = budget - sum(posts) on every D rank, 'weights + runtime state'
    is the first post, and the card had MORE free than rest (unaccounted > 0):
    the budget binds, a MiB less in the first post is a MiB more KV."""
    text = _lines("D")
    argv = next(ln for ln in text.splitlines() if ln.startswith("argv:"))
    budgets = [int(x) for x in re.search(r"--rank-gpu-memory-mib (\S+)", argv).group(1).split(",")]
    assert budgets == [27720, 17232, 17024]
    for m in re.finditer(
        r"\[world_rank (\d)\] KV budget posts \(GiB\): (.*?) \| rest=([\d.]+) \| "
        r"measured free=([\d.]+) \| unaccounted=([+-][\d.]+)",
        text,
    ):
        r = int(m.group(1))
        posts = [float(v) for v in re.findall(r"=([\d.]+)", m.group(2))]
        assert m.group(2).startswith("weights + runtime state=")
        assert budgets[r] / 1024 - sum(posts) == pytest.approx(float(m.group(3)), abs=0.002)
        assert float(m.group(5)) > 0


def test_rechnung_d_kv_after_the_drafter_residue_is_freed():
    """RECHNUNG, not a measurement: the freed MiB per rank (tms figure) become
    KV tokens at 32 KiB, the line's own uneven-DCP rule re-derives the vector
    (partition_units(64, capacities)) and the pool is min(cap // v) * 64."""
    from sglang.srt.distributed.utils import partition_units

    text = _lines("D")
    caps = [0, 0, 0]
    for m in re.finditer(
        r" TP(\d)\] KV pool sizing: available_bytes=(\d+) .*?cell_size=(\d+).*?"
        r"max_total_num_tokens=(\d+)",
        text,
    ):
        assert int(m.group(3)) == CELL
        caps[int(m.group(1))] = int(m.group(4))
    assert caps == [247875, 200274, 194258]

    def pool(c):
        v = list(partition_units(64, c))
        return v, min(c[r] // v[r] for r in range(3)) * sum(v)

    v0, before = pool(caps)
    assert (v0, before) == ([25, 20, 19], 634560)  # the boot's own line
    assert "EFFECTIVE max_total_num_tokens 634560" in text
    freed_mib = [431, 209, 199]
    after_caps = [caps[r] + freed_mib[r] * (1 << 20) // CELL for r in range(3)]
    v1, after = pool(after_caps)
    assert v1 == [25, 20, 19]
    assert after == 662272  # +27712 tokens (+4.4 %); TP1 binds instead of TP0
    assert min(range(3), key=lambda r: after_caps[r] // v1[r]) == 1


# ---------------------------------------------------------------------------
# 2. der Mechanismus auf der CPU: Pool-Routing modelliert, jede Allokation mit
#    dem Pool beschriftet, in dem sie geboren wurde
# ---------------------------------------------------------------------------


class _FakePools:
    """'tag' while a tag pool is active, 'outside' in a stepped-out block,
    'tag' again under back_into_tag_pool -- the same yields as the real ones."""

    def __init__(self, tag_pool_open=True):
        self.stack = ["tag" if tag_pool_open else "default"]
        self.outside_calls = []
        self.into = {}

    @contextmanager
    def outside_tag_pool(self, reason="", *, into="load"):
        self.outside_calls.append(reason)
        self.into[reason] = into
        if self.stack[-1] == "default":
            yield False
            return
        if self.stack[-1] == "outside":
            yield True
            return
        self.stack.append("outside")
        try:
            yield True
        finally:
            self.stack.pop()

    @contextmanager
    def back_into_tag_pool(self):
        if "outside" not in self.stack:
            yield False
            return
        self.stack.append("tag")
        try:
            yield True
        finally:
            self.stack.pop()


class _BirthRecorder(TorchFunctionMode):
    """First sighting of every storage = its birth pool.  Holds every tensor
    it saw, so no address is reused and a dead block stays identifiable."""

    def __init__(self, pools):
        super().__init__()
        self.pools = pools
        self.birth = {}
        self.keep = []

    def _see(self, t):
        if isinstance(t, torch.Tensor):
            st = t.untyped_storage()
            if st.nbytes() and st.data_ptr() not in self.birth:
                self.birth[st.data_ptr()] = (self.pools.stack[-1], st.nbytes())
            self.keep.append(t)
        elif isinstance(t, (tuple, list)):
            for x in t:
                self._see(x)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        self._see(out)
        return out


K, N = 256, 128


def _fake_repack(b_q_weight, perm, size_k, size_n, num_bits):
    out = torch.empty(
        (size_k // 16, size_n * 16 // (32 // num_bits)), dtype=b_q_weight.dtype
    )
    out.copy_(b_q_weight.reshape(out.shape))
    return out


def _run(monkeypatch, *, bits, group=64, actorder=None, symmetric=True, armed=True,
         tag_pool_open=True):
    from compressed_tensors.quantization import ActivationOrdering

    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_wNa16 as mod,
    )

    pools = _FakePools(tag_pool_open)
    monkeypatch.setattr(saver, "outside_tag_pool", pools.outside_tag_pool)
    monkeypatch.setattr(saver, "back_into_tag_pool", pools.back_into_tag_pool)
    monkeypatch.setattr(mod, "gptq_marlin_repack", _fake_repack, raising=False)
    monkeypatch.setattr(
        mod, "marlin_make_workspace", lambda device, *a: torch.zeros(8, dtype=torch.int)
    )
    scheme = mod.CompressedTensorsWNA16(
        strategy="group",
        num_bits=bits,
        group_size=group,
        symmetric=symmetric,
        actorder=ActivationOrdering.GROUP if actorder else None,
    )
    layer = torch.nn.Module()
    g = torch.Generator().manual_seed(bits * 10 + bool(actorder) + 2 * (not symmetric))
    rec = _BirthRecorder(pools)
    with envs.SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL.override(armed), rec:
        scheme.create_weights(
            layer,
            output_size=N,
            input_size=K,
            output_partition_sizes=[N],
            input_size_per_partition=K,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **k: None,
        )
        originals = {
            n: (p.untyped_storage().data_ptr(), p.untyped_storage().nbytes())
            for n, p in layer.named_parameters()
        }
        pools.stack.append("test-source")
        with torch.no_grad():
            for n, p in layer.named_parameters():
                if p.dtype == torch.int32:
                    p.copy_(torch.randint(-(1 << 30), 1 << 30, p.shape, generator=g,
                                          dtype=torch.int32))
                elif p.dtype == torch.int64:
                    p.copy_(torch.tensor([N, K]))
                else:
                    p.copy_(torch.rand(p.shape, generator=g).to(p.dtype))
            if actorder:
                layer.weight_g_idx.copy_(
                    (torch.randperm(K, generator=g) // group).to(torch.int32)
                )
        pools.stack.pop()
        scheme.process_weights_after_loading(layer)
    live = {
        t.untyped_storage().data_ptr()
        for t in [p for _n, p in layer.named_parameters()]
        + [layer.g_idx_sort_indices, scheme.workspace]
        if t.untyped_storage().nbytes()
    }
    dead_in_tag = {
        ptr: nb for ptr, (where, nb) in rec.birth.items() if where == "tag" and ptr not in live
    }
    return layer, scheme, rec, originals, live, dead_in_tag, pools


VARIANTS = [
    dict(bits=8, group=128),  # the DFlash2-W8 drafter's form
    dict(bits=8, actorder=True),  # g_idx survivors
    dict(bits=8, symmetric=False),  # zero-point survivor
    dict(bits=4),
    dict(bits=6),  # the line's widened sub-byte path
]


def _vid(v):
    return "-".join(f"{k}{x}" for k, x in v.items())


@pytest.mark.parametrize("v", VARIANTS, ids=_vid)
def test_armed_no_dead_block_in_the_tag_pool(monkeypatch, v):
    layer, scheme, rec, originals, live, dead, pools = _run(monkeypatch, **v)
    assert dead == {}, "born in the tag pool and dead after the repack: %s" % dead
    for ptr in live:  # every survivor pauses and resumes with its tag
        assert rec.birth[ptr][0] == "tag", ptr
    for name, (ptr, _nb) in originals.items():
        want = "tag" if name == "weight_shape" else "outside"
        assert rec.birth[ptr][0] == want, name
    assert pools.into == {"ct-dense-ckpt": "ckpt", "ct-dense-marlin": "load"}


def test_unarmed_is_the_measured_form_of_this_line(monkeypatch):
    """Switch off = the 2026-09-24 form: the 8-bit checkpoint tensor and the
    contiguous() of its transposed view die in the tag pool, 2.0x the repacked
    weight (plus the scales' copies) -- the residue section 1 measures."""
    layer, _s, _rec, _o, _live, dead, pools = _run(monkeypatch, bits=8, group=128, armed=False)
    wq = layer.weight_packed.untyped_storage().nbytes()
    ws = layer.weight_scale.untyped_storage().nbytes()
    assert 2.0 * wq <= sum(dead.values()) <= 2.0 * wq + 4 * ws + 4096
    assert pools.outside_calls == []


@pytest.mark.parametrize("v", VARIANTS, ids=_vid)
def test_armed_and_unarmed_give_the_same_bytes(monkeypatch, v):
    a = _run(monkeypatch, armed=True, **v)[0]
    b = _run(monkeypatch, armed=False, **v)[0]
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert list(pa) == list(pb)  # registration order unchanged
    for n in pa:
        assert pa[n].dtype == pb[n].dtype and pa[n].shape == pb[n].shape, n
        assert torch.equal(pa[n], pb[n]), n
    assert torch.equal(a.g_idx_sort_indices, b.g_idx_sort_indices)


def test_without_a_tag_pool_nothing_steps_anywhere(monkeypatch):
    layer, _s, rec, _o, _live, dead, _p = _run(monkeypatch, bits=8, tag_pool_open=False)
    assert {w for w, _nb in rec.birth.values()} - {"test-source"} == {"default"}
    assert dead == {}
    ref = _run(monkeypatch, bits=8, armed=False, tag_pool_open=False)[0]
    assert torch.equal(layer.weight_packed, ref.weight_packed)


def _form_env(profile):
    from sglang.srt.weg2.form import Weg2Form

    arch, experts, draft, kv = (("dense", "none", "dflash", "paged_dcp") if profile == "qwen27b"
                                else ("moe", "offload", "mtp", "qsa_forma"))
    return Weg2Form(arch=arch, experts=experts, draft=draft, p_draft="none", kv=kv,
                    flip="family", vision="off", profile=profile, model="m").env_value()


@pytest.mark.parametrize("profile,want", [("qwen27b", False), ("nextflash", True), (None, True)])
def test_the_default_is_per_profile(monkeypatch, profile, want):
    """UNIFY S2: 27B OFF, NF ON, no form ON -- one environ entry, no second one."""
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_wNa16 as mod,
    )

    monkeypatch.delenv("SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL", raising=False)
    if profile is None:
        monkeypatch.delenv("SGLANG_WEG2_FORM", raising=False)
    else:
        monkeypatch.setenv("SGLANG_WEG2_FORM", _form_env(profile))
    assert envs.SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL.get() is want
    assert mod.dense_repack_outside_pool_armed() is want


@pytest.mark.parametrize("profile", ["qwen27b", "nextflash", None])
@pytest.mark.parametrize("explicit,want", [("1", True), ("0", False)])
def test_an_explicit_value_wins_over_the_profile(monkeypatch, profile, explicit, want):
    if profile is None:
        monkeypatch.delenv("SGLANG_WEG2_FORM", raising=False)
    else:
        monkeypatch.setenv("SGLANG_WEG2_FORM", _form_env(profile))
    monkeypatch.setenv("SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL", explicit)
    assert envs.SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL.get() is want


@pytest.mark.parametrize("profile,explicit,want", [
    ("qwen27b", None, False), ("nextflash", None, True), (None, None, True),
    ("qwen27b", "1", True), ("nextflash", "0", False)])
def test_the_planner_reads_the_group_env_like_the_rank(profile, explicit, want):
    """H50's reader of the D-group env resolves the same default as the rank."""
    from sglang.srt.planner.expert_residency import dense_repack_outside_pool

    env = {}
    if profile is not None:
        env["SGLANG_WEG2_FORM"] = _form_env(profile)
    if explicit is not None:
        env["SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL"] = explicit
    assert dense_repack_outside_pool(env) is want


def test_environ_holds_exactly_one_entry():
    import ast
    import inspect

    from sglang.srt import environ as env_mod

    tree = ast.parse(inspect.getsource(env_mod))
    hits = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL"
                    for t in n.targets)]
    assert len(hits) == 1, hits
