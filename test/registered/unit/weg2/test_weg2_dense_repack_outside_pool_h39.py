"""H39: the dense Marlin repack leaves no dead block in a weg2 tag pool.

BEFUND (fnFL2x145, 9310d2893a, D-TP0 = 5090, the attention host of Form A;
literal lines in ``fixtures/d_card_h39/fnFL2x145.D.lines``): of the 4993 MiB
``private_free`` that WEG2-GRAPH-POOL names on TP0, the tag pools already held
4.8 GiB INACTIVE right after load -- the base ``weights`` pool (pool 0.2,
2494/1244 MiB) 1.22 GiB next to 1.21 GiB of live embed + lm_head, and every
layer band 0.18-0.25 GiB, where the expert-only TP1/TP2 bands hold 0.03-0.04.
The difference is the DENSE layers: ``compressed_tensors_wNa16`` (6-bit GDN /
attention / shared-expert widened to 8 bit, the 8-bit HC mixer, PLE and
lm_head) allocated its checkpoint-format tensor at construction and the whole
repack working set (widened packing, ``contiguous()`` of the transposed view)
INSIDE the live private tag pool, which never hands a freed block back.  The
lm_head alone: 606.25 MiB original + 606.25 MiB contiguous = 1212.5 MiB, the
1.22 GiB of the base pool (P's last stage, which holds the head: 1.21 GiB;
P's first stage, embedding only, no repack: 0.01).

FIX: the checkpoint-format tensors are born in the load's transient pool
(``_checkpoint_format_scope``), the repack runs there (``outside_tag_pool``),
and only the survivors are born in the tag pool.  The tests below model the
pool routing on CPU and assert the property the metal needs: every block the
dense post-load path allocates inside the tag pool is still alive afterwards.
"""

import os
from contextlib import contextmanager

import pytest
import torch
from torch.overrides import TorchFunctionMode

from sglang.srt.environ import envs
from sglang.srt.managers import weg2_memory_saver as saver
from sglang.srt.planner import expert_residency as er
from sglang.srt.planner import graph_pool_ledger as gpl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "d_card_h39")
MIB = float(1 << 20)
GIB = float(1 << 30)


def _x145():
    with open(os.path.join(FIX, "fnFL2x145.D.lines")) as fh:
        return fh.read()


def _occupancy(text):
    """{rank: {tag: (active_gib, inactive_gib)}} of the first after-load line."""
    import re

    out = {}
    for m in re.finditer(
        r" TP(\d)\] WEG2-TAG-POOL occupancy tag=(\S+) when=after-load "
        r"segments=\d+ active_blocks=\d+ active_gib=([\d.]+) inactive_gib=([\d.]+)",
        text,
    ):
        out.setdefault(int(m.group(1)), {}).setdefault(
            m.group(2), (float(m.group(3)), float(m.group(4)))
        )
    return out


# ---------------------------------------------------------------------------
# 1. der Befund, woertlich aus dem Log
# ---------------------------------------------------------------------------


def test_tp0_private_free_is_the_tag_pools_dead_blocks_after_load():
    text = _x145()
    occ = _occupancy(text)
    s = gpl.binding_sample(gpl.samples_from_log(text)[0])
    assert s.source == gpl.SOURCE_INSTRUMENT
    assert s.private_free_mib == pytest.approx(4993, abs=1)
    tag_dead_mib = sum(i for _a, i in occ[0].values()) * 1024
    # >= 95 % of the private-free is tag-pool dead weight from the LOAD, the
    # graph captures (post-capture - after-load) are the small rest
    assert tag_dead_mib >= 0.95 * s.private_free_mib
    assert tag_dead_mib <= s.private_free_mib
    # pool 0.2 is the base 'weights' pool: its free MiB are the occupancy's
    # inactive GiB, and the live part is embed + lm_head (2 x 606.25 MiB)
    import re

    line = next(
        ln for ln in text.splitlines()
        if "WEG2-GRAPH-POOL rank=0 phase=post-capture" in ln
    )
    pools = {
        pid: (float(t), float(u))
        for pid, t, u in re.findall(r"([\d.]+):([\d.]+)/([\d.]+)", line.split("pools=")[1].split()[0])
    }
    tot, used = pools["0.2"]
    act, ina = occ[0]["weights"]
    assert (tot - used) == pytest.approx(ina * 1024, abs=12)
    lm_head_mib = 248320 * 2560 / MIB  # 8-bit, one byte per weight
    assert lm_head_mib == pytest.approx(606.25)
    # the lm_head repack: original + contiguous() of the transposed view
    assert 2 * lm_head_mib <= ina * 1024 <= 2 * lm_head_mib + 40


def test_the_bands_of_the_attention_host_carry_the_dense_repack():
    occ = _occupancy(_x145())
    for r in (1, 2):  # expert-only workers
        bands = [i for t, (_a, i) in occ[r].items() if t.startswith("weights_") and t[8:].isdigit()]
        assert len(bands) == 16 and max(bands) <= 0.08, (r, bands)
        assert sorted(bands)[8] == pytest.approx(0.03)  # the median band
        assert occ[r]["weights"] == (0.0, 0.0)
    bands0 = [i for t, (_a, i) in occ[0].items() if t.startswith("weights_") and t[8:].isdigit()]
    assert len(bands0) == 16 and min(bands0) >= 0.15
    # what the fix can reach at most: TP0's bands down to the workers' floor
    # plus the whole base pool (a Rechnung, the next boot measures it)
    floor = 0.03  # the workers' median band: the expert-side residue stays
    assert sum(bands0) == pytest.approx(3.40, abs=0.005)
    reachable_gib = sum(b - floor for b in bands0) + occ[0]["weights"][1]
    assert reachable_gib == pytest.approx(4.14, abs=0.01)


def test_the_planner_reads_the_instrument_lines_of_a_new_boot():
    """--d-card-reference-logs takes a boot that carries WEG2-GRAPH-POOL (not
    only the legacy [vram-peak] + #1027): after the fix the next boot's TP0
    private-free is read from exactly these lines."""
    ref = er.d_card_reference_from_logs(
        [("fnFL2x145", _x145())],
        n_ranks=3,
        n_layers=48,
        slot_bytes=1297637376 / 512,
        model="Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist",
        rank_tp_ratio="1,0,0",
    )
    assert ref.private_free_mib[0] == pytest.approx(4993, abs=1)
    assert ref.private_free_mib[1] < 1000 and ref.private_free_mib[2] < 1000
    assert ref.buffer_rows[0] == 94


# ---------------------------------------------------------------------------
# 2. der Mechanismus, auf der CPU: Pool-Routing modelliert, jede Allokation
#    mit dem Pool beschriftet, in dem sie geboren wurde
# ---------------------------------------------------------------------------


class _FakePools:
    """The routing of weg2_memory_saver: 'tag' while a tag pool is active,
    'outside' in a stepped-out block, 'tag' again under back_into_tag_pool.
    Same yields as the real ones (outside: False without a tag pool, True when
    nested; back_into: False outside a stepped-out block)."""

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


K, N, G = 256, 128, 64


def _fake_repack(b_q_weight, perm, size_k, size_n, num_bits):
    out = torch.empty(
        (size_k // 16, size_n * 16 // (32 // num_bits)), dtype=b_q_weight.dtype
    )
    out.copy_(b_q_weight.reshape(out.shape))
    return out


def _run(monkeypatch, *, bits, actorder=None, symmetric=True, armed=True, tag_pool_open=True):
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
        group_size=G,
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
        # the checkpoint load: in-place copies into the params; the random
        # sources are the test's own tensors, labelled apart
        pools.stack.append("test-source")
        with torch.no_grad():
            for n, p in layer.named_parameters():
                if p.dtype == torch.int32:
                    p.copy_(torch.randint(-(1 << 30), 1 << 30, p.shape, generator=g, dtype=torch.int32))
                elif p.dtype == torch.int64:
                    p.copy_(torch.tensor([N, K]))
                else:
                    p.copy_(torch.rand(p.shape, generator=g).to(p.dtype))
            if actorder:
                layer.weight_g_idx.copy_(
                    (torch.randperm(K, generator=g) // G).to(torch.int32)
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
        ptr: nb
        for ptr, (where, nb) in rec.birth.items()
        if where == "tag" and ptr not in live
    }
    return layer, scheme, rec, originals, live, dead_in_tag, pools


VARIANTS = [
    dict(bits=6),  # GDN / attention / shared expert (widened 6 -> 8)
    dict(bits=8),  # HC mixer, PLE, lm_head
    dict(bits=8, actorder=True),  # g_idx survivors
    dict(bits=8, symmetric=False),  # zero-point survivor
    dict(bits=4),
]


@pytest.mark.parametrize("v", VARIANTS, ids=lambda v: "-".join(f"{k}{x}" for k, x in v.items()))
def test_armed_no_dead_block_in_the_tag_pool(monkeypatch, v):
    layer, scheme, rec, originals, live, dead, pools = _run(monkeypatch, **v)
    assert dead == {}, "blocks born in the tag pool and dead after the repack: %s" % dead
    # every survivor was born in the tag pool (pausable with its tag)
    for ptr in live:
        assert rec.birth[ptr][0] == "tag", ptr
    # the checkpoint-format tensors were born outside, weight_shape inside
    for name, (ptr, nb) in originals.items():
        want = "tag" if name == "weight_shape" else "outside"
        assert rec.birth[ptr][0] == want, name
    assert "ct-dense-marlin" in pools.outside_calls
    assert "ct-dense-ckpt" in pools.outside_calls
    # the originals go to their OWN load pool, the working set to the shared one
    assert pools.into == {"ct-dense-ckpt": "ckpt", "ct-dense-marlin": "load"}


@pytest.mark.parametrize("bits,ratio", [(6, 2.75), (8, 2.0)])
def test_unarmed_is_the_measured_leak(monkeypatch, bits, ratio):
    """The 2026-09-24 form, reproduced: dead tag-pool bytes = original +
    contiguous (+ widened for 6 bit) = 2.75x / 2.0x the repacked weight --
    the ratio the x144 bands and the 1.22 GiB lm_head pool show."""
    layer, _s, _rec, _o, _live, dead, _p = _run(monkeypatch, bits=bits, armed=False)
    wq = layer.weight_packed.untyped_storage().nbytes()
    ws = layer.weight_scale.untyped_storage().nbytes()
    # scales leak too (original + contiguous + index copy); weight dominates
    assert sum(dead.values()) >= ratio * wq
    assert sum(dead.values()) <= ratio * wq + 4 * ws + 4096


@pytest.mark.parametrize("v", VARIANTS, ids=lambda v: "-".join(f"{k}{x}" for k, x in v.items()))
def test_armed_and_unarmed_give_the_same_bytes(monkeypatch, v):
    a = _run(monkeypatch, armed=True, **v)[0]
    b = _run(monkeypatch, armed=False, **v)[0]
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys()
    assert list(pa) == list(pb)  # registration order unchanged
    for n in pa:
        assert pa[n].dtype == pb[n].dtype and pa[n].shape == pb[n].shape, n
        assert torch.equal(pa[n].view(torch.uint8) if pa[n].numel() else pa[n],
                           pb[n].view(torch.uint8) if pb[n].numel() else pb[n]), n
    assert torch.equal(a.g_idx_sort_indices, b.g_idx_sort_indices)


def test_without_a_tag_pool_nothing_steps_anywhere(monkeypatch):
    layer, _s, rec, _o, live, dead, pools = _run(monkeypatch, bits=6, tag_pool_open=False)
    assert {w for w, _nb in rec.birth.values()} - {"test-source"} == {"default"}
    assert dead == {}  # no tag pool, no tag-pool block at all
    ref = _run(monkeypatch, bits=6, armed=False, tag_pool_open=False)[0]
    assert torch.equal(layer.weight_packed, ref.weight_packed)


def test_the_switch_defaults_on_and_off_skips_every_step(monkeypatch):
    assert envs.SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL.get() is True
    *_, pools = _run(monkeypatch, bits=8, armed=False)
    assert "ct-dense-marlin" not in pools.outside_calls
    assert "ct-dense-ckpt" not in pools.outside_calls


# ---------------------------------------------------------------------------
# 3. die zwei Lade-Pools werden unabhaengig zurueckgegeben
# ---------------------------------------------------------------------------


def test_a_live_original_keeps_only_its_own_pool(monkeypatch, caplog):
    """A checkpoint-format tensor something still references must not pin the
    repack working set (2.2 GiB on D-TP0, 2.6-3.0 GiB per P stage): the ckpt
    pool is KEPT alone, the transient pool is released, and its log line is
    the one the boot readers parse, byte for byte."""
    import logging

    ckpt, load = object(), object()
    monkeypatch.setattr(saver, "_LOAD_CKPT_POOL", ckpt)
    monkeypatch.setattr(saver, "_LOAD_TRANSIENT_POOL", load)
    monkeypatch.setattr(saver, "_ACTIVE_TAG_POOL", None)
    monkeypatch.setattr(saver, "_STEPPED_OUT_POOLS", [])
    monkeypatch.setattr(saver, "_tms_cdll_in_region", lambda: None)
    kept = []
    monkeypatch.setattr(saver, "_KEPT_TRANSIENT_POOLS", kept)
    monkeypatch.setattr(saver, "_pool_has_live_blocks", lambda pool, reason: pool is ckpt)
    reserved = iter([10 * GIB, 8 * GIB])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: next(reserved))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    with caplog.at_level(logging.INFO, logger=saver.logger.name):
        freed = saver.release_load_transient_pool(reason="after-load")
    assert kept == [ckpt]
    assert saver._LOAD_CKPT_POOL is None and saver._LOAD_TRANSIENT_POOL is None
    assert freed == pytest.approx(2.0)
    line = [r.getMessage() for r in caplog.records if "RELEASED" in r.getMessage()]
    assert line == [
        "WEG2-TAG-POOL load transient pool RELEASED reason=after-load freed_gib=2.00 "
        "-- the repack's working set, reused across layers and handed back in one piece"
    ]
    # the fixture reader of section 1 parses exactly this shape
    import re

    assert re.search(r"load transient pool RELEASED.*freed_gib=([\d.]+)", line[0])
