"""P-COLD: the QSA indexer's TileLang prefill kernels are built at boot.

Hermetic (no CUDA, no TileLang needed). Metal (rc12z30c, P log
...09282039_f33274eac0_0928_203953.P.log): the first P forward of the boot
(pdflip-0-6, 30 new tokens) stalled PP1 '#1466 PASS-STALL pp_rank=1 ...
fwd_ms=12783'; its Triton windows all closed after 0.0 s, the gap
'_fused_qk_rmsnorm_rope_gate_kernel' 20:45:49 -> '_expand_qsa_block_indices_kernel'
20:45:58 is 'TileLang begins/completes to compile kernel' 20:45:49-55 and
20:45:55-58 (Z. 28557-28560): the MQA prefill kernel and the mask kernel.

What these cases pin:

* the prewarm reaches both kernels through the serving entry with the serving
  key (heads, head_dim, block_q = 128 // heads), bf16 q/k, int32 bounds;
* the geometry comes from the rank's own QSAIndexer modules (deduplicated);
* named skips (no TileLang / Form-A worker / no indexer), never raises;
* the scheduler runs it before the #603b sampling barrier (so before the first
  sleep and READY).
"""

import ast
import logging
import pathlib
from unittest import mock

import torch

from flliper.srt import rank_role
from flliper.srt.layers.attention.qsa import mqa
from flliper.srt.layers.attention.qsa import mqa_prewarm as pw
from flliper.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

REPO = pathlib.Path(__file__).resolve().parents[4]


def _indexer(heads, head_dim):
    m = object.__new__(QSAIndexer)
    m.__dict__.update(index_n_heads=heads, index_head_dim=head_dim)
    return m


class _Model:
    def __init__(self, *mods):
        self._mods = mods

    def modules(self):
        return iter(self._mods)


def test_metal_form_both_kernels_are_built_with_the_serving_key(monkeypatch):
    built = []

    def prefill_kernel(**key):
        built.append(("prefill", key))
        return lambda q, k, logits, starts, ends: None

    def mask_kernel(**key):
        built.append(("mask", key))
        return lambda logits, starts, ends: None

    monkeypatch.setattr(mqa, "HAS_TILELANG", True)
    monkeypatch.setattr(mqa, "_tilelang_qsa_mqa_prefill_kernel", prefill_kernel, raising=False)
    monkeypatch.setattr(mqa, "_tilelang_qsa_mqa_mask_kernel", mask_kernel, raising=False)
    res = pw.prewarm_mqa(signatures=((16, 128),), device="cpu")
    assert built == [("prefill", {"heads": 16, "head_dim": 128, "block_q": 8}), ("mask", {})]
    assert res.launched == 1


def test_the_launch_carries_the_serving_shapes():
    seen = []
    pw.prewarm_mqa(signatures=((16, 128), (4, 64)), device="cpu",
                   launch=lambda q, k, s, e: seen.append((q.shape, q.dtype, k.shape, k.dtype,
                                                          s.dtype, e.tolist())))
    assert seen[0][:5] == ((8, 16, 128), torch.bfloat16, (64, 1, 128), torch.bfloat16, torch.int32)
    assert seen[0][5] == [64] * 8  # every row attends every key: nothing masked away
    assert seen[1][0] == (32, 4, 64)  # block_q = 128 // heads


def test_the_geometry_is_the_ranks_own_indexers():
    model = _Model(_indexer(16, 128), torch.nn.Linear(2, 2), _indexer(16, 128), _indexer(8, 64))
    assert pw.indexer_prefill_signatures(model.modules()) == ((16, 128), (8, 64))
    assert pw.indexer_prefill_signatures(_Model(torch.nn.Linear(2, 2)).modules()) == ()


def test_named_skips(monkeypatch, caplog):
    launch = mock.Mock()
    monkeypatch.setattr(pw, "prewarm_mqa", launch)
    monkeypatch.setattr(mqa, "HAS_TILELANG", False)
    with caplog.at_level(logging.INFO):
        assert "no TileLang" in pw.run_boot_prewarm(model=_Model(_indexer(16, 128)), device="cpu").skipped
    monkeypatch.setattr(mqa, "HAS_TILELANG", True)
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: True)
    assert "Form-A" in pw.run_boot_prewarm(model=_Model(_indexer(16, 128)), device="cpu").skipped
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: False)
    assert "no QSAIndexer" in pw.run_boot_prewarm(model=_Model(), device="cpu").skipped
    assert "no QSAIndexer" in pw.run_boot_prewarm(model=None, device="cpu").skipped
    launch.assert_not_called()
    assert "P-COLD QSA-MQA-TILELANG-PREWARM skipped" in caplog.text


def test_a_rank_with_indexers_prewarms_and_logs(monkeypatch, caplog):
    monkeypatch.setattr(mqa, "HAS_TILELANG", True)
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: False)
    got = []
    monkeypatch.setattr(pw, "prewarm_mqa", lambda **kw: got.append(kw) or pw.MqaPrewarmResult(
        kw["signatures"], 1, 9000.0))
    with caplog.at_level(logging.INFO):
        res = pw.run_boot_prewarm(model=_Model(_indexer(16, 128)), device="cpu")
    assert got[0]["signatures"] == ((16, 128),) and res.launched == 1
    assert "QSA-MQA-TILELANG-PREWARM kernels=[mqa_prefill, mqa_mask] sigs=[heads=16 head_dim=128]" in caplog.text


def test_the_prewarm_never_raises(monkeypatch):
    monkeypatch.setattr(mqa, "HAS_TILELANG", True)
    monkeypatch.setattr(rank_role, "this_rank_is_form_a_worker", lambda: False)
    monkeypatch.setattr(pw, "prewarm_mqa", mock.Mock(side_effect=RuntimeError("nvcc")))
    assert pw.run_boot_prewarm(model=_Model(_indexer(16, 128)), device="cpu") is None


def test_the_scheduler_prewarms_before_the_sampling_barrier():
    src = (REPO / "python/flliper/srt/managers/scheduler.py").read_text()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    meths = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    delegate = ast.unparse(meths["warm_qsa_mqa_tilelang"])
    assert "mqa_prewarm import run_boot_prewarm" in delegate and "run_boot_prewarm(" in delegate
    caller = [m for m in meths.values()
              if "self.warm_sampling_backend()" in ast.unparse(m) and m.name != "warm_sampling_backend"]
    assert len(caller) == 1
    body = ast.unparse(caller[0])
    assert body.index("self.warm_qsa_mqa_tilelang()") < body.index("self.warm_sampling_backend()")
