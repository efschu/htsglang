# SPDX-License-Identifier: Apache-2.0
"""DP-NACHLAUF 02.10.: the small-q fusion rule of the uneven-DCP extend.

N6d (ec4d492f58): D's first P>D extend (5 new tokens over ~98k cached) spent
its extra ~80-140 ms in the 15 full-attention layers; the GPU check
(test_fi_small_q_prefix_split_gpu_1002) put flashinfer's prefix kernel at
~0.1 ms per layer with the stock plan already splitting, 0 ulp against a
forced split -- the cost is the DCP chain around it. For rows <= 16 of a real
extend (not verify, not inside a graph capture) the two existing fusions are
used whatever their global switches say. Pinned here (red before): the rule
itself; with the global switches OFF the forced merge is BIT-identical to the
two-collective a2a body and issues ONE collective instead of two (threaded
3-rank rendezvous that exchanges real inputs and would deadlock on a
mismatched sequence); KVQ fusability honours the force; the switch off and
wide/verify/captured forwards keep today's sequence; wiring.
"""

import importlib.util
import inspect
import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.dcp import comm  # noqa: E402

_H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dcp_collective_fusion_0926.py")
_spec = importlib.util.spec_from_file_location("_dcp_fusion_0926", _H)
F = importlib.util.module_from_spec(_spec)
sys.modules["_dcp_fusion_0926"] = F
_spec.loader.exec_module(F)


def _reset_all():
    F._reset()
    comm._SMALL_Q["on"] = None
    comm._SMALL_Q["rows"] = None


def test_rule(monkeypatch):
    for k in (comm.SMALL_Q_FUSE_ENV, comm.SMALL_Q_FUSE_ROWS_ENV):
        monkeypatch.delenv(k, raising=False)
    _reset_all()
    assert comm.small_q_fuse_applies(5, False, False)
    assert comm.small_q_fuse_applies(16, False, False)
    assert not comm.small_q_fuse_applies(17, False, False)       # wide
    assert not comm.small_q_fuse_applies(5, True, False)         # verify (force_prefix)
    assert not comm.small_q_fuse_applies(5, False, True)         # inside a graph capture
    monkeypatch.setenv(comm.SMALL_Q_FUSE_ENV, "0")
    _reset_all()
    assert not comm.small_q_fuse_applies(5, False, False)
    _reset_all()


def test_forced_merge_is_bit_identical_and_one_collective():
    _reset_all()
    for T, dtype, seed in ((5, torch.float32, 0), (1, torch.float32, 3), (16, torch.bfloat16, 5)):
        w = F._MergeWorld([12, 6, 6], T=T, seed=seed, dtype=dtype)

        def base_call(r, grp):
            return comm.cp_lse_ag_out_a2a_mha_uneven(w.o[r], w.lse[r], grp, w.counts, return_lse=True)

        def forced_call(r, grp):
            return comm.cp_lse_ag_out_a2a_mha_uneven(w.o[r], w.lse[r], grp, w.counts, return_lse=True,
                                                     force_fused=True)

        with mock.patch.dict(os.environ, {"SGLANG_DCP_LSE_MERGE": "a2a"}):
            os.environ.pop("SGLANG_DCP_LSE_MERGE_FUSED", None)        # the global switch stays OFF
            _reset_all()
            base, log_b = F._run_world(3, base_call)
            _reset_all()
            forced, log_f = F._run_world(3, forced_call)
        for r in range(3):
            for x, y in zip(forced[r], base[r]):
                assert F._bitwise(x, y), f"rank {r} T={T}: forced fused merge not bit-identical"
            assert [o[0] for o in log_b[r]] == ["all_gather", "a2a"]
            assert [o[0] for o in log_f[r]] == ["a2a"]
    _reset_all()


def test_kvq_fusable_honours_the_force(monkeypatch):
    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend as B

    monkeypatch.delenv("SGLANG_DCP_FUSE_KVQ_GATHER", raising=False)
    _reset_all()
    self_ = SimpleNamespace(uneven_dcp=True, dcp_kv_replicated_heads=False, weightless_kv=False)
    layer = SimpleNamespace(tp_k_head_num=2, tp_v_head_num=2, head_dim=8)
    q = torch.zeros(5, 4, 8, dtype=torch.bfloat16)
    k = v = torch.zeros(5, 2, 8, dtype=torch.bfloat16)
    assert not B._dcp_kvq_fusable(self_, layer, k, v, q)
    assert B._dcp_kvq_fusable(self_, layer, k, v, q, force=True)
    self_.weightless_kv = True
    assert not B._dcp_kvq_fusable(self_, layer, k, v, q, force=True)
    _reset_all()


def test_wiring():
    from sglang.srt.layers.attention import flashinfer_backend as fb

    src = inspect.getsource(fb.FlashInferAttnBackend._forward_extend_dcp)
    assert "_dcp_comm.small_q_fuse_applies(" in src
    assert "self._dcp_kvq_fusable(layer, k, v, q_local, force=_sq)" in src
    assert src.count("force_fused=_sq,") == 2
    msrc = inspect.getsource(fb._dcp_uneven_merge)
    assert "force_fused=True" in msrc
