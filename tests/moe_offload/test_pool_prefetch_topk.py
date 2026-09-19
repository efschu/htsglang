"""Task #45: the pool prefetch may fetch only the m most probable of the next
router's k predictions (SGLANG_MOE_POOL_PREFETCH_TOPK)."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.models import qwen2_moe as qm


def test_default_is_the_routers_k(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_POOL_PREFETCH_TOPK", raising=False)
    qm._POOL_PREFETCH_TOPK["m"] = None
    assert qm.pool_prefetch_topk(10) == 10


def test_env_cuts_to_m_and_never_above_k_or_below_1(monkeypatch):
    monkeypatch.setenv("SGLANG_MOE_POOL_PREFETCH_TOPK", "5")
    qm._POOL_PREFETCH_TOPK["m"] = None
    assert qm.pool_prefetch_topk(10) == 5 and qm.pool_prefetch_topk(3) == 3
    monkeypatch.setenv("SGLANG_MOE_POOL_PREFETCH_TOPK", "0")
    qm._POOL_PREFETCH_TOPK["m"] = None
    assert qm.pool_prefetch_topk(10) == 10
    monkeypatch.setenv("SGLANG_MOE_POOL_PREFETCH_TOPK", "x")
    qm._POOL_PREFETCH_TOPK["m"] = None
    assert qm.pool_prefetch_topk(10) == 10
    qm._POOL_PREFETCH_TOPK["m"] = None
