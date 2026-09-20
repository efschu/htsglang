"""fnFL2 v4-v6 (20.09.): under --flip-weights resident the weights load must
NOT run inside the memory-saver's private MemPool -- freed load transients in
a private pool stay reserved through torch.cuda.empty_cache() (metal probe:
1.8 GiB freed inside a MemPool, still reserved; PP1 13.05 GiB reserved vs
5.97 allocated -> KV sizing refused). The tag is still published."""

from sglang.srt.managers import weg2_memory_saver as ws


class _Adapter:
    def __init__(self):
        self.regions = []

    def region(self, tag, enable_cpu_backup=False):
        self.regions.append((tag, enable_cpu_backup))
        from contextlib import nullcontext

        return nullcontext()


def test_resident_arm_publishes_the_tag_but_opens_no_pool(monkeypatch):
    monkeypatch.setenv(ws.WEIGHTS_RESIDENT_ENV, "1")
    pools = []
    monkeypatch.setattr(ws, "tag_pool_scope", lambda tag: pools.append(tag) or __import__("contextlib").nullcontext())
    a = _Adapter()
    with ws.weights_region(a, ws.GPU_MEMORY_TYPE_WEIGHTS, enable_cpu_backup=False) as tag:
        assert tag == ws.GPU_MEMORY_TYPE_WEIGHTS
        assert ws.current_weights_region_tag() == ws.GPU_MEMORY_TYPE_WEIGHTS
    assert a.regions == [] and pools == []


def test_pausable_arm_keeps_the_pool(monkeypatch):
    monkeypatch.delenv(ws.WEIGHTS_RESIDENT_ENV, raising=False)
    pools = []
    monkeypatch.setattr(ws, "tag_pool_scope", lambda tag: pools.append(tag) or __import__("contextlib").nullcontext())
    a = _Adapter()
    with ws.weights_region(a, ws.GPU_MEMORY_TYPE_WEIGHTS, enable_cpu_backup=True) as tag:
        assert tag == ws.GPU_MEMORY_TYPE_WEIGHTS
    assert a.regions == [(ws.GPU_MEMORY_TYPE_WEIGHTS, True)] and pools == [ws.GPU_MEMORY_TYPE_WEIGHTS]


def test_illegal_tag_is_refused_on_both_arms(monkeypatch):
    import pytest

    for armed in ("1", "0"):
        monkeypatch.setenv(ws.WEIGHTS_RESIDENT_ENV, armed)
        with pytest.raises(ValueError):
            with ws.weights_region(_Adapter(), "weights_3", enable_cpu_backup=False):
                pass
