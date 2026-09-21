"""Task #65 (fnFL2v62, 21.09.): the sub-byte widening must not be born in a
weg2 tag pool.

A private ``torch.cuda.MemPool`` never hands a cached block back while it
lives (``emptyCache`` reaches ``graph_pools_freeable`` only), so every
transient allocated inside one keeps its segment for the whole boot.  The
v62 allocator snapshot measured what that costs on PP0: eleven chunk pools
holding 22.0 GiB reserved for 2.6 GiB of live weights, 82 of 302 segments
completely EMPTY, and the alloc history naming the source -- 161 GiB
cumulative in ``unpack_dense_subbyte`` plus 43 GiB in the widening, against
2.0 GiB of marlin repack output that actually stays.
"""

import types

import pytest
import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_wNa16 as wna16,
)
from sglang.srt.managers import weg2_memory_saver as MS


class _FakePool:
    def __init__(self, ident=(0, 7)):
        self.id = ident


@pytest.fixture
def stepped(monkeypatch):
    """An active tag pool plus a recorder for the begin/end allocate pair."""
    calls = []
    monkeypatch.setattr(MS, "_ACTIVE_TAG_POOL", _FakePool(), raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    mod = types.ModuleType("torch.cuda.memory")
    mod._cuda_endAllocateToPool = lambda dev, pid: calls.append(("end", dev, pid))
    mod._cuda_beginAllocateCurrentThreadToPool = lambda dev, pid: calls.append(
        ("begin", dev, pid)
    )
    monkeypatch.setitem(__import__("sys").modules, "torch.cuda.memory", mod)
    return calls


def test_without_an_active_pool_it_is_a_no_op():
    MS._ACTIVE_TAG_POOL = None
    with MS.outside_tag_pool(reason="x") as out:
        assert out is False


def test_it_leaves_and_comes_back_in_order(stepped):
    with MS.outside_tag_pool(reason="x") as out:
        assert out is True
        assert stepped == [("end", 0, (0, 7))]
    assert stepped == [("end", 0, (0, 7)), ("begin", 0, (0, 7))]


def test_it_comes_back_even_when_the_body_raises(stepped):
    with pytest.raises(RuntimeError):
        with MS.outside_tag_pool():
            raise RuntimeError("repack blew up")
    assert [c[0] for c in stepped] == ["end", "begin"]


def test_tag_pool_scope_publishes_and_restores_the_active_pool(monkeypatch):
    """Nesting a chunk scope inside the base scope must restore the outer pool."""
    import inspect

    src = inspect.getsource(MS.tag_pool_scope)
    assert "_ACTIVE_TAG_POOL = pool" in src
    assert "_ACTIVE_TAG_POOL = _prev_active" in src
    # the restore is in a finally, so an exception inside cannot strand it
    assert src.index("finally:") < src.index("_ACTIVE_TAG_POOL = _prev_active")


@pytest.mark.parametrize("src_bits", [4, 6])
def test_the_widening_is_numerically_unchanged(src_bits, monkeypatch):
    """The refactor must not move a bit: same input, same output."""
    torch.manual_seed(11)
    rows, in_features = 3, 64
    words = (in_features * src_bits + 31) // 32
    packed = torch.randint(
        -(2**31), 2**31 - 1, (rows, words), dtype=torch.int32
    )
    MS._ACTIVE_TAG_POOL = None
    got = wna16.widen_dense_packed_to_8bit(
        packed, src_bits=src_bits, in_features=in_features
    )
    want = wna16._widen_dense_packed_to_8bit(
        packed, src_bits=src_bits, in_features=in_features
    )
    assert torch.equal(got, want)
    assert got.dtype == torch.int32 and got.shape == (rows, in_features // 4)


def test_the_result_is_reallocated_inside_the_pool(stepped, monkeypatch):
    """With a pool active, the survivor is copied back in AFTER the step-in."""
    order = []
    real_empty_like = torch.empty_like

    def spy(t, *a, **kw):
        order.append("empty_like")
        return real_empty_like(t, *a, **kw)

    monkeypatch.setattr(torch, "empty_like", spy)
    packed = torch.zeros((2, 8), dtype=torch.int32)
    out = wna16.widen_dense_packed_to_8bit(packed, src_bits=4, in_features=64)
    # end -> (work) -> begin -> empty_like: the copy lands in the tag pool
    assert [c[0] for c in stepped] == ["end", "begin"]
    assert order == ["empty_like"]
    assert torch.equal(
        out, wna16._widen_dense_packed_to_8bit(packed, src_bits=4, in_features=64)
    )


def test_the_unpack_has_no_caller_outside_the_widening():
    """161 GiB of the measured waste is the unpack -- it must stay nested
    under the widening, or it needs its own step-out."""
    import pathlib
    import re

    root = pathlib.Path(wna16.__file__).resolve().parents[5]
    hits = []
    for py in root.rglob("*.py"):
        if "test" in py.name:
            continue
        try:
            text = py.read_text()
        except OSError:
            continue
        for m in re.finditer(r"^(?!def ).*\bunpack_dense_subbyte\s*\(", text, re.M):
            line = text[: m.start()].count(chr(10)) + 1
            hits.append((py.name, text[m.start() : m.end()].strip(), line))
    assert len(hits) == 1, hits
    name, call, line = hits[0]
    assert name == "compressed_tensors_wNa16.py", hits
    # and that one call sits inside the private impl, which runs stepped out
    body = pathlib.Path(wna16.__file__).read_text().split("\n")
    impl = next(
        i for i, l in enumerate(body, 1) if l.startswith("def _widen_dense_packed_to_8bit")
    )
    assert line > impl, (line, impl)
