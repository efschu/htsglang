"""NF-TOD 20.10. 20:08:05Z: the ROW-FENCE's stream registry never holds a weakref to a stream.

THE METAL (rc28 297198745d, boot ..._1020_174205): P PP1 (pid 909) died by SIGSEGV in
``PyObject_RichCompare`` <- ``WeakSet.add`` (_weakrefset.py:88) <- ``d_row_fence.register_row_stream`` <-
``expert_offload._CudaStreamOps.new_stream`` <- ``DeferredRowsFill.start`` (the #284b fill-start of the
D->P wake, epoch 62). Cause: torch 2.11's ``torch.cuda.Stream`` inherits its weakref slot from the C base
(``__weakrefoffset__`` 48 on torch.Stream, _CudaStreamBase and torch.cuda.Stream; no own ``__weakref__``),
so CPython's ``subtype_dealloc`` leaves the clearing to the C dealloc -- and ``THCPStream_dealloc`` never
calls ``PyObject_ClearWeakRefs``. A weakref to a dead stream keeps pointing at freed memory; the WeakSet
entry stays (no callback), and the next pool stream with the same handle (same hash) compares against it.

What must hold:
(1) registering a stream creates NO weakref to it (red on the WeakSet registry);
(2) the registry holds the stream strongly -- dropping every caller handle keeps it registered;
(3) every handle of one CUDA stream (same device_type / device_index / stream_id) is ONE entry;
(4) distinct streams stay distinct, objects without stream ids are keyed by identity (idempotent);
(5) d_row_fence imports no weakref at all (the class cannot come back by a later edit);
(6) the torch premise itself (no CUDA needed): the subclass carries no own weakref slot.
"""
from __future__ import annotations

import gc
import inspect
import os
import weakref

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.pdflip import d_row_fence as F  # noqa: E402


class _TorchLikeStream:
    """Carries the three int ids of ``torch.Stream`` and is weak-referenceable (like torch.cuda.Stream)."""

    def __init__(self, sid, dix=0, dty=1):
        self.stream_id, self.device_index, self.device_type = sid, dix, dty


class _Plain:
    pass


@pytest.fixture(autouse=True)
def _clean():
    F.reset_for_tests()
    yield
    F.reset_for_tests()


def test_1_register_creates_no_weakref():
    s = _TorchLikeStream(17)
    F.register_row_stream(s)
    assert weakref.getweakrefcount(s) == 0
    p = _Plain()
    F.register_row_stream(p)
    assert weakref.getweakrefcount(p) == 0


def test_2_registry_holds_the_stream_after_every_handle_died():
    F.register_row_stream(_TorchLikeStream(5))
    gc.collect()
    got = F.row_streams()
    assert len(got) == 1 and got[0].stream_id == 5


def test_3_handles_of_one_stream_are_one_entry():
    a, b = _TorchLikeStream(9), _TorchLikeStream(9)
    F.register_row_stream(a)
    F.register_row_stream(b)
    got = F.row_streams()
    assert len(got) == 1 and got[0] is a


def test_4_distinct_streams_and_identity_keys():
    F.register_row_stream(_TorchLikeStream(1))
    F.register_row_stream(_TorchLikeStream(1, dix=1))
    F.register_row_stream(_TorchLikeStream(2))
    p = _Plain()
    F.register_row_stream(p)
    F.register_row_stream(p)
    F.register_row_stream(None)
    got = F.row_streams()
    assert len(got) == 4 and sum(x is p for x in got) == 1
    # a non-int id (mock-like) never collapses distinct objects into one key
    m1, m2 = _Plain(), _Plain()
    m1.stream_id = m2.stream_id = object()
    m1.device_index = m2.device_index = 0
    m1.device_type = m2.device_type = 1
    F.register_row_stream(m1)
    F.register_row_stream(m2)
    assert len(F.row_streams()) == 6


def test_5_the_fence_module_imports_no_weakref():
    src = inspect.getsource(F)
    assert "import weakref" not in src and "WeakSet(" not in src


def test_6_torch_premise_the_cuda_stream_has_no_own_weakref_slot():
    torch = pytest.importorskip("torch")
    if not hasattr(torch._C, "_CudaStreamBase") or not hasattr(torch._C, "Stream"):
        pytest.skip("torch without the CUDA stream base")
    base_off = torch._C.Stream.__weakrefoffset__
    assert base_off != 0
    # the subclass inherits the base's slot: subtype_dealloc does not clear it, the C dealloc would have to
    assert torch.cuda.Stream.__weakrefoffset__ == base_off
    assert "__weakref__" not in torch.cuda.Stream.__dict__
