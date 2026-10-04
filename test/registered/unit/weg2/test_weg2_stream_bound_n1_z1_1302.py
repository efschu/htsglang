"""Q-1302 STREAM-BOUND: the index blocks of the E1 PLE apply (N1) and the E1
tail install (Z1) are bound to the stream that reads them.

Hermetic (no CUDA). Class of Fall E (Q-702, NF y9nf3 10040106 @ 458851283d,
P PP1 pid 707): a kernel queued on the FORWARD stream reads an index tensor
whose storage the scheduler thread's stream owns; the last Python reference
drops on the host before the kernel ran, the caching allocator hands the block
back to the schedule stream, the next small tensor overwrites it, the kernel
reads the garbage (device assert / silent write into another request's slot).

* N1 ``ple_state.queue`` clones the slot index at the admission commit
  (schedule stream); ``apply_pending`` (from ``_prepare_ple_batch``, inside
  the forward) pops the entry right after queueing ``index_copy_`` on the
  forward stream -- the clone's last reference dies with that iteration.
* Z1 ``tail_adopt._put`` (E1 install: ``rows`` / ``groups`` / ``slot`` made on
  the schedule stream) reads them with no copy; they lived on only through the
  verify thread's reference (SGLANG_WEG2_TAIL_VERIFY / the stage worker).

A CPU test cannot reproduce allocator reuse across streams, so the cases pin
the STRUCTURE: the index tensor the kernel is queued with gets
``record_stream(<the reading stream>)`` before the owner drops it. The stand-in
below is a CUDA-looking tensor (``is_cuda`` True) that records those calls.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.weg2 import ple_state
from sglang.srt.weg2 import tail_adopt as ta

FWD = object()  # the reading (forward) stream


class _CudaLike(torch.Tensor):
    """A CPU tensor that claims to be a CUDA block and logs record_stream."""

    log = []

    @property
    def is_cuda(self):
        return True

    def record_stream(self, stream):
        _CudaLike.log.append((self.data_ptr(), stream))


def _cuda_like(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype).as_subclass(_CudaLike)


@pytest.fixture(autouse=True)
def _stream(monkeypatch):
    _CudaLike.log.clear()
    ple_state.PENDING.clear()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: FWD)
    yield
    ple_state.PENDING.clear()


def test_n1_apply_pending_binds_the_queued_slot_to_the_reading_stream(monkeypatch):
    seen = []
    # the install of the queued rows; what it is handed is what its kernels read
    monkeypatch.setattr(ple_state, "install", lambda pool, slot, rows: seen.append(slot) or "")
    ple_state.queue("rid-n1", _cuda_like([7]), {"conv": torch.zeros(1)})
    assert not _CudaLike.log  # queue() runs on the schedule stream: nothing to bind yet
    assert ple_state.apply_pending(object(), ["rid-n1"]) == 1
    assert len(seen) == 1 and not ple_state.PENDING
    assert _CudaLike.log == [(seen[0].data_ptr(), FWD)], "the slot block the install reads was not record_stream'd"


def test_n1_refused_install_still_leaves_nothing_queued(monkeypatch):
    monkeypatch.setattr(ple_state, "install", lambda pool, slot, rows: "not_owner")
    ple_state.queue("rid-n1", _cuda_like([7]), {"conv": torch.zeros(1)})
    assert ple_state.apply_pending(object(), ["rid-n1"]) == 0 and not ple_state.PENDING


def test_hold_for_current_stream_is_a_noop_off_cuda():
    cpu = torch.tensor([3])
    ple_state.hold_for_current_stream(cpu, "unit")  # must not raise: CPU tensors have no stream
    ple_state.hold_for_current_stream(None, "unit")
    assert not _CudaLike.log


def test_z1_put_binds_an_uncopied_index_and_leaves_a_copied_one():
    dst, src = torch.zeros(8, 4, dtype=torch.uint8), torch.arange(8, dtype=torch.uint8).view(2, 4)
    idx = _cuda_like([5, 2])
    ta._put(dst, idx, src, None)
    assert _CudaLike.log == [(idx.data_ptr(), FWD)]
    assert dst[5].tolist() == [0, 1, 2, 3] and dst[2].tolist() == [4, 5, 6, 7]  # values unchanged
    _CudaLike.log.clear()
    # an index of another dtype is COPIED on the reading stream by the .to(): that copy is its own
    ta._put(dst, _cuda_like([1, 3], dtype=torch.int32), src, None)
    assert not _CudaLike.log and dst[1].tolist() == [0, 1, 2, 3]


def test_z1_e1_install_binds_rows_groups_and_slot(monkeypatch):
    """The whole E1 install path (attention rows + compressed groups at the
    first GDN read, the GDN state at the slot): every schedule-stream index
    the forward stream reads is bound, with no verify thread to hold it."""
    rows, groups, slot = _cuda_like([0, 1, 2, 3]), _cuda_like([0]), _cuda_like([4])
    fa_dst = (torch.zeros(8, 2, dtype=torch.uint8), torch.zeros(8, 2, dtype=torch.uint8),
              torch.zeros(2, 2, dtype=torch.uint8))
    host_fa = (torch.ones(4, 2, dtype=torch.uint8), torch.ones(4, 2, dtype=torch.uint8),
               torch.ones(1, 2, dtype=torch.uint8))
    gdn_dst = (torch.zeros(6, 2, dtype=torch.uint8),)
    inst = ta.Install(spec=SimpleNamespace(rid="rid-z1"), headers=[], fa={5: host_fa},
                      gdn={0: (torch.full((1, 2), 9, dtype=torch.uint8),)}, fa_dst={5: fa_dst},
                      gdn_dst={0: gdn_dst}, rows=rows, groups=groups, slot=slot, t0=0.0)
    finished = []
    monkeypatch.setattr(ta, "PENDING_INSTALLS", [inst])
    monkeypatch.setattr(ta, "_finish", lambda i, verify: finished.append(i))
    ta._install_step(inst, 0)
    assert finished == [inst] and not ta.PENDING_INSTALLS
    bound = {ptr for ptr, stream in _CudaLike.log if stream is FWD}
    assert {rows.data_ptr(), groups.data_ptr(), slot.data_ptr()} <= bound
    assert gdn_dst[0][4].tolist() == [9, 9] and fa_dst[0][3].tolist() == [1, 1]  # the rows landed
