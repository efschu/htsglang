"""PDFLIP-SLEEP-HOLDER on NF rc12z17 (103bfdf29a, boot dkrnfh91dprsavisnoadoptbar1dauer09281044, P.log
10:50:11/12 and 10:51:24/25, all three P ranks, every sleep): 'skipped (RuntimeError: Cannot access
data pointer of Tensor (e.g. FakeTensor, FunctionalTensor) ...)'. The heap scan of holder_report
reached a fake/functional tensor (torch.compile / dynamo leave them on the heap); its data_ptr()
raises, and the one exception discarded the whole report -- the hc_combine proof never printed.

Fix: an object whose data pointer cannot be read is passed over and counted (skipped_no_ptr=N on
every line); the scan names the holders of the real tensors as before."""

import torch

from flliper.srt.pdflip import sleep_staging as ss


class _NoPtr(torch.Tensor):
    """Stands in for FakeTensor/FunctionalTensor: a CUDA-looking tensor whose data_ptr raises."""

    @property
    def is_cuda(self):
        return True

    def data_ptr(self):
        raise RuntimeError(
            "Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor). "
            "If you're using torch.compile/export/fx, it is likely that we are erroneously tracing"
        )


class _AtPtr(torch.Tensor):
    """A CUDA-looking tensor at a chosen address (no GPU on the desk)."""

    _ptr = 0

    @property
    def is_cuda(self):
        return True

    def data_ptr(self):
        return type(self)._ptr


def _snap(addr, size):
    return {"segments": [{"address": addr, "blocks": [
        {"size": size, "state": "active_allocated",
         "frames": [{"filename": "/x/flliper/kernels/ops/elementwise/hc_combine.py", "line": 86,
                     "name": "hc_combine"}]},
    ]}]}


def test_a_fake_tensor_on_the_heap_does_not_discard_the_report():
    fake = torch.zeros(2).as_subclass(_NoPtr)  # noqa: F841 -- held by this frame, like dynamo's
    _AtPtr._ptr = 1 << 32
    real = torch.zeros(2).as_subclass(_AtPtr)

    def event_loop():
        result = real  # noqa: F841 -- the frame local the holder report must still name
        return ss.holder_report(_snap(1 << 32, 1 << 20), top=4, depth=3)

    lines = event_loop()
    assert len(lines) == 1, lines
    assert "frame event_loop local result" in lines[0], lines
    assert "skipped_no_ptr=" in lines[0] and "skipped_no_ptr=0" not in lines[0], lines
    # the scan's own temporaries are no holders: its (tensor, ptr) pairs, the hit slice, and the
    # (name, value) pairs of the live-stack index (kept alive by a frame cycle before this fix)
    assert "tuple" not in lines[0] and "list <-" not in lines[0], lines


def test_live_tensors_are_pointer_pairs_and_the_bad_ones_are_counted():
    fake = torch.zeros(2).as_subclass(_NoPtr)  # noqa: F841
    pairs, skipped = ss.live_cuda_tensor_ptrs()
    assert skipped >= 1
    assert all(isinstance(p, int) for _, p in pairs)
    assert not any(t is fake for t, _ in pairs)
