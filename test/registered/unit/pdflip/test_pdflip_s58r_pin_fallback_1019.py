"""S58r (review of S58 ad02005e6d): the exact-pin fallback must not leave a CUDA error behind.

S58 made the exact pin (anonymous mapping + ``cudaHostRegister``) the default for the tail-stage arena and the
draft image and falls back to ``torch.empty(pin_memory)`` when the registration fails. A failed CUDA runtime call
leaves its code in the calling thread's error variable; the next kernel-launch check of that thread
(``C10_CUDA_KERNEL_LAUNCH_CHECK`` = ``cudaGetLastError()``) raises it on an unrelated launch -- the "fallback"
would end in a CUDA error blamed on the boot's next kernel (the arena is pinned in ``tp_worker`` init, on the
scheduler's thread). ``pinned_exact_empty`` now reads the error out before it raises.

Hermetic, CPU only: the CUDA runtime is a recorded fake.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

import flliper.srt.layers.moe.expert_offload as eo  # noqa: E402
from flliper.srt.pdflip import tail_adopt as ta  # noqa: E402


class _FakeCudart:
    def __init__(self, err):
        self.err = err
        self.registered = []

    def cudaHostRegister(self, ptr, nbytes, flags):  # noqa: N802 - CUDA's name
        self.registered.append((ptr, nbytes, flags))
        return self.err

    def cudaHostUnregister(self, ptr):  # noqa: N802
        return 0


@pytest.fixture
def cuda_fake(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(eo, "release_torch_host_cache", lambda: None)
    monkeypatch.setattr(eo, "_clear_cuda_last_error", lambda: calls.append("clear"))

    def install(err):
        rt = _FakeCudart(err)
        monkeypatch.setattr(torch.cuda, "cudart", lambda: rt)
        return rt

    return install, calls


def test_a_failed_registration_clears_the_thread_cuda_error_before_it_raises(cuda_fake):
    install, calls = cuda_fake
    rt = install(2)   # cudaErrorMemoryAllocation
    with pytest.raises(RuntimeError, match="cudaHostRegister"):
        eo.pinned_exact_empty((4096,), torch.uint8)
    assert rt.registered and rt.registered[0][1] == 4096
    assert calls == ["clear"], "the fallback would inherit the error: the next kernel-launch check raises it"


def test_a_successful_registration_touches_no_error_state(cuda_fake):
    install, calls = cuda_fake
    install(0)
    t = eo.pinned_exact_empty((3 * 4096 + 1152,), torch.uint8)   # not a page multiple, like the D TP0 arena
    assert t.numel() == 3 * 4096 + 1152
    assert calls == []
    eo._pinned_exact_unregister(t.data_ptr())


def test_the_tail_stage_fallback_runs_after_the_error_is_cleared(cuda_fake, monkeypatch):
    install, calls = cuda_fake
    install(2)
    order = []
    real_empty = torch.empty

    def fake_empty(*args, **kw):
        if kw.get("pin_memory"):
            order.append(("torch.empty(pin)", list(calls)))
            kw = {k: v for k, v in kw.items() if k != "pin_memory"}
        return real_empty(*args, **kw)

    monkeypatch.setattr(ta.torch, "empty", fake_empty)
    arena = ta.PinArena(8192, pin=True)
    assert arena.nbytes == 8192 and arena.buf is not None
    assert order == [("torch.empty(pin)", ["clear"])], "the fallback pins only after the error was read out"


def test_clearing_never_raises_without_a_runtime(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", None)
    eo._clear_cuda_last_error()
    monkeypatch.setattr(torch.version, "cuda", "99.0")   # no libcudart.so.99 anywhere
    eo._clear_cuda_last_error()
