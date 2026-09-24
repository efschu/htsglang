"""fnFL2 H69: one callable that runs right after a model runner's next CUDA
graph replay has been LAUNCHED -- host side, the device already working
through the graph.

WHY. A few host tasks need their inputs from the device (so they cannot run
before the round's previous work has finished) but are consumed by the graph
only partway through its replay. The PLE stage of a verify round is the one
that exists: its rows depend on the draft's tokens and are read by decoder
layer 1 behind a device gate
(``models/qwen4_exp_ple_decode_pread.py``, ``SGLANG_WEG2_PLE_STAGE_BEHIND_REPLAY``).
Run before the launch, the graph waits for the host; run here, the host works
while the graph does.

CONTRACT. ``arm(owner, fn)`` before the forward, ``fire(owner)`` from the
replay site of the runner ``owner`` (``DecodeCudaGraphRunner.execute``, right
after ``backend.replay``), ``disarm(owner)`` after the forward -- it returns
the callable when no replay fired it (an eager forward), so the caller can run
it late instead of losing it. One pending callable per owner; nothing is
armed unless a caller asked for it, so ``fire`` on an empty registry is one
truth test.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

__all__ = ["arm", "disarm", "fire", "pending"]

_PENDING: Dict[int, Callable[[], object]] = {}


def arm(owner: object, fn: Callable[[], object]) -> None:
    """Run ``fn`` right after ``owner``'s next graph replay is launched."""
    key = id(owner)
    if key in _PENDING:
        raise RuntimeError(
            "post-replay hook armed twice for the same runner: the previous "
            "forward neither replayed nor disarmed it"
        )
    _PENDING[key] = fn


def fire(owner: object) -> bool:
    """Called by the replay site: run and drop ``owner``'s pending callable.
    False when none was armed."""
    if not _PENDING:
        return False
    fn = _PENDING.pop(id(owner), None)
    if fn is None:
        return False
    fn()
    return True


def disarm(owner: object) -> Optional[Callable[[], object]]:
    """Drop ``owner``'s pending callable and hand it back; None when a replay
    already ran it (or nothing was armed)."""
    return _PENDING.pop(id(owner), None)


def pending(owner: object) -> bool:
    return id(owner) in _PENDING
