"""MEMRANK-1015 leak fix 3: the front's ``_x_requeues`` (rid -> count) was never retired."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front  # noqa: E402


def test_count_semantics_unchanged():
    t = {}
    assert front.bump_x_requeues(t, "a") == 1
    assert front.bump_x_requeues(t, "a") == 2
    assert front.bump_x_requeues(t, "b") == 1
    assert front.bump_x_requeues(t, "a") == 3
    assert not front.x_requeue_capped(front.X_REQUEUE_CAP)
    assert front.x_requeue_capped(front.X_REQUEUE_CAP + 1)


def test_table_is_bounded_oldest_out():
    t = {}
    n = front.X_REQUEUES_SLOTS + 300
    for i in range(n):
        front.bump_x_requeues(t, "r%d" % i)
    assert len(t) == front.X_REQUEUES_SLOTS
    assert "r0" not in t and "r299" not in t
    assert "r300" in t and ("r%d" % (n - 1)) in t


def test_front_uses_the_helper_not_a_bare_dict_increment():
    import inspect

    src = inspect.getsource(front.Front._requeue_after_x_refusal) if hasattr(front, "Front") else ""
    if src:
        assert "bump_x_requeues(self._x_requeues" in src
        assert "self._x_requeues[rid] = n" not in src
