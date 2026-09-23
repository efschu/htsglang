"""weg2xsn297 (Task #13): the prefetch-pending skip names the intake stall
when the read cannot be funded (need > free pool, nothing running)."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import intake_stall as st  # noqa: E402


def test_unfundable_pending_read_with_nothing_running_is_a_stall():
    assert st.pending_prefetch_is_a_stall(need_tokens=99572, pool_free_tokens=23000,
                                          running_empty=True, waiting=1)
    # a fundable read is a slow read, never a stall
    assert not st.pending_prefetch_is_a_stall(need_tokens=4316, pool_free_tokens=23000,
                                              running_empty=True, waiting=1)
    # something running may free rows
    assert not st.pending_prefetch_is_a_stall(need_tokens=99572, pool_free_tokens=23000,
                                              running_empty=False, waiting=1)
    assert not st.pending_prefetch_is_a_stall(need_tokens=99572, pool_free_tokens=23000,
                                              running_empty=True, waiting=0)


def test_the_skip_site_feeds_the_watch():
    from sglang.srt.managers import scheduler as sch
    src = open(sch.__file__).read()
    i = src.index('_note_skip("prefetch_pending_pp0", req.rid)')
    blk = src[i - 1500:i]
    assert "pending_prefetch_is_a_stall(" in blk and "gate=prefetch_pending" in blk
    assert "self._weg2_intake_stall_observe(" in blk
