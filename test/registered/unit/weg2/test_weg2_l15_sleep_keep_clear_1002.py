# SPDX-License-Identifier: Apache-2.0
"""L15-D2FIX (LCWAKE finding D2): a D sleep that does NOT arm a hold clears
the TMS keep set when the L1.5 master is on.

A refused (W29) wake bails before the restore, so the keep set the previous
sleep armed stays on the allocations; the next sleep REPLACES keep sets only
when it retains. Every retain-skipping sleep took the plain flush, which had
no keep-clear -> stale keep spans pinned VRAM through the P phase. The flush
now clears them in the plain branch (master on, D parked) via the wake's own
helper (one base walk, no second enumeration).
"""

from __future__ import annotations

import pathlib

_SRC = (pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" /
        "srt" / "managers" / "scheduler.py")


def test_plain_branch_clears_keep_set_under_master():
    src = _SRC.read_text()
    i_plain = src.find("self.tree_cache.reset()\n                self.req_to_token_pool.clear()")
    assert i_plain != -1, "plain-flush branch moved -- update this seam test"
    window = src[i_plain - 2500:i_plain]
    assert "L15-D2FIX" in window, "no keep-clear before the plain flush"
    assert "_l15_clear_tms_keep_spans(self)" in window
    assert "L15-KEEP-CLEAR at=sleep" in window
    assert "master_on(os.environ)" in window, "must be gated on the master"
