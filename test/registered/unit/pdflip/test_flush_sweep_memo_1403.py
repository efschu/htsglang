# SPDX-License-Identifier: Apache-2.0
"""1403 (NF nf9 1004_105315): the flush publish sweep's no-progress memo.

DER BEFUND (nf9 P.log, Zeilen 144024/144160/144297/144421): each quiesce
``/flush_cache`` poll ran one full #1470 publish sweep whose rounds issued
nothing (host arena full, ARENA-DROP freed=0): "PDFLIP PUBLISH-SWEEP n=5707
unbacked=18 issued=0 refused=18 ... sweep_ms=1221.1 issue_ms=1219.8" and
"#1470 FLUSH-PUBLISH issued=0 unbacked_left=18 waited_ms=1221" -- four
identical ~1.2 s blocks inside the 9.8 s quiesce, plus the pass stalls they
cause on the pass slice ("#1466 PASS-STALL pp_rank=2 pass_ms=1177 fwd_ms=0").

THE MEMO: kvs2 W3 already breaks the loop after a round that issued nothing
with nothing in flight ("no pin frees, no write lands"). The memo carries
that verdict across polls: while the in-flight write-through/store count
stands where the no-progress round left it, the next poll skips the sweep
(FLLIPER_PDFLIP_FLUSH_SWEEP_MEMO, default off).
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _nb():
    from flliper.srt.managers import pdflip_flush_nonblock as nb

    return nb


class _Tree:
    def __init__(self, write_through=0, backup=0):
        self.ongoing_write_through = {i: object() for i in range(write_through)}
        self.ongoing_backup = {i: object() for i in range(backup)}


class _Sched:
    def __init__(self):
        self._pdflip_flush_sweep_memo = None


def test_no_progress_needs_issued_zero_pending_zero_unbacked_positive():
    nb = _nb()
    assert nb.sweep_no_progress({"issued": 0, "pending": 0, "unbacked": 18})
    assert not nb.sweep_no_progress({"issued": 3, "pending": 0, "unbacked": 18})
    assert not nb.sweep_no_progress({"issued": 0, "pending": 2, "unbacked": 18})
    assert not nb.sweep_no_progress({"issued": 0, "pending": 0, "unbacked": 0})
    assert not nb.sweep_no_progress({})


def test_inflight_count_sums_write_through_and_backup():
    nb = _nb()
    assert nb.inflight_count(None) == 0
    assert nb.inflight_count(_Tree(0, 0)) == 0
    assert nb.inflight_count(_Tree(3, 4)) == 7


def test_memo_hit_only_while_the_count_stands():
    nb = _nb()
    s = _Sched()
    tree = _Tree(2, 1)
    assert not nb.sweep_memo_hit(s, tree)  # unset
    nb.sweep_memo_set(s, tree)
    assert nb.sweep_memo_hit(s, tree)
    tree.ongoing_write_through.pop(0)  # a write-through acked: state moved
    assert not nb.sweep_memo_hit(s, tree)


def test_memo_switch_default_off_on_via_env(monkeypatch):
    nb = _nb()
    monkeypatch.delenv("FLLIPER_PDFLIP_FLUSH_SWEEP_MEMO", raising=False)
    assert not nb.sweep_memo_on()
    monkeypatch.setenv("FLLIPER_PDFLIP_FLUSH_SWEEP_MEMO", "1")
    assert nb.sweep_memo_on()


def test_scheduler_flush_handler_is_wired():
    """The handler checks the memo before its rounds, skips the loop on a hit,
    and arms the memo exactly where the kvs2 W3 break fires."""
    from flliper.srt.managers import scheduler as sched_mod

    src = inspect.getsource(sched_mod.Scheduler.flush_cache)
    assert "sweep_memo_hit" in src
    assert "sweep_memo_set" in src
    assert "range(64) if not _memo_hit else ()" in src
