# SPDX-License-Identifier: Apache-2.0
"""h16v S13 (Opus-Kurzpruefung A1/A2, K22): the flush sweep memo's stamp.

A1: D runs pp=1 with overlap; quiesce_end('batch') is only called in the PP loop, so the stamp must
    carry ``forward_ct`` (moves in _run_batch_forward of EVERY loop) -- a batch between two polls voids it.
A2: ``available_size()`` of the arena pool is ``slots - claimed`` (COMPLETE counts as free); the stamp
    reads ``(complete, claimed)`` of ``arena.stats()`` so a COMPLETE slot turning free moves it.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _nb():
    from sglang.srt.managers import weg2_flush_nonblock as nb

    return nb


class _Arena:
    def __init__(self, slots, complete, claimed):
        self.slots, self.complete, self.claimed = slots, complete, claimed

    def stats(self):
        return {"slots": self.slots, "complete": self.complete, "claimed": self.claimed}


class _ArenaPool:
    """available_size() like arena_pool.py:3094: (slots - claimed), COMPLETE counted as free."""

    def __init__(self, arena):
        self.arena = arena

    def available_size(self):
        return max(0, self.arena.slots - self.arena.claimed)


class _Tree:
    def __init__(self, arena):
        self.cache_controller = type("CC", (), {"mem_pool_host": _ArenaPool(arena)})()
        self.ongoing_write_through = {}
        self.ongoing_backup = {}


class _Sched:
    def __init__(self):
        self._weg2_flush_sweep_memo = None
        self.forward_ct = 7


def test_a1_batch_between_polls_without_quiesce_end_voids_the_memo():
    """Non-PP loop (D, pp=1 overlap): no bubble_end/quiesce_end('batch'); only forward_ct moves."""
    nb = _nb()
    s, tree = _Sched(), _Tree(_Arena(10, 3, 2))
    nb.sweep_memo_set(s, tree)
    assert nb.sweep_memo_hit(s, tree)  # second poll, nothing ran: skip
    s.forward_ct += 1  # a batch ran between the polls (state may be the same again afterwards)
    assert not nb.sweep_memo_hit(s, tree)  # first poll after it sweeps


def test_a1_forward_ct_is_the_stamp_part_and_run_batch_forward_moves_it():
    from sglang.srt.managers import scheduler as sched_mod

    nb = _nb()
    assert "forward_ct" in inspect.getsource(nb.sweep_state_stamp)
    assert "self.forward_ct += 1" in inspect.getsource(sched_mod.Scheduler._run_batch_forward)
    assert "sweep_memo_hit(\n                                 self, self.tree_cache)" in inspect.getsource(
        sched_mod.Scheduler.flush_cache) or "sweep_memo_hit" in inspect.getsource(sched_mod.Scheduler.flush_cache)


def test_a2_complete_slot_turning_free_moves_the_stamp():
    nb = _nb()
    s, arena = _Sched(), _Arena(10, 3, 2)
    tree = _Tree(arena)
    nb.sweep_memo_set(s, tree)
    avail = tree.cache_controller.mem_pool_host.available_size()
    arena.complete -= 1  # a COMPLETE slot became free (claimed unchanged)
    assert tree.cache_controller.mem_pool_host.available_size() == avail  # the old stamp part saw nothing
    assert not nb.sweep_memo_hit(s, tree)


def test_a2_claim_or_completion_moves_the_stamp_and_unchanged_does_not():
    nb = _nb()
    s, arena = _Sched(), _Arena(10, 3, 2)
    tree = _Tree(arena)
    nb.sweep_memo_set(s, tree)
    assert nb.sweep_memo_hit(s, tree)
    arena.claimed, arena.complete = 1, 4  # a claim completed: free unchanged, composition moved
    assert not nb.sweep_memo_hit(s, tree)


def test_a2_arena_free_docstring_names_the_real_source():
    doc = _nb().arena_free.__doc__
    assert "complete" in doc and "claimed" in doc
    assert _nb().arena_free(_Tree(_Arena(10, 3, 2))) == (3, 2)
