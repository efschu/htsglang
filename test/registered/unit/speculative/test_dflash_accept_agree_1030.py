# SPDX-License-Identifier: Apache-2.0
"""27b-accept-rank-uniform-1030: the DFLASH greedy accept's Triton path is switched off for all ranks of a TP group or for none.

CPU only. Two ranks run as two threads; their TP CPU group is a barrier stand-in that computes the element-wise MIN like
``reduce_min_cpu``. A rank that calls the reduction a different number of times than its peer breaks the barrier (timeout)
instead of hanging, so the test fails -- the falsifier of the hang class.

Bug (red before the fix): a one-sided Triton failure (OOM in the slot buffers, a launch error on sm_86 but not on sm_120) made
only the failing rank fall back to the eager path, which runs none of the DFLASH_ACCEPT_GREEDY broadcasts; the peers kept
waiting inside them and the TP group hung."""

import threading

import pytest

from flliper.srt.speculative.dflash_accept_agree import AGREEMENT_ROUNDS, resolve_triton_accept

_TIMEOUT_S = 10.0


class _Group:
    def __init__(self, n: int):
        self.calls = [0] * n
        self._vals = [None] * n
        self._out = None
        self._b = threading.Barrier(n, timeout=_TIMEOUT_S)

    def reducer(self, rank: int):
        def reduce_min(values):
            self.calls[rank] += 1
            self._vals[rank] = [int(v) for v in values]
            if self._b.wait() == 0:
                self._out = [min(col) for col in zip(*self._vals)]
            self._b.wait()
            return list(self._out)

        return reduce_min


def _run_ranks(group: _Group, errors, rounds_left: int):
    results = [None, None]

    def worker(rank: int):
        results[rank] = resolve_triton_accept(
            error=errors[rank], rounds_left=rounds_left, reduce_min=group.reducer(rank)
        )

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_one_sided_failure_sends_the_whole_group_to_eager():
    group = _Group(2)
    results = _run_ranks(group, errors=[None, RuntimeError("OOM on rank 1")], rounds_left=AGREEMENT_ROUNDS)
    assert results == [(False, AGREEMENT_ROUNDS - 1)] * 2
    assert group.calls == [1, 1]


def test_after_the_window_a_local_failure_raises_and_a_healthy_round_costs_no_collective():
    group = _Group(2)
    with pytest.raises(RuntimeError, match="peer ranks are inside"):
        resolve_triton_accept(error=ValueError("launch"), rounds_left=0, reduce_min=group.reducer(0))
    assert resolve_triton_accept(error=None, rounds_left=0, reduce_min=group.reducer(1)) == (True, 0)
    assert group.calls == [0, 0]


def test_single_rank_keeps_the_graceful_fallback():
    assert resolve_triton_accept(error=ValueError("launch"), rounds_left=AGREEMENT_ROUNDS, reduce_min=None) == (False, 0)
    assert resolve_triton_accept(error=None, rounds_left=AGREEMENT_ROUNDS, reduce_min=None) == (True, 0)
