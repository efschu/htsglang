# SPDX-License-Identifier: Apache-2.0
"""DP-NACHLAUF 02.10.: the uneven DCP merge hands barlink its group-wide
largest a2a block -- no gloo group_max (a CPU rendezvous of every rank) on
every merge call.

N6k (3e0b2cd3a3): D's first P>D extend, merge_scatter host 105-195 ms
against device 110-119 ms over 16 full-attention layers -- host-bound, no
device wait, switch interval already 0.2 ms. The merge's a2a passes BOTH
split lists, so barlink took the slot decision from _group_max over gloo.
Pinned (red before): with the hint barlink never calls _group_max and asks
the transport with max(rows) * row_bytes -- the value _group_max would have
returned; a hint below this rank's own block raises; without a hint the
group_max runs as before; the merge helper passes max(head_counts) only to a
group class that takes the parameter (the threaded test groups keep their
signature); GroupCoordinator forwards it.
"""
import inspect
import os
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.distributed.device_communicators import barlink as B  # noqa: E402
from flliper.srt.layers.dcp import comm  # noqa: E402


class _T:
    def __init__(self):
        self.asked = []

    def supports_a2a(self, largest):
        self.asked.append(largest)
        return True

    def barlink_all_to_all_single(self, comm_, output, inp, send_bytes, recv_bytes, rounds=None):
        return output


def _comm(t, rank=1, world=3):
    c = object.__new__(B.BarlinkCommunicator)
    c.disabled, c.world_size, c.rank = False, world, rank
    c.cpu_group, c._peer_table = None, None
    c._select = lambda op, n: t
    c._after_transport = lambda t_, op: None
    return c


def _io(counts, mine, tokens=5, dim=8):
    send = torch.zeros(sum(counts), tokens, dim)
    recv = torch.zeros(len(counts) * mine, tokens, dim)
    return recv, send, tokens * dim * 4


def test_hint_skips_group_max_and_matches_its_value(monkeypatch):
    counts = [12, 6, 6]
    t = _T()
    c = _comm(t, rank=1)
    recv, send, row_bytes = _io(counts, counts[1])

    def boom(*a, **k):
        raise AssertionError("gloo group_max called despite the hint")

    monkeypatch.setattr(B, "_group_max", boom)
    c.all_to_all_single(recv, send, [counts[1]] * 3, counts, largest_block_rows=max(counts))
    assert t.asked == [max(counts) * row_bytes]


def test_no_hint_runs_group_max_and_too_small_hint_raises(monkeypatch):
    counts = [12, 6, 6]
    t = _T()
    c = _comm(t, rank=2)
    recv, send, row_bytes = _io(counts, counts[2])
    calls = []
    monkeypatch.setattr(B, "_group_max", lambda v, g, table=None: (calls.append(v), 12 * row_bytes)[1])
    c.all_to_all_single(recv, send, [counts[2]] * 3, counts)
    assert calls and t.asked == [12 * row_bytes]
    with pytest.raises(ValueError, match="largest_block_rows"):
        c.all_to_all_single(recv, send, [counts[2]] * 3, counts, largest_block_rows=6)


def test_helper_passes_the_hint_only_to_classes_that_take_it(monkeypatch):
    monkeypatch.delenv(comm.A2A_BLOCK_HINT_ENV, raising=False)
    comm._A2A_HINT.update({"on": None, "cls": {}, "said": False})
    seen = {}

    class WithHint:
        def all_to_all_single_v(self, out, inp, output_split_sizes=None, input_split_sizes=None,
                                largest_block_rows=None):
            seen["with"] = largest_block_rows

    class Without:
        def all_to_all_single_v(self, out, inp, output_split_sizes=None, input_split_sizes=None):
            seen["without"] = True

    comm._a2a_v(WithHint(), None, None, [6] * 3, [12, 6, 6], 12)
    comm._a2a_v(Without(), None, None, [6] * 3, [12, 6, 6], 12)
    assert seen == {"with": 12, "without": True}
    monkeypatch.setenv(comm.A2A_BLOCK_HINT_ENV, "0")
    comm._A2A_HINT.update({"on": None})
    seen.clear()
    comm._a2a_v(WithHint(), None, None, [6] * 3, [12, 6, 6], 12)
    assert seen == {"with": None}
    comm._A2A_HINT.update({"on": None, "cls": {}, "said": False})


def test_wiring():
    from flliper.srt.distributed import parallel_state as ps

    src = inspect.getsource(ps.GroupCoordinator.all_to_all_single_v)
    assert "largest_block_rows=largest_block_rows" in src
    csrc = inspect.getsource(comm)
    assert csrc.count("_a2a_v(cp_group, recv, send, [mine] * world, counts, max(counts))") == 2
