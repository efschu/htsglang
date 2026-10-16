"""H16N-REVIEW 1015 (rank-agreement review of the h16n stack, raenge-nie-uneins).

TAIL-CONSUME-ON-FULL (60ad6ba20a) judged ``full`` on the UNPACKED reap values:
``_hp1_end_unpack(base, reduced) = max(0, reduced - base)`` subtracts THIS
rank's HP1 end base. On a Form A group the bases differ (TP0 0, the workers
32704 -- rc12z20) and the store probe MIN runs in length space (the shortest
span's 896 on every rank, ``probed_hit_tokens``). A worker then reads hit 0 ->
``reap_full`` True (the "unprobed / nothing to wait for" guard), TP0 reads
completed < 896 -> False: the workers remove the shared E2 tail files that
TP0 defers -- one verdict, two values. The verdict now reads the RAW reduced
ENDS, bit-identical on every rank.

The test drives the real reap site (``UnifiedRadixCache.check_prefetch_progress``)
for three ranks: phase 1 collects each rank's proposal vector, phase 2 feeds
the element-wise MIN back as the all_reduce and captures ``full``.
"""

import types

import pytest
import torch

from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.weg2 import tail_adopt

RID = "weg2-h16n-1"
W = 32704  # the workers' span start (rc12z20)
PROBE_MIN = 896  # the length-space probe MIN (the shortest span)


class _Collected(Exception):
    pass


class _Agreed(Exception):
    pass


def _rank(base, completed, monkeypatch, reduce_fn):
    op = types.SimpleNamespace(
        host_indices=torch.zeros(1, dtype=torch.int64),
        probed_hit_tokens=PROBE_MIN,
        pool_storage_result=types.SimpleNamespace(extra_pool_hit_pages={}),
    )
    s = types.SimpleNamespace()
    s.ongoing_prefetch = {RID: (None, None, op.host_indices, op, None, {})}
    s.can_terminate_prefetch = lambda operation, tail_hold=None: True
    s.cache_controller = types.SimpleNamespace(terminate_prefetch=lambda operation: (completed, [b"h"] * 7))
    s._reap_annotation_local = lambda operation, hv: UnifiedRadixCache._reap_annotation_local(s, operation, hv)
    s._reap_annotation_from_packed = UnifiedRadixCache._reap_annotation_from_packed
    s._hp1_end_base_by_rid = {RID: base} if base else {}
    s._anchor_reach_local = lambda c, hv, operation: urc._ANCHOR_ABSTAIN
    s.page_size = 128
    s.tp_world_size = 3
    s._all_reduce_attn_groups = reduce_fn
    return s


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(tail_adopt, "stage", lambda rid, tree_cache=None: None)
    monkeypatch.setattr(tail_adopt, "vote_hold", lambda rid: False)
    monkeypatch.setattr(tail_adopt, "local_vote", lambda rid: 1)
    monkeypatch.setattr(urc, "_weg2_read_no_writer_local", lambda rid, op: 0)


def _drive(ranks, monkeypatch):
    """ranks: [(end_base, completed_local)]. Returns the ``full`` each rank's reap passes to ``agree``."""
    proposals = []

    def collect(packed, op, label=None):
        proposals.append(packed.clone())
        raise _Collected()

    for base, comp in ranks:
        with pytest.raises(_Collected):
            UnifiedRadixCache.check_prefetch_progress(_rank(base, comp, monkeypatch, collect), RID)
    reduced = torch.stack(proposals).min(dim=0).values

    def feed(packed, op, label=None):
        packed.copy_(reduced)

    fulls = []

    def agree(rid, vote, full=None, completed_synced=None, hit_synced=None):
        fulls.append(full)
        raise _Agreed()

    monkeypatch.setattr(tail_adopt, "agree", agree)
    for base, comp in ranks:
        with pytest.raises(_Agreed):
            UnifiedRadixCache.check_prefetch_progress(_rank(base, comp, monkeypatch, feed), RID)
    return fulls


def test_form_a_short_host_read_is_one_verdict_on_every_rank(stubs, monkeypatch):
    """TP0 (base 0) completed 500 of its span, the workers (base 32704) their whole
    896: the reduced END is 500, the reduced hit END 896. One verdict: short."""
    fulls = _drive([(0, 500), (W, PROBE_MIN), (W, PROBE_MIN)], monkeypatch)
    assert fulls == [False, False, False], fulls


def test_form_a_full_read_is_full_on_every_rank(stubs, monkeypatch):
    fulls = _drive([(0, 33600), (W, PROBE_MIN), (W, PROBE_MIN)], monkeypatch)
    assert fulls == [True, True, True], fulls


def test_plain_group_unchanged(stubs, monkeypatch):
    """Off a Form A group (base 0 everywhere) the raw ENDS are the counts: as before."""
    assert _drive([(0, 500), (0, 900), (0, 900)], monkeypatch) == [False, False, False]
    assert _drive([(0, 900), (0, 900), (0, 900)], monkeypatch) == [True, True, True]


def test_the_unpacked_form_was_a_split():
    """What the reap passed before (the mutant): each rank's unpacked counts."""
    e_end, h_end = 500, PROBE_MIN
    per_rank = [tail_adopt.reap_full(urc._hp1_end_unpack(b, e_end), urc._hp1_end_unpack(b, h_end), True)
                for b in (0, W, W)]
    assert per_rank == [False, True, True]
