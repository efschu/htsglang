# SPDX-License-Identifier: Apache-2.0
"""NF-RANG-AUDIT 1014, Fix 3 (B2): the group MIN helpers stop by name when the
collective fails, instead of acting on this rank's local flags.

``_pdflip_group_min_flags`` / ``_pdflip_group_min_ints`` are the ONE verdict of every
park / settle / displace / capacity decision on group D. Their error arm answered an
exception from ``all_reduce`` with the LOCAL flags (INFO "local flags used") -- the
forbidden silent alignment: this rank acts on its view, its peers on theirs, nothing
names it.

RANK-EQUALITY MUTANT (``test_rank_equality_mutant_local_flags_after_a_failed_collective``):
``FLLIPER_PDFLIP_GROUP_MIN_STRICT=0`` = the arm of before: rank A (collective raised) keeps
its local [1, 0], rank B (collective fine) holds the MIN [0, 0] -> two release sets. On:
rank A raises ``PdFlipGroupVerdictFailed`` by name.
"""
from __future__ import annotations

import types

import pytest
import torch

from flliper.srt.managers import scheduler as sched_mod
from flliper.srt.pdflip import group_min_strict as GMS


def _rank(tp_size=3, group="g"):
    h = types.SimpleNamespace(ps=types.SimpleNamespace(tp_size=tp_size), tp_cpu_group=group)
    h._pdflip_group_min_flags = types.MethodType(sched_mod.Scheduler._pdflip_group_min_flags, h)
    h._pdflip_group_min_ints = types.MethodType(sched_mod.Scheduler._pdflip_group_min_ints, h)
    return h


@pytest.fixture
def dist(monkeypatch):
    monkeypatch.delenv(GMS.ENV, raising=False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    return monkeypatch


def _raising(*a, **k):
    raise RuntimeError("[gloo] op.preamble.length <= op.nbytes. 16 vs 4")


def _peer_min(peer):
    """an all_reduce MIN against one peer's vector"""
    def _ar(t, op=None, group=None):
        t.copy_(torch.minimum(t, torch.tensor(peer, dtype=t.dtype)))
    return _ar


def test_a_failed_collective_is_a_named_stop(dist):
    dist.setattr(torch.distributed, "all_reduce", _raising)
    r = _rank()
    with pytest.raises(GMS.PdFlipGroupVerdictFailed, match="GROUP-VERDICT FAILED flags n=2"):
        r._pdflip_group_min_flags([1, 0])
    with pytest.raises(GMS.PdFlipGroupVerdictFailed, match="GROUP-VERDICT FAILED ints n=1"):
        r._pdflip_group_min_ints([3])


def test_rank_equality_mutant_local_flags_after_a_failed_collective(dist):
    """Rank A's collective raises, rank B's reduces fine: with the old arm A releases
    [1, 0] while B holds [0, 0]. With the strict arm A stops by name."""
    a, b = _rank(), _rank()
    dist.setenv(GMS.ENV, "0")
    dist.setattr(torch.distributed, "all_reduce", _raising)
    got_a = a._pdflip_group_min_flags([1, 0])
    dist.setattr(torch.distributed, "all_reduce", _peer_min([0, 0]))
    got_b = b._pdflip_group_min_flags([1, 0])
    assert got_a == [1, 0] and got_b == [0, 0]  # the mutant: two release sets for one settle
    dist.delenv(GMS.ENV, raising=False)
    dist.setattr(torch.distributed, "all_reduce", _raising)
    with pytest.raises(GMS.PdFlipGroupVerdictFailed):
        a._pdflip_group_min_flags([1, 0])


def test_a_working_collective_and_no_group_are_unchanged(dist):
    dist.setattr(torch.distributed, "all_reduce", _peer_min([0, 1, 1]))
    r = _rank()
    assert r._pdflip_group_min_flags([1, 1, 0]) == [0, 1, 0]
    assert r._pdflip_group_min_ints([5, 2, 9]) == [0, 1, 1]
    assert r._pdflip_group_min_flags([]) == []
    dist.setattr(torch.distributed, "all_reduce", _raising)
    solo = _rank(tp_size=1, group=None)
    assert solo._pdflip_group_min_flags([1, 0]) == [1, 0]  # nobody to disagree with
    assert solo._pdflip_group_min_ints([4]) == [4]


def test_default_on():
    assert GMS.enabled({}) is True
    assert GMS.enabled({GMS.ENV: "0"}) is False
