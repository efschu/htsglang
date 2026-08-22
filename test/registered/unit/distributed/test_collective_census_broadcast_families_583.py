# SPDX-License-Identifier: Apache-2.0
"""The census must be able to see a broadcast/all_to_all desync (#583).

WHY THIS FILE EXISTS
--------------------
The 2026-08-06 05:53:59 production crash was a rank-arrival desync: ranks 0
and 1 entered a group collective that rank 2 never joined, and each burned its
full 60e9-cycle spin deadline. The census was armed and running the whole
time, and it reported nothing -- because all FOUR families it tracked agreed
exactly across the ranks (dcp.all_gather 10176x, dcp.all_reduce 2864x,
tp.all_gather 580x, tp.all_reduce 36824x on both rank 0 and rank 1).

Agreement across every family you count is not health when the failure is in a
family you do not count. The abort named an 8-byte collective served by the
a2a kernel, and neither ``broadcast`` nor ``all_to_all`` was censused at all.

These tests drive the REAL dispatch sites (``GroupCoordinator.broadcast``,
``GroupCoordinator.all_to_all_single``) and the REAL differ
(``CollectiveCensus._diff``), so they fail on a tree where the bump is
missing rather than on a tree where only a string changed. Hermetic: no
torch.distributed init, no group, no CUDA.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from sglang.srt.distributed.collective_census import CollectiveCensus
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    collective_clock_families,
)


def _coordinator(world_size=2):
    """A GroupCoordinator with exactly what the two seams read.

    ``object.__new__`` on purpose: the real ``__init__`` builds process
    groups. What is under test is the dispatch site, not construction.
    """
    g = object.__new__(GroupCoordinator)
    g.world_size = world_size
    g.rank_in_group = 0
    g.ranks = list(range(world_size))
    g.barlink_comm = None
    g.device_group = object()
    g.unique_name = "tp:0"
    # #631 (9c5ffaaab0) introduced ``_census_wire`` as a real __init__-time
    # attribute: ``_CENSUS_ON and self.world_size > 1`` (parallel_state.py:740).
    # ``object.__new__`` skips __init__ so it is never set here; every dispatch
    # site now reads it before touching the census (parallel_state.py:1999 et
    # al.). Every caller of this fixture in this file runs the collective
    # inside the ``_CENSUS_ON = True`` patch block in ``_drive`` (and the one
    # inline case in TestTheDispatchSitesCount does the same), so the formula
    # reduces to ``world_size > 1`` for every instance this fixture builds --
    # bind that reduction, not a bare True, so a fixture ever constructed with
    # world_size=1 still gets the correct (false) wire state.
    g._census_wire = world_size > 1
    # Set positionally and WITHOUT tuple-unpacking the whole return value, so
    # this harness builds against a tree that has not added the two families
    # yet. That is deliberate: the can-fail for this slice has to be "the
    # divergence passes silently", not "the test could not construct its
    # fixture". A ValueError on arity would prove nothing about the census.
    fams = collective_clock_families("tp")
    g._clock_family_all_reduce = fams[0]
    g._clock_family_all_gather = fams[1]
    g._clock_family_all_gatherv = fams[2]
    g._clock_family_reduce_scatterv = fams[3]
    g._clock_family_broadcast = "tp.broadcast"
    g._clock_family_all_to_all = "tp.all_to_all"
    return g


def _declare(census):
    """Declare the families the real ``GroupCoordinator.__init__`` declares.

    Tolerant of a tree without ``declare_families`` for the same reason
    ``_coordinator`` is arity-tolerant.
    """
    declare = getattr(census, "declare_families", None)
    if declare is not None:
        declare(collective_clock_families("tp"))
    return census


def _drive(census, *, broadcasts, all_to_alls, declare=True):
    """Run one rank's collectives through the real dispatch sites.

    The wire call is stubbed out; the census bump is not. Patching the module
    global is what makes the bump land in a per-rank census instead of the
    process-wide singleton, so two "ranks" can be built in one process.

    ``declare`` mirrors what the real ``GroupCoordinator.__init__`` does at
    construction. ``object.__new__`` skips it, so the harness has to do it, or
    the test would be measuring a state no production rank is ever in.
    """
    g = _coordinator()
    if declare:
        _declare(census)
    tensor = torch.zeros(2, dtype=torch.int64)
    with mock.patch(
        "sglang.srt.distributed.parallel_state._CENSUS", census
    ), mock.patch(
        "sglang.srt.distributed.parallel_state._CENSUS_ON", True
    ), mock.patch(
        "torch.distributed.broadcast"
    ), mock.patch(
        "sglang.srt.distributed.parallel_state.reg_all_to_all_single"
    ):
        for _ in range(broadcasts):
            g.broadcast(tensor, src=0)
        for _ in range(all_to_alls):
            g.all_to_all_single(tensor, tensor)
    return census


class TestTheDesyncIsVisible(unittest.TestCase):
    """The case the production census was blind to."""

    def test_a_broadcast_desync_is_named(self):
        rank0 = _drive(CollectiveCensus(), broadcasts=129, all_to_alls=4)
        rank1 = _drive(CollectiveCensus(), broadcasts=128, all_to_alls=4)

        found = CollectiveCensus._diff([rank0.snapshot(), rank1.snapshot()])

        families = {d.family for d in found}
        self.assertIn(
            "tp.broadcast",
            families,
            "a broadcast desync must be named; before #583 the family was "
            "not counted at all and this diff came back empty",
        )
        (div,) = [d for d in found if d.family == "tp.broadcast"]
        self.assertEqual(div.counts, (129, 128))
        self.assertEqual(div.leader, 129)
        self.assertEqual(div.behind, [(1, 1)])
        self.assertIn("rank 1 behind by 1", div.describe())

    def test_an_all_to_all_desync_is_named(self):
        rank0 = _drive(CollectiveCensus(), broadcasts=8, all_to_alls=17)
        rank1 = _drive(CollectiveCensus(), broadcasts=8, all_to_alls=16)

        found = CollectiveCensus._diff([rank0.snapshot(), rank1.snapshot()])

        self.assertEqual([d.family for d in found], ["tp.all_to_all"])
        self.assertEqual(found[0].behind, [(1, 1)])

    def test_agreement_on_the_new_families_stays_silent(self):
        """The instrument must not cry wolf: equal counts are not a finding."""
        rank0 = _drive(CollectiveCensus(), broadcasts=64, all_to_alls=9)
        rank1 = _drive(CollectiveCensus(), broadcasts=64, all_to_alls=9)

        self.assertEqual(
            CollectiveCensus._diff([rank0.snapshot(), rank1.snapshot()]), []
        )


class TestTheDispatchSitesCount(unittest.TestCase):
    """Each new site must bump its OWN family, at the real call."""

    def test_broadcast_bumps_broadcast(self):
        c = _drive(CollectiveCensus(), broadcasts=3, all_to_alls=0)
        self.assertEqual(c.snapshot()["tp.broadcast"], 3)
        self.assertEqual(c.snapshot()["tp.all_to_all"], 0)

    def test_all_to_all_bumps_all_to_all(self):
        c = _drive(CollectiveCensus(), broadcasts=0, all_to_alls=5)
        self.assertEqual(c.snapshot()["tp.all_to_all"], 5)
        self.assertEqual(c.snapshot()["tp.broadcast"], 0)

    def test_the_uneven_form_shares_the_family(self):
        """A rank that skips an uneven a2a has skipped an a2a."""
        c = CollectiveCensus()
        g = _coordinator()
        tensor = torch.zeros(2, dtype=torch.int64)
        with mock.patch(
            "sglang.srt.distributed.parallel_state._CENSUS", c
        ), mock.patch(
            "sglang.srt.distributed.parallel_state._CENSUS_ON", True
        ), mock.patch(
            "torch.distributed.all_to_all_single"
        ):
            g.all_to_all_single_v(tensor, tensor)
        self.assertEqual(c.snapshot().get("tp.all_to_all"), 1)


class TestPayloadWidthIsRankUniform(unittest.TestCase):
    """#610: pack size from replicated config, never from rank-local state."""

    def test_declared_families_make_the_key_set_identical(self):
        """Two ranks that executed DIFFERENT collectives still pack the same
        keys -- otherwise the payload width would diverge for the same reason
        the counts do."""
        busy = _drive(CollectiveCensus(), broadcasts=12, all_to_alls=3)
        idle = _drive(CollectiveCensus(), broadcasts=0, all_to_alls=0)

        self.assertEqual(set(busy.snapshot()), set(idle.snapshot()))
        self.assertEqual(idle.snapshot()["tp.broadcast"], 0)
        self.assertEqual(busy.snapshot()["tp.broadcast"], 12)

    def test_without_declaration_the_key_sets_diverge(self):
        """The falsifier: declaration is what fixes the width, not luck.

        Without it the key set is a function of what a rank has executed --
        exactly the rank-local state the census exists to catch diverging.
        """
        busy = _drive(CollectiveCensus(), broadcasts=12, all_to_alls=3, declare=False)
        idle = _drive(CollectiveCensus(), broadcasts=0, all_to_alls=0, declare=False)

        self.assertNotEqual(set(busy.snapshot()), set(idle.snapshot()))
        self.assertEqual(idle.snapshot(), {})

    def test_declaration_never_overwrites_a_live_count(self):
        c = _drive(CollectiveCensus(), broadcasts=7, all_to_alls=0, declare=False)
        c.declare_families(collective_clock_families("tp"))
        self.assertEqual(c.snapshot()["tp.broadcast"], 7)

    def test_declaration_is_idempotent(self):
        c = CollectiveCensus()
        c.declare_families(collective_clock_families("tp"))
        first = c.snapshot()
        c.declare_families(collective_clock_families("tp"))
        self.assertEqual(c.snapshot(), first)

    def test_a_never_fired_declared_family_is_visible_as_zero(self):
        """"never counted" and "not instrumented" must stop looking alike."""
        c = CollectiveCensus()
        c.declare_families(collective_clock_families("tp"))
        self.assertIn("tp.all_to_all", c.snapshot())
        self.assertEqual(c.snapshot()["tp.all_to_all"], 0)


class TestTheFamilyNamesExist(unittest.TestCase):
    def test_both_new_families_are_named_per_group(self):
        self.assertEqual(collective_clock_families("dcp")[4], "dcp.broadcast")
        self.assertEqual(collective_clock_families("dcp")[5], "dcp.all_to_all")


if __name__ == "__main__":
    unittest.main()
