"""MEMRANK-1015 leak fix 2: ``PPAdmissionCongruenceGuard._1059_observed`` was never retired.

One entry (rid -> smallest coverage promise) per request for the life of the
scheduler process; no pop site existed anywhere (checked with a table scan of
python/sglang/srt/{weg2,managers,mem_cache}). Bounded now like its sibling
``_offer_prefix_key``: oldest inserted rid out first, ``OBSERVED_SLOTS`` entries.
The table is small per entry (~250 B), so this is hygiene, not the cause of the
anon growth (see deskq/done/nf-memrank-1015.md).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.pp_admission_congruence import PPAdmissionCongruenceGuard as G  # noqa: E402


def test_table_is_bounded_and_keeps_the_newest():
    g = G()
    n = G.OBSERVED_SLOTS + 500
    for i in range(n):
        g.note_observed_coverage("r%d" % i, 100 + i)
    assert len(g._1059_observed) == G.OBSERVED_SLOTS
    # oldest out, newest kept, values intact
    assert g.uniform_prefix_for("r0") is None
    assert g.uniform_prefix_for("r499") is None
    assert g.uniform_prefix_for("r500") == 600
    assert g.uniform_prefix_for("r%d" % (n - 1)) == 100 + n - 1


def test_min_semantics_untouched_and_none_skipped():
    g = G()
    g.note_observed_coverage("a", 50)
    g.note_observed_coverage("a", 80)   # larger: the min stays
    g.note_observed_coverage("a", 30)   # smaller: the min follows
    g.note_observed_coverage("a", None)  # None is no fact
    g.note_observed_coverage("b", None)
    assert g.uniform_prefix_for("a") == 30
    assert g.uniform_prefix_for("b") is None
    assert "b" not in g._1059_observed


def test_recent_live_rid_survives_a_burst_below_the_cap():
    g = G()
    g.note_observed_coverage("live", 7)
    for i in range(G.OBSERVED_SLOTS - 1):
        g.note_observed_coverage("x%d" % i, 1)
    assert g.uniform_prefix_for("live") == 7
