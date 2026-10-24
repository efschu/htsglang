"""REVIEW REPRICE-REST-1024 (#224, SOLL R2 R7): review pins on the settle hold (settle_writer.held_read_holds_lapse).

Two danger directions of the hold, on the author's 26-232 harness (test_pdflip_reprice_rest_1024):

* UNDER-pricing (D-OOM / D prefill over X): the awaited re-read lands SHORT. The hold must not let the release price
  more than really landed -- the witness stays the record's absolute materialized depth (FIRST + what landed), the
  remainder N - (FIRST + SHORT) = 6844 > front X goes through RESUME-VIA-P, never as a D-RESUME-EXTEND of a "220".
  Mutant (killed by this pin): the #1324 arm writes ``deliverable`` instead of ``materialized`` as the witness.

* HANGING (a request held without end): the hold has no clock of its own; it ends when the re-read terminates (the
  prefetch reaper under policy=timeout: up to issue + 2 x (2 s + pages x 0.98 ms), under wait_complete: never by time).
  Review MUSS-1 asked for a NAMED upper bound (now settle_writer.hold_capped); the former xfail(strict) pin below is
  a plain test now (green on the capped head).
"""
from __future__ import annotations

import logging
import os
import sys
import time


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_pdflip_reprice_rest_1024 as base  # noqa: E402
from test_pdflip_reprice_rest_1024 import _env  # noqa: E402,F401  (autouse: needs-p dir, arena dir, group D)

from flliper.srt.mem_cache.hicache_storage import PrefetchOutcome  # noqa: E402
from flliper.srt.pdflip import d_norecompute as dn  # noqa: E402
from flliper.srt.pdflip import resume_via_p as rvp  # noqa: E402
from flliper.srt.pdflip import short_read as sr  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

SHORT = 20000   # the awaited re-read lands 20000 of its 26624 pages (store held them all again)


def test_review_held_reread_landing_short_is_never_underpriced(caplog):
    s, r, calls = base._settle_sched()
    base._to_the_lapse(s, r)
    assert s.pdflip_post_wake_settle == [r], "held past the bound while the re-read runs"
    # the awaited re-read terminates short (reap: requested = hit = 26624, completed 20000 -> held again)
    s.tree_cache.ongoing_prefetch.pop(r.rid, None)
    rec = PrefetchOutcome(SHORT, hit_tokens=base.REST, probed=True, matched=0, deliverable=base.REST, synced=SHORT,
                          cut_cause="late_complete")
    rec.span_start = base.FIRST
    s.tree_cache.prefetch_loaded_tokens_by_reqid[r.rid] = rec
    sr.note_reap(r.rid, base.REST, base.REST, SHORT, 1)
    with caplog.at_level(logging.INFO):
        base._tick(s, r)
    true_rest = base.N - (base.FIRST + SHORT)          # 6844 > front X 4096
    assert r not in s.pdflip_post_wake_settle, "the hold is spent once the awaited read answered"
    assert not any("D-RESUME-EXTEND rid=" + r.rid in m for m in caplog.messages), \
        "UNDER-pricing: D would extend a remainder over its front X"
    recs = rvp.take_requests(rvp.needs_p_dir())
    # exactly what is missing: the record's absolute materialized depth (FIRST + SHORT), not its deliverable
    # (a witness above what landed = the D-OOM direction; the divert clears the witness afterwards, keep_on_d)
    assert len(recs) == 1 and recs[0]["d_extent"] == true_rest, recs
    assert calls == [r.rid]


# REVIEW #224 MUSS-1 met (settle_writer.hold_capped, FLLIPER_PDFLIP_SETTLE_HELD_READ_MAX_S): the strict xfail marker is gone
def test_review_hold_has_a_named_upper_bound_when_the_reread_never_terminates():
    s, r, _calls = base._settle_sched()
    base._to_the_lapse(s, r)
    assert s.pdflip_post_wake_settle == [r]
    # ten minutes after the wake, the re-read still has not terminated (never popped from ongoing_prefetch)
    r._1471_since = time.monotonic() - 600.0
    base._tick(s, r)
    assert r not in s.pdflip_post_wake_settle, "a bound must end the hold (release or RESUME-VIA-P), by name"
    assert dn.d_would_compute(r) >= 0
