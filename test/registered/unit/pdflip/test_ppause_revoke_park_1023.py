"""PPAUSE-REVOKE (23.10.2026): the S82 pass-top park polled a REVOKED store read to its timeout.

Metal (27B AutoRound rc35i4ar, a45112428e, P = PP2 over 2x3080, P log
boot_weg2_dkr27bint4arbar1int4ar10231400_a45112428e_1023_140032.P.log, PP0):
  14:10:36  'PDFLIP-CLAIM-SHORT rid=pdflip-15-35 pp=0 asked=2294 kv=0 claimed=0 ... by=kv_prefix tokens=0'
            (20224 of its 22520 tokens resident on PP0's device; the 2294-token tail is in no store -> revoked)
  14:10:36  '#969N ADMIT slot=1 fwd_ct=68 bs=1 extend=2139 rids=[pdflip-14-34]'          (the older set)
  14:10:41  'S82 COMMIT-PARK n=3 outcome=read rid=pdflip-15-35 waited_ms=3994.0 ... reads=2 acks=0'
  14:10:41  '#1157 PREFETCH REAPED req=pdflip-15-35 probed=True requested_pages=2294 hit_pages=0 completed=0
             elapsed=4.24s budget=4.24s (completed_local=0 policy=timeout)'
  14:10:41  PP1 'PP-CHAIN-RECV #1460 CHAIN-RECV blocked 4278 ms' (the park withheld PP0's chain)
  14:10:44  '#PGAP pp_rank=0 fwd=70 tokens=2296 gpu_gap_ms=4100.0 ... host[plan=11 ... output_commit=95
             d2h_wait=981 process=25]'                                   -> monitor P-PAUSE 4.10 s, queue 2
Same class 14:08:05-15: 'S82 COMMIT-PARK n=1 outcome=read rid=pdflip-11-20 waited_ms=9672.3', REAPED 9.86 s.

A revoked operation is never terminated: its record leaves ``ongoing_prefetch`` only when the scheduler thread
drains the revoke queue once per round (#1400), and ``can_terminate_prefetch`` never calls an empty
``hash_value`` complete. The park held that thread. Pinned here (RED on a45112428e):
  1. the metal case: a held rid whose read was revoked before the pass top is not an open read -- no park;
  2. a revoke that lands DURING the park ends it at once ('read'), not at the bound;
  3. a read that is still running is untouched (the drain retires revoked records only) and still parks;
  4. a tree without the drain keeps the unchanged wait (no exception);
  5. P-MINIFWD-HOLD waits through the same draining predicate;
  6. #PGAP names the park: 's82_park' / 'pass_top' host terms, appended after the old keys.
"""

import inspect
import os
import sys
import time as _time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.managers import pdflip_p_overlap as pov  # noqa: E402
from flliper.srt.managers import pdflip_store_told as told  # noqa: E402
from flliper.srt.managers import scheduler_pp_mixin as ppm  # noqa: E402
from flliper.srt.pdflip import pp0_commit_park as cp  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402
from test_s82_commit_park_1019 import pp0  # noqa: E402,F401  (fixture: the PF/paced PP0 stand-in)

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

RID = "pdflip-15-35"
BOUND_S = 0.4


class _Tree:
    """UnifiedRadixCache stand-in of PP0's tree for one held rid: the record of its read sits in
    ``ongoing_prefetch``; a REVOKED read is never terminable (empty hash_value, timeout far away) until the
    revoke queue is drained -- ``drain_prefetch_revokes_uncoordinated`` pops it, as ``_drain_revoke`` does."""

    def __init__(self, revoked_at_call=None, terminable_at_call=None, with_drain=True):
        op = SimpleNamespace(host_indices=[1], hash_value=[])
        self.ongoing_prefetch = {RID: SimpleNamespace(operation=op)}
        self.revoke_queue = []
        self.drains = 0
        self.can_calls = 0
        self._revoked_at = revoked_at_call
        self._terminable_at = terminable_at_call
        if with_drain:
            self.drain_prefetch_revokes_uncoordinated = self._drain

    def _drain(self):
        self.drains += 1
        if self._revoked_at is not None and self.drains >= self._revoked_at:
            self.revoke_queue.append(RID)
        while self.revoke_queue:
            self.ongoing_prefetch.pop(self.revoke_queue.pop(0), None)
        return True

    def can_terminate_prefetch(self, op, tail_hold=0):
        self.can_calls += 1
        return self._terminable_at is not None and self.can_calls >= self._terminable_at

    def check_prefetch_progress(self, rid):
        return rid not in self.ongoing_prefetch


def _metal_pass_top(pp0, monkeypatch, tree):
    """PP0 at the top of the pass after launching pdflip-14-34: pdflip-15-35 held (#1400) with its store read
    registered, not yet paced (no told on the wire, no ack waiter), the older set's output owed."""
    req = pp0.waiting_queue[0]
    req.rid = RID
    req.prefetch_deferred = None
    pp0._pdflip_told_pacing = {}
    pp0._pdflip_fb_open = {}
    pp0._pdflip_store_held = {RID: req}
    pp0.tree_cache = tree
    monkeypatch.setattr(ppm, "_pp_proxy_readiness_budget_s", lambda: BOUND_S)
    monkeypatch.setattr(told._twin, "is_deferred", lambda s, rid: False)
    if hasattr(told, "_REVOKE_WAIT"):        # the base has no counter: the behaviour asserts below still run
        told._REVOKE_WAIT.update({"n": 0, "entry": 0, "poll": 0})
    return req


def _drained(site):
    return getattr(told, "_REVOKE_WAIT", {}).get(site, 0)


def test_metal_1410_a_read_revoked_before_the_top_is_no_waiter(pp0, monkeypatch):
    """RED on a45112428e: the park takes the revoked read as open and polls it to the bound
    ('outcome=read ... waited_ms=3994.0' on the metal, where the bound was the read's own 4.24 s timeout)."""
    tree = _Tree()
    tree.revoke_queue.append(RID)            # the probe answered 0 at 14:10:36, the revoke is queued
    _metal_pass_top(pp0, monkeypatch, tree)
    t0 = _time.monotonic()
    told._commit_park(pp0, [])
    waited = _time.monotonic() - t0
    assert cp.stats()["n"] == 0, "the pass parked on a read that was revoked before its top"
    assert waited < BOUND_S / 2
    assert RID not in tree.ongoing_prefetch and _drained("entry") == 1
    assert told._read_terminable(tree, RID)  # the publish loop reaches the rid in THIS pass


def test_a_revoke_landing_in_the_park_ends_it_at_once(pp0, monkeypatch):
    """The probe answers while PP0 already parks: the next poll drains it -- 'read', not 'bound'."""
    tree = _Tree(revoked_at_call=3)          # entry drain: nothing yet; the 2nd poll finds the revoke
    _metal_pass_top(pp0, monkeypatch, tree)
    t0 = _time.monotonic()
    told._commit_park(pp0, [])
    waited = _time.monotonic() - t0
    st = cp.stats()
    assert st["n"] == 1 and st.get("read") == 1 and st.get("bound", 0) == 0
    assert waited < BOUND_S / 2
    assert _drained("poll") == 1


def test_a_running_read_still_parks_and_is_not_dropped(pp0, monkeypatch):
    """Danger direction: the drain must never retire a read that is still loading -- the park keeps waiting
    for it and ends when it lands."""
    tree = _Tree(terminable_at_call=4)       # nothing revoked; the read lands at the 4th probe
    _metal_pass_top(pp0, monkeypatch, tree)
    told._commit_park(pp0, [])
    st = cp.stats()
    assert st["n"] == 1 and st.get("read") == 1
    assert RID in tree.ongoing_prefetch and _drained("n") == 0
    assert tree.drains >= 2                  # entry + polls asked, nothing was there to drain


def test_a_tree_without_the_drain_keeps_the_unchanged_wait(pp0, monkeypatch):
    tree = _Tree(with_drain=False, terminable_at_call=2)
    _metal_pass_top(pp0, monkeypatch, tree)
    told._commit_park(pp0, [])
    assert cp.stats()["n"] == 1 and cp.stats().get("read") == 1
    drain = getattr(told, "_drain_revokes_for", None)
    assert drain is not None and drain(pp0, SimpleNamespace(), [RID], "entry") == []   # no ongoing map: no-op


def test_minifwd_hold_waits_through_the_draining_predicate():
    src = inspect.getsource(told._minifwd_hold)
    assert '_drain_revokes_for(scheduler, tree, [r for r in queued if r in held], "entry")' in src
    assert "_read_terminable_draining(scheduler, tree, r)" in src
    park = inspect.getsource(told._commit_park)
    assert park.index('"entry")') < park.index("open_reads = [")
    assert "read_done=lambda r: _read_terminable_draining(scheduler, tree, r)" in park


def test_pgap_names_the_park(monkeypatch):
    assert pov._ORDER[-2:] == ("pass_top", "s82_park")
    assert pov._ORDER[:8] == ("plan", "proxy_recv", "launch", "fi_plan", "deferred_publish",
                              "output_commit", "d2h_wait", "process")   # old keys unchanged, in order
    host = {"plan": 11.0, "launch": 224.0, "output_commit": 95.0, "d2h_wait": 981.0, "process": 25.0,
            "pass_top": 4064.0, "s82_park": 3994.0}
    line = pov.format_line(0, 70, 2296, 4100.0, 1463.3, host)
    assert "process=25 pass_top=4064 s82_park=3994]" in line
    # wiring: the pass top is spanned at its call site, the waits inside pp0_publish
    pub = inspect.getsource(told.pp0_publish)
    assert pub.index('_pov.span("s82_park")') < pub.index("_minifwd_hold(scheduler, recv_reqs)") \
        < pub.index("_commit_park(scheduler, recv_reqs)")
    mix = inspect.getsource(ppm.SchedulerPPMixin._pp_forward_and_process_input_requests)
    assert mix.index('_pov.span("pass_top")') < mix.index("pdflip_store_told.pp0_publish(self, recv_reqs)")


def test_the_park_span_is_measured(pp0, monkeypatch):
    monkeypatch.setenv(pov.P_HOSTGAP_ENV, "1")
    tree = _Tree(terminable_at_call=3)
    _metal_pass_top(pp0, monkeypatch, tree)
    pov.take_spans()
    told.pp0_publish(pp0, [])
    spans = pov.take_spans()
    assert spans.get("s82_park", 0.0) > 0.0
