# SPDX-License-Identifier: Apache-2.0
"""#287 NEED0 (30.09., NF y4k dff1a7fed4): a 25-token request parked over 12 D
phases because its store read was taken for a budget refusal.

DER BEFUND (y4k, D- und Front-Log): ``weg2-0-4`` (SHORT, 25 Token, seat 1/6)
wurde 0,65 s nach der Zulassung geparkt (``RETAINED retained=0 of 26``); jeder
Wake las mit ``#915 PREFETCH REFUSED reason=vote_negative need=0
available=415040`` (533x je Rang), ``#1471b WAKE-READ BUDGET-REFUSED``, die
Settle-Schranke (20 s) wurde bei jedem Wake neu gestempelt und keine D-Phase
(10-18 s) liess sie ablaufen: ``WEG2-SERVED ... wall=525.55s`` fuer 2 Token,
ein D-Sitz den ganzen Bench lang belegt, bs6 lief als bs5.

(a) ``vote_negative`` ist nur mit need>0 und need>available eine
    Budget-Verweigerung (settle_writer.budget_refused + die Terme des Baums).
(b) Ein budget-verweigerter Read behaelt seine Settle-Uhr ueber Wakes.
(c) front.d_park_stuck in state.json: Parks in Folge ohne Ausgabe.
"""

from __future__ import annotations

import inspect
import types

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-test-cpu")

SETTLE_S = 20.0
D_PHASE_S = 14.0   # y4k: 10-18 s D phases
P_PHASE_S = 10.0


def _sw():
    from sglang.srt.weg2 import settle_writer as sw

    return sw


def _tree_with_refusal(reason, rid, need, available, occupied=0, limit=373536):
    """A radix tree stub that logged one #915 refusal through the REAL
    ``UnifiedRadixCache._log_prefetch_refused``."""
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    tree = types.SimpleNamespace()
    tree._prefetch_line_terms = lambda n: {
        "need": int(n), "available": int(available), "threshold": 256,
        "occupied": int(occupied), "limit": int(limit), "pool_id": 1, "epoch": 1,
        "phase": "pp", "generation": 0}
    UnifiedRadixCache._log_prefetch_refused(tree, reason, rid, need)
    return tree


# ---- (a) ----------------------------------------------------------------------------

def test_vote_negative_with_need_zero_is_an_answer_not_a_budget_refusal():
    sw = _sw()
    tree = _tree_with_refusal("vote_negative", "weg2-0-4", 0, 415040)
    # room = min(available 415040, limit 373536 - occupied 0)
    assert sw.refusal_terms(tree, "weg2-0-4") == ("vote_negative", 0, 373536)
    req = types.SimpleNamespace(rid="weg2-0-4")
    assert sw.note_read_verdict(req, "declined:vote_negative", 1.0, tree=tree) is False
    assert not sw.budget_pending(req)
    assert sw.gate_action("decide", req) == "decide"     # the settle may decide it now


def test_vote_negative_over_a_full_budget_stays_a_budget_refusal():
    sw = _sw()
    # y3u (0930_002717 D, every rank): need=77824 available=415040
    # occupied=392320 limit=373536 -- the host pool had room, the budget none
    tree = _tree_with_refusal("vote_negative", "weg2-30-52", 77824, 415040, 392320, 373536)
    assert sw.refusal_terms(tree, "weg2-30-52") == ("vote_negative", 77824, -18784)
    req = types.SimpleNamespace(rid="weg2-30-52")
    assert sw.note_read_verdict(req, "declined:vote_negative", 1.0, tree=tree) is True
    assert sw.gate_action("decide", req) == "poll"
    # without the rank's terms a vote_negative proves no shortage
    assert sw.budget_refused("declined:vote_negative") is False
    # the pure budget terms stay budget refusals
    assert sw.budget_refused("declined:rate_limited") is True
    assert sw.budget_refused("declined:too_short") is False


# ---- (a)+(b): the y4k pattern over twelve short D phases -----------------------------

def _run_wakes(sw, req, tree, wakes=12):
    """The #1471 settle over ``wakes`` D phases of D_PHASE_S: each wake reads
    (refused as y4k), re-stamps the settle clock the scheduler's way, and the
    request is released when the settle may decide it or its bound lapsed."""
    t = 0.0
    for k in range(1, wakes + 1):
        t += P_PHASE_S                                   # the P phase
        sw.note_read_verdict(req, "declined:vote_negative", t, tree=tree)
        req._1471_since = sw.settle_since_for_wake(req, t)
        sw.reset_for_wake(req)
        end = t + D_PHASE_S
        decided = sw.gate_action("decide", req) == "decide"
        lapsed = end - req._1471_since >= SETTLE_S
        if decided or lapsed:
            return k, ("decide" if decided else "lapsed")
        t = end
    return None, "parked"


def test_y4k_weg2_0_4_is_released_at_the_first_wake():
    sw = _sw()
    tree = _tree_with_refusal("vote_negative", "weg2-0-4", 0, 415040)
    req = types.SimpleNamespace(rid="weg2-0-4")
    assert _run_wakes(sw, req, tree) == (1, "decide")


def test_a_real_budget_refusal_lapses_over_wakes_instead_of_for_ever():
    sw = _sw()
    tree = _tree_with_refusal("vote_negative", "weg2-30-52", 77824, 415040, 392320, 373536)
    req = types.SimpleNamespace(rid="weg2-30-52")
    # first refused wake at t=10; the bound runs over the P phase: by the end
    # of the second D phase (t=48) 38 s have passed
    assert _run_wakes(sw, req, tree) == (2, "lapsed")


def test_an_unrefused_request_starts_the_bound_at_each_wake_as_before():
    sw = _sw()
    req = types.SimpleNamespace(rid="x", _1471_since=3.0)
    assert sw.settle_since_for_wake(req, 50.0) == 50.0


# ---- the wiring -----------------------------------------------------------------------

def test_the_scheduler_and_the_wake_read_pass_the_tree_and_keep_the_clock():
    from sglang.srt.managers import scheduler as sched_mod
    from sglang.srt.weg2 import park_l3

    src = inspect.getsource(sched_mod)
    assert src.count("_sw.note_read_verdict(req, verdict, now, tree=self.tree_cache)") == 2
    assert "_r._1471_since = _sw_nw.settle_since_for_wake(_r, _now)" in src
    psrc = inspect.getsource(park_l3._issue)
    assert 'tree=getattr(sched, "tree_cache", None)' in psrc


# ---- (c) the streak as IPC ------------------------------------------------------------

def test_park_streak_of_the_y4k_request_and_its_end():
    from sglang.srt.weg2.park_stuck import ParkStuck

    ps = ParkStuck()
    for _ in range(12):
        ps.note_park(["weg2-0-4"])       # parked again, no output in between
    ps.note_park(["weg2-2-6"])
    ps.note_output("weg2-2-6")
    ps.note_park(["weg2-2-6"])           # progress between its parks: streak restarts
    b = ps.block(3)
    assert b["stuck"] == 1 and b["rids"] == {"weg2-0-4": 12} and b["max_streak"] == 12
    ps.done("weg2-0-4")
    b = ps.block(3)
    assert b["stuck"] == 0 and b["max_streak"] == 1 and b["max_streak_boot"] == 12


def test_front_publishes_the_streak_in_its_state_fields():
    from sglang.srt.environ import envs
    from sglang.srt.weg2 import front as F

    assert envs.SGLANG_WEG2_PARK_STUCK_PHASES.get() == 3
    src = inspect.getsource(F.Front._ipc_front_fields)
    assert 'out["d_park_stuck"] = self._park_stuck().block(envs.SGLANG_WEG2_PARK_STUCK_PHASES.get())' in src
    fsrc = inspect.getsource(F)
    i = fsrc.index("self._d_parked[r] = t_park")
    assert "_ps().note_park(known + late)" in fsrc[i:i + 300]
    assert "self._park_stuck().note_output(rid)" in fsrc
    class _F:
        pass

    front = _F()
    ps = F.Front._park_stuck(front)
    ps.note_park(["a"]); ps.note_park(["a"]); ps.note_park(["a"])
    assert F.Front._park_stuck(front).block(3)["rids"] == {"a": 3}
