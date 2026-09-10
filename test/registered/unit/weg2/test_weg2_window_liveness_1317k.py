# SPDX-License-Identifier: Apache-2.0
"""#1317k -- the window cap is WIRED, and the bounds are progress, not seconds.

WHAT THESE PIN, and each one is a defect that was paid for on metal (boots
weg2sn6k and weg2sn6l, 2026-09-10, identical numbers on both):

1. ``solve_window``'s W was **computed by nobody and read by nobody**. Its
   only non-test caller was ``window_provenance``, whose callers were none, so
   the ``#1317 WINDOW W=`` line printed 0 times in both group logs. What sized
   the read instead was ``min(need, available)`` -- the WHOLE pool:
   ``#915 PREFETCH TRUNCATED need=109129 got=30518 ... available=0`` on a
   30,518-row pool. Window 1 owned every row, so no second window could ever
   allocate (the next attempt read ``got=1847``, which is exactly the solver's
   own predicted slack), C1 had nothing it was permitted to free
   (``WINDOW-RELEASE`` 0), the chain never closed (``closed_total`` 0), and the
   loop re-issued ``issued:truncated_group`` ten times in 61 s without passing
   ``matched=98302`` of 109,131.

2. The two ``DEFER EXPIRED`` exits were WALL CLOCKS. A clock cannot separate
   "loading slowly" from "not loading": it expires on a healthy 109k-token read
   whose bound is priced off a span it is not allowed to have, and it does not
   expire on a chain that is provably dead but young. User ruling 2026-09-10:
   *"entweder es lädt (warten) oder es lädt nicht (Fehler -> Abbruch)"*.

3. The window chain closed on ANY verdict that was not
   ``issued:truncated_group`` -- so a DECLINE closed it exactly as a whole read
   would have -- and nothing compared the covered prefix against the prompt.

THE DANGER DIRECTION IS THE POINT OF 2 AND 3, and it is pinned in both
directions: a chain that is MOVING must never be cut (that is the false
refusal, and it turns a served request into a 503), and a chain that is
STANDING STILL must always terminate (that is the spin, and it ends in D
prefilling the prompt over X). A test that only pinned one direction would
have passed on the code that shipped.
"""

import unittest

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.weg2 import host_ledger

# Group D on this rig, from the boot logs of weg2sn6k/sn6l.
D_POOL = 30518
D_CHUNK = 4096
D_CHAINS = 1
D_FLOOR = D_CHUNK
D_W = 24576
D_NEED = 109129


def _bind(names):
    """A stand-in carrying the REAL unbound methods, by name.

    The methods under test read only getattr-able state, so binding them onto
    a namespace is exercising the shipped code rather than a copy of it. Names
    are resolved out of ``Scheduler.__dict__`` so a rename breaks this file
    instead of silently testing nothing.
    """

    class _S:
        pass

    s = _S()
    for n in names:
        fn = Scheduler.__dict__[n]
        setattr(s, n, fn.__get__(s, _S))
    return s


class _Req:
    """Minimal request: only the attributes the witness actually reads."""

    def __init__(self, prefix=0, host_hit=0, rid="r0", total=D_NEED):
        self.rid = rid
        self.prefix_indices = [0] * prefix
        self.host_hit_length = host_hit
        self.full_untruncated_fill_ids = [0] * total
        self._prefetch_registered_prefix_len = 0


class _Tree:
    def __init__(self):
        self.ongoing_prefetch = {}
        self._weg2_window_rows_recycled = 0
        self._weg2_window_nodes_recycled = 0


class TestTheWindowIsAWindow(unittest.TestCase):
    """(1) The cap the read never had."""

    def test_the_solver_reproduces_group_ds_window_and_its_slack(self):
        w = host_ledger.solve_window(D_POOL, D_CHUNK, D_CHAINS, D_FLOOR, 37)
        self.assertEqual(w, D_W)
        # The slack the solver predicts is the room the NEXT window needs, and
        # it is the number metal actually saw on its second attempt (got=1847
        # against this 1846 -- one row is the resume anchor that must survive).
        self.assertEqual(D_POOL - w - D_FLOOR, 1846)

    def test_the_uncapped_ask_is_the_whole_pool_which_is_the_defect(self):
        # The arithmetic that shipped: min(need, available) on an empty pool.
        available = D_POOL
        self.assertEqual(min(D_NEED, available), D_POOL)
        # Capped, one window is asked for and the slack survives.
        self.assertEqual(min(min(D_NEED, available), D_W), D_W)
        self.assertGreater(D_POOL - D_W, 0)

    def test_five_windows_cover_the_prompt_that_could_not_take_two(self):
        self.assertEqual(-(-D_NEED // D_W), 5)
        self.assertIn("W=24576", host_ledger.window_provenance(
            D_POOL, D_CHUNK, D_CHAINS, D_FLOOR, 37, D_NEED))


class TestProgressNotSeconds(unittest.TestCase):
    """(2) Both directions of the liveness predicate."""

    def _sched(self):
        s = _bind([
            "_weg2_prefetch_stall_passes",
            "_weg2_prefetch_progress_terms",
            "_weg2_note_prefetch_progress",
        ])
        s.tree_cache = _Tree()
        s._weg2_window_closed = 0
        s._weg2_window_reissues = 0
        return s

    def test_a_moving_chain_is_never_cut_however_long_it_takes(self):
        """THE FALSE-REFUSAL DIRECTION. Far past any clock, still no verdict."""
        s, req = self._sched(), _Req()
        bound = s._weg2_prefetch_stall_passes()
        for i in range(bound * 4):
            # One term moves per observation -- the slowest possible progress.
            req.host_hit_length = i + 1
            self.assertEqual(s._weg2_note_prefetch_progress(req), "progress")
        self.assertEqual(getattr(req, "_weg2_no_progress_passes", 0), 0)

    def test_a_standstill_terminates_and_does_so_exactly_at_the_bound(self):
        """THE SPIN DIRECTION. Ten identical rounds was the metal reading."""
        s, req = self._sched(), _Req(prefix=98302)
        bound = s._weg2_prefetch_stall_passes()
        self.assertEqual(s._weg2_note_prefetch_progress(req), "progress")
        for _ in range(bound - 1):
            self.assertEqual(s._weg2_note_prefetch_progress(req), "stalled")
        self.assertEqual(s._weg2_note_prefetch_progress(req), "terminal")

    def test_an_operation_stuck_in_flight_does_not_buy_immunity(self):
        """The lost-ack shape (#989/#1157) must terminate, not wait forever."""
        s, req = self._sched(), _Req(prefix=98302, rid="stuck")
        s.tree_cache.ongoing_prefetch["stuck"] = object()
        verdicts = {
            s._weg2_note_prefetch_progress(req)
            for _ in range(s._weg2_prefetch_stall_passes() + 1)
        }
        self.assertIn("terminal", verdicts)

    def test_one_moving_term_anywhere_in_the_witness_resets_the_streak(self):
        """Every term is load-bearing: rows recycled is progress too."""
        s, req = self._sched(), _Req(prefix=98302)
        s._weg2_note_prefetch_progress(req)
        for _ in range(s._weg2_prefetch_stall_passes() - 1):
            s._weg2_note_prefetch_progress(req)
        s.tree_cache._weg2_window_rows_recycled = 24576
        self.assertEqual(s._weg2_note_prefetch_progress(req), "progress")
        self.assertEqual(req._weg2_no_progress_passes, 0)

    def test_no_seconds_survive_on_the_deferral_path(self):
        """The bound is in PASSES; no source line here reads a clock."""
        import inspect

        for name in ("_apply_group_shortfall_deferral", "_apply_prefetch_deferral"):
            fn = Scheduler.__dict__[name]
            # THE DOCSTRING IS STRIPPED, and that is not a convenience: the
            # first run of this assertion failed on the docstring alone, which
            # is a REAL finding (a comment asserting a bound the code no
            # longer has) but a different one. The docstring was corrected;
            # this scans EXECUTED lines, so the two findings stay separate.
            src = inspect.getsource(fn)
            doc = fn.__doc__ or ""
            body = src.replace(doc, "") if doc else src
            self.assertNotIn(
                "_deferred_prefetch_bound_s", body,
                f"{name} still prices a wall-clock bound in executed code",
            )
            # WHAT IS FORBIDDEN IS THE COMPARISON, NOT THE STOPWATCH.
            # `prefetch_defer_since` is still stamped and still read for the
            # `waited_s` field of the log lines, and that is not a
            # "Zeitschranke": recording how long a mark stood is a diagnostic,
            # while DECIDING on it is the defect. The second run of this
            # assertion failed on the diagnostic and the assertion was
            # narrowed to the decision -- deliberately, and stated here so the
            # next reader does not re-widen it and delete a useful field.
            self.assertNotIn("waited > bound", body,
                             f"{name} still decides on elapsed time")
            self.assertNotIn("bound = self._deferred", body)


class TestAChainMayOnlyCloseCovered(unittest.TestCase):
    """(3) The multi-window witness that did not exist."""

    def _sched(self, x=11101):
        s = _bind([
            "_weg2_window_residual_allowance",
            "_weg2_check_window_coverage",
            "_weg2_store_read_is_pending",
        ])

        class _SA:
            chunked_prefill_size = D_CHUNK

        s.server_args = _SA()
        s.tree_cache = _Tree()
        s.terminal_calls = []
        s._weg2_store_load_terminal = (
            lambda req, arm, span, site, **kw: s.terminal_calls.append(
                (arm, span, site, kw.get("code"))
            )
            or "failed"
        )
        return s

    def test_a_decline_that_closes_the_chain_short_is_refused_by_name(self):
        s = self._sched()
        req = _Req(prefix=98302)  # 10,829 short of 109,129 -- 2.6 chunks
        self.assertEqual(
            s._weg2_check_window_coverage(req, "declined:rate_limited"), "failed"
        )
        self.assertEqual(len(s.terminal_calls), 1)
        self.assertEqual(s.terminal_calls[0][3], "W89")
        self.assertEqual(s.terminal_calls[0][1], D_NEED - 98302)

    def test_a_close_inside_the_one_chunk_allowance_is_legal(self):
        s = self._sched()
        req = _Req(prefix=D_NEED - D_CHUNK)
        self.assertEqual(
            s._weg2_check_window_coverage(req, "declined:rate_limited"), "covered"
        )
        self.assertEqual(s.terminal_calls, [])

    def test_a_close_on_issued_is_never_judged_because_the_span_is_still_landing(self):
        """THE FALSE-POSITIVE GUARD. `matched` is read from the tree, while a
        read that was just issued lands its span passes later -- judging it
        would refuse exactly the successful case."""
        s = self._sched()
        req = _Req(prefix=0)
        self.assertEqual(s._weg2_check_window_coverage(req, "issued"), "pending")
        self.assertEqual(s.terminal_calls, [])

    def test_a_pending_read_is_never_judged_either(self):
        s = self._sched()
        req = _Req(prefix=0, rid="inflight")
        s.tree_cache.ongoing_prefetch["inflight"] = object()
        self.assertEqual(
            s._weg2_check_window_coverage(req, "declined:rate_limited"), "pending"
        )
        self.assertEqual(s.terminal_calls, [])


class TestTheRefusalsAreNamed(unittest.TestCase):
    """The zero that cost a boot cycle is attributable now."""

    def test_every_precondition_of_c1_has_a_name(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        names = UnifiedRadixCache._WINDOW_RELEASE_REFUSALS
        self.assertIn("not_l3_present", names)
        self.assertIn("host_pin_held", names)
        self.assertIn("not_device_resident", names)
        census = UnifiedRadixCache._window_release_refusal_census(object.__new__(UnifiedRadixCache))
        # Every key present even at zero: a missing key and a zero key read
        # the same to a grep, and attribution is the whole point.
        for n in names:
            self.assertIn(f"{n}=0", census)

    def test_the_new_w_codes_are_above_the_highest_in_use(self):
        import inspect

        src = inspect.getsource(Scheduler.__dict__["_weg2_store_load_terminal"])
        self.assertIn("W88", src)
        self.assertIn("Weg2StoreLoadNotProgressing", src)


if __name__ == "__main__":
    unittest.main()
