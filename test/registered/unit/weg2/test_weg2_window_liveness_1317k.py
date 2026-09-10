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
D_FLOOR = D_CHUNK
D_NEED = 109129
# TWO chain counts, and keeping both is the point. `solve_window`'s docstring
# works its example at chains=1 -> W=24,576, and reading that example as "group
# D's window" is exactly the drift that let W go unwired for a whole ticket:
# `chains` is `--max-running-requests`, which D's argv sets to **6**
# (`max_running_requests=6`, weg2sn6l's D log), so the number this rig actually
# gets is 4,096 -- ONE CHUNK, and 27 windows for the user's prompt rather than
# 5. Both are pinned below so neither can be mistaken for the other again.
D_CHAINS_SPEC = 1
D_W_SPEC = 24576
D_CHAINS_RIG = 6
D_W_RIG = 4096


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

    def test_the_solver_reproduces_the_specs_own_worked_example(self):
        w = host_ledger.solve_window(D_POOL, D_CHUNK, D_CHAINS_SPEC, D_FLOOR, 37)
        self.assertEqual(w, D_W_SPEC)
        # The slack the solver predicts is the room the NEXT window needs, and
        # it is the number metal actually saw on its second attempt (got=1847
        # against this 1846 -- one row is the resume anchor that must survive).
        self.assertEqual(D_POOL - w - D_FLOOR, 1846)

    def test_the_window_this_rig_actually_gets_is_one_chunk_not_the_example(self):
        """THE NUMBER A READER WILL GET WRONG. chains is --max-running-requests
        = 6 on group D, so W is 4,096 and the prompt needs 27 windows. Pinned
        beside the spec example because reading the example as the rig's number
        is the same drift that left W unwired: a quantity that looks derived,
        is derived, and describes a different configuration."""
        w = host_ledger.solve_window(D_POOL, D_CHUNK, D_CHAINS_RIG, D_FLOOR, 37)
        self.assertEqual(w, D_W_RIG)
        self.assertEqual(w, D_CHUNK, "one chunk per window at bs6")
        self.assertEqual(-(-D_NEED // w), 27)
        # And the pool stays wide open, which is the whole fix: 22,326 rows
        # free after window 1 instead of the metal reading of available=0.
        self.assertEqual(D_POOL - w - D_FLOOR, 22326)

    def test_the_uncapped_ask_is_the_whole_pool_which_is_the_defect(self):
        # The arithmetic that shipped: min(need, available) on an empty pool.
        available = D_POOL
        self.assertEqual(min(D_NEED, available), D_POOL)
        # Capped, one window is asked for and the slack survives -- and the
        # metal consequence of NOT capping is the reading this pins against:
        # occupied=30518 available=0, then 26,985 `reason=vote_negative`
        # refusals of the same rid against a threshold of only 256.
        self.assertEqual(min(min(D_NEED, available), D_W_RIG), D_W_RIG)
        self.assertGreater(D_POOL - D_W_RIG, 0)

    def test_the_provenance_line_names_every_term_of_whichever_w_it_answers(self):
        # The line the CAP now prints -- and the reason the acceptance derives
        # `got=<W>` from it instead of hardcoding a number.
        rig = host_ledger.window_provenance(
            D_POOL, D_CHUNK, D_CHAINS_RIG, D_FLOOR, 37, D_NEED)
        self.assertIn("W=4096", rig)
        self.assertIn("chains=6", rig)
        self.assertIn("windows_for_109129=27", rig)
        spec = host_ledger.window_provenance(
            D_POOL, D_CHUNK, D_CHAINS_SPEC, D_FLOOR, 37, D_NEED)
        self.assertIn("W=24576", spec)
        self.assertIn("chains=1", spec)


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


# ---------------------------------------------------------------------------
# #1317l -- the four killers boot weg2sn6n produced, after #1317k removed the
# host-pool wall it was built for. The wall moved; it did not vanish.
# ---------------------------------------------------------------------------


class TestADeclineIsNotAFinishedRead(unittest.TestCase):
    """sn6n's killer: the chain closed after window 1 of ~27 on a DECLINE.

        #1317 WINDOW-REISSUE n=1 verdict=declined:anchor_pool_exhausted
            still_owed=False matched=4096 total=109130 closed_total=1

    Both directions are pinned, because getting either one wrong is a
    different disaster: a decline that CLOSES the chain throws away 105,034
    tokens of the prompt (measured), and a decline that keeps it armed with
    nothing ending it is the pre-#1317k spin (26,985 refusals, also measured).
    """

    def _verdict_owed(self, verdict):
        """The shipped predicate, read out of the source rather than retyped.

        Retyping it here would let the test agree with a copy of the rule
        while the executed rule drifted -- which is the second-bookkeeping
        shape this fork deletes on sight.
        """
        import inspect
        import re

        src = inspect.getsource(Scheduler.__dict__["_weg2_issue_next_window"])
        m = re.search(r"_declined = (.+)\n\s+still_owed = (.+)", src)
        assert m, "the still_owed predicate changed shape; update this reader"
        _declined = eval(m.group(1), {}, {"verdict": verdict})  # noqa: S307
        return eval(m.group(2), {}, {"verdict": verdict, "_declined": _declined})  # noqa: S307

    def test_a_decline_keeps_the_window_owed(self):
        for verdict in (
            "declined:anchor_pool_exhausted",   # sn6n's own verdict
            "declined:rate_limited",
            "declined:already_in_flight",
            "declined:vote_negative",
        ):
            self.assertTrue(self._verdict_owed(verdict), verdict)

    def test_a_truncated_group_still_owes_the_next_window(self):
        self.assertTrue(self._verdict_owed("issued:truncated_group"))

    def test_only_a_whole_read_closes_the_chain(self):
        self.assertFalse(self._verdict_owed("issued"))

    def test_the_standstill_exit_is_the_precondition_of_keeping_it_armed(self):
        """Keeping a chain armed on every decline is only safe because a
        standstill still terminates. If the progress witness ever leaves the
        still_owed branch, this fix becomes the spin it replaced."""
        import inspect

        src = inspect.getsource(Scheduler.__dict__["_weg2_issue_next_window"])
        owed_arm = src.split("if still_owed:", 1)[1].split("else:", 1)[0]
        self.assertIn("_weg2_note_prefetch_progress", owed_arm)
        self.assertIn("_weg2_store_load_terminal", owed_arm)


class TestTheComponentsCoverWhatWasAllocated(unittest.TestCase):
    """sn6n's new wall: 13 anchor slots against an ask for ~27.

    `#1035 PREFETCH DROPPED (host anchor pool exhausted) prefetch_tokens=109128
    host_anchor_avail=0 host_anchor_size=13` fired while the KV pool sat at
    `available=26422`. Two pools, and only one of them was capped: the KV alloc
    was bounded but this loop still asked every component for
    `len(prefetch_key)` -- the whole remaining prompt -- because prefetch_key
    is trimmed only AFTER the group vote.
    """

    def _src(self):
        import inspect

        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        return inspect.getsource(UnifiedRadixCache.prefetch_from_storage)

    def test_the_component_ask_is_the_allocated_span_not_the_whole_key(self):
        src = self._src()
        self.assertIn("_comp_tokens = len(host_indices)", src)
        self.assertIn("prefetch_tokens=_comp_tokens", src)
        self.assertIn("token_ids=_comp_ids", src)
        # The defect spelling must be gone, or the anchor wall comes back.
        self.assertNotIn("prefetch_tokens=len(prefetch_key)", src)

    def test_the_vote_still_sees_the_full_remainder(self):
        """The trim may not reach `local_span`: that is what makes
        `truncated_group` true and arms the next window. Capping it would
        close every chain after one window -- the same outcome as the killer,
        reached from the other side."""
        src = self._src()
        self.assertIn("local_span = len(prefetch_key)", src)

    def test_one_window_needs_two_anchors_against_thirteen_slots(self):
        """The arithmetic that makes 13 slots sufficient once the ask is
        capped: W//chunk + 1, which is `solve_window`'s own W21 term."""
        anchors_live = D_W_RIG // D_CHUNK + 1
        self.assertEqual(anchors_live, 2)
        self.assertLessEqual(anchors_live, 13)
        # And the ask that actually fired on metal, for contrast.
        self.assertGreater(-(-109128 // D_CHUNK) + 1, 13)


class TestTheCensusRidesEverySeam(unittest.TestCase):
    """The #1317k promise was "a bare WINDOW-RELEASE=0 is now impossible", and
    sn6n showed it held only for a chain that DIES: that chain CLOSED, took the
    success path, and printed no census anywhere."""

    def test_the_reissue_line_carries_the_release_census(self):
        import inspect

        src = inspect.getsource(Scheduler.__dict__["_weg2_issue_next_window"])
        self.assertIn("release_refusals=[%s]", src)
        self.assertIn("_weg2_window_release_census()", src)

    def test_an_unanswerable_census_is_named_never_blank(self):
        """A missing census and an all-zero census read the same to a grep."""
        s = _bind(["_weg2_window_release_census"])
        s.tree_cache = None
        self.assertEqual(
            s._weg2_window_release_census(), "unnamed:no_census_on_this_tree"
        )

        class _Raises:
            def _window_release_refusal_census(self):
                raise RuntimeError("boom")

        s.tree_cache = _Raises()
        self.assertEqual(s._weg2_window_release_census(), "unnamed:census_raised")

    def test_a_real_tree_answers_with_every_key(self):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        s = _bind(["_weg2_window_release_census"])
        s.tree_cache = object.__new__(UnifiedRadixCache)
        got = s._weg2_window_release_census()
        for n in UnifiedRadixCache._WINDOW_RELEASE_REFUSALS:
            self.assertIn(f"{n}=0", got)


class TestTheExemptionDiesOnItsOwnTerms(unittest.TestCase):
    """D was admitted over X 15 times on sn6n and prefilled the whole 109k
    prompt WHILE the client was refused anyway -- the barrier that fed the
    exemption was gone but the exemption outlived it."""

    def test_the_exemption_is_gated_on_the_windowed_predicate(self):
        import inspect

        src = inspect.getsource(Scheduler.__dict__["_weg2_x_refuses"])
        self.assertIn("if _weg2_windowed_path(self) and carry > 0 "
                      "and uncached > carry:", src)
        # The retention path keeps the exemption, because there the wall it
        # was built for still exists.
        self.assertIn("elif carry > 0 and uncached > carry:", src)
        self.assertIn("verdict=exempt_carrier_exceeds", src)

    def test_the_guard_is_a_module_function_not_a_method(self):
        """THE THIRD FORM, and the first that works. A defensive METHOD failed
        the same way the direct call did -- the stand-ins bind a CURATED list
        of real methods, so a NEW method is missing from them exactly as the
        predicate was. Two gate runs, one lesson: the guard cannot live behind
        an attribute lookup on the receiver when the receiver is the
        incomplete thing."""
        import types

        from sglang.srt.managers import scheduler as sched_mod

        self.assertIsInstance(sched_mod._weg2_windowed_path, types.FunctionType)
        self.assertNotIn("_weg2_windowed_path", Scheduler.__dict__)
        # The exact shape the harnesses hand it: a namespace with none of this.
        self.assertFalse(sched_mod._weg2_windowed_path(types.SimpleNamespace()))

        class _Raises:
            def _weg2_windowed_store_read_active(self):
                raise RuntimeError("boom")

        self.assertFalse(sched_mod._weg2_windowed_path(_Raises()))

    def test_the_code_itself_named_this_retirement_condition(self):
        """Not a design deviation: the exemption's own comment set its sunset,
        and this is the commit that meets it."""
        import inspect

        src = inspect.getsource(Scheduler.__dict__["_weg2_x_refuses"])
        self.assertIn("the exemption dies in the same commit as the wall", src)


# ---------------------------------------------------------------------------
# #1317m -- boot weg2sn6o. The W cap and the retry both WORKED; the read was
# voted down at the ANCHOR pool, and two diagnoses died before the instrument
# existed to settle it.
# ---------------------------------------------------------------------------


class TestTheInstrumentNamesItsNumbers(unittest.TestCase):
    """`prefetch_tokens=109131` on the #1035 line was `len(prefetch_key)`, not
    the ask -- a field that was honest about its number and misleading about
    its NAME, and it cost a boot's worth of analysis (a reader concluded the
    mamba component was asking for the whole prompt; it asks for one slot).
    """

    def _src(self):
        import inspect

        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        return inspect.getsource(UnifiedRadixCache.prefetch_from_storage)

    def test_the_1035_line_prints_the_ask_AND_the_remainder_separately(self):
        src = self._src()
        self.assertIn("asked_tokens=%d", src)
        self.assertIn("remaining_tokens=%d", src)
        self.assertIn("anchors_asked=%d", src)
        # The misleading name must be gone: `prefetch_tokens=` on THIS line
        # read as "what this read asked for" and was the prefix length.
        self.assertNotIn("prefetch_tokens=%d", src)

    def test_the_two_numbers_are_not_the_same_expression(self):
        """Printing one value under two names would be the same defect."""
        src = self._src()
        head = src.index("asked_tokens=%d")
        args = src[head:]
        self.assertIn("_comp_tokens,", args)
        self.assertIn("len(prefetch_key),", args)


class TestTheMambaArmAsksForOneSlot(unittest.TestCase):
    """The refutation that saved a no-op boot, pinned so it cannot be
    re-proposed: the PREFETCH arm allocates ONE slot and reads neither
    `prefetch_tokens` nor `token_ids`, so capping either changes nothing."""

    def _arm(self):
        import inspect

        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        src = inspect.getsource(MambaComponent.build_hicache_transfers)
        return src.split("if phase == CacheTransferPhase.PREFETCH:", 1)[1]

    def test_it_allocates_exactly_one_slot(self):
        arm = self._arm()
        self.assertIn("_mamba_pool_host.alloc(1)", arm)

    def test_it_reads_neither_span_parameter(self):
        # EXECUTED LINES ONLY. The first run of this assertion tripped on the
        # site's own COMMENT, which says "reads neither `prefetch_tokens` nor
        # `token_ids`" -- the third time today a source-text assertion matched
        # prose instead of code (a docstring asserting a retired bound, a
        # comment naming a field). Comments are stripped, so the claim is about
        # what runs.
        arm = self._arm().split("return []", 1)[0]
        code = "\n".join(
            l for l in arm.split("\n") if not l.lstrip().startswith("#")
        )
        self.assertNotIn("prefetch_tokens", code)
        self.assertNotIn("token_ids", code)


class TestTheAnchorHolderIsNamed(unittest.TestCase):
    """sn6o died 84x at `avail=0 size=13` with an eviction between two allocs
    of ONE slot that freed nothing, and no instrument could say why. Two
    theories died first (the whole-prompt ask; the H-leaf blind spot -- refuted
    because `drive_host_eviction` walks the mamba host LRU and tombstones
    INTERNAL nodes), so this counts the candidate holders instead of picking
    one."""

    def _census(self):
        import inspect

        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        return inspect.getsource(MambaComponent.build_hicache_transfers)

    def test_all_three_candidate_holders_are_counted(self):
        src = self._census()
        self.assertIn("lru_anchor_holders=%d", src)
        self.assertIn("of_which_host_pinned=%d", src)
        self.assertIn("of_which_h_leaves=%d", src)

    def test_the_census_cannot_break_the_intake(self):
        """It runs on the collective path: an exception here would leave one
        rank out of a vote its peers are already in (#580)."""
        src = self._census()
        body = src.split("#1317m WHO HOLDS THE ANCHORS?", 1)[1].split("return []", 1)[0]
        self.assertIn("except Exception:", body)

    def test_the_refuted_theories_are_recorded_where_they_would_be_retried(self):
        """Both dead diagnoses are named at the site, so the next reader does
        not spend a window re-deriving them."""
        src = self._census()
        self.assertIn("refuted", src)


class TestTheCeilingLineDoesNotPickARefutedCause(unittest.TestCase):
    def test_it_prints_the_arithmetic_without_naming_a_holder(self):
        import inspect

        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache._weg2_window_alloc_cap)
        self.assertIn("ANCHOR CEILING", src)
        self.assertIn("stays OPEN", src)
        # Both pool readings side by side -- one without the other is what made
        # sn6o's two instruments look contradictory.
        self.assertIn("available=26422", src)
        self.assertIn("host_anchor_avail=0", src)
