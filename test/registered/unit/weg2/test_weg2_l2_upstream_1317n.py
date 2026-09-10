# SPDX-License-Identifier: Apache-2.0
"""#1317n -- D's L2 is DERIVED from the request cap, and the compensation is gone.

THE INVARIANT THIS RESTORES, in upstream's own words. `pool_host/base.py`
sizes the host pool as `ratio x device_pool` when `--hicache-size` is unset
(ratio 2.0, so L2 >= L1) and MIN-syncs a fixed size across ranks because "the
lockstep schedulers and the host radix index must agree on one slot count".
This fork shipped a fixed 1 GB to BOTH groups -- 30,518 rows on D, a #915
limit of 27,466 -- which broke it, and the #1317 window/chain/anchor layer plus
the 413 band was the compensation. The user ordered the compensation deleted
and the invariant restored (2026-09-10).

WHY THE ABSOLUTE FLAG STAYS RATHER THAN THE RATIO, priced not preferred: D's
profiled per-rank device capacity on boot weg2sn6p was [188788, 262358,
260566] tokens, so `ratio 2.0` against the largest is 17.2 GB on one rank and
51.5 GB across three, against 39 GiB MemAvailable and a reap mark at 95.90 GiB.
And the ratio path is not even admissible here: upstream's own docstring says
"ratio-based sizing already derives from the SYNCED device pool size", which is
true for even TP and false under this fork's uneven DCP -- an audit of the
MIN-synced consumers found `prefetch_budget.py` requiring `--hicache-size`
BECAUSE it is MIN-synced (the fork deleted its own MIN all_reduce in #1068
slice 2 on that basis) and `staging_write_ring.py` naming a rank-dependent
admission bound as the #645 defect. Uniform, derived, priced.
"""

import unittest

from sglang.srt.weg2 import host_ledger as hl

# Boot weg2sn6p, group D.
CAP = 262144
SHARE = 24 / 64          # installed ownership vector [17, 24, 23]
CELL = 32768
FRACTION = 0.9           # HICACHE_LOAD_POOL_USAGE_FRACTION


class TestTheDerivation(unittest.TestCase):
    def test_it_reproduces_the_ordered_numbers_exactly(self):
        t = hl.derive_d_hicache_size_gb(CAP, SHARE, CELL, FRACTION)
        self.assertEqual(int(t["s_gb"]), 4)
        self.assertEqual(int(t["rows_needed"]), 98304)
        self.assertEqual(int(t["rows"]), 122070)
        self.assertEqual(int(t["rows_lendable"]), 109863)
        self.assertAlmostEqual(t["gb_exact"], 3.58, places=2)

    def test_the_pool_holds_a_cap_sized_read_BEHIND_the_load_fraction(self):
        """The whole point: not "the pool is 4 GB" but "the controller will
        lend enough of it for one cap-sized read". The old 1 GB gave 30,518
        rows and a limit of 27,466 against the 98,304 a read needs."""
        t = hl.derive_d_hicache_size_gb(CAP, SHARE, CELL, FRACTION)
        self.assertGreaterEqual(t["rows_lendable"], t["rows_needed"])
        old_rows = int(1 * hl.GB // CELL)
        self.assertLess(int(old_rows * FRACTION), int(t["rows_needed"]))

    def test_a_misread_share_is_refused_not_sized(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                hl.derive_d_hicache_size_gb(CAP, bad, CELL, FRACTION)

    def test_the_provenance_line_carries_every_term(self):
        t = hl.derive_d_hicache_size_gb(CAP, SHARE, CELL, FRACTION)
        line = hl.d_hicache_provenance(t, "flag-default 0.40 vs installed unknown")
        for frag in ("hicache_size=4 GB derived", "cap=262144", "fraction=0.90",
                     "cell=32768", "rows_needed=98304", "rows=122070",
                     "rows_lendable=109863", "flag-default 0.40"):
            self.assertIn(frag, line)


class TestTheTwoGroupsArePricedApart(unittest.TestCase):
    """NO BOOT WITH S_D=S_P IN THE LEDGER: D carries 4 GB where P carries 1,
    which is +8.38 GiB of rings against a reap mark nobody may touch."""

    def _images(self):
        import inspect
        n = len(inspect.signature(hl.ImageTerms).parameters)
        return hl.ImageTerms(*([0.0] * n))

    def test_the_default_is_byte_identical_to_the_old_single_budget(self):
        z = self._images()
        self.assertEqual(
            hl.charge_terms(1, 600, 3, z)["rings_gib"],
            hl.charge_terms(1, 600, 3, z, s_gb_d=1)["rings_gib"],
        )

    def test_the_d_budget_is_charged_and_the_delta_is_the_ordered_one(self):
        z = self._images()
        base = hl.charge_terms(1, 600, 3, z)["rings_gib"]
        got = hl.charge_terms(1, 600, 3, z, s_gb_d=4)["rings_gib"]
        self.assertAlmostEqual(got - base, 8.38, places=2)
        self.assertAlmostEqual(
            got, (hl.RING_P_MULT_GB_PER_S * 1 + hl.RING_D_MULT_GB_PER_S * 4)
            * hl.GB / hl.GIB, places=6)

    def test_the_arm_line_prints_which_d_budget_it_charged(self):
        z = self._images()
        priced = hl.charge_terms(1, 600, 3, z, s_gb_d=4)
        self.assertEqual(hl._arm_s_d(type("A", (), {"terms": priced})()), "4")
        # An arm with no D term says so rather than printing P's number.
        self.assertEqual(hl._arm_s_d(type("A", (), {"terms": {}})()), "=S")


class TestTheCompensationIsGone(unittest.TestCase):
    """Deletion is the deliverable, so absence is asserted rather than assumed
    -- an emitter that survives in one module is the second bookkeeping the
    user ordered removed."""

    GONE = (
        "WINDOW-REISSUE", "release_staged_window", "_weg2_issue_next_window",
        "#1317m", "WINDOW-RELEASE", "exempt_carrier_exceeds",
        "windowed_carrier", "_weg2_window_alloc_cap",
    )

    @staticmethod
    def _code_only(src: str) -> str:
        """Source with comments and string literals removed.

        FIFTH INSTANCE TODAY of a source-text assertion matching PROSE instead
        of code, and the first one in a test I wrote after naming the class
        four times: this assertion tripped on the `front.py` COMMENT that
        DOCUMENTS the deletion ("That arm is DELETED"), and it cost a red gate.
        The comment is correct and must stay; the assertion is about what
        EXECUTES. Strings go too -- a deleted emitter's own log text is a
        string literal, and keeping one would be the same false positive
        wearing quotes.
        """
        import io
        import tokenize

        out = []
        try:
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type in (tokenize.COMMENT, tokenize.STRING):
                    continue
                out.append(tok.string)
        except (tokenize.TokenError, IndentationError):  # pragma: no cover
            return src
        return "\n".join(out)

    def test_no_window_chain_or_anchor_census_survives(self):
        import inspect

        from sglang.srt.managers import scheduler
        from sglang.srt.mem_cache import unified_radix_cache
        from sglang.srt.mem_cache.unified_cache_components import mamba_component
        from sglang.srt.weg2 import front

        for mod in (scheduler, unified_radix_cache, mamba_component, front):
            code = self._code_only(inspect.getsource(mod))
            for gone in self.GONE:
                self.assertNotIn(
                    gone, code, f"{mod.__name__} still EXECUTES {gone}")

    def test_the_absence_check_can_still_fail_plant_proof(self):
        """A stripper that removes too much turns this suite into a rubber
        stamp, so the check is proven able to FAIL: every name is planted as
        real code and must be caught, and the same name in a comment or a
        string must NOT be."""
        for gone in self.GONE:
            planted = f"def f():\n    return {gone.replace('-', '_').replace('#', 'n')}\n"
            self.assertIn(
                gone.replace("-", "_").replace("#", "n"),
                self._code_only(planted),
                f"the stripper swallowed planted CODE for {gone}",
            )
        # ...and prose with the same token is invisible, which is the whole point
        prose = '# this mentions WINDOW-REISSUE and exempt_carrier_exceeds\ndef f():\n    return 1\n'
        code = self._code_only(prose)
        self.assertNotIn("WINDOW-REISSUE", code)
        self.assertNotIn("exempt_carrier_exceeds", code)
        docstr = 'def f():\n    """mentions release_staged_window"""\n    return 1\n'
        self.assertNotIn("release_staged_window", self._code_only(docstr))

    def test_design_a_and_the_1246_bound_are_KEPT(self):
        """The delete list is not "everything #1317 touched": the X gate is the
        phase law, not L2 compensation, and the carrier bound is what the
        acceptance reads against cap x share."""
        from sglang.srt.managers.scheduler import Scheduler

        self.assertTrue(hasattr(Scheduler, "_weg2_x_refuses"))
        self.assertTrue(hasattr(Scheduler, "_weg2_host_carry_tokens"))
        self.assertTrue(hasattr(Scheduler, "_weg2_local_store_matches"))


if __name__ == "__main__":
    unittest.main()


class TestTheArmLineLabelsWhatItPriced(unittest.TestCase):
    """Boot weg2sn6r printed `S_D==S` while pricing S_D=4.

    A CORRECT PRICE WITH A WRONG LABEL, in the very line this ticket added.
    The value was only checkable by inverting the ring arithmetic (12.83 GiB is
    unreachable at S_D=1: (1.7778+3.0) x 0.931323 = 4.45), which is precisely
    the work an instrument exists to save. Root: `price()` builds `arm.terms`
    as a FRESH literal dict -- it is not `charge_terms()`'s return -- so the
    key added there never arrived.
    """

    def test_price_puts_the_d_budget_into_the_terms_the_arm_line_reads(self):
        import inspect

        src = inspect.getsource(hl.price)
        self.assertIn('"s_gb_d": float(s_gb if s_gb_d is None else s_gb_d)', src)

    def test_the_label_follows_the_price_for_both_shapes(self):
        # priced apart -> the number; priced together -> the honest "=S"
        self.assertEqual(hl._arm_s_d(type("A", (), {"terms": {"s_gb_d": 4.0}})()), "4")
        self.assertEqual(hl._arm_s_d(type("A", (), {"terms": {"s_gb_d": 1.0}})()), "1")
        self.assertEqual(hl._arm_s_d(type("A", (), {"terms": {}})()), "=S")

    def test_the_ring_term_and_the_label_cannot_disagree(self):
        """The sn6r reading, as an invariant: whatever S_D the label claims
        must be the S_D the rings were priced from."""
        import inspect
        z = hl.ImageTerms(*([0.0] * len(inspect.signature(hl.ImageTerms).parameters)))
        for sd in (1, 3, 4):
            t = hl.charge_terms(1, 600, 3, z, s_gb_d=sd)
            expected = (hl.RING_P_MULT_GB_PER_S * 1
                        + hl.RING_D_MULT_GB_PER_S * sd) * hl.GB / hl.GIB
            self.assertAlmostEqual(t["rings_gib"], expected, places=6)
            self.assertEqual(int(t["s_gb_d"]), sd)


class TestTheRefusalNamesTheLargestFundableCap(unittest.TestCase):
    """Order item 1's missing half: *"ist S_D nicht fundierbar -> benannte
    Refusal am Launch, die den GROESSTEN FUNDIERBAREN CAP nennt (kein stilles
    Verkleinern)"*. The first build shipped the refusal without the number, and
    boot weg2sn6r got a correct W20 that named Sigma H and the rung as levers
    but could not say which cap WOULD fit -- leaving the operator to invert the
    arithmetic by hand.
    """

    def test_it_names_the_cap_the_largest_fundable_budget_serves(self):
        # sn6r: S_D=4 misses the launch moment by 2.78 GiB; one GB of S_D is
        # 3.0 GB of rings = 2.79 GiB, so S_D=3 is the first fundable rung.
        fit = hl.largest_fundable_d_cap(lambda sd: sd <= 3, CAP, SHARE, CELL, FRACTION)
        self.assertEqual(int(fit["s_gb_d"]), 3)
        self.assertEqual(int(fit["rows"]), 91552)
        self.assertEqual(int(fit["rows_lendable"]), 82396)
        self.assertEqual(int(fit["cap_tokens"]), 219722)
        self.assertEqual(int(fit["asked_cap"]), CAP)

    def test_a_bigger_share_buys_a_smaller_cap(self):
        """The share is a divisor, so a margin costs cap -- which is the number
        the operator needs when choosing the flag."""
        at_040 = hl.largest_fundable_d_cap(lambda sd: sd <= 3, CAP, 0.40, CELL, FRACTION)
        self.assertEqual(int(at_040["cap_tokens"]), 205990)
        self.assertLess(at_040["cap_tokens"],
                        hl.largest_fundable_d_cap(
                            lambda sd: sd <= 3, CAP, SHARE, CELL, FRACTION)["cap_tokens"])

    def test_when_d_is_not_the_binding_term_it_says_so_instead_of_advising(self):
        """Advising a smaller cap when D's budget is not what binds would send
        the operator after the wrong lever."""
        self.assertIsNone(
            hl.largest_fundable_d_cap(lambda sd: False, CAP, SHARE, CELL, FRACTION))

    def test_the_refusal_text_carries_the_advice_and_the_warning(self):
        import inspect

        src = inspect.getsource(hl.choose)
        self.assertIn("LARGEST FUNDABLE CAP", src)
        # and it must warn against the wrong fix: shrinking D's L2 alone
        # reintroduces the shortfall the deleted window layer compensated.
        self.assertIn("do NOT lower", src)
