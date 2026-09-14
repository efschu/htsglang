# SPDX-License-Identifier: Apache-2.0
"""#1397 (Option 3, DESIGN_option3_band_credit_0914.md section 9) -- the
LAST missing link: launcher.py (argparse + terms publication) and
checkpoint_census.py (parameter pass-through), DESK9's own file boundary.

WHAT WAS MISSING, per DESK11's own prior-art gate (design doc section 9.1,
verified here rather than trusted): ``checkpoint_census.widest_layer_terms``
took ``pairs, depth, slot_bytes, n_lanes, max_tag_bytes, lanes_concurrent``
but never ``band_credit``/``n_cross_lanes`` -- the ONE call site between the
launcher and ``xchg_bounce.bounce_terms`` for every OTHER field. DESK10's
own commit (26bd28089a) independently confirmed the OTHER two links were
already sound: ``weight_updater.py`` has exactly two ``BounceTerms``
construction sites, both ``xb.read_published_terms()`` (never a local
``bounce_terms(...)`` reconstruction that could drop the two new fields),
and the phased cross path already builds
``CrossSlotRendezvous(sems, slots, pair=pair)`` for a real cross descriptor
-- ``_band_credit_leg``'s second precondition. So the gap was narrower than
"wire the whole feature": one function's signature, plus the launcher's own
CLI flag and the two call sites that resolve it.

ONE SOURCE, no second publication path: ``--xchg-band-credit`` (a plain
``store_true``, default off, byte-identical) -> ``main()`` resolves it
EXACTLY ONCE -> both the ranks' publication call
(``checkpoint_census.widest_layer_terms`` -> ``xchg_bounce.bounce_terms`` ->
``_TERM_FIELDS`` -> ``publish_terms``) and ``choose_host_ledger``'s own
pricing call read the IDENTICAL resolved bool. ``n_cross_lanes`` is
DELIBERATELY not exposed as a second CLI flag anywhere in this file --
design doc section 9.2's own finding: arming reduces to the single boolean,
``xchg_bounce.resolve_cross_lanes`` derives a safe, structural lower bound
from ``n_lanes`` (measured, #1358) and ``weight_exchange_region.N_CARDS``
(a pigeonhole argument), so there is nothing else to coordinate.

ORDER ITEM 1 -- READER ENUMERATION BEFORE CHANGING THE PRODUCER (verified
by file:line reading, not assumed from DESK10's own commit message):
``weight_updater.py`` reads a lane's buffer size in EXACTLY TWO places,
:1078 and :1213, BOTH ``terms = xb.read_published_terms()`` -- no third
reader, no local reconstruction of ``BounceTerms`` that could silently drop
``band_credit``/``n_cross_lanes`` the way a #1369-class defect would one
layer down (a consumer holding its own, un-refreshed copy of a decision the
producer changed). ``TheReadersAreEnumeratedAndStayThatWay`` below is the
regression tripwire.

ORDER ITEM 2 -- THE #1385 UNLINK INTERACTION, CHECKED (design doc's own
open question, `--xchg-lanes-concurrent`'s round-3/4 lesson: a cap on
SIMULTANEOUSLY PINNED buffers is not a cap on ACCUMULATED-unfreed files;
weight_updater.py:4170's own ``unlink_lane_buffer`` call is what closes
that gap for ``lanes_concurrent``). FINDING: band_credit does NOT touch the
unlink side, and needs no unlink logic of its own.
``weight_exchange_bounce.unlink_lane_buffer(boot_nonce, shm_root, lane)``
removes a file by PATH (the lane's own name, e.g. ``bounce.bin.p1``),
completely independent of what SIZE was ever written there -- band_credit
only changes the BYTE COUNT a cross lane's file holds, never its path, its
existence, or when it is freed. The call at weight_updater.py:4170 is
gated on `_lane_permit_active` (whether `--xchg-lanes-concurrent` armed a
cap), which is an INDEPENDENT axis from `band_credit`: with no cap, #1385's
own pricing already charges for every lane accumulating "by design" (that
comment's own words), and band_credit's smaller PER-LANE cross price is
still the correct charge for that assumption -- it changes the SIZE term
in a formula that already assumes full accumulation, not the accumulation
assumption itself. `TheUnlinkSideIsUntouchedByBandCredit` below pins this
as a source-level fact rather than leaving it as a claim in this docstring.

THE "WRONG REGION" DANGER DIRECTION (order item, coordinator-named): "ein
Band-Credit, der die Baender der falschen Region liest, schreibt fremde
Bytes = still falsch, teurer als jede Refusal" is a RUNTIME CORRECTNESS
question about ``CrossSlotRendezvous``/``_band_credit_leg`` --
weight_exchange_bounce.py and weight_updater.py, DESK11's and DESK10's own
file boundaries, never launcher.py/checkpoint_census.py (this file's own
boundary is PRICING and PUBLICATION, which cannot misidentify a region --
there are no descriptors, no ranks, no pairs at this layer, only byte
counts). DESK10's own commit (26bd28089a) already ran exactly this mutant
by hand against the real execution path: rendezvous construction switched
from `pair=pair` to a card-only identity for a cross group (pair identity
lost) -- the size-comparison test went red (24576==24576, no difference
survived), confirming the real allocator's cross/diagonal split depends on
the SAME pair identity a wrong-region read would corrupt. Not duplicated
here: this file's own danger-direction mutant (below) is the one reachable
from ITS boundary -- the flag failing to reach one of the two publication
call sites, which is the #1369-class defect this order's own precedent
names.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import inspect

from sglang.srt.weg2 import checkpoint_census as cc
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import xchg_bounce as xb
from sglang.test.test_utils import CustomTestCase

GIB = xb.GIB

# xsn31/3's own measured geometry -- the SAME fixture
# test_weg2_xchg_lanes_concurrent_1385.py uses, reused rather than a second
# invented checkpoint shape.
XSN31_3 = dict(
    bytes_per_direction=29119878266, n_layers=64,
    widest_layer_bytes=756323776, pairs=3, depth=2,
    n_lanes=5, max_tag_bytes=2907 * (1 << 20),
)


def _terms(slot_mib: int, **kw):
    base = dict(XSN31_3, slot_bytes=slot_mib * (1 << 20))
    base.update(kw)
    return xb.bounce_terms(**base)


class TheFlagIsByteIdenticalByDefault(CustomTestCase):
    def test_default_off_prices_the_pre_1397_total(self):
        off = _terms(128)
        self.assertFalse(off.band_credit)
        self.assertEqual(off.n_cross_lanes_priced, 0)
        self.assertAlmostEqual(off.total_bytes / GIB, 15.75, places=2)

    def test_the_cli_flag_defaults_to_false(self):
        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        self.assertIs(ns.xchg_band_credit, False)

    def test_the_cli_flag_parses_store_true(self):
        ns = L.build_parser().parse_args(
            ["--tree", "/t", "--tag", "t", "--xchg-band-credit"])
        self.assertIs(ns.xchg_band_credit, True)

    def test_no_n_cross_lanes_flag_exists_arming_is_the_one_boolean(self):
        """Design doc section 9.2's own finding, pinned: no second CLI knob
        anywhere for n_cross_lanes -- arming reduces to the single flag."""
        help_text = L.build_parser().format_help()
        self.assertNotIn("--xchg-n-cross-lanes", help_text)
        self.assertNotIn("--xchg-cross-lanes", help_text)


class OneSourceReachesBothPublicationSides(CustomTestCase):
    def test_all_three_layers_accept_band_credit(self):
        self.assertIn("band_credit",
                      inspect.signature(cc.widest_layer_terms).parameters)
        self.assertIn("n_cross_lanes",
                      inspect.signature(cc.widest_layer_terms).parameters)
        self.assertIn("band_credit",
                      inspect.signature(L.xchg_bounce_terms_for_arm).parameters)
        self.assertIn("band_credit",
                      inspect.signature(L.choose_host_ledger).parameters)

    def test_no_n_cross_lanes_parameter_at_the_launcher_layer(self):
        """xchg_bounce_terms_for_arm/choose_host_ledger deliberately do NOT
        expose n_cross_lanes -- it always auto-derives, one boolean only."""
        self.assertNotIn(
            "n_cross_lanes",
            inspect.signature(L.xchg_bounce_terms_for_arm).parameters)
        self.assertNotIn(
            "n_cross_lanes",
            inspect.signature(L.choose_host_ledger).parameters)

    def test_main_resolves_the_flag_exactly_once(self):
        src = inspect.getsource(L.main)
        self.assertEqual(
            src.count('getattr(ns, "xchg_band_credit", False)'), 1,
            "the flag must be read from ns exactly once in main() -- a "
            "second read is how the ranks' publication and the ledger's "
            "own pricing could resolve two different answers")

    def test_xchg_bounce_terms_for_arm_forwards_to_widest_layer_terms(self):
        src = inspect.getsource(L.xchg_bounce_terms_for_arm)
        i = src.index("checkpoint_census.widest_layer_terms(")
        self.assertIn("band_credit=bool(band_credit)", src[i:i + 400])

    def test_choose_host_ledger_forwards_the_same_local_both_call_sites(self):
        """main() must hand the ONE resolved local (`_band_credit`) to BOTH
        the ranks' publication call and choose_host_ledger -- never re-read
        `ns` a second time for the ledger side."""
        src = inspect.getsource(L.main)
        self.assertGreaterEqual(src.count("band_credit=_band_credit"), 2,
                                "expected _band_credit forwarded to both "
                                "the ranks' publication call and "
                                "choose_host_ledger")


class TheLedgerSeesTheSmallerCharge(CustomTestCase):
    def test_charge_terms_reflects_the_smaller_total_when_armed(self):
        from sglang.srt.weg2 import host_ledger as hl

        off = _terms(128, band_credit=False)
        on = _terms(128, band_credit=True)
        img = hl.ImageTerms(
            p_gib=38.63, d_gib=38.63, p_source="s", d_source="s",
            p_measured=True, d_measured=True, extra_p_gib=0.0, extra_d_gib=0.0,
        )
        charged_off = hl.charge_terms(
            1, 600, 3, img, s_gb_d=4, xchg_bounce_host_bytes=off.total_bytes)
        charged_on = hl.charge_terms(
            1, 600, 3, img, s_gb_d=4, xchg_bounce_host_bytes=on.total_bytes)
        self.assertLess(
            charged_on["xchg_bounce_gib"], charged_off["xchg_bounce_gib"],
            "the ledger must SEE the smaller cross-lane price, not merely "
            "a printed advisory (#1256 shape)")
        self.assertAlmostEqual(
            charged_off["xchg_bounce_gib"] - charged_on["xchg_bounce_gib"],
            5.25, places=2)


class DesignDocMathVerifiedAtTheLedger(CustomTestCase):
    """DESK11's own design doc arithmetic (section 9.4), verified through
    the REAL xb.bounce_terms call this order asked for -- not adopted."""

    def test_slot_128_mib_matches_the_design_doc_exactly(self):
        off, on = _terms(128, band_credit=False), _terms(128, band_credit=True)
        self.assertAlmostEqual(off.total_bytes / GIB, 15.75, places=2)
        self.assertAlmostEqual(on.total_bytes / GIB, 10.5, places=2)
        self.assertEqual(on.n_cross_lanes_priced, 2)
        self.assertEqual(on.n_diag_lanes_priced, 3)

    def test_slot_64_mib_diverges_slightly_from_the_design_docs_own_estimate(self):
        """The design doc's own section 9.4 labels this "gerechnet, nicht
        gemessen" (9.75 GiB) -- the REAL function returns 9.562 GiB. Order's
        own instruction: verify, don't adopt. The doc's hand arithmetic is
        the approximation; this number, from the real sizing function, is
        authoritative."""
        off, on = _terms(64, band_credit=False), _terms(64, band_credit=True)
        self.assertAlmostEqual(off.total_bytes / GIB, 15.062, places=2)
        self.assertAlmostEqual(on.total_bytes / GIB, 9.562, places=2)


class TheReadersAreEnumeratedAndStayThatWay(CustomTestCase):
    """Order item 1: enumerate the readers of the lane buffer size with
    file:line BEFORE changing the producer -- done (see module docstring),
    and pinned here as a regression tripwire against a THIRD reader
    appearing that reconstructs BounceTerms locally instead of reading the
    launcher's own published one."""

    def test_weight_updater_has_exactly_two_read_published_terms_call_sites(self):
        import sglang.srt.managers.scheduler_components.weight_updater as wu

        src = inspect.getsource(wu)
        count = src.count("xb.read_published_terms()")
        self.assertEqual(
            count, 2,
            "weight_updater.py must have EXACTLY two "
            "xb.read_published_terms() call sites (verified at #1397 "
            "wiring time: lines 1078 and 1213) -- a THIRD reader, or one "
            "replaced by a local bounce_terms(...) reconstruction, could "
            "silently miss band_credit/n_cross_lanes the way a #1369-class "
            "defect misses a decision one layer down")


class TheUnlinkSideIsUntouchedByBandCredit(CustomTestCase):
    """Order item 2: does band_credit touch the #1385-round-3 unlink side?
    Verified NO -- pinned here as a source-level fact."""

    def test_unlink_lane_buffer_takes_no_band_credit_aware_parameter(self):
        import sglang.srt.weg2.weight_exchange_bounce as wb

        sig = inspect.signature(wb.unlink_lane_buffer)
        self.assertNotIn("band_credit", sig.parameters)
        self.assertNotIn("cross", " ".join(sig.parameters))

    def test_the_unlink_call_site_gates_on_the_permit_not_on_band_credit(self):
        import sglang.srt.managers.scheduler_components.weight_updater as wu

        src = inspect.getsource(wu)
        i = src.index("bx.unlink_lane_buffer(")
        window = src[max(0, i - 300):i]
        self.assertIn("_lane_permit_active", window)
        self.assertNotIn("band_credit", window)


class TheDangerDirectionMutantForThisLayer(CustomTestCase):
    """THE mutant reachable from THIS file's own boundary (launcher.py):
    band_credit is armed at the CLI but the resolved value fails to reach
    ONE of the two call sites (the ranks' publication, or the ledger's own
    pricing) -- exactly the #1369-class defect the order's own precedent
    names (a consumer/producer disagreeing about one resolved decision).
    Reproduced by hand (matching this file's own convention) rather than
    by calling a mutated copy of main(): a caller that resolves
    `_band_credit` but only forwards it to ONE of the two functions below
    prices the ranks' allocation SMALL while the host ledger still charges
    FULL, or the reverse -- either way, a silent size disagreement between
    what was priced and what a rank allocates, the exact OOM-adjacent shape
    this order names.
    """

    def test_mutant_ledger_call_missing_band_credit_prices_the_bigger_total(self):
        """Direct sizing comparison (xsn31/3's own real geometry, not the
        tiny fake checkpoint below -- that checkpoint's layers are far
        smaller than even the 64 MiB default slot, so max_tag_bytes never
        exceeds the assemble-depth floor and band_credit's split collapses
        to a no-op there by construction, proving nothing about the wiring
        itself). This is the SAME comparison `xb.bounce_terms` itself
        already makes -- reproduced here to describe it as the mutant this
        order names: a caller that resolves band_credit but forwards it
        to only ONE of the two functions below (ranks vs. ledger) is
        exactly what `xchg_bounce_terms_for_arm(..., band_credit=False)`
        vs. `(..., band_credit=True)` demonstrates the SIZE consequence of.
        """
        off = _terms(128, n_lanes=5, band_credit=False)
        on = _terms(128, n_lanes=5, band_credit=True)
        self.assertGreater(
            off.total_bytes, on.total_bytes,
            "with band_credit silently dropped on one side, that side "
            "prices/allocates the FULL diagonal-floor total for every "
            "lane while the other side (correctly wired) charges less -- "
            "the two numbers must diverge, or a dropped argument would be "
            "invisible")


class TheDangerDirectionMutantAppliedLiveAndReverted(CustomTestCase):
    """THE SAME mutant as above, but applied to the REAL launcher.py source
    and reverted, matching this session's own established discipline
    (#1369's BOOT7 fix, #1395's reader mutant) rather than only a
    function-argument comparison. Kept as its own test so a CI run that
    executes this file always exercises it -- the live-mutant step itself
    was performed manually during development (revert-and-diff-empty,
    documented in the commit message) and this test is what would have
    caught it automatically had the revert been forgotten.
    """

    def test_choose_host_ledger_call_site_still_forwards_band_credit(self):
        src = inspect.getsource(L.choose_host_ledger)
        i = src.index("_bounce_charge_bytes, _bounce_lines = xchg_bounce_terms_for_arm(")
        call = src[i:i + 250]
        self.assertIn("band_credit", call,
                     "choose_host_ledger's own call to "
                     "xchg_bounce_terms_for_arm must forward band_credit -- "
                     "if this argument is ever dropped, the ledger silently "
                     "reverts to full diagonal-floor pricing regardless of "
                     "the flag, invisible on the ARM line's own total alone")
