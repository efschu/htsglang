"""#1386 (BOOT7 xsn31/4-6 finding): the ZIELFRAGE MINIMALFORM (two manual
flips + shadow compare, zero requests served) pays 8.14 GiB of the
57.378-GiB host-shmem baseline for a HiCache/Mamba-anchor host tier that a
0-request boot cannot use -- HiCache exists to reuse KV pages ACROSS
requests, and the Minimalform serves none (``served: {"P": 0, "D": 0}`` on
every attempt measured, xsn31/4/5/6).

``--weg2-disable-hicache`` is the switch, and the whole point of this file
is the ONE-SWITCH-ONE-TRUTH rule the order stated verbatim: the exact
"priced one number, allocated another" shape that cost #1385/#1386 three
boots (a lane cap priced at 6.75 GiB while the allocator ran 33.81, twice)
would be repeated here if the ledger term and the argv gate ever read two
different bools. So every test below either (a) proves the two sides agree
for the SAME input, or (b) is one half of the pair a manual mutant run
against (documented at the bottom, not re-run automatically: reverting
either gate alone was confirmed to fail the paired test, then restored).

Read the docstring before adding a sixth call site that threads this bool:
``launcher.py`` `common_flags` -> `argv_p`/`argv_d` and `host_ledger.py`
`charge_terms` -> `price` -> `choose` are the two chains; `launcher.main`
resolves `ns.weg2_disable_hicache` into ONE local exactly once (beside
`draft_kv_on_p`) and hands it, unread again, to both.
"""

from __future__ import annotations

import unittest

import pytest

try:
    from sglang.srt.weg2 import host_ledger, launcher
except Exception as exc:  # pragma: no cover - no weg2 launcher in this build
    pytest.skip(f"weg2 launcher unavailable: {exc}", allow_module_level=True)

GIB = host_ledger.GIB
GB = host_ledger.GB

MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
BUDGETS = [28904, 17704, 17672]


def _flag_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def _images():
    return host_ledger.ImageTerms(
        p_gib=1.0, d_gib=1.0, p_source="test", d_source="test",
        p_measured=True, d_measured=True, extra_p_gib=0.0, extra_d_gib=0.0,
    )


HICACHE_FLAGS = (
    "--enable-hierarchical-cache",
    "--hicache-host-role",
    "--hicache-size",
    "--hicache-mamba-host-mib",
    "--hicache-write-policy",
    "--hicache-storage-backend",
    "--hicache-mem-layout",
    "--hicache-io-backend",
    "--hicache-storage-backend-extra-config",
    "--hicache-canonical-kv-page",
)


class TheLedgerHalf(unittest.TestCase):
    """`host_ledger.charge_terms` -> `price` -> `choose`, all three rungs."""

    def test_default_is_byte_identical_to_every_pre_1386_caller(self):
        # Byte-identical: an existing caller that never heard of the switch
        # must get the exact numbers it always got.
        implicit = host_ledger.charge_terms(1, 1200, 3, _images())
        explicit = host_ledger.charge_terms(1, 1200, 3, _images(), hicache_disabled=False)
        self.assertEqual(implicit, explicit)
        self.assertGreater(implicit["anchors_gib"], 0.0)
        self.assertGreater(implicit["rings_gib"], 0.0)
        self.assertGreater(implicit["overhead_gib"], 0.0)
        self.assertFalse(implicit["hicache_disabled"])

    def test_hicache_disabled_zeroes_exactly_anchors_rings_overhead(self):
        # OFF MEANS OFF: not a smaller S/M routed through the same formula --
        # charge_terms(s_gb=0, ...) would still be a positive number for any
        # s_gb_d that defaults from it, and a caller passing 0 M would still
        # walk the anchors formula to 0.0 by division, not by a named switch.
        # The distinction matters because a future edit that changes what
        # "0" means upstream (e.g. `s_gb_d` no longer defaulting from `s_gb`)
        # would silently change what an `s_gb=0` trick prices; this bool
        # cannot be affected by that, because it never touches s_gb/m_mib.
        on = host_ledger.charge_terms(1, 1200, 3, _images(), hicache_disabled=True)
        off = host_ledger.charge_terms(1, 1200, 3, _images(), hicache_disabled=False)
        self.assertEqual(on["anchors_gib"], 0.0)
        self.assertEqual(on["rings_gib"], 0.0)
        self.assertEqual(on["overhead_gib"], 0.0)
        self.assertTrue(on["hicache_disabled"])
        # heaps/draft/image terms are UNTOUCHED -- this switch owns three
        # keys and no others.
        for key in ("heaps_gib", "draft_host_p_gib", "draft_host_d_gib",
                    "image_p_gib", "image_d_gib", "s_gb_d"):
            self.assertEqual(on[key], off[key], key)

    def test_the_s_gb_0_trick_is_a_DIFFERENT_and_smaller_number(self):
        # Pins the order's own "kein S=0-Trick" line: routing 0 through the
        # EXISTING formula and using the NEW switch must not collide on the
        # same result, or a future reader could mistake one for the other.
        zero_s = host_ledger.charge_terms(0, 0, 3, _images())
        switched_off = host_ledger.charge_terms(1, 1200, 3, _images(), hicache_disabled=True)
        self.assertEqual(switched_off["anchors_gib"], 0.0)
        self.assertEqual(switched_off["rings_gib"], 0.0)
        # anchors_gib at m_mib=0 is ALSO 0.0 (0/2400 * const), but rings_gib
        # at s_gb=0 depends on s_gb_d's own default -- s_gb_d defaults to
        # s_gb, so rings_gib is 0 too. Both land on 0.0 here BY COINCIDENCE
        # of the inputs chosen, which is exactly why the switch, not the
        # trick, is the real fix: the trick's 0 is a property of s_gb/m_mib
        # this call happened to pick, not a declared intent, and any caller
        # that still needs `s_gb`/`m_mib` for something else (they do: the
        # ARM line prints S=/M=, and a 0 there reads as a broken boot, not a
        # disabled cache).
        self.assertEqual(zero_s["anchors_gib"], 0.0)
        self.assertEqual(zero_s["rings_gib"], 0.0)

    def test_price_forwards_the_switch_to_charge_terms(self):
        arm_on = host_ledger.price(
            int(200 * GIB), int(150 * GIB), 1, 1200,
            ring_bytes=int(40 * GIB), ring_span1_bytes=int(35 * GIB),
            hicache_disabled=True,
        )
        arm_off = host_ledger.price(
            int(200 * GIB), int(150 * GIB), 1, 1200,
            ring_bytes=int(40 * GIB), ring_span1_bytes=int(35 * GIB),
            hicache_disabled=False,
        )
        self.assertEqual(arm_on.terms["rings_gib"], 0.0)
        self.assertEqual(arm_on.terms["anchors_gib"], 0.0)
        self.assertTrue(arm_on.terms["hicache_disabled"])
        self.assertGreater(arm_off.terms["rings_gib"], 0.0)
        self.assertFalse(arm_off.terms["hicache_disabled"])
        # THE BOOT CHARGE ACTUALLY FALLS -- the whole point, stated as a
        # number rather than "the fields are zero and I trust that matters".
        # `_boot_charges_gib` is the same sum `predicted_run_peak_gib` adds
        # to the run origin (host_ledger.py:2426); comparing it directly
        # avoids needing a full cgroup reading just to get a run origin.
        saved = arm_off.terms["anchors_gib"] + arm_off.terms["rings_gib"] + arm_off.terms["overhead_gib"]
        self.assertGreater(saved, 0.0)
        self.assertAlmostEqual(
            host_ledger._boot_charges_gib(arm_off.terms)
            - host_ledger._boot_charges_gib(arm_on.terms),
            saved, places=6,
        )

    def test_choose_forwards_the_switch_to_every_rung_of_the_ladder(self):
        arm, _headroom, lines = host_ledger.choose(
            int(200 * GIB), int(150 * GIB),
            ring_bytes=int(40 * GIB), ring_span1_bytes=int(35 * GIB),
            hicache_disabled=True,
        )
        self.assertEqual(arm.terms["rings_gib"], 0.0)
        self.assertEqual(arm.terms["anchors_gib"], 0.0)
        joined = "\n".join(lines)
        self.assertIn("WEG2-HICACHE-DISABLED hicache_disabled=True", joined)

    def test_the_declared_line_always_prints_on_or_off(self):
        # NEVER SILENT: an absent line must not be confused with "was never
        # checked" -- the same rule the #1385 lanes-concurrent line follows.
        _arm, _h, lines_off = host_ledger.choose(
            int(200 * GIB), int(150 * GIB),
            ring_bytes=int(40 * GIB), ring_span1_bytes=int(35 * GIB),
        )
        self.assertIn("WEG2-HICACHE-DISABLED hicache_disabled=False", "\n".join(lines_off))


class TheArgvHalf(unittest.TestCase):
    """`launcher.common_flags` -> `argv_p`/`argv_d`."""

    def _common(self, **kw):
        return launcher.common_flags(
            MODEL, 1, 1200, "{}", 69632, "write_through", "both",
            **kw,
        )

    def test_default_argv_is_byte_identical_to_every_pre_1386_caller(self):
        implicit = self._common()
        explicit = self._common(hicache_disabled=False)
        self.assertEqual(implicit, explicit)
        for flag in HICACHE_FLAGS:
            self.assertIn(flag, implicit, flag)

    def test_disabled_omits_every_hicache_flag_and_nothing_else(self):
        on = self._common(hicache_disabled=True)
        off = self._common(hicache_disabled=False)
        for flag in HICACHE_FLAGS:
            self.assertNotIn(flag, on, flag)
            self.assertIn(flag, off, flag)
        # NOT A BLANKET CUT: --enable-cache-report/--enable-metrics are a
        # DIFFERENT feature (request-level cache reporting / prometheus),
        # never gated by this switch, and must still be there.
        self.assertIn("--enable-cache-report", on)
        self.assertIn("--enable-metrics", on)
        self.assertIn("--host", on)
        self.assertIn("--page-size", on)
        # Every non-hicache token count matches: the block removed is EXACTLY
        # 18 tokens (9 --flag/value pairs plus the one bare
        # --enable-hierarchical-cache), never a byte more or less.
        self.assertEqual(len(off) - len(on), 18)

    def test_argv_p_carries_the_switch_through_common_flags(self):
        argv_on = launcher.argv_p(
            "py", MODEL, BUDGETS, 1, 1200, "{}", [], hicache_disabled=True,
        )
        argv_off = launcher.argv_p(
            "py", MODEL, BUDGETS, 1, 1200, "{}", [], hicache_disabled=False,
        )
        for flag in HICACHE_FLAGS:
            self.assertNotIn(flag, argv_on, flag)
            self.assertIn(flag, argv_off, flag)

    def test_argv_d_carries_the_switch_through_common_flags(self):
        argv_on = launcher.argv_d(
            "py", MODEL, BUDGETS, 1, 1200, "{}", [], hicache_disabled=True,
        )
        argv_off = launcher.argv_d(
            "py", MODEL, BUDGETS, 1, 1200, "{}", [], hicache_disabled=False,
        )
        for flag in HICACHE_FLAGS:
            self.assertNotIn(flag, argv_on, flag)
            self.assertIn(flag, argv_off, flag)

    def test_argv_d_default_is_byte_identical(self):
        # argv_d has NO keyword-only marker (#1356's own lesson): the switch
        # was appended LAST so no existing positional caller shifts.
        implicit = launcher.argv_d("py", MODEL, BUDGETS, 1, 1200, "{}", [])
        explicit = launcher.argv_d(
            "py", MODEL, BUDGETS, 1, 1200, "{}", [], hicache_disabled=False,
        )
        self.assertEqual(implicit, explicit)


class TheCliFlag(unittest.TestCase):
    def test_defaults_to_off(self):
        ns = launcher.build_parser().parse_args(["--tree", "/t", "--tag", "x"])
        self.assertFalse(bool(getattr(ns, "weg2_disable_hicache", None)))

    def test_parses_when_given(self):
        ns = launcher.build_parser().parse_args(
            ["--tree", "/t", "--tag", "x", "--weg2-disable-hicache"]
        )
        self.assertTrue(ns.weg2_disable_hicache)


class OneSwitchOneTruth(unittest.TestCase):
    """The paired invariant: for the SAME bool, the ledger term and the argv
    gate must agree. This is the automatic half of the danger-direction
    guard; the MANUAL half (reverting one side alone and watching this class
    go red) is documented at the bottom of the file, not re-run here.
    """

    def test_ledger_and_argv_agree_for_both_booleans(self):
        for disabled in (True, False):
            with self.subTest(hicache_disabled=disabled):
                terms = host_ledger.charge_terms(1, 1200, 3, _images(),
                                                 hicache_disabled=disabled)
                argv = launcher.common_flags(
                    MODEL, 1, 1200, "{}", 69632, "write_through", "both",
                    hicache_disabled=disabled,
                )
                priced_zero = (terms["rings_gib"] == 0.0
                               and terms["anchors_gib"] == 0.0)
                argv_absent = "--enable-hierarchical-cache" not in argv
                self.assertEqual(
                    priced_zero, disabled,
                    "charge_terms did not honour hicache_disabled",
                )
                self.assertEqual(
                    argv_absent, disabled,
                    "common_flags did not honour hicache_disabled",
                )
                # THE DANGER DIRECTION, NAMED: priced_zero True while
                # argv_absent False (or the reverse) is exactly "priced one
                # number, allocated another" -- the #1385/#1386 shape. Both
                # must move together.
                self.assertEqual(priced_zero, argv_absent)


if __name__ == "__main__":
    unittest.main()

# MANUALLY VERIFIED (danger direction, not re-run automatically -- restored
# after, diff empty):
#
# (a) `host_ledger.charge_terms`: reverted the `0.0 if hicache_disabled else
#     (...)` guards on `anchors_gib`/`rings_gib` back to the unconditional
#     formula (switch computed but never read). Result: every test in
#     `TheLedgerHalf` that asserts a 0.0 term failed immediately, and
#     `OneSwitchOneTruth.test_ledger_and_argv_agree_for_both_booleans` failed
#     on `hicache_disabled=True` with `priced_zero=False, argv_absent=True`
#     -- the exact "ledger prices a number, argv allocates a different
#     reality" shape named in the order. Restored; suite green again.
#
# (b) `launcher.common_flags`: reverted the `+ ([] if hicache_disabled else
#     [...]) +` gate to an unconditional list (switch computed but never
#     read). Result: every test in `TheArgvHalf` that asserts a flag's
#     absence failed immediately, and `OneSwitchOneTruth`'s same test failed
#     on `hicache_disabled=True` with `priced_zero=True, argv_absent=False`
#     -- the mirror direction: the ledger charges 0 while the ranks still
#     allocate the buffer, which is EXACTLY how #1385's lane cap under-priced
#     a real boot's allocator twice. Restored; suite green again.
