# SPDX-License-Identifier: Apache-2.0
"""#1385 (Wand 11b, #1378 step 5) -- `--xchg-lanes-concurrent` caps how many of
this boot's MEASURED exchange lanes may hold their assemble buffer at once.

WHY THIS EXISTS. Wand 11b (#1377) is the open host blocker for the goal-
question boot: the bounce prices 5 lanes x 3.00 GiB + 0.75 GiB staging =
15.75 GiB on xsn31/3's own checkpoint (`front.log:111 ARM ... xchg_bounce=
15.75`, `front.log:106 WEG2-XCHG-LANES ... lanes=5 source=measured`), and the
sleep-leg cushion floor (1.50 GiB, RATE_LATCH_CUSHION_FLOOR_GIB) latches at
0.20 (W98 Weg2HostRateLatched) because that growth ate the room. The #1370
fixes made the bounce PRICED HONESTLY; they did not make it SMALLER. This
flag is the named lever: fewer lanes concurrently pinned, at the cost of
serialising the rest -- named and counted, never a silent flip-time cost.

ONE SOURCE, exactly the shape `--xchg-bounce-depth` (48adc00da4) already
proved: flag -> `xchg_bounce.resolve_lanes_concurrent` -> `BounceTerms.
lanes_concurrent` -> `_TERM_FIELDS` publication -> P/D recompute the same
total through `bounce_terms`, and `choose_host_ledger` prices the identical
number through the identical function. No second formula anywhere.

DANGER DIRECTION (the mutant the operator named): the knob is set, but the
LEDGER still prices as if every lane runs -- `lanes_priced` silently reverts
to `n_lanes` regardless of the cap. That is not merely a wasted flag: it is
the #1256 shape ("printed advisory without a reader") landing on the exact
number Wand 11b needs shrunk, so an ARM line could claim
`lanes_concurrent=2` while `xchg_bounce=15.75` never moved and the cushion
floor breaches exactly as it did on xsn31/3 -- the boot would believe the
lever was pulled when it was not.
`test_the_cap_actually_shrinks_the_charge_this_is_the_OOM_direction_guard`
is the one test written specifically against that mutant.

REFUSAL, NOT SILENT DEGRADATION, for N < 1: an explicit
`--xchg-lanes-concurrent 0` must not collapse into the SAME sentinel
`BounceTerms` uses internally for "not stated" (also 0) -- that collapse
would run every lane concurrently while the argv says the opposite.
`resolve_lanes_concurrent` is the one function both call sites (the ranks'
publication, the ledger's own pricing) resolve through, so they cannot
diverge on which values are refused. W103 is the next free number after a
census of this tree (`grep -rhoE 'W[0-9]+' python/ test/` tops out at W102);
W52/W53/W54/W56/W58 are a documented collision class (#1265/#1306) of
guessed digits, so the number is taken from the census, not memory.
"""

from __future__ import annotations

import json
import os
import struct
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import xchg_bounce as xb
from sglang.test.test_utils import CustomTestCase

G = 1024 ** 3
MIB = xb.MIB

# xsn31/3's OWN measured geometry (BOOT_weg2xsn31_0913.md, "WAND 11"):
# largest tag 2907 MiB, 128 MiB slots -> 24 slots/lane (ceil(2907/128)=23,
# +1 shadow-compare slot) -> 24 x 128 MiB = 3072 MiB/lane; 3 cards x 2 slots
# x 128 MiB staging = 768 MiB; 5 measured lanes (front.log:106).
XSN31_3 = dict(
    bytes_per_direction=29119878266, n_layers=64,
    widest_layer_bytes=756323776,       # far below the tag-sized lane buffer
    pairs=3, depth=2, slot_bytes=128 * MIB,
    n_lanes=5, max_tag_bytes=2907 * MIB,
)


def _terms(**kw):
    base = dict(XSN31_3)
    base.update(kw)
    return xb.bounce_terms(**base)


# ---------------------------------------------------------------------------
# The default: unset is byte-identical
# ---------------------------------------------------------------------------


class TheDefaultIsByteIdentical(CustomTestCase):
    def test_unset_cap_prices_every_measured_lane_exactly_as_before_1385(self):
        t = _terms()
        self.assertEqual(t.lanes_concurrent, 0, "0 is the 'not stated' sentinel")
        self.assertEqual(t.lanes_priced, t.n_lanes)
        self.assertEqual(t.lanes_serialised, 0)
        # THE ORDERED NUMBER, as a number: this is xsn31/3's own ARM line.
        self.assertAlmostEqual(t.total_bytes / G, 15.75, places=2)

    def test_the_flag_defaults_to_unset_not_to_a_literal_zero(self):
        from sglang.srt.weg2 import launcher as lc

        ns = lc.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        self.assertIsNone(
            ns.xchg_lanes_concurrent,
            "the CLI default must be None (unset), never 0 -- 0 is a "
            "distinct, refused, EXPLICIT value")
        self.assertEqual(xb.resolve_lanes_concurrent(ns.xchg_lanes_concurrent), 0)

    def test_the_flag_parses_an_explicit_value(self):
        from sglang.srt.weg2 import launcher as lc

        ns = lc.build_parser().parse_args(
            ["--tree", "/t", "--tag", "t", "--xchg-lanes-concurrent", "2"])
        self.assertEqual(ns.xchg_lanes_concurrent, 2)


# ---------------------------------------------------------------------------
# THE DANGER DIRECTION: the cap must actually move the priced total
# ---------------------------------------------------------------------------


class TheCapActuallyShrinksTheCharge(CustomTestCase):
    def test_the_cap_actually_shrinks_the_charge_this_is_the_OOM_direction_guard(self):
        """THE mutant guard: a broken wiring that still prices `n_lanes`
        lanes regardless of the cap would pass every OTHER test in this file
        (they check properties, not that the number MOVED) but fails here.

        A cap that changes what the ARM line PRINTS without changing what the
        host ledger CHARGES is the #1256 shape landing on the one number
        Wand 11b needs smaller -- the boot would believe the lever was
        pulled while the cushion floor breaches exactly as on xsn31/3.
        """
        uncapped = _terms()
        capped = _terms(lanes_concurrent=2)
        self.assertLess(
            capped.total_bytes, uncapped.total_bytes,
            "the cap must actually reduce the priced total, not merely be "
            "recorded on the term")
        self.assertEqual(capped.lanes_priced, 2)
        self.assertEqual(capped.lanes_serialised, 3)
        # THE ORDERED NUMBERS: xsn31/3's checkpoint at concurrent=2.
        self.assertAlmostEqual(capped.total_bytes / G, 6.75, places=2)
        self.assertAlmostEqual(
            (uncapped.total_bytes - capped.total_bytes) / G, 9.00, places=2,
            msg="the freed bytes at concurrent=2 against xsn31/3's own 15.75 GiB")

    def test_the_cap_at_one_lane_is_the_floor(self):
        t = _terms(lanes_concurrent=1)
        self.assertEqual(t.lanes_priced, 1)
        self.assertEqual(t.lanes_serialised, 4)
        self.assertAlmostEqual(t.total_bytes / G, 3.75, places=2)

    def test_a_cap_above_the_measured_lane_count_is_a_no_op(self):
        """A cap larger than n_lanes must never INFLATE the charge -- that
        would be a knob pricing buffers no boot of this cut ever creates,
        the same defect class as #1358's under-charge in the other
        direction."""
        t = _terms(lanes_concurrent=10)
        u = _terms()
        self.assertEqual(t.lanes_priced, u.lanes_priced)
        self.assertEqual(t.total_bytes, u.total_bytes)
        self.assertEqual(t.lanes_serialised, 0)

    def test_the_formula_is_the_same_formula_never_a_second_one(self):
        """`total_bytes` must read `lanes_priced`, not re-derive a capped
        count inline -- checked as an absence of a second `min(` expression
        in the property."""
        import inspect

        src = inspect.getsource(xb.BounceTerms.total_bytes.fget)
        self.assertIn("lanes_priced", src)
        self.assertNotIn("min(", src, (
            "total_bytes must read the ONE lanes_priced property, not "
            "re-derive its own min() -- a second formula is how #1358's "
            "3-causes-for-one-number defect reproduces"))


# ---------------------------------------------------------------------------
# Refusal, not silent degradation
# ---------------------------------------------------------------------------


class RefusalNotSilentDegradation(CustomTestCase):
    def test_none_resolves_to_the_unset_sentinel(self):
        self.assertEqual(xb.resolve_lanes_concurrent(None), 0)

    def test_an_explicit_zero_is_refused_not_folded_into_unset(self):
        with self.assertRaises(xb.Weg2XchgLanesConcurrentInvalid) as ctx:
            xb.resolve_lanes_concurrent(0)
        self.assertIn("W103", str(ctx.exception))
        self.assertIn("--xchg-lanes-concurrent", str(ctx.exception))

    def test_a_negative_value_is_refused(self):
        for bad in (-1, -5):
            with self.subTest(n=bad):
                with self.assertRaises(xb.Weg2XchgLanesConcurrentInvalid):
                    xb.resolve_lanes_concurrent(bad)

    def test_a_positive_value_passes_through_unchanged(self):
        for n in (1, 2, 5, 100):
            self.assertEqual(xb.resolve_lanes_concurrent(n), n)

    def test_w103_is_the_next_free_number_not_a_collision(self):
        """The census the operator asked for, run as an assertion.

        W52/W53/W54/W56/W58 are a documented collision class (#1265/#1306):
        digits picked from memory landing on codes already spoken for. This
        pins that W103 -- picked by grepping the tree for the current
        maximum, not by memory -- is not already the code of any OTHER
        exception in the two modules that define the neighbouring W-codes
        this feature sits beside (W71, W100, W102).
        """
        from sglang.srt.weg2 import host_ledger as hl

        other_docs = "\n".join([
            xb.Weg2XchgBounceUnderCovered.__doc__ or "",
            hl.Weg2SleepLegCushionDeficit.__doc__ or "",
            hl.Weg2XchgLanesUnmeasured.__doc__ or "",
        ])
        self.assertNotIn("W103", other_docs)
        self.assertIn("W103", xb.Weg2XchgLanesConcurrentInvalid.__doc__ or "")


# ---------------------------------------------------------------------------
# Named and counted, never silent: the flip-time cost has a log line
# ---------------------------------------------------------------------------


class NamedAndCountedNeverSilent(CustomTestCase):
    def test_the_off_case_says_off_not_a_blank_or_a_zero(self):
        line = xb.lanes_concurrent_line(_terms())
        self.assertIn("WEG2-XCHG-LANES-CONCURRENT", line)
        self.assertIn("lanes_concurrent=off", line)
        self.assertIn("lanes_total=5", line)
        self.assertIn("lanes_priced=5", line)
        self.assertIn("serialised=0", line)

    def test_the_capped_case_names_and_counts_the_serialised_lanes(self):
        line = xb.lanes_concurrent_line(_terms(lanes_concurrent=2))
        self.assertIn("lanes_concurrent=2", line)
        self.assertIn("lanes_total=5", line)
        self.assertIn("lanes_priced=2", line)
        self.assertIn("serialised=3", line)

    def test_the_arm_helper_prints_the_line(self):
        """The line reaches the boot log, not only the term."""
        import inspect

        from sglang.srt.weg2 import launcher as lc

        src = inspect.getsource(lc.xchg_bounce_terms_for_arm)
        self.assertIn("lanes_concurrent_line", src)

    def test_a_caveat_line_names_the_unwired_runtime_when_a_cap_serialises(self):
        """HONESTY GATE: this commit wires the CAP THROUGH PRICING ONLY.

        `weight_updater.py`'s per-rank leg driver still allocates a buffer
        for every lane it owns unconditionally -- nothing there yet waits
        for a concurrency permit. A boot that arms a real cap must be told
        so in its OWN log, not merely in a commit message, or the #1358
        under-charge reproduces in the other direction: a smaller ARM number
        trusted as the real peak.
        """
        import inspect

        from sglang.srt.weg2 import launcher as lc

        src = inspect.getsource(lc.xchg_bounce_terms_for_arm)
        self.assertIn("lanes_serialised", src)
        self.assertIn("runtime_enforcement=NOT_WIRED", src)
        self.assertIn("NAMED HERE,", src)
        self.assertIn("NOT BUILT", src)


# ---------------------------------------------------------------------------
# ONE SOURCE: the flag reaches the ledger AND the ranks through ONE resolve
# ---------------------------------------------------------------------------


def _write_shard(path, tensors):
    """A REAL safetensors file, as `test_weg2_widest_layer_census_1332b.py`
    writes it -- duplicated rather than imported so this file stays a single
    unit with no cross-file fixture coupling."""
    header, off = {}, 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    pad = (-len(blob)) % 8
    blob += b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * off)


class TheCaveatLineIsExecutionSmoked(CustomTestCase):
    """Not just a source-string check: the line actually appears (or does
    not) through the real `xchg_bounce_terms_for_arm` call, end to end."""

    def test_no_cap_no_caveat(self):
        pytest_ckpt = self._ckpt()
        from sglang.srt.weg2 import launcher as lc

        lc.publish_weight_chunk_layers(2)
        _charged, lines = lc.xchg_bounce_terms_for_arm(
            "shadow", "host", str(pytest_ckpt), n_lanes=5)
        caveats = [ln for ln in lines if "LANES-CONCURRENT-CAVEAT" in ln]
        self.assertEqual(caveats, [], lines)

    def test_a_real_cap_prints_the_caveat_with_the_right_numbers(self):
        pytest_ckpt = self._ckpt()
        from sglang.srt.weg2 import launcher as lc

        lc.publish_weight_chunk_layers(2)
        _charged, lines = lc.xchg_bounce_terms_for_arm(
            "shadow", "host", str(pytest_ckpt), n_lanes=5, lanes_concurrent=2)
        caveats = [ln for ln in lines if "LANES-CONCURRENT-CAVEAT" in ln]
        self.assertEqual(len(caveats), 1, lines)
        self.assertIn("NOT_WIRED", caveats[0])
        self.assertIn("lanes_priced=2", caveats[0])
        capped_line = [ln for ln in lines if ln.startswith(
            "WEG2-XCHG-LANES-CONCURRENT ")][0]
        self.assertIn("lanes_concurrent=2", capped_line)
        self.assertIn("serialised=3", capped_line)

    @staticmethod
    def _ckpt():
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp()) / "ckpt"
        d.mkdir()
        _write_shard(d / "model-00001-of-00001.safetensors", {
            "model.layers.0.mlp.gate_proj.weight": ("BF16", (1024, 1024), 1 << 20),
            "model.layers.1.mlp.gate_proj.weight": ("BF16", (1024, 1024), 1 << 21),
            "model.layers.2.mlp.gate_proj.weight": ("BF16", (1024, 1024), 1 << 20),
        })
        return d


class OneSourceReachesBothSides(CustomTestCase):
    def test_the_cap_rides_with_the_inputs_and_round_trips(self):
        """A rank must recompute the capped total, never inherit one it
        cannot check -- the #1358 rule this tuple already exists for."""
        self.assertIn("lanes_concurrent", xb._TERM_FIELDS)
        for cap in (0, 1, 2, 5, 10):
            with self.subTest(lanes_concurrent=cap):
                t = _terms(lanes_concurrent=cap)
                back = xb.read_published_terms(xb.publish_terms(t))
                self.assertEqual(int(back.lanes_concurrent), cap)
                self.assertEqual(int(back.total_bytes), int(t.total_bytes))
                self.assertEqual(int(back.lanes_priced), int(t.lanes_priced))

    def test_choose_host_ledger_and_the_arm_helper_both_take_the_cap(self):
        import inspect

        from sglang.srt.weg2 import launcher as lc

        self.assertIn("lanes_concurrent",
                      inspect.signature(lc.choose_host_ledger).parameters)
        self.assertIn("lanes_concurrent",
                      inspect.signature(lc.xchg_bounce_terms_for_arm).parameters)
        self.assertIn("lanes_concurrent",
                      inspect.signature(lc.checkpoint_census.widest_layer_terms
                                        ).parameters)

    def test_choose_host_ledger_forwards_the_SAME_value_it_receives(self):
        import inspect

        src = inspect.getsource(
            __import__("sglang.srt.weg2.launcher",
                       fromlist=["choose_host_ledger"]).choose_host_ledger)
        i = src.index("xchg_bounce_terms_for_arm(")
        self.assertIn("lanes_concurrent", src[i:i + 200])

    def test_main_resolves_the_flag_exactly_once_before_either_call_site(self):
        """The ONE resolve: both the ranks' publication call and the
        ledger's own call must read the SAME resolved local, never re-read
        `ns.xchg_lanes_concurrent` (and re-validate it) twice -- two
        validations of one flag is how the two sides can disagree about
        which values are refused."""
        import inspect

        from sglang.srt.weg2 import launcher as lc

        src = inspect.getsource(lc.main)
        self.assertEqual(
            src.count("resolve_lanes_concurrent("), 1,
            "the flag must be resolved exactly once in main()")


# ---------------------------------------------------------------------------
# The ledger sees the smaller number (#1256 requirement: not a printed
# advisory without a reader)
# ---------------------------------------------------------------------------


class TheLedgerSeesTheSmallerCharge(CustomTestCase):
    def test_charge_terms_reflects_the_capped_total_not_the_uncapped_one(self):
        from sglang.srt.weg2 import host_ledger as hl

        uncapped = _terms()
        capped = _terms(lanes_concurrent=2)
        img = hl.ImageTerms(
            p_gib=38.63, d_gib=38.63, p_source="s", d_source="s",
            p_measured=True, d_measured=True, extra_p_gib=0.0, extra_d_gib=0.0,
        )
        charged_uncapped = hl.charge_terms(
            1, 600, 3, img, s_gb_d=4, xchg_bounce_host_bytes=uncapped.total_bytes)
        charged_capped = hl.charge_terms(
            1, 600, 3, img, s_gb_d=4, xchg_bounce_host_bytes=capped.total_bytes)
        self.assertLess(
            charged_capped["xchg_bounce_gib"], charged_uncapped["xchg_bounce_gib"],
            "the ledger must SEE the smaller bounce, not merely the flag")
        self.assertAlmostEqual(
            charged_uncapped["xchg_bounce_gib"] - charged_capped["xchg_bounce_gib"],
            9.00, places=2)


if __name__ == "__main__":
    unittest.main()
