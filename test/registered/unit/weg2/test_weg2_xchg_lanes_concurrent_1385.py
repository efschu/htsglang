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

STEP 2 (coordinator order, same branch, stacked on 45477c1745): step 1 wired
the cap through PRICING only, and the boot record's own honesty gate
(WEG2-XCHG-LANES-CONCURRENT-CAVEAT) said so as `runtime_enforcement=
NOT_WIRED` -- a ledger that charges LESS than the runtime actually pins
funds a boot that then dies at the host, the #1358 under-charge reproduced
in the other direction. This file's second half tests the RUNTIME half:
`weight_exchange_transport.LanePermit`, a boot-wide POSIX counting
semaphore the launcher creates (`weight_exchange_region.
create_lane_permit_semaphore`, sized to `lanes_priced`, never a second
count) and `weight_updater.py`'s lane loop acquires before each lane's
buffer and releases only after that lane's `run_bounce_leg` -- and so its
own `finally: bounce.close()` -- has returned or raised. The caveat line now
reads `runtime_enforcement=WIRED` for exactly the same reason it used to
read the opposite: it is a mechanical consequence of the code, checked here,
not a claim typed once and left behind the next time the runtime changes.

MUTANTS FOR THIS STEP (coordinator-named danger directions):
  M1  cap set, buffers still all held at once  -> the semaphore's own
      concurrency bound, proven with real threads and a real POSIX
      semaphore (`test_at_most_N_permits_are_ever_held_at_once`)
  M2  release before use instead of after      -> source-order check that
      `.release()` sits in a `finally` AFTER the `run_bounce_leg` call, and
      an execution check that a held permit is NOT available until release
      actually runs (`test_the_permit_is_unavailable_until_release_not_
      before`)
  M3  counter reports serialised without real serialisation -> the launcher
      creates the permit with EXACTLY `lanes_priced` slots (never `n_lanes`,
      never a hand count), checked by source and by an execution smoke
      through the real launcher function

STEP 3 (coordinator order, boot weg2xsn31/6, same branch, stacked on
461fdebca0): xsn31/6 BOOT-PROVED step 1's size fix (every observed lane
measured exactly 3,221,225,472 B = 128 MiB x 24, both instruments
deckungsgleich) -- but ALSO measured a THIRD instance of the same class,
this time in step 2's own runtime enforcement: with `--xchg-lanes-
concurrent 1`, up to FOUR lane files (c0+p1+p2+p4, 12.00 GiB) coexisted on
tmpfs at once, against a promised 3.00 GiB.

TRACED (not merely observed): the coordinator's own hypothesis --
"the semaphore caps within one group, not across P and D" -- does NOT hold
under source inspection. `create_lane_permit_semaphore` is called exactly
ONCE in `launcher.main`, keyed on `str(ring_plan.epoch)`, and
`prepare_xchg_env`'s returned `xchg_env` (built from that SAME boot_nonce)
is merged into BOTH groups' environment through the SAME `build_env(...,
xchg_env=xchg_env)` call (`group="P"` and `group="D"`, three call sites, all
passing the identical dict) -- so both `_weg2_xchg_deposit_before_sleep`
(D, hook=source) and `_weg2_xchg_inject_from_peer` (P, hook=authoritative)
read the identical `SGLANG_WEG2_XCHG_BOOT` value and therefore open the
IDENTICAL semaphore name. The permit genuinely is boot-wide, and
`test_at_most_N_permits_are_ever_held_at_once` above already proves the
PRIMITIVE enforces a real cross-process bound (POSIX named semaphores do
not care which process opens them).

THE REAL GAP: `LayerBounce.close()` deliberately never unlinks (module
docstring: cross-process store-and-forward needs the file to outlive
either side's own close). Step 2 released the PERMIT once `run_bounce_leg`
returned, which correctly let the NEXT lane start PINNING -- but the
PREVIOUS lane's FILE stayed on tmpfs, uncounted by the semaphore and
unremoved until full boot teardown. A concurrency cap that limits how many
buffers may be PINNED at any instant is not the same claim as a cap on how
many ACCUMULATE unfreed over one flip; by the time a flip has touched every
lane once, the two converge to the SAME uncapped total regardless of the
cap -- which is exactly consistent with "up to four files, growing as more
lanes were touched" and requires no cross-group semaphore failure to
explain.

THE FIX: `weight_exchange_bounce.unlink_lane_buffer`, called from the
COLLECTOR side (the side `CrossSlotRendezvous.post_drained` already tells
"this tag is out of the buffer, the depositor may reuse it") -- the same
moment already proves both sides' own file descriptors are closed, so
unlinking here can never race a reader. Gated on `_lane_permit_active`
exactly like the permit itself, so the unset/default arm -- whose own
`BounceTerms.total_bytes` formula already charges for every lane
accumulating, by design (`n_lanes` IS that accumulated count) -- is
untouched.
"""

from __future__ import annotations

import json
import os
import struct
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_bounce as bx
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

    def test_a_caveat_line_names_the_wired_runtime_when_a_cap_serialises(self):
        """HONESTY GATE, STEP 2: the caveat now says WIRED, because it is.

        The line must never claim more than the code does (coordinator
        order: "sie darf nie NOT_WIRED sagen, waehrend sie durchsetzt, und
        nie WIRED, waehrend sie es nicht tut"). This checks the STATIC half
        of that promise -- the string itself, and that the old NOT_WIRED
        claim is gone rather than left to coexist with the new one; the
        DYNAMIC half (the permit really is acquired/released) is
        `test_at_most_N_permits_are_ever_held_at_once` and the source-order
        checks in `RuntimeEnforcementIsWiredCorrectly` below.
        """
        import inspect

        from sglang.srt.weg2 import launcher as lc

        src = inspect.getsource(lc.xchg_bounce_terms_for_arm)
        self.assertIn("lanes_serialised", src)
        self.assertIn("runtime_enforcement=WIRED", src)
        self.assertNotIn("runtime_enforcement=NOT_WIRED", src, (
            "the old EMITTED claim must not coexist with the new one -- a "
            "caveat that could print either string depending on nothing "
            "real would be the same printed-advisory-without-a-reader "
            "shape (#1256) this whole feature exists to close")
        )


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
        self.assertIn("runtime_enforcement=WIRED", caveats[0])
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


# ---------------------------------------------------------------------------
# STEP 2: the RUNTIME half. A real POSIX semaphore, real threads, real
# concurrency proof -- not only a source-string check.
# ---------------------------------------------------------------------------


def _fresh_nonce(tag: str) -> str:
    """A boot nonce unique to this test process and this test, so parallel
    test runs (or a leftover from a previous crashed run) cannot collide on
    the same semaphore name."""
    import uuid

    return f"test1385-{tag}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


class TheLanePermitPrimitive(CustomTestCase):
    """`weight_exchange_region.create_lane_permit_semaphore` +
    `weight_exchange_transport.LanePermit`, exercised as REAL POSIX IPC --
    the same `/dev/shm` namespace a boot uses, not a mock of it. Every test
    unlinks its own name in `finally`, mirroring the launcher's own
    create-then-unlink discipline."""

    def test_create_refuses_permits_below_one(self):
        from sglang.srt.weg2 import weight_exchange_region as xr

        nonce = _fresh_nonce("refuse")
        for bad in (0, -1, -5):
            with self.subTest(permits=bad):
                with self.assertRaises(ValueError):
                    xr.create_lane_permit_semaphore(nonce, bad)

    def test_round_trip_create_open_acquire_release_unlink(self):
        from sglang.srt.weg2 import weight_exchange_region as xr
        from sglang.srt.weg2 import weight_exchange_transport as tp

        nonce = _fresh_nonce("roundtrip")
        xr.create_lane_permit_semaphore(nonce, 2)
        try:
            permit = tp.LanePermit(nonce)
            try:
                permit.acquire(budget_s=5.0, lane="p0", boot=nonce)
                permit.acquire(budget_s=5.0, lane="p1", boot=nonce)
                permit.release()
                permit.release()
            finally:
                permit.close()
        finally:
            self.assertTrue(xr.unlink_lane_permit_semaphore(nonce))

    def test_unlink_is_idempotent_like_the_24_name_census(self):
        from sglang.srt.weg2 import weight_exchange_region as xr

        nonce = _fresh_nonce("idempotent")
        xr.create_lane_permit_semaphore(nonce, 1)
        self.assertTrue(xr.unlink_lane_permit_semaphore(nonce))
        self.assertFalse(xr.unlink_lane_permit_semaphore(nonce),
                         "an absent name must not be an error")

    def test_a_rank_can_never_create_the_semaphore_by_opening_it(self):
        """SemSet's own rule, reused: an ENOENT here means the launcher never
        armed a cap, and it must say so rather than silently creating one
        with an unknown count (POSIX ignores the value on an existing name,
        so a silent adopt-or-create would be invisible)."""
        from sglang.srt.weg2 import weight_exchange_transport as tp

        nonce = _fresh_nonce("never-armed")
        permit = tp.LanePermit(nonce)
        with self.assertRaises(OSError):
            permit.acquire(budget_s=1.0)

    def test_a_wait_beyond_the_permit_count_is_bounded_and_named_W69(self):
        """M2's first half: a permit that cannot be granted refuses by name,
        in bounded time -- never hangs, never silently proceeds."""
        import time as _time

        from sglang.srt.weg2 import weight_exchange_region as xr
        from sglang.srt.weg2 import weight_exchange_transport as tp

        nonce = _fresh_nonce("timeout")
        xr.create_lane_permit_semaphore(nonce, 1)
        holder = tp.LanePermit(nonce)
        waiter = tp.LanePermit(nonce)
        try:
            holder.acquire(budget_s=5.0, lane="p0", boot=nonce)
            budget = 0.4
            t0 = _time.monotonic()
            with self.assertRaises(tp.Weg2XchgGateTimeout) as ctx:
                waiter.acquire(budget_s=budget, lane="p1", boot=nonce)
            elapsed = _time.monotonic() - t0
            self.assertGreaterEqual(elapsed, budget * 0.9,
                                    "must actually wait, not fail instantly")
            self.assertLess(elapsed, budget + 5.0,
                            "must not wait past its own budget -- an "
                            "unbounded waiter is the worse failure")
            self.assertIn("W69 Weg2XchgGateTimeout", str(ctx.exception))
            self.assertIn("lane=p1", str(ctx.exception))
            # AFTER the holder releases, the SAME waiter (or a new one)
            # succeeds -- proving the earlier refusal was a real timeout on
            # a real count, not a permanently broken semaphore.
            holder.release()
            waiter.acquire(budget_s=5.0, lane="p1", boot=nonce)
            waiter.release()
        finally:
            holder.close()
            waiter.close()
            xr.unlink_lane_permit_semaphore(nonce)

    def test_at_most_N_permits_are_ever_held_at_once(self):
        """M1, THE CORE MUTANT GUARD: 'cap set but buffers still all held
        concurrently'. Real threads, a real semaphore, a real shared counter
        -- if acquire/release were ever no-ops (the mutant), every thread
        would enter its critical section at once and this assertion catches
        it directly rather than inferring it from timing.
        """
        import threading
        import time as _time

        from sglang.srt.weg2 import weight_exchange_region as xr
        from sglang.srt.weg2 import weight_exchange_transport as tp

        nonce = _fresh_nonce("concurrency-bound")
        PERMITS = 2
        WORKERS = 6
        xr.create_lane_permit_semaphore(nonce, PERMITS)
        lock = threading.Lock()
        state = {"inside": 0, "peak": 0, "entries": 0}

        def worker():
            permit = tp.LanePermit(nonce)
            try:
                permit.acquire(budget_s=10.0, lane="w", boot=nonce)
                try:
                    with lock:
                        state["inside"] += 1
                        state["entries"] += 1
                        state["peak"] = max(state["peak"], state["inside"])
                    _time.sleep(0.05)
                finally:
                    with lock:
                        state["inside"] -= 1
                permit.release()
            finally:
                permit.close()

        threads = [threading.Thread(target=worker) for _ in range(WORKERS)]
        try:
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30.0)
                self.assertFalse(t.is_alive(), "a worker hung -- the "
                                 "unbounded-wait class this feature exists "
                                 "to refuse rather than reproduce")
        finally:
            xr.unlink_lane_permit_semaphore(nonce)
        self.assertEqual(state["entries"], WORKERS,
                         "every worker must have actually run")
        self.assertLessEqual(state["peak"], PERMITS,
                             f"peak concurrency {state['peak']} exceeded the "
                             f"{PERMITS} permits -- M1: the cap did not cap")
        self.assertGreaterEqual(
            state["peak"], 1, "the instrument itself must be live")


class RuntimeEnforcementIsWiredCorrectly(CustomTestCase):
    """Structural proof, over the REAL product source, that the permit is
    acquired around exactly the buffer's own lifetime -- the same
    source-order style `test_the_refusal_is_raised_before_the_first_
    allocation` (test_weg2_bounce_perlane_1358.py) already uses for this
    exact class of claim, because a full multi-rank GPU boot is not
    available to a hermetic desk test."""

    def test_the_permit_object_is_created_only_when_a_real_cap_is_active(self):
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu)
        i = src.index("_lane_permit_active = (")
        window = src[i:i + 4000]
        self.assertIn("lanes_concurrent", window)
        self.assertIn(
            "tp.LanePermit(str(boot_nonce)) if _lane_permit_active else None",
            window)

    def test_acquire_precedes_run_bounce_leg_precedes_release(self):
        """M2, the ordering half: acquire before the buffer, release after
        it closes -- never the reverse, never both before."""
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu)
        base = src.index("_lane_permit_active = (")
        acquire_i = src.index("_lane_permit.acquire(", base)
        run_i = src.index("bx.run_bounce_leg(", acquire_i)
        release_i = src.index("_lane_permit.release()", run_i)
        self.assertLess(acquire_i, run_i,
                        "the permit must be acquired BEFORE the buffer")
        self.assertLess(run_i, release_i,
                        "the permit must be released AFTER the buffer's own "
                        "leg (and so its close()) has returned or raised")

    def test_the_acquire_call_is_itself_conditioned_on_the_active_flag(self):
        """A structural guard against a mutant that keeps the semaphore
        object gated but calls `.acquire()` unconditionally (which would
        crash on the default path, but only AFTER a syscall the flag
        promises never happens)."""
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu)
        base = src.index("_lane_permit_active = (")
        acquire_stmt = src.index("_lane_permit.acquire(", base)
        preceding = src[max(0, acquire_stmt - 200):acquire_stmt]
        self.assertIn("if _lane_permit_active:", preceding)

    def test_release_lives_inside_a_finally_block(self):
        """M2's other half: a release that only runs on the happy path is
        the leak this whole feature exists to bound -- `run_bounce_leg`
        itself already earns this property for the buffer
        (`bounce.close()` sits in ITS OWN finally); the permit must match
        it or a raising lane holds its permit forever, exactly the
        unbounded-wait class named throughout this file."""
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu)
        base = src.index("_lane_permit_active = (")
        release_i = src.index("_lane_permit.release()", base)
        preceding = src[max(0, release_i - 300):release_i]
        self.assertIn("finally:", preceding)


class TheLauncherPricesTheOnlyPermitCount(CustomTestCase):
    """M3: the launcher creates the semaphore with EXACTLY `lanes_priced`,
    the SAME number the ledger just charged with -- never `n_lanes`, never a
    hand count, never a second computation of the cap."""

    def test_the_launcher_creates_with_lanes_priced_not_a_second_count(self):
        import inspect

        from sglang.srt.weg2 import launcher as lc

        src = inspect.getsource(lc.main)
        i = src.index("create_lane_permit_semaphore(")
        window = src[max(0, i - 200):i + 200]
        self.assertIn("bounce_terms_for_ranks.lanes_priced", window)
        self.assertNotIn("_pub_lane_n)", window)

    def test_the_launcher_only_creates_it_when_a_cap_is_requested(self):
        import inspect

        from sglang.srt.weg2 import launcher as lc

        src = inspect.getsource(lc.main)
        i = src.index("create_lane_permit_semaphore(")
        preceding = src[max(0, i - 1500):i]
        self.assertIn("if int(_lanes_concurrent) > 0:", preceding)

    def test_the_permit_name_shares_the_region_prefix_so_the_existing_sweep_cleans_it(self):
        """No second teardown path: naming this inside `REGION_PREFIX` means
        the launch-time residue sweep (`sweep_xchg_semaphores`, prefix-
        matched on `sem.<REGION_PREFIX>`) already catches a crashed boot's
        leftover permit exactly as it catches the other 36 names."""
        from sglang.srt.weg2 import weight_exchange_region as xr

        name = xr.lane_permit_sem_name("abc123")
        self.assertTrue(name.startswith(f"/{xr.REGION_PREFIX}"))
        # And it must never collide with a cross (`-<digit>-<digit>-<digit>-
        # {kind}`) or diagonal (`-card<n>-<digit>-{kind}`) name.
        self.assertNotIn("-card", name)
        for pair in range(xr.N_PAIRS):
            for slot in range(xr.SLOTS_PER_PAIR):
                for kind in ("empty", "full"):
                    self.assertNotEqual(
                        name, xr.sem_name("abc123", pair, slot, kind))


# ---------------------------------------------------------------------------
# STEP 3: the collector must FREE the lane's file, not only its permit, or
# the cap serialises PINNING while the FILES still accumulate. Real deposit,
# real collect, real bytes -- through the product's own
# `_weg2_xchg_bounce_leg` call site, twice (once per side, matching the two
# real processes a boot actually runs).
# ---------------------------------------------------------------------------


class TheCollectorFreesTheFileNotOnlyThePermit(CustomTestCase):
    """xsn31/6's own finding, reproduced and fixed at the desk: 3 lanes
    (one diagonal, two cross), cap=1, deposited then collected -- and after
    the round trip every file must be gone, while the SAME round trip
    WITHOUT a cap must leave them exactly as before (byte-identical
    default)."""

    #: A small-scale Option-1 geometry sharing the same STRUCTURAL relation
    #: as production (oncard slot far smaller than one layer, tag spanning
    #: more than one layer) -- reused from
    #: test_weg2_xchg_leg_slot_bytes_1385.py's own choice for the identical
    #: reason: fast, hermetic, no gigabyte allocations.
    SLOT_BYTES = 512

    def _terms(self, cap):
        from .test_weg2_xchg_bounce_execution_smoke_1273 import (
            DEPTH,
            LAYER0_BYTES,
            N_LAYERS,
            PLAIN_LAYER_BYTES,
        )

        kw = dict(n_lanes=3)   # THIS RANK's own lane count (one diagonal,
                               # two cross -- see _all_descs' own N_DST=3):
                               # the per-rank guard (#1358) refuses a term
                               # priced for fewer than the rank enumerates.
        if cap:
            kw["lanes_concurrent"] = cap
        return xb.bounce_terms(
            bytes_per_direction=PLAIN_LAYER_BYTES * N_LAYERS, n_layers=N_LAYERS,
            widest_layer_bytes=LAYER0_BYTES, pairs=6, depth=DEPTH,
            slot_bytes=self.SLOT_BYTES,
            max_tag_bytes=LAYER0_BYTES * 2, **kw,
        )

    def _round_trip(self, *, cap):
        """Deposit ALL of this rank's lanes for ONE tag, then collect them
        -- two separate calls, matching D's and P's separate processes --
        and return (terms, lane_paths, mismatched_rows)."""
        import tempfile

        from sglang.srt.weg2 import weight_exchange_region as xr
        from sglang.srt.weg2 import weight_exchange_transport as tp

        from .test_weg2_xchg_bounce_execution_smoke_1273 import (
            TAG,
            _all_descs,
            _manager,
            _mismatched_rows,
            _seed_source,
            as_single_hook_descs,
        )
        from .test_weg2_xchg_transport_1273 import FakeDeviceOps, _fresh_boot

        root = tempfile.mkdtemp()
        nonce = _fresh_boot()
        xr.create_semaphores(nonce)
        try:
            terms = self._terms(cap)
            if cap:
                xr.create_lane_permit_semaphore(nonce, terms.lanes_priced)
            try:
                descs_all = _all_descs()
                src_descs = [d for d in as_single_hook_descs(descs_all, is_source=True)
                            if d.tag == TAG]
                dst_descs = [d for d in as_single_hook_descs(descs_all, is_source=False)
                            if d.tag == TAG]
                ops = FakeDeviceOps(root, 0)
                _seed_source(ops)
                mgr = _manager()
                mgr._weg2_xchg_bounce_leg(
                    descs=src_descs, ops=ops, boot_nonce=nonce, terms=terms,
                    mode=wx.INJECT_AUTHORITATIVE, device=0, hook="source",
                    sems=tp.SemSet(nonce), tag=TAG, shm_root=root,
                )
                paths_after_deposit = {
                    lane: bx.bounce_path(nonce, root, lane)
                    for lane in ("c0", "p0", "p1")
                }
                exists_after_deposit = {
                    lane: os.path.exists(p) for lane, p in paths_after_deposit.items()
                }
                mgr._weg2_xchg_bounce_leg(
                    descs=dst_descs, ops=ops, boot_nonce=nonce, terms=terms,
                    mode=wx.INJECT_AUTHORITATIVE, device=0, hook="authoritative",
                    sems=tp.SemSet(nonce), tag=TAG, shm_root=root,
                )
                exists_after_collect = {
                    lane: os.path.exists(p) for lane, p in paths_after_deposit.items()
                }
                mismatched = _mismatched_rows(ops, descs_all)
            finally:
                if cap:
                    xr.unlink_lane_permit_semaphore(nonce)
        finally:
            xr.unlink_semaphores(nonce)
        return terms, exists_after_deposit, exists_after_collect, mismatched

    def test_with_a_cap_every_lane_file_is_gone_after_the_collector_is_done(self):
        terms, after_deposit, after_collect, mismatched = self._round_trip(cap=1)
        self.assertEqual(terms.lanes_priced, 1)
        # All three of THIS rank's lanes were deposited (the deposit side
        # does not know about the cap's runtime enforcement -- it is the
        # collector's job to free what it just finished with).
        self.assertTrue(all(after_deposit.values()), after_deposit)
        # xsn31/6's own regression: NONE may survive the collector.
        self.assertFalse(any(after_collect.values()),
                         f"a lane file survived its own collector: {after_collect}")
        self.assertEqual(mismatched, [], "the fix must not cost correctness")

    def test_without_a_cap_the_default_arm_is_byte_identical_files_persist(self):
        """The other half of the promise: `_lane_permit_active` gates the
        unlink exactly as it gates the permit, so a boot that never set
        `--xchg-lanes-concurrent` keeps its pre-existing behaviour (files
        outlive the collector, exactly as `BounceTerms.total_bytes`'s own
        `n_lanes`-wide charge already assumes)."""
        _terms, after_deposit, after_collect, mismatched = self._round_trip(cap=0)
        self.assertTrue(all(after_deposit.values()), after_deposit)
        self.assertTrue(all(after_collect.values()),
                        "the default arm must be UNCHANGED by this fix: "
                        f"a file disappeared with no cap set: {after_collect}")
        self.assertEqual(mismatched, [])

    def test_unlink_lane_buffer_is_idempotent(self):
        import tempfile

        root = tempfile.mkdtemp()
        nonce = "idem-test"
        path = bx.bounce_path(nonce, root, "p0")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"\0" * 16)
        self.assertTrue(bx.unlink_lane_buffer(nonce, root, "p0"))
        self.assertFalse(os.path.exists(path))
        self.assertFalse(bx.unlink_lane_buffer(nonce, root, "p0"),
                         "an absent file is not an error")


class TheUnlinkIsWiredAtExactlyTheRightPoint(CustomTestCase):
    """Structural proof, over the real product source: the unlink call sits
    where `post_drained` already sits (the collector's own "this tag is out
    of the buffer" moment), gated by the same `_lane_permit_active` boolean
    the permit itself uses -- never a second, independent condition that
    could drift from it."""

    def test_unlink_lane_buffer_is_called_after_post_drained(self):
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu)
        i = src.index("rv.post_drained(tag=str(tag))")
        j = src.index("bx.unlink_lane_buffer(", i)
        self.assertLess(i, j)
        self.assertLess(j - i, 2000,
                        "the unlink must be the very next thing this "
                        "collector does with the tag, not an unrelated "
                        "later call that happens to match")

    def test_the_unlink_is_gated_on_the_same_boolean_as_the_permit(self):
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu)
        i = src.index("rv.post_drained(tag=str(tag))")
        window = src[i:i + 2000]
        self.assertIn("if _lane_permit_active:", window)
        self.assertIn("bx.unlink_lane_buffer(boot_nonce, root, _lane_key)",
                      window)

    def test_the_unlink_only_ever_runs_on_the_collect_side(self):
        """`post_drained` (and so the unlink beside it) is reached only
        inside `if tag is not None and phase == bx.PHASE_COLLECT:` -- the
        depositor has no way to know when a cross-process peer has finished
        reading, so it must never be the one that removes the file."""
        import inspect

        from sglang.srt.managers.scheduler_components import weight_updater as wu

        src = inspect.getsource(wu)
        i = src.index("if tag is not None and phase == bx.PHASE_COLLECT:\n"
                      "                        rv.post_drained(tag=str(tag))")
        j = src.index("bx.unlink_lane_buffer(", i)
        self.assertLess(i, j)
        self.assertLess(j - i, 2200)


if __name__ == "__main__":
    unittest.main()
