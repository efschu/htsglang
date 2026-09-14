# SPDX-License-Identifier: Apache-2.0
"""#1369 -- the host ring's off-switch, PAKET A (launcher.py, host_ledger.py,
weight_exchange.py only; weight_updater.py/weg2_memory_saver.py/model_runner.py
/tms_csrc are packages B/C, untouched here).

User order 2026-09-14, verbatim: "DIE 48GB MUESSEN WEG. UND ZWAR PRONTO" and,
against the "the ring is a fallback net" argument: "auf der Festplatte liegt
ein Snapshot. Das ist auch ein Rueckfall. Aber wenn es korrekt implementiert
ist braucht es NIEMALS einen Rueckfall....!"

THE 46.40 GiB HOST RING (#1369) backed the WEIGHTS region unconditionally,
both groups, under all three weight-source arms -- including under an ARMED,
AUTHORITATIVE exchange, where the exchange itself is already the weight
source at the wake seam and the ring refills bytes nobody reads.

THREE LAYERS, each with its own test class below, because each is where the
old, wrong version of this ticket's own first commit (7ca55a35a0, corrected
by a033f2926a) could still have failed silently:

1. weight_exchange.weights_cpu_backup_armed() -- the CONTRACT. auto/on/off,
   explicit vs. environment, W107 on an unrecognised value.
2. launcher.common_flags/argv_p/argv_d -- the ARGV LEVER
   (--enable-weights-cpu-backup's own conditional presence).
3. launcher.weg2_weights_cpu_backup_ring_kw + host_ledger.choose/price -- the
   LEDGER'S OWN VIEW (Sigma H zeroed together with the declaration, never
   one without the other, and the printed Gegenprobe line).

PRIOR-ART, EXTENDED NOT REBUILT: launcher.ring_absent_by_design (B4f, boot
weg2xsn13) already answers "is the ring absent by design" for the
weight_source/inject_mode axis; #1369 does not replace it, it adds the
operator's own on/off/auto override ON TOP, via ONE function
(weg2_weights_cpu_backup_ring_kw) so the byte counts and the declaration can
never drift apart again -- exactly the #1358 defect class ("2.16 GiB" had
three causes) this order named by name.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import weight_exchange as wx
from sglang.test.test_utils import CustomTestCase


def _reset_env():
    for k in (wx.WEIGHT_SOURCE_ENV, wx.INJECT_ENV, wx.WEIGHTS_CPU_BACKUP_ENV):
        os.environ.pop(k, None)


class TheContractAutoMatchesRingAbsentByDesign(CustomTestCase):
    """weight_exchange.weights_cpu_backup_armed(): auto/on/off, explicit vs env."""

    def setUp(self):
        _reset_env()

    def tearDown(self):
        _reset_env()

    def test_ring_auto_backs_up(self):
        self.assertTrue(wx.weights_cpu_backup_armed())

    def test_exchange_plus_shadow_auto_STILL_backs_up(self):
        # THE CASE THE FIRST COMMIT (7ca55a35a0) GOT WRONG: under the DEFAULT
        # inject mode, the refill is still the authority and the ring is the
        # shadow leg's ground truth to grade against -- a boot armed this way
        # that dropped the ring would delete evidence, not a fallback.
        os.environ[wx.WEIGHT_SOURCE_ENV] = wx.WEIGHT_SOURCE_EXCHANGE
        self.assertTrue(wx.weights_cpu_backup_armed())

    def test_exchange_plus_authoritative_auto_drops_the_ring(self):
        os.environ[wx.WEIGHT_SOURCE_ENV] = wx.WEIGHT_SOURCE_EXCHANGE
        os.environ[wx.INJECT_ENV] = wx.INJECT_AUTHORITATIVE
        self.assertFalse(wx.weights_cpu_backup_armed())

    def test_explicit_on_wins_regardless_of_arm(self):
        os.environ[wx.WEIGHT_SOURCE_ENV] = wx.WEIGHT_SOURCE_EXCHANGE
        os.environ[wx.INJECT_ENV] = wx.INJECT_AUTHORITATIVE
        self.assertTrue(wx.weights_cpu_backup_armed(explicit="on"))

    def test_explicit_off_wins_regardless_of_arm_including_plain_ring(self):
        self.assertFalse(wx.weights_cpu_backup_armed(explicit="off"))

    def test_env_only_path_no_explicit_resolves_on(self):
        os.environ[wx.WEIGHTS_CPU_BACKUP_ENV] = "on"
        self.assertTrue(wx.weights_cpu_backup_armed())

    def test_env_only_path_no_explicit_resolves_off(self):
        os.environ[wx.WEIGHTS_CPU_BACKUP_ENV] = "off"
        self.assertFalse(wx.weights_cpu_backup_armed())

    def test_unrecognised_value_refuses_by_name_W107(self):
        with self.assertRaises(wx.Weg2WeightsCpuBackupModeUnknown) as ctx:
            wx.weights_cpu_backup_armed(explicit="maybe")
        self.assertIn("W107", str(ctx.exception))

    def test_empty_explicit_falls_through_to_environment(self):
        os.environ[wx.WEIGHTS_CPU_BACKUP_ENV] = "off"
        self.assertFalse(wx.weights_cpu_backup_armed(explicit=""))


class TheArgvLeverDropsOnlyItsOwnFlag(CustomTestCase):
    """common_flags/argv_p/argv_d: byte-identical default, clean diff when off."""

    def test_common_flags_default_is_byte_identical(self):
        default = L.common_flags("model", 1, 1, "{}", 100)
        explicit_true = L.common_flags("model", 1, 1, "{}", 100,
                                       weights_cpu_backup=True)
        self.assertEqual(default, explicit_true)
        self.assertIn("--enable-weights-cpu-backup", default)

    def test_common_flags_off_drops_only_that_one_flag(self):
        on = L.common_flags("model", 1, 1, "{}", 100, weights_cpu_backup=True)
        off = L.common_flags("model", 1, 1, "{}", 100, weights_cpu_backup=False)
        diff = [a for a in on if a not in off]
        self.assertEqual(diff, ["--enable-weights-cpu-backup"])
        self.assertIn("--enable-memory-saver", off)

    def test_argv_p_forwards_the_bool(self):
        on = L.argv_p("py", "model", [1, 1, 1], 1, 1, "{}", [])
        off = L.argv_p("py", "model", [1, 1, 1], 1, 1, "{}", [],
                       weights_cpu_backup=False)
        self.assertIn("--enable-weights-cpu-backup", on)
        self.assertNotIn("--enable-weights-cpu-backup", off)

    def test_argv_d_forwards_the_bool(self):
        on = L.argv_d("py", "model", [1, 1, 1], 1, 1, "{}", [])
        off = L.argv_d("py", "model", [1, 1, 1], 1, 1, "{}", [],
                       weights_cpu_backup=False)
        self.assertIn("--enable-weights-cpu-backup", on)
        self.assertNotIn("--enable-weights-cpu-backup", off)

    def test_cli_flag_parses_and_defaults_to_auto(self):
        ap = L.build_parser()
        ns = ap.parse_args(["--tree", "/tmp", "--tag", "x", "--model", "/tmp/x"])
        self.assertEqual(ns.weg2_weights_cpu_backup, wx.WEIGHTS_CPU_BACKUP_AUTO)
        ns2 = ap.parse_args(["--tree", "/tmp", "--tag", "x", "--model", "/tmp/x",
                             "--weg2-weights-cpu-backup", "off"])
        self.assertEqual(ns2.weg2_weights_cpu_backup, "off")
        with self.assertRaises(SystemExit):
            ap.parse_args(["--tree", "/tmp", "--tag", "x", "--model", "/tmp/x",
                          "--weg2-weights-cpu-backup", "bogus"])


class ThePublicationReachesBothGroupsFromOneDict(CustomTestCase):
    """Mutant (ii): the flag reaching only ONE rank means P and D disagree
    about the weight source -- per user law, disagreement between ranks is
    CRASH/STOP, never a degraded-but-running boot. #1369's own answer is
    structural, not a runtime detector: ONE xchg_env dict, merged into BOTH
    groups' build_env call by the SAME variable, so there is no code path
    left where they could read two different values.
    """

    def test_build_env_merges_the_same_dict_identically_for_P_and_D(self):
        xchg_env = {wx.WEIGHTS_CPU_BACKUP_ENV: "off",
                   wx.WEIGHT_SOURCE_ENV: wx.WEIGHT_SOURCE_EXCHANGE}
        env_p = L.build_env("tree", "venv", "0", "/tmp/store", False, "tag",
                            group="P", xchg_env=xchg_env)
        env_d = L.build_env("tree", "venv", "0", "/tmp/store", False, "tag",
                            group="D", xchg_env=xchg_env)
        self.assertEqual(env_p[wx.WEIGHTS_CPU_BACKUP_ENV], "off")
        self.assertEqual(env_d[wx.WEIGHTS_CPU_BACKUP_ENV], "off")
        self.assertEqual(env_p[wx.WEIGHTS_CPU_BACKUP_ENV],
                         env_d[wx.WEIGHTS_CPU_BACKUP_ENV])

    def test_exactly_one_xchg_env_variable_feeds_every_build_env_call(self):
        # MUTANT TRIPWIRE: a future edit that built a per-group
        # `xchg_env_p`/`xchg_env_d` (the #1358 "two spellings of one
        # decision" defect class, one publication path over) would still
        # pass typecheck and tests that only exercise ONE group -- this
        # counts the literal source occurrences of the ONE shared variable
        # name across every `build_env(...)` call site, which is the
        # structural guarantee no per-group copy has crept in.
        import inspect
        src = inspect.getsource(L)
        count = src.count("xchg_env=xchg_env)") + src.count("xchg_env=xchg_env,")
        self.assertGreaterEqual(
            count, 3,
            "expected every build_env(...) call site (group P, and D's two "
            "arms) to pass the SAME xchg_env local -- found fewer; a "
            "per-group copy would let P and D read different decisions, "
            "which is CRASH/STOP by user law, not a degrade")


def _main_resolve(mode, weight_source, inject_mode):
    """Mirrors main()'s own resolution -- see launcher.py's comment there for
    why this can't be a bare call to weights_cpu_backup_armed(explicit=mode)
    for the auto case (that function's auto branch reads os.environ, which
    the launcher's own process never populates -- the S6 fix E trap one
    predicate over)."""
    if mode == wx.WEIGHTS_CPU_BACKUP_AUTO:
        return not L.ring_absent_by_design(weight_source, inject_mode)
    return wx.weights_cpu_backup_armed(explicit=mode)


class TheMainProcessResolutionNeverReadsItsOwnBareEnvironment(CustomTestCase):
    def setUp(self):
        _reset_env()

    def tearDown(self):
        _reset_env()

    def test_auto_ring_stays(self):
        self.assertTrue(_main_resolve("auto", "ring", "shadow"))

    def test_auto_exchange_shadow_stays(self):
        self.assertTrue(_main_resolve("auto", "exchange", "shadow"))

    def test_auto_exchange_authoritative_goes(self):
        self.assertFalse(_main_resolve("auto", "exchange", "authoritative"))

    def test_explicit_off_kills_even_plain_ring(self):
        self.assertFalse(_main_resolve("off", "ring", "shadow"))

    def test_explicit_on_keeps_ring_under_authoritative_AB_instrument(self):
        self.assertTrue(_main_resolve("on", "exchange", "authoritative"))

    def test_resolution_is_immune_to_the_launcher_processs_own_bare_environ(self):
        # THE TRAP: pollute this test PROCESS's own os.environ with values
        # that contradict the arguments -- if `_main_resolve` (or, mutated,
        # a bare `weights_cpu_backup_armed(explicit="auto")`) ever consulted
        # os.environ for exchange_armed()/inject_authoritative() instead of
        # the plain weight_source/inject_mode values it was given, it would
        # read THESE and answer wrongly.
        os.environ[wx.WEIGHT_SOURCE_ENV] = wx.WEIGHT_SOURCE_RING
        os.environ[wx.INJECT_ENV] = wx.INJECT_SHADOW
        self.assertFalse(
            _main_resolve("auto", "exchange", "authoritative"),
            "resolution must follow the ARGUMENTS (this process's own real "
            "arm), never this test's polluted bare os.environ",
        )


class TheLedgerSeesTheRingsWegfall(CustomTestCase):
    """launcher.weg2_weights_cpu_backup_ring_kw + host_ledger.choose/price."""

    RING_BYTES = int(32964 * (1 << 20))
    RING_SPAN1_BYTES = int(29912 * (1 << 20))
    MEMTOTAL = 126_751_866_880
    MEMAVAIL = 111_196_077_056

    def test_ring_kw_passthrough_when_armed(self):
        self.assertEqual(
            L.weg2_weights_cpu_backup_ring_kw(100, 50, False, True),
            (100, 50, False))
        self.assertEqual(
            L.weg2_weights_cpu_backup_ring_kw(100, 50, True, True),
            (100, 50, True))

    def test_ring_kw_zeroes_BOTH_bytes_AND_forces_the_declaration(self):
        # THE MUTANT THIS PINS: a version that only forced the declaration
        # (`ring_absent_by_design = True`) WITHOUT also zeroing the bytes
        # would hand host_ledger.price() a contradiction it refuses by
        # raising -- "the ledger prices the ring anyway" would not even be
        # silent, it would be a boot-time crash on a phantom. Proven below
        # by feeding this function's own output straight into price().
        self.assertEqual(
            L.weg2_weights_cpu_backup_ring_kw(100, 50, False, False),
            (0, 0, True))
        self.assertEqual(
            L.weg2_weights_cpu_backup_ring_kw(100, 50, True, False),
            (0, 0, True))

    def test_price_accepts_the_zeroed_output_without_raising(self):
        rb, rs, absent = L.weg2_weights_cpu_backup_ring_kw(
            self.RING_BYTES, self.RING_SPAN1_BYTES, False, False)
        arm = hl.price(self.MEMTOTAL, self.MEMAVAIL, 1, 1200,
                       ring_bytes=rb, ring_span1_bytes=rs,
                       ring_absent_by_design=absent)
        self.assertEqual(arm.terms["host_ring_gib"], 0.0)

    def test_price_RAISES_if_declaration_and_bytes_disagree(self):
        # THE DANGER DIRECTION NAMED DIRECTLY: if `weg2_weights_cpu_backup_
        # ring_kw` (or a future caller) ever forced the declaration True
        # without zeroing the bytes, THIS is the guard that turns it into a
        # loud crash rather than a silently wrong price -- pinned so nobody
        # "fixes" it into a silent 0 later.
        with self.assertRaises(ValueError):
            hl.price(self.MEMTOTAL, self.MEMAVAIL, 1, 1200,
                    ring_bytes=self.RING_BYTES,
                    ring_span1_bytes=self.RING_SPAN1_BYTES,
                    ring_absent_by_design=True)

    def test_choose_prices_the_ring_when_present(self):
        _chosen, _headroom, lines = hl.choose(
            self.MEMTOTAL, self.MEMAVAIL,
            ring_bytes=self.RING_BYTES, ring_span1_bytes=self.RING_SPAN1_BYTES,
            ring_absent_by_design=False)
        line = next(l for l in lines if l.startswith("WEG2-WEIGHTS-CPU-BACKUP"))
        self.assertIn("host_ring_gib=32.19", line)
        self.assertIn("ring_absent_by_design=False", line)

    def test_choose_zeroes_the_ring_when_declared_absent(self):
        _chosen, _headroom, lines = hl.choose(
            self.MEMTOTAL, self.MEMAVAIL,
            ring_bytes=0, ring_span1_bytes=0, ring_absent_by_design=True)
        line = next(l for l in lines if l.startswith("WEG2-WEIGHTS-CPU-BACKUP"))
        self.assertIn("host_ring_gib=0.00", line)
        self.assertIn("ring_absent_by_design=True", line)

    def test_the_gegenprobe_line_is_ALWAYS_printed_never_only_on_change(self):
        # #1256-class: an advisory printed only sometimes is an advisory
        # nobody can trust, because its absence is indistinguishable from
        # "never checked". Both arms above must carry the line; this test
        # is redundant with the two above by construction but pins the
        # PRESENCE independently of the VALUE, which is the actual #1256
        # failure mode (a line silently dropped, not a line with a wrong
        # number).
        for absent, rb, rs in ((False, self.RING_BYTES, self.RING_SPAN1_BYTES),
                               (True, 0, 0)):
            _c, _h, lines = hl.choose(
                self.MEMTOTAL, self.MEMAVAIL,
                ring_bytes=rb, ring_span1_bytes=rs,
                ring_absent_by_design=absent)
            self.assertTrue(
                any(l.startswith("WEG2-WEIGHTS-CPU-BACKUP") for l in lines),
                f"missing Gegenprobe line for ring_absent_by_design={absent}")
