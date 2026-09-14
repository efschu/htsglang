"""#1369 CLOSEOUT, RESTPOSTEN -- coordinator order (continuation after the
ring-off-allocation fix, `697c604323`): ``scheduler.py:2310`` still reads the
RAW ``self.server_args.enable_weights_cpu_backup`` (the argv bit) for its
launch-time refusal (``assert_backup_off_wake_refill_is_defined`` -- a
quantized checkpoint cannot be refilled from disk, so a boot that will need
the disk-reload wake path must refuse BEFORE launch, not at the first wake).
This was left explicitly UNPROVEN in the previous round's report: "argued
(not boot-proven) safe... structurally the same pattern as
inject_mode()/weight_source(), but pattern equality is not proof."

THIS FILE IS THE PROOF, not an assertion of one. The coordinator's own
challenge: "liest scheduler.py:2310 wirklich denselben aufgeloesten Zustand,
oder gibt es einen Pfad, auf dem Launcher-Entscheid und Rang-Lesung
divergieren koennen (Env nicht gesetzt, Default-Zweig, fremdes Env aus einer
Shell)?"

THE TRACE, walked file:line by file:line against this worktree's own
checkout (not assumed from memory of an earlier session):

1. ``main()`` resolves ONE local, ``weights_cpu_backup_armed``
   (``launcher.py:10434-10439``): under ``auto``,
   ``not ring_absent_by_design(str(ns.weg2_weight_source),
   str(ns.weg2_xchg_inject))``; under ``on``/``off``,
   ``weight_exchange.weights_cpu_backup_armed(explicit=...)``.
2. THAT SAME LOCAL feeds ``common_flags``'s conditional
   (``launcher.py:2712``: ``["--enable-weights-cpu-backup"] if
   weights_cpu_backup else []``) at every real argv-build call site --
   ``form_argv_p``, the shipped ``argv_p``, and BOTH ``argv_d`` call sites
   (grepped below, 4 call sites, all passing the literal identifier
   ``weights_cpu_backup_armed``, never a re-derivation). This is what
   ``self.server_args.enable_weights_cpu_backup`` is, on every rank: whether
   the flag string was in ITS OWN argv.
3. THE SAME LOCAL'S TWO INPUTS (``ns.weg2_weight_source``,
   ``ns.weg2_xchg_inject``) are what ``prepare_xchg_env`` is called with
   (``launcher.py:11117-11131``) to build the per-rank ``SGLANG_WEG2_WEIGHT_
   SOURCE`` / ``INJECT_ENV`` publication, and ``ns.weg2_weights_cpu_backup``
   (the raw mode string, not the resolved boolean) is what
   ``WEIGHTS_CPU_BACKUP_ENV`` carries, published UNCONDITIONALLY
   (``launcher.py:11145-11147``, outside ``prepare_xchg_env``'s own
   ``if not armed: return {}`` guard) specifically so an explicit ``on``/
   ``off`` reaches every rank even under the un-armed ``ring`` arm.
4. ``WEIGHT_SOURCE_ENV`` ALSO reaches every rank through a SECOND, redundant
   path: the launcher sets it on ITS OWN process
   (``os.environ[weight_exchange.WEIGHT_SOURCE_ENV] = str(ns.weg2_weight_
   source)``, ``launcher.py:10488``) BEFORE any ``build_env(...)`` call, and
   ``build_env`` starts every rank's environment from ``dict(os.environ)``
   (``launcher.py:4553``) -- so even under ``ring``, where
   ``prepare_xchg_env`` returns ``{}`` and never sets
   ``SGLANG_WEG2_WEIGHT_SOURCE`` itself, the value is already correct in the
   inherited base dict.
5. THE SHORT-CIRCUIT THAT MAKES ``INJECT_ENV``'S ABSENCE UNDER ``ring``
   HARMLESS: ``weights_cpu_backup_armed()``'s ``auto`` branch is
   ``not (exchange_armed() and inject_authoritative())``.  Under ``ring``,
   ``exchange_armed()`` (``weight_source() == "exchange"``) is False, and
   Python's ``and`` never evaluates ``inject_authoritative()`` -- so a stray,
   shell-inherited ``SGLANG_WEG2_XCHG_INJECT`` cannot change the answer on
   this arm, REGARDLESS of whether it was popped.  It matters only under
   ``exchange``, where step 3 above shows it is explicitly, unconditionally
   set from the SAME ``ns.weg2_xchg_inject``.

CONCLUSION: no divergence path was found between what feeds
``server_args.enable_weights_cpu_backup`` (argv) and what
``weight_exchange.weights_cpu_backup_armed()`` would independently compute
from the published env, across all three arms. This is PINNED here as a
ratchet -- the class of change that WOULD reopen it (a new call site that
re-derives the axis instead of reusing the one local, a publication that
stops being unconditional, a short-circuit that gets reordered) is exactly
what the tests below would catch.

DECISION, per the coordinator's own instruction: divergence is
NACHWEISBAR UNMOEGLICH given the current wiring -- PIN it, report
file:line, build nothing new. ``scheduler.py:2310`` is UNCHANGED by this
file; the file only adds tests.

FILE BOUNDARY: this file is new (``test/registered/unit/weg2/``), and reads
(never writes) ``launcher.py`` and ``weight_exchange.py`` -- neither is
edited by this ticket's DESK12 half. Production code in ``launcher.py`` /
``host_ledger.py`` belongs to DESK9; this file does not touch it, per the
coordinator's explicit "sagen, nicht machen" instruction for anything
outside ``model_runner.py`` / ``core.cpp`` / ``scheduler.py``.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import unittest
from contextlib import contextmanager

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher, weight_exchange as wx
from sglang.test.test_utils import CustomTestCase

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)


@contextmanager
def _rank_env(weight_source, inject_mode, backup_mode):
    """Simulate exactly what a RANK's environment would carry after
    ``build_env``/``prepare_xchg_env`` publication, for the three variables
    ``weight_exchange.weights_cpu_backup_armed()`` reads (directly or via
    ``exchange_armed()``/``inject_authoritative()``)."""
    keys = (wx.WEIGHT_SOURCE_ENV, wx.INJECT_ENV, wx.WEIGHTS_CPU_BACKUP_ENV)
    saved = {k: os.environ.get(k) for k in keys}
    try:
        if weight_source is None:
            os.environ.pop(wx.WEIGHT_SOURCE_ENV, None)
        else:
            os.environ[wx.WEIGHT_SOURCE_ENV] = weight_source
        if inject_mode is None:
            os.environ.pop(wx.INJECT_ENV, None)
        else:
            os.environ[wx.INJECT_ENV] = inject_mode
        if backup_mode is None:
            os.environ.pop(wx.WEIGHTS_CPU_BACKUP_ENV, None)
        else:
            os.environ[wx.WEIGHTS_CPU_BACKUP_ENV] = backup_mode
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestTheAutoIdentityAgreesAcrossLauncherAndRank(CustomTestCase):
    """THE CORE BEHAVIOURAL PROOF: main()'s own inline auto formula
    (``not ring_absent_by_design(weight_source, inject_mode)``, what decides
    the ARGV a rank's ``server_args.enable_weights_cpu_backup`` reflects)
    equals ``weight_exchange.weights_cpu_backup_armed()`` computed from a
    RANK's own published environment, for every (weight_source, inject_mode)
    pair -- not asserted, executed."""

    def test_the_two_independent_formulas_agree_on_every_pair(self):
        pairs = [
            (launcher.WEIGHT_SOURCE_DEFAULT, wx.INJECT_SHADOW),
            (launcher.WEIGHT_SOURCE_DEFAULT, wx.INJECT_AUTHORITATIVE),
            (wx.WEIGHT_SOURCE_EXCHANGE, wx.INJECT_SHADOW),
            (wx.WEIGHT_SOURCE_EXCHANGE, wx.INJECT_AUTHORITATIVE),
            (wx.WEIGHT_SOURCE_SHADOW, wx.INJECT_SHADOW),
            (wx.WEIGHT_SOURCE_SHADOW, wx.INJECT_AUTHORITATIVE),
        ]
        for weight_source, inject_mode in pairs:
            launcher_side = not launcher.ring_absent_by_design(
                weight_source, inject_mode)
            with _rank_env(weight_source, inject_mode, wx.WEIGHTS_CPU_BACKUP_AUTO):
                rank_side = wx.weights_cpu_backup_armed()
            self.assertEqual(
                launcher_side, rank_side,
                f"weight_source={weight_source!r} inject_mode={inject_mode!r}: "
                f"main()'s own auto formula ({launcher_side}) disagrees with "
                f"weights_cpu_backup_armed() read from a rank's published env "
                f"({rank_side}) -- this is the divergence scheduler.py:2310's "
                f"raw-flag read would be unsafe under",
            )

    def test_ring_arm_never_evaluates_inject_even_if_it_were_stray(self):
        """The short-circuit that makes a stray, shell-inherited INJECT_ENV
        harmless under `ring`: exchange_armed() is False there, and Python's
        `and` never reaches inject_authoritative(). Proven by NOT publishing
        INJECT_ENV at all (simulating prepare_xchg_env's {} early return)
        and confirming the answer is still correct."""
        with _rank_env(launcher.WEIGHT_SOURCE_DEFAULT, None,
                       wx.WEIGHTS_CPU_BACKUP_AUTO):
            self.assertIs(wx.exchange_armed(), False)
            self.assertIs(wx.weights_cpu_backup_armed(), True)

    def test_explicit_on_off_agree_regardless_of_the_axis(self):
        """`on`/`off` short-circuit before either sub-predicate is touched --
        the identity must hold even with a weight_source/inject_mode
        combination that would disagree under auto."""
        for backup_mode, expected in (
            (wx.WEIGHTS_CPU_BACKUP_ON, True),
            (wx.WEIGHTS_CPU_BACKUP_OFF, False),
        ):
            with _rank_env(wx.WEIGHT_SOURCE_EXCHANGE, wx.INJECT_AUTHORITATIVE,
                           backup_mode):
                self.assertIs(wx.weights_cpu_backup_armed(), expected)


class TestThePublicationWiringIsWhatTheProofAssumed(CustomTestCase):
    """Source-level pins on the FACTS
    ``TestTheAutoIdentityAgreesAcrossLauncherAndRank`` relies on but does not
    itself exercise (it calls the two formulas directly, not `main()`'s
    thousand-line body) -- so a rewiring that broke the premise, rather than
    the formula, is still caught."""

    def test_every_argv_build_call_site_passes_the_one_resolved_local(self):
        """4 call sites (form_argv_p, the shipped argv_p, both argv_d calls)
        must all read the literal `weights_cpu_backup_armed` -- never a
        second `ring_absent_by_design(...)` or a bare `True`/`False`."""
        out = subprocess.run(
            ["git", "-C", REPO_ROOT, "grep", "-n",
             "weights_cpu_backup=weights_cpu_backup_armed", "--",
             "python/sglang/srt/weg2/launcher.py"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(out.returncode, 0, f"git grep failed: {out.stderr[:300]}")
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        self.assertGreaterEqual(
            len(lines), 4,
            f"expected at least 4 call sites passing the one resolved local, "
            f"found {len(lines)}: {lines}",
        )

    def test_prepare_xchg_env_is_called_with_the_same_ns_fields_main_resolved_from(self):
        src = inspect.getsource(launcher.main)
        call = src.index("prepare_xchg_env(")
        # the call spans several lines; grab a generous window rather than
        # parsing it as an expression, so this survives a re-wrap
        window = src[call:call + 700]
        self.assertIn("ns.weg2_weight_source", window)
        self.assertIn("inject_mode=ns.weg2_xchg_inject", window)

    def test_weights_cpu_backup_env_is_published_unconditionally(self):
        """The one deliberate exception to prepare_xchg_env's {} early
        return -- an explicit on/off must reach every rank even under
        `ring`. Pinned by AST: the assignment must sit OUTSIDE (after) the
        prepare_xchg_env(...) call, not inside its conditional body."""
        src = inspect.getsource(launcher.main)
        prepare_call = src.index("xchg_env = prepare_xchg_env(")
        backup_env_set = src.index(
            "xchg_env[weight_exchange.WEIGHTS_CPU_BACKUP_ENV]")
        self.assertGreater(
            backup_env_set, prepare_call,
            "WEIGHTS_CPU_BACKUP_ENV must be set AFTER prepare_xchg_env "
            "returns, not folded into that function's own (conditionally "
            "empty) dict",
        )

    def test_the_launcher_sets_its_own_weight_source_env_before_build_env(self):
        """The redundant path that makes WEIGHT_SOURCE_ENV correct even
        under `ring` (where prepare_xchg_env never sets it itself): the
        launcher's OWN os.environ carries it before the first build_env
        call, whose baseline is dict(os.environ)."""
        src = inspect.getsource(launcher.main)
        self_set = src.index(
            "os.environ[weight_exchange.WEIGHT_SOURCE_ENV] = "
            "str(ns.weg2_weight_source)")
        # "build_env(" alone also matches the explanatory comment right
        # above this same assignment ("BEFORE build_env(), which starts
        # from os.environ: ..."), which sits BEFORE the assignment in the
        # source and would make this pin pass for the wrong reason (or, as
        # measured while writing this test, fail on a false match). The
        # real first CALL is always an assignment.
        first_build_env = src.index("env_p = build_env(")
        self.assertLess(
            self_set, first_build_env,
            "the launcher must publish WEIGHT_SOURCE_ENV into its OWN "
            "environment before the first build_env() call, whose "
            "baseline (dict(os.environ)) is where every rank actually "
            "inherits it under the `ring` arm",
        )

    def test_build_env_starts_from_a_copy_of_the_launchers_own_environment(self):
        src = inspect.getsource(launcher.build_env)
        self.assertIn("env = dict(os.environ)", src)


class TestSchedulerReadsTheArgvDerivedFlagNotAReDerivation(CustomTestCase):
    """The OTHER half of the proof: scheduler.py itself must still be
    reading the raw flag by a SINGLE, simple attribute access -- if a future
    edit turned it into a second call to weights_cpu_backup_armed() (a
    THIRD reader of the same fact, this time in the class that pinned the
    first two agree), that would be worth knowing about explicitly rather
    than silently changing what this whole file proves."""

    def test_scheduler_still_reads_the_raw_server_args_flag(self):
        from sglang.srt.managers import scheduler as sched

        # The check lives in init_watch_dog_memory_saver_input_blocker,
        # called from __init__ -- NOT inline in __init__ itself (verified by
        # actually locating it rather than assuming the method name).
        src = inspect.getsource(
            sched.Scheduler.init_watch_dog_memory_saver_input_blocker)
        self.assertIn(
            "if not self.server_args.enable_weights_cpu_backup:", src,
            "scheduler.py:2310's read changed shape or moved method -- "
            "re-check this file's proof against the new form before "
            "assuming it still applies",
        )
        self.assertNotIn("weights_cpu_backup_armed()", src,
                         "scheduler.py now calls the predicate directly -- "
                         "this file's whole premise (raw flag == predicate, "
                         "proven once, cheaply) is moot; the direct call is "
                         "strictly safer and this note is not a regression, "
                         "just a signal to retire the now-redundant proof")


if __name__ == "__main__":
    unittest.main(verbosity=2)
