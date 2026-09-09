# SPDX-License-Identifier: Apache-2.0
"""#1248(a): a post-launch refusal must tear down what main() already spawned.

THE DEFECT. Refusals raised AFTER a group has already spawned (W7, W9, W10,
W45, and -- twice on boot weg2dec1 -- W53 from the D-vector position check)
used to reach `cli()`'s except block, print one line, drop the admin-key
file, and `return 2` WITHOUT tearing down the group(s) already running.
weg2dec1 arms 2/3: the fatal W53 left group P serving on :30031, ~1.4 GiB
held on the 5090, cleared only by a MANUAL teardown against the state json
(/spinning/gpu-arb/weg2/BOOT_weg2dec1_0909.md, read-only record).

THE FIX (upstream-minimal, one funnel). `main()` itself carries zero
try/except -- confirmed below -- so every exception raised anywhere in its
call chain already unwinds, uncaught, to `cli()`'s single
`except REFUSALS as e:` block. That block now reads `_ACTIVE_BOOT_STATE`
(the module global `main()` points at its own `BootState` early on, mirroring
the existing `_ADMIN_KEY` pattern): if no group has a pid yet, it takes
today's fast exit unchanged; otherwise it goes through the SAME `teardown()`
the operator's `--teardown PATH` mode and the killer path use (TERM the pid
set + a pgrep fallback, wait up to 60s, KILL leftovers, unmount the store,
drop the ring dir and vram credit counters), then prints:

    WEG2-LAUNCH REFUSED <W-code> teardown=done groups=<n> pids=<n> cards_empty=<bool>

`cards_empty` reads NVML the same way `teardown()`'s own final print does
(`nvml_registry.memory_snapshot()` / `tenant_used_mib`, the carve-out-honest
figure, #539) so the check and the print can never disagree about "empty".

WHY THIS IS NOT A LEXICAL "AFTER LINE N" TEST. `launch_group()` no-ops
before its `subprocess.Popen` when `dry=True`, and `main()`'s own
`if dry:` branch calls `d_tp_ratio_decision`, `build_env` and a SECOND
`launch_group` (for group D, in no-op form) before returning -- all
textually AFTER the first (real, group-P) `launch_group` call. A test that
classified "post-spawn" purely by source position would therefore also
have to reason about the dry branch to avoid misclassifying those calls.
This suite instead asks the two questions that actually matter: (1) does
`main()` have any try/except that could swallow a raise before it reaches
`cli()` (answer: no, checked structurally), and (2) does any of the named
helper functions `main()` calls swallow ITS OWN refusal internally before
it can propagate out of that helper (answer: no, per function, checked
structurally -- see `_locally_swallowed_refusal_lines` for why this is not
a blanket "no except anywhere" scan).

SCOPE, stated rather than silently narrowed. `REFUSALS` has four members;
this file's AST enumeration covers `main()`'s own inline raises plus the
raises of the SEVEN helper functions `main()` calls directly by name
(`NAMED_POST_SPAWN_HELPERS`) that themselves raise `Weg2LaunchRefused`. The
other three REFUSALS members (`Weg2RingRefused`, `Weg2HostLedgerRefused`,
`Weg2HostRunPeakRefused`) are raised from `ring_table.py` / `host_ledger.py`,
several calls deeper than `main()`'s own frame; their propagation relies on
the SAME "main() has zero try/except" fact this suite pins, but their own
internal call chains are not separately enumerated here. W53
(`Weg2TpOperatingPointDisablesUnevenAxis`, the class named in the weg2dec1
evidence) does not exist in this tree at all (base 80de2d31d1) -- it lives
only on orphaned sibling branches -- so this fix's coverage of it is by
DESIGN (state-based: any `Weg2LaunchRefused`, named or not, after a spawn)
rather than by a direct site check against that class in this tree.

#1248(b), --dry-run vs. the census region: see
`TestDryRunNeverReachesRealSpawnOrCensus`'s docstring for the "reached in
< 1h?" verdict and why not.
"""

import ast
import contextlib
import inspect
import io
import os
import tempfile
import textwrap
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.registry import nvml as nvml_registry
from sglang.srt.weg2 import launcher as L
from sglang.test.test_utils import CustomTestCase

#: The four exception names `cli()`'s `REFUSALS` tuple names or subclasses
#: of which it names (`Weg2RingFormUnproven` etc. inherit `Weg2RingRefused`
#: and are therefore already covered without appearing here).
WEG2_REFUSAL_NAMES = {
    "Weg2LaunchRefused", "Weg2RingRefused",
    "Weg2HostLedgerRefused", "Weg2HostRunPeakRefused",
}

#: The helpers `main()` calls, BY NAME, that are reachable in the REAL
#: (non-`--dry-run`) path after the dry branch's own `return` -- i.e. every
#: function whose own `raise Weg2LaunchRefused(...)` must escape uncaught to
#: reach `cli()`'s one funnel. Verified reachable (`test_the_named_helper_
#: set_is_reachable_after_the_dry_return`) rather than assumed.
NAMED_POST_SPAWN_HELPERS = (
    "wait_ready", "sleep_group", "d_tp_ratio_decision", "build_env",
    "gate_w11", "check_drafter_identity", "launch_group",
)

#: Pinned 2026-09-09 against this branch (base 80de2d31d1). A raise site
#: added or removed changes one of these numbers -- the point is that it
#: cannot change silently.
#:
#: serve-next5 train (2026-09-09): ``d_tp_ratio_decision`` 1 -> 2. The second
#: raise is #1241's operating-point refusal for a SHIPPED position (the
#: W61-W64 ``Weg2TpOperatingPoint*`` family re-raised as ``Weg2LaunchRefused``
#: with "position ... was SHIPPED, so the refusal is fatal here"), merged from
#: ``weg2/tp3-decode-dec2-0909``, whose tree predates this file and so never
#: ran this pin. Not swallowed: ``test_no_named_helper_locally_swallows_its_
#: own_refusal`` holds on the merged tree.
RAISE_COUNTS_BY_HELPER = {
    "wait_ready": 3,
    "sleep_group": 1,
    "d_tp_ratio_decision": 2,
    "build_env": 1,
    "gate_w11": 2,
    "check_drafter_identity": 0,
    "launch_group": 0,
}


# --------------------------------------------------------------------------
# AST helpers -- shared by the pinning tests and the meta-test that proves
# the swallow-detector itself is not vacuous.
# --------------------------------------------------------------------------

def _fn_ast(fn) -> ast.Module:
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


def _fn_ast_with_offset(fn):
    """`(tree, offset)` such that `node.lineno + offset` is the file's real
    line number -- needed only for `main()`, whose sites are cited by real
    file line in this module's docstring and in the #1248 task text."""
    src, start = inspect.getsourcelines(fn)
    return ast.parse(textwrap.dedent("".join(src))), start - 1


def _callee_name(node: ast.Call):
    f = node.func
    return f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)


def _raise_target_name(node: ast.Raise):
    if node.exc is None:
        return None
    call = node.exc
    f = call.func if isinstance(call, ast.Call) else call
    return f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)


def _raise_lines(tree: ast.AST, names) -> list[tuple]:
    """`[(lineno, name), ...]` for every `raise Name(...)` / `raise mod.Name(...)`
    matching `names`, anywhere in `tree`."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Raise):
            nm = _raise_target_name(n)
            if nm in names:
                out.append((n.lineno, nm))
    return out


def _call_lines(tree: ast.AST, name: str) -> list[int]:
    """Every call-site lineno of `name(...)` (bare or `x.name(...)`) in `tree`."""
    return [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _callee_name(n) == name]


def _main_tree_and_offset():
    return _fn_ast_with_offset(L.main)


def _spawn_marker_line() -> int:
    """The first `launch_group(...)` call in `main()` -- group P's real
    launch, the point #1248's task text calls 'the first Popen/subprocess
    site'. Derived by walking the AST, not hand-kept."""
    tree, offset = _main_tree_and_offset()
    lines = _call_lines(tree, "launch_group")
    return min(lines) + offset


def _dry_return_line() -> int:
    """The `return` ending `main()`'s top-level `if dry:` branch -- #1248(b):
    no real spawn and no carrier-census call has happened by this line."""
    tree, offset = _main_tree_and_offset()
    fn = tree.body[0]
    for n in fn.body:
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "dry":
            for stmt in reversed(n.body):
                if isinstance(stmt, ast.Return):
                    return stmt.lineno + offset
    raise AssertionError("main() no longer has a top-level `if dry:` branch")


def _handler_catches_broadly_without_reraise(handler: ast.ExceptHandler) -> bool:
    """True if `handler` would swallow outright: no type (bare `except:`) or
    `Exception`/`BaseException`, AND its last statement is not a `raise` (a
    re-raise still lets the refusal reach `cli()`, so that is not a swallow
    for this purpose)."""
    if handler.type is None:
        broad = True
    else:
        names: list[str] = []

        def collect(n):
            if isinstance(n, ast.Name):
                names.append(n.id)
            elif isinstance(n, ast.Attribute):
                names.append(n.attr)
            elif isinstance(n, ast.Tuple):
                for e in n.elts:
                    collect(e)

        collect(handler.type)
        broad = any(n in ("Exception", "BaseException") for n in names)
    if not broad:
        return False
    return not (handler.body and isinstance(handler.body[-1], ast.Raise))


def _raises_in_stmts(stmts, names) -> list[int]:
    out = []
    for stmt in stmts:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Raise):
                nm = _raise_target_name(n)
                if nm in names:
                    out.append(n.lineno)
    return out


def _locally_swallowed_lines_in_tree(tree: ast.AST) -> list[int]:
    """Lines where a `raise <Weg2 refusal>(...)` sits in the BODY of a `try`
    (not a handler/orelse/finally) whose OWN handler catches broadly and does
    not re-raise.

    NOT a blanket "no except in this function" scan: `d_tp_ratio_decision`
    carries two `except Exception:` blocks that are purely diagnostic
    (a geometry-estimate computation, a `load_measured_library()` call) and
    sit both lexically and dynamically AFTER its own W47 objective-check
    raise -- a blanket scan would flag that function and be wrong. Walking
    each `Try` node's OWN body for a refusal raise means a raise guarded by
    a DIFFERENT, later try/except in the same function is correctly left
    alone (see `test_the_swallow_detector_actually_detects_a_swallow` for
    the positive/negative proof this is not vacuous).
    """
    offenders: list[int] = []
    for try_node in [n for n in ast.walk(tree) if isinstance(n, ast.Try)]:
        hot = _raises_in_stmts(try_node.body, WEG2_REFUSAL_NAMES)
        if hot and any(_handler_catches_broadly_without_reraise(h) for h in try_node.handlers):
            offenders.extend(hot)
    return sorted(set(offenders))


def _locally_swallowed_refusal_lines(fn) -> list[int]:
    return _locally_swallowed_lines_in_tree(_fn_ast(fn))


def _boot_state(tag="t1248", pids=None, cards=None) -> "L.BootState":
    st = L.BootState(tag=tag, tip="deadbeef", tree="/spinning/wt-weg2-1248",
                      stamp="2026-09-09T00:00:00Z")
    st.pids = pids or {}
    st.cards = cards or []
    # A path that cannot possibly be a real mount on ANY box this test runs
    # on, so teardown()'s unmount branch is skipped by construction rather
    # than by luck of what happens to be mounted where the test runs.
    st.store_mount = "/nonexistent-1248-test-mount"
    st.ring_dir = ""
    st.ring_epoch = ""
    st.admin_key_file = ""
    st.helper_pids = []
    return st


def _fake_snapshot(used_mib_by_index: dict[int, int]):
    """`memory_snapshot()`-shaped fake: `[(DeviceInfo, MemoryInfo), ...]`,
    one pair per NVML index in `used_mib_by_index`, each 20480 MiB total
    (a 3080's size; the number does not matter to `_cards_empty`, which
    only reads `tenant_used_mib`)."""
    out = []
    for idx, used_mib in used_mib_by_index.items():
        dev = nvml_registry.DeviceInfo(
            index=idx, uuid=f"GPU-fake-{idx}", name="fake-card",
            total_bytes=20480 * nvml_registry.MIB, pci_bus_id="0000:00:00.0",
        )
        mem = nvml_registry.MemoryInfo(
            total_bytes=20480 * nvml_registry.MIB,
            free_bytes=(20480 - used_mib) * nvml_registry.MIB,
            used_bytes=used_mib * nvml_registry.MIB,
            tenant_used_bytes=used_mib * nvml_registry.MIB,
            carve_out_known=True,
        )
        out.append((dev, mem))
    return out


# --------------------------------------------------------------------------
# Structural pins: "any raise of a Weg2* refusal after the spawn marker must
# be inside the funnel."
# --------------------------------------------------------------------------

class TestNoLocalSwallowing(CustomTestCase):

    def test_main_has_zero_try_blocks(self):
        """The funnel is a SINGLE except-block at the `cli()` level, not
        per-site try/except inside `main()`. If this ever fails, refusal
        handling grew a second, parallel path -- exactly the class #1248
        exists to stop."""
        tree, _offset = _main_tree_and_offset()
        tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
        self.assertEqual(tries, [], "main() must have zero try/except blocks")

    def test_cli_is_the_one_funnel(self):
        tree = _fn_ast(L.cli)
        tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
        self.assertEqual(len(tries), 1, "cli() must be exactly one try/except")
        handler_names: list[str] = []

        def collect(n):
            if isinstance(n, ast.Name):
                handler_names.append(n.id)
            elif isinstance(n, ast.Tuple):
                for e in n.elts:
                    collect(e)

        for h in tries[0].handlers:
            collect(h.type)
        self.assertIn("REFUSALS", handler_names)

    def test_the_spawn_marker_and_dry_return_are_found(self):
        """DENOMINATOR FIRST: if these markers stop resolving, every other
        test below passes vacuously."""
        spawn = _spawn_marker_line()
        dry_ret = _dry_return_line()
        self.assertIsInstance(spawn, int)
        self.assertIsInstance(dry_ret, int)
        self.assertLess(spawn, dry_ret,
                         "group P's real launch must precede the dry return")

    def test_mains_own_raise_sites_are_pinned_pre_and_post_spawn(self):
        tree, offset = _main_tree_and_offset()
        raises = [(ln + offset, nm) for ln, nm in _raise_lines(tree, WEG2_REFUSAL_NAMES)]
        spawn = _spawn_marker_line()
        pre = [r for r in raises if r[0] < spawn]
        post = [r for r in raises if r[0] >= spawn]
        self.assertEqual(len(pre), 4, f"pre-spawn inline raises: {pre}")
        self.assertEqual(len(post), 5, f"post-spawn inline raises: {post}")

    def test_the_named_helper_set_is_reachable_after_the_dry_return(self):
        """Every helper in NAMED_POST_SPAWN_HELPERS must actually be called
        from main() somewhere PAST the dry-branch return in the real path --
        otherwise it is out of this enumeration's scope, or the enumeration
        is stale. (Several of these are ALSO called earlier -- inside the
        dry branch, or pre-spawn for group P -- that is fine; only
        reachability in the real post-spawn path is asserted here.)"""
        tree, offset = _main_tree_and_offset()
        dry_ret = _dry_return_line()
        for name in NAMED_POST_SPAWN_HELPERS:
            calls = [ln + offset for ln in _call_lines(tree, name)]
            self.assertTrue(
                any(c > dry_ret for c in calls),
                f"{name} is never called after the dry-return line {dry_ret}: {calls}",
            )

    def test_the_named_helper_raise_counts_are_pinned(self):
        actual = {
            name: len(_raise_lines(_fn_ast(getattr(L, name)), WEG2_REFUSAL_NAMES))
            for name in NAMED_POST_SPAWN_HELPERS
        }
        self.assertEqual(actual, RAISE_COUNTS_BY_HELPER)

    def test_no_named_helper_locally_swallows_its_own_refusal(self):
        offenders = {
            name: _locally_swallowed_refusal_lines(getattr(L, name))
            for name in NAMED_POST_SPAWN_HELPERS
        }
        offenders = {k: v for k, v in offenders.items() if v}
        self.assertEqual(offenders, {},
                          f"a named helper's own refusal is locally caught: {offenders}")

    def test_the_swallow_detector_actually_detects_a_swallow(self):
        """DENOMINATOR FIRST for the detector itself: an always-empty
        checker would pass the test above vacuously. Exercises the SAME
        `_locally_swallowed_lines_in_tree` the pinning test uses, on two
        synthetic functions it never sees otherwise."""
        swallowing = ast.parse(
            "def evil():\n"
            "    try:\n"
            "        raise Weg2LaunchRefused('nope')\n"
            "    except Exception:\n"
            "        pass\n"
        )
        reraising = ast.parse(
            "def ok():\n"
            "    try:\n"
            "        raise Weg2LaunchRefused('nope')\n"
            "    except Exception:\n"
            "        log_it()\n"
            "        raise\n"
        )
        self.assertEqual(_locally_swallowed_lines_in_tree(swallowing), [3])
        self.assertEqual(_locally_swallowed_lines_in_tree(reraising), [])


# --------------------------------------------------------------------------
# #1248(b): --dry-run vs. the census region.
# --------------------------------------------------------------------------

class TestDryRunNeverReachesRealSpawnOrCensus(CustomTestCase):
    """#1248(b): `--dry-run` returns before the census region.

    Confirmed 2026-09-09 (this branch, base 80de2d31d1): the `if dry:`
    branch calls `d_tp_ratio_decision`, `build_env` and `launch_group` for
    group D in NO-OP form (`launch_group` itself returns before its
    `subprocess.Popen` when `dry=True`) and then returns -- before group D's
    REAL `launch_group` call, before its `wait_ready`, and before the
    carrier-census block (`_cc.route_floor` / `_cc.decide_bound`) that only
    runs later in the real path.

    VERDICT on "reach it with a synthetic D-log hook in < 1h": not
    attempted, and here is why. Between the dry return and the census call,
    `main()` drives group P's REAL `launch_group` (a live `subprocess.Popen`
    plus a `wait_ready` HTTP poll against it), a read of P's OWN log for the
    kv/blob markers the W7/W10 guard checks, `sleep_group`'s authenticated
    RPC round trip, and a dormant-image NVML sample -- none of which the dry
    branch also exercises. A hook that jumps from `--dry-run` straight to
    the census call would have to fake or skip all of that: the same size of
    hermetic harness #1248(a) already built for a DIFFERENT code region
    (the refusal funnel), not a cheap addition to it. Short-circuiting
    `main()` to reach the census region without the P lifecycle means
    refactoring `main()` into smaller directly-testable units, which is out
    of scope for a bugfix task under a "no large code changes" rule. This
    docstring is the "otherwise state why not" the task asks for.
    """

    def test_the_dry_branch_ends_before_group_ds_real_launch(self):
        tree, offset = _main_tree_and_offset()
        dry_ret = _dry_return_line()
        real_d_launch = [ln + offset for ln in _call_lines(tree, "launch_group")
                          if ln + offset > dry_ret]
        self.assertTrue(real_d_launch, "no launch_group call found after the dry return")

    def test_the_dry_branch_ends_before_the_carrier_census_call(self):
        tree, offset = _main_tree_and_offset()
        dry_ret = _dry_return_line()
        census_calls = [ln + offset for ln in _call_lines(tree, "route_floor")]
        self.assertTrue(census_calls, "route_floor() is no longer called from main()")
        self.assertGreater(min(census_calls), dry_ret)


# --------------------------------------------------------------------------
# Behavioral, hermetic: real cli() + real teardown(), fake Popen/pids/NVML.
# --------------------------------------------------------------------------

class TestPreSpawnRefusalFastExitUnchanged(CustomTestCase):
    """A refusal BEFORE any spawn keeps today's fast exit: no teardown, no
    new print line. Must stay true, or the fix changed unrelated behavior."""

    def setUp(self):
        self._orig_state = L._ACTIVE_BOOT_STATE
        self.addCleanup(setattr, L, "_ACTIVE_BOOT_STATE", self._orig_state)
        L.set_admin_key(None)
        self.addCleanup(lambda: L.set_admin_key(None))

    def _run_with(self, fake_main):
        with (
            mock.patch.object(L, "teardown") as fake_teardown,
            mock.patch.object(L, "main", side_effect=fake_main),
        ):
            rc = L.cli(["--fake"])
        return rc, fake_teardown

    def test_no_active_boot_state_at_all(self):
        def fake_main(argv=None):
            L._ACTIVE_BOOT_STATE = None
            raise L.Weg2LaunchRefused("W1 fake pre-spawn refusal")

        rc, fake_teardown = self._run_with(fake_main)
        self.assertEqual(rc, 2)
        fake_teardown.assert_not_called()

    def test_boot_state_exists_but_no_group_has_a_pid_yet(self):
        def fake_main(argv=None):
            L._ACTIVE_BOOT_STATE = _boot_state(pids={"P": 0, "D": 0})
            raise L.Weg2LaunchRefused("W2 fake pre-spawn refusal")

        rc, fake_teardown = self._run_with(fake_main)
        self.assertEqual(rc, 2)
        fake_teardown.assert_not_called()


class TestPostSpawnRefusalRunsTeardown(CustomTestCase):
    """#1248 fix: a refusal AFTER a group has spawned goes through the same
    `teardown()` the killer / `--teardown PATH` path uses, then prints the
    one summary line before exiting 2.

    Evidence this reproduces: boot weg2dec1, arms 2/3, W53 -- group P kept
    serving on :30031 (~1.4 GiB held on the 5090) after the fatal refusal,
    cleared only by a manual teardown against the state json
    (/spinning/gpu-arb/weg2/BOOT_weg2dec1_0909.md).
    """

    def setUp(self):
        self._orig_state = L._ACTIVE_BOOT_STATE
        self.addCleanup(setattr, L, "_ACTIVE_BOOT_STATE", self._orig_state)
        L.set_admin_key(None)
        self.addCleanup(lambda: L.set_admin_key(None))

        self._tmpdir = tempfile.mkdtemp(prefix="weg2-1248-test-")
        self.addCleanup(self._rm_tmpdir)

        # #1248's own teardown-path writes the state json under `GPU_ARB`
        # (`/spinning/gpu-arb` by default) -- redirected to a tempdir so this
        # hermetic test never touches shared rig state.
        self._patches = [
            mock.patch.object(L, "GPU_ARB", self._tmpdir),
            # Fake pids (999001, ...) are never real live processes, so
            # `_alive()` already reads them as dead via a real ESRCH from a
            # real os.kill -- this patch only removes the (small) chance of
            # colliding with an actually-live pid on the box running the
            # test, and any residual TERM/KILL send.
            mock.patch("os.kill", side_effect=ProcessLookupError("no such process")),
            mock.patch("time.sleep"),
            # Backs BOTH `session_pids`'s `ps -eo pid,sid` and `teardown`'s
            # `pgrep -f ...` fallback -- an empty result from each, so the
            # pid set teardown reports is EXACTLY the one this test hands it.
            mock.patch("subprocess.run", return_value=mock.Mock(stdout="")),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _rm_tmpdir(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _run_with(self, fake_main, snapshot):
        buf = io.StringIO()
        with (
            mock.patch.object(nvml_registry, "memory_snapshot", return_value=snapshot),
            mock.patch.object(L, "main", side_effect=fake_main),
            contextlib.redirect_stdout(buf),
        ):
            rc = L.cli(["--fake"])
        return rc, buf.getvalue()

    def test_two_groups_teardown_and_cards_empty(self):
        def fake_main(argv=None):
            L._ACTIVE_BOOT_STATE = _boot_state(
                pids={"P": 999001, "D": 999002},
                cards=[
                    {"nvml_index": 0, "uuid": "GPU-0", "name": "3080",
                     "total_mib": 20480, "reserved_mib": 0},
                    {"nvml_index": 1, "uuid": "GPU-1", "name": "5090",
                     "total_mib": 32607, "reserved_mib": 0},
                ],
            )
            # W45 (``Weg2CarrierCensusRefused``) is a REAL post-spawn refusal
            # of this tree -- the docstring above enumerates it. A synthetic
            # number, or a real number paired with an exception that does not
            # hold it, is a live grep hit that outlives the fake (serve-next4
            # refuter A, 2026-09-09).
            raise L.Weg2LaunchRefused("W45 fake post-spawn refusal")

        rc, out = self._run_with(fake_main, _fake_snapshot({0: 0, 1: 0}))
        self.assertEqual(rc, 2)
        self.assertIn(
            "WEG2-LAUNCH REFUSED W45 teardown=done groups=2 pids=2 cards_empty=True",
            out,
        )

    def test_one_group_cards_not_empty_mirrors_the_weg2dec1_evidence(self):
        """~1.4 GiB left on the 5090 after the refusal: cards_empty must
        read False, never a false 'clean'."""
        def fake_main(argv=None):
            L._ACTIVE_BOOT_STATE = _boot_state(
                pids={"P": 999003, "D": 0},
                cards=[{"nvml_index": 1, "uuid": "GPU-1", "name": "5090",
                        "total_mib": 32607, "reserved_mib": 0}],
            )
            raise L.Weg2LaunchRefused("W45 Weg2CarrierCensusRefused: fake")

        rc, out = self._run_with(fake_main, _fake_snapshot({1: 1434}))
        self.assertEqual(rc, 2)
        self.assertIn(
            "WEG2-LAUNCH REFUSED W45 teardown=done groups=1 pids=1 cards_empty=False",
            out,
        )

    def test_w_code_absent_prints_w_question_mark(self):
        def fake_main(argv=None):
            L._ACTIVE_BOOT_STATE = _boot_state(
                pids={"P": 999004, "D": 0}, cards=[],
            )
            raise L.Weg2LaunchRefused("PYTORCH_CUDA_ALLOC_CONF refused, no code here")

        rc, out = self._run_with(fake_main, _fake_snapshot({}))
        self.assertEqual(rc, 2)
        self.assertIn(
            "WEG2-LAUNCH REFUSED W? teardown=done groups=1 pids=1 cards_empty=True",
            out,
        )


if __name__ == "__main__":
    unittest.main()
