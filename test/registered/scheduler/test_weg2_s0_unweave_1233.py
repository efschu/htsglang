"""Weg-2 slice S0 -- the un-weave, asserted three ways (#1233).

Weg 2 has no in-process cutover: a process is a prefill process or a decode
process for its whole life, and the only thing that crosses a flip is the
canonical page store.  ``scheduler.py``, ``scheduler_pp_mixin.py`` and
``server_args.py`` must therefore stop knowing what a cutover is -- before the
22 flip modules are ``git rm``'d in S7, because the un-weave is the work and
the deletion is an afternoon.

Three assertions, each falsifiable on its own:

* :class:`TestSchedulerImportsWithoutFlipModules` -- the spec's named
  red-first: import ``scheduler`` with every ``phase_flip_*`` module made
  UNIMPORTABLE and assert no ``ImportError``.  This is S7's world simulated
  today.  It only sees *import-time* coupling.
* :class:`TestNoExecutableFlipReference` -- the static form, which also sees
  the function-body (lazy) imports the import test structurally cannot.  It
  scans EXECUTABLE source only: comments and docstrings are excluded on
  purpose, because a comment is not knowledge the program has, and a
  name-based sweep over prose is exactly what §11.4 of the spec warns
  destroys surviving mechanisms.
* :class:`TestEnablePhaseFlipIsNotParseable` -- the CLI half.  §7(4) of the
  spec: hard-remove, no silent shim.

Plus GREEN pins for the invariants that SURVIVE the cut, which is the danger
direction of this slice: the smallest cut that satisfies the acceptance line
must not take the stock PP=3 path with it.
"""

import ast
import io
import re
import subprocess
import sys
import tokenize
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRT = _REPO_ROOT / "python" / "sglang" / "srt"

#: The three files S0 owns.  Paths, not module names: the assertion is about
#: the source of record, and it must hold whether or not the module imports.
_S0_FILES = (
    _SRT / "managers" / "scheduler.py",
    _SRT / "managers" / "scheduler_pp_mixin.py",
    _SRT / "server_args.py",
)

#: Every token that names the in-process cutover machinery.  ``flip`` alone is
#: deliberately NOT in here: it matches ``flip`` in unrelated identifiers and,
#: more importantly, ``hicache_flip_writeback`` is a module the spec KEEPS
#: (§11.4) -- it is named for a mechanism that dies and implements one that
#: lives.  The pattern names the mechanism, not the word.
_FLIP_TOKEN = re.compile(r"phase_flip|PhaseFlip|PHASE_FLIP|cutover|Cutover|CUTOVER")

#: Module prefixes that vanish at S7.  An S0-clean file must not reach for any
#: of them, at import time or lazily inside a function body.
_DEAD_MODULE_PREFIXES = (
    "sglang.srt.managers.phase_flip_",
    "sglang.srt.managers.cutover_participants",
    "sglang.srt.managers.gdn_flip_mover",
    "sglang.srt.managers.hicache_phase_binding",
    "sglang.srt.managers.phase_domain_verdict",
    "sglang.srt.layers.dcp.phase_flip_plan",
    "sglang.srt.layers.dcp.gdn_flip_plan",
    "sglang.srt.mem_cache.kvso_flip_contract",
)


def _executable_lines(path: Path):
    """``[(lineno, text)]`` for lines matching the flip pattern in EXECUTABLE
    source: comments and docstrings removed.

    Comments are BLANKED, not dropped: ``tokenize`` gives the exact span, so
    only the comment text is removed and the code before it on the same line is
    still scanned.  (Dropping whole lines was the first shape of this function
    and mutant M6 survived it -- ``x = getattr(self, "phase_flip_abort_window",
    None)  # note`` was invisible because of the trailing comment alone.  A
    ``#`` regex is not an option either: it would eat a ``#`` inside a string.)
    Docstrings are dropped by AST position.  A string literal passed to
    ``logger.info`` is NOT a docstring and stays -- a log line that narrates a
    cutover is a line the program still executes.
    """
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            lines[row - 1] = lines[row - 1][:col]
    drop = set()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if not body:
            continue
        head = body[0]
        if (
            isinstance(head, ast.Expr)
            and isinstance(head.value, ast.Constant)
            and isinstance(head.value.value, str)
        ):
            for ln in range(head.lineno, (head.end_lineno or head.lineno) + 1):
                drop.add(ln)
    return [
        (i, line.rstrip())
        for i, line in enumerate(lines, 1)
        if i not in drop and _FLIP_TOKEN.search(line)
    ]


def _imported_modules(path: Path):
    """Every module name this file imports, INCLUDING inside function bodies.

    The lazy import is the coupling the import-time test cannot see: today
    ``scheduler.py`` reaches for ``phase_flip_runtime`` from inside six method
    bodies, and every one of them is an ``ImportError`` the day S7 lands.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


# --------------------------------------------------------------------------
# RED 1 -- the spec's named red-first
# --------------------------------------------------------------------------

#: Run in a child so the blocker is installed before the first import of the
#: package.  In-process this test would be a no-op: pytest has already
#: imported half of sglang by collection time, and a module already in
#: ``sys.modules`` never reaches a meta-path finder.
_BLOCKED_IMPORT_PROBE = r"""
import sys
import importlib.abc


DEAD = (
    "phase_flip",
    "cutover_participants",
    "gdn_flip_mover",
    "gdn_flip_plan",
    "hicache_phase_binding",
    "phase_domain_verdict",
    "kvso_flip_contract",
)


class Guillotine(importlib.abc.MetaPathFinder):
    '''S7's world, today: the flip modules are simply not there.'''

    def find_spec(self, fullname, path=None, target=None):
        leaf = fullname.rsplit(".", 1)[-1]
        if any(leaf.startswith(d) or leaf == d for d in DEAD):
            raise ImportError(
                "weg2-s0: %s is deleted in Weg 2 and must never be imported "
                "by scheduler.py / scheduler_pp_mixin.py / server_args.py"
                % fullname
            )
        return None


sys.meta_path.insert(0, Guillotine())

import sglang.srt.server_args  # noqa: F401
import sglang.srt.managers.scheduler_pp_mixin  # noqa: F401
import sglang.srt.managers.scheduler as sched  # noqa: F401

# The module must still be the scheduler, not a husk: name a few surviving
# symbols so an accidental over-cut is a failure here, not at R1 on metal.
for sym in ("Scheduler", "run_scheduler_process", "default_pp_micro_batch_size"):
    assert hasattr(sched, sym), "S0 over-cut: scheduler.%s is gone" % sym

print("WEG2_S0_IMPORT_OK")
"""


class TestSchedulerImportsWithoutFlipModules(unittest.TestCase):
    def test_scheduler_imports_with_every_flip_module_absent(self):
        proc = subprocess.run(
            [sys.executable, "-c", _BLOCKED_IMPORT_PROBE],
            capture_output=True,
            text=True,
            timeout=600,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": "/root",
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": str(_REPO_ROOT / "python"),
            },
        )
        self.assertIn(
            "WEG2_S0_IMPORT_OK",
            proc.stdout,
            "importing the scheduler still reaches a phase_flip module.\n"
            "--- stdout ---\n%s\n--- stderr (tail) ---\n%s"
            % (proc.stdout[-2000:], proc.stderr[-4000:]),
        )


# --------------------------------------------------------------------------
# RED 2 -- the static form: lazy imports and every other executable reference
# --------------------------------------------------------------------------


class TestNoExecutableFlipReference(unittest.TestCase):
    def test_no_lazy_import_of_a_dead_module(self):
        for path in _S0_FILES:
            with self.subTest(file=path.name):
                bad = [
                    m
                    for m in _imported_modules(path)
                    if m.startswith(_DEAD_MODULE_PREFIXES)
                ]
                self.assertEqual(
                    [],
                    bad,
                    "%s imports modules Weg 2 deletes: %s" % (path.name, bad),
                )

    def test_no_executable_flip_or_cutover_token(self):
        for path in _S0_FILES:
            with self.subTest(file=path.name):
                hits = _executable_lines(path)
                self.assertEqual(
                    [],
                    hits,
                    "%s still executes %d flip/cutover line(s); first ten:\n%s"
                    % (
                        path.name,
                        len(hits),
                        "\n".join("  %d: %s" % (i, t[:140]) for i, t in hits[:10]),
                    ),
                )


# --------------------------------------------------------------------------
# RED 3 -- the CLI half (mutant M2's target)
# --------------------------------------------------------------------------


class TestEnablePhaseFlipIsNotParseable(unittest.TestCase):
    def test_server_args_has_no_enable_phase_flip_field(self):
        from sglang.srt.server_args import ServerArgs

        fields = sorted(getattr(ServerArgs, "__dataclass_fields__", {}))
        surviving = [f for f in fields if "phase_flip" in f]
        self.assertEqual(
            [],
            surviving,
            "ServerArgs still carries flip fields: %s" % surviving,
        )
        with self.assertRaises(TypeError):
            ServerArgs(model_path="dummy", enable_phase_flip=True)

    def test_argparse_rejects_the_flag(self):
        import argparse

        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--model-path", "dummy", "--enable-phase-flip"])


# --------------------------------------------------------------------------
# GREEN pins -- the danger direction: what must SURVIVE the cut
# --------------------------------------------------------------------------


class TestStockPpPathSurvives(unittest.TestCase):
    """The acceptance line is "boots as a stock PP=3 server".

    Every pin below except the first was true before S0 with the flag off and
    must still be true after -- they are the over-cut detector.  The first one
    is RED before S0 by construction (the signature still demands the flip
    keyword) and encodes S0's danger direction: the cut must delete the flip
    BRANCH, not promote it to the default.
    """

    def test_pp_micro_batch_default_still_divides_by_pp_size(self):
        from sglang.srt.managers.scheduler import default_pp_micro_batch_size

        # Classic PP: each stage may only hold its share of the concurrency
        # cap.  The flip branch returned the UNDIVIDED cap; if the cut left
        # that branch behind as the new default, this goes red.
        self.assertEqual(
            2, default_pp_micro_batch_size(max_running_requests=6, pp_size=3)
        )
        self.assertEqual(
            1, default_pp_micro_batch_size(max_running_requests=4, pp_size=3)
        )
        self.assertEqual(
            7, default_pp_micro_batch_size(max_running_requests=7, pp_size=1)
        )
        # Degenerate inputs keep their old floor of 1.
        self.assertEqual(
            1, default_pp_micro_batch_size(max_running_requests=0, pp_size=3)
        )
        self.assertEqual(
            1, default_pp_micro_batch_size(max_running_requests=-5, pp_size=3)
        )

    def test_pp_mixin_still_carries_the_pp_event_loop(self):
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        for sym in ("event_loop_pp", "_event_loop_pp_body"):
            self.assertTrue(
                hasattr(SchedulerPPMixin, sym),
                "S0 over-cut: SchedulerPPMixin.%s is gone" % sym,
            )

    def test_reference_pp3_server_args_still_build(self):
        from sglang.srt.server_args import ServerArgs

        sa = ServerArgs(
            model_path="dummy",
            pp_size=3,
            enable_hierarchical_cache=True,
            page_size=1,
            max_running_requests=6,
        )
        self.assertEqual(3, sa.pp_size)
        self.assertTrue(sa.enable_hierarchical_cache)

    def test_token_vector_still_resolves_from_the_env(self):
        """Spec §6.2 rule 1, re-expressed rather than ported.

        ``test_phase_flip_boot.TestFlipTokenVector`` was the only executable
        statement of this invariant and it dies with ``parse_flip_token_vector``
        (whose fallback was the flip WEIGHT vector).  The surviving half --
        ``SGLANG_UNEVEN_TOKEN_VECTOR`` names the per-rank KV TOKEN split, is
        validated against the group's rank count, and is ``None`` when unset --
        now lives in ``ServerArgs._pp_cut_token_shares`` and is pinned here.
        """
        import os

        from sglang.srt.server_args import ServerArgs

        sa = ServerArgs(model_path="dummy", pp_size=3)
        prev = os.environ.get("SGLANG_UNEVEN_TOKEN_VECTOR")
        try:
            os.environ.pop("SGLANG_UNEVEN_TOKEN_VECTOR", None)
            # Unset: None, and None is the ANSWER (pp_cut reads it as "no
            # second arena to split against"), not a missing value.
            self.assertIsNone(sa._pp_cut_token_shares())

            os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = "7,39,18"
            shares = sa._pp_cut_token_shares()
            self.assertEqual(3, len(shares))
            self.assertAlmostEqual(1.0, sum(shares), places=9)
            self.assertAlmostEqual(7 / 64, shares[0], places=9)
            # A gcd-reduced vector is the SAME vector.
            os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = "14,78,36"
            self.assertEqual(shares, sa._pp_cut_token_shares())

            os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = "7,39"
            with self.assertRaises(ValueError):
                sa._pp_cut_token_shares()
            os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = "7,x,18"
            with self.assertRaises(ValueError):
                sa._pp_cut_token_shares()
            os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = "7,0,18"
            with self.assertRaises(ValueError):
                sa._pp_cut_token_shares()
        finally:
            os.environ.pop("SGLANG_UNEVEN_TOKEN_VECTOR", None)
            if prev is not None:
                os.environ["SGLANG_UNEVEN_TOKEN_VECTOR"] = prev

    def test_prefix_len_survives_in_a_module_that_survives(self):
        """The tensor-truthiness landmine (W37-B) keeps ONE definition, and it
        is now in a module Weg 2 does not delete."""
        import torch

        from sglang.srt.managers.schedule_batch import prefix_len

        class _Req:
            pass

        r = _Req()
        r.prefix_indices = torch.zeros(5, dtype=torch.int32)
        self.assertEqual(5, prefix_len(r))  # `or ()` would raise here
        r.prefix_indices = torch.zeros(0, dtype=torch.int32)
        self.assertEqual(0, prefix_len(r))
        r.prefix_indices = None
        self.assertEqual(0, prefix_len(r))
        r.prefix_indices = [1, 2, 3]
        self.assertEqual(3, prefix_len(r))

    def test_hierarchical_cache_no_longer_needs_a_flip_validator(self):
        # The two flip validators refused a combination that only existed
        # under the flip.  Their removal must not take the surviving
        # hierarchical-cache path with them.
        from sglang.srt.server_args import ServerArgs

        sa = ServerArgs(model_path="dummy", disable_radix_cache=True)
        self.assertTrue(sa.disable_radix_cache)


if __name__ == "__main__":
    unittest.main()
