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
#:
#: THE HYPHEN IS PART OF THE MECHANISM'S NAME, not a spelling variant.  Every
#: CLI flag of the dead family is hyphenated (``--enable-phase-flip``,
#: ``--phase-flip-tp-vector``), and an underscore-only pattern is blind to all
#: of them: a `parser.add_argument("--phase-flip-tp-vector", ...)` added back
#: to ``add_cli_args`` is executable source this gate must see.  Measured
#: 2026-09-07 on the S0 tip: the underscore-only pattern reported 0 executable
#: hits across the three files where the hyphenated one reports 15, two of
#: which were live operator-facing text naming a flag S0 had just made
#: unparsable.
_FLIP_TOKEN = re.compile(
    r"phase[_-]flip|PhaseFlip|PHASE[_-]FLIP|cutover|Cutover|CUTOVER"
)

#: The modules that vanish at S7, DERIVED FROM SPEC §11.1 rather than from the
#: ``phase_flip`` name shape.  Category A (14) + B (5) + C (3) = 22, and the
#: count is asserted below so a future edit to §11.1 that is not mirrored here
#: fails loudly instead of silently shrinking the gate.
#:
#: WHY NOT PREFIXES.  The first shape of this list was eight ``startswith``
#: prefixes built from the flip NAMING, and six of the 22 modules do not carry
#: that naming at all -- ``phase_req_pool_binding``, ``hicache_phase_guard``,
#: ``mamba_state_pool``, ``seam_coverage``, ``seam_slope``, ``seam_holdback``.
#: The token regex misses them by the same name coincidence, so a lazy import
#: of any of the six was invisible to all three gates (measured 2026-09-07: a
#: lazy ``from sglang.srt.managers.seam_coverage import enabled`` planted in
#: ``Scheduler.release_host_resources`` left the whole file green).  Two more
#: entries had the WRONG PACKAGE: ``kvso_flip_contract`` lives in ``managers``
#: and ``hicache_phase_binding`` in ``mem_cache``, so those two prefixes could
#: never have matched a real import either.  A name-shaped list cannot catch
#: its own class; an enumerated one can.
_DEAD_MODULES = (
    # Category A -- pure flip machinery (14)
    "sglang.srt.managers.phase_flip_runtime",
    "sglang.srt.managers.phase_flip_boot",
    "sglang.srt.managers.phase_flip_spill",
    "sglang.srt.managers.phase_flip_seam_reserve",
    "sglang.srt.managers.phase_flip_presence",
    "sglang.srt.managers.phase_flip_draft_bootstrap",
    "sglang.srt.managers.gdn_flip_mover",
    "sglang.srt.managers.phase_flip_seam_census",
    "sglang.srt.layers.dcp.phase_flip_plan",
    "sglang.srt.managers.phase_flip_counters",
    "sglang.srt.managers.phase_flip_output_trace",
    "sglang.srt.managers.cutover_participants",
    "sglang.srt.managers.kvso_flip_contract",
    "sglang.srt.layers.dcp.gdn_flip_plan",
    # Category B -- rebind / binding bookkeeping (5)
    "sglang.srt.managers.phase_domain_verdict",
    "sglang.srt.mem_cache.hicache_phase_binding",
    "sglang.srt.managers.phase_req_pool_binding",
    "sglang.srt.mem_cache.hicache_phase_guard",
    "sglang.srt.mem_cache.mamba_state_pool",
    # Category C -- seam economics satellites (3)
    "sglang.srt.managers.seam_coverage",
    "sglang.srt.managers.seam_slope",
    "sglang.srt.planner.seam_holdback",
)


#: The dead-module imports S0 does NOT remove, named one by one with the slice
#: that owns each.  This is a DEFERRAL, stated, not a gate that quietly passes.
#:
#: Both entries were invisible to the gate's first shape for two independent
#: reasons, which is why they surface only now: ``hicache_phase_binding`` was
#: listed under the wrong package (``managers``; it lives in ``mem_cache``), and
#: ``phase_domain_verdict`` is imported as ``from sglang.srt.managers import
#: phase_domain_verdict``, a shape the old collector recorded only as the
#: package name.
#:
#: WHY NOT CUT THEM HERE.  Neither is flip plumbing that collapses to a
#: constant; each is a live mechanism whose re-rooting is a design decision the
#: spec assigns elsewhere:
#:
#: * ``phase_domain_verdict`` (§11.1 Category B) carries the #1068 phase-domain
#:   slice of the uniform MIN-reduce and RAISES a group STOP on disagreement
#:   (``scheduler.py:6725``, read back at ``:6783``).  Deleting it deletes a
#:   ranks-never-disagree stop; §11.4's warning about name-based sweeps is
#:   exactly this shape.
#: * ``hicache_phase_binding`` supplies ``current_generation`` /
#:   ``bound_phase`` to four cache-probe and premise-stamp seams
#:   (``scheduler.py:4922``, ``:5041``, ``:5195``; ``scheduler_pp_mixin.py:2797``,
#:   ``:2816``, ``:2863``).  §11.4 names the re-root -- "reads ``BindingState``
#:   generations, which vanish; re-root on the store key" -- and books it
#:   BEFORE the Category B cut, i.e. S5/S7, not here.
_KNOWN_RESIDUAL_DEAD_IMPORTS = {
    "scheduler.py": (
        "sglang.srt.managers.phase_domain_verdict",
        "sglang.srt.mem_cache.hicache_phase_binding",
        "sglang.srt.mem_cache.hicache_phase_binding.bound_phase",
        "sglang.srt.mem_cache.hicache_phase_binding.current_generation",
    ),
    "scheduler_pp_mixin.py": ("sglang.srt.mem_cache.hicache_phase_binding",),
    "server_args.py": (),
}


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


def _package_of(path: Path) -> str:
    """The dotted package a source file lives in, e.g. ``sglang.srt.managers``.

    Derived from the path under ``python/`` so a relative import can be
    resolved to the absolute name the dead-module list is written in.
    """
    rel = path.resolve().relative_to((_REPO_ROOT / "python").resolve())
    return ".".join(rel.parts[:-1])


def _imported_modules(path: Path):
    """Every module name this file imports, INCLUDING inside function bodies.

    The lazy import is the coupling the import-time test cannot see: today
    ``scheduler.py`` reaches for ``phase_flip_runtime`` from inside six method
    bodies, and every one of them is an ``ImportError`` the day S7 lands.

    THREE SHAPES, ALL RESOLVED TO ABSOLUTE NAMES, because a gate that only
    understands one spelling is a gate the next edit walks past:

    * ``import a.b.c`` -- the alias name.
    * ``from a.b import c`` -- BOTH ``a.b`` and ``a.b.c``, because the second
      is how you import a MODULE with the ``from`` form, and the dead list
      names modules.
    * ``from .c import X`` / ``from . import c`` -- relative, resolved against
      this file's own package.  The first shape of this function dropped every
      relative import before the check (``node.level == 0``), and all three S0
      files sit in the same package as the modules that die, so the relative
      form is legal there and was invisible (measured 2026-09-07: a lazy
      ``from .gdn_flip_mover import GdnFlipPools`` planted in
      ``Scheduler.release_host_resources`` left the whole file green; the tree
      carries 131 relative imports under ``python/sglang/srt``, so this is the
      ordinary spelling, not a contrived one).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg_parts = _package_of(path).split(".")
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                # ``level`` 1 is this package, 2 is its parent, and so on.
                keep = len(pkg_parts) - (node.level - 1)
                base = ".".join(pkg_parts[:keep]) if keep > 0 else ""
                if node.module:
                    base = "%s.%s" % (base, node.module) if base else node.module
            if not base:
                continue
            names.append(base)
            names.extend("%s.%s" % (base, a.name) for a in node.names)
    return names


def _dead_imports(path: Path):
    """The names in *path* that reach a module Weg 2 deletes."""
    return sorted(
        {
            m
            for m in _imported_modules(path)
            if any(m == d or m.startswith(d + ".") for d in _DEAD_MODULES)
        }
    )


def _literal_assignments(path: Path, func: str, target: str):
    """Every literal assigned to *target* inside ``def func`` in *path*.

    Reads the SOURCE, not a running object, because the values this pins live
    inside ``Scheduler.__init__``-time methods that need a model to run.  A
    non-literal right-hand side yields nothing, which is the point: the danger
    direction of S0 is a flip BRANCH promoted to the default, and a promoted
    branch is an expression, never a literal.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != func:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                targets, value = sub.targets, sub.value
            elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
                targets, value = [sub.target], sub.value
            else:
                continue
            for t in targets:
                name = (
                    t.attr
                    if isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                    else t.id
                    if isinstance(t, ast.Name)
                    else None
                )
                if name != target:
                    continue
                try:
                    out.append((sub.lineno, ast.literal_eval(value)))
                except (ValueError, TypeError):
                    out.append((sub.lineno, _NOT_A_LITERAL))
    return out


class _NotALiteral:
    def __repr__(self):  # pragma: no cover - only ever shown in a failure
        return "<not a literal: an expression, i.e. a surviving branch>"


_NOT_A_LITERAL = _NotALiteral()


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
    def test_the_dead_module_list_is_the_whole_deletion_surface(self):
        """Spec §11.1 names 22 modules; the gate must know all 22.

        The denominator of every claim this class makes.  If §11.1 grows or
        shrinks, this fails and the list is re-derived by hand rather than
        drifting silently.
        """
        self.assertEqual(22, len(_DEAD_MODULES))
        self.assertEqual(len(set(_DEAD_MODULES)), len(_DEAD_MODULES))
        # Every named module must actually exist in the tree today -- a typo
        # or a wrong package is a gate that can never fire (two of the eight
        # prefixes this list replaced had exactly that defect).
        missing = [
            d
            for d in _DEAD_MODULES
            if not (_REPO_ROOT / "python" / (d.replace(".", "/") + ".py")).is_file()
        ]
        self.assertEqual(
            [], missing, "dead-module list names absent files: %s" % missing
        )

    def test_no_lazy_import_of_a_dead_module(self):
        """EXACT equality against the named residual, not ``<=``.

        A new coupling fails here because it is not in the list; a residual
        that is finally cut ALSO fails here, which is what forces the list to
        shrink deliberately rather than rot.  The denominator is printed in
        ``_KNOWN_RESIDUAL_DEAD_IMPORTS`` above, with the reason each entry is
        deferred and the slice that owns it.
        """
        for path in _S0_FILES:
            with self.subTest(file=path.name):
                self.assertEqual(
                    sorted(_KNOWN_RESIDUAL_DEAD_IMPORTS.get(path.name, ())),
                    _dead_imports(path),
                    "%s: the set of dead-module imports moved.  Anything new "
                    "here is an S7 ImportError shipped early; anything missing "
                    "means a residual was cut and this list must shrink with "
                    "it." % path.name,
                )

    def test_the_lazy_import_gate_can_fail_on_every_shape_it_claims(self):
        """Can-it-fail proof, in-file rather than by planting a mutant.

        Each of the four spellings below is a real import statement the gate
        must catch; the three that once slipped past it are named.
        """
        shapes = (
            # absolute, module-level, the shape the original gate did catch
            "import sglang.srt.managers.phase_flip_runtime\n",
            # absolute ``from`` naming the MODULE as the imported name
            "from sglang.srt.managers import seam_coverage\n",
            # relative, inside a function body -- mutant B2's shape
            "def f():\n    from .gdn_flip_mover import GdnFlipPools\n",
            # relative, bare, one package up
            "def g():\n    from ..mem_cache import hicache_phase_guard\n",
        )
        probe = _SRT / "managers" / "_weg2_s0_gate_probe.py"
        for i, src in enumerate(shapes):
            with self.subTest(shape=i):
                probe.write_text(src, encoding="utf-8")
                try:
                    self.assertNotEqual(
                        [],
                        _dead_imports(probe),
                        "the lazy-import gate is blind to shape %d:\n%s" % (i, src),
                    )
                finally:
                    probe.unlink(missing_ok=True)

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


#: Every CLI spelling S0 removes.  Ten fields are deleted outright and the
#: eleventh -- ``phase_flip_canonical_kv_page`` -- is RENAMED to
#: ``--hicache-canonical-kv-page`` (builder deviation F1, pulled forward from
#: S5 because deleting the field would have turned Weg 2's ONE carrier
#: silently off between the two slices).  Both cases share one obligation
#: here: the OLD spelling must not parse.  Spec §11.5 rule 5 is "hard-remove
#: with a loud refusal", and one probed flag is not a proof about eleven.
_REMOVED_FLIP_FLAGS = (
    "--enable-phase-flip",
    "--phase-flip-policy",
    "--phase-flip-purity",
    "--phase-flip-tp-vector",
    "--phase-flip-spill-depth",
    "--phase-flip-corridor-floor-mib",
    "--phase-flip-image-file-backed",
    "--phase-flip-canonical-kv-page",
    "--phase-flip-writeback",
    "--phase-flip-writeback-deadline-s",
    "--phase-flip-rebind-hicache",
)


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
        for flag in _REMOVED_FLIP_FLAGS:
            with self.subTest(field=flag):
                kw = flag.lstrip("-").replace("-", "_")
                with self.assertRaises(TypeError):
                    ServerArgs(model_path="dummy", **{kw: True})

    def test_argparse_rejects_every_removed_flag(self):
        """Not one flag -- all eleven.

        The single-flag form of this test was one eleventh of a gate: it
        probed ``--enable-phase-flip`` and said nothing about the ten other
        spellings the same commit removed.
        """
        import argparse

        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        for flag in _REMOVED_FLIP_FLAGS:
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--model-path", "dummy", flag])

    def test_the_renamed_canonical_page_flag_parses_under_its_new_name(self):
        """The other half of F1: the carrier must still be reachable.

        ``--phase-flip-canonical-kv-page`` is rejected above; if the rename
        had merely deleted it, Weg 2's ONE carrier across a flip would be
        silently off and no test above would notice.
        """
        import argparse

        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        ns = parser.parse_args(["--model-path", "dummy", "--hicache-canonical-kv-page"])
        self.assertTrue(getattr(ns, "hicache_canonical_kv_page"))
        self.assertIn("hicache_canonical_kv_page", ServerArgs.__dataclass_fields__)

    def test_no_refusal_message_names_a_flag_argparse_rejects(self):
        """An error message that tells the operator to pass an unparsable
        flag is a dead end with a helpful tone.

        Scans the executable source of the three S0 files for the removed
        spellings; the token gate above catches the same lines, but this one
        says WHY they are wrong, and it survives a future narrowing of that
        regex.
        """
        for path in _S0_FILES:
            with self.subTest(file=path.name):
                hits = [
                    (i, t)
                    for i, t in _executable_lines(path)
                    if any(f in t for f in _REMOVED_FLIP_FLAGS)
                ]
                self.assertEqual(
                    [],
                    hits,
                    "%s executes text naming a removed flag:\n%s"
                    % (
                        path.name,
                        "\n".join("  %d: %s" % (i, t.strip()[:140]) for i, t in hits),
                    ),
                )


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

    #: The collapsed values the commit body names, and the denominator of the
    #: claim "every value this commit pins is the value the expression already
    #: produced with the flag off".  Five of the seven are pinned below as
    #: statement-level literals; ``phase_policy_hook=None`` and the shut warmup
    #: window are keyword arguments inside nested calls and are covered by the
    #: slice boot's O-1 line instead -- stated, not assumed.
    _COLLAPSED = (
        ("init_parked_decode_set", "want", False),
        ("init_request_receiver", "pp_flip_counters", None),
        ("init_request_receiver", "pp_chain_receiver", None),
        ("abort_request", "deferred", False),
    )

    def test_the_named_collapsed_defaults_are_still_flag_off_literals(self):
        """The over-cut detector, applied to the values the commit names.

        A flip branch promoted to the stock default is the danger direction of
        this slice, and it is invisible to every other test here: the promoted
        expression still imports, still parses, and names no flip token.
        Measured 2026-09-07: replacing ``want = False`` with
        ``want = mamba_allocator is not None`` -- which is TRUE on the
        reference model, so it arms the parked set and silently discounts
        admission on a stock PP=3 boot -- left this whole file green.
        """
        sched = _SRT / "managers" / "scheduler.py"
        for func, target, expected in self._COLLAPSED:
            with self.subTest(where="%s.%s" % (func, target)):
                found = _literal_assignments(sched, func, target)
                self.assertNotEqual(
                    [], found, "no assignment to %s in %s()" % (target, func)
                )
                for lineno, value in found:
                    self.assertIs(
                        expected,
                        value,
                        "scheduler.py:%d %s() collapsed %s to %r, not the "
                        "flag-off value %r" % (lineno, func, target, value, expected),
                    )

    def test_parked_decode_verdict_default_agrees_with_its_readers(self):
        """The initializer, the writer and both readers must be ONE type.

        ``_parked_decode_verdict`` was a ``(phase, blocked)`` tuple; S0 reduced
        it to a bool because there is no other phase to compare against.  The
        writer is now unreachable (``parked_decode_set.enabled`` is a hardwired
        ``False``), so whatever the initializer holds is what every reader sees
        for the whole life of the process -- and ``bool((None, False))`` is
        ``True``.  That makes ``_decode_forbidden_this_phase()`` report a
        prohibition on EVERY boot, on the per-round admission path, in a
        process that has exactly one role.

        Read as a value from the source and pushed through the real methods, so
        this pins the behaviour rather than the spelling.
        """
        from sglang.srt.managers.scheduler import Scheduler

        sched = _SRT / "managers" / "scheduler.py"
        found = _literal_assignments(
            sched, "init_parked_decode_set", "_parked_decode_verdict"
        )
        self.assertNotEqual([], found, "no _parked_decode_verdict initializer found")

        class _ParkedSet:
            enabled = False
            resident_count = 0

            def carrier_discount(self):
                # Deliberately non-zero: if the guard above it stops working,
                # the discount leaks through and this test says so.
                return 99

        for lineno, value in found:
            stub = type("_SchedStub", (), {})()
            stub._parked_decode_verdict = value
            stub.parked_decode_set = _ParkedSet()
            self.assertIs(
                False,
                Scheduler._decode_forbidden_this_phase(stub),
                "scheduler.py:%d initialises _parked_decode_verdict to %r, "
                "which the bool readers see as a standing decode prohibition"
                % (lineno, value),
            )
            self.assertEqual(
                0,
                Scheduler._parked_carrier_discount(stub, 4),
                "scheduler.py:%d initialises _parked_decode_verdict to %r, "
                "which discounts carriers on a boot that parks none" % (lineno, value),
            )

    def test_hierarchical_cache_no_longer_needs_a_flip_validator(self):
        # The two flip validators refused a combination that only existed
        # under the flip.  Their removal must not take the surviving
        # hierarchical-cache path with them.
        from sglang.srt.server_args import ServerArgs

        sa = ServerArgs(model_path="dummy", disable_radix_cache=True)
        self.assertTrue(sa.disable_radix_cache)


if __name__ == "__main__":
    unittest.main()
