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
import types
import unittest
from pathlib import Path
from typing import Optional

from sglang.srt.removed_cli_flags import (
    REMOVED_FLIP_FLAGS as _SHIPPED_REMOVED_FLIP_FLAGS,
)
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

#: EVERY SOURCE FILE THIS SLICE EDITS OR CREATES, dead modules included.  This
#: is the RAW denominator, and it is written out rather than derived from ``git
#: diff --name-only aef3ae7676..HEAD -- 'python/*.py'`` because a test may not
#: assume git, a reachable base ref, or a checkout carrying history.
#:
#: WHY IT IS A LIST AND NOT A COUNT.  The previous shape pinned the surviving
#: subset directly and asserted ``len(...) == 12`` beside a docstring saying
#: "fourteen source files".  Measured 2026-09-07: the diff held FIFTEEN, and
#: the missing one was ``mem_cache/hicache_flip_writeback.py`` -- the module
#: §11.4 marks KEEP as the STORE PATH, i.e. the one carrier Weg 2 rests on,
#: carrying two live imports of a Category-B dead module
#: (``:534  from sglang.srt.mem_cache.hicache_phase_guard import
#: seam_refusals``) that no gate could see.  A hand-counted literal cannot
#: notice its own omission; the arithmetic below can, because the surviving set
#: is COMPUTED from this one and the exemption is named rather than implied.
_S0_EDITED_SOURCE_FILES = (
    _SRT / "entrypoints" / "http_server.py",
    _SRT / "managers" / "cache_controller.py",
    _SRT / "managers" / "kv_session_offload.py",
    _SRT / "managers" / "phase_flip_counters.py",
    _SRT / "managers" / "phase_flip_draft_bootstrap.py",
    _SRT / "managers" / "pp_wire_channels.py",
    _SRT / "managers" / "schedule_batch.py",
    _SRT / "managers" / "scheduler.py",
    _SRT / "managers" / "scheduler_components" / "batch_result_processor.py",
    _SRT / "managers" / "scheduler_components" / "output_streamer.py",
    _SRT / "managers" / "scheduler_pp_mixin.py",
    _SRT / "managers" / "tokenizer_control_mixin.py",
    _SRT / "mem_cache" / "hicache_flip_writeback.py",
    _SRT / "model_executor" / "model_runner.py",
    _SRT / "model_executor" / "model_runner_kv_cache_mixin.py",
    _SRT / "removed_cli_flags.py",
    _SRT / "server_args.py",
    _SRT / "speculative" / "eagle_worker_v2.py",
)


def _module_name_of(path: Path) -> str:
    """``sglang.srt.managers.scheduler`` for a path under ``python/``."""
    rel = path.resolve().relative_to((_REPO_ROOT / "python").resolve())
    return ".".join(rel.parts)[: -len(".py")]


#: EVERY SURVIVING FILE S0 EDITS, which is the denominator the lazy-import gate
#: needs and ``_S0_FILES`` is not.  Scoping the S7-ImportError gate to the three
#: files S0 OWNS left the rest unscanned, including ``pp_wire_channels.py`` --
#: the module this very slice created and ``scheduler_pp_mixin`` imports.
#: Measured 2026-09-07: a lazy ``from sglang.srt.managers import
#: phase_flip_runtime`` planted in ``pp_wire_channels.py`` left the whole slice
#: file green (17 passed, 39 subtests), because the path was not in
#: ``_S0_FILES``.
#:
#: The exemption is COMPUTED: an edited file that is itself in ``_DEAD_MODULES``
#: is deleted at S7, so a dead-module import inside it is not an S7 ImportError.

#: The files the EXECUTABLE-TOKEN gate holds to zero: the three S0 owns plus
#: the one it created.  A new module born inside this slice must be born clean;
#: the three touched files that still carry flip tokens are named as a deferral
#: in ``_KNOWN_FLIP_TOKEN_FILES`` below rather than left unstated.
_TOKEN_CLEAN_FILES = _S0_FILES + (_SRT / "managers" / "pp_wire_channels.py",)

#: The touched files that still execute flip vocabulary, with the slice that
#: owns each.  Asserted by EXACT equality below, so a new one fails here and a
#: cleaned one forces this list to shrink.
#:
#: * ``cache_controller.py`` reads ``phase_flip_tp_vector`` and names a cutover
#:   in two messages; its Category-B binding imports are S5/S7's re-root.
#: * ``model_runner.py`` carries the ``is_phase_flip_tp_stack`` constructor
#:   argument -- the SECOND stack, which S3 replaces with a second process.
#: * ``model_runner_kv_cache_mixin.py`` carries the seam-reserve and spill
#:   readers; §11.5 item 1 books the seam fund as a PLANNER change first.
#: * ``hicache_flip_writeback.py`` is §11.4's KEEP + RENAME -- "named for a
#:   mechanism that dies, implements one that lives".  Four of its seven hits
#:   are prose about what a cutover did to a partially written tree and three
#:   are ``getattr(server_args, "phase_flip_writeback*", ...)`` reads whose
#:   field S0 removed, so they answer the flag-off default (``False`` /
#:   ``None``) the stock boot always saw.  The RENAME is S5's, with the store.
#: * ``removed_cli_flags.py`` is the §7 item 4 refusal ITSELF.  Its whole job is
#:   to spell the removed flags back at an operator who passes one; a gate that
#:   held it to zero would forbid the refusal the spec orders.
_KNOWN_FLIP_TOKEN_FILES = {
    "cache_controller.py",
    "hicache_flip_writeback.py",
    "model_runner.py",
    "model_runner_kv_cache_mixin.py",
    "removed_cli_flags.py",
}

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


_TOUCHED_FILES = tuple(
    p for p in _S0_EDITED_SOURCE_FILES if _module_name_of(p) not in _DEAD_MODULES
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
#: * ``cache_controller.py`` and ``model_runner_kv_cache_mixin.py`` join the
#:   list when the gate widens from the three S0 files to all twelve it
#:   touches.  Neither is S0's to cut: the first reads the same
#:   ``hicache_phase_binding`` generations §11.4 books for the S5/S7 re-root,
#:   the second reads ``phase_flip_seam_reserve`` / ``phase_flip_spill`` /
#:   ``seam_slope``, which §11.5 item 1 orders removed from the PLANNER first
#:   and separately.  They are listed so the gate says "deferred, and here is
#:   the owner" instead of "not scanned".
_KNOWN_RESIDUAL_DEAD_IMPORTS = {
    "scheduler.py": (
        "sglang.srt.managers.phase_domain_verdict",
        "sglang.srt.mem_cache.hicache_phase_binding",
        "sglang.srt.mem_cache.hicache_phase_binding.bound_phase",
        "sglang.srt.mem_cache.hicache_phase_binding.current_generation",
    ),
    "scheduler_pp_mixin.py": ("sglang.srt.mem_cache.hicache_phase_binding",),
    "server_args.py": (),
    # §11.4's STORE PATH, and the reason the denominator above is computed now.
    # `_seam_refusals_or_none` (`:518`) reads the guard's write-refusal counter
    # to grade its own writeback; the re-root belongs with the store key
    # (S5/S7), together with `hicache_phase_binding`'s generations, because the
    # counter is the guard's and not this module's to reinvent.
    "hicache_flip_writeback.py": (
        "sglang.srt.mem_cache.hicache_phase_guard",
        "sglang.srt.mem_cache.hicache_phase_guard.seam_refusals",
    ),
    "cache_controller.py": (
        "sglang.srt.mem_cache.hicache_phase_binding",
        "sglang.srt.mem_cache.hicache_phase_binding.current_generation",
        "sglang.srt.mem_cache.hicache_phase_binding.host_pool_for_generation",
        "sglang.srt.mem_cache.hicache_phase_binding.write_back_stamp_is_current",
        "sglang.srt.mem_cache.hicache_phase_guard",
        "sglang.srt.mem_cache.hicache_phase_guard.active_phase",
        "sglang.srt.mem_cache.hicache_phase_guard.device_tier_disarmed",
    ),
    "model_runner_kv_cache_mixin.py": (
        "sglang.srt.managers.phase_flip_boot",
        "sglang.srt.managers.phase_flip_boot.checkpoint_param_dict",
        "sglang.srt.managers.phase_flip_boot.parse_flip_vector",
        "sglang.srt.managers.phase_flip_runtime",
        "sglang.srt.managers.phase_flip_runtime.PP_TO_TP",
        "sglang.srt.managers.phase_flip_runtime.TP_TO_PP",
        "sglang.srt.managers.phase_flip_runtime.derive_pp_full_attn_layer_map",
        "sglang.srt.managers.phase_flip_seam_reserve",
        "sglang.srt.managers.phase_flip_spill",
        "sglang.srt.managers.phase_flip_spill.cold_stack_deferred",
        "sglang.srt.managers.seam_slope",
        "sglang.srt.managers.seam_slope.derive_seam_slope_for_rank",
        "sglang.srt.managers.seam_slope.received_attention_layers",
    ),
}


def _executable_source_lines(path: Path):
    """``[(lineno, text)]`` for every line of EXECUTABLE source in *path*:
    comments and docstrings removed.

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
        (i, line.rstrip()) for i, line in enumerate(lines, 1) if i not in drop
    ]


def _executable_lines(path: Path):
    """The subset of :func:`_executable_source_lines` naming the cutover."""
    return [(i, t) for i, t in _executable_source_lines(path) if _FLIP_TOKEN.search(t)]


def _package_of(path: Path) -> str:
    """The dotted package a source file lives in, e.g. ``sglang.srt.managers``.

    Derived from the path under ``python/`` so a relative import can be
    resolved to the absolute name the dead-module list is written in.
    """
    rel = path.resolve().relative_to((_REPO_ROOT / "python").resolve())
    return ".".join(rel.parts[:-1])


#: Recorded in place of a dynamically imported module whose name is computed.
#: Never equal to any ``_DEAD_MODULES`` entry, so it is a finding to read, not a
#: silent pass: a touched file that computes a module name has to say why.
_DYNAMIC_IMPORT_ARG = "<dynamic>"


def _is_dynamic_import(func) -> bool:
    """``importlib.import_module`` / ``import_module`` / ``__import__``."""
    if isinstance(func, ast.Name):
        return func.id in ("import_module", "__import__")
    if isinstance(func, ast.Attribute):
        return func.attr in ("import_module", "__import__")
    return False


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
    * ``importlib.import_module("a.b.c")`` / ``__import__("a.b.c")`` -- the
      DYNAMIC form, which is a statement to no AST import node at all.  This is
      the shape that made the slice's whole premise ("S7 deletes 22 modules and
      nothing breaks") only half-gated: measured 2026-09-07, an
      ``importlib.import_module("sglang.srt.managers.seam_coverage")`` planted
      in ``Scheduler.init_parked_decode_set`` -- a method that runs at boot --
      was invisible to ALL THREE coupling gates at once (the token regex misses
      it because Category-C modules carry no flip naming, this collector missed
      it because it walked only ``ast.Import``/``ast.ImportFrom``, and the
      runtime Guillotine probe covers import time while this is a function
      body).  The shape is live in this tree, not contrived: 28
      ``import_module(`` sites under ``python/sglang/srt``.
      A NON-CONSTANT argument is recorded as ``<dynamic>`` rather than dropped,
      so a computed module name in a touched file surfaces as an unresolvable
      import instead of as silence.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pkg_parts = _package_of(path).split(".")
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_dynamic_import(node.func):
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                names.append(first.value)
            else:
                names.append(_DYNAMIC_IMPORT_ARG)
            continue
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


def _binds(node) -> Optional[str]:
    """The plain name an assignment target binds: ``x`` or ``self.x``."""
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def _literal_assignments(path: Path, func: str, target: str):
    """Every literal bound to *target* inside ``def func`` in *path*.

    Reads the SOURCE, not a running object, because the values this pins live
    inside ``Scheduler.__init__``-time methods that need a model to run.  A
    non-literal right-hand side yields ``_NOT_A_LITERAL``, which is the point:
    the danger direction of S0 is a flip BRANCH promoted to the default, and a
    promoted branch is an expression, never a literal.

    THREE BINDING SHAPES, because a collapsed default is not always a
    statement.  The first shape of this function walked ``ast.Assign`` /
    ``ast.AnnAssign`` only, and two of the seven values the commit itself names
    as collapsed are neither:

    * ``name = <literal>`` -- statement assignment (and the annotated form).
    * ``f(..., name=<literal>, ...)`` -- a KEYWORD ARGUMENT.  Measured
      2026-09-07: replacing ``phase_policy_hook=None`` (``scheduler.py:2959``,
      inside ``init_request_receiver``) with ``getattr(self,
      "_policy_recv_hook", None) or (lambda reqs: reqs[:1])`` -- which silently
      discards every recv batch but its first request on the live intake path
      (``request_receiver.py:196``/``:205``) -- left the whole slice file green.
    * ``return <literal>`` -- pass ``target="<return>"``.  The mixin's
      collapsed values are all of this shape (``_pp_flip_epoch`` at
      ``scheduler_pp_mixin.py:9653`` returns ``None``), and a rank-derived
      expression there is a cross-rank identity mismatch, not a local one.
    * ``name += <expr>`` / ``name |= <expr>`` -- an AUGMENTED assignment, and
      ``name := <expr>`` -- a walrus, and ``setattr(self, "name", <expr>)``.
      None of the three is an ``Assign``, an ``AnnAssign``, a keyword or a
      ``Return``, so all three re-arm a collapsed default with the detector
      looking straight at it.  Measured 2026-09-07: ``want |= mamba_allocator
      is not None`` one line after ``want = False`` in
      ``Scheduler.init_parked_decode_set`` -- True on the reference hybrid
      model, so the parked-decode set arms on a stock PP=3 boot and silently
      discounts admission -- left this whole file green.  An augmented
      assignment to a value pinned as a flag-off constant is ALWAYS the defect,
      so it is recorded as ``_NOT_A_LITERAL`` whatever its right-hand side; the
      pin then fails on it by construction.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []

    def _record(lineno, value):
        try:
            out.append((lineno, ast.literal_eval(value)))
        except (ValueError, TypeError):
            out.append((lineno, _NOT_A_LITERAL))

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != func:
            continue
        for sub in ast.walk(node):
            if target == _RETURN_TARGET:
                if isinstance(sub, ast.Return):
                    _record(sub.lineno, sub.value)
                continue
            if isinstance(sub, ast.keyword) and sub.arg == target:
                _record(sub.value.lineno, sub.value)
                continue
            if isinstance(sub, ast.AugAssign) and _binds(sub.target) == target:
                # Never a literal, by construction: `x = False` followed by
                # `x |= <anything>` is a re-armed default whatever the
                # right-hand side evaluates to.
                out.append((sub.lineno, _NOT_A_LITERAL))
                continue
            if isinstance(sub, ast.NamedExpr) and _binds(sub.target) == target:
                _record(sub.lineno, sub.value)
                continue
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "setattr"
                and len(sub.args) == 3
                and isinstance(sub.args[1], ast.Constant)
                and sub.args[1].value == target
            ):
                _record(sub.lineno, sub.args[2])
                continue
            if isinstance(sub, ast.Assign):
                targets, value = sub.targets, sub.value
            elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
                targets, value = [sub.target], sub.value
            else:
                continue
            for t in targets:
                if _binds(t) != target:
                    continue
                _record(sub.lineno, value)
    return out


#: Sentinel target for :func:`_literal_assignments`: pin the function's RETURN
#: values rather than a name it binds.
_RETURN_TARGET = "<return>"

#: The statements after which nothing in the same block can run.
_TERMINAL_STATEMENTS = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def _unreachable_statements(path: Path):
    """``[(func, lineno, kind)]`` for every statement that can never run.

    A statement that follows ``return`` / ``raise`` / ``continue`` / ``break``
    in the SAME block is dead, and an edit that prepends an unconditional
    ``return`` to a function while leaving its body in place is exactly the
    shape S0 shipped (``_pp_flip_hold_slot``: ``Return@6200`` followed by five
    live-looking statements, five of which read a mechanism whose only caller
    was the last of them).

    ruff cannot see this class here -- no unreachable-code rule is enabled --
    so the commit's "ruff parity EXACT" was a green number over a check blind
    to the edit's error class.  Three lines of AST are not.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        scope = getattr(node, "name", None) or type(node).__name__
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for i, stmt in enumerate(block[:-1]):
                if isinstance(stmt, _TERMINAL_STATEMENTS):
                    nxt = block[i + 1]
                    out.append((scope, nxt.lineno, type(nxt).__name__))
                    break
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
        for path in _TOUCHED_FILES:
            with self.subTest(file=path.name):
                self.assertEqual(
                    sorted(_KNOWN_RESIDUAL_DEAD_IMPORTS.get(path.name, ())),
                    _dead_imports(path),
                    "%s: the set of dead-module imports moved.  Anything new "
                    "here is an S7 ImportError shipped early; anything missing "
                    "means a residual was cut and this list must shrink with "
                    "it." % path.name,
                )

    def test_the_scanned_set_is_every_surviving_file_the_slice_touches(self):
        """The denominator of the lazy-import gate, DERIVED rather than counted.

        The previous shape asserted a hand-written ``12`` beside a docstring
        claiming fourteen edited files; the diff held fifteen, and the file it
        omitted was ``hicache_flip_writeback.py`` -- §11.4's store path, with
        two live imports of a Category-B dead module.  A literal cannot notice
        its own omission, so the arithmetic is asserted instead: the surviving
        set is exactly the edited set minus the edited files that are
        themselves deleted at S7, every exemption resolves to a real
        ``_DEAD_MODULES`` entry, and every path exists on disk.
        """
        exempt = [
            p
            for p in _S0_EDITED_SOURCE_FILES
            if _module_name_of(p) in _DEAD_MODULES
        ]
        self.assertEqual(
            len(_S0_EDITED_SOURCE_FILES) - len(exempt),
            len(_TOUCHED_FILES),
            "the surviving set is not the edited set minus the dead modules",
        )
        self.assertNotEqual(
            [], exempt, "no edited file is exempt -- the exemption arm is dead"
        )
        self.assertEqual(len(set(_TOUCHED_FILES)), len(_TOUCHED_FILES))
        self.assertEqual(
            len(set(_S0_EDITED_SOURCE_FILES)), len(_S0_EDITED_SOURCE_FILES)
        )
        missing = [p for p in _S0_EDITED_SOURCE_FILES if not p.is_file()]
        self.assertEqual([], missing, "edited-file list names absent files")
        # Every S0-owned file is inside the wider set: the narrow gate must be
        # a subset of the wide one, never a parallel one that drifts.
        for p in _S0_FILES:
            self.assertIn(p, _TOUCHED_FILES)
        for p in _TOKEN_CLEAN_FILES:
            self.assertIn(p, _TOUCHED_FILES)

    def test_no_unreachable_statement_in_a_touched_file(self):
        """An unconditional ``return`` prepended to a live body is dead code.

        S0's own error class: ``_pp_flip_hold_slot`` was collapsed by prefixing
        ``return self._1173_forget_stashed_frame()`` and leaving five statements
        behind it, which silently orphaned the #1173 launched-pass GROUP STOP
        without a word in the commit message.  ruff's enabled rule set here has
        no unreachable-code rule, so the commit's green "ruff parity EXACT"
        number was measured by a check blind to the edit.
        """
        for path in _TOUCHED_FILES:
            with self.subTest(file=path.name):
                dead = _unreachable_statements(path)
                self.assertEqual(
                    [],
                    dead,
                    "%s carries %d unreachable statement(s):\n%s"
                    % (
                        path.name,
                        len(dead),
                        "\n".join(
                            "  %s: line %d (%s)" % row for row in dead[:10]
                        ),
                    ),
                )

    def test_the_lazy_import_gate_can_fail_on_every_shape_it_claims(self):
        """Can-it-fail proof, in-file rather than by planting a mutant.

        Each of the six spellings below is a real import this gate must catch;
        the five that once slipped past it are named.  The docstring used to
        say "every shape it claims" while claiming only ``import``/``from``
        spellings, which is honest and still leaves the slice's premise --
        runtime reachability, not statement shape -- half-gated.  The last two
        rows close that half.
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
            # DYNAMIC, inside a function body -- mutant M-RB1's shape, and the
            # one that was invisible to all three coupling gates at once
            "import importlib\n"
            "def h():\n"
            "    importlib.import_module('sglang.srt.managers.seam_coverage')\n",
            # dynamic, the builtin spelling
            "def i():\n"
            "    __import__('sglang.srt.mem_cache.mamba_state_pool')\n",
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
        for path in _TOKEN_CLEAN_FILES:
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

    def test_the_touched_files_that_still_carry_flip_vocabulary_are_named(self):
        """EXACT equality against the deferral, not a silent narrow scope.

        S0 touches twelve surviving files and holds four of them to zero flip
        tokens.  The other three are deferred to a named slice; asserting the
        SET means a fourth one appearing fails here, and one being cleaned fails
        here too, so the deferral shrinks deliberately.
        """
        dirty = {p.name for p in _TOUCHED_FILES if _executable_lines(p)}
        self.assertEqual(_KNOWN_FLIP_TOKEN_FILES, dirty)


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
#:
#: READ FROM THE SHIPPED REFUSAL, not re-typed here.  The first shape of this
#: gate kept its own eleven-entry tuple beside ``srt/removed_cli_flags.py``'s
#: -- two lists of the same names, which is the fork-owned second bookkeeping
#: the upstream-minimal law forbids, and the failure it produces is the worst
#: kind: the gate stays green while the refusal quietly stops covering a flag.
#: The count is asserted below so a shrinking refusal fails loudly instead.
_REMOVED_FLIP_FLAGS = tuple(_SHIPPED_REMOVED_FLIP_FLAGS)


#: The surviving files that still put a removed flag spelling into executable
#: text, with a count per file, asserted by EXACT equality.  S0 FIXES the two
#: instances on Weg 2's CARRIER path (``mem_cache/hicache_flip_writeback.py``,
#: which §11.4 marks KEEP as the store path) and DEFERS the rest, named:
#:
#: * ``managers/phase_purity.py`` (2) -- the purity refusal; the mechanism dies
#:   with §11.1 Category A, and its message dies with it.
#: * ``mem_ledger/engine.py`` (3) -- the DCP-group ledger note explaining why a
#:   group exists under the flip; a ledger EXPLANATION, not an instruction.
#: * ``model_executor/model_runner_kv_cache_mixin.py`` (1) -- an ``_abstain``
#:   reason string on the seam-reserve path §11.5 item 1 books to the planner.
#: * ``model_executor/weights_arena.py`` (2) -- ``flag=`` metadata of the arena
#:   family, which §11 deletes together with the arena.
#: * ``planner/rejected.py`` (2) -- the VERWORFENES register, which records
#:   what was rejected and when; rewriting history there would be the defect.
#:
#: None of the five is on the store path a Weg-2 operator follows; every one is
#: owned by a later slice, and the equality makes that deferral fail loudly the
#: day a seventh file appears or one of these is finally cut.
#:
#: The SIXTH entry is not a deferral at all: ``srt/removed_cli_flags.py`` is the
#: §7 item 4 refusal, whose whole job is to spell these flags back at an
#: operator who passes one and name the replacement where there is one.  It is
#: listed here, with its count, rather than exempted by a rule -- so the day it
#: stops naming all eleven, or starts naming a twelfth, this equality says so.
_KNOWN_RESIDUAL_FLAG_TEXT = {
    "python/sglang/srt/removed_cli_flags.py": 11,
    "python/sglang/srt/managers/phase_purity.py": 2,
    "python/sglang/srt/mem_ledger/engine.py": 3,
    "python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py": 1,
    "python/sglang/srt/model_executor/weights_arena.py": 2,
    "python/sglang/srt/planner/rejected.py": 2,
}


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
        """Not one flag -- all eleven, and ABSENCE asserted rather than inferred.

        THE PROBE THAT WAS FIVE ELEVENTHS OF A GATE.  The first shape here was
        ``parse_args(["--model-path", "dummy", flag])`` with no VALUE for the
        flag, and six of the eleven removed spellings are value-taking
        (``_StoreAction``: ``--phase-flip-policy``, ``--phase-flip-purity``,
        ``--phase-flip-tp-vector``, ``--phase-flip-spill-depth``,
        ``--phase-flip-corridor-floor-mib``,
        ``--phase-flip-writeback-deadline-s``).  argparse exits 2 with "expected
        one argument" for those whether or not the option exists, so those six
        subtests were green on the PARENT, where all eleven flags are present.
        Measured 2026-09-07: re-adding ``phase_flip_policy`` to ``ServerArgs``
        made ``--phase-flip-policy auto`` parse and this gate still reported
        eleven passing subtests.

        So the assertion is now the direct one -- the option string is not in
        the parser at all -- with the exit probe kept as a second, correctly
        shaped assertion: value-taking flags get a dummy value, so the only
        reason left to exit is that the option is unknown.
        """
        import argparse

        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        registered = parser._option_string_actions
        for flag in _REMOVED_FLIP_FLAGS:
            with self.subTest(flag=flag):
                self.assertNotIn(
                    flag,
                    registered,
                    "%s is still a registered option; §11.5 rule 5 is "
                    "hard-remove with a loud refusal, not a silent shim" % flag,
                )
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--model-path", "dummy", flag, "x"])

    def test_the_absence_probe_can_fail_on_a_value_taking_flag(self):
        """Can-it-fail proof for the shape that could not fail before.

        A parser carrying a value-taking option under one of the removed
        spellings must be REJECTED by the assertion above.  Built here rather
        than planted in ``ServerArgs`` so the proof is in the file.
        """
        import argparse

        probe = argparse.ArgumentParser()
        probe.add_argument("--model-path")
        probe.add_argument("--phase-flip-policy", type=str, default=None)
        self.assertIn("--phase-flip-policy", probe._option_string_actions)
        # And the OLD probe shape is blind to exactly this: no value given, so
        # argparse exits 2 for "expected one argument" on a LIVE option.
        with self.assertRaises(SystemExit):
            probe.parse_args(["--model-path", "dummy", "--phase-flip-policy"])
        # With a value it parses, which is what the new assertion catches.
        ns = probe.parse_args(["--model-path", "dummy", "--phase-flip-policy", "auto"])
        self.assertEqual("auto", ns.phase_flip_policy)

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

        SCOPED TO THE WHOLE SURVIVING TREE, not to the three files S0 owns.  A
        rename that breaks an operator instruction three modules away is the
        same defect as one that breaks it in place, and the first shape of this
        test could not see either of the two real instances S0 shipped:

        * ``mem_cache/hicache_flip_writeback.py:241`` told the operator to
          "Enable --phase-flip-canonical-kv-page", the spelling S0 renamed to
          ``--hicache-canonical-kv-page`` -- in the module §11.4 marks KEEP as
          the STORE PATH, i.e. on the one carrier Weg 2 rests on.
        * the same file at ``:895`` raised "``--phase-flip-writeback`` is set
          but ...", naming a flag S0 deleted outright.

        Files under ``_DEAD_MODULES`` are exempt: they are deleted at S7, and a
        message inside one is not an instruction anyone can still reach.  The
        cheap ``in`` pre-filter runs over raw bytes so only the handful of files
        that mention a removed spelling at all pay for tokenize + AST.
        """
        dead_paths = {
            _REPO_ROOT / "python" / (d.replace(".", "/") + ".py") for d in _DEAD_MODULES
        }
        offenders = {}
        scanned = 0
        for path in sorted(_SRT.rglob("*.py")):
            if path in dead_paths:
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")
            if not any(f in raw for f in _REMOVED_FLIP_FLAGS):
                continue
            scanned += 1
            hits = [
                i
                for i, t in _executable_lines(path)
                if any(f in t for f in _REMOVED_FLIP_FLAGS)
            ]
            if hits:
                offenders[str(path.relative_to(_REPO_ROOT))] = len(hits)
        self.assertEqual(
            _KNOWN_RESIDUAL_FLAG_TEXT,
            offenders,
            "the set of surviving files naming an unparsable flag moved (%d "
            "file(s) mention one at all).  Anything new here is an operator "
            "instruction S0's rename broke; anything missing means a residual "
            "was fixed and this list must shrink with it." % scanned,
        )


# --------------------------------------------------------------------------
# RED 4 -- the surfaces that OUTLIVED the mechanism
# --------------------------------------------------------------------------


def _dispatcher_registered_types(path: Path):
    """Every request type ``Scheduler.init_request_dispatcher`` registers.

    Read from the SOURCE and not from a live ``Scheduler``: the constructor
    needs a model, a device and a process group, and this question is about the
    dispatch table's contents, which are literal names in one list.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "init_request_dispatcher":
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if not (
                isinstance(call.func, ast.Name)
                and call.func.id == "TypeBasedDispatcher"
            ):
                continue
            for entry in call.args[0].elts:
                if isinstance(entry, ast.Tuple) and isinstance(
                    entry.elts[0], ast.Name
                ):
                    out.add(entry.elts[0].id)
    return out


def _request_types_constructed(path: Path):
    """``{name: [lineno, ...]}`` for every ``XxxReq``/``XxxReqInput`` built."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name.endswith("ReqInput") or name.endswith("Req"):
            out.setdefault(name, []).append(node.lineno)
    return out


class TestNoLiveSenderOutlivesItsHandler(unittest.TestCase):
    """A removed handler is a SERVING KILLER while its sender is still wired.

    THE FAILURE THIS PINS, in full, because it is not a style point.  S0 deleted
    ``(PhaseFlipReqInput, self.handle_phase_flip)`` from the scheduler's
    dispatch table (it stood at ``aef3ae7676:scheduler.py:2493``) and left the
    route that sends one: ``POST /phase_flip`` in ``http_server.py``, through
    ``TokenizerManager.phase_flip``, onto the request stream that every rank
    reads.  ``Scheduler.process_input_requests`` calls
    ``self._request_dispatcher(recv_req)`` with no ``try``/``except``
    (``scheduler.py:2840``); ``TypeBasedDispatcher`` has no fallback registered
    on this table, so it reaches ``sglang/utils.py:673  raise ValueError(f"Invalid
    object: {obj}")``; ``run_scheduler_process`` catches, logs and sends SIGQUIT
    to the parent.  The request is broadcast, so ALL THREE schedulers die
    together.  At the base commit the same POST returned an arming verdict.

    THE GATE IS THE PROPERTY, NOT THE NAME.  Asserting "``/phase_flip`` is gone"
    would pin this one instance; the class is "a control-plane sender whose
    scheduler handler no longer exists".  So: every request type the tokenizer
    manager's control mixin CONSTRUCTS -- which is exactly the set it puts on
    the scheduler's socket -- must be a type that dispatch table registers.
    """

    def test_every_type_the_control_mixin_sends_has_a_scheduler_handler(self):
        registered = _dispatcher_registered_types(_SRT / "managers" / "scheduler.py")
        self.assertGreater(
            len(registered), 20, "the dispatch table did not parse; gate is blind"
        )
        sent = _request_types_constructed(
            _SRT / "managers" / "tokenizer_control_mixin.py"
        )
        orphans = {n: lines for n, lines in sent.items() if n not in registered}
        self.assertEqual(
            {},
            orphans,
            "tokenizer_control_mixin.py builds %d request type(s) no scheduler "
            "handler accepts; each one is a ValueError in "
            "TypeBasedDispatcher and a SIGQUIT on every rank: %s"
            % (len(orphans), orphans),
        )

    def test_no_entrypoint_still_names_a_removed_request_type(self):
        """The route half: the HTTP surface may not carry the dead type either.

        ``http_server.py`` imported ``PhaseFlipReqInput`` and annotated the
        route body with it, so the type was constructed by FastAPI from the
        request JSON before any of our code ran.  Scanning executable source
        for the name covers the annotation, the import and any construction.
        """
        for path in (
            _SRT / "entrypoints" / "http_server.py",
            _SRT / "managers" / "tokenizer_control_mixin.py",
        ):
            with self.subTest(file=path.name):
                hits = [
                    (i, t)
                    for i, t in _executable_source_lines(path)
                    if "PhaseFlip" in t
                ]
                self.assertEqual(
                    [],
                    hits,
                    "%s still names the removed request type: %s"
                    % (path.name, hits[:5]),
                )

    def test_the_dispatcher_really_raises_on_an_unregistered_type(self):
        """Can-it-fail proof for the premise, run rather than asserted.

        The whole finding rests on "an unregistered type reaches
        ``raise ValueError``".  If ``TypeBasedDispatcher`` had a fallback, or
        matched by duck typing, the gate above would be guarding nothing.
        """
        from sglang.utils import TypeBasedDispatcher

        class _Registered:
            pass

        class _NotRegistered:
            pass

        d = TypeBasedDispatcher([(_Registered, lambda o: "handled")])
        self.assertEqual("handled", d(_Registered()))
        with self.assertRaises(ValueError):
            d(_NotRegistered())


def _rank_gated_exits_before_the_barrier(path: Path, func: str):
    """``[(lineno, kind, test_src)]`` for every early exit this rank may take
    alone, lexically before the #753 per-iteration lockstep barrier.

    THE PROPERTY, NOT THE NAME.  The fix-2 commit removed ``_pp_flip_hold_slot``
    -- a ``continue`` under ``if not self.pp_group.is_last_rank:`` sitting
    immediately in front of ``self.pp_group.barrier()`` -- and the gate that
    replaced it asserted the ABSENCE OF THREE METHOD NAMES.  Measured
    2026-09-07: ``if not self.pp_group.is_last_rank and getattr(self,
    "_pp_proxy_drops", 0): continue``, planted at the same indent and the same
    place, reachable (``_pp_proxy_drops`` is written at
    ``scheduler_pp_mixin.py:10209``), rank-asymmetric and carrying zero flip
    vocabulary, left the slice and neighbour battery byte-identical to the clean
    baseline (4 failed / 60 passed / 13 skipped / 59 subtests).

    A barrier is a rendezvous: whoever skips it leaves its peers blocked in it
    for ever.  So what must hold is not "those three methods are gone" but
    "every path to the barrier is rank-symmetric".
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    src = path.read_text(encoding="utf-8").splitlines()
    body = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            body = node
            break
    if body is None:
        raise AssertionError("no %s() in %s" % (func, path.name))

    barriers = [
        n.lineno
        for n in ast.walk(body)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "barrier"
    ]
    if len(barriers) != 1:
        raise AssertionError(
            "expected exactly one barrier call in %s(); found %d at %s -- the "
            "gate's denominator moved and it must be re-read, not widened"
            % (func, len(barriers), barriers)
        )
    barrier_line = barriers[0]

    rank_tokens = ("is_last_rank", "is_first_rank", "pp_rank")
    out = []

    def walk(node, guards):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                text = ast.get_source_segment("\n".join(src), child.test) or ""
                walk(child, guards + [(child.lineno, " ".join(text.split()))])
                continue
            if isinstance(child, (ast.Continue, ast.Break, ast.Return)):
                if child.lineno >= barrier_line:
                    continue
                for lineno, text in guards:
                    if any(tok in text for tok in rank_tokens):
                        out.append(
                            (
                                child.lineno,
                                type(child).__name__,
                                "%d: %s" % (lineno, text),
                            )
                        )
                        break
                continue
            walk(child, guards)

    walk(body, [])
    return out


class TestTheLockstepBarrierIsRankSymmetric(unittest.TestCase):
    def test_no_rank_gated_exit_precedes_the_753_barrier(self):
        mixin = _SRT / "managers" / "scheduler_pp_mixin.py"
        bad = _rank_gated_exits_before_the_barrier(mixin, "_event_loop_pp_body")
        self.assertEqual(
            [],
            bad,
            "a rank-conditional early exit sits in front of the #753 "
            "per-iteration lockstep barrier; the ranks that take it leave the "
            "others blocked in the rendezvous for ever:\n%s"
            % "\n".join("  line %d: %s under %s" % row for row in bad),
        )

    def test_the_rank_symmetry_gate_can_fail(self):
        """Can-it-fail proof, in-file: the exact mutant that survived.

        Planting it in the tree and reverting is a two-minute manual loop that
        nobody repeats; running the detector over the source WITH the mutant
        spliced in is the same evidence, every run.
        """
        mixin = _SRT / "managers" / "scheduler_pp_mixin.py"
        text = mixin.read_text(encoding="utf-8")
        needle = "                if self._pp_gapped_wire:\n"
        self.assertIn(needle, text, "the barrier guard moved; re-read the gate")
        mutant = (
            "                if not self.pp_group.is_last_rank and getattr(\n"
            '                    self, "_pp_proxy_drops", 0\n'
            "                ):\n"
            "                    continue\n"
        ) + needle
        probe = _SRT / "managers" / "_weg2_s0_barrier_probe.py"
        probe.write_text(text.replace(needle, mutant, 1), encoding="utf-8")
        try:
            found = _rank_gated_exits_before_the_barrier(probe, "_event_loop_pp_body")
            self.assertNotEqual(
                [],
                found,
                "the rank-symmetry gate is blind to the mutant it exists for",
            )
        finally:
            probe.unlink(missing_ok=True)


class TestTheProxyReceiveDecidesAcceptOrDrop(unittest.TestCase):
    """S0 turned the #631/#1004 identity REFUSAL into an unconditional drop.

    The refusal was group-fatal and therefore self-testing: nothing could take
    the wrong branch quietly.  A drop is silent, so the accept/drop decision of
    the surviving PP wire needs a behavioural test or it has none.  Measured
    2026-09-07: turning the drop's ``return None`` (``scheduler_pp_mixin.py:10225``)
    into ``return PPProxyTensors(raw)`` -- which computes a forward on a proxy
    frame that names a DIFFERENT slot, the mispair the deleted RuntimeError
    named verbatim -- left the slice and neighbour battery byte-identical to the
    clean baseline.  The one test that touched this family called the PREDICATE
    ``pp_proxy_stamp_names_pass`` directly and never the receive path that
    consumes its verdict.
    """

    def _holder(self, raw):
        from sglang.srt.managers import scheduler_pp_mixin as ppm

        class _Holder:
            require_attn_tp_allgather = False
            attn_tp_group = None
            forward_ct = 7
            mbs = None

            def __init__(self):
                self.pp_group = types.SimpleNamespace(is_first_rank=False)
                self._pp_gapped_wire = False
                self._pp_proxy_drops = 0
                self.readiness_calls = []

            def _pp_wait_for_proxy_readiness(self, mb_id):
                self.readiness_calls.append(mb_id)

            def _pp_recv_typed_dict(self, expected_kind=None, all_gather_group=None):
                assert expected_kind == "proxy"
                return dict(raw)

        return ppm, _Holder()

    def test_a_stamp_naming_this_slot_is_accepted_with_its_payload(self):
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        ppm, holder = self._holder(
            {"hidden_states": "HS", "__stamp__": (1, 17, 3, -1)}
        )
        out = ppm.SchedulerPPMixin._pp_recv_proxy_tensors(holder, mb_id=1)
        self.assertIsInstance(out, PPProxyTensors)
        self.assertEqual("HS", out.tensors["hidden_states"])
        self.assertNotIn("__stamp__", out.tensors)
        self.assertEqual(0, holder._pp_proxy_drops)

    def test_a_stamp_naming_another_slot_is_dropped_and_counted(self):
        ppm, holder = self._holder(
            {"hidden_states": "HS", "__stamp__": (2, 17, 3, -1)}
        )
        out = ppm.SchedulerPPMixin._pp_recv_proxy_tensors(holder, mb_id=1)
        self.assertIsNone(
            out,
            "a proxy frame stamped for another slot was ACCEPTED: this pairs "
            "one request set's hidden states with another's metadata, which is "
            "silent wrong output whenever the two widths agree",
        )
        self.assertEqual(
            1,
            holder._pp_proxy_drops,
            "the drop is not counted, so the boot log cannot show it happened",
        )

    def test_the_two_early_exits_above_the_wire_still_short_circuit(self):
        """Neither exit may reach the wire; both existed before S0."""
        ppm, holder = self._holder({"hidden_states": "HS"})
        holder.pp_group = types.SimpleNamespace(is_first_rank=True)
        self.assertIsNone(
            ppm.SchedulerPPMixin._pp_recv_proxy_tensors(holder, mb_id=1)
        )
        self.assertEqual([], holder.readiness_calls)

        ppm, holder = self._holder({"hidden_states": "HS"})
        holder._pp_gapped_wire = True
        self.assertIsNone(
            ppm.SchedulerPPMixin._pp_recv_proxy_tensors(holder, mb_id=1)
        )
        self.assertEqual([], holder.readiness_calls)


class TestTheRemovalRefusesLoudlyAtParseTime(unittest.TestCase):
    """§7 item 4 has TWO halves; the slice shipped only the first.

    "**The CLI flags are a compatibility surface.** Decide once: **hard-remove
    with a loud \'flag removed, see WEG2\' refusal at parse time** -- no silent
    shim -- and sweep ``scripts/`` in the same commit."  §11.5 item 5 repeats it.

    What shipped in fix 2 was the removal alone: eleven spellings, including the
    RENAMED carrier flag, produced only argparse\'s generic ``unrecognized
    arguments`` -- the same sentence a typo produces.  It neither says the
    mechanism was removed nor names ``--hicache-canonical-kv-page`` as the new
    spelling of the page Weg 2\'s only carrier rides on, so an operator reading
    it drops the flag and boots without the carrier.
    """

    def test_every_removed_spelling_is_refused_by_name(self):
        from sglang.srt.removed_cli_flags import refuse_removed_flip_flags

        self.assertEqual(
            11,
            len(_REMOVED_FLIP_FLAGS),
            "the shipped refusal no longer covers eleven spellings",
        )
        for flag in _REMOVED_FLIP_FLAGS:
            for argv in (
                ["--model-path", "m", flag],
                ["--model-path", "m", "%s=x" % flag],
            ):
                with self.subTest(argv=argv[-1]):
                    with self.assertRaises(ValueError) as caught:
                        refuse_removed_flip_flags(argv)
                    msg = str(caught.exception)
                    self.assertIn(flag, msg)
                    self.assertIn("#1233", msg)
                    self.assertIn("WEG 2", msg)

    def test_the_renamed_flag_names_its_replacement(self):
        from sglang.srt.removed_cli_flags import refuse_removed_flip_flags

        with self.assertRaises(ValueError) as caught:
            refuse_removed_flip_flags(["--phase-flip-canonical-kv-page"])
        self.assertIn("--hicache-canonical-kv-page", str(caught.exception))

    def test_a_clean_launch_line_is_untouched(self):
        """The can-it-fail proof\'s other half: it must not refuse everything.

        The reference launch line of the stock PP=3 boot goes through
        unchanged; a refusal that fires on a clean argv is a serving killer of
        its own.
        """
        from sglang.srt.removed_cli_flags import refuse_removed_flip_flags

        refuse_removed_flip_flags(
            [
                "--model-path", "m", "--tp-size", "1", "--pp-size", "3",
                "--hicache-canonical-kv-page", "--enable-hierarchical-cache",
                "--pp-stage-ratio", "2,1,1",
            ]
        )

    def test_the_refusal_is_wired_before_the_parser_sees_the_argv(self):
        """A refusal nothing calls is a comment.

        Pinned on ``prepare_server_args`` because that is the ONE choke point
        every launcher shares (``launch_server.py:68``, ``cli/serve.py:139``,
        ``multimodal_gen/runtime/launch_server.py:783``), and pinned as an
        ORDER: after the ``--config`` merge, so a flag arriving from a config
        file is refused on the same terms, and before ``parse_args``.
        """
        src = (_SRT / "server_args.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "prepare_server_args"
        )
        call_line = merge_line = parse_line = None
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and (
                node.func.id == "refuse_removed_flip_flags"
            ):
                call_line = node.lineno
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "merge_config_with_args":
                    merge_line = node.lineno
                if node.func.attr == "parse_args":
                    parse_line = node.lineno
        self.assertIsNotNone(call_line, "prepare_server_args never refuses")
        self.assertIsNotNone(merge_line)
        self.assertIsNotNone(parse_line)
        self.assertLess(merge_line, call_line, "the config merge runs after it")
        self.assertLess(call_line, parse_line, "argparse sees the argv first")


#: Every file under ``scripts/`` that still SPELLS a removed flag, with a count,
#: asserted by exact equality.  §7 item 4 orders the sweep "in the same commit";
#: what a sweep can and cannot mean is decided per file and stated here:
#:
#: * The five BOOT/RELAUNCH recipes each carry a ``#1233`` refusal with
#:   ``exit 1`` ahead of every removed spelling, asserted structurally below.
#:   Their bodies are left byte-for-byte: ``route_a_631_prod_boot.sh`` is shaped
#:   by the flip end to end (the ``SGLANG_PHASE_POLICY_*`` tunables, the
#:   30,17,17 TP vector, a BAR1 aperture split that exists because the flip
#:   builds ``flip_tp:0``/``flip_dcp:0`` beside ``world:0``/``pp:0``), so
#:   deleting five argv lines would leave a recipe that boots something nobody
#:   sized.  A half-swept recipe that STARTS is worse than one that refuses.
#:   ``route_a_631_ava_boot.sh`` is the non-flip recipe of that same family and
#:   is named in the refusal as the replacement; it is now clean.
#: * ``s34_extract.sh`` echoed the SPEC-ITEM table of an archived #656
#:   acceptance run and is now CLEAN: its two mentions were instructions
#:   ("--phase-flip-purity strict") and were reworded into a record of what
#:   that boot ran, which is why it does not appear below.
#: * ``cert_485/certify_485.py`` no longer ASKS the parser for the flag; the one
#:   mention is the comment saying why the entry left.
_SCRIPT_FLAG_MENTIONS = {
    "scripts/cert_485/certify_485.py": 1,
    "scripts/int8_368/boot_855_gdncov.sh": 8,
    "scripts/route_a_631_flip_boot.sh": 3,
    "scripts/route_a_631_prod_boot.sh": 8,
    "scripts/s29_reboot_depth_draft.sh": 3,
    "scripts/s30_reboot_corridor_guard.sh": 2,
}

#: The recipes that must refuse before they can reach a removed spelling.
_REFUSING_SCRIPTS = (
    "scripts/int8_368/boot_855_gdncov.sh",
    "scripts/route_a_631_flip_boot.sh",
    "scripts/route_a_631_prod_boot.sh",
    "scripts/s29_reboot_depth_draft.sh",
    "scripts/s30_reboot_corridor_guard.sh",
)


class TestTheScriptsSweep(unittest.TestCase):
    def test_the_set_of_scripts_naming_a_removed_flag_is_the_named_one(self):
        found = {}
        for path in sorted((_REPO_ROOT / "scripts").rglob("*")):
            if not path.is_file():
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            n = sum(
                1
                for line in raw.splitlines()
                if any(f in line for f in _REMOVED_FLIP_FLAGS)
            )
            if n:
                found[str(path.relative_to(_REPO_ROOT))] = n
        self.assertEqual(
            _SCRIPT_FLAG_MENTIONS,
            found,
            "the scripts/ sweep moved.  A NEW entry is a launch line that now "
            "hard-refuses at parse time; a MISSING one means a sweep landed "
            "and this map must shrink with it.",
        )

    def test_no_script_reaches_a_removed_flag_without_refusing_first(self):
        """The PROPERTY: a recipe may still SPELL the flag, never PASS it.

        The counts above say what is left; this says none of it is reachable.
        The refusal must carry ``#1233`` and an ``exit 1``, and both must sit
        before the first line that uses a removed spelling outside a comment.
        """
        for rel in _REFUSING_SCRIPTS:
            with self.subTest(script=rel):
                lines = (_REPO_ROOT / rel).read_text(encoding="utf-8").splitlines()
                exits = [
                    i
                    for i, line in enumerate(lines)
                    if line.strip() == "exit 1"
                ]
                self.assertNotEqual([], exits, "%s never refuses" % rel)
                refusal = min(exits)
                self.assertTrue(
                    any("#1233" in t for t in lines[max(0, refusal - 30) : refusal]),
                    "%s exits without naming #1233; a bare exit is not a "
                    "refusal an operator can act on" % rel,
                )
                uses = [
                    i
                    for i, line in enumerate(lines)
                    if not line.strip().startswith("#")
                    and any(f in line for f in _REMOVED_FLIP_FLAGS)
                ]
                self.assertNotEqual(
                    [], uses, "%s has nothing left to guard; drop it from the "
                    "refusing set instead of pretending" % rel
                )
                self.assertLess(
                    refusal,
                    min(uses),
                    "%s reaches line %d (%s) before it refuses"
                    % (rel, min(uses) + 1, lines[min(uses)].strip()[:80]),
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

    def test_dispatch_event_loop_reaches_the_dispatch_table_on_a_pp_boot(self):
        """The acceptance line is "boots as a stock PP=3 server; R1 READY".

        THE BOOT KILLER THIS EXISTS FOR.  S0 removed the
        ``if server_args.enable_phase_flip:`` guard from ``dispatch_event_loop``
        and left its ``return`` behind at the inner indent, where it re-parented
        into the preceding ``if _pp is not None and world_size > 1:`` block.
        Every PP rank then returned from ``dispatch_event_loop`` WITHOUT
        entering any event loop and fell straight through
        ``run_scheduler_process``'s shutdown ``finally``.

        ``test_pp_mixin_still_carries_the_pp_event_loop`` above cannot see this:
        ``hasattr(SchedulerPPMixin, "event_loop_pp")`` was true the whole time.
        Nothing here scanned CONTROL FLOW, so the gate had to become
        behavioural: call the function and record which loop it entered.
        """
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler import dispatch_event_loop

        class _Stub:
            def __init__(self, pp_size):
                self.pp_group = types.SimpleNamespace(
                    world_size=pp_size, warmup_p2p_pairs=lambda: None
                )
                self._pp_p2p_warmed = True
                self.server_args = types.SimpleNamespace(pp_size=pp_size)
                self.disaggregation_mode = DisaggregationMode.NULL
                self.enable_pdmux = False
                self.enable_overlap_mlx = False
                self.enable_overlap = False
                self.entered = []

            def event_loop_pp(self):
                self.entered.append("pp")

            def event_loop_normal(self):
                self.entered.append("normal")

            def event_loop_overlap(self):
                self.entered.append("overlap")

            def event_loop_overlap_mlx(self):
                self.entered.append("overlap_mlx")

            def event_loop_pdmux(self):
                self.entered.append("pdmux")

        pp = _Stub(pp_size=3)
        dispatch_event_loop(pp)
        self.assertEqual(
            ["pp"],
            pp.entered,
            "dispatch_event_loop returned without entering event_loop_pp on a "
            "pp_size=3 boot -- every PP rank falls through to shutdown",
        )

        # The mirror, so the probe is proved sensitive rather than trivially
        # green: pp_size==1 must reach the SAME table and take the other arm.
        single = _Stub(pp_size=1)
        dispatch_event_loop(single)
        self.assertEqual(["normal"], single.entered)

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
    #:
    #: ``phase_policy_hook`` is now pinned too: it is a KEYWORD ARGUMENT, the
    #: shape ``_literal_assignments`` was blind to, and the hook is live on the
    #: intake path (``request_receiver.py:196``/``:205``) rather than dead
    #: plumbing.  The seventh named value -- "warmup window shut" -- is NOT
    #: pinnable and is not claimed to be: the post-cutover JIT warmup was
    #: deleted outright rather than collapsed to a value, so there is no
    #: surviving expression to pin (``git diff aef3ae7676..HEAD`` removes the
    #: ``cold_build_window`` import and its whole wrapper; ``grep -n "_warm"``
    #: on the tip leaves only the sampler warmup and the p2p warm flag, neither
    #: flip-related).
    _COLLAPSED = (
        ("init_parked_decode_set", "want", False),
        ("init_request_receiver", "pp_flip_counters", None),
        ("init_request_receiver", "pp_chain_receiver", None),
        ("init_request_receiver", "phase_policy_hook", None),
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

    def test_the_armed_slot_hold_is_gone_from_the_loop_and_from_the_mixin(self):
        """The hold family had ONE premise -- an armed window -- and it is gone.

        S0 collapsed ``_pp_flip_hold_slot`` to an unconditional
        ``return self._1173_forget_stashed_frame()`` but left the CALL in
        ``_event_loop_pp_body``, re-parented under ``if not
        self.pp_group.is_last_rank:`` -- a rank-asymmetric ``continue``
        immediately in front of the #753 per-iteration lockstep barrier.  Taken
        on non-last ranks only, that ``continue`` skips a barrier the last rank
        still enters: a group hang, inert only because of the unconditional
        return nothing stated or tested.

        The whole family therefore goes: the hold, the #1173 D2b
        launched-pass GROUP STOP it was the only caller of, and their shared
        bookkeeping.  That removal is booked by name in
        ``WEG2_BUILD_DECISIONS_0906.md`` (S0 FIXER round 2), because a
        ranks-never-disagree stop must never leave as dead code.
        """
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        for sym in (
            "_pp_flip_hold_slot",
            "_pp_flip_stashed_frame_forces_advance",
            "_1173_forget_stashed_frame",
        ):
            self.assertFalse(
                hasattr(SchedulerPPMixin, sym),
                "SchedulerPPMixin.%s is back: its premise (an armed window) "
                "does not exist in Weg 2" % sym,
            )

        # And the name must not survive in EXECUTABLE source either -- the call
        # site in `_event_loop_pp_body` is the one that mattered, and it sat
        # immediately in front of the #753 barrier.  Comments are excluded the
        # same way the token gate excludes them: prose is not knowledge the
        # program has.
        mixin = _SRT / "managers" / "scheduler_pp_mixin.py"
        hits = [
            (i, t)
            for i, t in _executable_source_lines(mixin)
            if "_pp_flip_hold_slot" in t
        ]
        self.assertEqual([], hits, "the hold is still called: %s" % hits)

    def test_the_surviving_pp_flip_family_is_exactly_the_named_set(self):
        """``_FLIP_TOKEN`` is blind to the ``_pp_flip_*`` methods, by design.

        ``phase[_-]flip`` does not match ``_pp_flip_hold_slot``, so the whole
        surviving family was a blind spot for all three static gates.  Rather
        than widen the regex -- which would go red on every one of these,
        including the wire-ring accessors the stock PP loop needs -- the family
        is ENUMERATED, and the enumeration is asserted by exact equality so a
        new ``_pp_flip_*`` method cannot appear unnoticed in a later slice.
        """
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        family = {
            n
            for n in vars(SchedulerPPMixin)
            if n.startswith("_pp_flip") or n.startswith("pp_flip")
        }
        self.assertEqual(
            {
                # the PP wire ring: rank/size and the two neighbours, read by
                # every send and receive of the stock loop
                "_pp_flip_ring",
                "_pp_flip_upstream",
                "_pp_flip_downstream",
                # the wire's own counters
                "_pp_flip_bump_sent",
                "_pp_flip_bump_attempted",
                "_pp_flip_bump_consumed",
                "_pp_flip_pass_tick",
                # the wire's drains and flushes, all on the stock loop
                "pp_flip_consume_inbound",
                "pp_flip_drain_leftover_dicts",
                "pp_flip_flush_drained_sends",
                "pp_flip_flush_pending_dict_sends",
                "pp_flip_retire_undeclared_stash",
                "pp_flip_channels_empty",
                # the proxy stamp's epoch component, collapsed to None
                "_pp_flip_epoch",
            },
            family,
            "the surviving _pp_flip_* family moved; every entry must be a "
            "mechanism the stock PP loop needs, or it belongs to the cut",
        )

    def test_the_proxy_epoch_is_rank_independent(self):
        """RANK-DIVERGENT is the direction S0 made silent, so it is pinned.

        ``_pp_flip_epoch`` is stamped into the proxy frame by the SENDER
        (``scheduler_pp_mixin.py:_pp_proxy_stamp``) and compared by the
        RECEIVER (``pp_proxy_stamp_names_pass``).  A rank-derived value there
        makes every cross-rank comparison mismatch -- and S0 rewrote that
        mismatch branch into an unconditional bypass, so the failure mode is a
        silently DROPPED proxy frame rather than a stop.

        Measured 2026-09-07: ``return int(getattr(self, "pp_rank", 0))`` in
        place of ``return None`` left the whole slice file green (17 passed, 39
        subtests).  Two assertions close it: the value is identical on two
        ranks, and a stamp made by one rank names the pass another rank is on.
        """
        from sglang.srt.managers import scheduler_pp_mixin as ppm

        epochs = set()
        for rank in (0, 2):
            stub = types.SimpleNamespace(pp_rank=rank, ps=types.SimpleNamespace(pp_rank=rank))
            epochs.add(ppm.SchedulerPPMixin._pp_flip_epoch(stub))
        self.assertEqual(
            {None},
            epochs,
            "_pp_flip_epoch is rank-derived: two ranks produced %r, so every "
            "cross-rank proxy stamp comparison mismatches" % (sorted(map(str, epochs)),),
        )

        # The consumer half, pushed through the real predicate: a stamp made on
        # a peer's epoch must name this rank's pass on the same slot.
        peer_epoch = ppm.SchedulerPPMixin._pp_flip_epoch(
            types.SimpleNamespace(pp_rank=1, ps=types.SimpleNamespace(pp_rank=1))
        )
        mine = ppm.SchedulerPPMixin._pp_flip_epoch(
            types.SimpleNamespace(pp_rank=0, ps=types.SimpleNamespace(pp_rank=0))
        )
        stamp = (2, 17, 119, -1 if peer_epoch is None else peer_epoch)
        self.assertTrue(ppm.pp_proxy_stamp_names_pass(stamp, 2, mine))
        # Can-fail: a different SLOT is still refused, so the predicate is not
        # answering True to everything.
        self.assertFalse(ppm.pp_proxy_stamp_names_pass(stamp, 1, mine))

    def test_the_named_return_collapses_are_still_flag_off_literals(self):
        """The ``return``-shaped half of the over-cut detector.

        Same argument as the assignment pins above, for the shape the mixin
        uses: a flip branch promoted back into a ``return`` is an expression,
        never a literal, so ``_NOT_A_LITERAL`` here IS the failure.
        """
        mixin = _SRT / "managers" / "scheduler_pp_mixin.py"
        found = _literal_assignments(mixin, "_pp_flip_epoch", _RETURN_TARGET)
        self.assertNotEqual([], found, "no return statement in _pp_flip_epoch")
        for lineno, value in found:
            self.assertIsNone(
                value,
                "scheduler_pp_mixin.py:%d returns %r from _pp_flip_epoch, not "
                "the flag-off None" % (lineno, value),
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
