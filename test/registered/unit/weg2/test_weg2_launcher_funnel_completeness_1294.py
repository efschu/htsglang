# SPDX-License-Identifier: Apache-2.0
"""#1294: two post-spawn raises bypassed BOTH funnels as raw tracebacks.

THE DEFECT (found reading #1248's own "out of scope, named exactly" note,
SECTION 1at of WEG2_BUILD_DECISIONS_0906.md): ``chunk_tag_cards()``
(``weg2_memory_saver.py``, called from ``main()`` while building the flip's
chunk-tag-to-card map) and ``route_floor()`` (``carrier_census.py``, called
from ``main()``'s carrier-census block) each raised a BARE builtin
(``ValueError``, ``RuntimeError``) that is neither in ``REFUSALS`` nor named
``Weg2*`` -- #1248's teardown-on-refusal fix (SECTION 1at) does not see
either raise, so a boot that hits one leaves the sglang groups alive on the
cards exactly like the pre-#1248 defect, just from two different sites.

THE FIX (``launcher.py``): ``Weg2ChunkCardMismatch(Weg2LaunchRefused,
ValueError)`` (W52) and ``Weg2CarrierFloorUnreachable(Weg2LaunchRefused)``
(W54), raised at the two sites via a deferred import (the leaf module
imports the control-plane class only inside the failing branch, so the
launcher's import graph is paid for only on that path). Multiple
inheritance on the first class only, because it is the only one of the two
an EXISTING test asserts a builtin type against
(``test_weg2_flip_order_1233.py::TestChunkTagCards::test_card_list_that_does_not_match_the_stages_is_refused``,
``assertRaises(ValueError)``) -- checked and left untouched, since
``Weg2ChunkCardMismatch`` IS a ``ValueError`` too. ``route_floor``'s branch
has no such caller (checked: no ``assertRaises``/``RuntimeError`` anywhere in
``test_weg2_1246_carrier_census.py``), so its class needs no second base.

THE RATCHET (the class, not the instance). Rather than re-enumerate a fixed
set of good names the way #1248's own ``WEG2_REFUSAL_NAMES`` allowlist does
(a NEW raise site would simply not appear in it, silently), this file asks
the GENERIC question at every raise site on the reachable path: does the
raised name resolve, at runtime, to a class that IS ``Weg2LaunchRefused`` or
a subclass of it (or is the raise a bare re-raise)? An unresolvable name
fails CLOSED (counts as a violation), so an allowlist gap cannot hide a new
bare raise either. Scope, matching the task's own bound: ``main()`` after
the dry-return line, the same ``NAMED_POST_SPAWN_HELPERS`` #1248's own test
enumerates (reachability for that set is #1248's proof, not re-derived
here), plus the two cross-module functions this ticket's two sites live in
(``weg2_memory_saver.chunk_tag_cards``, ``carrier_census.route_floor``) --
reachability for THOSE two is proven fresh below
(``TestCrossModuleSitesAreReachable``), since #1248's own inventory predates
this ticket and does not cover them.

RED-FIRST: at ``6dff25c7c3`` (before this ticket's two edits),
``TestEveryReachableRaiseIsAFunnelSubtype.test_every_raise_on_the_reachable_path_is_compliant``
was RED -- the real (not synthetic) ``raise ValueError(...)`` in
``chunk_tag_cards`` and ``raise RuntimeError(...)`` in ``route_floor`` are
exactly the two violations it found, both failing closed because neither
builtin name resolves against ``Weg2LaunchRefused``'s known homes. See
SECTION 1at-b for the run id and evidence.

Hermetic: ``CUDA_VISIBLE_DEVICES=""`` and no server; both fixed functions are
pure (geometry / bisection only, no NVML or CUDA), so the behavioral class
below calls them for real.
"""

import ast
import inspect
import os
import textwrap
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_memory_saver
from sglang.srt.weg2 import carrier_census
from sglang.srt.weg2 import front as front_mod
from sglang.srt.weg2 import launcher as L
from sglang.test.test_utils import CustomTestCase

#: Mirrors ``test_weg2_launcher_teardown_1248.NAMED_POST_SPAWN_HELPERS``
#: exactly. #1248's own reachability proof
#: (``test_the_named_helper_set_is_reachable_after_the_dry_return``) already
#: covers this set from ``main()``'s side; not re-derived here, and not
#: imported from that sibling file either -- there is no precedent anywhere
#: in this test directory for one weg2 test file importing symbols from
#: another (checked), so this stays a small, self-contained local copy
#: instead of the first such cross-file dependency. A change to the set
#: fails #1248's own test first, which is the actual guard on drift.
NAMED_POST_SPAWN_HELPERS = (
    "wait_ready",
    "sleep_group",
    "d_tp_ratio_decision",
    "build_env",
    "gate_w11",
    "check_drafter_identity",
    "launch_group",
)

#: The two #1294 cross-module sites: (owning module, function name).
CROSS_MODULE_HELPERS = (
    (weg2_memory_saver, "chunk_tag_cards"),
    (carrier_census, "route_floor"),
)


# --------------------------------------------------------------------------
# AST helpers. Shapes mirror test_weg2_launcher_teardown_1248.py's own
# helpers of the same name; `_all_raise_lines` differs on purpose from that
# file's `_raise_lines` (which filters by a known-good name set): the whole
# point of a ratchet against an EMPTY allowlist is to see every raise, not
# only the ones already expected.
# --------------------------------------------------------------------------

def _fn_ast(fn) -> ast.Module:
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


def _fn_ast_with_offset(fn):
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


def _all_raise_lines(tree: ast.AST) -> list:
    """``[(lineno, name_or_None), ...]`` for EVERY ``ast.Raise`` in ``tree``.

    ``name`` is ``None`` for a bare ``raise`` (a re-raise). Never filtered by
    a fixed set -- an unenumerated name must still show up here for the
    ratchet to see it.
    """
    return [
        (n.lineno, _raise_target_name(n))
        for n in ast.walk(tree)
        if isinstance(n, ast.Raise)
    ]


def _call_lines(tree: ast.AST, name: str) -> list:
    return [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _callee_name(n) == name
    ]


def _main_tree_and_offset():
    return _fn_ast_with_offset(L.main)


def _dry_return_line() -> int:
    tree, offset = _main_tree_and_offset()
    fn = tree.body[0]
    for n in fn.body:
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "dry":
            for stmt in reversed(n.body):
                if isinstance(stmt, ast.Return):
                    return stmt.lineno + offset
    raise AssertionError("main() no longer has a top-level `if dry:` branch")


def _resolve_class(name: str):
    """Look ``name`` up among the modules a raise inside this file's scope
    could plausibly name: the launcher module itself, plus the two
    submodules ``launcher.py`` imports at module level and whose refusal
    bases ``main()``/its helpers are known to raise directly (``ring_table``,
    ``host_ledger`` -- the same pair #1248's own ``REFUSALS`` tuple names
    beside ``Weg2LaunchRefused``). ``None`` if no such module carries a class
    of that name -- UNRESOLVED, which the caller treats as a violation, not
    a silent pass.
    """
    homes = [L]
    for attr in ("ring_table", "host_ledger"):
        mod = getattr(L, attr, None)
        if mod is not None:
            homes.append(mod)
    for home in homes:
        cls = getattr(home, name, None)
        if isinstance(cls, type):
            return cls
    return None


def _raise_is_compliant(name) -> bool:
    """``True`` for a bare re-raise (``name is None``) or a name that
    resolves to ``Weg2LaunchRefused`` itself or a subclass. ``False`` -- a
    violation -- for everything else, INCLUDING a name this function cannot
    resolve at all: the ratchet fails closed rather than allow-listing an
    unknown by omission.
    """
    if name is None:
        return True
    cls = _resolve_class(name)
    if cls is None:
        return False
    return issubclass(cls, L.Weg2LaunchRefused)


# --------------------------------------------------------------------------
# Reachability: the two #1294 cross-module sites are actually on the path.
# --------------------------------------------------------------------------

class TestCrossModuleSitesAreReachable(CustomTestCase):
    """#1248's own ``TestDryRunNeverReachesRealSpawnOrCensus`` already proves
    ``route_floor`` is called after the dry-return
    (``test_the_dry_branch_ends_before_the_carrier_census_call``); this class
    adds the analogous proof for ``chunk_tag_cards``, which #1248's own
    inventory predates and does not cover, plus a same-file re-check of
    ``route_floor`` so this file's own ratchet does not silently depend on a
    fact only proven in a sibling module.
    """

    def test_chunk_tag_cards_is_called_after_the_dry_return(self):
        tree, offset = _main_tree_and_offset()
        dry_ret = _dry_return_line()
        calls = [ln + offset for ln in _call_lines(tree, "chunk_tag_cards")]
        self.assertTrue(calls, "chunk_tag_cards() is no longer called from main()")
        self.assertGreater(
            min(calls), dry_ret,
            "chunk_tag_cards() must be called after the dry-return",
        )

    def test_route_floor_is_called_after_the_dry_return(self):
        tree, offset = _main_tree_and_offset()
        dry_ret = _dry_return_line()
        calls = [ln + offset for ln in _call_lines(tree, "route_floor")]
        self.assertTrue(calls, "route_floor() is no longer called from main()")
        self.assertGreater(
            min(calls), dry_ret,
            "route_floor() must be called after the dry-return",
        )


# --------------------------------------------------------------------------
# THE RATCHET: every raise on the reachable path names a funnel subtype.
# --------------------------------------------------------------------------

class TestEveryReachableRaiseIsAFunnelSubtype(CustomTestCase):

    def _violations(self):
        violations = []

        # main(), post-dry-return only (mirrors #1248's own spawn/dry-return
        # split -- main()'s PRE-spawn raises are out of scope for this
        # ticket, which is specifically about post-spawn teardown coverage).
        tree, offset = _main_tree_and_offset()
        dry_ret = _dry_return_line()
        for lineno, name in _all_raise_lines(tree):
            real_line = lineno + offset
            if real_line > dry_ret and not _raise_is_compliant(name):
                violations.append(("main", real_line, name))

        # The launcher-local named helpers #1248 already proved reachable.
        for helper_name in NAMED_POST_SPAWN_HELPERS:
            fn = getattr(L, helper_name)
            for lineno, name in _all_raise_lines(_fn_ast(fn)):
                if not _raise_is_compliant(name):
                    violations.append((helper_name, lineno, name))

        # The two #1294 cross-module sites (reachability proven above).
        for module, fn_name in CROSS_MODULE_HELPERS:
            fn = getattr(module, fn_name)
            for lineno, name in _all_raise_lines(_fn_ast(fn)):
                if not _raise_is_compliant(name):
                    violations.append((f"{module.__name__}.{fn_name}", lineno, name))

        return violations

    def test_every_raise_on_the_reachable_path_is_compliant(self):
        """THE RATCHET. Empty allowlist: a violation is anything that is
        neither a bare re-raise nor a name resolving to ``Weg2LaunchRefused``
        or a subclass. RED at ``6dff25c7c3`` (two violations: ``ValueError``
        in ``chunk_tag_cards``, ``RuntimeError`` in ``route_floor`` -- neither
        name resolves via ``_resolve_class`` at all, so both fail CLOSED);
        GREEN after this ticket's fix. See SECTION 1at-b for the red-first
        run id.
        """
        violations = self._violations()
        self.assertEqual(
            violations, [],
            f"raise(s) on the post-spawn reachable path bypass the funnel: "
            f"{violations}",
        )

    def test_the_two_1294_sites_are_specifically_present_and_compliant(self):
        """Names the two sites directly, so a future refactor that happens to
        keep the total violation count at zero cannot silently swap the two
        intended classes for two different, also-compliant ones without this
        test noticing.
        """
        cc_names = {n for _ln, n in _all_raise_lines(_fn_ast(carrier_census.route_floor))}
        self.assertIn("Weg2CarrierFloorUnreachable", cc_names)

        wms_names = {n for _ln, n in _all_raise_lines(_fn_ast(weg2_memory_saver.chunk_tag_cards))}
        self.assertIn("Weg2ChunkCardMismatch", wms_names)

    def test_the_new_classes_are_funnel_subtypes_and_the_first_also_a_valueerror(self):
        """Runtime confirmation, not just AST name-matching: resolves the two
        new classes and checks the actual MRO, plus the ``ValueError``
        compatibility ``Weg2ChunkCardMismatch`` exists for (the pre-existing
        ``test_weg2_flip_order_1233.py`` assertion).
        """
        self.assertTrue(issubclass(L.Weg2ChunkCardMismatch, L.Weg2LaunchRefused))
        self.assertTrue(issubclass(L.Weg2ChunkCardMismatch, ValueError))
        self.assertTrue(issubclass(L.Weg2CarrierFloorUnreachable, L.Weg2LaunchRefused))
        self.assertIn(L.Weg2LaunchRefused, L.REFUSALS)


# --------------------------------------------------------------------------
# Behavioral: the two sites actually raise the new classes, for real.
# --------------------------------------------------------------------------

class TestTheTwoSitesRaiseForReal(CustomTestCase):
    """Calls the two fixed functions with inputs that trigger their refusal
    branch (both are pure, hermetic, no NVML or CUDA) and checks the caught
    exception's REAL runtime type -- the AST tests above prove the SOURCE
    names the right class; this proves the class that source names is what
    actually comes out at the raise site.
    """

    def test_chunk_tag_cards_raises_weg2_launch_refused_and_stays_a_valueerror(self):
        with self.assertRaises(L.Weg2LaunchRefused) as ctx:
            weg2_memory_saver.chunk_tag_cards([4, 4], 2, 4, card_of_stage=[0])
        self.assertIsInstance(ctx.exception, ValueError)
        self.assertIn("W52 Weg2ChunkCardMismatch", str(ctx.exception))

    def test_route_floor_raises_weg2_launch_refused(self):
        # front.price_remainder mocked to always return remainder=0: with
        # chunk=short_bound=100, `_not_short` (remainder > chunk) can then
        # never be true for any hi, so the bisection seed's doubling runs
        # off route_floor's own "unreachable for any sane divisor" guard
        # deterministically, regardless of CHARS_PER_TOKEN or any other
        # front constant.
        with (
            mock.patch.object(front_mod, "price_remainder", return_value=(0, 0, True)),
            self.assertRaises(L.Weg2LaunchRefused) as ctx,
        ):
            carrier_census.route_floor(short_bound=100)
        self.assertIn("W54 Weg2CarrierFloorUnreachable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
