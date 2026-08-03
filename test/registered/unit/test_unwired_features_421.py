"""Task #421 pins: features that are BUILT but not WIRED.

Each test below pins a *specific, verified* reachability fact that the #421
audit established (see ``docs/dev/AUDIT_421_UNWIRED.md``). They are written
inverted on purpose: they assert that the feature is still unreachable.

WHEN ONE OF THESE FAILS, that is the good outcome -- somebody wired the
feature. The correct response is to delete the pin (and its entry in the
audit document), NOT to widen it. A pin that is silently relaxed re-creates
exactly the failure mode the audit exists to catch.

Why pin an absence at all: this fork has twice shipped a feature whose
parameter existed, whose tests passed, and which no production call path ever
reached -- the #197 escape hatch that ``lm_head`` ignored, and the #394
cold-shard apportionment that no caller supplied a ratio to. Both were
invisible to the test suite because the tests called the feature directly.
A reachability assertion is the only kind of test that can see the gap
between "the unit works" and "the product uses it".

These are source-structure assertions (AST over the repo tree), not imports:
hermetic, no torch, no CUDA, no GPU.
"""

import ast
import pathlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SRT = _REPO_ROOT / "python" / "sglang" / "srt"


def _is_test_path(path: pathlib.Path) -> bool:
    rel = path.relative_to(_REPO_ROOT).as_posix()
    return (
        rel.startswith("test/")
        or "/test/" in "/" + rel
        or path.name.startswith("test_")
        or "/benchmark" in "/" + rel
        or rel.startswith("benchmark/")
    )


def _production_py_files():
    """Every .py under python/sglang that is not itself test/bench code."""
    for path in (_REPO_ROOT / "python" / "sglang").rglob("*.py"):
        if not _is_test_path(path):
            yield path


def _parse(path: pathlib.Path):
    return ast.parse(path.read_text(errors="ignore"), filename=str(path))


def _calls_named(tree, func_name):
    """Yield every ast.Call in ``tree`` whose callee is ``func_name``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = (
            callee.id
            if isinstance(callee, ast.Name)
            else callee.attr if isinstance(callee, ast.Attribute) else None
        )
        if name == func_name:
            yield node


def _production_callers_of(func_name, defining_rel_paths=()):
    """Production files (outside the defining module) that call ``func_name``."""
    hits = []
    skip = set(defining_rel_paths)
    for path in _production_py_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in skip:
            continue
        try:
            tree = _parse(path)
        except SyntaxError:
            continue
        for call in _calls_named(tree, func_name):
            hits.append(f"{rel}:{call.lineno}")
    return hits


def _production_importers_of(dotted_module, exclude_package=False):
    """Production files that import ``dotted_module`` (from-import or import).

    ``exclude_package`` drops the module's own package directory: a package
    ``__init__.py`` that re-exports its submodules is not a consumer of the
    feature, it is part of it. Counting it would make every packaged feature
    look wired.
    """
    hits = []
    own = "python/" + dotted_module.replace(".", "/") + ".py"
    pkg = own.rsplit("/", 1)[0] + "/"
    for path in _production_py_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel == own or (exclude_package and rel.startswith(pkg)):
            continue
        try:
            tree = _parse(path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == dotted_module:
                hits.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == dotted_module:
                        hits.append(f"{rel}:{node.lineno}")
    return hits


# RETIRED PINS -- #421 F1 and F2 are FIXED (task #428).
#
# ``TestKvPressureLadderAutoIsUnreachable`` and
# ``TestOffloadRegisterProfileIsUnreachable`` lived here. Both asserted an
# absence that no longer holds, and the pins' own instruction is to delete
# rather than widen them:
#
# * F1 -- ``--kv-pressure-ladder auto`` now gets the #272 planner's table
#   injected at ``managers/kv_pressure_runtime.py``. Replaced by
#   ``test/registered/unit/managers/test_kv_ladder_auto_421.py``, which
#   asserts the POSITIVE fact (the runtime builds) at the same call site.
# * F2 -- the three ``--lane-offload-*`` flags now reach the register through
#   ``configure_global_register_from_server_args`` at ModelRunner init.
#   Replaced by
#   ``test/registered/unit/model_executor/test_offload_register_wiring_421.py``,
#   which additionally pins the CALL SITE, so a refactor cannot move the
#   configure step after the first adapter read and silently restore the
#   fallback register.
#
# Both replacements carry a can-fail proof (see their module docstrings).


class TestRuntimeDraftLifecycleIsUnreachable(CustomTestCase):
    """#309 runtime drafter attach/detach exists but nothing drives it.

    ``speculative/runtime_draft.py`` implements the full state machine, is
    covered by 65 hermetic tests and documented in
    ``docs/dev/TASK_309_RUNTIME_DRAFT.md``. Its own docstring says the weight
    load and VRAM return "are executed by the scheduler at the boundary this
    machine hands them" -- but no scheduler code imports it, so no operator
    can attach or detach a drafter on a running server.
    """

    def test_no_production_importer(self):
        importers = _production_importers_of("sglang.srt.speculative.runtime_draft")
        self.assertEqual(
            importers,
            [],
            "GOOD NEWS: the #309 lifecycle is now driven by production code "
            f"({importers}). #421 finding F3 is fixed -- delete this pin.",
        )

    def test_arm_selection_helper_has_no_production_caller(self):
        """``draft_selection.arms_from_server_args`` is the #309 sibling."""
        callers = _production_callers_of(
            "arms_from_server_args",
            defining_rel_paths=("python/sglang/srt/speculative/draft_selection.py",),
        )
        self.assertEqual(
            callers,
            [],
            "GOOD NEWS: arm selection is now reached from production "
            f"({callers}). #421 finding F3 is fixed -- delete this pin.",
        )


# RETIRED PIN -- #421 F4 is FIXED (task #394 slice 2).
#
# ``TestColdTierShmIsUnreachable`` asserted that
# ``layers/moe/cold_tier_shm.py`` had no production importer. It has one now:
# ``layers/moe/cold_tier_fetch.py`` is the routing half, reached from the GGUF
# streaming door (``fused_moe_triton/layer.py``) and from the launcher
# (``entrypoints/engine.py``). Per this file's own rule the pin is deleted
# rather than widened, and replaced by
# ``test/registered/unit/moe/test_cold_tier_wiring_394.py``, which asserts the
# POSITIVE fact and pins the CALL SITES -- so a refactor cannot quietly drop
# the fetch route and leave the module importable but unreached again.
#
# The replacement carries can-fail proofs (see its module docstring), and the
# hermetic behaviour of the chain is covered by
# ``tests/moe_offload/test_cold_tier_fetch.py``.


# RETIRED PIN -- #421 F6 is FIXED (task #286).
#
# ``TestMemTierRegistryHasNoConsumers`` asserted that ``srt/memtier/`` had zero
# production importers, i.e. that the catalog's "all new spill/offload
# consumers must pick targets from it" was aspirational rather than enforced.
#
# ``model_executor/short_term_offload_register.py`` is the first production
# consumer: it imports ``memtier.registry`` and ``memtier.tiers`` at module
# scope and picks its park target through ``TierRegistry.select`` rather than
# from ``offload_register.PARK_TARGETS``. The pin therefore fires, and per this
# module's own rule it is deleted rather than widened.
#
# The replacement asserts the POSITIVE fact and pins the call site -- the same
# treatment #394's cold-tier pin got above -- in
# ``test/registered/unit/model_executor/test_short_term_offload_register.py``
# (``MemTierIsNowWiredTest``): the module-scope imports are pinned by AST, and
# the priced target is pinned to be a ``TierId`` that is NOT one of the three
# hand-written ``PARK_TARGETS`` strings.
#
# What is NOT yet true, and is stated in ``FEATURE_CATALOG.md`` §3 rather than
# pinned here: the OTHER consumers (expert offload, the #394 cold tier, the
# rest of the #286 park-target ladder) still carry their own target lists.
# One consumer is not the reconciliation.


if __name__ == "__main__":
    unittest.main()
