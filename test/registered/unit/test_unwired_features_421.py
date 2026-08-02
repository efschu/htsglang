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
            else callee.attr
            if isinstance(callee, ast.Attribute)
            else None
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


class TestKvPressureLadderAutoIsUnreachable(CustomTestCase):
    """``--kv-pressure-ladder auto`` cannot be used: no caller injects a table.

    ``build_ladder_from_server_args(server_args, *, table_fn=None)`` raises
    ValueError when the spec is ``auto`` and ``table_fn`` is None. Its help
    text advertises auto as "computed once from the rig/model profile by the
    #272 planner", and argument-time validation accepts the value. But the
    sole production construction site --
    ``managers/kv_pressure_runtime.build_kv_pressure_runtime`` -- calls it
    positionally with no ``table_fn``, so the advertised mode is a guaranteed
    late crash rather than a feature.

    The fix is to inject ``planner.kv_ladder_table.build_ladder_table``
    (which exists, and is tested) at that call site.
    """

    CALL_SITE = "python/sglang/srt/managers/kv_pressure_runtime.py"

    def test_sole_production_caller_still_omits_table_fn(self):
        tree = _parse(_REPO_ROOT / self.CALL_SITE)
        calls = list(_calls_named(tree, "build_ladder_from_server_args"))
        self.assertEqual(
            len(calls),
            1,
            "expected exactly one construction site; the pin needs rewriting "
            f"if that changed (found {len(calls)})",
        )
        kwargs = {kw.arg for kw in calls[0].keywords}
        self.assertNotIn(
            "table_fn",
            kwargs,
            "GOOD NEWS: the auto ladder table is now injected. #421 finding "
            "F1 is fixed -- delete this pin and its audit entry.",
        )

    def test_no_production_module_supplies_the_planner_table_builder(self):
        callers = _production_callers_of(
            "build_ladder_table",
            defining_rel_paths=("python/sglang/srt/planner/kv_ladder_table.py",),
        )
        self.assertEqual(
            callers,
            [],
            "GOOD NEWS: build_ladder_table now has a production caller "
            f"({callers}). #421 finding F1 is fixed -- delete this pin.",
        )


class TestOffloadRegisterProfileIsUnreachable(CustomTestCase):
    """The three ``--lane-offload-*`` flags never reach the offload register.

    ``server_args._handle_lane_offload_register`` parses and validates
    ``--lane-offload-profile``, ``--lane-offload-class-policy`` and
    ``--lane-offload-park-targets``, then discards the resolved values with
    the comment "recomputed at configure time".

    ``configure_global_register(profile, class_policy_overrides, ...)`` is
    that configure-time entry point, and its docstring says it is "called
    once at runner init when the register is enabled". No production module
    calls it. With ``SGLANG_OFFLOAD_REGISTER=1`` the register is instead
    built by ``get_global_register()``'s fallback, which constructs a bare
    ``OffloadRegister()`` on the default (latency) profile -- so the operator's
    profile choice is silently discarded.
    """

    DEFINING = "python/sglang/srt/model_executor/offload_register.py"

    def test_configure_global_register_has_no_production_caller(self):
        callers = _production_callers_of(
            "configure_global_register", defining_rel_paths=(self.DEFINING,)
        )
        self.assertEqual(
            callers,
            [],
            "GOOD NEWS: the offload register is now configured from the CLI "
            f"flags ({callers}). #421 finding F2 is fixed -- delete this pin.",
        )

    def test_park_target_order_never_reaches_the_movement_layer(self):
        """The parsed park order is validated in server_args and dropped.

        Only the defining module and the argument-time validator mention it;
        no runtime consumer takes the operator's order.
        """
        callers = _production_callers_of(
            "parse_park_target_order",
            defining_rel_paths=(self.DEFINING,),
        )
        # server_args.py calls it purely to raise on bad syntax.
        self.assertEqual(
            callers,
            ["python/sglang/srt/server_args.py:6542"],
            "the only production call should be the argument-time validator; "
            f"found {callers}. If a runtime consumer appeared, #421 finding "
            "F2 is (partly) fixed -- re-check and update the pin.",
        )


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


class TestColdTierShmIsUnreachable(CustomTestCase):
    """#394 reachability slice 1: the cold expert tier has no production user.

    ``layers/moe/cold_tier_shm.py`` provides the shared-host-memory segment
    primitive (create/seal/publish/attach/peer views). The merge that landed
    it recorded the placement policy as "inert pending reachability slice 2".
    This pin makes that state visible to the test suite instead of only to a
    commit message.
    """

    def test_no_production_importer(self):
        importers = _production_importers_of("sglang.srt.layers.moe.cold_tier_shm")
        self.assertEqual(
            importers,
            [],
            "GOOD NEWS: the cold tier is now attached from production "
            f"({importers}). #421 finding F4 is fixed -- delete this pin.",
        )


class TestMemTierRegistryHasNoConsumers(CustomTestCase):
    """#407 memory-tier registry: no production code picks tiers from it.

    ``FEATURE_CATALOG.md`` §3 states "All new spill/offload consumers must
    pick targets from it" -- a normative rule. At this tip the package
    ``srt/memtier/`` (registry, tiers, probe, profile, reservations) has zero
    production importers and zero production symbol references; every
    reference outside the package is a unit test. The two offload consumers
    audited under #421 (the #286 offload register and the #394 cold tier)
    both pick targets without it.

    So the rule is aspirational. That is a legitimate state for a freshly cut
    node layer -- the merge said "no consumers yet" -- but a catalog that
    states it as an active constraint invites the next author to assume a
    reconciler exists.
    """

    MEMTIER_MODULES = (
        "sglang.srt.memtier.registry",
        "sglang.srt.memtier.tiers",
        "sglang.srt.memtier.probe",
        "sglang.srt.memtier.profile",
        "sglang.srt.memtier.reservations",
    )

    def test_no_production_importer_of_any_memtier_module(self):
        found = {}
        for dotted in self.MEMTIER_MODULES:
            importers = _production_importers_of(dotted, exclude_package=True)
            if importers:
                found[dotted] = importers
        self.assertEqual(
            found,
            {},
            "GOOD NEWS: the memtier registry now has a production consumer "
            f"({found}). #421 finding F6 is fixed -- delete this pin and "
            "re-check whether the catalog rule is now enforced.",
        )


if __name__ == "__main__":
    unittest.main()
