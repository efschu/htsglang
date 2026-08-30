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


class TestDrafterParkHasNoCaller(CustomTestCase):
    """#309 (c): the drafter PARK half is unwired too, and the docs omit it.

    Added by the #309 remainder determination (2026-08-17). The two pins above
    cover the attach/detach DECISION layer; this covers the other half a reader
    would assume exists, because #286 declares ``drafter_heads`` as an offload
    asset class (``model_executor/offload_register.py:115``) and
    ``dual_group_lane.py:1915-1932`` registers a lane's drafter head into it.

    That registration is real code, so a reader checking only for a descriptor
    concludes the park path is live. It is not, for two independent reasons
    beyond the default-off ``SGLANG_OFFLOAD_REGISTER`` flag:

    * no movement payload is bound -- the registration site says so itself
      ("No payload bind yet ... binding a TensorPayload here would be refused
      by the backend"), so nothing could move even if park were called;
    * ``rung1_evict`` is the ONLY function that parks a ``drafter_heads`` item,
      and nothing calls it; ``AdaptiveGraphStateMover``, the class that would
      move the pages, is never instantiated outside its own definition.

    #286's own commit states it: "DESK-ONLY -- no page has ever moved."
    TASK_309_RUNTIME_DRAFT.md's honest remainder names the #286 register as
    where freed state must be tagged, but does not record that the register's
    own park path has never run -- so the destination reads as available. That
    is the gap this pin closes: prose can go stale, a reachability assertion
    cannot.
    """

    _DEFINER = "python/sglang/srt/model_executor/short_term_offload_register.py"

    def test_nothing_calls_the_only_function_that_parks_a_drafter(self):
        callers = _production_callers_of(
            "rung1_evict", defining_rel_paths=(self._DEFINER,)
        )
        self.assertEqual(
            callers,
            [],
            "GOOD NEWS: the RUNG-1 eviction that parks a drafter head is now "
            f"reached from production ({callers}). #309 (c) is no longer "
            "desk-only -- delete this pin and replace it with a wiring test.",
        )

    def test_the_page_mover_is_never_instantiated_in_production(self):
        callers = _production_callers_of(
            "AdaptiveGraphStateMover", defining_rel_paths=(self._DEFINER,)
        )
        self.assertEqual(
            callers,
            [],
            "GOOD NEWS: the adaptive graph-state mover is constructed in "
            f"production ({callers}). #286's 'no page has ever moved' no "
            "longer holds -- delete this pin and re-check the #309 (c) verdict.",
        )


class TestRealMovementBackendIsNeverConstructed(CustomTestCase):
    """#778/#286: the movement half has a real backend that nothing builds.

    Added 2026-08-30 from the "solved, never shipped" inventory
    (``/spinning/gpu-arb/GELOEST-NIE-GESHIPPT-INVENTAR-0830.md``, table B3).

    THREE COMPOUNDING LAYERS, which is why fixing any one alone changes
    nothing and why this is pinned rather than left to prose:

    1. ``SGLANG_OFFLOAD_REGISTER`` is set by no launcher on this rig.
    2. Even with the gate on, ``offload_register.py:572`` reads
       ``self._backend = backend or CpuFakeMovementBackend()`` -- the register
       falls back to the CPU FAKE whenever no backend is injected, and nothing
       injects one.
    3. ``RealMovementBackend`` is constructed in no production file at all.

    THE PROSE ALREADY DISAGREES WITH ITSELF, which is the reason a
    reachability assertion is the only trustworthy record here:
    ``offload_register.py``'s ``MovementBackend`` docstring states "The real
    backend is ``offload_movement.RealMovementBackend``", while
    ``registry/tick.py`` states "It does not call ``#286``'s
    ``RealMovementBackend``". Exactly one of those is true about production,
    and this test is the one that cannot go stale.

    RETIRE WHEN: a production file constructs ``RealMovementBackend``. Deleting
    this pin is then correct ONLY together with checking layer 2 -- a
    constructor that runs while the register still defaults to the fake leaves
    the spill mover just as dark. Replace with a wiring test that pins the
    injection SITE, the way #421 F2's replacement does.
    """

    _DEFINER = "python/sglang/srt/model_executor/offload_movement.py"

    def test_no_production_file_constructs_the_real_backend(self):
        callers = _production_callers_of(
            "RealMovementBackend", defining_rel_paths=(self._DEFINER,)
        )
        self.assertEqual(
            callers,
            [],
            "GOOD NEWS: the real movement backend is constructed in production "
            f"({callers}). #778/#286 is no longer desk-only -- delete this pin, "
            "and verify the register no longer defaults to "
            "CpuFakeMovementBackend when that construction happens.",
        )


class TestMlpRebalanceAdvisoryIsEmitOnly(CustomTestCase):
    """The uneven-TP self-calibration computes a better vector and can only
    ask a human to restart with it.

    Added 2026-08-30 from the "solved, never shipped" inventory (table B1).
    MEASURED on the standing boot
    (``/spinning/evidence-665-f1/boot_855_gdncovB2_0840f82601_0830_053228.log``
    line 673)::

        [PP0] uneven TP: restart with SGLANG_UNEVEN_MLP_VECTOR=1009,38,41
              to raise the KV pool from 449314 to ~561293 tokens

    That is +111,979 tokens (+24.9%) on the rank that BINDS the MIN-synced
    pool, computed by the server about itself, and discarded when the process
    exits. ``_maybe_suggest_mlp_rebalance`` says so in its own docstring:
    "Purely advisory -- nothing is resized in-process; the hint asks for a
    restart".

    THIS IS THE #797 SHAPE ON A SECOND AXIS. #797 was the same defect on the
    TOKEN vector -- an advisory printed and consumed by nobody -- and it was
    closed by giving the measurement a consumer, which is why the same boot
    shows the measured ownership vector [30,17,17] installed OVER the
    pre-boot estimate. The MLP and MOE axes never got that second half.

    WHAT IS PINNED, precisely: no production file writes either calibration
    environment variable, so there is no in-process path from the advisory
    back into the resolved configuration. The loop can only be closed by a
    human reading a log line and editing a launcher -- and on this rig, for
    the entire life of the flag, nobody has.

    RETIRE WHEN: the in-flight solve-boot lands a consumer for the computed
    vector. At that point this pin goes RED and must be DELETED, not widened;
    replace it with a positive test that the suggested vector reaches the
    resolver, mirroring #797's own replacement.
    """

    _FAMILY_ENVS = ("SGLANG_UNEVEN_MLP_VECTOR", "SGLANG_UNEVEN_MOE_VECTOR")

    def _environ_writes(self, name):
        """Production ``os.environ[name] = ...`` assignments."""
        hits = []
        for path in _production_py_files():
            rel = path.relative_to(_REPO_ROOT).as_posix()
            try:
                tree = _parse(path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                        and target.value.attr == "environ"
                        and isinstance(target.slice, ast.Constant)
                        and target.slice.value == name
                    ):
                        hits.append(f"{rel}:{node.lineno}")
        return hits

    def test_nothing_feeds_the_suggested_vector_back_in_process(self):
        for name in self._FAMILY_ENVS:
            with self.subTest(env=name):
                writes = self._environ_writes(name)
                self.assertEqual(
                    writes,
                    [],
                    f"GOOD NEWS: {name} is now written in production "
                    f"({writes}) -- the self-calibration may have gained an "
                    "in-process consumer. Delete this pin and replace it with "
                    "a positive test that the suggested vector reaches the "
                    "resolver, the way #797 was closed on the token axis.",
                )


# RETIRED PIN -- #421 F4 is FIXED (task #394 slice 2).
#
# ``TestColdTierShmIsUnreachable`` asserted that
# ``layers/moe/cold_tier_shm.py`` had no production importer. It has one now:
# ``layers/moe/cold_tier_fetch.py`` is the routing half, reached from the GGUF
# streaming door (``fused_moe_triton/layer.py``) and from the launcher
# (``entrypoints/engine.py``). Per this file's own rule the pin is deleted
# rather than widened, and replaced by
# ``test/registered/unit/layers/moe/test_cold_tier_wiring_394.py``, which asserts the
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
# #421 finding F6 -- "the memtier registry has zero production consumers" --
# was FIXED by #410 (server-side session checkpoints), which resolves its
# checkpoint tier through ``memtier.consumers.checkpoint_tier_targets`` on a
# real control request. The inverted pin that asserted the absence is
# therefore retired, and replaced below by the POSITIVE pin that keeps the
# call site from quietly disappearing again -- the same substitution the #394
# cold tier made when it was wired.
#
# The catalog rule ("all new spill/offload consumers must pick targets from
# it") is still not enforced for the PRE-EXISTING consumers: the #286 offload
# register and the #394 cold tier both still carry their own target lists,
# and migrating them is memtier cuts 4 and 5. That remains an open item in
# ``docs/dev/AUDIT_421_UNWIRED.md``; what changed is only that the rule now
# has one consumer honouring it instead of none.


class TestMemTierRegistryHasItsFirstConsumer(CustomTestCase):
    """#407 memory-tier registry: #410 picks its checkpoint tier from it.

    Pins the CALL SITE, not the module: a refactor that leaves
    ``consumers.py`` importable but unreached would restore F6 without
    failing any other test in the suite, which is exactly how the #197 escape
    hatch and the #394 apportionment stayed invisible.
    """

    CONSUMER_MODULE = "sglang.srt.memtier.consumers"
    EXPECTED_CALLER = "python/sglang/srt/managers/session_checkpoint.py"

    def test_the_checkpoint_runtime_imports_the_consumer_shim(self):
        importers = _production_importers_of(self.CONSUMER_MODULE, exclude_package=True)
        self.assertTrue(
            any(hit.startswith(self.EXPECTED_CALLER) for hit in importers),
            "#410's checkpoint runtime no longer imports "
            f"{self.CONSUMER_MODULE}. If the tier selection moved, move this "
            "pin with it; do NOT delete it -- an unreached registry is #421 "
            f"finding F6 all over again. Importers found: {importers}",
        )

    def test_the_checkpoint_runtime_calls_the_selection_helper(self):
        callers = _production_callers_of(
            "checkpoint_tier_targets",
            defining_rel_paths=("python/sglang/srt/memtier/consumers.py",),
        )
        self.assertTrue(
            any(hit.startswith(self.EXPECTED_CALLER) for hit in callers),
            "nothing in production calls checkpoint_tier_targets any more; "
            "the #410 checkpoint would then be placing bytes without asking "
            f"the registry. Callers found: {callers}",
        )


if __name__ == "__main__":
    unittest.main()
