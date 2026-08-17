"""Ratchet guard: server_args mutations outside the resolution pipeline.

After ``ServerArgs.__post_init__`` returns, the instance carries the resolved
configuration. Every audited runtime adjustment goes through
``ServerArgs.override(source, **fields)`` -- the single mutation entry point,
which records provenance and keeps whitelisted fields consistent with the
declaration stash. A bare assignment to a resolved ServerArgs is the violation
this file catches.

WHY THIS FILE IS AN ALLOWLIST AND NOT A SINGLE COUNT
====================================================
The upstream form of this test pinned one integer, ``_BASELINE = 0``, and
excluded the pipeline BY PATH (``srt/server_args.py``, ``srt/arg_groups/``).
That worked upstream, where resolution lives entirely in those two paths. This
fork moved substantial resolution logic OUT of them -- uneven-TP planning into
``srt/uneven_perf.py``, PD topology into ``srt/disaggregation/topology.py``,
cross-algorithm shape forcing into ``srt/speculative/cross_algo_utils.py`` --
while those functions are still called from inside ``__post_init__``. The
path-based exclusion could not see that, so the count went to 11 and the test
went red and STAYED red.

A standing-red ratchet is worse than no ratchet: it is disarmed, and every new
mutation hides in its noise. That is not hypothetical here. NOTE_493
(2026-08-03) recorded this test red at count 10 and moved on. By the time it
was picked up the count was 11, and the ONE site that had slipped in unnoticed
--  ``srt/mem_cache/gdn_slot_ladder.py:288``, commit 7a7742d9ee (2026-08-09) --
is a genuine post-resolution violation, not one of the benign resolution-time
ones. The ratchet caught a real defect and nobody could hear it.

So the pin is now per file, with a verdict per entry. Each entry is sorted into
one of two kinds, and the difference is load-bearing:

RESOLUTION_TIME
    The call runs INSIDE ``ServerArgs.__post_init__``, before
    ``materialize_declarations`` (``server_args.py:6471``) sets the flag that
    arms the runtime ``__setattr__`` guard. Plain assignment is the correct
    mutation style there -- it is what the in-pipeline code does. These are
    only flagged because they live outside the excluded PATHS, so the entry
    records the ``__post_init__`` call site that proves the timing.

FILED_VIOLATION
    The call runs AFTER resolution, on a live ServerArgs the engine is already
    using. These are real, they are debt, and they are recorded here rather
    than quietly folded into the allowed set, so that reading this file tells
    you the ratchet is carrying known debt and exactly how much.

Adding a new mutation anywhere -- including in a file already listed -- fails
this test. Removing one also fails it, asking you to lower the pin and lock in
the progress.

KNOWN LIMIT, STATED SO NOBODY DISCOVERS IT THE HARD WAY: the pin is a per-file
COUNT, so removing one mutation from a file and adding another to the same file
in the same commit nets to zero and passes. Tightening that means pinning
statements rather than counts, which trades this blind spot for line-number
churn on every edit. The runtime guard (``SGLANG_STRICT_CONFIG_MUTATION=1``,
under which a bare post-resolution assignment raises) is what actually covers
the post-resolution case; this static scan exists for the sites the tests never
execute.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import re
import unittest
from pathlib import Path
from typing import Dict, Tuple

import sglang
from sglang.test.test_utils import CustomTestCase

_SGLANG_ROOT = Path(next(iter(sglang.__path__)))

# Assignments to a server_args attribute (``server_args.x = ...``,
# ``self.server_args.x = ...``, and the ``sa`` alias used by a few helpers).
# ``==`` comparisons are excluded by the negative lookahead.
_MUTATION_PATTERNS = [
    # (?![=}]) skips ``==`` comparisons and f-string ``{x=}`` debug specs.
    re.compile(r"\bserver_args\.[a-z0-9_]+\s*=(?![=}])"),
    re.compile(r"\bsa\.[a-z0-9_]+\s*=(?![=}])"),
    re.compile(r"get_(?:global_)?server_args\(\)\.[a-z0-9_]+\s*=(?![=}])"),
    # setattr is the same write with the attribute name behind a variable.
    re.compile(
        r"setattr\(\s*(?:[\w.]+\.)?(?:server_args|sa|get_(?:global_)?server_args\(\))\s*,"
    ),
]

# The resolution pipeline itself (mutation is its job) and multimodal_gen,
# whose ServerArgs is a different class outside this contract.
_EXCLUDED = (
    "srt/server_args.py",
    "srt/arg_groups",
    "multimodal_gen",
)

RESOLUTION_TIME = "resolution-time"
FILED_VIOLATION = "filed-violation"

#: ``relpath -> (count, kind, justification)``. Every entry names the commit
#: that introduced the mutation and the evidence for its kind.
_PINNED: Dict[str, Tuple[int, str, str]] = {
    "srt/disaggregation/topology.py": (
        1,
        RESOLUTION_TIME,
        "pp_size, in apply_pd_topology (topology.py:718). Introduced by "
        "bb40020afd 'disaggregation: free PD topology choice (#107)'. Reached "
        "from __post_init__: server_args.py:6199 -> _handle_pd_disaggregation "
        "-> arg_groups/pd_disaggregation_hook.py:431. Resolution-time.",
    ),
    "srt/speculative/cross_algo_utils.py": (
        2,
        RESOLUTION_TIME,
        "speculative_draft_load_format (:789) and the _SHAPE_FIELDS setattr "
        "loop (:1015), both in normalize_cross_algorithm_args (:673). "
        "Introduced by 234a9feee0 (GGUF cross-algo target routing) and "
        "bc627f72f4 (T156 stage 2 meta-worker). Reached from __post_init__: "
        "server_args.py:6357 -> handle_speculative_decoding -> "
        "arg_groups/speculative_hook.py:142. Resolution-time.",
    ),
    "srt/uneven_perf.py": (
        6,
        RESOLUTION_TIME,
        "_measured_kv_budget_registry_path (:6435), rank_mlp_ratio (:7666), "
        "rank_kv_capacity_seed (:7796, :7842), rank_kv_ratio (:7799), "
        "rank_kv_speed_weights (:7869) -- all in apply_auto_performance "
        "(:6421). Introduced by 36d9f6a506, 17ac81168f, 4a4cbffa91 (x2), "
        "c0837d4a12, a564aaf8466. Reached from __post_init__: "
        "server_args.py:6240 -> _handle_uneven_tp -> server_args.py:10517. "
        "Resolution-time.",
    ),
    "srt/mem_cache/gdn_slot_ladder.py": (
        1,
        FILED_VIOLATION,
        "DEBT. setattr(server_args, '_gdn_profiled_state_slots', ...) at :288, "
        "in remember_profiled_state_slots (:259). Introduced by 7a7742d9ee "
        "'[#631] Make the state-slot cap and the flip compose'. Runs POST "
        "resolution, during ModelRunner KV profiling "
        "(model_runner_kv_cache_mixin.py:2157). Two aggravating facts: the "
        "same caller uses server_args.override(...) about thirty lines "
        "earlier, so the sanctioned path was in view; and the attribute is "
        "underscore-prefixed BY DESIGN to slip past the __setattr__ strict "
        "guard (gdn_slot_ladder.py:250-255 says so outright). This is also "
        "the one site that entered while this test stood red. Not rewritten "
        "here: it is a serving-path change that wants a boot to validate, "
        "which this hermetic cut cannot give it.",
    ),
    "srt/model_executor/dual_group_lane.py": (
        1,
        FILED_VIOLATION,
        "DEBT. dual_group_lane_eager = True at :5495, in build_dual_group_lanes "
        "(:5454). Introduced by df08e51baa '#274 Familien-Slice 2'. Runs POST "
        "resolution on scheduler.server_args, inside Scheduler.__init__ "
        "(managers/scheduler.py:802) -- the live object the engine serves on. "
        "override('dual_group_lane.eager', dual_group_lane_eager=True) is the "
        "shape this wants. Not rewritten here for the same reason as above: "
        "no boot in a hermetic cut.",
    ),
}


def _counts_by_file() -> Dict[str, int]:
    found: Dict[str, int] = {}
    for path in sorted(_SGLANG_ROOT.rglob("*.py")):
        rel = path.relative_to(_SGLANG_ROOT).as_posix()
        if rel.startswith(_EXCLUDED):
            continue
        source = path.read_text()
        n = sum(len(p.findall(source)) for p in _MUTATION_PATTERNS)
        if n:
            found[rel] = n
    return found


class TestServerArgsMutationRatchet(CustomTestCase):
    def test_out_of_pipeline_mutations_match_the_pins(self):
        found = _counts_by_file()
        pinned = {rel: entry[0] for rel, entry in _PINNED.items()}

        new_files = sorted(set(found) - set(pinned))
        self.assertFalse(
            new_files,
            f"new files mutate server_args outside the resolution pipeline: "
            f"{new_files}. Configuration is resolved in "
            f"ServerArgs.__post_init__; declare through the pipeline (passes / "
            f"declare_load_time_override) or go through "
            f"ServerArgs.override(source, ...) instead of assigning fields. If "
            f"the call really runs inside __post_init__, add a RESOLUTION_TIME "
            f"pin naming the call site that proves it.",
        )

        grew = {
            rel: (found[rel], pinned[rel])
            for rel in sorted(set(found) & set(pinned))
            if found[rel] > pinned[rel]
        }
        self.assertFalse(
            grew,
            f"server_args mutations grew in already-pinned files "
            f"{{file: (found, pinned)}}: {grew}. Use "
            f"ServerArgs.override(source, ...), or justify the new site the "
            f"way the existing pins are justified.",
        )

        shrank = {
            rel: (found.get(rel, 0), pinned[rel])
            for rel in sorted(pinned)
            if found.get(rel, 0) < pinned[rel]
        }
        self.assertFalse(
            shrank,
            f"server_args mutations shrank {{file: (found, pinned)}}: "
            f"{shrank}. Lower the pin in this file to lock in the progress.",
        )

    def test_every_pin_carries_a_kind_and_a_justification(self):
        """A pin without a reason is a silent re-baseline with extra steps."""
        for rel, (count, kind, why) in sorted(_PINNED.items()):
            with self.subTest(file=rel):
                self.assertGreater(count, 0)
                self.assertIn(kind, (RESOLUTION_TIME, FILED_VIOLATION))
                self.assertGreater(
                    len(why), 80, f"{rel}: justification is too thin to audit"
                )

    def test_resolution_time_pins_cite_the_post_init_path(self):
        """The claim 'this runs during resolution' is the whole reason these
        are allowed, so each one must show its work."""
        for rel, (_, kind, why) in sorted(_PINNED.items()):
            if kind != RESOLUTION_TIME:
                continue
            with self.subTest(file=rel):
                self.assertIn("__post_init__", why)
                self.assertIn("server_args.py:", why)

    def test_filed_violations_are_visible_as_debt(self):
        """These are recorded, not excused. If this list ever empties, delete
        the concept -- do not let it quietly refill."""
        filed = {
            rel: entry[0]
            for rel, entry in _PINNED.items()
            if entry[1] == FILED_VIOLATION
        }
        self.assertEqual(
            filed,
            {
                "srt/mem_cache/gdn_slot_ladder.py": 1,
                "srt/model_executor/dual_group_lane.py": 1,
            },
            "the filed post-resolution violations changed; that is either "
            "progress worth lowering the pin for, or a new violation that "
            "needs its own item -- neither should pass silently",
        )

    def test_every_pinned_file_still_exists(self):
        """Guards against a pin outliving the file it describes."""
        for rel in sorted(_PINNED):
            with self.subTest(file=rel):
                self.assertTrue((_SGLANG_ROOT / rel).is_file())


if __name__ == "__main__":
    unittest.main()
