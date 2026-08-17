"""#584 R1's own falsifier, built: no NEW private VRAM constant outside the ledger.

R1 says it in one line -- "Every VRAM decision anywhere resolves through
``mem_ledger`` terms. No component keeps a private constant, a private
fraction, or a private reserve" -- and names its own gate:

    *Falsifier:* grep gate -- a new module-level MiB constant used in memory
    arithmetic outside ``mem_ledger`` fails CI. (The three constants #582
    already replaced, 1280/1536/600, are the template for what must not
    recur.)

That gate did not exist. The mem_ledger suite pins terms, wiring, calibration
and reconciliation, and `test_module_state_ratchet.py` guards module-level
MUTABLE state -- a different concern. Nothing refused a new constant.

WHY A RATCHET AND NOT A BAN. The tree holds 43 module-level ``*_BYTES`` /
``*_MIB`` constants outside the ledger and most are NOT R1 violations: protocol
header widths (``_HEADER_BYTES``), unit constants (``GIB_BYTES``), probe sizes.
R1 is about VRAM DEMAND arithmetic, not about every number with a byte unit. So
this pins the MiB/MB/GiB-scale population -- the demand scale -- by name, and a
NEW one fails. Shrinking the list is always allowed; growing it is the failure.

THE ONE THAT IS A REAL VIOLATION, and it is why this gate is worth having:
``distributed/utils.py::_CP_TOKEN_OVERHEAD_MIB = 1536`` is subtracted from each
rank's VRAM budget to size the uneven KV split --

    avail[r] = rank_gpu_memory_mib[r] - checkpoint_share - _CP_TOKEN_OVERHEAD_MIB

-- which is a private VRAM reserve deciding a per-rank budget outside the
ledger, and its value is literally 1536, one of the three constants R1 names as
the template for what must not recur. The ledger has a measured term for
exactly this (``TERM_HARDWARE_RESIDUAL``) and REFUSES to default it: "This term
is NOT defaulted to a constant -- a constant here is the _PREDICT_OVERHEAD_MIB
guess this ledger replaces" (``mem_ledger/engine.py:1525-1533``).

It is pinned as a KNOWN VIOLATION rather than quietly accepted, so the list
carries the verdict and not just the count.
"""

import ast
import pathlib
import re
import unittest

SRT = pathlib.Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"

#: MiB/MB/GiB/GB scale only -- the VRAM DEMAND scale. Byte-scale constants
#: (header widths, probe sizes) are not what R1 is about.
DEMAND_SCALE = re.compile(r"(_MIB|_MB|_GIB|_GB)$")

#: Survivors, by "module::NAME". Each is either a legitimate non-demand use or
#: a recorded violation; the verdict lives beside the name so the list cannot
#: decay into a count.
KNOWN = {
    # ======================================================================
    # VERIFIED R1 VIOLATIONS -- private VRAM arithmetic outside the ledger
    # ======================================================================
    "distributed/utils.py::_CP_TOKEN_OVERHEAD_MIB": (
        "VIOLATION: a private VRAM reserve (1536 MiB) subtracted from each "
        "rank's budget to size the uneven KV split (utils.py:596). The ledger "
        "owns this as TERM_HARDWARE_RESIDUAL and REFUSES to default it. 1536 "
        "is one of the three values R1 names as what must not recur."
    ),
    "uneven_perf.py::_PREDICT_OVERHEAD_MIB": (
        "VIOLATION, and the sharpest one: this is the constant the ledger's "
        "own refusal message names -- 'a constant here is the "
        "_PREDICT_OVERHEAD_MIB guess this ledger replaces' "
        "(mem_ledger/engine.py:1533). The ledger replaced it; the constant "
        "did not leave."
    ),
    "uneven_perf.py::_PREDICT_MAMBA_ACT_RESERVE_MIB": (
        "VIOLATION (same family): a predicted activation reserve in MiB, "
        "outside the ledger that owns activation terms (#595)."
    ),
    "uneven_perf.py::_SOLO_HOST_WORKSPACE_MIB": (
        "VIOLATION (same family): a workspace footprint constant outside the "
        "ledger."
    ),
    # ======================================================================
    # NEEDS AUDIT -- demand-shaped, but I have not read the call site, so the
    # verdict is withheld rather than guessed. Listed so they are visible.
    # ======================================================================
    "model_executor/model_runner_kv_cache_mixin.py::MAMBA_AUTO_ACTIVATION_RESERVE_MIB": (
        "NEEDS AUDIT: an activation reserve in MiB on the sizing path. Likely "
        "a ledger term (#595 activation coverage); not read here."
    ),
    "model_executor/model_runner_kv_cache_mixin.py::MAMBA_CEILING_FIT_MIN_KV_MIB": (
        "NEEDS AUDIT: a minimum-KV floor used in ceiling fitting; may be a "
        "policy floor rather than a demand estimate."
    ),
    "models/deepseek_common/attention_forward_methods/forward_mha.py::DEFAULT_ATTN_SCRATCH_BUDGET_MIB": (
        "NEEDS AUDIT: an attention scratch BUDGET, which is demand-shaped; "
        "#595 carries attention-workspace terms."
    ),
    "planner/key_solver.py::FIXED_PROCESS_POST_MIB": (
        "NEEDS AUDIT: a fixed process post inside the PLANNER. The planner is "
        "an authority, so this may be legitimate -- but a fixed post is "
        "exactly the shape #605's measured posts replace."
    ),
    "planner/graphmem.py::BASE_MIB": (
        "NEEDS AUDIT: graph-memory base estimate inside the planner; #586 "
        "recalibrated the graph-capture term and #707 prices it per rung."
    ),
    "planner/graphmem.py::DRAFT_DECODE_MIB": (
        "NEEDS AUDIT, as BASE_MIB: a draft-decode graph footprint estimate "
        "inside the planner, of the kind #586/#707 measure per rung."
    ),
    "planner/graphmem.py::DRAFT_EXTEND_MIB": (
        "NEEDS AUDIT, as BASE_MIB: a draft-extend graph footprint estimate "
        "inside the planner, of the kind #586/#707 measure per rung."
    ),
    # ======================================================================
    # LEGITIMATE -- stated policy, unit conversion, or an allocation the
    # ledger PRICES rather than a second decision about size
    # ======================================================================
    "managers/corridor_guard.py::CORRIDOR_LAW_MIB": (
        "POLICY, not demand: the corridor law is the operator's stated free-"
        "VRAM floor (1024 MiB). The ledger prices demand; this is headroom, "
        "which R2 says is the user's."
    ),
    "managers/corridor_guard.py::DEFAULT_SEAM_ENTRY_RESERVE_MIB": (
        "POLICY default for the seam draw allowance; derived into the arming "
        "floor with the law (corridor_guard.arming_floor_mib)."
    ),
    "managers/corridor_guard.py::DEFAULT_DELTA_MIB": (
        "POLICY: how far above the floor a reclaim frees, to avoid paying a "
        "spill per allocation. Hysteresis, not demand."
    ),
    "managers/corridor_admission.py::WANT_CAP_MIB": (
        "POLICY cap on a single admission ask; bounds a request, does not "
        "size a pool."
    ),
    "managers/corridor_rebalance.py::DEFAULT_MIN_SHED_MIB": (
        "POLICY hysteresis: the smallest shed worth doing, to avoid churn."
    ),
    "managers/corridor_rebalance.py::DEFAULT_MIN_YIELD_MIB": (
        "POLICY hysteresis: the smallest yield worth acting on."
    ),
    "managers/corridor_steering.py::DEFAULT_MIN_SPREAD_MIB": (
        "POLICY hysteresis: the spread below which steering does nothing."
    ),
    "managers/kv_reshard.py::CORRIDOR_FLOOR_MIB": (
        "POLICY mirror of the corridor law for the reshard path."
    ),
    "managers/phase_flip_runtime.py::DEFAULT_SEAM_ENTRY_MARGIN_MIB": (
        "POLICY margin for seam entry; a decision threshold, not a demand "
        "estimate."
    ),
    "managers/phase_flip_runtime.py::DEFAULT_SEAM_CAP_RETIRE_HYSTERESIS_MIB": (
        "POLICY hysteresis so the seam cap's install/retire pair cannot flap."
    ),
    "managers/phase_flip_seam_reserve.py::DEFAULT_MARGIN_MIB": (
        "POLICY margin on the seam reserve: headroom above the measured draw, "
        "which R2 assigns to the user rather than to the demand model."
    ),
    "managers/phase_flip_spill.py::_MIB": (
        "UNIT CONVERSION (1024*1024), not a demand quantity."
    ),
    "planner/graphmem.py::_GIB_TO_MIB": (
        "UNIT CONVERSION (1024), not a demand quantity: it changes the units "
        "of a number, it does not decide how much VRAM anything gets."
    ),
    "layers/attention/flashinfer_workspace.py::WORKSPACE_ARCH_MIB": (
        "ALLOCATION the ledger prices: the backend's workspace width. #595 "
        "carries the ledger term for it; this is the size that term prices, "
        "not a second decision about how much VRAM to take."
    ),
    "layers/attention/flashinfer_workspace.py::WORKSPACE_DETERMINISTIC_MIB": (
        "ALLOCATION priced by the ledger, as WORKSPACE_ARCH_MIB "
        "(deterministic-mode variant)."
    ),
    "layers/attention/trtllm_mha_backend.py::DEFAULT_WORKSPACE_SIZE_MB": (
        "ALLOCATION priced by the ledger: an upstream backend default whose "
        "resulting allocation #595's workspace term accounts for."
    ),
    "layers/attention/trtllm_mla_backend.py::DEFAULT_WORKSPACE_SIZE_MB": (
        "ALLOCATION priced by the ledger: upstream backend default, MLA "
        "variant, accounted the same way as the MHA one."
    ),
    "boot_matrix/sweep.py::VRAM_IDLE_MIB": (
        "HARNESS threshold in the boot-matrix sweep, not a serving-path "
        "decision about how much VRAM anything gets."
    ),
}


def _module_constants():
    """(module, NAME, value) for module-level demand-scale numeric constants."""
    found = []
    for path in sorted(SRT.rglob("*.py")):
        rel = path.relative_to(SRT).as_posix()
        if rel.startswith("mem_ledger/"):
            continue  # the ledger IS the authority; constants belong there
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            continue
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if not DEMAND_SCALE.search(target.id):
                    continue
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(
                    value.value, (int, float)
                ):
                    found.append((rel, target.id, value.value))
    return found


class TestNoNewPrivateVramConstant(unittest.TestCase):
    """R1's grep gate. Shrinking the list is allowed; growing it fails."""

    def test_the_srt_tree_is_scannable(self):
        """A gate that silently scanned nothing would pass forever."""
        self.assertTrue(SRT.is_dir(), f"{SRT} not found")
        self.assertGreater(len(list(SRT.rglob("*.py"))), 100)

    def test_no_unpinned_demand_scale_constant_exists(self):
        found = {f"{mod}::{name}" for mod, name, _v in _module_constants()}
        new = sorted(found - set(KNOWN))
        self.assertEqual(
            new,
            [],
            "new module-level VRAM-scale constant(s) outside mem_ledger: "
            f"{new}. R1: every VRAM decision resolves through ledger terms. "
            "If this is a ledger term, put it there; if it is not a VRAM "
            "demand decision, add it to KNOWN with the reason.",
        )

    def test_the_pin_has_not_gone_stale(self):
        """A KNOWN entry that no longer exists means the list is describing a
        tree that has moved -- shrink it, so the gate keeps meaning something."""
        found = {f"{mod}::{name}" for mod, name, _v in _module_constants()}
        stale = sorted(set(KNOWN) - found)
        self.assertEqual(stale, [], f"KNOWN lists names that no longer exist: {stale}")

    def test_every_survivor_carries_a_verdict(self):
        for key, why in KNOWN.items():
            with self.subTest(constant=key):
                self.assertGreater(
                    len(why), 40, f"{key} is listed without a real reason"
                )


class TestTheRecordedViolationStaysVisible(unittest.TestCase):
    """The point of a ratchet is that a known defect cannot quietly become
    background. This keeps the one real violation legible until it is fixed."""

    KEY = "distributed/utils.py::_CP_TOKEN_OVERHEAD_MIB"

    def test_it_is_recorded_as_a_violation_not_a_survivor(self):
        self.assertIn("VIOLATION", KNOWN[self.KEY])

    def test_its_value_is_still_one_of_the_three_r1_names(self):
        """R1 names 1280 / 1536 / 600 as the template for what must not
        recur. If this value changes, the finding needs re-reading rather
        than the number quietly drifting."""
        values = {
            v for mod, name, v in _module_constants()
            if f"{mod}::{name}" == self.KEY
        }
        self.assertEqual(values, {1536})

    def test_the_ledger_has_the_term_this_should_resolve_through(self):
        from sglang.srt.mem_ledger.engine import TERM_HARDWARE_RESIDUAL

        self.assertTrue(TERM_HARDWARE_RESIDUAL)


if __name__ == "__main__":
    unittest.main()
