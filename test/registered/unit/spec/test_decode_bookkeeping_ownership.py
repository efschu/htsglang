"""Ownership contract for per-request bookkeeping clocks.

Per-request accounting state (`decode_batch_idx` / `extend_batch_idx` iter
clocks, `kv_committed_len` / `kv_allocated_len` KV watermarks,
`spec_verify_ct`, and the `maybe_evict_swa()` call) must only be advanced by
the reviewed owner sites in _OWNER_SITES; spec-v2 draft workers must not
repeat any of them (the scheduler-driven free function / resolve path already
does).
A clock that runs fast fires SWA eviction in the overlap race window and
releases the SWA prefix lock early; neither shows up in e2e CI or the idle
leak checker, hence this AST-level guard.
"""

import ast
import unittest
import warnings
from collections import Counter
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SRT_DIR = _REPO_ROOT / "python" / "sglang" / "srt"
_SPECULATIVE_DIR = _SRT_DIR / "speculative"
assert _SRT_DIR.is_dir(), f"srt dir not found: {_SRT_DIR}"

_TRACKED_ATTRS = (
    "decode_batch_idx",
    "extend_batch_idx",
    "kv_committed_len",
    "kv_allocated_len",
    "spec_verify_ct",
)
_EVICT_METHOD = "maybe_evict_swa"

# {(path relative to srt/, scope, kind): mutation count}. Kind is the mutated
# attribute (`= 0` resets exempt) or "evict" for a `maybe_evict_swa()` call.
# Any added/removed/recounted site fails until reviewed here.
_SB = "managers/schedule_batch.py"
_EAGLE_DECODE = ("speculative/eagle_utils.py", "eagle_prepare_for_decode")
_RESOLVE = (
    "managers/scheduler_components/batch_result_processor.py",
    "SchedulerBatchResultProcessor._resolve_spec_v2_tokens",
)
_SS = "session/streaming_session.py"
_LANE = "model_executor/dual_group_lane.py"
_OWNER_SITES = {
    # non-spec scheduler
    (_SB, "ScheduleBatch.prepare_for_decode", "decode_batch_idx"): 1,
    (_SB, "ScheduleBatch.prepare_for_decode", "kv_committed_len"): 1,
    (_SB, "ScheduleBatch.prepare_for_decode", "kv_allocated_len"): 1,
    (_SB, "ScheduleBatch.prepare_for_extend", "extend_batch_idx"): 1,
    (_SB, "ScheduleBatch.prepare_for_extend", "kv_committed_len"): 1,
    (_SB, "ScheduleBatch.prepare_for_extend", "kv_allocated_len"): 1,
    ("mem_cache/common.py", "alloc_for_extend", "evict"): 1,
    ("mem_cache/common.py", "alloc_for_decode", "evict"): 1,
    # spec v2: no pre-claim; resolve commits the full accepted run uniformly.
    (*_EAGLE_DECODE, "decode_batch_idx"): 1,
    (*_EAGLE_DECODE, "evict"): 1,
    (*_EAGLE_DECODE, "kv_allocated_len"): 1,
    (*_RESOLVE, "kv_committed_len"): 1,
    (*_RESOLVE, "spec_verify_ct"): 1,
    (
        "speculative/dflash_info_v2.py",
        "DFlashDraftInputV2.prepare_for_decode",
        "kv_allocated_len",
    ): 1,
    # disaggregation decode prealloc
    (
        "disaggregation/decode.py",
        "DecodePreallocQueue._pre_alloc",
        "kv_committed_len",
    ): 1,
    (
        "disaggregation/decode.py",
        "DecodePreallocQueue._pre_alloc",
        "kv_allocated_len",
    ): 1,
    # streaming session slot save/restore and tail trimming
    (_SS, "SessionSlot.save_from_req", "kv_committed_len"): 1,
    (_SS, "SessionSlot.save_from_req", "kv_allocated_len"): 1,
    (_SS, "SessionSlot.restore_to_req", "kv_committed_len"): 1,
    (_SS, "SessionSlot.restore_to_req", "kv_allocated_len"): 1,
    (_SS, "StreamingSession._free_tail", "kv_committed_len"): 2,
    (_SS, "StreamingSession._free_tail", "kv_allocated_len"): 2,
    (_SS, "StreamingSession._trim_overshoot", "kv_committed_len"): 1,
    (_SS, "StreamingSession._trim_overshoot", "kv_allocated_len"): 1,
    (_SS, "StreamingSession.try_cache_finished_req", "kv_allocated_len"): 1,
    # Inherit the authoritative finished length (not the lagging req clock).
    (_SS, "StreamingSession.try_cache_finished_req", "kv_committed_len"): 1,
    # -- dual-group lane (#274): a SEPARATE mini-runtime, not the scheduler --
    #
    # The lane builds its own ModelRunner and calls `alloc_memory_pool()` on
    # it, so its `req_to_token_pool` / `token_to_kv_pool_allocator` are
    # disjoint from the serving group's. Its requests are lane-private
    # (`_make_batch` constructs `Req(rid="dual-group-lane-<n>")`; they never
    # enter the scheduler's running_batch), the batch carries
    # `_LaneTreeCacheStub` (`supports_swa()` False, `evict()` a no-op) and is
    # built with `enable_overlap=False`. The hazard this register guards --
    # a clock running fast fires SWA eviction inside the overlap race window
    # and releases the SWA prefix lock early -- is therefore structurally
    # absent here, and there is no second worker on these pools (#444/#450):
    # the lane tick is rank-local and serial by contract.
    # The lane drives its own decode/verify forwards instead of going through
    # `prepare_for_decode`, which is why it must settle these fields itself.
    #
    # The verify commits `kept` positions the verify forward consumed.
    (_LANE, "DualGroupLane._verify_by_target_verify", "decode_batch_idx"): 1,
    (_LANE, "DualGroupLane._verify_by_target_verify", "kv_committed_len"): 1,
    (_LANE, "DualGroupLane._verify_by_target_verify", "kv_allocated_len"): 1,
    # Rollback of the head's over-drafted tail: the clock moves BACKWARDS
    # (`max(0, idx - surplus)`) after the surplus slots are handed back, so it
    # cannot run fast. `batch_d.reqs[0]` is the draft head's own lane request.
    (_LANE, "DualGroupLane._truncate_draft", "decode_batch_idx"): 1,
    (_LANE, "DualGroupLane._truncate_draft", "kv_committed_len"): 1,
    (_LANE, "DualGroupLane._truncate_draft", "kv_allocated_len"): 1,
    # Debug probes, env-gated off by default (SGLANG_LANE_SPEC_ROW_ORACLE /
    # `_dbg_on`): both restore the counters to a value they already held.
    (_LANE, "DualGroupLane._dbg_rollback_one", "kv_committed_len"): 1,
    (_LANE, "DualGroupLane._dbg_rollback_one", "kv_allocated_len"): 1,
    (_LANE, "DualGroupLane._dbg_row_oracle._restore", "decode_batch_idx"): 1,
    (_LANE, "DualGroupLane._dbg_row_oracle._restore", "kv_committed_len"): 1,
    (_LANE, "DualGroupLane._dbg_row_oracle._restore", "kv_allocated_len"): 1,
    # -- kv-session offload: reclaim of the speculative overhang before a spill
    #
    # This one IS a scheduler-owned request. It only LOWERS the watermark, and
    # only after `token_to_kv_pool_allocator.free()` actually returned the
    # slots in `[snap.free_from, kv_allocated_len)` -- the same range stock
    # retraction reclaims via `pop_overallocated_kv_cache`. It is ordered after
    # the in-flight forward (`_wait_forward_stream()` runs before the snapshot
    # is taken), and `kv_committed_len` is left to the resolve path, which
    # remains its sole owner. A lowered allocated watermark cannot fire the
    # early-eviction hazard this register guards.
    # NOTE, and it is NOT what this register tracks: the same block also sets
    # `kv_overallocated_freed = True` (kv_session_offload.py:3360) BEFORE three
    # later decline points (:3394 empty spill plan, :3411 budget regler, :3434
    # tail exceeds the host region), each of which returns False and leaves the
    # request running. Only the resume path clears the flag again (:5075), so a
    # reclaim-then-decline leaves it set on a live request and the eventual
    # `pop_overallocated_kv_cache()` (schedule_batch.py:1141) asserts. Open
    # defect, filed separately -- it does not bear on the ownership verdict
    # above, which is about the watermark write.
    (
        "managers/kv_session_offload.py",
        "KVSessionOffloadManager.try_spill",
        "kv_allocated_len",
    ): 1,
}


def _iter_scoped_nodes(tree):
    """Yield (node, dotted Class.method scope) for every node."""
    scope_of = {}

    def visit(node, scope):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}" if scope else node.name
        scope_of[node] = scope
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, "")
    return scope_of.items()


def _is_zero_reset(node):
    return isinstance(node, ast.Assign) and (
        isinstance(node.value, ast.Constant) and node.value.value == 0
    )


def _scan_tree(tree):
    """Count bookkeeping sites in an AST as Counter[(scope, kind)]."""
    sites = Counter()
    for node, scope in _iter_scoped_nodes(tree):
        if isinstance(node, (ast.AugAssign, ast.Assign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr in _TRACKED_ATTRS
                    and not _is_zero_reset(node)
                ):
                    sites[(scope, target.attr)] += 1
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == _EVICT_METHOD
        ):
            sites[(scope, "evict")] += 1
    return sites


def _parse(path: Path):
    # utf-8-sig: some srt files carry a BOM that breaks plain-utf-8 ast.parse.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(path.read_text(encoding="utf-8-sig"))


def _scan_srt():
    """Count all bookkeeping sites in srt/ as Counter[(rel, scope, kind)]."""
    found = Counter()
    for path in sorted(_SRT_DIR.rglob("*.py")):
        rel = path.relative_to(_SRT_DIR).as_posix()
        for (scope, kind), count in _scan_tree(_parse(path)).items():
            found[(rel, scope, kind)] += count
    return found


def _draft_worker_classes():
    """All transitive EagleDraftWorkerBase subclasses under speculative/."""
    by_name = {}
    for path in sorted(_SPECULATIVE_DIR.glob("*.py")):
        rel = path.relative_to(_SRT_DIR).as_posix()
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ClassDef):
                bases = {
                    b.id if isinstance(b, ast.Name) else getattr(b, "attr", None)
                    for b in node.bases
                }
                by_name[node.name] = (rel, node, bases)

    workers = {"EagleDraftWorkerBase"}
    changed = True
    while changed:
        changed = False
        for name, (_, _, bases) in by_name.items():
            if name not in workers and bases & workers:
                workers.add(name)
                changed = True
    return [
        (rel, node)
        for name, (rel, node, _) in sorted(by_name.items())
        if name in workers and name != "EagleDraftWorkerBase"
    ]


def _scan_class_subtree(class_node):
    """Scan one ClassDef subtree; returns (method_scope, kind) sites."""
    module = ast.Module(body=[class_node], type_ignores=[])
    sites = set()
    for scope, kind in _scan_tree(module):
        # Strip the leading class name; keep method-level scope.
        sites.add((scope.split(".", 1)[1] if "." in scope else scope, kind))
    return sites


class TestDecodeBookkeepingOwnership(CustomTestCase):
    def test_bookkeeping_sites_match_owner_allowlist(self):
        found = _scan_srt()
        allow = Counter(_OWNER_SITES)
        unexpected = found - allow
        missing = allow - found
        msg = []
        if unexpected:
            msg.append(
                "New bookkeeping mutation(s) beyond the recorded counts:\n  "
                + "\n  ".join(f"{site} x{n}" for site, n in sorted(unexpected.items()))
                + "\nThese are owned by the sites in _OWNER_SITES -- do not "
                "repeat them; a genuinely new owner must be recorded there."
            )
        if missing:
            msg.append(
                "Recorded site(s) no longer exist (update _OWNER_SITES):\n  "
                + "\n  ".join(f"{site} x{n}" for site, n in sorted(missing.items()))
            )
        self.assertFalse(msg, "\n\n".join(msg))

    def test_spec_v2_draft_workers_do_no_scheduler_bookkeeping(self):
        classes = _draft_worker_classes()
        names = {node.name for _, node in classes}
        # Discovery sanity: fail loudly instead of silently guarding nothing.
        self.assertIn("EagleDraftWorker", names)
        self.assertIn("FrozenKVMTPDraftWorker", names)

        violations = []
        for rel, node in classes:
            for scope, kind in _scan_class_subtree(node):
                violations.append((rel, f"{node.name}.{scope}", kind))
        self.assertFalse(
            violations,
            "Spec-v2 draft worker(s) repeat scheduler-owned bookkeeping:\n  "
            + "\n  ".join(map(str, sorted(violations)))
            + "\nUnder spec v2 the iter-clock ticks, `maybe_evict_swa`, and "
            "KV watermark settlement are owned by the scheduler-driven "
            "free function / resolve path. Remove these from the worker.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
