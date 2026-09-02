"""#1068 WEG 1 slice 0: the per-request seam copy (A) is DELETED and stays deleted.

THE LAW. `upstream-minimal-statt-eigenbau`: the seam copy was a second carrier
beside the HiCache store (`#706` canonical pages + the GDN leaf anchor). Two
carriers for one prefix is the defect class that produced #913 (copying rows
whose backing was gone), #920 (global slot ids handed to a compacted pool),
#941 (layout refusal on EVERY pp_to_tp cutover) and #875/#1043 (a rank-local
carry that scored 162 refusals and 0 successes). After this slice the store is
the single carrier and the bounded loss is one chunked_prefill_size per
resident and cutover (WEG1_BUILD_SPEC_0901.md section 4.5).

WHAT THIS PINS. Not behaviour -- ABSENCE. A deleted mechanism grows back one
helper at a time ("just a small restore for the mid-chunk case"), and the
first re-grown helper reads exactly like a bug fix. So the pin is an AST walk
over `python/sglang/srt`: none of the seam-copy names may be DEFINED, no call
may pass `copy_state=`, `retract_all`/`release_req` may not accept it, `Req`
may not carry the four copy-side fields, and the acceptance log markers that
only the copy ever printed may not exist as string literals.

RED on 228a66db32 because every one of those is present:
schedule_batch.py:3136 seam_copy_state, :3231 restore_seam_state,
phase_flip_runtime.py:925 seam_copy_addresses_the_bound_pool,
schedule_batch.py:2741 _mamba_cpu_copy_is_mine, :2985 copy_state, :663
SEAM_STATE_PREFIX, phase_flip_runtime.py:1920 the #920 marker.

WHAT STAYS, pinned in the second test so an over-eager sweep cannot take the
upstream core with it: `Req.offload_kv_cache` / `Req.load_kv_cache` (the
decode-disaggregation retraction backup, upstream), `check_cpu_copy_layers`
and `check_cpu_copy_rows` (both consumed by the retained `get_cpu_copy` /
`load_cpu_copy` bodies, memory_pool.py -- a named refusal on a live
disaggregation path is kept, never converted into a silent illegal access).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import ast
import pathlib
import unittest

import sglang
from sglang.test.test_utils import CustomTestCase

SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"

#: Functions / methods that must not be DEFINED anywhere under srt/.
DELETED_DEFS = (
    "seam_copy_state",
    "restore_seam_state",
    "seam_copy_addresses_the_bound_pool",
    "_mamba_cpu_copy_is_mine",
    "_seam_extent_of",
    "_seam_prefill_is_complete",
    "supports_mamba_cpu_copy",
    "cpu_copy_layout",
)
#: Classes that must not be defined.
DELETED_CLASSES = ("CpuCopyLayout",)
#: Module-level names that must not be assigned.
DELETED_NAMES = (
    "SEAM_COPY_GLOBAL_ROW_LAYOUTS",
    "SEAM_STATE_PREFIX",
    "_SEAM_STATE_COUNTS",
)
#: Attributes that must not be written on any object (`x.<name> = ...`) nor
#: declared as annotated class fields.
DELETED_FIELDS = (
    "mamba_state_cpu",
    "kv_cache_cpu_extent",
    "kv_cache_cpu_layout",
    "mamba_state_cpu_layout",
)
#: Functions that must not take a `copy_state` parameter.
NO_COPY_STATE_PARAM = ("retract_all", "release_req")
#: Log markers only the deleted copy ever printed.
DELETED_MARKERS = (
    "[#783 seam-state]",
    "the cutover will NOT copy",
    "SEAM RESTORE",
    "SEAM COPY DECLINED",
)


def _iter_sources():
    for path in sorted(SRT.rglob("*.py")):
        # utf-8-sig: one model file carries a BOM (qwen2_classification.py),
        # which plain utf-8 leaves as U+FEFF and ast.parse rejects.
        yield path, path.read_text(encoding="utf-8-sig", errors="replace")


def _param_names(fn: ast.AST):
    a = fn.args
    names = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return names


def seam_copy_zombies():
    """Every surviving occurrence, as `path:line what`. Empty means deleted."""
    found = []
    for path, src in _iter_sources():
        rel = str(path.relative_to(SRT.parent.parent))
        for marker in DELETED_MARKERS:
            for lineno, line in enumerate(src.splitlines(), 1):
                if marker in line:
                    found.append(f"{rel}:{lineno} marker {marker!r}")
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:  # a file that does not parse is a finding too
            found.append(f"{rel}:{exc.lineno} does not parse: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in DELETED_DEFS:
                    found.append(f"{rel}:{node.lineno} def {node.name}")
                if node.name in NO_COPY_STATE_PARAM and "copy_state" in _param_names(
                    node
                ):
                    found.append(f"{rel}:{node.lineno} {node.name} takes copy_state")
            elif isinstance(node, ast.ClassDef):
                if node.name in DELETED_CLASSES:
                    found.append(f"{rel}:{node.lineno} class {node.name}")
            elif isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "copy_state":
                        found.append(f"{rel}:{node.lineno} call passes copy_state=")
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in DELETED_NAMES:
                        found.append(f"{rel}:{node.lineno} assigns {tgt.id}")
                    if isinstance(tgt, ast.Attribute) and tgt.attr in DELETED_FIELDS:
                        found.append(f"{rel}:{node.lineno} writes .{tgt.attr}")
            elif isinstance(node, ast.AnnAssign):
                tgt = node.target
                if isinstance(tgt, ast.Name) and tgt.id in DELETED_FIELDS:
                    found.append(f"{rel}:{node.lineno} declares field {tgt.id}")
                if isinstance(tgt, ast.Attribute) and tgt.attr in DELETED_FIELDS:
                    found.append(f"{rel}:{node.lineno} writes .{tgt.attr}")
    return found


class TestSeamCopyIsDeleted(CustomTestCase):
    def test_no_seam_copy_symbols_remain(self):
        zombies = seam_copy_zombies()
        self.assertEqual(
            zombies,
            [],
            "#1068 seam copy A must be deleted, not gated. Surviving pieces "
            f"({len(zombies)}):\n  " + "\n  ".join(zombies),
        )

    def test_the_upstream_core_survives(self):
        # The sweep may not take the decode-disaggregation retraction backup
        # or the pool-level guards its bodies call. Named here so the deletion
        # has a lower bound as well as an upper one.
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.mem_cache import memory_pool

        self.assertTrue(callable(getattr(Req, "offload_kv_cache", None)))
        self.assertTrue(callable(getattr(Req, "load_kv_cache", None)))
        self.assertTrue(callable(getattr(memory_pool, "check_cpu_copy_layers", None)))
        self.assertTrue(callable(getattr(memory_pool, "check_cpu_copy_rows", None)))
        self.assertTrue(
            callable(getattr(memory_pool.MHATokenToKVPool, "get_cpu_copy", None))
        )
        self.assertTrue(
            callable(getattr(memory_pool.MHATokenToKVPool, "load_cpu_copy", None))
        )


if __name__ == "__main__":
    unittest.main()
