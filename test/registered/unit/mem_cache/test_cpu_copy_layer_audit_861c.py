"""#861c: every `load_cpu_copy` in the tree either checks the copy's layer count or says why not.

THIS IS THE ANSWER TO "how does this class become discoverable by INSPECTION
instead of by crash". The two fix-side files pin the two instances that were
found; this one makes the NEXT instance fail at the desk.

THE CLASS: a per-layer structure is carried across a cutover and indexed with
the DESTINATION layout's geometry. `get_cpu_copy` sizes a list with the copying
pool's `layer_num`; `load_cpu_copy` walks the restoring pool's. Within one phase
those are the same object and the defect is invisible; across the phase flip
they are different pools with different layer counts (`--pp-stage-ratio`), and
the mismatch is an IndexError one way and a silent wrong-layer write the other.

WHY AN AUDIT AND NOT A BASE-CLASS HOOK. The pools do not share a base --
`MHATokenToKVPool`/`MLATokenToKVPool` descend from `KVCache`, `MambaPool` does
not, the NPU pools live in another package, and the composite pools forward
rather than loop. A guard installed on one ancestor is inert on the others; that
is the finding `check_cpu_copy_rows` (memory_pool.py) already records for the
row axis, and it holds unchanged here. What CAN be enforced tree-wide is the
obligation: if you loop your own layer count over a list you were handed, you
check first.

THE SAME SHAPE ALREADY EXISTS IN THE TREE FOR A NEIGHBOURING PATH, and this
follows it rather than inventing a second one:
  * `test_kv_store_bound_unity.py` (#355) audits every KV writer by AST and
    fails on a NEW unbounded one -- the "audit with a reasoned allowlist" idiom
    reused verbatim here.
  * `hicache_phase_binding.check_shapes` (#719) refuses a HiCache rebind whose
    host and device pools report different layer counts, with the same argument
    in its docstring: "the single most dangerous rebind is the one that
    succeeds: matching row ids, mismatched widths, and a copy that runs".
    #861c is that finding on the seam copy path, where it had not been applied.

THREE CLASSIFICATIONS, and every method in the tree must be one of them:
  GUARDED     -- loops a per-layer count and calls `check_cpu_copy_layers`.
  UNSUPPORTED -- raises, so there is no restore to get wrong.
  FORWARDER   -- contains no per-layer loop of its own; it hands the payload to
                 sub-pools, which are audited on their own account. A forwarder
                 that grows a loop stops being one and this test fails.

A NEW `load_cpu_copy` is UNCLASSIFIED and fails, which is the point: the
allowlist is by NAME and by REASON, so adding a pool is a decision someone
records rather than a silence someone inherits.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import unittest
from pathlib import Path

import sglang.srt.mem_cache.memory_pool as memory_pool
from sglang.test.test_utils import CustomTestCase

# Anchored on a MODULE file rather than on `sglang.srt.__file__`: that package
# is a namespace package here and its `__file__` is None.
SRT_ROOT = Path(memory_pool.__file__).parent.parent

# Methods with no per-layer loop of their own. Each entry names WHY, and the
# structural check below re-derives the claim rather than trusting the list:
# an entry that grows a loop fails as UNGUARDED, not as an allowlist hit.
FORWARDERS = {
    "mem_cache/allocator/base.py::BaseTokenToKVPoolAllocator": (
        "raises; the paged allocator has no pool to forward to here"
    ),
    "mem_cache/allocator/paged.py::PagedTokenToKVPoolAllocator": (
        "forwards to self._kvcache"
    ),
    "mem_cache/allocator/token.py::TokenToKVPoolAllocator": "forwards to self._kvcache",
    "mem_cache/allocator/swa.py::SWATokenToKVPoolAllocator": (
        "forwards to self._kvcache"
    ),
    "mem_cache/memory_pool.py::KVCache": "abstract; raises NotImplementedError",
    "mem_cache/memory_pool.py::PageMajorMHATokenToKVPool": (
        "raises: CPU offload is unsupported under the page-major layout"
    ),
    "mem_cache/memory_pool.py::HybridLinearKVPool": (
        "splits the payload across full_kv_pool and mamba_pool, both audited"
    ),
    "mem_cache/swa_memory_pool.py::SWAKVPool": (
        "splits the payload across full/swa sub-pools, both audited"
    ),
    "mem_cache/unified_memory_pool.py::UnifiedSWAKVPool": (
        "splits the payload across full/swa sub-pools, both audited"
    ),
    "mem_cache/hisparse_memory_pool.py::HiSparseDSATokenToKVPool": (
        "raises NotImplementedError"
    ),
    "mem_cache/deepseek_v4_memory_pool.py::HiSparseC4DevicePool": (
        "raises NotImplementedError"
    ),
}

# The per-layer iteration shapes that make a method the audit's business.
_LAYER_ATTRS = ("layer_num", "num_layers")


def _iterates_own_layers(fn: ast.AST) -> bool:
    """Does this body walk a count that belongs to SELF rather than to the copy?

    Two shapes, both live in the tree:
      `for x in range(self.layer_num)`         -- the KV pools
      `for i, c in enumerate(self.mamba_cache.conv)` -- MambaPool, whose
                                                 per-layer container is a list
    Both index a PARAMETER with that loop variable, which is the defect.
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.For):
            continue
        it = node.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name):
            if it.func.id == "range" and it.args:
                arg = it.args[0]
                if (
                    isinstance(arg, ast.Attribute)
                    and arg.attr in _LAYER_ATTRS
                    and isinstance(arg.value, ast.Name)
                    and arg.value.id == "self"
                ):
                    return True
            if it.func.id == "enumerate" and it.args:
                if isinstance(it.args[0], ast.Attribute) and _roots_at_self(it.args[0]):
                    return True
    return False


def _roots_at_self(node: ast.AST) -> bool:
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id == "self"


def _calls(fn: ast.AST) -> set:
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def _load_cpu_copy_methods():
    """(`path::Class`, function node) for every `load_cpu_copy` under srt/.

    Keyed by CLASS and not only by file: `memory_pool.py` holds several, and a
    file-level allowlist entry would silently cover a new one added beside an
    exempt sibling."""
    found = []
    for path in sorted(SRT_ROOT.rglob("*.py")):
        try:
            src = path.read_text()
        except OSError:  # pragma: no cover
            continue
        if "load_cpu_copy" not in src:  # cheap prefilter; the walk is the slow part
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:  # pragma: no cover - a parse failure is its own bug
            continue
        rel = str(path.relative_to(SRT_ROOT))
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            for node in cls.body:
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "load_cpu_copy"
                ):
                    found.append((f"{rel}::{cls.name}", node))
    return found


class TestEveryRestoreIsClassified(CustomTestCase):
    def test_the_audit_actually_finds_the_methods(self):
        """CONTROL. An audit that walks an empty set is green for the wrong
        reason, and that is indistinguishable from an audit that works."""
        found = _load_cpu_copy_methods()
        self.assertGreaterEqual(
            len(found),
            10,
            "the AST sweep found almost nothing -- the walk is broken, not the tree",
        )
        paths = {p for p, _ in found}
        self.assertTrue(
            any(p.startswith("mem_cache/memory_pool.py::") for p in paths),
            "memory_pool.py contributed nothing -- the sweep is broken",
        )

    def test_the_specimen_site_is_seen_as_a_looper(self):
        """CONTROL for the classifier itself: the method W40 died in must be
        recognised as one that walks its own layer count. If this goes green
        because the shape test is inert, every red below is unreachable."""
        loopers = [
            p
            for p, fn in _load_cpu_copy_methods()
            if p.startswith("mem_cache/memory_pool.py::") and _iterates_own_layers(fn)
        ]
        self.assertTrue(
            loopers, "no per-layer looper found in memory_pool.py -- classifier inert"
        )

    def test_every_per_layer_restore_checks_the_copys_layer_count(self):
        unguarded = []
        for path, fn in _load_cpu_copy_methods():
            if not _iterates_own_layers(fn):
                continue
            if "check_cpu_copy_layers" not in _calls(fn):
                unguarded.append(f"{path}:{fn.lineno}")
        self.assertEqual(
            [],
            unguarded,
            "these `load_cpu_copy` methods walk their OWN layer count over a "
            "list they were handed, without checking that the two agree. "
            "Across a phase flip that is an IndexError one way (the W40 crash) "
            "and a silent wrong-layer write the other. Call "
            "`check_cpu_copy_layers(len(copy), self.layer_num, 'restore', "
            "'layer')` before the first store: " + ", ".join(unguarded),
        )

    def test_every_non_looping_restore_is_a_declared_forwarder(self):
        """The other half. A method that does not loop is fine ONLY because it
        forwards to pools that do -- which is a claim, and claims get recorded.
        A new pool lands here and has to be classified by a person."""
        undeclared = []
        for path, fn in _load_cpu_copy_methods():
            if _iterates_own_layers(fn):
                continue
            if path not in FORWARDERS:
                undeclared.append(f"{path}:{fn.lineno}")
        self.assertEqual(
            [],
            undeclared,
            "these `load_cpu_copy` methods neither loop per-layer nor appear in "
            "FORWARDERS. Add them with the reason they need no check of their "
            "own (they forward, or they raise), or add the check: "
            + ", ".join(undeclared),
        )

    def test_the_forwarder_list_has_no_dead_entries(self):
        """An allowlist that outlives its entries is how an audit rots into a
        rubber stamp."""
        live = {p for p, fn in _load_cpu_copy_methods() if not _iterates_own_layers(fn)}
        self.assertEqual(
            set(),
            set(FORWARDERS) - live,
            "FORWARDERS names paths that no longer have a non-looping "
            "`load_cpu_copy`; drop them so the list keeps meaning something",
        )


if __name__ == "__main__":
    unittest.main()
