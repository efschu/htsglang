"""HiCache storage prefetch must not be armed without a HiCache tree.

The scheduler used to derive ``enable_hicache_storage`` from the storage
backend name alone::

    self.enable_hicache_storage = server_args.hicache_storage_backend is not None

but *which* tree cache gets built is decided by
``enable_hierarchical_cache`` alone (``mem_cache/registry.py``). Configure a
storage backend with ``--enable-hierarchical-cache`` off and the prefetch
path was armed against a plain ``RadixCache`` / ``MambaRadixCache``, whose
class never defines ``hicache_storage_pass_prefix_keys``. Every scheduler
process then died with an ``AttributeError`` inside its first request.

The test has two halves, because the fix is only correct if both hold:

1. the derivation says False for that combination (and True when both are
   on), and
2. the attribute the prefetch path dereferences really is HiCache-only --
   otherwise the guard would be cosmetic.

Both halves are hermetic: no GPU, no memory pool, no scheduler process.
"""

import ast
import pathlib
import unittest

from sglang.srt.managers.scheduler import derive_enable_hicache_storage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_MEM_CACHE = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "mem_cache"
)

#: Reached only when ``enable_hierarchical_cache`` is on (registry.py).
_HICACHE_MODULES = ("hiradix_cache.py", "hi_mamba_radix_cache.py")
#: Reached when it is off -- these are the ones the bug handed to the
#: prefetch path.
_PLAIN_MODULES = (
    "radix_cache.py",
    "mamba_radix_cache.py",
    "swa_radix_cache.py",
    "chunk_cache.py",
)

#: The first attribute ``Scheduler._prefetch_kvcache`` dereferences on the
#: tree cache after the flag lets it in.
_GUARDED_ATTR = "hicache_storage_pass_prefix_keys"


class _Args:
    """Only the two fields the derivation reads."""

    def __init__(self, backend, hierarchical):
        self.hicache_storage_backend = backend
        self.enable_hierarchical_cache = hierarchical


def _assigns_attr(path: pathlib.Path, attr: str) -> bool:
    """True if the module contains a ``self.<attr> = ...`` anywhere."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == attr
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                return True
    return False


class TestHicacheStorageNeedsHierarchical(CustomTestCase):
    def test_storage_backend_alone_does_not_arm_prefetch(self):
        """The exact configuration that killed all three schedulers."""
        self.assertFalse(
            derive_enable_hicache_storage(_Args("file", hierarchical=False)),
            msg=(
                "a storage backend without --enable-hierarchical-cache builds "
                "a plain RadixCache, which has no "
                f"{_GUARDED_ATTR} -- the prefetch path must stay off"
            ),
        )

    def test_both_on_arms_prefetch(self):
        self.assertTrue(derive_enable_hicache_storage(_Args("file", True)))

    def test_no_backend_never_arms_prefetch(self):
        self.assertFalse(derive_enable_hicache_storage(_Args(None, True)))
        self.assertFalse(derive_enable_hicache_storage(_Args(None, False)))

    def test_guarded_attribute_is_hicache_only(self):
        """Half two: the guard is load-bearing, not decoration.

        If a plain cache ever grows the attribute this test goes red, and the
        guard above can be revisited on purpose instead of by accident.
        """
        for name in _HICACHE_MODULES:
            path = _MEM_CACHE / name
            self.assertTrue(
                _assigns_attr(path, _GUARDED_ATTR),
                msg=f"{name} was expected to define self.{_GUARDED_ATTR}",
            )
        for name in _PLAIN_MODULES:
            path = _MEM_CACHE / name
            if not path.exists():
                continue
            self.assertFalse(
                _assigns_attr(path, _GUARDED_ATTR),
                msg=(
                    f"{name} now defines self.{_GUARDED_ATTR}; the "
                    "hierarchical-cache half of the guard may no longer be "
                    "the right condition"
                ),
            )


if __name__ == "__main__":
    unittest.main()
