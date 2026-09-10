"""#1329 -- ``slots=True`` + a lazily-assigned attribute = AttributeError ON THE WRITE.

BOOT weg2xsn7 (@376ae2a475) died on this in 24 of 24 legs, BOTH groups, both
hooks, and #1328's note is what finally named it:

    W79 Weg2XchgShadowRankLocalSkip rank=1 leg=0 reason=no-plan
      detail=manifest-unagreed:hook=source group=D rank=1
      manifest=manifest-failed:AttributeError:
        'SchedulerWeightUpdaterManager' object has no attribute
        '_weg2_shadow_region_cache' @ weight_updater.py:1590

`SchedulerWeightUpdaterManager` is `@dataclass(kw_only=True, slots=True)`.
Under `slots=True` there is no `__dict__`, so `self._x = v` for an undeclared
`_x` raises `AttributeError` -- on the WRITE, not the read. The two S6b shadow
caches were assigned lazily, so every shadow leg of every boot on this arm died
at the first write. That is why `WEG2-XCHG-PLAN` was 0 on both groups and why
the shadow has never once run, on xsn5, xsn6 or xsn7.

THE CLASS PREDICTED IT, THREE COMMITS EARLY, in its own comment on
`weg2_leg_ledger`: *"A FIELD for the third time in this class: ``slots=True``
turns a lazily-assigned ``self._weg2_leg_ledger`` into an ``AttributeError``
raised ONLY on the retry path, i.e. only after a failure has already happened
-- the worst possible place to learn it."* These are the fourth and fifth.

SO THE PIN IS STRUCTURAL, NOT PER-ATTRIBUTE. Two spot fixes would leave the
next lazy assignment to the next boot; the AST sweep below is the future check
the root-before-effect law asks for, and it is cheap enough to keep forever.
"""

import ast
import inspect
import types
import unittest

from sglang.srt.managers.scheduler_components import weight_updater as wu

MANAGER = "SchedulerWeightUpdaterManager"


def _class_node(name):
    src = inspect.getsource(wu)
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _declared_fields(cls_node):
    return {
        n.target.id
        for n in cls_node.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }


def _self_writes(cls_node):
    out = {}
    for n in ast.walk(cls_node):
        targets = []
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        for t in targets:
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == "self"):
                out.setdefault(t.attr, []).append(n.lineno)
    return out


class SlotsClassHasNoLazyAttributes(unittest.TestCase):
    """THE RATCHET. One assertion, the whole class, every future attribute."""

    def test_the_manager_is_still_a_slots_dataclass(self):
        """The premise. If this ever stops being true the pin below is moot,
        and a reader must be told rather than left with a green test that
        checks nothing."""
        cls = getattr(wu, MANAGER)
        self.assertFalse(
            hasattr(cls, "__dict__") and isinstance(
                getattr(cls, "__dict__", None), types.MappingProxyType)
            and "__dict__" in cls.__dict__,
            "the class must not carry a per-instance __dict__",
        )
        self.assertTrue(hasattr(cls, "__slots__"), "slots=True was removed")

    def test_every_self_assignment_targets_a_declared_field(self):
        """M1: the sweep, as the pin.

        Measured on 376ae2a475 BEFORE the fix: 2 undeclared
        (`_weg2_shadow_region_cache` @1590, `_weg2_shadow_manifest_cache`
        @1632) of 8 distinct `self.X` writes against 15 declared fields.
        """
        cls_node = _class_node(MANAGER)
        fields = _declared_fields(cls_node)
        writes = _self_writes(cls_node)
        undeclared = {k: sorted(set(v)) for k, v in writes.items()
                      if k not in fields}
        self.assertEqual(
            undeclared, {},
            f"lazily-assigned attributes on a slots=True dataclass raise "
            f"AttributeError ON THE WRITE -- declare each as a field, as "
            f"`weg2_leg_ledger` and the two #1329 caches are: {undeclared}",
        )
        # The sweep must be able to SEE something, or an empty result is
        # vacuous rather than clean (a rename of the class would do it).
        self.assertGreaterEqual(len(writes), 5, "the sweep found almost nothing")
        self.assertGreaterEqual(len(fields), 10)


class TheTwoShadowCachesExistAfterConstruction(unittest.TestCase):
    """The behavioural half: for EVERY group form, both hooks, no CUDA."""

    @staticmethod
    def _manager():
        return getattr(wu, MANAGER)(
            tp_worker=None, draft_worker=None, tp_cpu_group=None,
            memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
            is_fully_idle=lambda *a, **k: True,
        )

    def test_both_caches_are_readable_and_writable_after_init(self):
        """M2: the write is what failed on metal, so the write is the pin."""
        m = self._manager()
        # Present by construction -- no group, no hook, no leg needed.
        self.assertEqual(m._weg2_shadow_region_cache, "unset")
        self.assertIsNone(m._weg2_shadow_manifest_cache)
        # And ASSIGNABLE, which is the operation that raised 24/24 on xsn7.
        m._weg2_shadow_region_cache = None
        m._weg2_shadow_manifest_cache = ((1, 0, 0, 4096, 7),)
        self.assertIsNone(m._weg2_shadow_region_cache)
        self.assertEqual(len(m._weg2_shadow_manifest_cache), 1)

    def test_the_unset_sentinel_is_distinct_from_a_cached_none(self):
        """M3: the semantics the getattr default carried must survive.

        A region that legitimately opens to ``None`` (a boot with no region)
        must CACHE that answer; defaulting the field to ``None`` would make
        every leg re-open and re-fail instead.
        """
        m = self._manager()
        self.assertNotEqual(m._weg2_shadow_region_cache, None)
        self.assertEqual(m._weg2_shadow_region_cache, "unset")
        # The reader's contract, exercised through the real method's logic.
        cached = getattr(m, "_weg2_shadow_region_cache", "unset")
        self.assertEqual(cached, "unset", "a fresh manager is NOT yet cached")
        m._weg2_shadow_region_cache = None
        self.assertNotEqual(
            getattr(m, "_weg2_shadow_region_cache", "unset"), "unset",
            "a cached None must read as CACHED, not as unset")

    def test_the_region_accessor_returns_none_without_a_region_and_caches_it(self):
        """M4: the whole path that died, driven hermetically.

        No `SGLANG_WEG2_XCHG_REGION` in the environment, so the accessor takes
        its `path and boot` guard, caches `None`, and -- the point -- does not
        raise on the way out.
        """
        import os

        m = self._manager()
        keep = {k: os.environ.pop(k) for k in list(os.environ)
                if k.startswith("SGLANG_WEG2_XCHG_REGION")}
        try:
            self.assertIsNone(m._weg2_shadow_region())
            self.assertIsNone(m._weg2_shadow_region_cache)
            self.assertIsNone(m._weg2_shadow_region())  # second call: cached
        finally:
            os.environ.update(keep)


if __name__ == "__main__":
    unittest.main()
