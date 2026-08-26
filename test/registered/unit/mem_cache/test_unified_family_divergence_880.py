"""#880: the Unified* family, checked from the BASE instead of from a list.

THE QUESTION THIS ANSWERS, ASKED OF MY OWN EARLIER WORK. Three duck-typed
attribute misses on the Unified family have been found one at a time:

    #832   `evictable_size` on UnifiedRadixCache   -- killed all three ranks (W29)
    #872   `_drain_storage_control_queues_local`   -- the fence never waited
    #872b  `value` / `*lock_ref` on UnifiedTreeNode -- rows unreadable, locks permissive

All three were INSTANCE fixes. The two conformance suites written with #872/#872b
were the first class-level checks in the tree, but they are not enough, and
measuring them says so plainly:

  * the #872 suite reads its probed names out of ONE module's source
    (`hicache_flip_writeback`), so it is derived -- but only for that module;
  * the #872b suite carries `expected_unreadable = {"UnifiedTreeNode"}` and
    pins `value` plus the `*lock_ref` trio. That is a LIST. `TreeNode` exposes
    SIX names `UnifiedTreeNode` lacks -- `value`, `lock_ref`, `host_value`,
    `host_ref_counter`, `protect_host`, `release_host` -- so the list covered
    two of six, i.e. exactly the ones that had already hurt.

A list extended at each new finding is a COUNTER, not a gate. So this check is
derived from the BASE CLASS: whatever the non-Unified counterpart provides and
the Unified one does not IS the surface, computed at test time, and a new
divergence enters it without anyone remembering to add it.

THE NAMING TRAP THAT MADE THIS FAMILY HARD TO SEE, recorded because it is the
reason two separate readers reached the same wrong conclusion: ``Unified*``
spans TWO unrelated subsystems. ``UnifiedRadixCache`` / ``UnifiedTreeNode`` are
the radix TREE and are LIVE on this rig; the unified memory POOLS
(``UnifiedKVPool``, ``UnifiedSWAKVPool``) are gated behind
``--enable-unified-memory``, which is False by default, is set nowhere in the
tree, and whose handler asserts ``speculative_algorithm is None`` while this rig
always runs MTP. A shared prefix suggests a shared gate that does not exist.

THE DISCRIMINATOR, without which this check cries wolf. Not every `hasattr` on
a diverging name is a cross-class probe. ``session_radix_cache`` guards
``_session_leaves`` with `hasattr` and then CREATES it
(`_ensure_session_radix_state`) -- lazy init, entirely correct, and it would be
a false positive here. The rule that separates them: a probe is a cross-class
miss only if the probing module NEVER ASSIGNS the attribute itself. A gate that
fires on a legitimate idiom gets muted, which is how the original condition
survived.

WHERE THIS CHECK STRUCTURALLY DOES NOT REACH, named rather than papered over:
  * factory indirection -- a class produced by a builder rather than named here;
  * `getattr(obj, some_variable)` -- a probe whose name is computed, not literal;
  * registry / dispatch-table lookup, where neither side appears in the source;
  * a consumer that reads `obj.attr` DIRECTLY and lets AttributeError fly. That
    is loud rather than silent and is a different (better) failure, so it is out
    of scope by intent, not by oversight.

Hermetic: source inspection only. No CUDA, no pools, no boot.
"""

import inspect
import pathlib
import re
import unittest

from sglang.test.test_utils import CustomTestCase

#: (unified module, unified class, base module, base class). The PAIRING is the
#: only hand-written input; the SURFACE is derived from it. A new Unified class
#: with no pair here fails `test_every_unified_class_is_paired`, so the pairing
#: cannot silently fall behind either.
PAIRS = (
    (
        "sglang.srt.mem_cache.unified_radix_cache",
        "UnifiedRadixCache",
        "sglang.srt.mem_cache.radix_cache",
        "RadixCache",
    ),
    (
        "sglang.srt.mem_cache.unified_radix_cache",
        "UnifiedTreeNode",
        "sglang.srt.mem_cache.radix_cache",
        "TreeNode",
    ),
)

#: Divergences a consumer duck-types, each with a decided verdict. This is NOT
#: the surface (that is derived) -- it is the RECORD that each derived finding
#: was looked at. An unrecorded finding fails the test, which is the whole
#: point: the fourth instance lands here instead of in a boot.
RECORDED = {
    ("UnifiedTreeNode", "value"): (
        "#872b: rows live in component_data[ct].value. Miss returns None = "
        "'no rows' -> the #662 evict rung frees nothing, silently. ALARMED at "
        "kv_radix_watermark._warn_if_node_rows_unreadable; NOT repaired, "
        "because a widened read without a matching actuator is #718."
    ),
    ("UnifiedTreeNode", "host_value"): (
        "#872b landmine, still UNREACHABLE: hicache_demotion is imported only "
        "by hiradix_cache.py and has zero references from unified_radix_cache, "
        "so demote-on-evict is not wired on the live cache. Recorded so it "
        "cannot become live unnoticed."
    ),
    ("UnifiedRadixCache", "_delete_leaf"): (
        "#872b: the watermark's unlink primitive. Absent here; the real shape "
        "is evict(EvictParams) / _evict_component_and_detach_lru / "
        "_remove_leaf_from_parent. Masked today by the `value` miss above."
    ),
}


def _srt_root() -> pathlib.Path:
    """The `sglang/srt` tree to scan.

    Via `__path__`, not `inspect.getfile`: `sglang.srt` is a NAMESPACE package
    with no `__file__`, and getfile raises "is a built-in module" on it.
    """
    import sglang.srt as srt

    return pathlib.Path(list(srt.__path__)[0])


def _surface(cls) -> set:
    """Every attribute an INSTANCE of ``cls`` can carry.

    Class attributes plus `self.x =` anywhere in the MRO -- the same rule the
    #872 suite established, because instance attributes assigned in __init__
    are invisible to `hasattr` at class level and treating them as missing
    produced a false positive there.
    """
    out = {k for k in dir(cls) if not k.startswith("__")}
    for base in inspect.getmro(cls):
        if base is object:
            continue
        try:
            src = inspect.getsource(base)
        except (OSError, TypeError):  # pragma: no cover
            continue
        out |= set(re.findall(r"self\.([a-zA-Z_]\w*)\s*(?::[^=\n]+)?=", src))
    return out


def _load(mod_name: str, cls_name: str):
    return getattr(__import__(mod_name, fromlist=[cls_name]), cls_name, None)


def _divergences() -> dict:
    """{unified_cls_name: names the BASE provides and it does not}."""
    out = {}
    for m1, c1, m2, c2 in PAIRS:
        a, b = _load(m1, c1), _load(m2, c2)
        if a is None or b is None:  # pragma: no cover
            continue
        out[c1] = _surface(b) - _surface(a)
    return out


_PROBE = re.compile(r"(?:getattr|hasattr)\(\s*\w+\s*,\s*\"([a-zA-Z_]\w*)\"")


def _module_assigns(text: str, name: str) -> bool:
    """Does this module assign the attribute itself? Then it is lazy init.

    `session_radix_cache` guards `_session_leaves` with `hasattr` and then
    creates it. Counting that as a cross-class probe would be a false alarm on
    a correct idiom.
    """
    return re.search(rf"self\.{re.escape(name)}\s*(?::[^=\n]+)?=", text) is not None


def _findings() -> dict:
    """{(unified_cls, name): [files]} -- derived, every run."""
    div = _divergences()
    root = _srt_root()
    hits: dict = {}
    for path in root.rglob("*.py"):
        if path.name == "unified_radix_cache.py":
            continue  # the class's own module defines the other half
        try:
            text = path.read_text(errors="ignore")
        except OSError:  # pragma: no cover
            continue
        for name in set(_PROBE.findall(text)):
            if _module_assigns(text, name):
                continue
            for cls_name, names in div.items():
                if name in names:
                    hits.setdefault((cls_name, name), []).append(path.name)
    return hits


class TestUnifiedFamilyDivergence(CustomTestCase):
    def test_the_surface_is_derived_and_is_wider_than_the_known_pains(self):
        """The direct answer to 'does this cover the family or the three pains'.

        `TreeNode` exposes six names `UnifiedTreeNode` lacks. If the derived
        surface ever collapses to the two that were fixed, this check has
        turned back into the list it replaced.
        """
        div = _divergences()
        node = div.get("UnifiedTreeNode", set())
        self.assertGreaterEqual(
            len(node),
            5,
            f"UnifiedTreeNode divergence collapsed to {sorted(node)}; the "
            "derived surface should carry every name TreeNode has and it does "
            "not",
        )
        for expected in ("value", "host_value", "lock_ref"):
            self.assertIn(expected, node)
        self.assertGreaterEqual(len(div.get("UnifiedRadixCache", set())), 10)

    def test_it_finds_a_site_no_instance_fix_ever_touched(self):
        """Generality, proved on a name that has NOT hurt yet.

        `host_value` is not one of the three fixed pains and appears in none of
        the #872/#872b assertions. A check that only knew those three could not
        surface it; this one does, from the base.
        """
        found = _findings()
        self.assertIn(
            ("UnifiedTreeNode", "host_value"),
            found,
            "the derived check no longer surfaces the host_value probe in "
            "hicache_demotion -- it has narrowed to the already-known names",
        )

    def test_every_derived_finding_has_a_recorded_verdict(self):
        """THE GATE. An unrecorded divergence fails here, not in a boot."""
        found = _findings()
        unrecorded = sorted(k for k in found if k not in RECORDED)
        self.assertEqual(
            unrecorded,
            [],
            "consumer code duck-types names the Unified class does not carry, "
            "and nobody has decided what that means: "
            + ", ".join(
                f"{c}.{n} (in {', '.join(found[(c, n)])})" for c, n in unrecorded
            ),
        )

    def test_lazy_init_is_not_counted_as_a_cross_class_probe(self):
        """No crying wolf on a correct idiom.

        `session_radix_cache._session_leaves` diverges AND is `hasattr`-probed,
        but the same module creates it. Counting it would make this gate fire
        on working code, and a gate that fires on working code gets muted.
        """
        self.assertTrue(
            _module_assigns("        self._session_leaves = x\n", "_session_leaves")
        )
        self.assertNotIn(("UnifiedRadixCache", "_session_leaves"), _findings())

    def test_a_NEW_divergence_is_detected_without_touching_this_file(self):
        """The can-fail the class question turns on.

        Synthetic pair: a base carrying an attribute its 'unified' counterpart
        lacks. The name is invented, so it is in no list anywhere -- if the
        derived surface catches it, the mechanism is a gate; if only the three
        known names are ever caught, it is a counter.
        """

        class _Base:
            def __init__(self):
                self.freshly_invented_attr = 1
                self.value = 2

        class _Unified:
            def __init__(self):
                self.value = 2

        div = _surface(_Base) - _surface(_Unified)
        self.assertIn("freshly_invented_attr", div)
        self.assertNotIn("value", div)

    def test_every_unified_class_is_paired(self):
        """A new Unified* TREE class must be paired, or this check silently
        stops covering it. Scoped to the radix-tree module on purpose: the
        unified memory POOLS are a different subsystem behind a different gate
        (--enable-unified-memory), and pairing them here would assert a
        relationship that does not exist."""
        import sglang.srt.mem_cache.unified_radix_cache as M

        declared = {
            n
            for n, o in vars(M).items()
            if inspect.isclass(o)
            and n.startswith("Unified")
            and o.__module__ == M.__name__
        }
        paired = {c1 for _, c1, _, _ in PAIRS}
        missing = sorted(declared - paired - {"UnifiedLRUList"})
        self.assertEqual(
            missing,
            [],
            f"unpaired Unified tree classes: {missing}. Pair them in PAIRS or "
            "record why they have no non-Unified counterpart.",
        )


if __name__ == "__main__":
    unittest.main()
