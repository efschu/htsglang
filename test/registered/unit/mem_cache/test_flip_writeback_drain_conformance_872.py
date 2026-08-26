"""#872: the flip writeback fence never waited for a single storage ack.

THE DEFECT, IN ONE LINE. ``_await_storage_acks`` finds its drain by name:

    drain = getattr(tree_cache, "_drain_storage_control_queues_local", None)
    if drain is None:
        return 0, before

The live tree cache in this build is ``UnifiedRadixCache``, which is
``(KVCacheEventMixin, BasePrefixCache)`` -- it does NOT descend from
``HiRadixCache``. It implements ``_drain_storage_control_queues_impl`` and, at
the time of writing, no ``_local`` wrapper; only ``HiRadixCache`` and
``HiMambaRadixCache`` carry that name. So the probe misses, the function
returns ``(0, before)`` at once, and the fence's whole contract -- "wait for
the storage acknowledgements before the tree is dropped" -- is void. It waits
zero seconds and reports ``acked=0``.

WHAT IT LOOKED LIKE IN THE FIELD, and why nobody saw it. On boot
``boot_w40_857strict_0826_0516.log``: 21 fence reports (7 cutovers x 3 ranks),
``acked=0`` on EVERY ONE, and three of them printed

    deadline reached with backups still in flight ... outstanding=1
    elapsed=0.000s/2.000s

a deadline "reached" after zero of its two seconds. That impossible pair is
the signature of the early return, not of a slow backend.

WHY THE EXISTING SUITE IS GREEN. ``test_flip_writeback_703.py``'s ``_FakeTree``
defines ``_drain_storage_control_queues_local``. The fake was written from the
CALLER's assumption about the name and never checked against the class that
actually gets bound, so the probe hits in the test and misses in production.
A fake built to the probe cannot falsify the probe.

THE CLASS OF DEFECT: a duck-typed ``getattr`` capability probe whose miss path
degrades to the same value a healthy no-op returns. ``(0, before)`` is
indistinguishable from "there was nothing to acknowledge", so the absence of
the capability is unobservable -- no log, no raise, no counter. That is why
this survived a boot with the #871 streak alarm armed: the alarm reads
``persisted_nothing``, and ``already_staged`` from the normal write policy kept
that False on the fences that had anything at all.

WHAT THIS SUITE PINS

* ``test_unified_radix_cache_provides_the_local_drain`` -- the instance. The
  bound class must carry the name the fence probes for.
* ``test_fence_acknowledges_against_the_live_class_surface`` -- the behaviour,
  through a fake mirroring the REAL surface (``_impl``, no ``_local``): the
  fence must acknowledge, not return zero.
* ``test_every_probed_name_exists_on_every_bindable_cache`` -- THE
  FUTURE-DETECTING CHECK. The probed names are read out of the module source,
  so a probe added or renamed later is covered without anyone remembering to
  extend this file, and a cache class that cannot serve the fence is named at
  desk time instead of costing a boot.
* ``test_an_unservable_cache_is_loud_not_silent`` -- the class fix. A probe
  that cannot resolve must SAY SO. A future fifth cache class will meet this
  same wall; it must meet it as a log line, not as a silent zero.

Hermetic: no CUDA, no pool, no server.
"""

import inspect
import pathlib
import re
import types
import unittest

from sglang.srt.mem_cache import hicache_flip_writeback
from sglang.srt.mem_cache.hicache_flip_writeback import flip_writeback
from sglang.test.test_utils import CustomTestCase

# The fence's drain is looked up under this name. It is spelled out here rather
# than imported so that renaming the probe without renaming the implementations
# fails LOUDLY in this suite instead of quietly in a boot.
LOCAL_DRAIN = "_drain_storage_control_queues_local"


def _bindable_cache_classes():
    """Every tree cache that can be ``scheduler.tree_cache`` with storage on.

    Imported lazily and individually: a class that is absent from a given
    build is skipped by name, never silently dropped from the sweep.
    """
    out = []
    for mod_name, cls_name in (
        ("sglang.srt.mem_cache.unified_radix_cache", "UnifiedRadixCache"),
        ("sglang.srt.mem_cache.hiradix_cache", "HiRadixCache"),
        ("sglang.srt.mem_cache.hi_mamba_radix_cache", "HiMambaRadixCache"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
        except Exception:  # pragma: no cover - build without that backend
            continue
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            out.append(cls)
    return out


def _provides(cls, name: str) -> bool:
    """Does an INSTANCE of ``cls`` carry ``name``?

    ``hasattr(cls, ...)`` alone is wrong here and would have produced a false
    positive on ``ongoing_backup``, which every one of these classes assigns in
    ``__init__`` and none declares at class level. So: class attribute, or an
    ``self.<name> =`` assignment anywhere in the class or its BASES.

    The MRO walk is not padding. Checking only ``cls``'s own body reported
    ``root_node`` missing from ``HiRadixCache`` and ``HiMambaRadixCache``,
    which is false -- both inherit it from the ``RadixCache``/``MambaRadixCache``
    base that assigns it in ``reset()``. A conformance check that cries wolf on
    inheritance gets muted, and a muted check is the failure mode this whole
    ticket is about.
    """
    if hasattr(cls, name):
        return True
    pattern = re.compile(rf"self\.{re.escape(name)}\s*(?::[^=\n]+)?=")
    for base in inspect.getmro(cls):
        if base is object:
            continue
        try:
            src = inspect.getsource(base)
        except (OSError, TypeError):  # pragma: no cover - C or builtin base
            continue
        if pattern.search(src):
            return True
    return False


def _unified_radix_cache_cls():
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    return UnifiedRadixCache


def _assigned_anywhere(name: str) -> bool:
    """Does anything under ``srt/mem_cache`` assign ``<obj>.name``?

    Evidence that a probed STATE attribute is injected from outside the class
    -- which the assembler really does, e.g. ``mamba_cache.cache_controller =
    cache_controller`` at ``hybrid_pool_assembler.py:1583``. Reading the class
    body alone reports such a name missing, which is a false alarm.
    """
    root = pathlib.Path(inspect.getfile(hicache_flip_writeback)).parent
    pattern = re.compile(rf"\.{re.escape(name)}\s*(?::[^=\n]+)?=\s*[^=]")
    for path in root.rglob("*.py"):
        try:
            if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                return True
        except OSError:  # pragma: no cover
            continue
    return False


def _probed_names():
    """Names ``hicache_flip_writeback`` duck-types off the tree cache.

    Read out of the source on purpose. A hand-kept list would drift the moment
    someone adds a probe, and drift in THIS list is the exact failure the
    module is being tested for.
    """
    src = inspect.getsource(hicache_flip_writeback)
    return sorted(set(re.findall(r'getattr\(\s*tree_cache\s*,\s*"([^"]+)"', src)))


class _Backend:
    def __init__(self):
        self.canonical_kv_page = object()  # non-None: the #706 format is on
        self.written = []

    def set(self, key, value):
        self.written.append((key, value))
        return True


class _Node:
    def __init__(self, hash_value):
        self.hash_value = [hash_value]
        self.host_value = None
        self.children = {}
        self.parent = None

    @property
    def backuped(self):
        return self.host_value is not None


class _UnifiedShapedTree:
    """A tree cache carrying ``UnifiedRadixCache``'s REAL local-drain wrapper.

    ``_drain_storage_control_queues_local`` below is the production function
    object, taken off the class under test and bound here -- not a
    reimplementation. That is what makes this a test of the fix rather than a
    test of a copy of the fix, and it is precisely what
    ``test_flip_writeback_703.py``'s ``_FakeTree`` got wrong: that fake wrote
    its OWN ``_local``, so it proved the fence works against a cache that has
    the method and said nothing about the cache that is actually bound.

    Only ``_impl`` is faked, with the five-argument signature
    ``UnifiedRadixCache`` really uses -- including ``extra_release_counts``,
    which the two ``HiRadixCache``-family signatures do not take, so a wrapper
    that called ``_impl`` positionally or omitted that keyword fails here.

    The real function is bound in ``__init__`` rather than in the class body on
    purpose: a class-body lookup raises ``AttributeError`` at IMPORT when the
    method is missing, which makes the whole module uncollectable and takes the
    sibling tests -- including the conformance sweep that names the problem --
    down with it. An absent method must fail this file as an assertion, not as
    a collection error.
    """

    def __init__(self, *, backend, n_pending=2):
        self.cache_controller = type("_CC", (), {"storage_backend": backend})()
        self.enable_storage = True
        self.root_node = _Node(hash_value=None)
        self.root_node.hash_value = None
        self._backend = backend
        self.ongoing_backup = {}
        self.impl_drains = 0
        real = getattr(_unified_radix_cache_cls(), LOCAL_DRAIN, None)
        if callable(real):
            self._drain_storage_control_queues_local = types.MethodType(real, self)
        for i in range(n_pending):
            node = _Node(hash_value=f"warm{i}")
            node.parent = self.root_node
            self.root_node.children[i] = node

    def write_backup(self, node, write_back=False):
        if node.host_value is not None:
            return 0
        node.host_value = b"payload"
        return 1

    def writing_check(self, write_back=False):
        for i, node in self.root_node.children.items():
            if node.host_value is not None and i not in self.ongoing_backup:
                self.ongoing_backup[i] = node

    def _drain_storage_control_queues_impl(
        self,
        n_revoke=None,
        n_backup=None,
        n_release=None,
        extra_release_counts=None,
        log_metrics=False,
    ):
        self.impl_drains += 1
        for op_id, node in list(self.ongoing_backup.items()):
            self._backend.set(node.hash_value[0], node.host_value)
            self.ongoing_backup.pop(op_id, None)


class _UndrainableTree(_UnifiedShapedTree):
    """A cache the fence genuinely cannot drain: the local wrapper is absent.

    This stands in for the NEXT cache class to be written without one -- the
    condition the class-level fix has to make visible.
    """

    _drain_storage_control_queues_local = None

    def __init__(self, *, backend, n_pending=2):
        super().__init__(backend=backend, n_pending=n_pending)
        # Undo the base's binding of the real method: this stand-in is defined
        # by NOT having one.
        self.__dict__.pop(LOCAL_DRAIN, None)


class TestFlipWritebackDrainConformance(CustomTestCase):
    # -- the instance -------------------------------------------------------

    def test_unified_radix_cache_provides_the_local_drain(self):
        """The class that is actually bound must answer the fence's probe.

        RED before the fix: ``UnifiedRadixCache`` had ``_impl`` and no
        ``_local``, so the fence returned ``(0, before)`` without draining and
        every fence of every boot reported ``acked=0``.
        """
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        self.assertTrue(
            _provides(UnifiedRadixCache, LOCAL_DRAIN),
            f"UnifiedRadixCache has no {LOCAL_DRAIN}; the flip writeback fence "
            f"probes for exactly that name and silently skips the ack drain "
            f"without it -- acked=0 on every fence, retention void.",
        )

    # -- the behaviour ------------------------------------------------------

    def test_fence_acknowledges_against_the_live_class_surface(self):
        """Driven through a fake with the real surface, the fence must wait.

        RED before the fix: ``acknowledged == 0`` and ``outstanding == 2``,
        with ``elapsed_s`` at zero -- the log's impossible "deadline reached"
        after none of its two seconds.
        """
        backend = _Backend()
        tree = _UnifiedShapedTree(backend=backend, n_pending=2)
        report = flip_writeback(tree, deadline_s=2.0)

        self.assertEqual(
            report.outstanding,
            0,
            f"fence left backups in flight against a cache it could have "
            f"drained: {report.as_log()}",
        )
        self.assertGreater(
            report.acknowledged,
            0,
            f"fence acknowledged nothing: {report.as_log()}",
        )
        self.assertFalse(report.persisted_nothing)
        self.assertTrue(report.complete)
        self.assertGreater(
            tree.impl_drains, 0, "the drain implementation was never called"
        )

    # -- the future-detecting check ----------------------------------------

    def test_every_probed_name_exists_on_every_bindable_cache(self):
        """THE CHECK THAT MAKES THE CLASS DISCOVERABLE WITHOUT A BOOT.

        Every name the fence duck-types off the tree cache, against every cache
        class that can be bound. Probed names are read from the module source,
        so this keeps covering probes nobody thought to add here.
        """
        probed = _probed_names()
        self.assertIn(LOCAL_DRAIN, probed, "the drain probe vanished from the module")
        classes = _bindable_cache_classes()
        self.assertTrue(classes, "no bindable cache class could be imported")

        # TWO TIERS, because the two kinds of probe fail differently.
        #
        # A METHOD probe must resolve on the class or its bases -- nothing
        # injects a method onto a cache from outside, so an unresolved one is
        # the #872 defect exactly. STATE may legitimately be injected by the
        # assembler (`hybrid_pool_assembler.py:1583` sets
        # `mamba_cache.cache_controller`), which no amount of reading the class
        # body will show; for those the requirement is that SOMETHING in the
        # package assigns them. Collapsing the two tiers would either miss the
        # defect or cry wolf on the assembler, and a check that cries wolf gets
        # muted.
        missing_methods = [
            f"{cls.__name__}.{name}"
            for cls in classes
            for name in probed
            if name.startswith("_drain") and not _provides(cls, name)
        ]
        self.assertEqual(
            missing_methods,
            [],
            "flip writeback probes METHODS these bindable caches do not carry, "
            "so the fence degrades into a no-op on them: " + ", ".join(missing_methods),
        )

        unassigned = [
            name
            for name in probed
            if not name.startswith("_drain")
            and not any(_provides(cls, name) for cls in classes)
            and not _assigned_anywhere(name)
        ]
        self.assertEqual(
            unassigned,
            [],
            "flip writeback probes state nothing in the package ever assigns, "
            "so the probe can only ever return its default: " + ", ".join(unassigned),
        )

    # -- the class fix ------------------------------------------------------

    def test_an_unservable_cache_is_loud_not_silent(self):
        """A probe that cannot resolve must be observable.

        This is the root of the CLASS, not of the instance: naming the drain on
        one more class fixes today's cache and leaves the next one to fail the
        same invisible way. The miss must produce a log record, and the report
        must not be able to claim completeness while carrying work it never
        drained.
        """
        backend = _Backend()
        tree = _UndrainableTree(backend=backend)
        with self.assertLogs(hicache_flip_writeback.__name__, level="ERROR") as cm:
            report = flip_writeback(tree, deadline_s=0.05)
        joined = "\n".join(cm.output)
        self.assertIn("#872", joined)
        self.assertIn(type(tree).__name__, joined)
        self.assertGreater(
            report.outstanding, 0, "an undrained backup must be reported outstanding"
        )
        self.assertFalse(report.complete)


if __name__ == "__main__":
    unittest.main()
