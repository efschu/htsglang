"""#1206 / R1: the transfer domain is the ENTRIES' KEY SPACE, not a count.

WHAT IS BROKEN, AND WHERE
-------------------------
The host->device restore loop's iteration domain is a boot-frozen COUNT
(``transfer_layer_num``) while the pool mappings are keyed by GLOBAL layer
id (``memory_pool.py:1932``
``self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}``).
``_make_layer_mapper`` (``hybrid_pool_assembler.py:44-53``) bounds a GLOBAL id
by that COUNT at ``:49`` ``if not 0 <= layer_id < transfer_layer_num:``, so on
a PP stage whose ids start above the count the mapper answers ``None`` for
EVERY key it holds: with ``tln=18`` and keys ``32..49`` (PP1) nothing is
restored at all, and the tree still calls the prefix resident.

The counter that the restore loop drives is a SECOND consumer of the same
count, and the two halves of the tree wait on it in two different index
spaces: ``HybridReqToTokenPool._wait_for_mamba_layer`` waits on the GLOBAL id
(``memory_pool.py:2001``) while ``HybridLinearKVPool._wait_for_layer`` waits on
``self.local_slot(layer_id)`` (``:5004``). One counter, two index spaces.

WHAT THESE TESTS PIN (spec rows T-1, T-2, T-2b, T-3, T-10, T-11, T-12)
----------------------------------------------------------------------
* T-1  -- a ``PoolEntry`` keeps its MAPPING and can answer for every key it
         holds; the group's domain is ``1 + max(key)`` and equals the device
         side's key space (the ``owned == driven`` predicate at group level).
* T-2  -- the group's per-layer load visits every mapped layer of EVERY entry
         exactly once, anchor included.
* T-2b -- the draft tier's own bound is MEMBERSHIP in its key space, not a
         count compared against a global id.
* T-3  -- the driven domain follows the rebind, because the loop reads it
         through ``self.mem_pool_host`` and ``_stamp`` moves that attribute.
* T-10 -- the index a consumer waits on IS the index at which the producer
         completed that layer's copy. This is the wrong-answer hazard the
         wider domain CREATES if only one of the two halves is landed.
* T-11 -- the mamba wait's bound is the counter's own width.
* T-12 -- the counter is resized to EXACTLY the driven domain, in place, and
         never grow-only: an unrecorded event queries True, so a counter wider
         than the domain acks a load whose copies have not landed.

Hermetic: CPU only, no CUDA. ``device_module.Event()`` constructs and
``query()`` answers without a device; ``record()`` does not, so no test here
records one.
"""

import ast
import inspect
import textwrap
import types
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.mem_cache.hicache_phase_binding import PhasePools, _stamp
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry

# Qwen3.8-27B, 64 layers, ``full_attention_interval = 4``: full attention sits
# at [3, 7, 11, ..., 63] and every other layer is GDN (48 GDN + 16 full).
# Config: /spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8/config.json
# The three PP stages of this boot are [0,32), [32,50) and [50,64); each has
# its own test method below rather than a subTest loop, so a FAILED name line
# and the summary count cannot disagree.


def _stage_maps(start, end):
    """The two device maps for one PP stage, built as the device pools build
    them: GLOBAL layer id -> dense local index (``memory_pool.py:1932`` for
    mamba, ``:4841-4843`` for KV)."""
    full_ids = [i for i in range(start, end) if i % 4 == 3]
    mamba_ids = [i for i in range(start, end) if i % 4 != 3]
    full_map = {layer_id: i for i, layer_id in enumerate(full_ids)}
    mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_ids)}
    return full_map, mamba_map


def _host_pool_stub(layer_num=4):
    pool = MagicMock()
    pool.page_size = 1
    pool.layout = "layer_first"
    pool.device = "cpu"
    pool.size = 16
    pool.can_use_write_back_jit = False
    pool.layer_num = layer_num
    return pool


def _group_for_stage(start, end):
    full_map, mamba_map = _stage_maps(start, end)
    anchor = PoolEntry(
        name=PoolName.KV,
        host_pool=_host_pool_stub(len(full_map)),
        device_pool=MagicMock(),
        layer_mapping=full_map,
        is_primary_index_anchor=True,
    )
    extra = PoolEntry(
        name=PoolName.MAMBA,
        host_pool=_host_pool_stub(len(mamba_map)),
        device_pool=MagicMock(),
        layer_mapping=mamba_map,
    )
    return HostPoolGroup([anchor, extra]), full_map, mamba_map


class TestPoolEntryKeySpace(unittest.TestCase):
    """T-1."""

    def _check_stage(self, start, end):
        group, full_map, mamba_map = _group_for_stage(start, end)

        # (a) per entry: the entry answers for every key it holds.
        for entry in group.entries:
            for key in entry.layer_mapping:
                self.assertIsNotNone(
                    entry.local_layer(key),
                    f"{entry.name} cannot answer for its own key {key}",
                )

        # (b) owned == driven at group level: the DEVICE maps' key space
        # against the HOST group's entries' key space. Two objects, one kind.
        owned = set(mamba_map) | set(full_map)
        driven = set().union(*(set(e.layer_mapping) for e in group.entries))
        self.assertEqual(owned, driven)

        # (c) the same equality one level up, scalar against scalar.
        self.assertEqual(group.transfer_layer_domain, 1 + max(owned))

    def test_pool_entry_key_space_is_inside_its_transfer_domain_pp0(self):
        self._check_stage(0, 32)

    def test_pool_entry_key_space_is_inside_its_transfer_domain_pp1(self):
        self._check_stage(32, 50)

    def test_pool_entry_key_space_is_inside_its_transfer_domain_pp2(self):
        self._check_stage(50, 64)

    def test_an_empty_group_has_a_zero_domain(self):
        """The ``max()`` of an empty key set is not an exception."""
        entry = PoolEntry(
            name=PoolName.KV,
            host_pool=_host_pool_stub(0),
            device_pool=MagicMock(),
            layer_mapping={},
            is_primary_index_anchor=True,
        )
        self.assertEqual(HostPoolGroup([entry]).transfer_layer_domain, 0)


class TestGroupLoadCoverage(unittest.TestCase):
    """T-2."""

    def _check_stage(self, start, end):
        group, full_map, mamba_map = _group_for_stage(start, end)
        seen = {PoolName.KV: [], PoolName.MAMBA: []}

        def _record(name):
            def load_to_device_per_layer(
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
            ):
                seen[name].append(layer_id)

            return load_to_device_per_layer

        for entry in group.entries:
            entry.host_pool.load_to_device_per_layer = _record(entry.name)

        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([1], dtype=torch.int64),
            device_indices=torch.tensor([2], dtype=torch.int64),
        )
        for i in range(group.transfer_layer_domain):
            group.load_to_device_per_layer(
                MagicMock(),
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([0], dtype=torch.int64),
                i,
                "direct",
                pool_transfers=[transfer],
            )

        self.assertEqual(sorted(seen[PoolName.KV]), sorted(full_map.values()))
        self.assertEqual(sorted(seen[PoolName.MAMBA]), sorted(mamba_map.values()))

    def test_group_load_visits_every_mapped_layer_exactly_once_pp0(self):
        self._check_stage(0, 32)

    def test_group_load_visits_every_mapped_layer_exactly_once_pp1(self):
        self._check_stage(32, 50)

    def test_group_load_visits_every_mapped_layer_exactly_once_pp2(self):
        self._check_stage(50, 64)


class TestDraftTierDomain(unittest.TestCase):
    """T-2b."""

    def test_draft_tier_loop_uses_a_domain_not_a_frozen_count(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            _host_pool_covers_layer,
        )

        covered = {0: 0, 2: 1, 5: 2}
        draft = types.SimpleNamespace(
            entries=[types.SimpleNamespace(layer_mapping=covered)],
            layer_num=len(covered),
        )
        visited = [i for i in range(8) if _host_pool_covers_layer(draft, i)]
        self.assertEqual(visited, [0, 2, 5])

        # The defect, stated as an inequality between the two candidates: the
        # frozen count and the key space disagree on this tier, so a count
        # bound visits 1 (uncovered) and skips 5 (covered).
        frozen = [i for i in range(8) if i < draft.layer_num]
        self.assertNotEqual(visited, frozen)

    def test_a_plain_draft_tier_is_dense_from_zero(self):
        """A non-composite host pool: membership IS ``range(layer_num)``, so
        the conversion is byte-identical there."""
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            _host_pool_covers_layer,
        )

        plain = types.SimpleNamespace(layer_num=4)
        self.assertEqual(
            [i for i in range(8) if _host_pool_covers_layer(plain, i)],
            [0, 1, 2, 3],
        )


class TestDomainFollowsRebind(unittest.TestCase):
    """T-3."""

    def test_controller_layer_domain_follows_the_rebind(self):
        boot_group, _, _ = _group_for_stage(32, 50)
        tp_full = {i: i for i in range(64) if i % 4 == 3}
        tp_mamba = {i: i for i in range(64) if i % 4 != 3}
        tp_anchor = PoolEntry(
            name=PoolName.KV,
            host_pool=_host_pool_stub(len(tp_full)),
            device_pool=MagicMock(),
            layer_mapping=tp_full,
            is_primary_index_anchor=True,
        )
        tp_extra = PoolEntry(
            name=PoolName.MAMBA,
            host_pool=_host_pool_stub(len(tp_mamba)),
            device_pool=MagicMock(),
            layer_mapping=tp_mamba,
        )
        tp_group = HostPoolGroup([tp_anchor, tp_extra])

        controller = types.SimpleNamespace(mem_pool_host=boot_group)
        self.assertEqual(controller.mem_pool_host.transfer_layer_domain, 50)

        _stamp(
            controller,
            PhasePools(
                phase="tp",
                device_pool=None,
                allocator=None,
                host_pool=tp_group,
            ),
            7,
        )
        self.assertEqual(
            set(range(controller.mem_pool_host.transfer_layer_domain)),
            set(range(64)),
        )

    def test_the_loop_reads_the_domain_through_the_swapped_attribute(self):
        """The half that makes the rebind assertion load-bearing: a domain
        cached on the CONTROLLER would survive the swap above and still be
        wrong. The loop must read it through ``self.mem_pool_host``."""
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(HybridCacheController.start_loading))
        )
        bounds = [
            node.iter
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and getattr(node.iter.func, "id", None) == "range"
        ]
        self.assertEqual(len(bounds), 1)
        arg = bounds[0].args[0]
        self.assertEqual(ast.unparse(arg), "self.mem_pool_host.transfer_layer_domain")


class TestCounterIndexSpace(unittest.TestCase):
    """T-10 and T-11."""

    @staticmethod
    def _recording_counter():
        counter = types.SimpleNamespace(completed=[], waited=[], num_layers=50)
        counter.wait_until = counter.waited.append
        return counter

    def test_layer_done_counter_index_space_matches_its_waiters(self):
        full_map, mamba_map = _stage_maps(32, 50)
        domain = 1 + max(set(full_map) | set(mamba_map))
        counter = self._recording_counter()

        # PRODUCER: the loop completes step i for every i of the domain.
        completed_at = {}
        for i in range(domain):
            completed_at[i] = i

        # CONSUMER, KV half: the real method, on a stand-in that owns this
        # stage's full-attention ids.
        kv = types.SimpleNamespace(
            layer_transfer_counter=counter,
            local_slot=lambda layer_id: full_map[layer_id],
        )
        for global_id in full_map:
            HybridLinearKVPool._wait_for_layer(kv, global_id)

        # CONSUMER, mamba half: the real method.
        mamba = types.SimpleNamespace(
            layer_transfer_counter=counter,
            _mamba_transfer_frame=len(full_map) + len(mamba_map),
        )
        for global_id in mamba_map:
            HybridReqToTokenPool._wait_for_mamba_layer(mamba, global_id)

        self.assertEqual(
            counter.waited,
            [completed_at[g] for g in list(full_map) + list(mamba_map)],
        )
        # The finish sentinel is load_events[-1], so the last index of the
        # domain must be one the producer actually completes.
        self.assertIn(domain - 1, completed_at)

    def test_mamba_layer_wait_bound_is_the_counter_width(self):
        counter = self._recording_counter()
        counter.num_layers = 64
        pool = types.SimpleNamespace(
            layer_transfer_counter=counter,
            _mamba_transfer_frame=18,
        )
        HybridReqToTokenPool._wait_for_mamba_layer(pool, 49)
        self.assertEqual(counter.waited, [49])

    def test_no_counter_or_no_frame_is_still_a_silent_no_wait(self):
        """The ``None`` semantics ``swa_memory_pool`` relies on: the presence
        test survives; only the BOUND moves to the counter's width."""
        counter = self._recording_counter()
        no_counter = types.SimpleNamespace(
            layer_transfer_counter=None, _mamba_transfer_frame=18
        )
        HybridReqToTokenPool._wait_for_mamba_layer(no_counter, 3)
        no_frame = types.SimpleNamespace(
            layer_transfer_counter=counter, _mamba_transfer_frame=None
        )
        HybridReqToTokenPool._wait_for_mamba_layer(no_frame, 3)
        self.assertEqual(counter.waited, [])


class TestCounterResize(unittest.TestCase):
    """T-12."""

    def test_counter_width_equals_the_driven_domain(self):
        counter = LayerDoneCounter(18)
        counter_id = id(counter)
        event_ids = [id(e) for e in counter.events]

        counter.resize(64)
        self.assertEqual(counter.num_layers, 64)

        counter.resize(50)
        self.assertEqual(counter.num_layers, 50)
        self.assertEqual(id(counter), counter_id)
        self.assertEqual([id(e) for e in counter.events], event_ids)
        for event in counter.events:
            self.assertEqual(len(event.load_events), 50)
            self.assertEqual(event._num_layers, 50)
            # DANGER DIRECTION: a grow-only resize leaves the finish sentinel
            # on load_events[63], an event the 50-domain loop never records.
            # An unrecorded event queries True, so the ACK fires before the
            # copies land.
            self.assertIs(event.finish_event, event.load_events[49])
            self.assertTrue(event.finish_event.query())


# ---------------------------------------------------------------------------
# S1 FIX 1. The four holes the fixer round measured in this file: the attach
# instrument had no test at all, the loop's own body was never driven (T-2b
# and T-10's producer half were fixture arithmetic), and the two lines that
# WIRE the derived domain -- the counter resize in ``__init__`` and the draft
# guard's third conjunct -- could each be deleted with the suite still green.
# ---------------------------------------------------------------------------


class _RecordingProducerEvent:
    """Stands in for ``LayerLoadingEvent`` without a device.

    ``record()``/``wait()`` are the two calls that need a CUDA context; the
    only thing under test here is the SEQUENCE handed to ``complete``.
    """

    def __init__(self):
        self.completed = []
        self.start_event = types.SimpleNamespace(
            record=lambda: None, wait=lambda _stream: None
        )
        self.finish_event = types.SimpleNamespace()

    def complete(self, layer_index):
        self.completed.append(layer_index)


class _StubDeviceModule:
    """``device_module.stream(...)`` as a no-op context manager."""

    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    def stream(self, _stream):
        return self._Ctx()


class _LoopDriver:
    """Drive ``HybridCacheController.start_loading``'s BODY hermetically.

    HAZARD this closes: the loop is the one place where the domain, the draft
    guard and the producer index meet, and every earlier test in this file
    drove a re-implementation of it from the test side. A re-implementation
    cannot fail when the tree's own composition changes.
    """

    def __init__(self, group, draft_host=None, draft_armed=False):
        self.controller = object.__new__(HybridCacheController)
        c = self.controller
        c.load_queue = [MagicMock()]
        c.ack_load_queue = []
        c.load_stream = None
        c.mem_pool_host = group
        c.mem_pool_device = MagicMock()
        c.mem_pool_host_draft = draft_host
        c.mem_pool_device_draft = MagicMock()
        c.io_backend = "direct"
        c.layer_done_counter = MagicMock()
        c.layer_done_counter.update_producer.return_value = 0
        self.producer_event = _RecordingProducerEvent()
        c.layer_done_counter.events = [self.producer_event]
        self.host_indices = torch.arange(4)
        self.device_indices = torch.arange(4)
        c.move_hybrid_indices = lambda op: (
            self.host_indices,
            self.device_indices,
            [],
        )
        c._dcp_kv_transfer_pairs = lambda h, d: (h, d)
        c.draft_tier_armed = lambda _direction: draft_armed
        self.draft_visited = []
        if draft_host is not None:
            draft_host.load_to_device_per_layer = (
                lambda *a, **kw: self.draft_visited.append(a[3])
            )

    def run(self):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hcc

        saved_dm = hcc.device_module
        saved_gate = hcc.consume_gate
        hcc.device_module = _StubDeviceModule()
        # The phase gate is S2/#760's question, not this one; neutered so the
        # loop's own composition is what the assertion reads.
        hcc.consume_gate = lambda _c, _q, _d: True
        try:
            HybridCacheController.start_loading(self.controller)
        finally:
            hcc.device_module = saved_dm
            hcc.consume_gate = saved_gate


class _CountingHostGroup:
    """A group with a real domain whose per-layer load only records.

    It carries `entries` as well as the domain, because the two numbers a PP
    stage keeps apart are exactly `len(keys)` and `1 + max(key)`: a stand-in
    that only knows the domain cannot tell a loop driving the right COUNT of
    layers from one driving the right ONES.
    """

    def __init__(self, domain, keys=None):
        self.transfer_layer_domain = domain
        keys = range(domain) if keys is None else keys
        self.entries = [
            types.SimpleNamespace(layer_mapping={k: i for i, k in enumerate(keys)})
        ]
        self.visited = []

    def load_to_device_per_layer(self, _dev, _h, _d, layer_id, _backend, **kw):
        self.visited.append(layer_id)


class _SparseDraftHostPool:
    """A draft host tier whose covered ids are NOT ``range(layer_num)``."""

    def __init__(self, covered):
        self.entries = [
            types.SimpleNamespace(
                layer_mapping={key: i for i, key in enumerate(covered)}
            )
        ]
        self.layer_num = len(covered)


class TestTheLoopBodyItself(unittest.TestCase):
    """T-2b (the LOOP, not the predicate) and T-10's producer half."""

    def test_the_producer_completes_every_global_id_of_the_domain_in_order(self):
        """E-1's producer contract, driven through the tree's own loop.

        DANGER DIRECTION: feeding ``step`` (or ``i - start_layer``) to
        ``producer_event.complete`` looks identical in a coverage line -- the
        right COUNT of layers is completed -- while the mamba waiter, which
        waits on the GLOBAL id, joins an index the producer never records.
        """
        # PP1's key space: 18 keys spanning 32..49, so `len(keys)` and the
        # domain are 18 and 50 -- the two numbers a step-indexed loop confuses.
        group = _CountingHostGroup(50, keys=range(32, 50))
        driver = _LoopDriver(group)
        driver.run()

        self.assertEqual(group.visited, list(range(50)))
        self.assertEqual(driver.producer_event.completed, list(range(50)))

    def test_the_draft_load_visits_exactly_the_covered_global_ids(self):
        """T-2b as the spec states it: every covered draft layer visited, no
        uncovered one -- through the loop's own guard composition."""
        group = _CountingHostGroup(50)
        draft = _SparseDraftHostPool([32, 35, 40, 49])
        driver = _LoopDriver(group, draft_host=draft, draft_armed=True)
        driver.run()

        self.assertEqual(driver.draft_visited, [32, 35, 40, 49])

    def test_a_disarmed_draft_tier_is_not_loaded_at_all(self):
        group = _CountingHostGroup(8)
        draft = _SparseDraftHostPool([0, 1])
        driver = _LoopDriver(group, draft_host=draft, draft_armed=False)
        driver.run()

        self.assertEqual(driver.draft_visited, [])


def _conjuncts(node):
    out = []
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        for value in node.values:
            out.extend(_conjuncts(value))
    else:
        out.append(node)
    return out


class TestTheTwoWiringLines(unittest.TestCase):
    """The two lines that connect the derived domain to its consumers.

    HAZARD these close: each is a single statement whose deletion leaves every
    behavioural test in this file green -- the counter keeps the base's narrow
    width, and ``_host_pool_covers_layer`` becomes dead code beside a restored
    frozen bound. A numbered change with no kill relation is a numbered change
    a later edit can drop silently.
    """

    def test_the_controller_sizes_the_counter_from_the_group(self):
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(HybridCacheController.__init__))
        )
        resizes = [
            call
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "resize"
        ]
        self.assertEqual(len(resizes), 1, "exactly one resize call in __init__")
        self.assertEqual(ast.unparse(resizes[0].func), "self.layer_done_counter.resize")
        self.assertEqual(
            ast.unparse(resizes[0].args[0]),
            "self.mem_pool_host.transfer_layer_domain",
        )

    def test_a_constructed_controller_carries_the_groups_domain_on_its_counter(self):
        """The same wiring as BEHAVIOUR, through the real ``__init__``.

        HAZARD the AST pin above cannot close: a statement that is written is
        not a statement that RUNS, and neither pin alone shows that the counter
        the pools were already handed is the one that widens. The identity
        assertion is the second half -- a fresh ``LayerDoneCounter`` would
        satisfy the width and leave the device and req pools holding the narrow
        object nobody rebinds.
        """
        from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hcc

        group = _CountingHostGroup(50, keys=range(32, 50))
        # 18 = the base's own `mem_pool_device.layer_num` meaning: this PP
        # stage's layer COUNT, the number the domain must replace.
        counter = LayerDoneCounter(18)

        def _base_init(self, **kwargs):
            # The two attributes the real base sets that the line under test
            # reads: `cache_controller.py:657` and `:732`. Nothing else of the
            # base is needed, and standing in for the rest is what keeps this
            # hermetic.
            self.mem_pool_host = kwargs["mem_pool_host"]
            self.layer_done_counter = counter

        saved = hcc.BaseHiCacheController.__init__
        hcc.BaseHiCacheController.__init__ = _base_init
        try:
            controller = hcc.HybridCacheController(
                token_to_kv_pool_allocator=MagicMock(),
                mem_pool_host=group,
                page_size=1,
                tp_group=MagicMock(),
                load_cache_event=MagicMock(),
            )
        finally:
            hcc.BaseHiCacheController.__init__ = saved

        self.assertIs(controller.layer_done_counter, counter)
        self.assertEqual(counter.num_layers, 50)
        for event in counter.events:
            self.assertEqual(len(event.load_events), 50)
            self.assertEqual(event._num_layers, 50)

    def test_the_draft_guard_reads_the_membership_predicate(self):
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(HybridCacheController.start_loading))
        )
        guards = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and "draft_tier_armed" in ast.unparse(node.test)
        ]
        self.assertEqual(len(guards), 1, "one draft guard in start_loading")
        terms = [ast.unparse(t) for t in _conjuncts(guards[0].test)]
        self.assertIn(
            "_host_pool_covers_layer(self.mem_pool_host_draft, i)",
            terms,
            "the call site must read the membership predicate, not a count",
        )
        self.assertNotIn("i < self.mem_pool_host_draft.layer_num", terms)


class TestTheExpectedDomainAccessor(unittest.TestCase):
    """S1-C18 / D-74. Slot 15's right-hand term is read through this NAME.

    HAZARD this closes: S0's reader treats a MISSING accessor as healthy
    (``phase_domain_verdict.py`` returns the MIN-neutral 1 when the name is
    absent), so an omitted method is a STOP that can never fire rather than a
    crash anyone would see.
    """

    def test_the_accessor_is_a_method_returning_the_groups_own_driven_domain(self):
        group, _full, _mamba = _group_for_stage(32, 50)
        self.assertTrue(callable(group.expected_transfer_layer_domain))
        self.assertEqual(group.expected_transfer_layer_domain("tp"), 50)
        self.assertEqual(
            group.expected_transfer_layer_domain("pp"), group.transfer_layer_domain
        )
        # D-12: a READ over the entries, not a stored copy -- so it follows a
        # group whose entries name a different key space.
        other, _f, _m = _group_for_stage(0, 32)
        self.assertEqual(other.expected_transfer_layer_domain("pp"), 32)

    def test_the_accessor_takes_the_bound_phase_parameter(self):
        """B6 fills the body from a phase-keyed table; a B1 signature without
        the parameter forces S7 to widen a signature it may not touch."""
        sig = inspect.signature(HostPoolGroup.expected_transfer_layer_domain)
        self.assertEqual(list(sig.parameters), ["self", "bound_phase"])


class TestTheAttachInstrument(unittest.TestCase):
    """T-43, the PP half. S1-C15's line is the ONLY instrument that can show
    a domain/key-space mismatch, and it had no test at all: the whole call
    could be deleted with this file still green.
    """

    def _attach_records(self, group, pp_rank=2):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa

        cache = MagicMock()
        cache.components = {}
        kvcache = MagicMock()
        params = MagicMock()
        params.pp_rank = pp_rank
        result = hpa.StackBuildResult(
            host_pool_group=group,
            cache_controller=MagicMock(),
            component_host_pools={},
            sidecars=[],
            register_req_to_token_counter=False,
            pools_desc="KV + MAMBA",
        )
        with self.assertLogs(hpa.logger, level="INFO") as captured:
            hpa._apply_stack_result(cache, kvcache, params, result)
        return [
            line for line in captured.output if "#1206 TRANSFER DOMAIN attach" in line
        ]

    def test_attach_line_names_the_domain_and_the_per_pool_key_ranges(self):
        group, full_map, mamba_map = _group_for_stage(32, 50)
        lines = self._attach_records(group, pp_rank=2)

        self.assertEqual(len(lines), 1, "exactly ONE attach line per stack")
        line = lines[0]
        self.assertIn("rank=2", line)
        self.assertIn("stack=pp", line)
        # The GROUP property, printed as an integer -- not the count that
        # could not show the mismatch.
        self.assertIn("domain=50", line)
        # Each pool's OWN key set, named by the pool -- so a reader cannot
        # take the domain for a key range or one pool's span for another's.
        self.assertIn(
            "%s: keys[%d..%d] n=%d"
            % (PoolName.KV, min(full_map), max(full_map), len(full_map)),
            line,
        )
        self.assertIn(
            "%s: keys[%d..%d] n=%d"
            % (PoolName.MAMBA, min(mamba_map), max(mamba_map), len(mamba_map)),
            line,
        )

    def test_the_attach_line_prints_this_ranks_own_number(self):
        group, _f, _m = _group_for_stage(0, 32)
        lines = self._attach_records(group, pp_rank=0)
        self.assertEqual(len(lines), 1)
        self.assertIn("rank=0", lines[0])
        self.assertIn("domain=32", lines[0])


class _DraftHostPool:
    def __init__(self, layer_num, size=8):
        self.layer_num = layer_num
        self.size = size


class TestTheDraftTierDomainTerm(unittest.TestCase):
    """T-36, S1-C12's LOCAL half -- WITHDRAWN, and this class is what is left.

    THE COMPARISON WAS WRONG, AND IT KILLED A BOOT. It read the DRAFT tier's
    layer count against the TARGET tier's ``transfer_layer_domain``, i.e. 1
    against 64 on every speculative config, and slot 10 is an AND slot. The
    premise behind it -- "one loop is driven over two domains, so whichever is
    narrower is silently truncated" -- is false of this loop: the draft half
    is bounded by the draft tier's OWN count at ``cache_controller.py:1961-1963``
    (``and i < self.mem_pool_host_draft.layer_num``), so the layers the draft
    does not cover are skipped BY CONSTRUCTION and not silently.

    The record's remaining alternative -- compare the draft tier against its
    OWN domain -- is a tautology on a plain dense-from-0 draft tier and a NEW
    false STOP on a composite one (a PP-sharded group legitimately has
    ``layer_num`` 18 and ``transfer_layer_domain`` 50). So the term is deleted
    with that reason, slot 10 keeps its declaration and reads its MIN-neutral
    1, and these arms pin the silence.
    """

    def _controller(self, domain):
        from sglang.srt.managers import cache_controller as cc

        controller = object.__new__(cc.HiCacheController)
        controller.has_draft = False
        controller.mem_pool_device_draft = None
        controller.mem_pool_host_draft = None
        controller.mem_pool_host = types.SimpleNamespace(transfer_layer_domain=domain)
        controller.draft_owner_phase = None
        controller.draft_binding_generation = None
        controller.draft_identity = None
        controller._maybe_register_draft_with_storage = lambda: None
        return controller

    def test_registering_a_narrower_draft_tier_is_silent(self):
        """RED BEFORE THE FIX: a 1-layer MTP draft tier under a 64-layer
        target domain emitted one ERROR line and voted the group down."""
        from sglang.srt.managers import cache_controller as cc

        for draft_layers in (1, 18, 64):
            with self.subTest(draft_layers=draft_layers):
                controller = self._controller(64)
                with self.assertLogs(cc.logger, level="INFO") as captured:
                    cc.HiCacheController.set_draft_kv_pool(
                        controller, MagicMock(), _DraftHostPool(draft_layers)
                    )
                self.assertEqual(
                    [
                        line
                        for line in captured.output
                        if "#1206 DRAFT TIER DOMAIN MISMATCH" in line
                    ],
                    [],
                )

    def test_the_wrong_comparison_is_gone_from_both_halves(self):
        """The DELETION, read off the modules themselves, so a reader who
        re-introduces either half is red rather than merely un-pinned. Both
        halves computed the SAME wrong predicate, and a fix to one alone
        leaves the other voting or logging it."""
        import inspect

        from sglang.srt.managers import cache_controller as cc
        from sglang.srt.managers import phase_domain_verdict as pdv

        self.assertFalse(hasattr(pdv, "_draft_tier_domain_matches"))
        self.assertNotIn(
            "DRAFT TIER DOMAIN MISMATCH",
            inspect.getsource(cc.HiCacheController.set_draft_kv_pool),
        )

    def test_the_draft_half_of_the_restore_loop_bounds_itself(self):
        """WHY the term is gone rather than repaired: the loop already bounds
        the draft half by the draft tier's own count, so there is no second
        domain for it to be truncated against."""
        import inspect

        from sglang.srt.managers import cache_controller as cc

        src = inspect.getsource(cc.HiCacheController.start_loading)
        self.assertIn("i < self.mem_pool_host_draft.layer_num", src)


class _CounterSpy:
    """A ``LayerDoneCounter`` stand-in that records resizes.

    ``consumer_index`` IS ITS OWN KNOB, and it is separate from the
    controller's ``ack_load_queue`` on purpose. A single ``quiescent`` flag
    setting both at once is what let an off-by-one on the consumer arm survive
    the whole slice: the queue alone carried every refusal, so no assertion
    ever read the consumer term. The two states this counter can be in --
    "a load's events are mid-record" and "a forward pass was pointed at an
    event slot" -- are different facts and are driven separately.
    """

    def __init__(self, num_layers, *, consumer_index=-1):
        self.num_layers = num_layers
        self.consumer_index = consumer_index
        self.resized_to = []

    def resize(self, num_layers):
        self.resized_to.append(num_layers)
        self.num_layers = num_layers


class TestTheCounterFollowsTheRebind(unittest.TestCase):
    """The rebind's own half of slot 15: the counter is RESIZED to the
    incoming group's driven domain, in place, at the one site that moves the
    readers.

    WHAT WAS BROKEN. ``HybridCacheController.__init__`` resized the counter
    once, at boot, and nothing resized it again -- so after the pp->tp cutover
    the restore loop drove a 64-wide domain against a counter of width 32
    (PP0) / 50 (PP1). Measured on the metal at
    ``boot_855_weg1b12s1_6917f46dd5_0906_115410.log``: two
    ``#1206 REBIND DOMAIN OUTSIDE DRIVEN`` lines, ``driven=64
    counter_width=32`` and ``counter_width=50``. The detector was S1's own and
    it was right; the actuator was missing.
    """

    def _rebind_with(self, controller, group):
        from sglang.srt.mem_cache import hicache_phase_binding as hpb

        incoming = MagicMock(spec=PhasePools)
        incoming.phase = "tp"
        incoming.host_pool = group
        incoming.device_pool = MagicMock()
        incoming.device_pool_hybrid = None
        incoming.allocator = MagicMock()
        incoming.layer_num = lambda: group.transfer_layer_domain
        saved = (hpb.check_shapes, hpb.check_pool_coverage)
        hpb.check_shapes = lambda _i: None
        hpb.check_pool_coverage = lambda _r, _i: None
        state = hpb.binding_state()
        before = (state.phase, state.generation)
        try:
            with self.assertLogs(hpb.logger, level="INFO") as captured:
                hpb.rebind({"cache_controller": controller}, incoming)
        finally:
            hpb.check_shapes, hpb.check_pool_coverage = saved
            state._phase, state._generation = before
        return captured.output

    def _controller(self, counter, *, loads_in_flight=0):
        """``loads_in_flight`` is the ack queue's DEPTH, and it is the only
        knob this helper owns -- the counter's ``consumer_index`` is set on the
        counter itself, so the two arms of the precondition can be driven one
        at a time."""
        return types.SimpleNamespace(
            layer_done_counter=counter,
            ack_load_queue=[object() for _ in range(loads_in_flight)],
            mem_pool_host=None,
            mem_pool_device=None,
            mem_pool_device_hybrid=None,
            token_to_kv_pool_allocator=None,
            hicache_binding_generation=None,
        )

    def test_a_quiescent_rebind_resizes_the_counter_and_is_silent(self):
        """RED BEFORE THE FIX: the counter kept its boot width, the detector
        fired, and slot 15 voted the group down at the next packed reduce."""
        from sglang.srt.managers import phase_domain_verdict as pdv

        tp_group, _f, _m = _group_for_stage(0, 64)
        self.assertEqual(tp_group.transfer_layer_domain, 64)
        counter = _CounterSpy(50)
        controller = self._controller(counter)
        lines = self._rebind_with(controller, tp_group)

        self.assertEqual(counter.resized_to, [64])
        self.assertEqual(counter.num_layers, 64)
        self.assertEqual(
            [x for x in lines if "#1206 REBIND DOMAIN OUTSIDE DRIVEN" in x], []
        )
        self.assertEqual(
            pdv._rebind_domain_within_driven(controller, tp_group, "tp"), 1
        )

    def test_the_real_counter_class_is_the_one_that_follows(self):
        """The spy above proves the CALL; this proves the tree's own class
        answers it -- ``LayerDoneCounter.resize`` rebuilds every event list,
        not only the scalar the term reads."""
        from sglang.srt.managers.cache_controller import LayerDoneCounter

        tp_group, _f, _m = _group_for_stage(0, 64)
        counter = LayerDoneCounter(50)
        controller = self._controller(counter)
        self._rebind_with(controller, tp_group)

        self.assertEqual(counter.num_layers, 64)
        for event in counter.events:
            self.assertEqual(len(event.load_events), 64)

    def test_a_load_in_flight_refuses_the_resize_and_names_both_numbers(self):
        """THE PRECONDITION, and it is not decoration: rebuilding the event
        list under an in-flight load hands the ACK an event nothing recorded.
        The resize is declined, ONE line names the reason, and the detector
        still fires so slot 15 carries the group STOP.

        THE ACK QUEUE ARM ALONE, with the consumer arm parked at -1. The two
        were driven together until this round, so a change to the consumer arm
        moved no assertion at all.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        tp_group, _f, _m = _group_for_stage(0, 64)
        counter = _CounterSpy(50, consumer_index=-1)
        controller = self._controller(counter, loads_in_flight=1)
        lines = self._rebind_with(controller, tp_group)

        self.assertEqual(counter.resized_to, [])
        self.assertEqual(counter.num_layers, 50)
        refusals = [
            x for x in lines if "#1206 REBIND COUNTER RESIZE REFUSED" in x
        ]
        self.assertEqual(len(refusals), 1)
        self.assertIn("driven=64", refusals[0])
        self.assertIn("counter_width=50", refusals[0])
        self.assertIn("ack_load_queue=1", refusals[0])
        self.assertIn("consumer_index=-1", refusals[0])
        self.assertEqual(
            len([x for x in lines if "#1206 REBIND DOMAIN OUTSIDE DRIVEN" in x]),
            1,
            "the detector still fires, so slot 15 carries the STOP",
        )
        self.assertEqual(
            pdv._rebind_domain_within_driven(controller, tp_group, "tp"), 0
        )

    def test_a_consumer_index_alone_does_not_decline_the_resize(self):
        """THE FIFTH-CUTOVER BOOT KILLER, driven at the state the metal was in.

        boot_855_weg1b12s1f3_6d78227979_0906_134543.log, generation=5, all
        three ranks: ``#1206 REBIND COUNTER RESIZE REFUSED reader=cache_controller
        driven=64 counter_width=32 ack_load_queue=0 consumer_index=1`` on PP0
        and the same with ``counter_width=50`` on PP1. The ack queue was EMPTY;
        the consumer term alone declined the resize, the counter kept the
        outgoing width, the detector fired, and slot 15 stopped the group.

        WHY THE EMPTY QUEUE IS ALREADY THE PROOF. An ack is appended the
        instant the copies are enqueued (`hybrid_cache_controller.py:714-720`,
        `cache_controller.py:1985`) and is popped only after its finish event
        has queried True (`unified_radix_cache.py:5328-5332` counts the leading
        ready entries, `:5392-5393` pops exactly those). An empty queue is
        therefore proof that no load's events are mid-record.
        ``consumer_index`` is not that fact: it is the event-slot pointer the
        last forwarded batch left on the counter (`tp_worker.py:553` ->
        `:487-489`), and it stays >= 0 until a batch without a load or a
        ``reset()`` moves it.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        tp_group, _f, _m = _group_for_stage(0, 64)
        counter = _CounterSpy(50, consumer_index=1)
        controller = self._controller(counter, loads_in_flight=0)
        lines = self._rebind_with(controller, tp_group)

        self.assertEqual(counter.resized_to, [64])
        self.assertEqual(counter.num_layers, 64)
        self.assertEqual(
            [x for x in lines if "#1206 REBIND COUNTER RESIZE REFUSED" in x],
            [],
            "an empty ack queue is quiescent; the consumer pointer is not a "
            "load in flight",
        )
        self.assertEqual(
            [x for x in lines if "#1206 REBIND DOMAIN OUTSIDE DRIVEN" in x], []
        )
        self.assertEqual(
            pdv._rebind_domain_within_driven(controller, tp_group, "tp"),
            1,
            "slot 15 votes healthy, so no group STOP at the fifth cutover",
        )

    def test_the_counter_narrows_when_the_incoming_domain_is_smaller(self):
        """THE NARROWING DIRECTION -- the danger direction the spec names.

        `WEG1_BUILD_SPEC_0905.md:406-410`: a counter WIDER than the domain the
        loop drives hands the ACK a `load_events[63]` that was never recorded
        and queries True immediately, so the load is acked before its copies
        land. Every other actuator arm here widens 50 -> 64, and a grow-only
        actuator passes all of them.
        """
        tp_group, _f, _m = _group_for_stage(0, 32)
        self.assertEqual(tp_group.transfer_layer_domain, 32)
        counter = _CounterSpy(64)
        controller = self._controller(counter)
        lines = self._rebind_with(controller, tp_group)

        self.assertEqual(counter.resized_to, [32])
        self.assertEqual(counter.num_layers, 32)
        self.assertEqual(
            [x for x in lines if "#1206 REBIND DOMAIN OUTSIDE DRIVEN" in x], []
        )

    def test_the_real_counter_class_narrows_its_event_lists_too(self):
        """The narrowing arm on the tree's own class, and the assertion a
        grow-only actuator cannot pass: ``finish_event`` is ``load_events[-1]``
        (`cache_controller.py:77-79`), so after narrowing to 32 the ACK's
        finish event must BE the 32nd event and not a 64th nothing records."""
        from sglang.srt.managers.cache_controller import LayerDoneCounter

        tp_group, _f, _m = _group_for_stage(0, 32)
        counter = LayerDoneCounter(64)
        controller = self._controller(counter)
        self._rebind_with(controller, tp_group)

        self.assertEqual(counter.num_layers, 32)
        for event in counter.events:
            self.assertEqual(len(event.load_events), 32)
            self.assertEqual(event._num_layers, 32)
            self.assertIs(event.finish_event, event.load_events[31])

    def test_a_reader_without_a_counter_is_not_given_one(self):
        """CAN-NOT-FIRE PIN. The three readers hold different subsets; a
        rebind that invented a counter on one that never had one would be a
        fourth, silent binding."""
        controller = types.SimpleNamespace(
            mem_pool_host=None,
            mem_pool_device=None,
            mem_pool_device_hybrid=None,
            token_to_kv_pool_allocator=None,
            hicache_binding_generation=None,
        )
        tp_group, _f, _m = _group_for_stage(0, 64)
        self._rebind_with(controller, tp_group)
        self.assertFalse(hasattr(controller, "layer_done_counter"))


class TestTheRebindDomainTerm(unittest.TestCase):
    """T-48, S1-C16's LOCAL half, arms (0), (1) and (2).

    TWO OBJECTS IN BOTH ARMS: the counter's live width against the bound
    group's own expected domain. A version reading both from one object is
    the guard-that-cannot-fire this refusal exists to avoid.
    """

    def _readers(
        self,
        counter_width,
        group,
        *,
        loads_in_flight=0,
        consumer_index=-1,
        counter=None,
    ):
        if counter is None:
            counter = _CounterSpy(counter_width, consumer_index=consumer_index)
        controller = types.SimpleNamespace(
            layer_done_counter=counter,
            ack_load_queue=[object() for _ in range(loads_in_flight)],
            mem_pool_host=None,
            mem_pool_device=None,
            mem_pool_device_hybrid=None,
            token_to_kv_pool_allocator=None,
            hicache_binding_generation=None,
        )
        return controller

    def _incoming(self, group):
        pools = MagicMock(spec=PhasePools)
        pools.phase = "tp"
        pools.host_pool = group
        pools.device_pool = MagicMock()
        pools.device_pool_hybrid = None
        pools.allocator = MagicMock()
        pools.layer_num = lambda: group.transfer_layer_domain
        return pools

    def _rebind(self, counter_width, group, *, loads_in_flight=0):
        from sglang.srt.mem_cache import hicache_phase_binding as hpb

        controller = self._readers(
            counter_width, group, loads_in_flight=loads_in_flight
        )
        incoming = self._incoming(group)
        # The two shape guards are S2's subject and refuse a stand-in tier;
        # neutered so what this assertion reads is the domain term alone.
        saved = (hpb.check_shapes, hpb.check_pool_coverage)
        hpb.check_shapes = lambda _i: None
        hpb.check_pool_coverage = lambda _r, _i: None
        # `rebind` ADVANCES THE PROCESS-WIDE BINDING STATE, and leaving it
        # advanced moves `bound_phase()` for every later test in the same
        # pytest process -- measured: 204 unrelated hicache cases in
        # `test_unified_radix_cache_unittest.py` stopped skipping and went
        # red on a whole-directory run while passing in isolation. The state
        # is restored here, not "expected to be reset by the next test".
        state = hpb.binding_state()
        before = (state.phase, state.generation)
        try:
            with self.assertLogs(hpb.logger, level="INFO") as captured:
                hpb.rebind({"controller": controller}, incoming)
        finally:
            hpb.check_shapes, hpb.check_pool_coverage = saved
            # Restore EXACTLY what was there, private fields included:
            # `reset()` returns the BOOT phase and generation 0, which is
            # right in a clean process and wrong if anything upstream had
            # already advanced it.
            state._phase, state._generation = before
        self.assertEqual((state.phase, state.generation), before)
        return controller, [
            line
            for line in captured.output
            if "#1206 REBIND DOMAIN OUTSIDE DRIVEN" in line
        ]

    def test_arm_2_a_stamp_without_a_resize_logs_both_numbers_and_votes_zero(self):
        """The counter is now RESIZED at the rebind, so the one way it can
        still lag is a resize the quiescence precondition declined -- which is
        the state this arm drives. The detector's subject is unchanged: the
        readers moved and the counter did not follow."""
        from sglang.srt.managers import phase_domain_verdict as pdv

        tp_group, _f, _m = _group_for_stage(0, 64)
        self.assertEqual(tp_group.transfer_layer_domain, 64)
        controller, lines = self._rebind(50, tp_group, loads_in_flight=1)

        self.assertEqual(len(lines), 1, "exactly one line at ERROR")
        self.assertIn("64", lines[0])
        self.assertIn("50", lines[0])
        self.assertIn("generation=", lines[0])
        self.assertEqual(
            pdv._rebind_domain_within_driven(controller, tp_group, "tp"), 0
        )

    def test_arm_2b_a_counter_WIDER_than_driven_is_named_too(self):
        """THE DETECTOR'S OTHER DIRECTION, and it is the wrong-answer one.

        Arm 2 above drives 50 against a driven 64 -- the counter too NARROW.
        A detector written `width >= driven` is silent on the opposite state,
        which is the one `WEG1_BUILD_SPEC_0905.md:406-410` calls a wrong answer:
        a counter WIDER than the domain the loop drives hands the ACK an event
        nothing recorded. Reached whenever the resize is declined across a
        tp->pp cutover, where the outgoing width is the larger one.
        """
        from sglang.srt.managers import phase_domain_verdict as pdv

        pp_group, _f, _m = _group_for_stage(0, 32)
        self.assertEqual(pp_group.transfer_layer_domain, 32)
        controller, lines = self._rebind(64, pp_group, loads_in_flight=1)

        self.assertEqual(len(lines), 1, "exactly one line at ERROR")
        self.assertIn("driven=32", lines[0])
        self.assertIn("counter_width=64", lines[0])
        self.assertEqual(
            pdv._rebind_domain_within_driven(controller, pp_group, "pp"), 0
        )

    def test_arm_1_a_counter_that_followed_logs_nothing_and_votes_one(self):
        from sglang.srt.managers import phase_domain_verdict as pdv

        tp_group, _f, _m = _group_for_stage(0, 64)
        controller, lines = self._rebind(64, tp_group)

        self.assertEqual(lines, [])
        self.assertEqual(
            pdv._rebind_domain_within_driven(controller, tp_group, "tp"), 1
        )

    def test_arm_0_before_any_rebind_the_term_is_the_min_neutral(self):
        """CAN-NOT-FIRE PIN. A controller with no counter votes the neutral
        rather than dereferencing one."""
        from sglang.srt.managers import phase_domain_verdict as pdv

        group, _f, _m = _group_for_stage(32, 50)
        controller = types.SimpleNamespace(layer_done_counter=None)
        self.assertEqual(pdv._rebind_domain_within_driven(controller, group, "pp"), 1)


if __name__ == "__main__":
    unittest.main()
