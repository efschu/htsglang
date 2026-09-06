"""#1206 / S1: the BOOT half of the transfer-domain verdicts.

WHAT IS BROKEN, AND WHERE
-------------------------
``build_phase_flip_host_pools`` (``phase_flip_boot.py:2274``) had TEN abrupt
exits -- five ``return``s and five ``raise``s -- and the boot verdicts S1
produces are voted on ONE MIN ``all_reduce`` that runs after it, at
``scheduler.py:961-968``. Every one of those exits lies UPSTREAM of that
reduce, so a rank taking a raising arm died with a named error while its two
peers blocked in a collective nobody would join: the one failure mode that
produces no evidence at all.

The four terms this file drives, all on ``BOOT_REDUCE_LAYOUT``:

* row 0 ``owned_equals_driven``       -- S1-C14, the SHORTFALL predicate
* row 1 ``layer_mapping_non_empty``   -- S1-C13
* row 2 ``counter_index_space_known`` -- S1-C11's group half
* row 12 ``host_pool_build_ok``       -- S1-C14 (e), the three post-build pins

and, on the PACKED bus, S1-C12's slot 10 and S1-C16's slot 15.

WHAT THESE TESTS PIN (spec rows T-10b, T-37, T-38, T-38b..e, T-41, T-48, T-53)
------------------------------------------------------------------------------
Every arm drives THREE stand-in ranks through the SAME element-wise MIN the
boot reduce takes, so "the group stops" is asserted rather than argued, and
"none hangs" is checkable: a rank that never reaches the reduce shows up as a
rank whose vote is missing from the fabric.

Hermetic: CPU only. ``torch.distributed`` is never initialised; the fabric
substitutes the element-wise MIN the collective would compute.
"""

import logging
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers import phase_domain_verdict as pdv
from sglang.srt.managers import phase_flip_boot as pfb
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_LOGGER = logging.getLogger("sglang.srt.managers.phase_flip_boot")


# ---------------------------------------------------------------------------
# Stand-ins. The two device maps are built the way the device pools build
# them: GLOBAL layer id -> dense local index.
# ---------------------------------------------------------------------------


def _host_pool_stub(layer_num):
    pool = MagicMock()
    pool.page_size = 1
    pool.layout = "layer_first"
    pool.device = "cpu"
    pool.size = 16
    pool.can_use_write_back_jit = False
    pool.layer_num = layer_num
    return pool


def _kv_device(keys):
    return types.SimpleNamespace(
        full_attention_layer_id_mapping={k: i for i, k in enumerate(sorted(keys))}
    )


def _mamba_device(keys):
    return types.SimpleNamespace(mamba_map={k: i for i, k in enumerate(sorted(keys))})


def _group(kv_driven, mamba_driven, kv_owned=None, mamba_owned=None):
    """A HostPoolGroup whose HOST mappings and DEVICE maps are set apart.

    Two objects, deliberately: ``driven`` is the host entry's own key set and
    ``owned`` is the device pool's. A version reading both from the group is a
    guard whose two terms are one object.
    """
    kv_owned = kv_driven if kv_owned is None else kv_owned
    mamba_owned = mamba_driven if mamba_owned is None else mamba_owned
    anchor = PoolEntry(
        name=PoolName.KV,
        host_pool=_host_pool_stub(max(1, len(kv_driven))),
        device_pool=_kv_device(kv_owned),
        layer_mapping={k: i for i, k in enumerate(sorted(kv_driven))},
        is_primary_index_anchor=True,
    )
    extra = PoolEntry(
        name=PoolName.MAMBA,
        host_pool=_host_pool_stub(max(1, len(mamba_driven))),
        device_pool=_mamba_device(mamba_owned),
        layer_mapping={k: i for i, k in enumerate(sorted(mamba_driven))},
    )
    return HostPoolGroup([anchor, extra])


def _stage(start, end):
    full = [i for i in range(start, end) if i % 4 == 3]
    mamba = [i for i in range(start, end) if i % 4 != 3]
    return full, mamba


class _SafeWaiter:
    """A pool class row 2 accepts, standing in for the registered one."""


_SafeWaiter.__name__ = "HybridLinearKVPool"


class _LocalIndexWaiter:
    """The ``dsa`` / ``deepseek_v4`` shape: a LOCAL index on a global-id
    producer's counter."""


def _device_owner(kv_keys=(), mamba_keys=(), kvcache_class=None):
    """A stack's DEVICE side: the two GLOBAL-id maps `owned` is read from.

    They live on the wrapper and on the req-to-token pool, one object out from
    `PoolEntry.device_pool`, which is why the reader takes the stack rather
    than the entry.
    """
    kv = types.SimpleNamespace(
        full_attention_layer_id_mapping={k: i for i, k in enumerate(sorted(kv_keys))}
    )
    if kvcache_class is not None:
        kv = kvcache_class()
        kv.full_attention_layer_id_mapping = {
            k: i for i, k in enumerate(sorted(kv_keys))
        }
    return types.SimpleNamespace(
        token_to_kv_pool=kv,
        req_to_token_pool=types.SimpleNamespace(
            mamba_map={k: i for i, k in enumerate(sorted(mamba_keys))}
        ),
    )


def _sched(kvcache_class=_SafeWaiter, rank=0, world=True):
    """A stand-in with the ATTRIBUTE SURFACE THE REAL `Scheduler` HAS.

    The parallel identity is on `ParallelState` (`scheduler.py:860`
    `        self.ps = ParallelState(`) and NOWHERE ELSE -- the #583 fact,
    still pinned by `test_census_attribute_surface_583.py`. A stand-in that
    grants a bare `pp_rank` proves only that the stand-in has it, and every
    #1206 boot line would label itself `rank=0` on the metal while this
    harness read the rank back correctly.
    """
    allocator = types.SimpleNamespace(
        get_kvcache=lambda: (None if kvcache_class is None else kvcache_class())
    )
    return types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_rank=rank, tp_rank=rank),
        token_to_kv_pool_allocator=allocator,
        world_group=(types.SimpleNamespace(cpu_group="cpu-group") if world else None),
    )


def _fresh_result():
    return {
        "owned_equals_driven": 1,
        "layer_mapping_non_empty": 1,
        "host_pool_build_ok": 1,
        "host_pool_build_msg": None,
    }


def _result_from_groups(rank, stacks):
    result = _fresh_result()
    pfb._vote_transfer_domain_terms(result, _LOGGER, rank, stacks)
    return result


def _stack(name, group, kv_owned, mamba_owned):
    return (name, group, _device_owner(kv_owned, mamba_owned))


# ---------------------------------------------------------------------------
# The fabric: three ranks meeting in ONE element-wise MIN.
# ---------------------------------------------------------------------------


class _BootFabric:
    """Run ``vote_phase_flip_boot_verdict`` on every rank against the group
    MIN of all their payloads.

    WHY IT PRE-COMPUTES: a rank that never reaches the reduce is exactly the
    defect under test, so the fabric records who voted. A harness that let one
    rank's absence silently shrink the MIN could not tell a routed refusal
    from a deleted one.
    """

    def __init__(self, cases):
        #: cases: list of (scheduler, result)
        self.cases = cases
        self.voted = []
        #: The REDUCTION OPERATOR and the payload DTYPE the code under test
        #: actually passed, one entry per rank. Recorded rather than ignored:
        #: a fabric that accepts `op` and throws it away, then hand-computes a
        #: MIN, pins nothing about the operator -- and the operator IS the
        #: rank-uniformity mechanism of this bus (one rank's 0 becomes every
        #: rank's raise only under MIN). Measured before this was added: the
        #: mutant `ReduceOp.MIN` -> `ReduceOp.MAX` left every test in this
        #: file passing while all three ranks proceeded past a pin one of
        #: them had refused.
        self.ops = []
        self.dtypes = []
        self._reduced = None

    def _group_min(self):
        payloads = []
        for rank, (sched, result) in enumerate(self.cases):
            terms = pfb.phase_flip_boot_terms(sched, result, _LOGGER, rank)
            payloads.append(pdv.build_boot_reduce_payload(terms))
        return [min(col) for col in zip(*payloads)]

    def reply(self, tensor, op=None, group=None):
        """The stand-in for ``torch.distributed.all_reduce``.

        A METHOD, not a closure, so the refusal below can be driven directly:
        an assertion on a value this object merely recorded is not a pin
        unless the recorder can also fail.
        """
        self.voted.append(group)
        self.ops.append(op)
        self.dtypes.append(tensor.dtype)
        if op is not torch.distributed.ReduceOp.MIN:
            raise AssertionError(
                "#1068 the boot ballot is a MIN reduction; this call passed "
                "%r. Under any other operator a rank that voted 0 stops being "
                "every rank's raise, and the group proceeds past a fact one "
                "of them refused." % (op,)
            )
        # The reply is built in the CALLER'S dtype, never in a dtype of this
        # fabric's own choosing: `copy_` casts silently, so a narrowed payload
        # would truncate inside the copy where no assertion can see it.
        tensor.copy_(torch.tensor(self._reduced, dtype=tensor.dtype))

    def run(self):
        """Returns, per rank: (raised exception or None, ERROR/INFO lines)."""
        self._reduced = self._group_min()
        _fake_all_reduce = self.reply

        out = []
        for rank, (sched, result) in enumerate(self.cases):
            with patch.object(
                torch.distributed, "all_reduce", new=_fake_all_reduce
            ), self.assertLogsOrNone() as captured:
                try:
                    pfb.vote_phase_flip_boot_verdict(sched, result)
                    raised = None
                except Exception as exc:  # noqa: BLE001 - the subject
                    raised = exc
            out.append((raised, captured.lines))
        return out

    class _Capture(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.DEBUG)
            self.lines = []

        def emit(self, record):
            self.lines.append(record.getMessage())

        _NAMES = (
            "sglang.srt.managers.phase_flip_boot",
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler",
        )

        def __enter__(self):
            self._saved = {}
            for name in self._NAMES:
                log = logging.getLogger(name)
                # The LOGGER's own level filters before any handler sees the
                # record, so an INFO instrument is invisible to a handler-only
                # capture -- the shape that made this harness read "no line".
                self._saved[name] = log.level
                log.setLevel(logging.DEBUG)
                log.addHandler(self)
            return self

        def __exit__(self, *exc):
            for name in self._NAMES:
                log = logging.getLogger(name)
                log.removeHandler(self)
                log.setLevel(self._saved[name])
            return False

    def assertLogsOrNone(self):
        return self._Capture()


class TestTheBootReduceLayout(CustomTestCase):
    """T-53. The bus is THIRTEEN wide and every unfilled row is neutral."""

    def test_the_boot_reduce_layout_is_thirteen_wide_and_neutral_when_unfilled(self):
        self.assertEqual(pdv.PHASE_BOOT_REDUCE_SLOTS, 13)
        payload = pdv.build_boot_reduce_payload()
        self.assertEqual(len(payload), 13)
        # An AND row's neutral is 1 and a digest pair's is (0, 0). A bus
        # declared narrower and appended to later makes the unpack return None
        # on a width it does not recognise, and every boot verdict silently
        # stops being read.
        self.assertEqual(payload, [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        for name in (
            "owned_equals_driven",
            "layer_mapping_non_empty",
            "counter_index_space_known",
            "host_pool_build_ok",
        ):
            self.assertEqual(payload[pdv.boot_index_of(name)], 1)

    def test_s1_fills_five_rows_and_row_3_has_no_producer_here(self):
        owners = {row.name: row.owner for row in pdv.BOOT_REDUCE_LAYOUT}
        self.assertEqual(owners["owned_equals_driven"], "S1-C14")
        self.assertEqual(owners["layer_mapping_non_empty"], "S1-C13")
        self.assertEqual(owners["counter_index_space_known"], "S1-C11")
        self.assertEqual(owners["host_pool_build_ok"], "S1-C14")
        self.assertEqual(
            owners["draft_tier_domain_matches_at_boot"], "none on this rig"
        )


class TestTheBallotIsAMinOnTheWorldGroup(CustomTestCase):
    """The three properties of the boot collective that no other test reads.

    Every arm above asserts what the REDUCED VALUE does. None of them asserted
    anything about the CALL that produced it, so three one-token edits at
    ``phase_flip_boot.py`` were free: the operator (MIN -> MAX deletes the
    group STOP while every detection stays intact), the payload dtype (a
    narrowing truncates the S7 digest rows at B6 and reads agreement where
    there is divergence), and the group handle (``tp_cpu_group`` is world size
    1 at boot on this configuration, so the reduce is a no-op and the raise
    below it rank-local -- the D-20 violation produced by the handle rather
    than by the code shape, which the site's own comment names).
    """

    def _healthy_cases(self):
        return [(_sched(rank=r), _fresh_result()) for r in range(3)]

    def test_the_boot_reduce_is_a_min(self):
        fabric = _BootFabric(self._healthy_cases())
        fabric.run()
        self.assertEqual(len(fabric.ops), 3)
        for rank, op in enumerate(fabric.ops):
            self.assertIs(
                op, torch.distributed.ReduceOp.MIN, f"rank {rank} did not vote a MIN"
            )

    def test_the_payload_is_int64(self):
        """The bus carries a ``(x, -x)`` digest pair from B6 on. int8 wraps at
        128, so two ranks whose digests differ by a multiple of 256 truncate
        to the same value and the pair reads agreement where there is none --
        a deleted STOP on the divergence detector itself."""
        fabric = _BootFabric(self._healthy_cases())
        fabric.run()
        self.assertEqual(len(fabric.dtypes), 3)
        for rank, dtype in enumerate(fabric.dtypes):
            self.assertIs(dtype, torch.int64, f"rank {rank} packed {dtype}")

    def test_the_reduce_is_taken_on_the_world_group(self):
        cases = self._healthy_cases()
        fabric = _BootFabric(cases)
        fabric.run()
        self.assertEqual(
            fabric.voted,
            [sched.world_group.cpu_group for sched, _result in cases],
            "the handle passed is the world group's cpu group, by identity",
        )

    def test_the_fabric_itself_refuses_a_non_min_operator(self):
        """CAN-FAIL PROOF for the three assertions above. Asserting on a
        recorded value proves nothing unless the recorder can also refuse, so
        the fabric's OWN reply path is driven here with a MAX and must raise.
        Without this arm, a fabric that recorded the op and never looked at it
        would read identically to one that pins it."""
        fabric = _BootFabric(self._healthy_cases())
        reduced = fabric._group_min()
        tensor = torch.tensor(reduced, dtype=torch.int64)
        with self.assertRaises(AssertionError) as cm:
            fabric.reply(tensor, op=torch.distributed.ReduceOp.MAX, group="g")
        self.assertIn("MIN reduction", str(cm.exception))
        self.assertEqual(fabric.ops, [torch.distributed.ReduceOp.MAX])


class TestTheShortfallTermStopsTheGroup(CustomTestCase):
    """T-38 (row 0) and T-38b (the can-not-fire counterpart)."""

    def _pp_stack(self, name, start, end, drop=()):
        full, mamba = _stage(start, end)
        group = _group(
            [k for k in full if k not in drop],
            [k for k in mamba if k not in drop],
            kv_owned=full,
            mamba_owned=mamba,
        )
        return _stack(name, group, full, mamba)

    def test_boot_owned_not_equal_driven_raises_on_every_rank(self):
        """T-38. One rank's host tier maps fewer layers than its device pool
        owns; ALL THREE ranks must raise, and none may reach a state where it
        proceeds while its peers block."""
        cases = []
        for rank, (start, end) in enumerate([(0, 32), (32, 50), (50, 64)]):
            drop = (33,) if rank == 1 else ()
            stack = self._pp_stack("pp", start, end, drop=drop)
            cases.append((_sched(rank=rank), _result_from_groups(rank, [stack])))
        fabric = _BootFabric(cases)
        outcomes = fabric.run()

        self.assertEqual(len(fabric.voted), 3, "every rank reached the reduce")
        for rank, (raised, _lines) in enumerate(outcomes):
            self.assertIsInstance(
                raised, pdv.PhaseDomainDivergence, f"rank {rank} must raise"
            )
            self.assertIn("owned_equals_driven", str(raised))

    def test_the_shortfall_stop_carries_the_reason_on_the_raise(self):
        """T-38, second half: *the message names ``driven``, ``owned`` and
        ``missing=[...]`` with their denominators*.

        The log line already named them; the RAISE did not, and the raise is
        what the boot dies with. A STOP that says only ``reason=None`` on all
        three ranks cannot say which rank refused or why -- the operator then
        reads three identical lines and has to go find a fourth instrument.
        Rank 1 carries its OWN recorded reason; ranks 0 and 2 say a peer
        refused, because an int64 MIN carries no string.
        """
        cases = []
        for rank, (start, end) in enumerate([(0, 32), (32, 50), (50, 64)]):
            drop = (33,) if rank == 1 else ()
            stack = self._pp_stack("pp", start, end, drop=drop)
            cases.append((_sched(rank=rank), _result_from_groups(rank, [stack])))
        outcomes = _BootFabric(cases).run()

        refusing = str(outcomes[1][0])
        self.assertIn("#1206 TRANSFER DOMAIN SHORTFALL rank=1", refusing)
        self.assertIn("missing=[33]", refusing)
        self.assertIn("driven=", refusing)
        self.assertIn("owned=", refusing)
        self.assertIn("at least one rank refused", refusing)
        self.assertNotIn("the ranks do not agree", refusing)
        for rank in (0, 2):
            healthy = str(outcomes[rank][0])
            self.assertIn("peer refused", healthy)
            self.assertNotIn("missing=[33]", healthy)

    def test_the_shortfall_names_the_missing_layers_on_the_rank_that_has_them(self):
        stack = self._pp_stack("pp", 32, 50, drop=(33,))
        result = _fresh_result()
        with self.assertLogs(_LOGGER, level="ERROR") as captured:
            pfb._vote_transfer_domain_terms(result, _LOGGER, 1, [stack])
        lines = [x for x in captured.output if "TRANSFER DOMAIN SHORTFALL" in x]
        self.assertEqual(len(lines), 1)
        self.assertIn("missing=[33]", lines[0])
        self.assertEqual(result["owned_equals_driven"], 0)

    def test_the_shortfall_fires_on_the_kv_pool_as_well_as_the_mamba_one(self):
        """PIN (not red-first): the KV arm of the ``owned`` read.

        WHY IT EXISTS. Both other SHORTFALL drives above drop layer ``33``,
        and for stage ``[32, 50)`` the split is
        ``full == [35, 39, 43, 47]`` / ``33 in mamba`` -- so ``owned !=
        driven`` was only ever made true on the MAMBA entry and the KV branch
        of ``_device_layer_keys`` was only ever driven on its equal, silent
        arm. A branch no drive can disturb is not established, and this branch
        is the one that carries row 0's ``owned`` term for
        ``PoolName.KV``. Dropping ``35`` -- a FULL-ATTENTION key of that same
        stage -- drives it.
        """
        stack = self._pp_stack("pp", 32, 50, drop=(35,))
        result = _fresh_result()
        with self.assertLogs(_LOGGER, level="ERROR") as captured:
            pfb._vote_transfer_domain_terms(result, _LOGGER, 1, [stack])
        lines = [x for x in captured.output if "TRANSFER DOMAIN SHORTFALL" in x]
        self.assertEqual(len(lines), 1)
        self.assertIn(str(PoolName.KV), lines[0])
        self.assertIn("missing=[35]", lines[0])
        self.assertEqual(result["owned_equals_driven"], 0)

    def test_the_kv_map_is_reached_through_the_allocator_on_the_pp_stack(self):
        """PIN (not red-first): the OWNER SHAPE the live pp stack has.

        ``build_phase_flip_host_pools`` passes ``("pp", pp_host, scheduler)``
        into the vote, and the ``Scheduler`` has NO ``token_to_kv_pool``
        (``scheduler.py:944`` keeps only
        ``self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator``;
        ``model_runner_kv_cache_mixin.py:2933`` is where the direct attribute
        is set, on the TP side). So on every real boot the pp stack's row-0
        ``owned`` term can only come from the allocator route -- the one arm
        no other test drives, because the stand-in owner sets
        ``token_to_kv_pool`` directly, which is the TP shape.
        """
        full, mamba = _stage(0, 32)
        kv = _kv_device(full)
        pp_owner = types.SimpleNamespace(
            token_to_kv_pool_allocator=types.SimpleNamespace(get_kvcache=lambda: kv),
            req_to_token_pool=types.SimpleNamespace(
                mamba_map={k: i for i, k in enumerate(mamba)}
            ),
        )
        self.assertEqual(pfb._device_layer_keys(pp_owner, PoolName.KV), set(full))
        self.assertEqual(pfb._device_layer_keys(pp_owner, PoolName.MAMBA), set(mamba))

    def test_the_owner_field_the_docstring_forbids_is_not_the_one_read(self):
        """DANGER DIRECTION for the KV arm, and the one the docstring names:
        reading ``PoolEntry.device_pool`` / an owner-side ``device_pool``
        answers ``None`` on every real boot, which leaves row 0 permanently
        neutral -- a guard that cannot fire, dressed as a comparison. An owner
        that carries ONLY that field must therefore read NOT COMPARABLE, and
        one that carries the real map must not.
        """
        full, _mamba = _stage(0, 32)
        decoy = types.SimpleNamespace(device_pool=_kv_device(full))
        self.assertIsNone(pfb._device_layer_keys(decoy, PoolName.KV))
        self.assertEqual(
            pfb._device_layer_keys(
                types.SimpleNamespace(token_to_kv_pool=_kv_device(full)), PoolName.KV
            ),
            set(full),
        )

    def test_boot_reduce_is_silent_on_a_correct_domain(self):
        """T-38b. CAN-NOT-FIRE PIN (excluded from the red-first total).

        DANGER DIRECTION, and it is the mirror of the usual one: reading
        ``driven`` as the GROUP's scalar domain instead of the matching
        ENTRY's key set makes this fail on all four stand-ins (24 == 32,
        14 == 50, 10 == 64, 48 == 64), so the guard fires ALWAYS and the boot
        never starts.
        """
        cases = []
        stages = [(0, 32), (32, 50), (50, 64)]
        for rank, (start, end) in enumerate(stages):
            pp_stack = self._pp_stack("pp", start, end)
            pin_full, pin_mamba = _stage(0, 64)
            pin = _group(pin_full, pin_mamba)
            cases.append(
                (
                    _sched(rank=rank),
                    _result_from_groups(
                        rank, [pp_stack, _stack("tp", pin, pin_full, pin_mamba)]
                    ),
                )
            )
        fabric = _BootFabric(cases)
        outcomes = fabric.run()

        for rank, (raised, lines) in enumerate(outcomes):
            self.assertIsNone(raised, f"rank {rank} raised: {raised}")
            self.assertEqual([x for x in lines if "TRANSFER DOMAIN SHORTFALL" in x], [])
            self.assertEqual([x for x in lines if "EMPTY LAYER MAPPING" in x], [])

    def test_a_correct_pin_prints_its_own_attach_line_at_the_vote_site(self):
        """T-43's TP half: the pin was unlogged, which is why a mismatch on
        it was invisible while the pp stacks printed a count."""
        pin_full, pin_mamba = _stage(0, 64)
        pin = _group(pin_full, pin_mamba)
        result = _result_from_groups(2, [_stack("tp", pin, pin_full, pin_mamba)])
        result["tp"] = pin
        fabric = _BootFabric([(_sched(rank=2), result)])
        ((raised, lines),) = fabric.run()

        self.assertIsNone(raised)
        attach = [x for x in lines if "#1206 TRANSFER DOMAIN attach" in x]
        self.assertEqual(len(attach), 1)
        self.assertIn("rank=2", attach[0])
        self.assertIn("stack=tp", attach[0])
        self.assertIn("domain=64", attach[0])


class TestTheEmptyMappingTerm(CustomTestCase):
    """T-41 (row 1)."""

    def test_boot_empty_layer_mapping_raises_on_every_rank(self):
        cases = []
        for rank in range(3):
            full, mamba = _stage(0, 32)
            if rank == 1:
                # A named entry that maps no layer while its device half maps
                # none either -- so `owned == driven` HOLDS and stays silent.
                group = _group(full, [], kv_owned=full, mamba_owned=[])
                stack = _stack("pp", group, full, [])
            else:
                group = _group(full, mamba)
                stack = _stack("pp", group, full, mamba)
            cases.append((_sched(rank=rank), _result_from_groups(rank, [stack])))
        fabric = _BootFabric(cases)
        outcomes = fabric.run()

        self.assertEqual(len(fabric.voted), 3)
        for rank, (raised, _lines) in enumerate(outcomes):
            self.assertIsInstance(raised, pdv.PhaseDomainDivergence)
            self.assertIn("layer_mapping_non_empty", str(raised))

    def test_the_two_terms_are_not_redundant(self):
        """CAN-FAIL: with the empty pool's device half also empty the
        SHORTFALL term stays silent while this one still refuses."""
        full, _mamba = _stage(0, 32)
        group = _group(full, [], kv_owned=full, mamba_owned=[])
        result = _fresh_result()
        with self.assertLogs(_LOGGER, level="ERROR") as captured:
            pfb._vote_transfer_domain_terms(
                result, _LOGGER, 1, [_stack("pp", group, full, [])]
            )
        self.assertEqual(result["owned_equals_driven"], 1)
        self.assertEqual(result["layer_mapping_non_empty"], 0)
        lines = [x for x in captured.output if "EMPTY LAYER MAPPING" in x]
        self.assertEqual(len(lines), 1)
        self.assertIn(str(PoolName.MAMBA), lines[0])


class TestTheCounterIndexSpaceTerm(CustomTestCase):
    """T-10b (row 2), S1-C11's group half."""

    def test_a_local_index_waiter_is_refused_under_the_global_producer(self):
        for shape in ("DsaCacheLayerSplit", "NpuMemoryPool", "DeepseekV4KVPool"):
            with self.subTest(shape=shape):
                cls = type(shape, (), {})
                sched = _sched(kvcache_class=cls, rank=0)
                with self.assertLogs(_LOGGER, level="ERROR") as captured:
                    term = pfb._counter_index_space_known(sched, _LOGGER, 0)
                self.assertEqual(term, 0)
                lines = [
                    x
                    for x in captured.output
                    if "#1206 COUNTER INDEX SPACE UNKNOWN" in x
                ]
                self.assertEqual(len(lines), 1)
                self.assertIn(shape, lines[0])

    def test_the_two_enumerated_waiters_are_accepted(self):
        for name in ("HybridLinearKVPool", "HybridReqToTokenPool"):
            with self.subTest(name=name):
                cls = type(name, (), {})
                self.assertEqual(
                    pfb._counter_index_space_known(
                        _sched(kvcache_class=cls), _LOGGER, 0
                    ),
                    1,
                )

    def test_an_absent_allocator_votes_the_min_neutral(self):
        sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=0, tp_rank=0))
        self.assertEqual(pfb._counter_index_space_known(sched, _LOGGER, 0), 1)

    def test_one_ranks_unknown_index_space_stops_the_group(self):
        cases = []
        for rank in range(3):
            cls = _LocalIndexWaiter if rank == 2 else _SafeWaiter
            full, mamba = _stage(0, 32)
            group = _group(full, mamba)
            cases.append(
                (
                    _sched(kvcache_class=cls, rank=rank),
                    _result_from_groups(rank, [_stack("pp", group, full, mamba)]),
                )
            )
        outcomes = _BootFabric(cases).run()
        for raised, _lines in outcomes:
            self.assertIsInstance(raised, pdv.PhaseDomainDivergence)
            self.assertIn("counter_index_space_known", str(raised))


class TestTheRoutedExits(CustomTestCase):
    """T-38c (the two pre-``try`` raises) and T-38d (the two NOT-APPLICABLE
    exits)."""

    def _sa(self, **over):
        base = dict(
            phase_flip_rebind_hicache=True,
            chunked_prefill_size=4096,
            max_running_requests=8,
            page_size=1,
            hicache_mem_layout="layer_first",
            hicache_storage_backend="file",
            hicache_host_role="staging",
            hicache_size=6,
            hicache_mamba_host_mib=0,
        )
        base.update(over)
        return types.SimpleNamespace(**base)

    def _builder_sched(self, *, rows=32768, cell=8192, layers=4, **sa_over):
        kv = types.SimpleNamespace(
            size=rows,
            size_per_token=cell,
            device_pool=types.SimpleNamespace(layer_num=layers),
        )
        pp_host = types.SimpleNamespace(
            anchor_entry=types.SimpleNamespace(
                host_pool=kv, device_pool=types.SimpleNamespace(layer_num=layers)
            )
        )
        tree = types.SimpleNamespace(token_to_kv_pool_host=pp_host)
        stacks = types.SimpleNamespace(
            tp_worker=types.SimpleNamespace(
                model_runner=types.SimpleNamespace(
                    token_to_kv_pool=types.SimpleNamespace(size_per_token=cell)
                )
            )
        )
        sched = _sched(rank=0)
        sched.server_args = self._sa(**sa_over)
        sched.tree_cache = tree
        sched.phase_flip_stacks = stacks
        return sched

    def test_the_two_pre_try_refusals_still_refuse_after_the_routing(self):
        """T-38c. Rank 1 satisfies the predicate, ranks 0 and 2 do not:
        all three must raise AFTER the vote, and the two preserved messages
        must still name their rank."""
        for arm, over in (
            ("cell", dict(cell=16383)),
            ("wave", dict(rows=30518)),
        ):
            with self.subTest(arm=arm):
                cases = []
                for rank in range(3):
                    sched = self._builder_sched(**(over if rank == 1 else {}))
                    sched.ps = types.SimpleNamespace(pp_rank=rank, tp_rank=rank)
                    result = pfb.build_phase_flip_host_pools(sched)
                    cases.append((sched, result))
                # The refusing rank RETURNED rather than raising -- that is
                # what separates a routing from a deletion.
                self.assertEqual(cases[1][1]["owned_equals_driven"], 0)
                msg = cases[1][1]["host_pool_build_msg"]
                self.assertIn("rank=1", msg)
                self.assertIn(
                    (
                        "#1068 TP PIN CELL UNDERIVABLE"
                        if arm == "cell"
                        else "#1068 HOST POOL BELOW ONE WAVE"
                    ),
                    msg,
                )
                fabric = _BootFabric(cases)
                outcomes = fabric.run()
                self.assertEqual(len(fabric.voted), 3, "none hangs")
                for rank, (raised, _lines) in enumerate(outcomes):
                    self.assertIsInstance(
                        raised, pdv.PhaseDomainDivergence, f"rank {rank}"
                    )
                # D-45 (ii), the half that was specified and not built: the
                # vote raises group-uniformly CARRYING THIS RANK'S REASON, so
                # the log still says CELL UNDERIVABLE / BELOW ONE WAVE on the
                # rank that hit it and `peer refused` on the others. Without
                # it all three ranks die with `reason=None` and the boot's
                # cause lives only in a log line nothing on the STOP path
                # points at.
                named = (
                    "#1068 TP PIN CELL UNDERIVABLE rank=1"
                    if arm == "cell"
                    else "#1068 HOST POOL BELOW ONE WAVE rank=1"
                )
                self.assertIn(named, str(outcomes[1][0]))
                # WHAT THIS FIXTURE CAN AND CANNOT SHOW, stated rather than
                # implied. Ranks 0 and 2 do not satisfy rank 1's predicate,
                # but they are NOT healthy either: their stand-in tp device
                # pool cannot be shape-matched, so they take the `#847` soft
                # refusal and vote row 0 = 0 with no reason of their own. So
                # the only claim available here is that no rank prints a
                # PEER'S string -- an int64 MIN carries none -- and that a
                # rank refusing without a recorded reason says exactly that
                # rather than claiming to be healthy. The healthy-peer half
                # ("peer refused") is driven where the fixture supports it,
                # in `test_the_shortfall_stop_carries_the_reason_on_the_raise`.
                for rank in (0, 2):
                    self.assertEqual(cases[rank][1]["owned_equals_driven"], 0)
                    self.assertIsNone(cases[rank][1]["host_pool_build_msg"])
                    self.assertNotIn(named, str(outcomes[rank][0]))
                    self.assertIn(
                        pdv.BOOT_REFUSED_NO_REASON_LINE, str(outcomes[rank][0])
                    )
                # A ROUTING, not a downgrade to the soft-refusal shape: no
                # rank returns a pin mapping.
                for rank in range(3):
                    self.assertNotIn("tp", cases[rank][1])

    def test_the_not_applicable_exits_vote_the_neutral_and_the_boot_proceeds(self):
        """T-38d. ``:2324`` (rebind sub-flag off) and ``:2338`` (no host tier)
        are supported configurations; routing them to the refusing value would
        be a guard that cannot NOT fire."""
        for arm, mutate in (
            (
                "flag-off",
                lambda s: setattr(s.server_args, "phase_flip_rebind_hicache", False),
            ),
            (
                "no-host-tier",
                lambda s: setattr(s.tree_cache, "token_to_kv_pool_host", None),
            ),
        ):
            with self.subTest(arm=arm):
                cases = []
                for rank in range(3):
                    sched = self._builder_sched()
                    sched.ps = types.SimpleNamespace(pp_rank=rank, tp_rank=rank)
                    mutate(sched)
                    result = pfb.build_phase_flip_host_pools(sched)
                    # NO DEFAULT anywhere: a missing key must be a KeyError,
                    # never a healthy vote.
                    self.assertEqual(result["host_pool_build_ok"], 1)
                    self.assertEqual(result["owned_equals_driven"], 1)
                    self.assertEqual(result["layer_mapping_non_empty"], 1)
                    cases.append((sched, result))
                fabric = _BootFabric(cases)
                outcomes = fabric.run()
                self.assertEqual(len(fabric.voted), 3)
                for rank, (raised, _lines) in enumerate(outcomes):
                    self.assertIsNone(raised, f"rank {rank} raised: {raised}")

    def test_a_missing_row_12_key_is_a_key_error_on_every_rank(self):
        """Mutant 21's expected symptom, asserted: the SAME raise on all
        three ranks and none hanging -- which is what makes the head-assembly
        convention checkable and a ``.get(..., 1)`` default rejectable."""
        cases = []
        for rank in range(3):
            sched = self._builder_sched()
            sched.ps = types.SimpleNamespace(pp_rank=rank, tp_rank=rank)
            result = pfb.build_phase_flip_host_pools(sched)
            del result["host_pool_build_ok"]
            cases.append((sched, result))
        for rank, (sched, result) in enumerate(cases):
            with self.assertRaises(KeyError):
                pfb.phase_flip_boot_terms(sched, result, _LOGGER, rank)


class TestThePostBuildChecksStopTheGroup(CustomTestCase):
    """T-38e (row 12)."""

    def _cases(self, failing_msg):
        cases = []
        for rank in range(3):
            result = _fresh_result()
            if rank == 1:
                pfb._record_build_failure(result, failing_msg)
            cases.append((_sched(rank=rank), result))
        return cases

    def test_the_three_post_build_checks_stop_the_group_instead_of_one_rank(self):
        for arm, msg in (
            ("c", "#1068 TP PIN ROW MISMATCH rank=1 tp_rows=1 pp_rows=2"),
            ("d", "#1068 TP PIN ANCHOR SLOT MISMATCH rank=1 tp_slots=129"),
            ("e", "--hicache-mamba-host-mib too small: 10 slots (rank=1)"),
        ):
            with self.subTest(arm=arm):
                fabric = _BootFabric(self._cases(msg))
                outcomes = fabric.run()
                self.assertEqual(len(fabric.voted), 3, "none hangs")
                for rank, (raised, _lines) in enumerate(outcomes):
                    self.assertIsInstance(
                        raised, pdv.PhaseDomainDivergence, f"rank {rank}"
                    )
                    self.assertIn("host_pool_build_ok", str(raised))
                # The rank that RECORDED the failure carries its own string
                # verbatim; an int64 MIN carries no string, so the two healthy
                # ranks say so and point at its log instead of printing it.
                self.assertIn(msg, str(outcomes[1][0]))
                for rank in (0, 2):
                    self.assertIn(
                        pdv.HOST_POOL_BUILD_OK_PEER_LINE, str(outcomes[rank][0])
                    )
                    self.assertNotIn(msg, str(outcomes[rank][0]))

    def test_arm_f_no_failing_check_is_silent(self):
        """CAN-NOT-FIRE PIN (excluded from the red-first total)."""
        cases = [(_sched(rank=r), _fresh_result()) for r in range(3)]
        outcomes = _BootFabric(cases).run()
        for raised, lines in outcomes:
            self.assertIsNone(raised)
            self.assertEqual([x for x in lines if "host_pool_build" in x], [])

    def test_the_first_failing_check_wins(self):
        result = _fresh_result()
        pfb._record_build_failure(result, "first")
        pfb._record_build_failure(result, "second")
        self.assertEqual(result["host_pool_build_msg"], "first")
        self.assertEqual(result["host_pool_build_ok"], 0)


class TestThePackedBusTermsStopTheGroup(CustomTestCase):
    """T-37 (slot 10) and T-48 arm (3) (slot 15), through S0's pack/unpack."""

    def _reduce(self, per_rank_terms):
        payloads = [pdv.pack_phase_domain_payload(t) for t in per_rank_terms]
        return [min(col) for col in zip(*payloads)]

    def _stop_on_every_rank(self, term_name, bad_rank=1):
        per_rank = []
        for rank in range(3):
            terms = {term_name: 0 if rank == bad_rank else 1}
            per_rank.append(terms)
        reduced = self._reduce(per_rank)
        raised = []
        for rank in range(3):
            with self.assertRaises(pdv.PhaseDomainDivergence) as cm:
                pdv.unpack_phase_domain(
                    reduced,
                    rank=rank,
                    local=per_rank[rank],
                    world_size=3,
                    phase="tp",
                )
            raised.append(str(cm.exception))
        return raised

    def test_draft_tier_mismatch_on_one_rank_stops_the_group(self):
        for line in self._stop_on_every_rank("draft_tier_domain_matches"):
            self.assertIn("draft_tier_domain_matches", line)

    def test_an_and_slot_stop_does_not_claim_the_ranks_disagreed(self):
        """INSTRUMENT-TEXT, class A: the line said what the code does not do.

        An AND slot at 0 and a MAX pair above 0 say *at least one rank
        refused*; a MIN carries no count, so neither can tell one refusing
        rank from all of them. Rendering that as *the ranks do not agree*
        sends the reader hunting for a divergence -- measured on the metal,
        `local=0 group_min=0 group_max=0 per_rank=[1,1,1]` under the
        divergence sentence."""
        for line in self._stop_on_every_rank("rebind_domain_within_driven"):
            self.assertIn("at least one rank refused", line)
            self.assertNotIn("the ranks do not agree", line)

    def test_a_divergence_pair_still_says_the_ranks_disagreed(self):
        """The must-not-fire partner: `min != max` really IS a disagreement,
        so the sweep above must not have flattened both sentences into one."""
        payloads = [
            pdv.pack_phase_domain_payload({"d_domain": v}) for v in (32, 64, 64)
        ]
        reduced = [min(col) for col in zip(*payloads)]
        with self.assertRaises(pdv.PhaseDomainDivergence) as cm:
            pdv.unpack_phase_domain(reduced, rank=0, local={}, world_size=3)
        self.assertIn("the ranks do not agree", str(cm.exception))

    def test_rebind_domain_term_fires_when_the_counter_did_not_follow(self):
        for line in self._stop_on_every_rank("rebind_domain_within_driven"):
            self.assertIn("rebind_domain_within_driven", line)

    def test_each_rank_names_its_own_local_value(self):
        per_rank = [{"draft_tier_domain_matches": 1 if r != 1 else 0} for r in range(3)]
        reduced = self._reduce(per_rank)
        with self.assertRaises(pdv.PhaseDomainDivergence) as cm:
            pdv.unpack_phase_domain(
                reduced, rank=1, local=per_rank[1], world_size=3, phase="tp"
            )
        self.assertIn("local=0", str(cm.exception))


class TestTheVoteSiteItself(CustomTestCase):
    """The reduce must be reached by EVERY rank whichever exit it took.

    HAZARD these close: a reduce sited inside `build_phase_flip_host_pools`,
    or guarded by anything a rank can fail, leaves the skipping rank running
    while its peers block in a collective nobody joins -- a boot that hangs
    with no message, the one failure mode that produces no evidence.
    """

    def test_the_builder_has_no_abrupt_exit_left(self):
        import ast
        import inspect
        import textwrap

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(pfb.build_phase_flip_host_pools))
        )
        fn = tree.body[0]
        raises = []
        for node in ast.walk(fn):
            if isinstance(node, ast.FunctionDef) and node is not fn:
                continue
            if isinstance(node, ast.Raise):
                raises.append(node.lineno)
        # The `try:` body may still re-raise into its own `except`; what must
        # not survive is a raise that ESCAPES the function, i.e. one outside
        # every try. Those are the ten exits D-65 enumerates.
        escaping = []
        tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
        covered = set()
        for t in tries:
            for node in ast.walk(t):
                if isinstance(node, ast.Raise):
                    covered.add(node.lineno)
        escaping = [ln for ln in raises if ln not in covered]
        self.assertEqual(escaping, [], "every escaping raise is routed into the vote")

    def test_the_vote_is_not_called_from_inside_the_builder(self):
        import inspect

        src = inspect.getsource(pfb.build_phase_flip_host_pools)
        self.assertNotIn("vote_phase_flip_boot_verdict", src)

    def test_the_scheduler_votes_under_the_two_flags_and_nothing_else(self):
        import ast
        import inspect
        import textwrap

        from sglang.srt.managers.scheduler import Scheduler

        tree = ast.parse(textwrap.dedent(inspect.getsource(Scheduler.__init__)))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "vote_phase_flip_boot_verdict"
        ]
        self.assertEqual(len(calls), 1)
        target = calls[0]

        conditions = []

        def walk(node, guards):
            if isinstance(node, ast.If):
                for child in node.body:
                    walk(child, guards + [ast.unparse(node.test)])
                for child in node.orelse:
                    walk(child, guards + ["not (%s)" % ast.unparse(node.test)])
                return
            for sub in ast.walk(node):
                if sub is target:
                    conditions.append(list(guards))
                    return
            return

        for stmt in tree.body[0].body:
            walk(stmt, [])
        self.assertEqual(len(conditions), 1)
        self.assertEqual(
            conditions[0],
            [
                "self.server_args.enable_phase_flip",
                "self.enable_hierarchical_cache",
            ],
            "flip AND hicache, and nothing a rank can answer differently",
        )


class TestThePackedPayloadCarriesTheTerms(CustomTestCase):
    """The FORGOTTEN VOTE, on both S1 slots of the packed bus.

    A value computed at the payload-build site and left out of the payload is
    a detection with no vote -- a deleted STOP wearing a log line -- and it is
    invisible to any test that packs the term by hand.
    """

    def _sched_with_controller(self, controller, group):
        return types.SimpleNamespace(
            ps=types.SimpleNamespace(tp_rank=0),
            tree_cache=types.SimpleNamespace(
                cache_controller=controller, components=None
            ),
            req_to_token_pool=None,
        )

    def test_slot_10_is_neutral_on_a_speculative_config(self):
        """THE BOOT-KILLER, pinned in the direction it fired.

        A 1-layer MTP draft tier under a 64-layer target domain is the
        ordinary shape of every speculative config on this rig. Slot 10 is an
        AND slot, so a 0 here is a deterministic group STOP at the FIRST
        cutover -- measured on the metal at
        ``boot_855_weg1b12s1_6917f46dd5_0906_115410.log`` (six
        ``DRAFT TIER DOMAIN MISMATCH`` lines, six ``PhaseDomainDivergence``).
        The term compared the DRAFT tier's layer count against the TARGET
        tier's key-space domain, which is false by construction whenever the
        two tiers legitimately differ in width, i.e. always under MTP.
        """
        for draft_layers in (1, 18, 64):
            with self.subTest(draft_layers=draft_layers):
                controller = types.SimpleNamespace(
                    has_draft=True,
                    mem_pool_host_draft=types.SimpleNamespace(
                        layer_num=draft_layers
                    ),
                    mem_pool_host=types.SimpleNamespace(transfer_layer_domain=64),
                    layer_done_counter=None,
                )
                sched = self._sched_with_controller(controller, None)
                payload = pdv.build_phase_domain_payload(sched)
                self.assertEqual(
                    pdv.slot_of(payload, "draft_tier_domain_matches"), 1
                )

    def test_slot_10_has_no_producer_left_in_the_payload_builder(self):
        """The DELETION itself, not only its effect. A term left computed but
        pinned to 1 would read identically above while still carrying the
        wrong comparison for a later reader to re-enable."""
        terms = pdv.read_phase_domain_terms(
            self._sched_with_controller(
                types.SimpleNamespace(
                    has_draft=True,
                    mem_pool_host_draft=types.SimpleNamespace(layer_num=1),
                    mem_pool_host=types.SimpleNamespace(transfer_layer_domain=64),
                    layer_done_counter=None,
                ),
                None,
            )
        )
        self.assertNotIn("draft_tier_domain_matches", terms)

    def test_slot_15_reaches_the_payload(self):
        full, mamba = _stage(0, 64)
        group = _group(full, mamba)
        controller = types.SimpleNamespace(
            has_draft=False,
            mem_pool_host_draft=None,
            mem_pool_host=group,
            layer_done_counter=types.SimpleNamespace(num_layers=50),
        )
        sched = self._sched_with_controller(controller, group)
        payload = pdv.build_phase_domain_payload(sched)
        self.assertEqual(pdv.slot_of(payload, "rebind_domain_within_driven"), 0)

    def test_both_slots_read_the_min_neutral_before_their_events(self):
        """CAN-NOT-FIRE PIN. An un-fired producer must vote 1, or every rank
        STOPs at the first TP-phase reduce after this batch."""
        controller = types.SimpleNamespace(
            has_draft=False,
            mem_pool_host_draft=None,
            mem_pool_host=None,
            layer_done_counter=None,
        )
        sched = self._sched_with_controller(controller, None)
        payload = pdv.build_phase_domain_payload(sched)
        self.assertEqual(pdv.slot_of(payload, "draft_tier_domain_matches"), 1)
        self.assertEqual(pdv.slot_of(payload, "rebind_domain_within_driven"), 1)


class TestTheRankLabelIsReadWhereTheSchedulerKeepsIt(CustomTestCase):
    """Every #1206 boot line must name the rank it was emitted on.

    THE HAZARD. ``Scheduler`` has no ``pp_rank`` and no ``tp_rank``: it keeps
    its parallel identity on ``ParallelState``, ``self.ps``
    (``scheduler.py:860`` ``        self.ps = ParallelState(``). The tree has
    paid for that fact twice already and says so in its own comments --
    ``scheduler.py:1678-1680`` (*"The first version read `self.tp_rank`, which
    the Scheduler does not have"*) and ``:8185-8186`` (*"The first cut of this
    used `self.tp_rank`, which does not exist"*) -- and
    ``test_census_attribute_surface_583.py`` pins it against the real class.

    A boot reader taking the bare names answers 0 on EVERY rank. Nothing
    crashes: the attach lines, the SHORTFALL and EMPTY-LAYER-MAPPING lines and
    the boot STOP all render, all labelled ``rank=0``. So a correct 3-rank
    boot emits six attach lines that split 4/1/1 by their own label, a grader
    reading the per-rank count fails a passing boot, and no line on the boot
    can say which rank refused. A wrong ANSWER, not a traceback, which is why
    it needs a pin.
    """

    def test_the_scheduler_has_no_rank_of_its_own(self):
        """THE FALSIFIER, in the #583 form: if this ever becomes false the
        reader below is over-built; while it is true, a bare read is a bug."""
        from sglang.srt.managers.scheduler import Scheduler

        self.assertFalse(hasattr(Scheduler, "pp_rank"))
        self.assertFalse(hasattr(Scheduler, "tp_rank"))

    def test_the_parallel_state_is_where_the_identity_lives(self):
        from sglang.srt.distributed.parallel_state_wrapper import ParallelState

        for field in ("pp_rank", "tp_rank"):
            with self.subTest(field=field):
                self.assertIn(field, ParallelState.__annotations__)

    def test_the_boot_rank_comes_off_the_parallel_state(self):
        for rank in range(3):
            with self.subTest(rank=rank):
                sched = types.SimpleNamespace(
                    ps=types.SimpleNamespace(pp_rank=rank, tp_rank=0)
                )
                self.assertEqual(pfb._boot_rank(sched), rank)

    def test_pp_rank_is_the_name_that_separates_the_three_ranks(self):
        """The boot topology is ``pp_size=3, tp_size=1``, so ``tp_rank`` is 0
        on all three and only ``pp_rank`` distinguishes them. A reader that
        preferred ``tp_rank`` would relabel every line 0 again."""
        sched = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=2, tp_rank=0))
        self.assertEqual(pfb._boot_rank(sched), 2)

    def test_a_bare_attribute_is_not_a_second_home_for_the_rank(self):
        """DANGER DIRECTION. A fallback to the bare name keeps every stand-in
        green while the metal reads 0 -- one fact with two homes and the wrong
        one live on the boot."""
        self.assertEqual(pfb._boot_rank(types.SimpleNamespace(pp_rank=2)), 0)
        self.assertEqual(pfb._boot_rank(types.SimpleNamespace(tp_rank=2)), 0)

    def test_a_scheduler_without_a_parallel_state_reads_zero(self):
        """An instrument may never be the thing that breaks a boot."""
        self.assertEqual(pfb._boot_rank(types.SimpleNamespace()), 0)
        self.assertEqual(pfb._boot_rank(types.SimpleNamespace(ps=None)), 0)

    def test_each_ranks_pin_attach_line_names_that_rank(self):
        """§2.2 grades the attach rows PER RANK. Three ranks, three distinct
        labels -- the count is only gradeable if the labels differ."""
        pin_full, pin_mamba = _stage(0, 64)
        cases = []
        for rank in range(3):
            pin = _group(pin_full, pin_mamba)
            result = _result_from_groups(rank, [_stack("tp", pin, pin_full, pin_mamba)])
            result["tp"] = pin
            cases.append((_sched(rank=rank), result))
        outcomes = _BootFabric(cases).run()

        labels = []
        for rank, (raised, lines) in enumerate(outcomes):
            self.assertIsNone(raised, f"rank {rank} raised: {raised}")
            attach = [x for x in lines if "#1206 TRANSFER DOMAIN attach" in x]
            self.assertEqual(len(attach), 1)
            self.assertIn(f"rank={rank}", attach[0])
            labels.append(attach[0])
        self.assertEqual(len(set(labels)), 3, "three ranks, three labels")

    def test_the_boot_stop_names_the_rank_it_is_raised_on(self):
        """One rank's tier maps fewer layers than its device pool owns; all
        three STOP, and each STOP carries ITS OWN rank rather than 0."""
        cases = []
        for rank, (start, end) in enumerate([(0, 32), (32, 50), (50, 64)]):
            full, mamba = _stage(start, end)
            drop = (33,) if rank == 1 else ()
            group = _group(
                [k for k in full if k not in drop],
                [k for k in mamba if k not in drop],
                kv_owned=full,
                mamba_owned=mamba,
            )
            cases.append(
                (
                    _sched(rank=rank),
                    _result_from_groups(rank, [_stack("pp", group, full, mamba)]),
                )
            )
        outcomes = _BootFabric(cases).run()

        for rank, (raised, _lines) in enumerate(outcomes):
            self.assertIsInstance(raised, pdv.PhaseDomainDivergence)
            self.assertIn(f"rank={rank}", str(raised))


if __name__ == "__main__":
    unittest.main()
