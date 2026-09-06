"""#1068 weg1 S0, operator ruling R-B1-4: the two D-68 attributes the bus reads
WITHOUT a default must be DECLARED in the tree S0 boots on, not in S1's.

WHAT THIS EXISTS FOR. `phase_domain_verdict.py:624/:630` read
`host_ring_discard_ok` and `d_backup_width` by NAME off the bound
`HostPoolGroup` with NO default -- deliberately, because a `getattr` default on
a ledger path (#606) would turn a missing declaration into a silently healthy
vote on the one term that exists to STOP the group. That policy is right and is
NOT what this file changes. What it pins is the policy's PRECONDITION, which
until now was carried by nothing: the declaration lived on S1's branch alone
(`memory_pool_host.py`, S1-C18), so S0 on any tree without S1 died on the first
TP-phase pass after the first cutover with
`AttributeError: 'HostPoolGroup' object has no attribute 'host_ring_discard_ok'`
-- all three ranks, deterministic, 25 s after READY (boot weg1b12s0,
9ad00b1b1b, 12:30:01Z).

WHY A TEST AND NOT AN ARGUMENT. The S0 suite was green through that boot
because every group it drove was a stand-in that declares the attributes
itself. The one arm that did drive the tree's own class asserted the ABSENCE
and the AttributeError -- i.e. the suite pinned the defect as the expected
state. This file drives the REAL class, so the argument about who declares the
neutral can fail here instead of on metal.

MERGE NOTE, and the reason `_pool_entry` reads the dataclass instead of naming
one field: S1 renames `PoolEntry.layer_mapper` (a closure) to `layer_mapping`
(a dict) in the SAME batch as this fix. This file asserts nothing about that
field; constructing the entry from whatever the tree's dataclass declares keeps
the two reads under test identical before and after that merge.

Hermetic: no CUDA, no torch.distributed, no scheduler process. Run with
CUDA_VISIBLE_DEVICES="".
"""

import dataclasses
import unittest

from sglang.srt.managers import phase_domain_verdict as pdv
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool_host import HostPoolGroup, PoolEntry
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2)


class _EntryHostPool:
    """The five attributes the real `HostPoolGroup.__init__` reads off an
    entry's host pool. Enough to build the tree's own group -- which is the
    whole point: a stand-in group that declares the attributes itself is what
    let the killer through."""

    layout = "layer_first"
    page_size = 1
    device = "cpu"
    size = 8
    can_use_write_back_jit = False


def _pool_entry():
    names = {f.name for f in dataclasses.fields(PoolEntry)}
    kwargs = {
        "name": PoolName.KV,
        "host_pool": _EntryHostPool(),
        "device_pool": None,
    }
    if "layer_mapping" in names:
        kwargs["layer_mapping"] = {0: 0}
    else:
        kwargs["layer_mapper"] = lambda layer_id: layer_id
    return PoolEntry(**kwargs)


def _real_group():
    return HostPoolGroup([_pool_entry()])


class _Counter:
    num_layers = 4


class _Controller:
    def __init__(self, group):
        self.mem_pool_host = group
        self.has_draft = False
        self.mem_pool_host_draft = None
        self.layer_done_counter = _Counter()


class _TreeCache:
    def __init__(self, controller):
        self.cache_controller = controller
        self.components = {}


class _PS:
    tp_rank = 0


class _Scheduler:
    """The smallest object `build_phase_domain_payload` reads, with the tree's
    own group bound where `cache_controller.mem_pool_host` binds it."""

    def __init__(self, group):
        self.tree_cache = _TreeCache(_Controller(group))
        self.ps = _PS()
        self.req_to_token_pool = None


class TheBusReadsAreDeclaredWhereS0Boots(unittest.TestCase):
    def test_the_class_declares_both_no_default_attributes(self):
        """The two names, read off the CLASS exactly as the bus reads them --
        one `getattr` with no default, through the module's own constants.

        Values, not just presence: the AND slot's MIN-neutral is 1 (a 0 here
        would STOP every group from the first pass), and the width term's
        neutral is 0 (its vote is the DELTA against the scheduler's bookmark,
        so a non-zero neutral votes a backup that never happened).
        """
        for attribute, neutral in (
            (pdv.HOST_RING_DISCARD_OK_ATTR, 1),
            (pdv.D_BACKUP_WIDTH_ATTR, 0),
        ):
            with self.subTest(attribute=attribute):
                self.assertEqual(getattr(HostPoolGroup, attribute), neutral)

    def test_a_real_group_answers_both_reads(self):
        """The tree's own instance, not a double. This is the expression at
        `phase_domain_verdict.py:624` and `:630`, executed."""
        group = _real_group()
        self.assertEqual(int(getattr(group, pdv.HOST_RING_DISCARD_OK_ATTR)), 1)
        self.assertEqual(int(getattr(group, pdv.D_BACKUP_WIDTH_ATTR) or 0), 0)

    def test_the_payload_builds_neutral_against_the_tree_s_own_group(self):
        """The KILLER'S OWN PATH: `build_phase_domain_payload` with the real
        group bound, which on 9ad00b1b1b raised `AttributeError` on all three
        ranks at the first TP pass after the first cutover.

        The two terms are read BY NAME, not by index, so this arm says nothing
        about the slot layout and survives a batch that renumbers it.
        """
        payload = pdv.build_phase_domain_payload(_Scheduler(_real_group()))
        self.assertEqual(pdv.slot_of(payload, "host_ring_discarded"), 1)
        self.assertEqual(pdv.pair_of(payload, "d_backup_width"), (0, 0))

    # THE POLICY THIS FIX DOES NOT SOFTEN is pinned where it already lives:
    # `test_phase_domain_verdict.py`'s T-46 arm 7 (b) drives a group-shaped
    # holder that WITHHOLDS each name and requires the AttributeError, naming
    # holder and attribute. That arm stays green through this change, and a
    # `getattr(group, name, 1)` at `:624` -- the #606 shape, rejected by name --
    # would turn it red. Nothing here re-pins it.


if __name__ == "__main__":
    unittest.main()
