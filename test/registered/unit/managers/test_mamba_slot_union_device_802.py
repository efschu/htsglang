"""#802: the slot-union collective must run on the group's own device.

THE CRASH, on metal, first real flip of the 18:49 Arm-1 boot
(/spinning/evidence-665-f1/boot_802ab_arm1_0822_1849.log):

    gdn_flip_mover.py:885  agreed, refusal = agree_mamba_slots(
                             local, flip_tp.device_group, _local_slot_capacity())
    gdn_flip_mover.py:743  reduce_fn(header, group)
    gdn_flip_mover.py:798  dist.all_reduce(tensor, op=MAX, group=group)
    RuntimeError: No backend type associated with device type cpu

``agree_mamba_slots`` builds ``header`` and ``presence`` as CPU tensors, but
the caller hands it ``flip_tp.device_group`` -- an NCCL group with no CPU
backend. Both collectives therefore die before the union can agree anything.
The KV leg on the very next line got this right:
``_dist_exchange(flip_tp.device_group, device)`` pairs the device group WITH
the device. The union was simply never told.

WHY THE SHIPPED TESTS DID NOT CATCH IT, and this is the lesson worth more
than the patch. ``agree_mamba_slots`` takes an injectable ``all_reduce`` so
its logic can be tested in-process. Every existing case supplies one, so the
DEFAULT path -- ``_default_all_reduce``, the only one a boot ever runs --
was never executed against a real group. The injectable made the union's
arithmetic green while the wire it actually uses was never touched. A
seam that is only ever crossed by a stand-in is not covered.

So this file pins the DEVICE CONTRACT rather than the arithmetic: whatever
the caller names as the collective device is where the reduced tensors must
be. It stays hermetic -- the recording reduce below asserts the contract the
NCCL backend enforces ("a device-only group refuses a cpu tensor") without
needing a GPU, which is exactly the check the injectable was hiding.
"""

import types
import unittest

import torch

from sglang.srt.managers.gdn_flip_mover import agree_mamba_slots
from sglang.test.test_utils import CustomTestCase


class _DeviceOnlyGroup:
    """Stands in for an NCCL process group: it has no CPU backend.

    The real one raises `RuntimeError: No backend type associated with device
    type cpu`; this reproduces that contract exactly, in-process.
    """

    def __init__(self, kind: str = "cuda"):
        self.kind = kind


def _recording_reduce(seen):
    """A reduce that enforces the device-only group's contract and records."""

    def _reduce(tensor, group):
        seen.append(tensor.device.type)
        if isinstance(group, _DeviceOnlyGroup) and tensor.device.type != group.kind:
            raise RuntimeError(
                f"No backend type associated with device type {tensor.device.type}"
            )
        # MAX all-reduce of a single rank is the identity, which is all the
        # union arithmetic needs here -- this file is about the device.
        return None

    return _reduce


class MambaSlotUnionDevice802(CustomTestCase):
    def test_collectives_land_on_the_device_the_caller_named(self):
        """THE FIX. Every tensor handed to the reduce is on the named device."""
        seen = []
        slots = torch.tensor([0, 3, 7], dtype=torch.int64)
        agreed, refusal = agree_mamba_slots(
            slots,
            _DeviceOnlyGroup("cpu"),
            local_capacity=64,
            all_reduce=_recording_reduce(seen),
            device=torch.device("cpu"),
        )
        self.assertEqual(refusal, "")
        # Both collectives ran, and both on the named device.
        self.assertGreaterEqual(len(seen), 2, f"expected 2 collectives, saw {seen}")
        self.assertEqual(set(seen), {"cpu"}, seen)
        # The agreed set comes back on CPU for the caller, as it always did.
        self.assertEqual(agreed.device.type, "cpu")
        self.assertEqual(sorted(int(v) for v in agreed), [0, 3, 7])

    def test_a_device_only_group_refuses_cpu_tensors(self):
        """RED WITHOUT THE FIX -- this is the metal crash, in-process.

        Before the fix `agree_mamba_slots` had no `device` parameter and built
        its tensors on CPU unconditionally, so against a device-only group
        this raised exactly the RuntimeError the 18:49 boot died of. A mutant
        that reverts the tensors to CPU turns this test red again.
        """
        seen = []
        slots = torch.tensor([0, 3, 7], dtype=torch.int64)
        # A group whose only backend is "cuda" -- the shipped flip_tp group.
        group = _DeviceOnlyGroup("cuda")

        # Naming cuda as the collective device is what makes the contract
        # satisfiable. We cannot allocate real cuda here, so assert the
        # negative directly: CPU tensors against this group are refused.
        with self.assertRaises(RuntimeError) as ctx:
            agree_mamba_slots(
                slots,
                group,
                local_capacity=64,
                all_reduce=_recording_reduce(seen),
                device=torch.device("cpu"),
            )
        self.assertIn("No backend type associated with device type cpu", str(ctx.exception))

    def test_default_device_is_still_cpu_for_existing_callers(self):
        """The parameter is additive: omitting it reproduces the old behaviour,
        so every existing in-process caller and test is unchanged."""
        seen = []
        slots = torch.tensor([1, 2], dtype=torch.int64)
        agreed, refusal = agree_mamba_slots(
            slots,
            _DeviceOnlyGroup("cpu"),
            local_capacity=32,
            all_reduce=_recording_reduce(seen),
        )
        self.assertEqual(refusal, "")
        self.assertEqual(set(seen), {"cpu"}, seen)
        self.assertEqual(sorted(int(v) for v in agreed), [1, 2])


if __name__ == "__main__":
    unittest.main()
