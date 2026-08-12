"""#695 risk 2: the exact-size pin must be switchable without a code revert.

WHY THIS GATE EXISTS
--------------------
The #695 arena change replaced ``torch.zeros(..., pin_memory=True)`` with an
exact-size ``MAP_SHARED|MAP_ANONYMOUS`` mapping plus ``cudaHostRegister``,
which removed 13.65 GiB of power-of-two rounding and raised measured host
headroom under load by 13.8 GiB. MERGE-R5 §6 left ONE number open against it:
flip restore p50 2036 -> 2153 ms (+5.7 %), measured across two windows with
n=18 vs n=546 and different load profiles -- a confounded comparison that was
explicitly called neither a regression nor clean.

Settling that question requires running the two allocation paths through the
SAME harness. Building the revert arm as a second tree makes the two arms
differ by a build as well as by the allocator; this gate makes them differ by
one environment variable and nothing else, so both arms can run from one
md5-identical tree.

The gate therefore has two jobs, and the second outlives the measurement: if
the latency cost turns out to be real and material, the exact-size path can be
made opt-in by flipping this default rather than by unpicking a merged commit.

WHAT IS **NOT** GATED, DELIBERATELY
-----------------------------------
``_register_image_post`` runs on BOTH sides. The host-post registration and
the ``honest_host_memory_bytes`` pricing are separate defects from the same
commit -- the images were invisible to the only registry that sums host posts,
and cgroup-v2 shmem was priced as reclaimable page cache while nine OOM kills
happened. Those are correct under either allocator, and an opt-out that also
silenced the ledger would trade a diagnosis for a benchmark.

CAN-FAIL
--------
Delete the ``SGLANG_PHASE_FLIP_EXACT_PIN`` check from ``_alloc_host_image``
and ``test_opt_out_routes_to_the_torch_pinned_allocator``,
``test_opt_out_preserves_the_zero_false_empty_path`` and
``test_opt_out_still_registers_the_host_post`` all go red. Gate
``_register_image_post`` behind the same env and
``test_opt_out_still_registers_the_host_post`` goes red on its own. Invert the
default and ``test_default_is_the_exact_size_path`` goes red.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor import weights_arena
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Routing:
    """Records which allocation path a call took, without allocating pinned
    memory -- these tests must run on a CPU box with no CUDA context."""

    def __init__(self):
        self.exact = []
        self.torch_zeros = []
        self.torch_empty = []
        self.posts = []

    def install(self, case):
        def _exact(dims, dtype, device, pin_memory, allocator):
            self.exact.append(int(dims[0]))
            return torch.zeros(dims, dtype=dtype)

        def _zeros(total):
            self.torch_zeros.append(int(total))
            return torch.zeros(total, dtype=torch.uint8)

        def _empty(total):
            self.torch_empty.append(int(total))
            return torch.empty(total, dtype=torch.uint8)

        def _post(nbytes):
            self.posts.append(int(nbytes))

        for name, fn in (
            ("_alloc_with_host_register", _exact),
            ("_torch_pinned_zeros", _zeros),
            ("_torch_pinned_empty", _empty),
            ("_register_image_post", _post),
        ):
            original = getattr(weights_arena, name)
            setattr(weights_arena, name, fn)
            case.addCleanup(setattr, weights_arena, name, original)
        return self


class ExactPinOptOut(unittest.TestCase):
    def setUp(self):
        self.routing = _Routing().install(self)

    def test_default_is_the_exact_size_path(self):
        """Unset env keeps the shipped behaviour. The memory result is proven
        under load; the latency question is not a reason to change the
        default before it is answered."""
        envs.SGLANG_PHASE_FLIP_EXACT_PIN.clear()
        out = weights_arena._alloc_host_image(1234567, pin=True)
        self.assertEqual(self.routing.exact, [1234567])
        self.assertEqual(self.routing.torch_zeros, [])
        self.assertEqual(out.numel(), 1234567)

    def test_opt_out_routes_to_the_torch_pinned_allocator(self):
        """=0 must reproduce the PRE-#695 allocation exactly, because the arm
        it builds is only a comparand if it is the old code path."""
        with envs.SGLANG_PHASE_FLIP_EXACT_PIN.override(False):
            out = weights_arena._alloc_host_image(1234567, pin=True)
        self.assertEqual(self.routing.exact, [])
        self.assertEqual(self.routing.torch_zeros, [1234567])
        self.assertEqual(out.numel(), 1234567)

    def test_opt_out_preserves_the_zero_false_empty_path(self):
        """``arena_image`` passes zero=False and pre-#695 that was
        ``torch.empty(..., pin_memory=True)``, not zeros. Faulting in a 13 GiB
        image that is about to be overwritten is exactly the boot-time cost
        the zero=False flag exists to avoid, so the opt-out arm must not
        quietly acquire it."""
        with envs.SGLANG_PHASE_FLIP_EXACT_PIN.override(False):
            weights_arena._alloc_host_image(4096, pin=True, zero=False)
        self.assertEqual(self.routing.torch_empty, [4096])
        self.assertEqual(self.routing.torch_zeros, [])

    def test_opt_out_still_registers_the_host_post(self):
        """The ledger is not part of the arena hunk. Both arms must remain
        visible to the registry that sums pinned host posts."""
        with envs.SGLANG_PHASE_FLIP_EXACT_PIN.override(False):
            weights_arena._alloc_host_image(999, pin=True)
        self.assertEqual(self.routing.posts, [999])

    def test_opt_out_does_not_touch_the_unpinned_path(self):
        """pin=False is the CPU/unit path; neither arm may reach CUDA."""
        with envs.SGLANG_PHASE_FLIP_EXACT_PIN.override(False):
            out = weights_arena._alloc_host_image(4096, pin=False)
        self.assertEqual(self.routing.exact, [])
        self.assertEqual(self.routing.torch_zeros, [])
        self.assertEqual(self.routing.torch_empty, [])
        self.assertEqual(self.routing.posts, [])
        self.assertEqual(out.numel(), 4096)

    def test_the_env_is_read_per_call_not_at_import(self):
        """The A/B sets the variable in the boot environment, but a value
        frozen at import time would also survive a test monkeypatch and give a
        silently single-armed measurement. Read it per call."""
        with envs.SGLANG_PHASE_FLIP_EXACT_PIN.override(False):
            weights_arena._alloc_host_image(11, pin=True)
        with envs.SGLANG_PHASE_FLIP_EXACT_PIN.override(True):
            weights_arena._alloc_host_image(22, pin=True)
        self.assertEqual(self.routing.torch_zeros, [11])
        self.assertEqual(self.routing.exact, [22])


if __name__ == "__main__":
    unittest.main()
