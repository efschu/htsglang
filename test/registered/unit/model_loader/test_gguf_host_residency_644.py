# SPDX-License-Identifier: Apache-2.0
"""GGUF MoE experts must not stay in host RAM after the parameter is filled (#644).

``FusedMoE.materialize_gguf_weights`` turns the loader's per-expert GGUF
tensors into one ``[E, ...]`` parameter. The loader put every expert in TWO
holders on the way in (``fused_moe_triton/layer.py:1323-1324``)::

    param.expert_data_map[key] = loaded_weight
    param.data_container.append(loaded_weight)

Every branch of the materializer that ends the parameter's life as an
``UninitializedParameter`` releases both holders again -- the streaming drain
at ``layer.py:1649-1650`` and the load-time offload branch at
``layer.py:2772``/``2780`` -- except the DEFAULT, fully-resident branch, which
before #644 read::

    if plan is None:
        stacked = torch.stack([get(i) for i in range(count)], dim=0)
        param.materialize(stacked.shape, dtype=stacked.dtype)
        param.data.copy_(stacked)
        continue

Two costs, both host-side and both invisible to the page-cache half of the
loader (``ConsumedPageDropper``, which can only madvise mmapped checkpoint
pages -- these are anonymous copies):

1. ``torch.stack`` builds a SECOND full copy of the expert set before the
   copy into the parameter, so the host peak is twice the expert bytes.
2. ``continue`` leaves both holders populated for the lifetime of the
   process. The expert set stays resident in host anon memory even though it
   is already resident in ``param.data``, on the card. That is the
   double-residency the box was OOM-killed by, twice.

``HazardTest`` states the mechanism on its own terms -- two holders, one
storage, a copy that does not release -- so the hazard is demonstrated rather
than merely asserted, and stays demonstrated whatever the fix does.
``GuardTest`` then drives the REAL ``materialize_gguf_weights`` default path
and requires that it release. Before the fix, ``GuardTest`` fails.

No CUDA, no checkpoint: the expert payloads are small synthetic byte tensors
and the "device" the parameter materializes onto is the CPU, which is enough
because the defect is that the SOURCE references outlive the copy -- the
destination's location is irrelevant to it.
"""

import gc
import unittest
import weakref

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# --------------------------------------------------------------------------
# Synthetic layer
# --------------------------------------------------------------------------

# The object-level tests only need enough experts to tell a per-expert copy
# from a whole-set stack; 4 x 256 x 2048 B == 2 MiB of payload is plenty.
NUM_EXPERTS = 4
ROWS = 128  # per expert and per shard; w13 carries a w1 and a w3 half
ROW_BYTES = 2048

# The corroborating RSS reading needs blocks comfortably above glibc's initial
# mmap threshold (128 KiB), so that freeing an expert really does unmap its
# pages instead of parking them on the heap. 2 MiB per expert tensor, 16 MiB
# of payload in total.
RSS_ROWS = 1024


class _FakeGGUFMoEMethod:
    """Stands in for the CUDA ``GGUFMoEMethod`` by class name only."""


_FakeGGUFMoEMethod.__name__ = "GGUFMoEMethod"


def _payload(seed: int, rows: int) -> torch.Tensor:
    """One expert's quantized bytes. Content only has to be recognizable."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (rows, ROW_BYTES), dtype=torch.uint8, generator=g)


def _stub_gguf_moe_layer(keep_sources: bool, rows: int = ROWS):
    """A minimal loaded-GGUF ``FusedMoE`` stand-in on the default path.

    Borrows the real methods under test rather than reimplementing them, and
    carries only the attributes they read. ``_expert_offload_fraction = 1.0``
    is what puts it on the default branch: ``_gguf_moe_offload_eligible``
    returns False for a fraction >= 1.0 (``layer.py:2534-2536``), so ``plan``
    is None and the fully-resident path runs -- the same way a server without
    ``SGLANG_MOE_RESIDENT_EXPERT_FRACTION`` reaches it.

    Returns ``(layer, source_refs, payload_bytes)``. ``source_refs`` are weak
    references to every tensor the loader handed the parameters; when
    ``keep_sources`` is False nothing else in the test holds them, so a live
    referent afterwards means the parameter's own holders are keeping the
    expert set in host memory.
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.gguf import GGUFUninitializedParameter

    class _StubGGUFMoELayer(torch.nn.Module):
        materialize_gguf_weights = FusedMoE.materialize_gguf_weights
        # ``FusedMoE._gguf_expert_source`` unwraps to a plain function on
        # attribute access; re-wrap so the stub keeps it a staticmethod.
        _gguf_expert_source = staticmethod(FusedMoE._gguf_expert_source)
        _gguf_moe_offload_eligible = FusedMoE._gguf_moe_offload_eligible
        _gguf_moe_offload_eligible_uncached = (
            FusedMoE._gguf_moe_offload_eligible_uncached
        )
        _drain_gguf_stream_stagers = FusedMoE._drain_gguf_stream_stagers

    layer = _StubGGUFMoELayer()
    layer.layer_id = 3
    layer.quant_method = _FakeGGUFMoEMethod()
    layer.num_experts = NUM_EXPERTS
    layer.num_local_experts = NUM_EXPERTS
    layer._gguf_expert_shard = False
    layer._gguf_expert_range = (0, NUM_EXPERTS)
    layer._expert_offload_fraction = 1.0

    source_refs = []
    payload_bytes = 0
    kept = {"w13_qweight": {}, "w2_qweight": {}}

    half = rows // 2
    for attr in ("w13_qweight", "w2_qweight"):
        param = GGUFUninitializedParameter(requires_grad=False)
        param.is_gguf_weight = True
        param.tensor_shape = (NUM_EXPERTS, rows, ROW_BYTES)
        param.data_container = []
        param.expert_data_map = {}
        for e in range(NUM_EXPERTS):
            full = _payload(1000 * (1 if attr.startswith("w13") else 2) + e, rows)
            if keep_sources:
                kept[attr][e] = full
            if attr == "w13_qweight":
                # The loader delivers gate and up as separate shards; the
                # materializer concatenates them per expert.
                shards = [("w1", full[:half].clone()), ("w3", full[half:].clone())]
            else:
                shards = [("w2", full)]
            for shard_id, tensor in shards:
                param.expert_data_map[(e, shard_id)] = tensor
                param.data_container.append(tensor)
                source_refs.append(weakref.ref(tensor))
                payload_bytes += tensor.numel() * tensor.element_size()
        layer.register_parameter(attr, param)

    layer._test_sources = kept if keep_sources else None
    return layer, source_refs, payload_bytes


def _live(source_refs):
    """The still-alive referents among ``source_refs``, after a collection."""
    gc.collect()
    return [r() for r in source_refs if r() is not None]


def _rss_bytes() -> int:
    """Resident set size from ``/proc/self/statm``.

    Read directly rather than through ``psutil``: under this LXC/lxcfs box
    ``psutil.virtual_memory()`` reports the host's numbers, not the
    container's. ``statm`` is per-process and unaffected.
    """
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096


# --------------------------------------------------------------------------
# The hazard, on its own terms
# --------------------------------------------------------------------------


class HazardTest(CustomTestCase):
    """Why holding the containers costs real bytes.

    None of this exercises the materializer; it models the two mechanisms the
    default branch used to combine, so the cost is demonstrated independently
    of the code that pays it. Both tests stay green after the fix.
    """

    def test_a_copy_does_not_release_the_source(self):
        """Copying into a destination leaves the source fully alive.

        This is the whole of the double residency: after ``dst.copy_(src)``
        the bytes exist twice, and the second copy only goes away when the
        last reference to ``src`` does. A holder that outlives the copy --
        ``param.data_container`` / ``param.expert_data_map`` -- makes "twice"
        permanent.
        """
        src = torch.ones(1024, 1024, dtype=torch.uint8)
        holder_a = [src]  # stands in for param.data_container
        holder_b = {("e0", "w2"): src}  # stands in for param.expert_data_map
        ref = weakref.ref(src)

        dst = torch.empty_like(src)
        dst.copy_(src)
        del src

        self.assertIsNotNone(ref(), "the copy alone must not free the source")

        # Releasing ONE holder frees nothing: the other still refers to the
        # same storage. This is why the fix has to clear both, and why
        # ``_gguf_expert_source.drop`` pops from both.
        holder_a.clear()
        self.assertIsNotNone(ref(), "one holder released is not enough")

        holder_b.clear()
        gc.collect()
        self.assertIsNone(ref(), "with both holders released the bytes go")
        self.assertTrue(bool(dst.all()), "the destination still has the data")

    def test_stacking_allocates_a_second_full_copy(self):
        """``torch.stack`` is an allocation the size of the whole expert set.

        It cannot alias its inputs -- the result is one contiguous storage --
        so on the default branch the host briefly held the loaded experts AND
        their stacked duplicate, before the parameter (a third copy) was even
        written.
        """
        experts = [torch.ones(64, 4096, dtype=torch.uint8) for _ in range(4)]
        expert_bytes = sum(t.numel() * t.element_size() for t in experts)

        stacked = torch.stack(experts, dim=0)

        self.assertEqual(stacked.numel() * stacked.element_size(), expert_bytes)
        source_ptrs = {t.untyped_storage().data_ptr() for t in experts}
        self.assertNotIn(
            stacked.untyped_storage().data_ptr(),
            source_ptrs,
            "the stack is a distinct storage, i.e. a second full copy",
        )


# --------------------------------------------------------------------------
# The guard: the real default path must release
# --------------------------------------------------------------------------


class GuardTest(CustomTestCase):
    """``materialize_gguf_weights``' default branch, driven for real."""

    def test_default_path_produces_the_same_bytes(self):
        """Behaviour first: the parameter is what it always was.

        Pinned here so the #644 change cannot be mistaken for a byte-level
        change. Expected to pass both before and after the fix.
        """
        layer, _, _ = _stub_gguf_moe_layer(keep_sources=True)
        layer.materialize_gguf_weights()

        self.assertEqual(tuple(layer.w13_qweight.shape), (NUM_EXPERTS, ROWS, ROW_BYTES))
        self.assertEqual(tuple(layer.w2_qweight.shape), (NUM_EXPERTS, ROWS, ROW_BYTES))
        for attr in ("w13_qweight", "w2_qweight"):
            for e in range(NUM_EXPERTS):
                self.assertTrue(
                    torch.equal(
                        getattr(layer, attr).data[e], layer._test_sources[attr][e]
                    ),
                    f"{attr} expert {e} changed",
                )

    def test_default_path_releases_both_holders(self):
        """#644: the holders are dead once the parameter is materialized.

        The object-level proof, and the deterministic one -- it does not
        depend on the allocator returning anything to the OS.
        """
        layer, source_refs, _ = _stub_gguf_moe_layer(keep_sources=False)
        layer.materialize_gguf_weights()

        for attr in ("w13_qweight", "w2_qweight"):
            param = getattr(layer, attr)
            self.assertEqual(
                list(getattr(param, "data_container", [])),
                [],
                f"{attr}.data_container still holds the loaded experts",
            )
            self.assertEqual(
                dict(getattr(param, "expert_data_map", {})),
                {},
                f"{attr}.expert_data_map still holds the loaded experts",
            )

        still_alive = _live(source_refs)
        retained = sum(t.numel() * t.element_size() for t in still_alive)
        self.assertEqual(
            still_alive,
            [],
            f"{len(still_alive)} loaded expert tensors ({retained} bytes) are "
            "still resident in host memory after the parameter was filled",
        )

    def test_default_path_does_not_build_a_second_full_stack(self):
        """#644: no whole-expert-set duplicate on the way into the parameter.

        The peak half of the defect. ``torch.stack`` over the expert list is
        the only place the default branch allocated expert-set-sized host
        memory, so its absence is the check. Per-expert allocations (the w13
        gate/up ``torch.cat``) are unchanged and not affected by this.
        """
        layer, _, _ = _stub_gguf_moe_layer(keep_sources=True)

        real_stack = torch.stack
        big_stacks = []

        def spy(tensors, *args, **kwargs):
            seq = list(tensors)
            if len(seq) >= NUM_EXPERTS:
                big_stacks.append(len(seq))
            return real_stack(seq, *args, **kwargs)

        torch.stack = spy
        try:
            layer.materialize_gguf_weights()
        finally:
            torch.stack = real_stack

        self.assertEqual(
            big_stacks,
            [],
            "the default path still stacks the whole expert set into a "
            "second full host copy before writing the parameter",
        )

    def test_default_path_returns_the_host_bytes(self):
        """Corroboration only: RSS actually falls back.

        Secondary to ``test_default_path_releases_both_holders`` -- the
        allocator is free to keep freed pages, so the threshold is
        deliberately generous (half the payload). The payload blocks are
        sized above glibc's initial mmap threshold, and every free happens
        after every allocation, so in practice the pages are unmapped.
        """
        gc.collect()
        before = _rss_bytes()

        layer, _, payload_bytes = _stub_gguf_moe_layer(
            keep_sources=False, rows=RSS_ROWS
        )
        layer.materialize_gguf_weights()
        gc.collect()

        after = _rss_bytes()
        # The parameter itself is legitimately resident; anything on top of it
        # is the loaded copy that should have been released.
        param_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
        overhead = after - before - param_bytes
        self.assertLess(
            overhead,
            payload_bytes // 2,
            f"{overhead} bytes of host memory beyond the {param_bytes}-byte "
            f"parameter survived materialization of a {payload_bytes}-byte "
            "expert set",
        )


if __name__ == "__main__":
    unittest.main()
