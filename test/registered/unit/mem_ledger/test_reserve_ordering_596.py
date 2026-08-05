"""#596: the reserve must be decided from the RESOLVED activation profile.

Window 8 ran the payout retry with a VRAM calibration that MATCHED the rig and
still OOMed, because the reserve was decided before the profile that keys the
footprint existed.

``chunked_prefill_size`` and ``cuda_graph_config.decode.max_bs`` are both None
until ``_handle_gpu_memory_settings``, and the reserve is decided earlier, in
``_handle_uneven_tp``. A profile built from unset fields digests differently,
misses every footprint, and the full-demand path then refuses -- truthfully,
by its own rules -- while a calibration for that exact rig sits in the cache.

The refusal fired on the FIRST call, the one whose number is installed, so the
boot kept the #590 reserve that had already OOMed in window 7. A LATER call,
after config resolution, logged the correct full demand into a log that no
budget was read from. Both numbers were in the same log file, 1 second apart.

So the bind proof here is specifically about ORDER: with the profile unresolved
at call time, the number that comes OUT must still be the full-model one.
"""

import types
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GPU_MEM = 20480
FULL_DEMAND_MIB = 3182  # what the ledger prices this card at, window 8 desk check
FALLBACK_MIB = 1958  # what #590 installed instead, and what OOMed


class _WindowEightStub:
    """A ServerArgs at the moment the reserve is decided.

    The two profile fields are None, exactly as they are inside
    ``_handle_uneven_tp``. ``_build_card_ledgers`` stands in for the footprint
    lookup and prices the card ONLY when the profile is resolved, which is the
    dependency that made window 8 refuse.
    """

    def __init__(self):
        self.chunked_prefill_size = None
        self.cuda_graph_config = types.SimpleNamespace(
            decode=types.SimpleNamespace(max_bs=None)
        )
        self.tp_size = 3
        self.profile_resolved_at_build = None
        self.capacity_defaults_calls = 0

    # -- the real methods under test -------------------------------------
    def ledger_full_demand_per_gpu(self, gpu_mem=None):
        return ServerArgs.ledger_full_demand_per_gpu(self, gpu_mem)

    def reserve_demand_per_gpu(self, gpu_mem, counts):
        return ServerArgs.reserve_demand_per_gpu(self, gpu_mem, counts)

    # -- the seam ---------------------------------------------------------
    def _apply_gpu_mem_capacity_defaults(self, gpu_mem):
        """The real one is idempotent and fills only unset values."""
        self.capacity_defaults_calls += 1
        if self.chunked_prefill_size is None:
            self.chunked_prefill_size = 2048
        if self.cuda_graph_config.decode.max_bs is None:
            self.cuda_graph_config.decode.max_bs = 24

    def _build_card_ledgers(self):
        resolved = (
            self.chunked_prefill_size is not None
            and self.cuda_graph_config.decode.max_bs is not None
        )
        self.profile_resolved_at_build = resolved
        if not resolved:
            # The window-8 behaviour: an unresolved profile digests to a key
            # nothing was cached under, so activation and capture come back
            # unpriced and the whole card refuses.
            return [
                types.SimpleNamespace(
                    gpu_id=1,
                    card="NVIDIA GeForce RTX 3080",
                    unbounded=(
                        "runtime activation + metadata on NVIDIA GeForce RTX 3080",
                        "CUDA graph capture on NVIDIA GeForce RTX 3080",
                    ),
                    terms=(),
                )
            ]
        return [
            types.SimpleNamespace(
                gpu_id=1,
                card="NVIDIA GeForce RTX 3080",
                unbounded=(),
                terms=(
                    types.SimpleNamespace(
                        name="runtime activation + metadata", mib=1766
                    ),
                    types.SimpleNamespace(name="CUDA graph capture", mib=640),
                    types.SimpleNamespace(name="GDN prefill scratch", mib=464),
                    types.SimpleNamespace(name="attention workspace", mib=0),
                    types.SimpleNamespace(
                        name="hardware residual (per process)", mib=312
                    ),
                ),
            )
        ]

    # -- fallback path, unchanged from #590 -------------------------------
    def _reserve_card_uuids(self, gpu_ids):
        return {}

    def ladder_reserve_gpu_id(self):
        return None

    def derived_rank_auto_reserve_mib(self, gpu_mem, cnt, **kw):
        return FALLBACK_MIB


class TestReserveIsDecidedFromTheResolvedProfile(unittest.TestCase):
    def setUp(self):
        ServerArgs._full_demand_refusal_named = False

    def test_bind_proof_installed_reserve_carries_the_full_model_number(self):
        """THE bind proof.

        The profile is unresolved when the reserve is asked for -- the window-8
        state -- and the number that comes back must still be the full model's.
        Getting FALLBACK_MIB here is not a smaller number; it is the number
        that OOMed twice.
        """
        stub = _WindowEightStub()
        self.assertIsNone(stub.chunked_prefill_size)
        self.assertIsNone(stub.cuda_graph_config.decode.max_bs)

        out = stub.reserve_demand_per_gpu(GPU_MEM, {1: 1})

        self.assertEqual(
            out,
            {1: FULL_DEMAND_MIB},
            "the installed reserve is not the full-model number; with the "
            "profile unresolved at decision time this is window 8 again",
        )
        self.assertNotEqual(out[1], FALLBACK_MIB)

    def test_the_profile_was_resolved_before_the_footprint_was_looked_up(self):
        """Names the mechanism, so a future refactor that reorders it fails
        here rather than in a boot."""
        stub = _WindowEightStub()
        stub.reserve_demand_per_gpu(GPU_MEM, {1: 1})
        self.assertTrue(
            stub.profile_resolved_at_build,
            "the ledger was built while chunked_prefill_size / decode.max_bs "
            "were still unset, so its footprint lookup used a digest nothing "
            "is cached under",
        )
        self.assertEqual(stub.chunked_prefill_size, 2048)
        self.assertEqual(stub.cuda_graph_config.decode.max_bs, 24)

    def test_no_refusal_is_logged_when_the_profile_can_be_resolved(self):
        """Window 8 logged a REFUSES line naming 6 terms. With the ordering
        fixed there is nothing to refuse."""
        stub = _WindowEightStub()
        with self.assertNoLogs("sglang.srt.server_args", level="WARNING"):
            stub.reserve_demand_per_gpu(GPU_MEM, {1: 1})

    def test_resolution_is_idempotent_and_does_not_overwrite_explicit_values(self):
        """An operator who pinned these must keep them: the helper fills only
        unset values, and the reserve must not silently re-tier a pinned
        config."""
        stub = _WindowEightStub()
        stub.chunked_prefill_size = 4096
        stub.cuda_graph_config.decode.max_bs = 8
        stub.reserve_demand_per_gpu(GPU_MEM, {1: 1})
        self.assertEqual(stub.chunked_prefill_size, 4096)
        self.assertEqual(stub.cuda_graph_config.decode.max_bs, 8)

    def test_without_gpu_mem_the_path_does_not_invent_a_tier(self):
        """gpu_mem is how the tier is chosen; with none there is nothing to
        resolve from, and guessing one would be the #593 partial-sum mistake in
        a new place."""
        stub = _WindowEightStub()
        stub.ledger_full_demand_per_gpu(None)
        self.assertEqual(stub.capacity_defaults_calls, 0)


class TestWhyNotTheOtherTwoDirections(unittest.TestCase):
    """Direction (c) was rejected on evidence, and this pins the evidence.

    Dropping decode_max_bs from the digest would make one bs's measurement
    serve another. The reference footprints carry a capture term per card, and
    capture memory is the graphs' -- it scales with the captured batch sizes.
    A digest without decode_max_bs would hand a bs=24 capture number to a
    bs=160 boot, which is an under-reserve of the same shape as window 7's.
    """

    def test_the_profile_digest_still_depends_on_decode_max_bs(self):
        from sglang.srt.mem_ledger.activation import ActivationProfile, profile_key

        base = dict(
            architectures=("Qwen3_5ForConditionalGeneration",),
            chunked_prefill_size=2048,
            tp_size=3,
            pp_size=1,
            kv_cache_dtype="fp8_e4m3",
            speculative_num_draft_tokens=4,
        )
        self.assertNotEqual(
            profile_key(ActivationProfile(decode_max_bs=24, **base)),
            profile_key(ActivationProfile(decode_max_bs=160, **base)),
            "if these digests ever collapse, a capture measurement taken at "
            "one decode batch size will be served to a boot at another",
        )

    def test_the_profile_digest_still_depends_on_chunked_prefill_size(self):
        from sglang.srt.mem_ledger.activation import ActivationProfile, profile_key

        base = dict(
            architectures=("Qwen3_5ForConditionalGeneration",),
            tp_size=3,
            pp_size=1,
            kv_cache_dtype="fp8_e4m3",
            speculative_num_draft_tokens=4,
            decode_max_bs=24,
        )
        self.assertNotEqual(
            profile_key(ActivationProfile(chunked_prefill_size=2048, **base)),
            profile_key(ActivationProfile(chunked_prefill_size=8192, **base)),
        )


if __name__ == "__main__":
    unittest.main()
