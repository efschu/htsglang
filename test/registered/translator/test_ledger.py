# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Ledger registration AND a real park/restore. CPU, hermetic.

The standing rule this file answers to: registration that is only asserted is
decoration. So the tests move actual bytes -- park frees the tensors and the
module really becomes unusable, restore brings it back and the weights are
bit-identical. A park that quietly kept the tensors alive would pass a
"registered: true" assertion and fail the only thing the ledger is for.

CPU throughout: the park path is device-agnostic (`.to("meta")` /
`to_empty()`), so the mechanism is exercised without holding a card, which
matters because the GPU window is held by another agent.

    CUDA_VISIBLE_DEVICES=99 python -m pytest test/registered/translator/test_ledger.py -v
"""

import unittest

import torch
from torch import nn

from sglang.srt.translator.ledger import (
    OFFLOAD_CLASS,
    AudioAssetLedger,
    ParkError,
)


def tiny_module(in_features=64, out_features=128):
    torch.manual_seed(466)
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.Linear(out_features, in_features),
    )


class TestAssetClassIsDeclared(unittest.TestCase):
    def test_the_class_exists_in_the_runtime_register(self):
        from sglang.srt.model_executor.offload_register import OFFLOAD_CLASSES
        from sglang.srt.model_executor.short_term_offload_register import (
            ASSET_CLASSES,
            LadderRank,
        )

        self.assertIn(OFFLOAD_CLASS, OFFLOAD_CLASSES)
        descriptor = ASSET_CLASSES[OFFLOAD_CLASS]
        self.assertEqual(descriptor.ladder_rank, LadderRank.COLD_SECOND_MODEL)
        self.assertTrue(descriptor.wired)
        self.assertFalse(descriptor.va_stable_required)
        self.assertEqual(descriptor.dimension_presets, ("module",))

    def test_it_participates_in_the_global_eviction_order(self):
        # The point of registering rather than tracking privately: the
        # translator's assets must be comparable against everything else on
        # the ONE ladder, not on a private victim list.
        from sglang.srt.model_executor.short_term_offload_register import (
            ASSET_CLASSES,
            LadderRank,
        )

        ours = ASSET_CLASSES[OFFLOAD_CLASS].ladder_rank
        self.assertLess(
            ours,
            LadderRank.IDLE_SESSIONS,
            "an idle translator must be given up BEFORE a user's idle session",
        )
        self.assertLess(ours, LadderRank.ACTIVE_WORK)


class TestRegistration(unittest.TestCase):
    def test_registering_reports_a_real_size(self):
        ledger = AudioAssetLedger()
        module = tiny_module()
        asset = ledger.register("trunk", module)
        expected = sum(t.nbytes for t in module.state_dict().values())
        self.assertEqual(asset.size_bytes(), expected)
        self.assertGreater(asset.size_bytes(), 0)

    def test_double_registration_is_a_hard_error(self):
        ledger = AudioAssetLedger()
        ledger.register("trunk", tiny_module())
        with self.assertRaises(ParkError) as ctx:
            ledger.register("trunk", tiny_module())
        self.assertIn("lifecycle bug", str(ctx.exception))

    def test_an_unknown_asset_is_named_in_the_error(self):
        with self.assertRaises(ParkError) as ctx:
            AudioAssetLedger().park("nope")
        self.assertIn("nope", str(ctx.exception))

    def test_the_report_shows_residency(self):
        ledger = AudioAssetLedger()
        ledger.register_all([("trunk", tiny_module()), ("codec", tiny_module())])
        report = ledger.to_json()
        self.assertEqual(report["offload_class"], OFFLOAD_CLASS)
        self.assertEqual(len(report["assets"]), 2)
        self.assertGreater(report["resident_mib"], 0.0)
        self.assertTrue(all(not a["parked"] for a in report["assets"]))


class TestParkRestoreActuallyMovesBytes(unittest.TestCase):
    """The smoke the standing rule demands: not 'registered', but 'moved'."""

    def setUp(self):
        self.ledger = AudioAssetLedger()
        self.module = tiny_module()
        self.reference = {
            k: v.detach().clone() for k, v in self.module.state_dict().items()
        }
        self.probe = torch.randn(4, 64)
        with torch.inference_mode():
            self.expected = self.module(self.probe).clone()
        self.ledger.register("trunk", self.module)

    def test_park_frees_the_tensors_for_real(self):
        freed = self.ledger.park("trunk")
        self.assertGreater(freed, 0)
        self.assertTrue(self.ledger.get("trunk").parked)
        # THE proof: the parameters are gone, not copied. A park that kept the
        # tensors alive would free nothing and pass a weaker assertion.
        for tensor in self.module.state_dict().values():
            self.assertEqual(
                tensor.device.type,
                "meta",
                "park left a live tensor behind; no VRAM was actually freed",
            )

    def test_a_parked_module_cannot_be_used(self):
        self.ledger.park("trunk")
        with self.assertRaises(Exception):
            with torch.inference_mode():
                self.module(self.probe)

    def test_restore_brings_back_bit_identical_weights(self):
        self.ledger.park("trunk")
        elapsed = self.ledger.restore("trunk")
        self.assertGreater(elapsed, 0.0)
        self.assertFalse(self.ledger.get("trunk").parked)
        restored = self.module.state_dict()
        self.assertEqual(set(restored), set(self.reference))
        for key, original in self.reference.items():
            self.assertTrue(
                torch.equal(restored[key].cpu(), original),
                f"{key} changed across a park/restore cycle",
            )

    def test_the_module_computes_identically_after_a_round_trip(self):
        self.ledger.park("trunk")
        self.ledger.restore("trunk")
        with torch.inference_mode():
            after = self.module(self.probe)
        # Bit-identical, not merely close: nothing in a park is numeric.
        self.assertTrue(torch.equal(after, self.expected))

    def test_the_measured_restore_cost_replaces_the_estimate(self):
        self.assertIsNone(self.ledger.get("trunk").measured_restore_ms)
        self.ledger.park("trunk")
        self.ledger.restore("trunk")
        measured = self.ledger.get("trunk").measured_restore_ms
        self.assertIsNotNone(measured)
        self.assertGreater(measured, 0.0)
        self.assertEqual(
            self.ledger.to_json()["assets"][0]["restore_ms"], round(measured, 1)
        )

    def test_repeated_cycles_are_stable(self):
        for _ in range(3):
            self.ledger.park("trunk")
            self.ledger.restore("trunk")
        with torch.inference_mode():
            after = self.module(self.probe)
        self.assertTrue(torch.equal(after, self.expected))

    def test_parking_twice_is_a_no_op_not_an_error(self):
        first = self.ledger.park("trunk")
        second = self.ledger.park("trunk")
        self.assertGreater(first, 0)
        self.assertEqual(second, 0)

    def test_restoring_a_resident_asset_is_a_no_op(self):
        self.assertEqual(self.ledger.restore("trunk"), 0.0)


class TestMultiModulePolicy(unittest.TestCase):
    def test_modules_park_independently_at_module_grain(self):
        ledger = AudioAssetLedger()
        modules = {name: tiny_module() for name in ("talker", "predictor", "codec")}
        ledger.register_all(modules.items())
        before = ledger.to_json()["resident_mib"]

        ledger.park("codec")

        report = ledger.to_json()
        self.assertLess(report["resident_mib"], before)
        parked = {a["name"]: a["parked"] for a in report["assets"]}
        self.assertTrue(parked["codec"])
        self.assertFalse(parked["talker"])
        self.assertFalse(parked["predictor"])

    def test_ensure_resident_restores_only_what_is_parked(self):
        ledger = AudioAssetLedger()
        ledger.register_all([("talker", tiny_module()), ("codec", tiny_module())])
        ledger.park("codec")
        restored = ledger.ensure_resident()
        self.assertEqual(list(restored), ["codec"])
        self.assertEqual(ledger.to_json()["assets"][0]["parked"], False)

    def test_park_all_then_restore_all(self):
        ledger = AudioAssetLedger()
        ledger.register_all([("talker", tiny_module()), ("codec", tiny_module())])
        freed = ledger.park_all()
        self.assertGreater(freed, 0)
        self.assertEqual(ledger.to_json()["resident_mib"], 0.0)
        ledger.ensure_resident()
        self.assertGreater(ledger.to_json()["resident_mib"], 0.0)


class _RotaryLike(nn.Module):
    """A module shaped like the failure: a NON-PERSISTENT buffer beside a
    normal weight. ``inv_freq`` in the translator's rotary modules is
    registered exactly this way (``qwen3_tts_compat`` documents why), which is
    what makes it absent from ``state_dict()``."""

    def __init__(self, dim=8):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(dim, dtype=torch.float32))
        self.register_buffer(
            "inv_freq",
            1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)),
            persistent=False,
        )
        self.register_buffer("running", torch.ones(dim), persistent=True)


class _Nested(nn.Module):
    def __init__(self):
        super().__init__()
        self.rot = _RotaryLike()
        self.head = nn.Linear(8, 4)


class TestNonPersistentBuffersSurviveAParkRound(unittest.TestCase):
    """#568: a park/restore round must not silently replace a buffer with
    uninitialised memory.

    THE DEFECT. ``park`` captured ``module.state_dict()``, which by definition
    OMITS non-persistent buffers, and ``restore`` did ``to_empty(device=...)``
    followed by ``load_state_dict(..., strict=True)``. ``to_empty`` DOES
    re-materialise every buffer, including the non-persistent ones -- with
    whatever was in that memory -- and the strict load then fills only the keys
    the state dict has. So each non-persistent buffer came back as garbage,
    while park and restore both reported success and the measured restore
    latency looked healthy.

    For the translator that buffer is the rotary ``inv_freq``: garbage there
    gives NaN cos/sin, NaN attention scores, NaN logits, and the first thing
    anyone sees is "probability tensor contains inf, nan or element < 0" from
    ``torch.multinomial`` -- one park and thirty layers from the cause. The
    LOAD path already had a repair for exactly this failure
    (``qwen3_tts_compat.refresh_rotary_buffers``); the RESTORE path never had
    an equivalent, and nothing connected the two.

    These tests assert CORRECTNESS AFTER RESTORE -- the values -- rather than
    how many such buffers a model happens to have, which is a property of the
    model and not a constant worth pinning.

    CAN-FAIL: drop the non-persistent capture in ``park`` (or the write-back in
    ``restore``) and every test here goes red, because ``to_empty`` leaves the
    buffer holding whatever the allocator handed out.
    """

    def _ledger_with(self, module, name="rot"):
        ledger = AudioAssetLedger(tenant_id="t-568", pin_host_copies=False)
        ledger.register(name, module)
        return ledger

    def test_a_non_persistent_buffer_comes_back_bit_identical(self):
        module = _RotaryLike()
        expected = module.inv_freq.detach().clone()
        # The premise, asserted rather than assumed: it really is absent from
        # the state dict, so load_state_dict cannot be what restores it.
        self.assertNotIn("inv_freq", module.state_dict())

        ledger = self._ledger_with(module)
        ledger.park("rot")
        self.assertTrue(ledger.get("rot").parked)
        ledger.restore("rot")

        self.assertTrue(
            torch.equal(module.inv_freq, expected),
            "the non-persistent buffer did not survive the park round",
        )
        self.assertTrue(torch.isfinite(module.inv_freq).all())

    def test_it_stays_non_persistent_so_a_second_round_still_works(self):
        module = _RotaryLike()
        expected = module.inv_freq.detach().clone()
        ledger = self._ledger_with(module)
        for _ in range(2):
            ledger.park("rot")
            ledger.restore("rot")
        self.assertNotIn("inv_freq", module.state_dict())
        self.assertTrue(torch.equal(module.inv_freq, expected))

    def test_nested_submodule_buffers_are_restored_too(self):
        """The write-back has to find the owning submodule, not just the root."""
        module = _Nested()
        expected = module.rot.inv_freq.detach().clone()
        ledger = self._ledger_with(module, name="nested")
        ledger.park("nested")
        ledger.restore("nested")
        self.assertTrue(torch.equal(module.rot.inv_freq, expected))

    def test_persistent_state_is_unaffected(self):
        """The fix must not disturb what already worked."""
        module = _RotaryLike()
        w = module.weight.detach().clone()
        r = module.running.detach().clone()
        ledger = self._ledger_with(module)
        ledger.park("rot")
        ledger.restore("rot")
        self.assertTrue(torch.equal(module.weight.detach(), w))
        self.assertTrue(torch.equal(module.running, r))

    def test_a_module_without_non_persistent_buffers_is_unchanged(self):
        """Byte-identical to the pre-fix path when there is nothing to carry."""
        module = tiny_module()
        before = {k: v.detach().clone() for k, v in module.state_dict().items()}
        ledger = self._ledger_with(module, name="plain")
        ledger.park("plain")
        ledger.restore("plain")
        for k, v in module.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]))


if __name__ == "__main__":
    unittest.main()


class _OnlyNonPersistentBuffers(torch.nn.Module):
    """A module whose ENTIRE state is non-persistent buffers.

    Not contrived: a rotary cache is exactly this shape, and the audio stack
    carries them as standalone submodules.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", torch.arange(256.0), persistent=False)
        self.register_buffer("cos_cached", torch.ones(128), persistent=False)


class _MixedWithNested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(32, 32))
        self.register_buffer("inv_freq", torch.arange(64.0), persistent=False)
        self.rot = _OnlyNonPersistentBuffers()


class TestNonPersistentBuffersAreAccountedFor(unittest.TestCase):
    """#568, second half: the park path carries these buffers, but the LEDGER
    did not count them and the RESTORE guard did not look for them.

    Both are the same omission as the original bug -- ``state_dict()`` is not
    the module's state -- one layer up.
    """

    def _ledger_with(self, module, name="a"):
        ledger = AudioAssetLedger(tenant_id="t-568b", pin_host_copies=False)
        ledger.register(name, module)
        return ledger

    def _true_bytes(self, module):
        seen = {}
        for t in list(module.parameters()) + list(module.buffers()):
            seen[id(t)] = t.nbytes
        return sum(seen.values())

    def test_registered_size_matches_what_park_actually_frees(self):
        """The ledger prices victims on size_bytes() and reports parked_bytes
        from the park. They described different sets of tensors, so a parked
        module reported freeing MORE than it was ever registered as holding."""
        for module in (_RotaryLike(), _MixedWithNested(), _OnlyNonPersistentBuffers()):
            with self.subTest(module=type(module).__name__):
                ledger = self._ledger_with(module)
                registered = ledger.get("a").size_bytes()
                self.assertEqual(registered, self._true_bytes(module))
                freed = ledger.park("a")
                self.assertEqual(
                    freed,
                    registered,
                    "park freed a different number of bytes than the ledger "
                    "registered -- the two enumerate different tensor sets",
                )

    def test_a_module_of_only_non_persistent_buffers_survives_the_round_trip(self):
        """The sharp case. Its state dict is EMPTY, so the restore guard used to
        declare it unrecoverable while its bytes sat on the host -- a park that
        succeeded followed by a restore that refused."""
        module = _OnlyNonPersistentBuffers()
        expected = module.inv_freq.detach().clone()
        self.assertEqual(len(module.state_dict()), 0)

        ledger = self._ledger_with(module)
        self.assertGreater(ledger.get("a").size_bytes(), 0)
        ledger.park("a")
        ledger.restore("a")

        self.assertTrue(torch.equal(module.inv_freq, expected))
        self.assertTrue(torch.isfinite(module.cos_cached).all())

    def test_such_a_module_is_not_silently_treated_as_empty(self):
        """Before the fix its tensors() was empty, which sent park down the
        'nothing to do' path: marked parked, nothing freed, nothing saved."""
        module = _OnlyNonPersistentBuffers()
        ledger = self._ledger_with(module)
        self.assertEqual(len(ledger.get("a").tensors()), 2)
        self.assertGreater(ledger.park("a"), 0)

    def test_a_genuinely_interrupted_park_is_still_refused(self):
        """The guard must not become permissive: with BOTH stores empty the
        module really is unrecoverable, and saying so is the point."""
        module = _RotaryLike()
        ledger = self._ledger_with(module)
        ledger.park("a")
        asset = ledger.get("a")
        asset._cpu_state = {}
        asset._cpu_nonpersistent = {}
        with self.assertRaises(ParkError):
            ledger.restore("a")
