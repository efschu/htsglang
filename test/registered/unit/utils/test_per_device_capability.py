"""One process, two architectures: capability answers must not be shared (#343).

The dual-group lane host holds parts of ONE model on TWO cards inside ONE
process and loads each part under ``torch.cuda.device(part_gpu_id)``. Every
capability gate in this tree used to be either

  * ``lru_cache(maxsize=1)`` -- one slot, so whichever card asked first froze
    the answer for the other; or
  * an import-time module constant (``_is_sm120_supported = ...``,
    ``_ENABLE_PDL = ...``, a function-default ``= cutlass_fp8_supported()``)
    -- device 0's answer, frozen for the life of the process; or
  * a ``device_id: int = 0`` default -- device 0 no matter where the caller
    actually runs.

The observable damage: on a 5090 (sm120) + 3080 (sm86) host the sm86 part was
handed the sm120 part's ``triton`` FP8 backend and died on
``type fp8e4nv not supported in this architecture`` instead of taking its own
Marlin route. The same shape bites a plain TP rank on a heterogeneous rig that
was not given a per-rank ``CUDA_VISIBLE_DEVICES``.

These tests mock a two-architecture rig -- no GPU is touched -- and assert
that the SAME process gets DIFFERENT answers for device 0 and device 1, and
that a falsifier (asking both about one id) still agrees. Each also pins the
single-card / GPU-passive behaviour, which must not move.
"""

import contextlib
import unittest
from unittest import mock

from sglang.srt.utils import common

# The rig this was found on: card 0 is the 5090, card 1 a 3080.
_SM120 = (12, 0)
_SM86 = (8, 6)
_RIG = {0: _SM120, 1: _SM86}


@contextlib.contextmanager
def two_architecture_rig(rig=None, initialized=True, current=0):
    """A mocked CUDA context where each device index has its own capability.

    ``initialized`` / ``current`` model what ``resolve_capability_device_id``
    reads: while torch.cuda is uninitialized the current device is 0 by
    definition, which is what keeps GPU-passive processes unchanged.
    """
    rig = _RIG if rig is None else rig
    common.clear_per_device_gate_caches()
    try:
        with mock.patch.object(common, "is_cuda", lambda: True), mock.patch.object(
            common,
            "get_device_capability_no_init",
            lambda device=None: rig[common.resolve_capability_device_id(device)],
        ), mock.patch.object(
            common.torch.cuda, "is_available", lambda: True
        ), mock.patch.object(
            common.torch.cuda, "is_initialized", lambda: initialized
        ), mock.patch.object(
            common.torch.cuda, "current_device", lambda: current
        ), mock.patch.object(
            common.torch,
            "version",
            mock.Mock(hip=None, cuda="12.8"),
        ):
            yield
    finally:
        common.clear_per_device_gate_caches()


class TestResolveCapabilityDeviceId(unittest.TestCase):
    def test_explicit_id_wins(self):
        with two_architecture_rig(current=1):
            self.assertEqual(common.resolve_capability_device_id(0), 0)
            self.assertEqual(common.resolve_capability_device_id(7), 7)

    def test_none_is_the_current_device_once_cuda_is_up(self):
        with two_architecture_rig(current=1):
            self.assertEqual(common.resolve_capability_device_id(None), 1)

    def test_none_is_zero_while_cuda_is_uninitialized(self):
        """The GPU-passive guarantee (task #237): no CUDA context, and the
        answer is exactly the one the old ``device_id=0`` default gave."""
        with two_architecture_rig(initialized=False, current=1):
            self.assertEqual(common.resolve_capability_device_id(None), 0)


class TestArchitectureGatesAreNotShared(unittest.TestCase):
    """The ``lru_cache(maxsize=1)`` family."""

    def test_sm120_gate_answers_per_device(self):
        with two_architecture_rig():
            self.assertTrue(common.is_sm120_supported(0))
            self.assertFalse(common.is_sm120_supported(1))
            # ... and again, now that both are cached: the second card's
            # answer must not have been overwritten by the first's.
            self.assertTrue(common.is_sm120_supported(0))
            self.assertFalse(common.is_sm120_supported(1))

    def test_ask_order_does_not_change_the_answer(self):
        """The falsifier for the maxsize=1 bug: with one slot, whichever id is
        asked first wins, so reversing the order flips the second answer."""
        with two_architecture_rig():
            self.assertFalse(common.is_sm120_supported(1))
            self.assertTrue(common.is_sm120_supported(0))

    def test_ampere_gate_answers_per_device(self):
        with two_architecture_rig():
            self.assertFalse(common.is_ampere_with_cuda_12_3(0))
            self.assertTrue(common.is_ampere_with_cuda_12_3(1))

    def test_blackwell_gate_answers_per_device(self):
        with two_architecture_rig():
            self.assertTrue(common.is_blackwell_supported(0))
            self.assertFalse(common.is_blackwell_supported(1))

    def test_default_follows_the_current_device(self):
        """No caller has to plumb an id: the lane part loader already runs the
        whole load under ``torch.cuda.device(part_gpu_id)``."""
        with two_architecture_rig(current=0):
            self.assertTrue(common.is_sm120_supported())
        with two_architecture_rig(current=1):
            self.assertFalse(common.is_sm120_supported())

    def test_raw_readers_default_to_the_current_device(self):
        with two_architecture_rig(current=1):
            self.assertEqual(common.get_cuda_sm(), 86)
            self.assertEqual(common.get_cuda_sm(0), 120)
            self.assertTrue(common.cuda_sm_at_least(8))
            self.assertFalse(common.cuda_sm_at_least(9))
            self.assertTrue(common.cuda_sm_at_least(9, device_id=0))
            self.assertTrue(common.cuda_sm_in_range((8, 0), (8, 9)))
            self.assertFalse(common.cuda_sm_in_range((8, 0), (8, 9), device_id=0))


class TestPerDeviceGateHelper(unittest.TestCase):
    def test_it_caches_per_device_and_can_be_cleared(self):
        calls = []

        @common.per_device_gate
        def probe(device_id: int) -> int:
            calls.append(device_id)
            return device_id * 10

        self.assertEqual(probe(0), 0)
        self.assertEqual(probe(1), 10)
        self.assertEqual(probe(0), 0)
        self.assertEqual(calls, [0, 1])  # cached, one call per device

        probe.cache_clear()
        self.assertEqual(probe(1), 10)
        self.assertEqual(calls, [0, 1, 1])


class TestFp8DispatchIsPerDevice(unittest.TestCase):
    """The FP8 dispatch layer, which is where this was found."""

    def _fp8_utils(self):
        from sglang.srt.layers.quantization import fp8_utils

        return fp8_utils

    def test_cutlass_gate_answers_per_device(self):
        fp8_utils = self._fp8_utils()
        with two_architecture_rig(), mock.patch.object(
            fp8_utils, "_is_cuda", True
        ), mock.patch.object(fp8_utils, "get_cuda_version", lambda: (12, 8)):
            # sm120 -> major >= 9 -> yes; sm86 -> neither >= 9 nor exactly
            # (8, 9) -> no.
            self.assertTrue(fp8_utils.cutlass_fp8_supported(0))
            self.assertFalse(fp8_utils.cutlass_fp8_supported(1))

    def test_marlin_gate_answers_per_device(self):
        fp8_utils = self._fp8_utils()
        with two_architecture_rig():
            # can_auto_enable_marlin_fp8 is exactly sm80..88.
            self.assertFalse(fp8_utils.can_auto_enable_marlin_fp8(0))
            self.assertTrue(fp8_utils.can_auto_enable_marlin_fp8(1))

    def test_the_bug_verbatim_sm86_part_does_not_inherit_triton(self):
        """``initialize_fp8_gemm_config`` is called ONCE per process, before
        any part is placed. The sm120 host must still get ``triton`` and the
        sm86 part must NOT -- that inheritance is what produced
        ``type fp8e4nv not supported in this architecture``."""
        fp8_utils = self._fp8_utils()
        server_args = mock.Mock(fp8_gemm_runner_backend="auto", quantization=None)
        with two_architecture_rig():
            fp8_utils.initialize_fp8_gemm_config(server_args)
            host = fp8_utils.get_fp8_gemm_runner_backend(0)
            part = fp8_utils.get_fp8_gemm_runner_backend(1)
        self.assertTrue(host.is_triton(), host)
        self.assertFalse(part.is_triton(), part)
        self.assertTrue(part.is_auto(), part)

    def test_explicit_backend_still_applies_to_every_device(self):
        """``--fp8-gemm-backend`` is a user statement about the launch, not a
        capability probe: it must NOT become per-device."""
        fp8_utils = self._fp8_utils()
        server_args = mock.Mock(fp8_gemm_runner_backend="triton", quantization=None)
        with two_architecture_rig():
            fp8_utils.initialize_fp8_gemm_config(server_args)
            self.assertTrue(fp8_utils.get_fp8_gemm_runner_backend(0).is_triton())
            self.assertTrue(fp8_utils.get_fp8_gemm_runner_backend(1).is_triton())

    def test_reinitialising_drops_the_previous_answers(self):
        fp8_utils = self._fp8_utils()
        with two_architecture_rig():
            fp8_utils.initialize_fp8_gemm_config(
                mock.Mock(fp8_gemm_runner_backend="triton", quantization=None)
            )
            self.assertTrue(fp8_utils.get_fp8_gemm_runner_backend(1).is_triton())
            fp8_utils.initialize_fp8_gemm_config(
                mock.Mock(fp8_gemm_runner_backend="auto", quantization=None)
            )
            self.assertFalse(fp8_utils.get_fp8_gemm_runner_backend(1).is_triton())


class TestPdlIsPerDevice(unittest.TestCase):
    """#267, the same family: an import-time ``_ENABLE_PDL`` constant.

    PDL is a ``tl.constexpr`` baked into the compiled kernel, so a wrong True
    is a kernel the card cannot launch rather than a slow one.
    """

    def test_pdl_answers_per_device(self):
        from sglang.srt.layers import fused_qk_rmsnorm_rope_gate as mod

        rig = {0: (9, 0), 1: (8, 6)}
        with two_architecture_rig(rig=rig):
            self.assertTrue(mod._pdl_supported(0))
            self.assertFalse(mod._pdl_supported(1))

    def test_launch_asks_about_the_activation_s_card(self):
        from sglang.srt.layers import fused_qk_rmsnorm_rope_gate as mod

        rig = {0: (9, 0), 1: (8, 6)}
        with two_architecture_rig(rig=rig):
            self.assertTrue(mod._enable_pdl(_FakeDevice(0)))
            self.assertFalse(mod._enable_pdl(_FakeDevice(1)))


class _FakeDevice:
    def __init__(self, index):
        self.index = index


class TestMinVisibleCapability(unittest.TestCase):
    """The floor, for the decisions that genuinely are process-wide.

    A ``tl.constexpr`` baked at import and a JIT backend switched on for the
    whole process cannot be re-asked per device. For those, device 0's answer
    is wrong in the dangerous direction (the newer card arms a kernel variant
    the older one cannot run); the floor is safe on every visible card and
    keeps ONE numeric path through the model.
    """

    def _min_visible(self, count, capabilities):
        common.min_visible_cuda_capability_no_init.cache_clear()
        with mock.patch.object(common, "is_cuda", lambda: True), mock.patch.object(
            common.torch.cuda, "is_initialized", lambda: True
        ), mock.patch.object(
            common.torch.cuda, "device_count", lambda: count
        ), mock.patch.object(
            common.torch.cuda, "get_device_capability", lambda i: capabilities[i]
        ):
            try:
                return common.min_visible_cuda_capability_no_init()
            finally:
                common.min_visible_cuda_capability_no_init.cache_clear()

    def test_floor_over_a_mixed_rig(self):
        self.assertEqual(self._min_visible(2, {0: _SM120, 1: _SM86}), _SM86)

    def test_order_does_not_matter(self):
        self.assertEqual(self._min_visible(2, {0: _SM86, 1: _SM120}), _SM86)

    def test_homogeneous_rig_is_unchanged(self):
        self.assertEqual(self._min_visible(3, {i: _SM86 for i in range(3)}), _SM86)

    def test_no_visible_device_is_none_not_a_guess(self):
        self.assertIsNone(self._min_visible(0, {}))


if __name__ == "__main__":
    unittest.main()
