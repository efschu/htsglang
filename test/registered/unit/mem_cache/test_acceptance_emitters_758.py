"""The three acceptance emitters must FIRE and must STAY SILENT (#758).

WHY THESE EXIST. The comp4 load ladder (2026-08-18) ran the full session-load
probe and could not accept three features, not because they failed but because
nothing said anything about them:

    anchor cadence          0 lines matching "anchor"      in a 300 s window
    mamba host resume       0 lines matching "host-backed" in a 300 s window
    file-backed refill time no emitter anywhere in the tree at all

Zero lines cannot distinguish "working and silent" from "inert" -- the #742
silently-inert-flag class. So the acceptance was recorded as NOT EVIDENCED
rather than passed, and these emitters are what close that.

WHY EACH TEST HAS BOTH HALVES. An emitter that always logs is as useless as one
that never does: it would turn the ladder's "1 host-backed resume observed" into
a tautology. Every case below therefore asserts the positive AND the negative on
the SHIPPED code path, so a future change that welds one on is caught here.

SCOPE NOTE, stated rather than glossed. The host-resume and refill emitters are
exercised directly through the shipped functions. The anchor emitter sits inside
``ScheduleBatch``'s mamba tracking step, which needs a batch, a request and a
pool to reach; what is tested here is the exact predicate that GATES it
(``mamba_checkpoint_track_target``, the only place a target is chosen), on and
off the grid. Its logging line is proven live by the boot ladder, which is the
measurement it was written for.
"""

import logging
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15)


class AnchorGate(unittest.TestCase):
    """(1 of 3) anchor-written -- the gate must be able to say NO."""

    def setUp(self):
        from sglang.srt.mem_cache.mamba_ckpt_utils import (
            mamba_checkpoint_track_target,
        )

        self.target = mamba_checkpoint_track_target

    def test_an_anchor_is_reachable_on_the_750_grid(self):
        """8192 with a 512 chunk budget is the #750 divisibility case the
        composite boots with; a step that spans the boundary must yield one."""
        got = self.target(
            prefix_len=7680, extend_len=512, interval=8192, chunk_size=512
        )
        self.assertEqual(got, 8192, "no anchor chosen where one is reachable")

    def test_no_anchor_when_the_step_reaches_no_boundary(self):
        """CAN-FAIL HALF. A step wholly inside one interval must yield None --
        if this ever returns a target the emitter would log an anchor per
        step and the cadence number becomes meaningless."""
        self.assertIsNone(
            self.target(prefix_len=100, extend_len=200, interval=8192, chunk_size=512)
        )

    def test_no_anchor_when_the_boundary_is_chunk_unaligned(self):
        """The other refusal: reachable position, unreachable by the kernel's
        chunk grid."""
        self.assertIsNone(
            self.target(prefix_len=8000, extend_len=300, interval=8192, chunk_size=512)
        )


class HostResume(unittest.TestCase):
    """(2 of 3) mamba-host-resume -- fires ONLY on a host-only acceptance."""

    def _predicate(self, interval=8192):
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        holder = types.SimpleNamespace(
            component_type="mamba", mamba_checkpoint_interval=interval
        )
        # #815 STUB DRIFT: `create_match_validator` grew a third read of its
        # holder when afa332e6bb [#783] started measuring the checkpoint grid in
        # RAW tokens -- `self._raw_token_pos`, which in turn reads
        # `self.cache.is_eagle`. This holder predates that (#758, 2026-08-18).
        #
        # THE REAL METHOD IS BOUND HERE, not a stand-in returning the depth. A
        # lambda would pass today and would go on passing if `_raw_token_pos`
        # ever changed -- which is precisely the EAGLE bigram correction #783
        # exists to carry, so faking it would make this file blind to the thing
        # it now depends on. `cache.is_eagle=False` is the production default
        # (CacheInitParams.is_eagle), and it is the value these tests already
        # assume: they pass plain raw depths, which is the identity case the
        # docstring names ("Identity outside EAGLE").
        holder.cache = types.SimpleNamespace(is_eagle=False)
        holder._raw_token_pos = MambaComponent._raw_token_pos.__get__(holder)
        fn = MambaComponent.create_match_validator(holder, match_device_only=False)
        return fn, MambaComponent

    def _node(self, device_value, host_value):
        return types.SimpleNamespace(
            component_data={
                "mamba": types.SimpleNamespace(
                    value=device_value, host_value=host_value
                )
            }
        )

    def test_host_only_acceptance_emits(self):
        fn, cls = self._predicate()
        cls._host_resume_count = 0
        node = self._node(device_value=None, host_value=object())
        with self.assertLogs(level=logging.INFO) as cm:
            ok = fn(node, 8192)
        self.assertTrue(ok, "a host-backed anchor on the grid must match")
        self.assertTrue(
            any("MAMBA-HOST-RESUME" in m for m in cm.output),
            f"emitter did not fire: {cm.output}",
        )

    def test_a_resident_device_state_is_silent(self):
        """CAN-FAIL HALF. The common case must NOT log, or the ladder's
        '>=1 host-backed resume' becomes true for every ordinary match."""
        fn, cls = self._predicate()
        cls._host_resume_count = 0
        node = self._node(device_value=object(), host_value=object())
        logger = logging.getLogger(
            "sglang.srt.mem_cache.unified_cache_components.mamba_component"
        )
        with self.assertNoLogs(logger, level=logging.INFO):
            self.assertTrue(fn(node, 8192))

    def test_off_grid_host_state_is_refused_and_silent(self):
        fn, cls = self._predicate()
        cls._host_resume_count = 0
        node = self._node(device_value=None, host_value=object())
        logger = logging.getLogger(
            "sglang.srt.mem_cache.unified_cache_components.mamba_component"
        )
        with self.assertNoLogs(logger, level=logging.INFO):
            self.assertFalse(fn(node, 8193), "off-grid anchor must not match")


class RefillTiming(unittest.TestCase):
    """(3 of 3) per-rank refill timing, and it must name the image mode."""

    def _stack(self, monkey_refill):
        # #809/W28: the shipped copy is now the ROTATION, imported inside
        # `_timed_arena_refill` at call time, so this is the seam to stub.
        import sglang.srt.model_executor.rotation_executor as rx
        from sglang.srt.managers import phase_flip_boot

        rx.rotate_arena = monkey_refill  # shipped call site, stubbed copy
        holder = types.SimpleNamespace(arena=None, rotation_image=None)
        holder.image_holds = "pp"
        holder._images_are_file_backed = types.MethodType(
            phase_flip_boot.PhaseFlipStacks._images_are_file_backed, holder
        )
        holder._timed_arena_refill = types.MethodType(
            phase_flip_boot.PhaseFlipStacks._timed_arena_refill, holder
        )
        return holder

    def test_the_refill_is_timed_and_reported(self):
        calls = []
        holder = self._stack(lambda *a, **k: calls.append(1))
        layout = types.SimpleNamespace(total_bytes=1048576 * 64)
        with self.assertLogs(level=logging.INFO) as cm:
            holder._timed_arena_refill("tp_to_pp", layout, layout, "pp")
        self.assertEqual(len(calls), 1, "the real refill must still be called once")
        line = "\n".join(cm.output)
        self.assertIn("REFILL tp_to_pp", line)
        self.assertIn("MiB/s", line)

    def test_the_mode_is_named_so_the_baseline_cannot_be_misread(self):
        """A pinned refill and a file-backed one are different measurements;
        reading one against the other's ~3.1 s baseline is the mistake this
        label exists to prevent."""
        import os

        holder = self._stack(lambda *a, **k: None)
        layout = types.SimpleNamespace(total_bytes=1048576)
        prev = os.environ.get("SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED")
        try:
            os.environ["SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED"] = "1"
            # The rotation alternates the marker, so each leg is re-armed.
            holder.image_holds = "tp"
            with self.assertLogs(level=logging.INFO) as cm:
                holder._timed_arena_refill("pp_to_tp", layout, layout, "tp")
            self.assertIn("file-backed", "\n".join(cm.output))

            os.environ.pop("SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED")
            # The rotation alternates the marker, so each leg is re-armed.
            holder.image_holds = "tp"
            with self.assertLogs(level=logging.INFO) as cm:
                holder._timed_arena_refill("pp_to_tp", layout, layout, "tp")
            self.assertIn("pinned", "\n".join(cm.output))
        finally:
            if prev is None:
                os.environ.pop("SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED", None)
            else:
                os.environ["SGLANG_PHASE_FLIP_IMAGE_FILE_BACKED"] = prev


if __name__ == "__main__":
    unittest.main()
