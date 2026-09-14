# SPDX-License-Identifier: Apache-2.0
"""#1378 Schritt B (ring-off boot weg2xsn34): the launch-time W4 lock learns
the one arm where its premise does not hold.

THE WALL, measured three times (A2/xsn32 died on it, the arm script's 5d gate
locked a fourth attempt this morning): ``scheduler.py``'s launch half refuses
every ``--enable-weights-cpu-backup``-less group on a QUANTIZED checkpoint,
because its premise is "the wake refills the weights with
``update_weights_from_disk``".  Under ``exchange`` + ``authoritative`` that
premise is FALSE: the wake refill is the exchange COLLECT, whose completeness
is guarded per tag -- BEFORE the pause mutates VRAM -- by the sleep-side W106
(``weight_updater._weg2_xchg_wake_source_gap`` -> ``Weg2XchgWakeSourceGapRefused``,
tag + measured byte count).  A blanket launch refusal there condemns a DEFINED
arm.

THE SHAPE: the exemption is decided INSIDE the assert
(``assert_backup_off_wake_refill_is_defined(..., exchange_owns_wake_refill=)``
) from ONE authority, ``weg2_memory_saver.exchange_owns_wake_refill()`` --
the SAME conjunction ``weights_cpu_backup_armed``'s auto branch reads
(exchange_armed() AND inject_authoritative(); ring absence is a property of
the inject arm, boot weg2xsn13's lesson).  The scheduler passes the authority,
never an inline env re-parse.  The DRAFT's disk-reload call site
(``weight_updater._weg2_xchg_draft_reload_from_disk``) keeps the lock
UNCONDITIONAL: that path really is ``update_weights_from_disk``, undefined on
a quantized checkpoint, and its own refusal is the finding -- never widened.

RED-FIRST RECORD (2026-09-14, against the pre-fix tree): the exchange case
raised W4 (the wall that killed A2); the kwarg did not exist (TypeError).
"""

from __future__ import annotations

import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.weg2_memory_saver import (  # noqa: E402
    assert_backup_off_wake_refill_is_defined,
    exchange_owns_wake_refill,
    Weg2WakeRefused,
)
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402

QUANT = "compressed-tensors"


class TheLaunchLockLearnsTheExchangeArm(unittest.TestCase):
    """Behavioural: the assert's own four cases, no scheduler construction."""

    def test_exchange_authoritative_launch_passes_the_lock(self):
        """THE WALL ITSELF: quantized + backup-OFF + exchange owning the
        refill must NOT refuse at launch -- the exchange collect is the
        refill and W106 guards its completeness per tag."""
        assert_backup_off_wake_refill_is_defined(
            quantization=QUANT, context="launch (scheduler init)",
            exchange_owns_wake_refill=True,
        )

    def test_ring_arm_launch_on_quantized_STILL_refuses(self):
        """The regression guard, and the mutant's target: a backup-OFF boot
        whose refill IS the disk reload (ring/shadow arm) must keep the
        refusal. Widening the exemption to any exchange arm -- or to
        ``exchange_armed() OR inject_authoritative()`` -- dies HERE."""
        with self.assertRaises(Weg2WakeRefused) as cm:
            assert_backup_off_wake_refill_is_defined(
                quantization=QUANT, context="launch (scheduler init)",
                exchange_owns_wake_refill=False,
            )
        self.assertIn("W4 Weg2WakeRefused", str(cm.exception))

    def test_unquantized_backup_off_still_passes(self):
        """Unchanged case: an unquantized checkpoint's disk reload IS
        defined -- the lock never fired for it and must not start."""
        assert_backup_off_wake_refill_is_defined(
            quantization=None, context="launch (scheduler init)",
            exchange_owns_wake_refill=False,
        )

    def test_default_keeps_the_refusal(self):
        """The parameter is keyword-only with default False: every EXISTING
        caller (the draft's disk-reload path) keeps today's behaviour
        byte for byte unless it names the exemption."""
        with self.assertRaises(Weg2WakeRefused):
            assert_backup_off_wake_refill_is_defined(
                quantization=QUANT, context="wake path")


class TheAuthorityIsOneConjunction(unittest.TestCase):
    """``exchange_owns_wake_refill()`` reads BOTH axes through the ONE
    predicates, and answers False unless BOTH hold. The four env shapes are
    the mutant kill: an OR-wiring (or a dropped conjunct) answers True on
    the one-axis shapes and dies here."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       (wx.WEIGHT_SOURCE_ENV, wx.INJECT_ENV)}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _arm(self, source: str, inject: str) -> bool:
        os.environ[wx.WEIGHT_SOURCE_ENV] = source
        os.environ[wx.INJECT_ENV] = inject
        return exchange_owns_wake_refill()

    def test_exchange_authoritative_is_true(self):
        self.assertTrue(self._arm("exchange", "authoritative"))

    def test_exchange_shadow_is_false(self):
        """Shadow grades AGAINST the ring's ground truth: the lock stays."""
        self.assertFalse(self._arm("exchange", "shadow"))

    def test_ring_authoritative_is_false(self):
        self.assertFalse(self._arm("ring", "authoritative"))

    def test_no_env_at_all_is_false(self):
        for k in (wx.WEIGHT_SOURCE_ENV, wx.INJECT_ENV):
            os.environ.pop(k, None)
        self.assertFalse(exchange_owns_wake_refill())


class TheCallSitesAreWiredOnce(unittest.TestCase):
    """Structural pins in the tree's own inspect-based style: the scheduler's
    launch half hands the authority; the draft's disk-reload half keeps the
    lock unconditional -- widening THAT one is the silent-undefined direction
    (a quantized checkpoint's reload would run because a typo said so)."""

    def test_scheduler_launch_call_passes_the_authority(self):
        from sglang.srt.managers import scheduler
        src = inspect.getsource(scheduler)
        self.assertIn("exchange_owns_wake_refill=exchange_owns_wake_refill()",
                      src,
                      "the launch half must hand the ONE authority's result, "
                      "not an inline re-parse")

    def test_the_draft_disk_reload_keeps_the_lock_unconditional(self):
        from sglang.srt.managers.scheduler_components import weight_updater
        fn = getattr(weight_updater.SchedulerWeightUpdaterManager,
                     "_weg2_xchg_draft_reload_from_disk")
        src = inspect.getsource(fn)
        self.assertIn("assert_backup_off_wake_refill_is_defined(", src,
                      "the draft reload must still run the lock")
        self.assertNotIn("exchange_owns_wake_refill", src,
                         "the draft's disk reload is update_weights_from_disk "
                         "on a possibly-quantized checkpoint -- its lock has "
                         "no exchange exemption")


if __name__ == "__main__":
    unittest.main()
