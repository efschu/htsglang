# SPDX-License-Identifier: Apache-2.0
"""27b-flipverdict-1023, review gap for the #739 guard: a queue that is 20+ s old
does NOT alarm while prefill chunks are still landing (last_prefill_progress_time
fresh). The #739 guard in invariant_checker.admission_wedge_verdict suppresses
the admission-wedge alarm when the queue age (first-token age capped at the
queue-non-empty age, deskq 1981) is >= 20 s but a prefill chunk completed
< 20 s ago: a mega-prefill produces no first token for minutes by
construction, so the box is PREFILLING, not wedged. The existing test stub
keeps last_prefill_progress_time = None, so that direction was untested.
Counter-probe: the identical timeline with no prefill-progress signal still
alarms (a genuine stand is not silenced by the guard).
"""
from __future__ import annotations

import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.managers.scheduler_components import invariant_checker as IC  # noqa: E402
from flliper.test.test_utils import CustomTestCase  # noqa: E402

QC = "FLLIPER_ADMISSION_WEDGE_QUEUE_CLOCK"
DUAL = "FLLIPER_PDFLIP_DUAL_LAYOUT"


def _unset_qc():
    # same harness as test_pdflip_flipverdict_1023: queue clock at its default
    # (armed in the dual layout since the commit), nothing else set
    e = {k: v for k, v in os.environ.items() if k != QC}
    e[DUAL] = "1"
    return mock.patch.dict(os.environ, e, clear=True)


class _Sched:
    def __init__(self, last_token, queued=0, running=0):
        self.is_initializing = False
        self.waiting_queue = [object()] * queued
        self.running_batch = types.SimpleNamespace(reqs=[object()] * running)
        self.last_first_token_progress_time = last_token
        self.last_prefill_progress_time = None
        self.forward_ct = 0


class PrefillProgressGuard739(CustomTestCase):
    def test_old_queue_with_fresh_prefill_chunk_does_not_alarm(self):
        """t=0: P's last first token. t=10: a request arrives, the queue goes
        non-empty (the queue clock stamps the first poll that sees it). At the
        t=35 poll the queue is 25 s old (>= 20 s) with no first token since
        t=0 -- but a prefill chunk completed at t=34, 1 s ago (< 20 s): the #739
        guard keeps the verdict on PREFILLING, not wedged, so NO alarm."""
        with _unset_qc():
            s = _Sched(last_token=0.0, queued=1)
            alarm, detail = IC.check_admission_wedge_once(s, now=10.0)
            self.assertFalse(alarm, detail)  # 0 s old at first sight: stamp only
            s.last_prefill_progress_time = 34.0  # a chunk just landed
            alarm, detail = IC.check_admission_wedge_once(s, now=35.0)
            self.assertFalse(alarm, detail)  # 25 s old but prefilling: no alarm

    def test_same_timeline_without_prefill_progress_still_alarms(self):
        """Counter-probe: identical timeline, but the prefill-progress signal is
        absent (last_prefill_progress_time = None, the pre-#739 stub state).
        A genuine stand (queued 25 s, 0 running, nothing serving it) still
        alarms -- the guard does not over-silence."""
        with _unset_qc():
            s = _Sched(last_token=0.0, queued=1)  # last_prefill_progress_time = None
            alarm, detail = IC.check_admission_wedge_once(s, now=10.0)
            self.assertFalse(alarm, detail)
            alarm, detail = IC.check_admission_wedge_once(s, now=35.0)
            self.assertTrue(alarm, detail)
