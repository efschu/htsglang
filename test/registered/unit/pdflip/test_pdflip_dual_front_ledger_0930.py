# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3 metal replay dual15 (45nh9g, ...09301211 @292108fda7).

(1) D raised pressure on P: GROW-SHORT at 12:21:26 with pressure_on_P=150208512,
    GROUP-WAIT at 12:21:58 with pressure_on_P=754188288. P never paused. The
    front read /dev/shm/wkv-828dfeb4b2b5 (5090). The launcher derives that
    path from NVML's uuid "GPU-31d7ef41-...". The ranks derive their path
    from torch's device uuid, which carries no "GPU-" prefix, so they wrote
    wkv-04eb21e67816. The front read a ledger nobody writes: pressure 0,
    always.
(3) LEDGER-PHYS OVER-PROMISE by 628359168 B, steady from 12:17:24: card use
    outside the KV pools (activations, graph pools, links) that the budget
    did not book. It must go into the budget. A persistent over-promise is
    booked atomically against the physical free bytes and never twice; a
    one-off spike is not.
"""
from __future__ import annotations

import os
import tempfile
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import card_kv_ledger as K
from flliper.srt.pdflip import dual_p_kv_stage as S
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MIB = 1 << 20
TAG = "dkr27bnvfp4dual1gbar1fs09301211"


class FrontReadsTheRanksLedger(CustomTestCase):
    def test_nvml_and_torch_uuid_forms_name_one_ledger(self):
        nvml = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
        torch_form = "31d7ef41-f574-4d0e-21ad-e773fd938f6d"
        self.assertEqual(K.ledger_path(TAG, nvml), K.ledger_path(TAG, torch_form))
        self.assertEqual(K.ledger_path(TAG, nvml.upper()), K.ledger_path(TAG, torch_form))

    def test_launcher_paths_are_the_rank_paths(self):
        from flliper.srt.pdflip import launcher as L

        ns = types.SimpleNamespace(dual_share=True, tag=TAG)
        cards = [types.SimpleNamespace(uuid="GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"),
                 types.SimpleNamespace(uuid="GPU-5c648f96-be1d-42d5-0221-34d11ab137f7")]
        self.assertEqual(L.dual_kv_ledger_paths(ns, cards),
                         [K.ledger_path(TAG, "31d7ef41-f574-4d0e-21ad-e773fd938f6d"),
                          K.ledger_path(TAG, "5c648f96-be1d-42d5-0221-34d11ab137f7")])

    def test_the_front_sees_the_pressure_d_raised(self):
        root = tempfile.mkdtemp(prefix="wkvfr")
        at = K.ledger_path(TAG, "31d7ef41-f574-4d0e-21ad-e773fd938f6d", root=root)   # the ranks' form
        rank = K.CardKvLedger(at, "D")
        rank.contribute(10 * MIB, committed=10 * MIB)
        p = K.CardKvLedger(at, "P")
        p.contribute(10 * MIB)
        p.request(8 * MIB)
        rank.request(9 * MIB)                                    # D short -> pressure on P
        front = [K.ledger_path(TAG, "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d", root=root)]
        self.assertGreater(K.p_pressure(front), 0)


class PersistentOverPromiseIsBooked(CustomTestCase):
    def _card(self):
        path = os.path.join(tempfile.mkdtemp(prefix="wkvbk"), "card")
        d = K.CardKvLedger(path, "D")
        d.contribute(4538 * MIB, committed=320 * MIB)
        K.CardKvLedger(path, "P").contribute(0)
        return path, d

    def _run(self, actor, phys_seq, t0=0.0):
        clock = [t0]
        seq = list(phys_seq)
        with mock.patch.object(S, "_now", lambda: clock[0]), \
                mock.patch.object(S, "phys_free_bytes", lambda: seq[0]):
            for i, ph in enumerate(phys_seq):
                seq[0] = ph
                clock[0] = t0 + i * S.PHYS_CHECK_S
                S.phys_check(actor, "D")

    def test_steady_over_promise_goes_into_the_budget_once(self):
        path, d = self._card()
        free = K.peek(path).free
        phys = free - 628359168
        a = types.SimpleNamespace(ledger=d)
        b = types.SimpleNamespace(ledger=K.CardKvLedger(path, "P"))   # the other process on the same card
        self._run(a, [phys] * 4)
        self._run(b, [phys] * 4, t0=1.0)
        st = K.peek(path)
        self.assertEqual(st.free, phys, "the unbooked card use was not booked, or booked twice")
        self.assertEqual(st.budget, 4538 * MIB - 628359168)

    def test_a_single_spike_is_not_booked(self):
        path, d = self._card()
        free = K.peek(path).free
        a = types.SimpleNamespace(ledger=d)
        self._run(a, [free, free - 500 * MIB, free, free - 400 * MIB, free])
        self.assertEqual(K.peek(path).budget, 4538 * MIB)

    def test_books_the_persistent_over_measured_now(self):
        # gmps7 (D 17:53:15): booking the window MINIMUM (209 MB of 209/477/466)
        # left the ledger over-promising 466 MB -> PP0 cuMemCreate OOM. Persistence
        # decides WHETHER to book; the gap measured now is WHAT is booked.
        path, d = self._card()
        free = K.peek(path).free
        a = types.SimpleNamespace(ledger=d)
        self._run(a, [free - 700 * MIB, free - 600 * MIB, free - 650 * MIB])
        self.assertEqual(K.peek(path).budget, 4538 * MIB - 650 * MIB)


if __name__ == "__main__":
    import unittest

    unittest.main()
