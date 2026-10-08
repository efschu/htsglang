"""RANK-PROGRESS (30.09.): every scheduler rank publishes a monotone progress
counter in its rankstate directory -- the reader is progress_watch.

Anlass (27B): progress_watch las nur front.served / served_tokens aus state.json.
Die bewegen sich erst am Request-Ende; ein 62k-P-Prefill lief 80 s ohne
served-Zuwachs -> Fehlalarm. Die RankState-Dateien (<G>.tp<t>pp<p>.json) sind
statisch (seq=1); die Zaehler liegen im rankstats-Nachbarn
(<G>.tp<t>pp<p>.rankstats, pdflip/rankstats.py, Timer-Thread, atomar).

Gepinnt:
* rankstats ist ein Instrument und per Default an (1 s Takt), aus per Env;
* der Datensatz traegt ``progress`` = {fwd_ct, tokens_done, prefill_tokens,
  decode_tokens}, tokens_done = prefill + decode, beides kumulativ;
* ein Prefill-CHUNK bewegt ihn (kein Request-Ende noetig);
* der Rundenpfad schreibt nichts (der Timer liest nur).
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from flliper.srt.environ import envs
from flliper.srt.pdflip import rankstats
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _sched():
    mr = SimpleNamespace(prefill_tokens_total=0, gen_tokens_total=0,
                         spec_total_num_accept_tokens=0, spec_total_num_forward_ct=0)
    return SimpleNamespace(forward_ct=0, metrics_reporter=mr, waiting_queue=[],
                           running_batch=SimpleNamespace(reqs=[]))


class TestProgress(unittest.TestCase):
    def tearDown(self):
        if rankstats._CURRENT is not None:
            rankstats._CURRENT.stop()
            rankstats._CURRENT = None

    def test_instrument_on_by_default_one_second(self):
        self.assertTrue(envs.FLLIPER_PDFLIP_ENABLE_RANKSTATS.get())
        self.assertEqual(envs.FLLIPER_PDFLIP_RANKSTATS_PERIOD_S.get(), 1.0)
        self.assertTrue(rankstats.enabled())

    def test_the_progress_block_is_monotone_per_chunk(self):
        s = _sched()
        c0 = rankstats.scheduler_counters(s)["progress"]
        self.assertEqual(c0, {"fwd_ct": 0, "tokens_done": 0, "prefill_tokens": 0, "decode_tokens": 0})
        # a 62k P prefill: 4 chunks, no request end in between
        seen = []
        for chunk in (16384, 16384, 16384, 12928):
            s.forward_ct += 1
            s.metrics_reporter.prefill_tokens_total += chunk
            seen.append(rankstats.scheduler_counters(s)["progress"])
        self.assertEqual([p["fwd_ct"] for p in seen], [1, 2, 3, 4])
        self.assertEqual([p["tokens_done"] for p in seen], [16384, 32768, 49152, 62080])
        s.forward_ct += 1
        s.metrics_reporter.gen_tokens_total += 3
        p = rankstats.scheduler_counters(s)["progress"]
        self.assertEqual((p["tokens_done"], p["decode_tokens"]), (62083, 3))

    def test_the_file_carries_it_next_to_the_rank_state(self):
        d = tempfile.mkdtemp(prefix="rkprog-")
        s = _sched()
        s.forward_ct, s.metrics_reporter.prefill_tokens_total = 3, 49152
        with envs.FLLIPER_PDFLIP_RANK_STATE_DIR.override(d), \
                envs.FLLIPER_PDFLIP_RANKSTATS_PERIOD_S.override(0.2), \
                mock.patch.dict(os.environ, {"FLLIPER_PDFLIP_GROUP": "p"}):
            self.assertIsNotNone(rankstats.maybe_start(s, tp_rank=0, pp_rank=2))
            path = os.path.join(d, "P.tp0pp2.rankstats")
            deadline = time.time() + 5
            while time.time() < deadline and not os.path.exists(path):
                time.sleep(0.05)
            with open(path) as f:
                rec = json.load(f)
        self.assertEqual(rec["progress"], {"fwd_ct": 3, "tokens_done": 49152,
                                           "prefill_tokens": 49152, "decode_tokens": 0})
        self.assertIsInstance(rec["ts"], float)
        self.assertGreaterEqual(rec["seq"], 1)


if __name__ == "__main__":
    unittest.main()
