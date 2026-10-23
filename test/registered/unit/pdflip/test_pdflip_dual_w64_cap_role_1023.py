# SPDX-License-Identifier: Apache-2.0
"""W64-DUAL: the MAX_LOGS cap counts only dual-share D logs of the asked model.

Rotation 23.10. (dry run 22:05Z, rc36p and rc31b5 alike): the 27b evidence dir is shared by every 27B container.
26 int4ar/duo D logs of 23.10. were newer than the newest dual-share D log of 22.10. (place 27 by mtime), the
old cut "newest 24 of ALL boot_*.D.log" dropped it, and W64 refused a dual boot that the measurement lifts
(W64-DUAL: no measured dual-share D log ... the model verdict stands).

DANGER DIRECTIONS guarded here:
* foreign boots (other model, same model but flip/duo D) in front of the dual log must NOT displace it;
* the cap still bounds the search: 24 newer dual logs of the model with OTHER weights end it;
* a same-model non-dual log stops at its first "KV pool sizing" (no ratios before it) and is never a match.
"""
from __future__ import annotations

import os
import tempfile
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import dual_w64 as W
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MODEL = "/m/Qwen3.8-27B-NVFP4-RadixArk"
OTHER = "/m/Huihui-Qwen3.8-27B-Abliterated-INT4-W4A16-AutoRound-ct"
POSTS = ((12.020, 1.388, 0.098, 1.000), (4.527, 0.694, 0.049, 1.000), (4.512, 0.694, 0.049, 1.000))


def _log(model=MODEL, dual=True, tp=(58, 25, 25), ranks=3, filler=0):
    lines = [f"[2026-10-22 02:16:20] server_args=ServerArgs(model_path='{model}', tokenizer_path='{model}')"]
    lines += [f"[2026-10-22 02:16:30 TP0] filler {i}" for i in range(filler)]
    if dual:
        for r in range(ranks):
            lines.append(f"[2026-10-22 02:17:07 TP{r}] PDFLIP-UNION D ratios published for the dual P stage: "
                         f"{{'tp': {list(tp)}, 'families': {{'mlp': [98, 19, 19]}}}} -> /dev/shm/wu-x/d_ratios.json")
    for r in range(ranks):
        a, b, c, d = POSTS[r % 3]
        lines.append(f"[2026-10-22 02:17:26 TP{r}] [world_rank {r}] KV budget posts (GiB): weights + runtime state={a:.3f}, "
                     f"mamba state pool={b:.3f}, speculative intermediate state={c:.3f}, prefill activation reserve={d:.3f}, "
                     f"GGUF dequant scratch=0.000 | rest=2.650 | measured free=17.764 | unaccounted=+15.114")
        lines.append(f"[2026-10-22 02:17:26 TP{r}] KV pool sizing: available_bytes=3955292160 (3.684 GiB), "
                     f"cell_size=32768, page_size=1 -> max_total_num_tokens=120000")
    if not dual:   # a flip/duo D may print a ratios-like line much later; it must not turn the log into a dual one
        lines += [f"[2026-10-23 12:53:30 TP0] filler late {i}" for i in range(50)]
        lines.append("[2026-10-23 12:54:00 TP0] PDFLIP-UNION D ratios published for the dual P stage: "
                     "{'tp': [58, 25, 25], 'families': {'mlp': [98, 19, 19]}} -> /dev/shm/wu-y/d_ratios.json")
    return "\n".join(lines) + "\n"


def _dir(texts_oldest_first):
    d = tempfile.mkdtemp(prefix="w64cap")
    now = time.time()
    n = len(texts_oldest_first)
    for i, t in enumerate(texts_oldest_first):
        p = os.path.join(d, f"boot_weg2_t{i:03d}_x_1023.D.log")
        with open(p, "w") as f:
            f.write(t)
        os.utime(p, (now - 60 * (n - i),) * 2)
    return d


class DualW64CapByRole(CustomTestCase):
    def test_thirty_foreign_logs_in_front_do_not_displace_the_dual_log(self):
        # the 23.10. shape: 1 dual log, then 30 newer foreign ones (other model + same-model duo/flip D)
        foreign = [_log(model=OTHER, dual=False) if i % 2 else _log(model=MODEL, dual=False, ranks=2)
                   for i in range(30)]
        d = _dir([_log()] + foreign)
        self.assertGreater(len(foreign), W.MAX_LOGS)   # the old cut over ALL logs would have dropped it
        m = W.find_dual_d_measurement([d], MODEL, (58, 25, 25))
        self.assertIsNotNone(m, "a dual-share D log behind 30 foreign logs must still be found")
        self.assertEqual(m.weights, (58, 25, 25))
        self.assertEqual(m.source, "boot_weg2_t000_x_1023.D.log")

    def test_cap_still_bounds_dual_logs_of_other_weights(self):
        others = [_log(tp=(50, 25, 25)) for _ in range(W.MAX_LOGS)]
        d = _dir([_log()] + others)
        self.assertIsNone(W.find_dual_d_measurement([d], MODEL, (58, 25, 25)))
        d2 = _dir([_log()] + others[:-1])   # one fewer: the matching one is inside the cap again
        self.assertIsNotNone(W.find_dual_d_measurement([d2], MODEL, (58, 25, 25)))

    def test_same_model_non_dual_log_stops_at_kv_sizing_and_never_matches(self):
        d = _dir([_log(model=MODEL, dual=False)])
        self.assertIsNone(W._scan(os.path.join(d, "boot_weg2_t000_x_1023.D.log"), MODEL, 3))
        self.assertIsNone(W.find_dual_d_measurement([d], MODEL, (58, 25, 25)))

    def test_dual_log_with_long_preamble_is_still_read(self):
        d = _dir([_log(filler=6000)])
        self.assertIsNotNone(W.find_dual_d_measurement([d], MODEL, (58, 25, 25)))

    def test_other_model_dual_log_does_not_count(self):
        d = _dir([_log(model=OTHER)])
        self.assertIsNone(W.find_dual_d_measurement([d], MODEL, (58, 25, 25)))


if __name__ == "__main__":
    import unittest
    unittest.main()
