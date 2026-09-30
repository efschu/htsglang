# SPDX-License-Identifier: Apache-2.0
"""DUAL-TP3PP3: W64 for group D under --dual-layout is judged against D's own
MEASURED budget posts instead of the family model.

Metal 30.09. (boot a3t5js ...09300606, D TP0 on the 5090): the model prices
D r0 at weights 13362 + mamba 1522 + reserves 2304 MiB, the runtime's own posts
say 12.020 + 1.388 + 0.098 + 1.000 GiB (~14853 MiB) and it sized 79138 KV
tokens where the model predicted 4224 -- a ~2.3 GiB pessimism that caps P's
5090 budget in the dual layout.

DANGER DIRECTIONS guarded here:
* only under dual_layout -- the default (flip) path keeps the model verdict;
* only a D log of the SAME model and the SAME installed weight vector (the
  dual-share D publishes it) counts; anything else leaves the refusal;
* a rank still needs >= the model's minimum tokens on its MEASURED posts --
  the override can refuse too, and says why;
* no measurement -> the model's refusal stands.
"""
from __future__ import annotations

import os
import tempfile
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import dual_w64 as W
from flliper.srt.pdflip import launcher as L
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MODEL = "/m/Qwen3.8-27B-NVFP4-RadixArk"


# the runtime's own posts of boot a3t5js (D TP0 5090, TP1/TP2 3080)
POSTS = ((12.020, 1.388, 0.098, 1.000), (4.527, 0.694, 0.049, 1.000), (4.512, 0.694, 0.049, 1.000))


def _log(tp=(58, 25, 25), model=MODEL, ranks=3, posts=POSTS, cell=32768):
    lines = [f"[2026-09-30 06:06:40] server_args=ServerArgs(model_path='{model}', tokenizer_path='{model}')"]
    for r in range(ranks):
        lines.append(
            f"[2026-09-30 06:07:25 TP{r}] PDFLIP-UNION D ratios published for the dual P stage: "
            f"{{'tp': {list(tp)}, 'families': {{'mlp': [98, 19, 19]}}}} -> /dev/shm/wu-x/d_ratios.json")
    for r in range(ranks):
        a, b, c, d = posts[r]
        lines.append(
            f"[2026-09-30 06:07:35 TP{r}] [world_rank {r}] KV budget posts (GiB): weights + runtime state={a:.3f}, "
            f"mamba state pool={b:.3f}, speculative intermediate state={c:.3f}, prefill activation reserve={d:.3f}, "
            f"GGUF dequant scratch=0.000 | rest=2.650 | measured free=17.764 | unaccounted=+15.114")
        lines.append(f"[2026-09-30 06:07:35 TP{r}] KV pool sizing: available_bytes=2593200128 (2.415 GiB), "
                     f"cell_size={cell}, page_size=1 -> max_total_num_tokens=79138")
    return "\n".join(lines) + "\n"


def _dir(*texts):
    d = tempfile.mkdtemp(prefix="w64dual")
    for i, t in enumerate(texts):
        p = os.path.join(d, f"boot_weg2_t{i}_x_0930.D.log")
        with open(p, "w") as f:
            f.write(t)
        os.utime(p, (time.time() - 100 * (len(texts) - i),) * 2)
    return d


class DualW64Measured(CustomTestCase):
    def test_posts_parsed_and_feasibility_on_measured_posts(self):
        d = _dir(_log())
        m = W.find_dual_d_measurement([d], MODEL, (58, 25, 25))
        self.assertIsNotNone(m)
        need = (12.020 + 1.388 + 0.098 + 1.000) * 1024
        self.assertAlmostEqual(m.posts_mib[0], need, delta=1.0)
        self.assertEqual(m.cell_bytes, 32768)
        v = W.judge(m, [17392, 13072, 13488], min_tokens=4096)
        self.assertTrue(v.feasible, v.line)
        self.assertGreater(v.tokens[0], 70000)
        # a budget below posts + 4096 tokens is refused, by the measurement itself
        tight = int(need + 4096 * 32768 / 2**20) - 5
        self.assertFalse(W.judge(m, [tight, 13072, 13488], min_tokens=4096).feasible)

    def test_other_vector_or_model_does_not_count(self):
        d = _dir(_log(tp=(56, 26, 26)), _log(model="/m/other"))
        self.assertIsNone(W.find_dual_d_measurement([d], MODEL, (58, 25, 25)))
        d2 = _dir(_log(), _log(tp=(56, 26, 26)))  # newest is the wrong vector -> older right one wins
        self.assertIsNotNone(W.find_dual_d_measurement([d2], MODEL, (58, 25, 25)))

    def test_non_dual_log_does_not_count(self):
        txt = "\n".join(l for l in _log().splitlines() if "PDFLIP-UNION D ratios" not in l)
        self.assertIsNone(W.find_dual_d_measurement([_dir(txt)], MODEL, (58, 25, 25)))

    def test_launcher_override_only_under_dual(self):
        d = _dir(_log())
        refusal = ("W64 PdFlipTpOperatingPointInfeasible: position decode-bs1 derives weights [58, 25, 25], "
                   "which PerfCostModel.predict_capacity marks feasible=False ...")
        ok, note = L.dual_w64_override(refusal, [17392, 13072, 13488], MODEL, dual_layout=True,
                                       evidence_dirs=[d])
        self.assertTrue(ok, note)
        self.assertIn("MEASURED", note)
        ok, note = L.dual_w64_override(refusal, [17392, 13072, 13488], MODEL, dual_layout=False,
                                       evidence_dirs=[d])
        self.assertFalse(ok)
        ok, note = L.dual_w64_override(refusal, [17392, 13072, 13488], MODEL, dual_layout=True,
                                       evidence_dirs=[_dir()])
        self.assertFalse(ok)
        self.assertIn("no measured", note)
        # a non-W64 refusal is never overridden
        ok, _ = L.dual_w64_override("W61 PdFlipTpOperatingPointUnpriced: ...", [17392, 13072, 13488], MODEL,
                                    dual_layout=True, evidence_dirs=[d])
        self.assertFalse(ok)

    def test_wiring(self):
        import inspect

        src = inspect.getsource(L.d_tp_ratio_decision)
        self.assertIn("dual_w64_override(", src)
        self.assertEqual(inspect.signature(L.d_tp_ratio_decision).parameters["dual_layout"].default, False)
        main = inspect.getsource(L.main)
        self.assertGreaterEqual(main.count('dual_layout=bool(getattr(ns, "dual_layout", False))'), 3)
