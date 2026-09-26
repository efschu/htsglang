"""27B line (24.09.): group D's planner prices the pool the runtime builds.

Befund xsn420: the launcher's D-OPERATING-POINTS line said 'world pool 330939'
while D realised [317127, 150862, 166098] = 634087 tokens (installed vector
[32, 15, 17] -> 625280). Three planner terms were wrong, one display pinned a
vector no position ships:

  (a) the SSM state priced fp32 -- the planner read SGLANG_MAMBA_SSM_DTYPE off
      the LAUNCHER's env, never set there (the rank gets bfloat16);
  (b) a flat 2304 MiB/rank overhead instead of the measured posts;
  (c) a 34816 B cell (16 attention + the target's MTP layer) where the DFLASH
      window pool charges the 32768 B target cell + a constant draft reserve;
  (d) decode positions priced 'funded ctx' with a rate-pinned token vector,
      while the runtime installs the capacity-matched one on every position.

Checkpoint-dependent cases skip when the rig's checkpoints are absent.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt import uneven_perf as UP
from sglang.srt.weg2 import launcher as L
from sglang.srt.weg2 import form as F  # UNIFY S7: LineIdentity -> form.CalibrationIdentity

MODEL = L.MODEL_DEFAULT
DRAFT = L.DFLASH_DRAFT_PATH_DEFAULT
HAVE_CKPT = os.path.isfile(os.path.join(MODEL, "config.json")) and os.path.isfile(
    os.path.join(DRAFT, "config.json"))
BUDGETS = [27720, 17232, 17024]


def _dflash():
    L.apply_spec_form(types.SimpleNamespace(
        spec_form="DFLASH", dflash_draft_path=DRAFT, dflash_block=8, dflash_window=2048,
        dflash_produce_on_p="off"))


def _nextn():
    L.apply_spec_form(types.SimpleNamespace(spec_form="NEXTN"))


class TestWindowPoolSlots(unittest.TestCase):
    def test_formula_matches_the_rank(self):
        self.assertEqual(UP.dflash_window_pool_slots(2048, 8, 6), 24673)  # xsn420 D.log
        from sglang.srt.model_executor.pool_configurator import window_pool_draft_slots

        for w, b, m in ((2048, 8, 6), (2048, 8, 1), (1024, 4, 3), (4096, 16, 2)):
            sa = types.SimpleNamespace(speculative_num_draft_tokens=b, max_running_requests=m)
            self.assertEqual(UP.dflash_window_pool_slots(w, b, m), window_pool_draft_slots(sa, w))

    def test_factor_constants_pinned(self):
        from sglang.srt.speculative import dflash_solo_pool as S

        self.assertEqual(UP._WINDOW_POOL_FACTOR_ENV, S.SOLO_POOL_FACTOR_ENV)
        self.assertEqual(UP._WINDOW_POOL_FACTOR_DEFAULT, S.DEFAULT_SOLO_POOL_FACTOR)


class TestLauncherFacts(unittest.TestCase):
    def tearDown(self):
        _nextn()

    def test_plan_carries_group_facts(self):
        _dflash()
        plan = L.d_plan_inputs(MODEL, 3, 6)
        self.assertEqual(plan.mamba_ssm_dtype, "bfloat16")
        self.assertTrue(plan.dflash_window_pool)
        self.assertEqual(plan.speculative_draft_window_size, 2048)
        self.assertIsNone(plan.overhead_mib_by_rank)
        self.assertEqual(L.d_plan_inputs(MODEL, 3, 6, [1.0, 2.0, 3.0]).overhead_mib_by_rank, [1.0, 2.0, 3.0])

    def test_no_window_pool_under_nextn(self):
        _nextn()
        plan = L.d_plan_inputs(MODEL, 3, 6)
        self.assertFalse(plan.dflash_window_pool)
        self.assertIsNone(plan.speculative_draft_window_size)
        self.assertEqual(plan.mamba_ssm_dtype, "bfloat16")


@unittest.skipUnless(HAVE_CKPT, "27B checkpoint / DFlash2 draft not on this box")
class TestPlannerTerms(unittest.TestCase):
    def tearDown(self):
        _nextn()

    def _pcm(self, **plan_kw):
        _dflash()
        plan = L.d_plan_inputs(MODEL, 3, 6)
        for k, v in plan_kw.items():
            plan = __import__("dataclasses").replace(plan, **{k: v})
        return UP.PerfCostModel(plan, list(L._gcd_reduce(BUDGETS)), BUDGETS)

    def test_window_pool_cell_and_reserve(self):
        pcm = self._pcm()
        self.assertEqual(pcm.kv_cell_bytes, 32768)          # D.log 'cell 43008 -> 32768'
        self.assertEqual(pcm.window_pool_reserve_bytes, 24673 * 10240)  # 240 MiB
        legacy = self._pcm(dflash_window_pool=False)
        self.assertEqual(legacy.kv_cell_bytes, 34816)       # unchanged without the fact
        self.assertEqual(legacy.window_pool_reserve_bytes, 0.0)

    def test_ssm_dtype_from_the_plan_not_the_env(self):
        env = os.environ.pop("SGLANG_MAMBA_SSM_DTYPE", None)
        try:
            bf16 = self._pcm().mamba_pool_bytes_for(None)
            fp32 = self._pcm(mamba_ssm_dtype=None).mamba_pool_bytes_for(None)
        finally:
            if env is not None:
                os.environ["SGLANG_MAMBA_SSM_DTYPE"] = env
        # xsn420 D.log: mamba state pool + speculative intermediate state
        # = 1.215 + 1.534 GiB on TP0 -- the bf16 price, to the MiB
        self.assertAlmostEqual(bf16[0] / 2**30, 2.749, places=2)
        self.assertGreater(fp32[0], 1.9 * bf16[0])

    def test_measured_overhead_replaces_the_flat_constant(self):
        from sglang.srt.distributed.utils import partition_units

        pcm = self._pcm()
        mlp = list(partition_units(int(pcm.mlp_units), list(L._gcd_reduce(BUDGETS))))
        p_flat = pcm.predict_capacity(mlp)["p"]
        p_meas = self._pcm(overhead_mib_by_rank=[1024.0, 1024.0, 1024.0]).predict_capacity(mlp)["p"]
        for a, b in zip(p_flat, p_meas):
            # 2304 - 1024 = 1280 MiB more KV per rank = 1280 * 32 tokens at 32 KiB
            self.assertAlmostEqual(b - a, 1280 * 32, delta=1)


def _git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True, text=True).stdout.strip()


@unittest.skipUnless(HAVE_CKPT, "27B checkpoint / DFlash2 draft not on this box")
class TestCalibration(unittest.TestCase):
    """d_overhead_calibration reproduces the boot it was measured on EXACTLY."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        root = self._td.name
        self.repo = os.path.join(root, "repo")
        os.makedirs(self.repo)
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "t@t")
        _git(self.repo, "config", "user.name", "t")
        with open(os.path.join(self.repo, "f"), "w") as f:
            f.write("x")
        _git(self.repo, "add", "f")
        _git(self.repo, "commit", "-qm", "base")
        self.tip = _git(self.repo, "rev-parse", "--short=10", "HEAD")
        self.ev = os.path.join(root, "ev")
        os.makedirs(self.ev)
        stem = os.path.join(self.ev, f"boot_weg2_xsn420t_{self.tip}_0924_123212")
        with open(stem + ".front.log", "w") as f:
            f.write(f"[t] WEG2-LAUNCH group P argv: /v/python -m sglang.launch_server --model-path {MODEL}\n")
            f.write(f"[t] WEG2-LAUNCH group D argv: /v/python -m sglang.launch_server --model-path {MODEL} "
                    f"--rank-gpu-memory-mib {','.join(str(b) for b in BUDGETS)} --rank-tp-ratio auto "
                    f"--max-running-requests 6 --tp-size 3\n")
        self.kv = {0: 10391631872, 1: 4943476736, 2: 5442721792}  # xsn420 D.log:3529-3533
        with open(stem + ".D.log", "w") as f:
            for r, b in self.kv.items():
                f.write(f"[2026-09-24 12:34:10 TP{r}] KV pool sizing: available_bytes={b} (x GiB), "
                        f"cell_size=32768, page_size=1 -> max_total_num_tokens={b // 32768}\n")
        self.line = F.CalibrationIdentity(model=MODEL, evidence_dir=self.ev, form=None,
                                          fields=("checkpoint", "line"), repo=self.repo)
        # UNIFY S7: the 27B planner semantics (early-read facts, the capacity
        # vector on every position) follow the published qwen27b profile.
        self._form_saved = os.environ.get(F.FORM_ENV)
        os.environ[F.FORM_ENV] = F.Weg2Form(arch="dense", experts="none", draft="dflash",
                                          p_draft="none", kv="paged_dcp", flip="family",
                                          vision="off", profile="qwen27b", model="m").env_value()

    def tearDown(self):
        self._td.cleanup()
        _nextn()
        os.environ.pop(F.FORM_ENV, None)
        if self._form_saved is not None:
            os.environ[F.FORM_ENV] = self._form_saved

    def test_reproduces_the_measured_capacities(self):
        _dflash()
        ovh, prov = L.d_overhead_calibration(MODEL, self.line, evidence_dir=self.ev)
        self.assertIsNotNone(ovh, prov)
        self.assertIn("MEASURED on", prov)
        from sglang.srt.distributed.utils import partition_units

        weights = list(L._gcd_reduce(BUDGETS))
        pcm = UP.PerfCostModel(L.d_plan_inputs(MODEL, 3, 6, ovh), weights, BUDGETS)
        mlp = list(partition_units(int(pcm.mlp_units), weights))
        cap = pcm.predict_capacity(mlp, L.d_row_attn_units(pcm, weights))  # the row's call
        want = [self.kv[r] // 32768 for r in range(3)]            # [317127, 150862, 166098]
        for got, w in zip(cap["p"], want):
            self.assertAlmostEqual(got, w, delta=1)
        # the D-OPERATING-POINTS row itself: world pool = the realised sum,
        # funded ctx = the runtime's installed pool after quantisation
        cards = [types.SimpleNamespace(name=n) for n in
                 ("NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3080", "NVIDIA GeForce RTX 3080")]
        rows, _ = L.d_operating_point_rows(cards, BUDGETS, MODEL, 6, overhead_mib_by_rank=ovh)
        maxkv = next(r for r in rows if r.position == "maxkv")
        self.assertAlmostEqual(maxkv.world_pool_tokens, 634087, delta=3)
        self.assertEqual(list(maxkv.attn_token_units), [32, 15, 17])   # D.log:3541
        self.assertEqual(maxkv.funded_ctx_tokens, 625280)               # D.log:3963
        for r in rows:  # every position: the runtime installs the capacity vector
            self.assertGreater(r.funded_ctx_tokens, 600000, r.position)

    def test_refuses_without_a_line_boot(self):
        _dflash()
        other = F.CalibrationIdentity(model=MODEL, evidence_dir=os.path.join(self._td.name, "empty"),
                                      form=None, fields=("checkpoint", "line"), repo=self.repo)
        os.makedirs(other.evidence_dir)
        ovh, why = L.d_overhead_calibration(MODEL, other, evidence_dir=other.evidence_dir)
        self.assertIsNone(ovh)
        self.assertIn("no D log of this line", why)
        self.assertEqual(L.d_overhead_calibration(MODEL, None)[0], None)

    def test_refuses_a_cell_that_does_not_transfer(self):
        _nextn()  # planner cell 34816 vs measured 32768
        ovh, why = L.d_overhead_calibration(MODEL, self.line, evidence_dir=self.ev)
        self.assertIsNone(ovh)
        self.assertIn("would not transfer", why)


if __name__ == "__main__":
    unittest.main()
