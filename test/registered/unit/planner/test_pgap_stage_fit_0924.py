"""--pp-cut-stage-fit / --pp-cut-depth-profile, the planner half (27B line).

Pinned without a GPU: a synthetic group-P log whose #PGAP gpu_fwd follows a
known line per rank is read back (#969N ADMIT join, prefix reconstruction),
fitted to that line, turned into per-stage layer / attention / per-forward
costs by the card rule, and priced over a depth profile by the solver. The
filters are pinned both ways: a host-PACED forward (launch ~ gpu_fwd AND the
card idle before it) is excluded, a host-BLOCKED one (launch ~ gpu_fwd, card
never idle) is kept; a stall is trimmed; the default solver path is unchanged.
"""

import os
import tempfile
import unittest

from sglang.srt.planner import pgap_stage_fit as F
from sglang.srt.planner import pp_cut as PC
from sglang.srt.planner import pp_cut_launch as PL
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CARDS = ["NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3080", "NVIDIA GeForce RTX 3080"]
# the xsn428 fit, per 512 chunk: a (ms at prefix+256 = 0) and b (ms per 1k)
LINES = [(41.4, 1.4358), (33.65, 1.0389), (37.86, 1.0568)]
CHUNK = 512


def _server_args(counts=(42, 11, 11), attn=(10, 3, 3), chunk=CHUNK):
    return (
        "[2026-09-24 18:27:59] server_args=ServerArgs(model_path='m', "
        "chunked_prefill_size=%d, pp_size=3, pp_stage_ratio=[%s], "
        "pp_attn_stage_ratio=[%s], enable_x=True)\n"
        % (chunk, ", ".join(map(str, counts)), ", ".join(map(str, attn)))
    )


def _pass(rank, fct, ext, rid, gpu, launch=2.0, gap="0.2"):
    return (
        "[2026-09-24 18:30:00 PP%d] #969N ADMIT slot=0 fwd_ct=%d bs=1 extend=%d "
        "input_ids=None rids=[%s]\n"
        "[2026-09-24 18:30:00 PP%d] #PGAP pp_rank=%d fwd=%d tokens=%d gpu_gap_ms=%s "
        "gpu_fwd_ms=%.4f host[plan=1 proxy_recv=0 launch=%.1f fi_plan=0 "
        "deferred_publish=0 output_commit=0 d2h_wait=0 process=1] "
        "plan_parts[anchor=0 publish=0 rowcheck=0 evict_drain=0] overlap=1\n"
        % (rank, fct, ext, rid, rank, rank, fct + 1, ext, gap, gpu, launch)
    )


def _write_log(prompts=(40960, 81920), lines=LINES, extra=None, deep_from=0, deep_b0=None):
    fct = [0, 0, 0]
    out = [_server_args()]
    for i, p in enumerate(prompts):
        rid = ("weg2-%d-%d" % (i, i))[:8]
        prefix = 0
        chunks = []
        while prefix < p - 1:
            ext = min(CHUNK, p - 1 - prefix)
            chunks.append((prefix, ext))
            prefix += ext
        chunks.append((prefix, 1))  # the end anchor
        for pre, ext in chunks:
            for r in range(3):
                a, b = lines[r]
                if r == 0 and deep_b0 is not None and pre >= deep_from:
                    b = deep_b0
                gpu = a + b * (pre + CHUNK / 2) / 1000.0 if ext == CHUNK else 9.0
                out.append(_pass(r, fct[r], ext, rid, gpu))
                fct[r] += 1
    if extra:
        out.extend(extra)
    fd, path = tempfile.mkstemp(suffix=".P.log")
    with os.fdopen(fd, "w") as fh:
        fh.write("".join(out))
    return path


class TestReadAndFit(CustomTestCase):
    def setUp(self):
        self.path = _write_log()

    def tearDown(self):
        os.unlink(self.path)

    def test_reads_the_cut_the_chunk_and_every_pass(self):
        log = F.read_pgap_log(self.path)
        self.assertEqual(log.chunk_tokens, CHUNK)
        self.assertEqual(log.counts, (42, 11, 11))
        self.assertEqual(log.attn, (10, 3, 3))
        self.assertEqual(log.unjoined, 0)
        n_full = (40959 // CHUNK) + (81919 // CHUNK)
        self.assertEqual([len(s) for s in log.samples], [n_full] * 3)
        self.assertEqual(log.samples[0][0].prefix, 0)
        self.assertEqual(log.samples[0][1].prefix, CHUNK)

    def test_fit_recovers_the_lines(self):
        lines = F.fit_rank_lines(F.read_pgap_log(self.path))
        for got, (a, b) in zip(lines, LINES):
            self.assertAlmostEqual(got.a_ms, a, places=3)
            self.assertAlmostEqual(got.b_ms_per_1k, b, places=3)

    def test_card_rule_shares_the_middle_stage_rate(self):
        cost, prov = F.fit_stage_cost(F.read_pgap_log(self.path), CARDS)
        self.assertAlmostEqual(cost.layer_ms[0], 41.4 / 42, places=3)
        self.assertAlmostEqual(cost.layer_ms[1], 33.65 / 11, places=3)
        self.assertAlmostEqual(cost.layer_ms[2], 33.65 / 11, places=3)
        self.assertAlmostEqual(cost.stage_fixed_ms[2], 37.86 - 33.65, places=3)
        self.assertAlmostEqual(cost.attn_ms_per_1k[0], 1.4358 / 10, places=3)
        # reproduces the measured cut at any depth
        for pre in (0, 16384, 131072):
            got = cost.stage_ms((42, 11, 11), (10, 3, 3), pre)
            for g, (a, b) in zip(got, LINES):
                self.assertAlmostEqual(g, a + b * (pre + 256) / 1000.0, places=3)
        self.assertIn("ASSUMPTION", prov)

    def test_chunk_refused_when_missing(self):
        fd, bad = tempfile.mkstemp(suffix=".P.log")
        with os.fdopen(fd, "w") as fh:
            fh.write(_pass(0, 0, 512, "r", 50.0))
        try:
            with self.assertRaises(F.StageFitRefused):
                F.read_pgap_log(bad)
        finally:
            os.unlink(bad)


class TestFilters(CustomTestCase):
    def test_host_paced_is_excluded_blocked_is_kept(self):
        paced = F.PgapSample(prefix=1024, gpu_ms=55.6, launch_ms=57.0, gap_ms=8.8)
        blocked = F.PgapSample(prefix=80000, gpu_ms=157.8, launch_ms=147.0, gap_ms=0.3)
        fast = F.PgapSample(prefix=80000, gpu_ms=117.2, launch_ms=17.0, gap_ms=41.4)
        self.assertTrue(paced.host_paced)
        self.assertFalse(blocked.host_paced)
        self.assertFalse(fast.host_paced)

    def test_a_stall_is_trimmed_not_fitted(self):
        stall = [_pass(0, 100000, 512, "zzzzzzzz", 900.0)]
        path = _write_log(extra=stall)
        try:
            # the stall has no prefix run of its own: prefix 0, gpu 900 ms
            line = F.fit_rank_lines(F.read_pgap_log(path))[0]
            self.assertAlmostEqual(line.a_ms, LINES[0][0], places=3)
            self.assertGreaterEqual(line.trimmed, 1)
        finally:
            os.unlink(path)

    def test_too_few_samples_is_refused(self):
        path = _write_log(prompts=(4096,))
        try:
            with self.assertRaises(F.StageFitRefused):
                F.fit_rank_lines(F.read_pgap_log(path))
        finally:
            os.unlink(path)


class TestPiecewise(CustomTestCase):
    def test_deep_split_log_fits_two_segments(self):
        path = _write_log(deep_from=10240, deep_b0=0.86)
        try:
            log = F.read_pgap_log(path)
            cost, prov = F.fit_stage_cost(log, CARDS, split_from_prefix=10240)
            self.assertEqual(cost.deep_from_prefix, 10240)
            self.assertAlmostEqual(cost.deep_attn_ms_per_1k[0], 0.086, places=3)
            self.assertAlmostEqual(cost.attn_ms_per_1k[0], 0.14358, places=3)
            shallow = cost.stage_ms((42, 11, 11), (10, 3, 3), 4096)[0]
            deep = cost.stage_ms((42, 11, 11), (10, 3, 3), 65536)[0]
            self.assertAlmostEqual(shallow, 41.4 + 1.4358 * 4.352, places=3)
            self.assertAlmostEqual(deep, 41.4 + 0.86 * 65.792, places=3)
            self.assertIn("prefix>=10240", prov)
        finally:
            os.unlink(path)


class TestDepthProfile(CustomTestCase):
    def test_ladder_weights_each_rung_equally(self):
        pts, prov = F.depth_profile("ladder:2048,8192,32768", CHUNK)
        self.assertEqual(len(pts), 4 + 16 + 64)
        self.assertAlmostEqual(sum(w for _, w in pts), 1.0, places=9)
        self.assertAlmostEqual(sum(w for _, w in pts[:4]), 1.0 / 3, places=9)
        self.assertAlmostEqual(sum(w for _, w in pts[4:20]), 1.0 / 3, places=9)
        self.assertAlmostEqual(sum(w for _, w in pts[20:]), 1.0 / 3, places=9)
        self.assertEqual([d for d, _ in pts[:4]], [0.0, 512.0, 1024.0, 1536.0])
        self.assertIn("rungs weighted equally", prov)

    def test_bad_spec_refused(self):
        with self.assertRaises(ValueError):
            F.depth_profile("ladder:", CHUNK)
        with self.assertRaises(ValueError):
            F.depth_profile("fit", CHUNK)
        with self.assertRaises(ValueError):
            F.depth_profile("nonsense", CHUNK)


def _pool_model():
    return PC.PhasePoolModel(
        free_mib=(26040.0, 15776.0, 15496.0), weight_mib_per_layer=363.4,
        kv_mib_per_token_per_attn_layer=2048.0 / PC.MIB, arming_floor_mib=(1229.0,) * 3,
        mamba_mib_per_linear_layer_per_slot=1.5588, mamba_slots=3, page_size=1,
        stage_fixed_mib=(2342.0, 1105.5, 3518.0), activation_reserve_mib=1024.0,
        corridor_holdback_mib=1800.0, prefill_graph_pool_mib=(160.0,) * 3,
        zero_posts_acknowledged=(
            "mamba pre-capture reserve", "speculative intermediate state", "GGUF dequant scratch"))


def _families():
    return tuple(
        PC.LAYER_FAMILY_ATTENTION if i % 4 == 3 else PC.LAYER_FAMILY_LINEAR for i in range(64)
    )


class TestSolverProfile(CustomTestCase):
    def _solve(self, cost, **kw):
        return PL.solve_launch_cut(
            layer_families=_families(),
            incumbent_layers=(32, 18, 14),
            measured_ms_per_layer=(8.10, 35.16, 33.59),
            measured_provenance="test",
            card_names=CARDS,
            pool_model=_pool_model(),
            cap_tokens=262144,
            family_cost=cost,
            per_pair_crossing_ms={(0, 1): 1.03, (1, 2): 1.03, (0, 2): 0.58,
                                  (1, 0): 1.03, (2, 1): 1.03, (2, 0): 0.58},
            enumerate_gapped=False,
            objective="makespan",
            pool_floor=262656,
            **kw,
        )

    def _cost(self):
        path = _write_log()
        try:
            return F.fit_stage_cost(F.read_pgap_log(path), CARDS)[0]
        finally:
            os.unlink(path)

    def test_default_path_is_unchanged(self):
        cost = self._cost()
        a = self._solve(cost, design_prefix_tokens=16384)
        b = self._solve(cost, design_prefix_tokens=16384, depth_profile=None)
        self.assertEqual(a.chosen.layers, b.chosen.layers)
        self.assertEqual(a.chosen.makespan_ms, b.chosen.makespan_ms)

    def test_profile_prices_the_mean_of_the_per_depth_max(self):
        cost = self._cost()
        pts, _ = F.depth_profile("ladder:2048,8192,32768", CHUNK)
        d = self._solve(cost, design_prefix_tokens=0, depth_profile=pts)
        c = d.chosen
        want = sum(w * max(cost.stage_ms(c.layers, c.attn, x)) for x, w in pts)
        self.assertAlmostEqual(c.makespan_ms, want, places=3)
        self.assertEqual(c.depth_tokens, int(round(sum(x * w for x, w in pts))))
        # the 512-measured costs move the ladder's cut off 42,11,11 (desk
        # solve 24.09. on the xsn428 fit: 40,13,11 / 10,3,3 at -3.8 %, with
        # 39,13,12 / 9,4,3 within 0.7 % of it) -- which neighbour wins is the
        # fit's business, not this test's; that 42,11,11 loses by > 3 % is.
        self.assertGreaterEqual(c.pool_tokens, 262656)
        self.assertNotEqual(tuple(c.layers), (42, 11, 11))
        base = sum(w * max(cost.stage_ms((42, 11, 11), (10, 3, 3), x)) for x, w in pts)
        self.assertLess(c.makespan_ms, 0.97 * base)


if __name__ == "__main__":
    unittest.main()
