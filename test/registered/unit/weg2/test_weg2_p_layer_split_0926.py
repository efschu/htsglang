"""--p-layer-split (27B, 26.09.): the chunk-to-chunk moving P cut.

Pinned without a GPU (design: /spinning/gpu-arb/docs/DYN_LAYER_SPLIT.md):
  * GEOMETRY -- hybrid layout, home/swing/resident sets, the window and the
    attention cap, every forward's executed sets partition the layers;
  * PLAN -- sums and cut validity, the frontier rule (V1: above home only on
    a fresh first chunk, then never rising), the home plan wins under the
    hysteresis and is always priced, the prediction is the exact flow shop
    of the returned plan, a window of 0 is the static cut, the stage model is
    SG's p_stage_model (and FamilyStageModel prices a home cut exactly like
    p_chunk_policy's StageModel);
  * RANKS NEVER DISAGREE -- PP0's LeaderCursor decides every forward's cut
    (min over the batch, frontier per request), the row crosses a mock wire,
    every FollowerCursor derives the same partition; a missing, foreign,
    replayed or out-of-geometry row is a crash-stop;
  * MIGRATION IS BIT-EXACT -- a toy hybrid pipeline (attention over a per-rank
    KV pool with rank-specific slot permutations, a GDN-like recurrent state
    per rank) run under the static home cut and under a moving cut with the
    mirror + write-back gives bit-identical outputs and bit-identical HOME
    pools; a prefix gather/scatter over two ranks' own req_to_token rows
    moves exactly the request's rows;
  * STATIC IS IDENTICAL -- no env, no launcher env/argv change by default;
    'dynamic' arms group P's env (executor: test_weg2_p_layer_split_exec_0926).
"""

from __future__ import annotations

import json
import os
import random
import unittest

import torch

from sglang.srt.weg2 import p_chunk_policy as P
from sglang.srt.weg2 import p_layer_split as S
from sglang.srt.weg2 import p_stage_model as M
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

DATA = os.path.join(os.path.dirname(M.__file__), "p_stage_model_data")
FAM = S.hybrid_families(64, 4)
LIM = P.ChunkLimits(max_tokens=2048, min_tokens=512, fixed_tokens=512, page=1,
                    graph_buckets=(512,), eager=True)


def _drift_model():
    """A family model with a strong drift: stage 0's GDN/MLP 6x faster than
    stages 1/2 but its attention only 2x (the NVFP4 shape, exaggerated)."""
    return S.FamilyStageModel(layer_ms=(1.0, 6.0, 6.0), attn_ms_per_1k=(0.3, 0.6, 0.6),
                              stage_fixed_ms=(0.0, 0.0, 0.0), chunk_tokens=512)


class TestGeometry(unittest.TestCase):
    def test_hybrid_layout(self):
        self.assertEqual([i for i, f in enumerate(FAM) if f == S.FAMILY_ATTENTION][:4], [3, 7, 11, 15])
        self.assertEqual(sum(1 for f in FAM if f == S.FAMILY_ATTENTION), 16)
        m = M.load_model(os.path.join(DATA, "27b_nvfp4_rc9j.json"))
        S.check_layout(S.SplitGeometry(FAM, (48, 56), (0, 0)), m)

    def test_sets(self):
        g = S.SplitGeometry(FAM, (41, 53), (7, 3))
        self.assertEqual(tuple(g.home(0)), tuple(range(0, 41)))
        self.assertEqual(tuple(g.swing_window(0)), tuple(range(41, 48)))
        self.assertEqual(g.resident(1), tuple(range(41, 53)) + (53, 54, 55))
        self.assertEqual(g.resident(2), tuple(range(53, 64)))
        self.assertEqual(g.swing((45, 54), 0), (41, 42, 43, 44))
        self.assertEqual(g.swing((45, 54), 1), (53,))
        self.assertEqual(g.home_rank(44), 1)
        self.assertEqual(g.counts((41, 53)), ((41, 10), (12, 3), (11, 3)))

    def test_window_and_attention_cap(self):
        g = S.SplitGeometry(FAM, (41, 53), (7, 3), (1, 3))
        self.assertTrue(g.valid((44, 53)))        # 41..43: one attention layer (43)
        self.assertFalse(g.valid((48, 53)))       # 43 and 47: two
        self.assertFalse(g.valid((40, 53)))       # below home: never
        self.assertFalse(g.valid((49, 53)))       # beyond the window
        for c in g.cut_options():
            self.assertTrue(g.valid(c))
        with self.assertRaises(S.LayerSplitError):
            S.SplitGeometry(FAM, (41, 53), (12, 3))   # takes stage 1's whole home span

    def test_executed_sets_partition(self):
        g = S.SplitGeometry(FAM, (41, 53), (7, 3))
        for cut in g.cut_options():
            seen = []
            for s in range(3):
                seen.extend(g.executed(cut, s))
            self.assertEqual(seen, list(range(64)))


class TestPlan(unittest.TestCase):
    def test_window_zero_is_static(self):
        g = S.SplitGeometry(FAM, (42, 53), (0, 0))
        res = S.plan_split(32768, g, _drift_model(), LIM)
        self.assertTrue(all(c == (42, 53) for c in res.cuts))
        self.assertEqual(res.candidate, "home")
        self.assertEqual(sum(res.chunks), 32768)

    def test_family_model_prices_like_p_chunk_policy(self):
        """One model: the home cut priced here == p_chunk_policy's flow shop
        on the StageModels of the same numbers."""
        fm = _drift_model()
        g = S.SplitGeometry(FAM, (42, 53), (0, 0))
        stages = []
        for s, (n, a) in enumerate(g.counts(g.home_cuts)):
            stages.append(P.StageModel(((512, n * fm.layer_ms[s]),), a * fm.attn_ms_per_1k[s] / 512.0))
        chunks = P._fixed(0, 16384, LIM)
        ours = S.makespan(chunks, [g.home_cuts] * len(chunks), 0, g, fm, LIM)
        theirs = P.makespan_ms(chunks, 0, stages, LIM)
        self.assertAlmostEqual(ours, theirs, places=6)

    def test_plan_is_valid_and_exactly_priced(self):
        g = S.SplitGeometry(FAM, (36, 50), (12, 6))
        fm = _drift_model()
        for n in (8192, 32768, 131072):
            res = S.plan_split(n, g, fm, LIM, min_gain=0.0)
            self.assertEqual(sum(res.chunks), n)
            self.assertLessEqual(max(res.chunks), LIM.max_tokens)
            for c in res.cuts:
                self.assertTrue(g.valid(c))
            for a, b in zip(res.cuts, res.cuts[1:]):
                self.assertTrue(S.dominated(b, a), f"cut rose mid-request: {a} -> {b}")
            self.assertAlmostEqual(res.predicted_ms, S.makespan(res.chunks, res.cuts, 0, g, fm, LIM),
                                   places=6)
            self.assertLessEqual(res.predicted_ms, res.home_ms + 1e-9)

    def test_drift_moves_the_cut_down_and_wins(self):
        g = S.SplitGeometry(FAM, (36, 50), (12, 6))
        res = S.plan_split(131072, g, _drift_model(), LIM, min_gain=0.0)
        self.assertGreater(res.gain, 0.02)
        self.assertGreater(res.cuts[0][0], res.cuts[-1][0])   # PP0 sheds layers with depth

    def test_warm_request_stays_home_in_v1(self):
        g = S.SplitGeometry(FAM, (36, 50), (12, 6))
        res = S.plan_split(32768, g, _drift_model(), LIM, start=65536, min_gain=0.0)
        self.assertTrue(all(c == g.home_cuts for c in res.cuts))

    def test_pull_mode_may_rise_and_pays(self):
        g = S.SplitGeometry(FAM, (36, 50), (12, 6))
        wbp = S.WritebackPrice(2048, 1634304, (6.5, 6.5))
        free = S.ExtraCosts(wbp, (), (1e9, 1e9))
        slow = S.ExtraCosts(wbp, (), (1e-4, 1e-4))
        r_free = S.plan_split(32768, g, _drift_model(), LIM, start=65536, min_gain=0.0, extra=free)
        r_slow = S.plan_split(32768, g, _drift_model(), LIM, start=65536, min_gain=0.0, extra=slow)
        self.assertTrue(any(c != g.home_cuts for c in r_free.cuts))
        self.assertTrue(all(c == g.home_cuts for c in r_slow.cuts))

    def test_hysteresis_keeps_home(self):
        g = S.SplitGeometry(FAM, (36, 50), (12, 6))
        res = S.plan_split(131072, g, _drift_model(), LIM, min_gain=0.99)
        self.assertTrue(all(c == g.home_cuts for c in res.cuts))
        self.assertEqual(res.candidate, "home(hysteresis)")

    def test_sg_model_home_never_loses(self):
        """On SG's rc9j models the planner never predicts worse than home."""
        for fmt, home in (("int8", (41, 53)), ("nvfp4", (48, 56))):
            m = M.load_model(os.path.join(DATA, f"27b_{fmt}_rc9j.json"))
            g = S.SplitGeometry(FAM, home, (3, 3))
            ex = S.ExtraCosts(S.WritebackPrice(2048, 1634304, (6.5, 6.5)), (1.5, 3.8, 3.8))
            res = S.plan_split(32768, g, m, LIM, extra=ex)
            self.assertLessEqual(res.predicted_ms, res.home_ms + 1e-9)

    def test_swing_slab_price(self):
        g = S.SplitGeometry(FAM, (41, 53), (7, 3))
        sl = S.swing_slab(g, 0, pool_tokens=271118, kv_bytes_per_token_layer=2048, state_slots=24,
                          state_bytes_per_layer=1634304, weight_bytes_attn=255_000_000,
                          weight_bytes_gdn=267_000_000)
        self.assertEqual((sl.attn_layers, sl.gdn_layers), (2, 5))
        self.assertEqual(sl.kv_bytes, 2 * 271119 * 2048)
        self.assertEqual(sl.state_bytes, 5 * 25 * 1634304)
        self.assertEqual(sl.weight_bytes, 2 * 255_000_000 + 5 * 267_000_000)


class TestSpecEnv(unittest.TestCase):
    def _spec(self, model=None):
        g = S.SplitGeometry(FAM, (41, 53), (7, 3), (2, 1))
        m = model or M.load_model(os.path.join(DATA, "27b_nvfp4_rc9j.json"))
        ex = S.ExtraCosts(S.WritebackPrice(2048, 1634304, (6.5, 6.5)), (1.5, 3.8, 3.8))
        return S.SplitSpec(g, m, LIM, 0.01, "every", "test", ex)

    def test_roundtrip_and_digest(self):
        for model in (None, _drift_model()):
            sp = self._spec(model)
            back = S.SplitSpec.from_json(sp.to_json())
            self.assertEqual(back.to_json(), sp.to_json())
            self.assertEqual(back.digest(), sp.digest())

    def test_env(self):
        self.assertIsNone(S.split_from_env({}))
        self.assertIsNone(S.split_from_env({S.POLICY_ENV: "static"}))
        with self.assertRaises(S.LayerSplitError):
            S.split_from_env({S.POLICY_ENV: "dynamic"})
        with self.assertRaises(S.LayerSplitError):
            S.split_from_env({S.POLICY_ENV: "sideways"})
        sp = self._spec()
        got = S.split_from_env({S.POLICY_ENV: "dynamic", S.SPEC_ENV: sp.to_json()})
        self.assertEqual(got.digest(), sp.digest())


class _Wire:
    """Mock collective: PP0's row reaches every rank as JSON (what the #791
    decision row would carry)."""

    def __init__(self):
        self.sent = []

    def broadcast(self, row: S.ForwardRow):
        raw = json.dumps(row.encode())
        self.sent.append(raw)
        return json.loads(raw)


class TestRanksAgree(unittest.TestCase):
    def _spec(self):
        g = S.SplitGeometry(FAM, (36, 50), (12, 6))
        return S.SplitSpec(g, _drift_model(), LIM, 0.0)

    def test_leader_rows_followers_agree(self):
        spec = self._spec()
        lead = S.LeaderCursor(spec)
        fol = [S.FollowerCursor(spec, s) for s in range(3)]
        wire = _Wire()
        pos, end, cuts = 0, 32768, []
        while pos < end:
            w = lead.next_width("r1", pos, end)
            row = lead.decide([("r1", pos, w)])
            raw = wire.broadcast(row)
            got = [f.adopt(raw) for f in fol]
            parts = [f.executed(r) for f, r in zip(fol, got)]
            self.assertEqual(sum((list(p) for p in parts), []), list(range(64)))
            for s in (1, 2):
                self.assertEqual(fol[s].incoming(got[s]), fol[s - 1].swing(got[s - 1]))
            # per stage: a stage keeps its graph while ITS range is home
            self.assertEqual(fol[0].graph_ok(got[0]), got[0].cut[0] == spec.geometry.home_cuts[0])
            self.assertEqual(fol[2].graph_ok(got[2]), got[2].cut[1] == spec.geometry.home_cuts[1])
            cuts.append(row.cut)
            pos += w
        for a, b in zip(cuts, cuts[1:]):
            self.assertTrue(S.dominated(b, a))
        self.assertNotEqual(cuts[0], spec.geometry.home_cuts)

    def test_batch_takes_the_minimum_and_warm_forces_home(self):
        spec = self._spec()
        lead = S.LeaderCursor(spec)
        w1 = lead.next_width("fresh", 0, 131072)
        want = lead.wanted_cut("fresh", 0)
        self.assertNotEqual(want, spec.geometry.home_cuts)
        lead.next_width("warm", 65536, 131072)     # start > 0: frontier = home
        row = lead.decide([("fresh", 0, w1), ("warm", 65536, 512)])
        self.assertEqual(row.cut, spec.geometry.home_cuts)
        # the fresh request's frontier fell with the batch: it never rises again
        row2 = lead.decide([("fresh", w1, 512)])
        self.assertEqual(row2.cut, spec.geometry.home_cuts)

    def test_divergence_is_a_crash_stop(self):
        spec = self._spec()
        lead = S.LeaderCursor(spec)
        f = S.FollowerCursor(spec, 1)
        w = lead.next_width("r", 0, 8192)
        raw = lead.decide([("r", 0, w)]).encode()
        with self.assertRaises(S.LayerSplitDivergence):
            f.adopt(None)
        foreign = list(raw)
        foreign[1] = "0" * 16
        with self.assertRaises(S.LayerSplitDivergence):
            f.adopt(foreign)
        f.adopt(raw)
        with self.assertRaises(S.LayerSplitDivergence):
            f.adopt(raw)                         # replayed version
        bad = list(raw)
        bad[0] = raw[0] + 1
        bad[2] = [30, 50]                        # below home
        with self.assertRaises(S.LayerSplitDivergence):
            f.adopt(bad)
        with self.assertRaises(S.LayerSplitDivergence):
            f.adopt([1, 2, 3])                   # malformed


# ---------------------------------------------------------------------------
# bit-exact migration on a toy hybrid pipeline


H = 8
TOY_FAM = S.hybrid_families(12, 4)        # attention at 3, 7, 11


class _Rank:
    """One PP rank: its OWN slot permutation (allocator), KV buffers for its
    resident attention layers, GDN state for its resident GDN layers."""

    def __init__(self, stage, resident, pool=96, seed=0):
        rng = random.Random(1000 + seed)
        self.free = list(range(1, pool))
        rng.shuffle(self.free)
        self.stage = stage
        self.req_to_token = torch.zeros((4, 64), dtype=torch.int64)
        self.kv = {l: (torch.zeros(pool, H), torch.zeros(pool, H))
                   for l in resident if TOY_FAM[l] == S.FAMILY_ATTENTION}
        self.state = {l: (torch.zeros(8, H), torch.zeros(8, H, H))
                      for l in resident if TOY_FAM[l] == S.FAMILY_LINEAR}
        self.state_idx = 3 + stage          # the request's mamba index differs per rank

    def alloc(self, row, start, n):
        loc = torch.tensor([self.free.pop() for _ in range(n)], dtype=torch.int64)
        self.req_to_token[row, start:start + n] = loc
        return loc


def _weights(l):
    g = torch.Generator().manual_seed(77 + l)
    return [torch.randn(H, H, generator=g) * 0.3 for _ in range(4)]


def _layer(rank, l, x, row, pos, loc):
    wq, wk, wv, wo = _weights(l)
    q, k, v = x @ wq, x @ wk, x @ wv
    if TOY_FAM[l] == S.FAMILY_ATTENTION:
        kb, vb = rank.kv[l]
        kb.index_copy_(0, loc, k)
        vb.index_copy_(0, loc, v)
        ctx = rank.req_to_token[row, : pos + x.shape[0]]
        K, V = kb[ctx], vb[ctx]
        att = (q @ K.T) / H ** 0.5
        n = x.shape[0]
        mask = torch.ones(n, pos + n, dtype=torch.bool).tril(pos)
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        out = att @ V
    else:
        conv, ssm = rank.state[l]
        st = ssm[rank.state_idx].clone()
        outs = []
        for t in range(x.shape[0]):
            st = 0.9 * st + torch.outer(k[t], v[t])
            outs.append(q[t] @ st)
        ssm[rank.state_idx] = st
        conv[rank.state_idx] = x[-1]
        out = torch.stack(outs)
    return x + torch.tanh(out @ wo)


def _run(cuts_per_chunk, chunks, geom):
    ranks = [_Rank(s, geom.resident(s), seed=s) for s in range(geom.stages)]
    g = torch.Generator().manual_seed(5)
    tokens = torch.randn(sum(chunks), H, generator=g)
    outs, pos = [], 0
    for c, cut in zip(chunks, cuts_per_chunk):
        x = tokens[pos:pos + c]
        payload = {}
        for s, rk in enumerate(ranks):
            loc = rk.alloc(0, pos, c)
            kv_of = lambda l, rk=rk: rk.kv[l]
            st_of = lambda l, rk=rk: rk.state[l]
            idx = torch.tensor([rk.state_idx])
            inc = geom.swing(cut, s - 1) if s > 0 else ()
            S.apply_writeback(payload,
                              [l for l in inc if TOY_FAM[l] == S.FAMILY_ATTENTION],
                              [l for l in inc if TOY_FAM[l] == S.FAMILY_LINEAR],
                              kv_of, st_of, loc, idx, True)
            for l in geom.executed(cut, s):
                x = _layer(rk, l, x, 0, pos, loc)
            sw = geom.swing(cut, s)
            payload = S.writeback_payload(
                [l for l in sw if TOY_FAM[l] == S.FAMILY_ATTENTION],
                [l for l in sw if TOY_FAM[l] == S.FAMILY_LINEAR], kv_of, st_of, loc, idx, True)
        outs.append(x)
        pos += c
    return torch.cat(outs), ranks


class TestMigrationBitExact(unittest.TestCase):
    def test_moving_cut_equals_static_bitwise(self):
        geom = S.SplitGeometry(TOY_FAM, (4, 8), (3, 2))
        chunks = [6, 5, 7, 4, 6]
        static = [geom.home_cuts] * len(chunks)
        moving = [(7, 10), (7, 9), (6, 9), (5, 8), (4, 8)]     # rises at chunk 0, then falls
        for c in moving:
            self.assertTrue(geom.valid(c))
        out_s, r_s = _run(static, chunks, geom)
        out_d, r_d = _run(moving, chunks, geom)
        self.assertTrue(torch.equal(out_s, out_d))
        n = sum(chunks)
        for l in range(12):
            home = geom.home_rank(l)
            a, b = r_s[home], r_d[home]
            if TOY_FAM[l] == S.FAMILY_ATTENTION:
                la, lb = a.req_to_token[0, :n], b.req_to_token[0, :n]
                self.assertTrue(torch.equal(a.kv[l][0][la], b.kv[l][0][lb]), f"K of layer {l}")
                self.assertTrue(torch.equal(a.kv[l][1][la], b.kv[l][1][lb]), f"V of layer {l}")
            else:
                self.assertTrue(torch.equal(a.state[l][1][a.state_idx], b.state[l][1][b.state_idx]),
                                f"state of layer {l}")
                self.assertTrue(torch.equal(a.state[l][0][a.state_idx], b.state[l][0][b.state_idx]))

    def test_missing_writeback_is_a_crash_stop(self):
        rk = _Rank(1, (4, 5, 6, 7))
        loc = rk.alloc(0, 0, 3)
        with self.assertRaises(S.LayerSplitDivergence):
            S.apply_writeback({}, [7], [], lambda l: rk.kv[l], lambda l: rk.state[l], loc,
                              torch.tensor([0]), True)

    def test_prefix_pull_over_own_rows(self):
        """Phase-1 claim 1: indices differ per rank, a gather over the
        sender's req_to_token row and a scatter over the receiver's move
        exactly the request's rows -- no index crosses the wire."""
        a, b = _Rank(0, (3,), seed=1), _Rank(1, (3,), seed=2)
        la, lb = a.alloc(2, 0, 20), b.alloc(2, 0, 20)
        self.assertFalse(torch.equal(la, lb))
        a.kv[3][0].index_copy_(0, la, torch.arange(20 * H, dtype=torch.float32).view(20, H))
        rows = S.gather_rows(a.kv[3][0], S.request_loc(a.req_to_token, 2, 0, 20))
        S.scatter_rows(b.kv[3][0], S.request_loc(b.req_to_token, 2, 0, 20), rows)
        self.assertTrue(torch.equal(b.kv[3][0][lb], a.kv[3][0][la]))
        untouched = torch.ones(b.kv[3][0].shape[0], dtype=torch.bool)
        untouched[lb] = False
        self.assertEqual(float(b.kv[3][0][untouched].abs().sum()), 0.0)


class TestStaticIdentical(unittest.TestCase):
    def test_launcher_default_static(self):
        from sglang.srt.weg2 import launcher as L

        ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        self.assertEqual(ns.p_layer_split, "static")
        L.apply_p_layer_split(ns)
        self.assertEqual(L.p_layer_split_env(), {})

    def test_launcher_dynamic_arms(self):
        from sglang.srt.weg2 import launcher as L

        base = ["--tree", "/t", "--tag", "t", "--p-layer-split", "dynamic"]
        ns = L.build_parser().parse_args(base)
        with self.assertRaises(SystemExit):          # incomplete spec
            L.apply_p_layer_split(ns)
        self.assertEqual(L.p_layer_split_env(), {})
        ns = L.build_parser().parse_args(base + [
            "--p-layer-split-home", "48,56", "--p-layer-split-window", "3,3",
            "--p-layer-split-model", "27b_nvfp4_rc9j"])
        try:
            L.apply_p_layer_split(ns)                # complete spec: armed, env for group P
            lines = "\n".join(L._P_LAYER_SPLIT["lines"])
            self.assertIn("P-LAYER-SPLIT armed", lines)
            self.assertIn("P-LAYER-SPLIT plan", lines)
            self.assertIn("P-LAYER-SPLIT slab", lines)
            self.assertEqual(L.p_layer_split_env()[S.POLICY_ENV], "dynamic")
        finally:
            L.apply_p_layer_split(L.build_parser().parse_args(["--tree", "/t", "--tag", "t"]))
        self.assertEqual(L.p_layer_split_env(), {})


if __name__ == "__main__":
    unittest.main()
