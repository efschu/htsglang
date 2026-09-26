"""--p-attn-head-split (27B, 26.09., release table row 22): attention by heads.

Pinned without a GPU (design: /spinning/gpu-arb/docs/ATTN_HEAD_SPLIT.md, module
weg2/attn_head_split.py):
  * BIT-EQUAL ON A MINI MODEL -- an owner stage whose last layers hand one or
    two kv groups to a helper produces, chunk by chunk, EXACTLY the hidden
    states of the unsplit stage (torch.equal), with and without a KV scale,
    including a chunk that does not split (owner computes all heads) and the
    in-place k/v division of set_kv_buffer (the pack must precede it);
  * RANKS NEVER DISAGREE -- the split rule is a pure function of the forward
    sequence, identical on every rank that sees the same sequence; a rank
    blind to graph forwards WOULD diverge on a rewinding request (pinned), so
    the rule runs in ModelRunner.forward; a payload header that differs from
    the helper's expectation, and an owner that never sends, are named
    crash-stops;
  * THE POST -- one function prices the mirror for the runtime post and the
    planner; the planner charges exactly that vector, the runtime post name is
    in RUNTIME_BUDGET_POSTS;
  * OFF IS IDENTICAL -- no env: no config, no post, no runtime, no launcher
    env or line; the model's hook attributes default to False.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from collections import defaultdict, deque
from types import SimpleNamespace

import torch

from sglang.srt.weg2 import attn_head_split as AH
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

FP8 = torch.float8_e4m3fn
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
SRT = os.path.join(REPO, "python", "sglang", "srt")


# ---------------------------------------------------------------------------
# fakes: an in-process barlink pair and a mini owner stage
# ---------------------------------------------------------------------------


class FakeNet:
    def __init__(self):
        self.box = defaultdict(deque)
        self.pub = defaultdict(int)


class FakeWire:
    """send/recv/published/peek_header with barlink's counting: a pair's
    publish counter is the seq of its last published piece (seq starts at 1)."""

    def __init__(self, net, rank, p2p_bytes=1 << 30):
        self.net, self.rank, self.p2p_bytes = net, rank, p2p_bytes
        self.on_empty = None

    def pieces(self, nbytes):
        return max(1, -(-int(nbytes) // self.p2p_bytes))

    def send(self, t, dst):
        self.net.box[(self.rank, dst)].append(t.detach().clone().reshape(-1))
        self.net.pub[(self.rank, dst)] += self.pieces(t.numel() * t.element_size())

    def recv(self, t, src):
        q = self.net.box[(src, self.rank)]
        while not q and self.on_empty is not None and self.on_empty():
            pass
        if not q:
            raise AssertionError(f"FakeWire: nothing from {src} for {self.rank}")
        t.view(-1).copy_(q.popleft())

    def published(self, src):
        return self.net.pub[(src, self.rank)]

    def peek_header(self, src, seq):
        return [int(x) for x in self.net.box[(src, self.rank)][0][: AH.HDR_BF16].view(torch.int64)]


def mini_cfg(n_layers=2, n_groups=1, cap=256, min_w=32):
    return AH.AHConfig(
        delegations=(AH.Delegation(owner=1, helper=0, n_layers=n_layers, n_groups=n_groups),),
        min_w=min_w, cap_tokens=cap, max_w=64, head_dim=16, gqa=2, num_kv_heads=4,
    )


class MiniOwnerStage:
    """Three attention layers of an owner stage: q/k/v/o projections, an fp8
    pool written like set_kv_buffer (IN PLACE division, then cast), and the
    reference paged attention over it."""

    def __init__(self, cfg, attn_layers=(3, 7, 11), hidden=64, k_scale=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.cfg, self.ids, self.hidden = cfg, attn_layers, hidden
        H, Hk, D = cfg.num_kv_heads * cfg.gqa, cfg.num_kv_heads, cfg.head_dim
        self.H, self.Hk, self.D = H, Hk, D
        self.k_scale = k_scale
        self.sm = D ** -0.5
        mk = lambda *s: (torch.randn(*s, generator=g) * 0.2).to(torch.bfloat16)
        self.w = {lid: (mk(hidden, H * D), mk(hidden, Hk * D), mk(hidden, Hk * D), mk(H * D, hidden))
                  for lid in attn_layers}
        self.pool = {lid: (torch.zeros(cfg.cap_tokens, Hk, D, dtype=FP8),
                           torch.zeros(cfg.cap_tokens, Hk, D, dtype=FP8)) for lid in attn_layers}

    def _store(self, lid, k3, v3, p):
        # MHATokenToKVPool.set_kv_buffer: cache_k.div_(k_scale) IN PLACE, then .to(dtype)
        if self.k_scale is not None:
            k3.div_(self.k_scale)
            v3.div_(self.k_scale)
        pk, pv = self.pool[lid]
        pk[p:p + k3.shape[0]] = k3.to(FP8)
        pv[p:p + v3.shape[0]] = v3.to(FP8)

    def own_attention_fn(self, lid, p):
        def fn(q_own, k3, v3, kv_own):
            self._store(lid, k3, v3, p)
            pk, pv = self.pool[lid]
            return AH.reference_attention(q_own, pk[:, :kv_own], pv[:, :kv_own], p, self.sm,
                                          self.k_scale, self.k_scale).reshape(q_own.shape[0], -1)
        return fn

    def forward(self, x, p, row=None, owner=None):
        for lid in self.ids:
            wq, wk, wv, wo = self.w[lid]
            q, k, v = x @ wq, x @ wk, x @ wv
            if row is not None and owner is not None and lid in owner.layers:
                a = owner.side.attention(row, lid, q, k, v, self.own_attention_fn(lid, p))
            else:
                w = x.shape[0]
                q3 = q.view(w, self.H, self.D)
                a = self.own_attention_fn(lid, p)(q3, k.view(w, self.Hk, self.D).clone(),
                                                  v.view(w, self.Hk, self.D).clone(), self.Hk)
            x = x + a @ wo
        return x


def build_pair(cfg, stage, k_scale):
    """Owner (rank 1) + helper (rank 0) wired through one FakeNet."""
    net = FakeNet()
    ow, hw = FakeWire(net, 1), FakeWire(net, 0)
    d = cfg.of_owner(1)
    layers = AH.owner_layers(stage.ids, d.n_layers)
    infos = {(1, lid): AH.LayerInfo(lid, stage.sm, 0.0, k_scale, k_scale) for lid in layers}

    def attend(owner, info, q_d, mk, mv, p, w):
        return AH.reference_attention(q_d, mk, mv, p, info.sm_scale, info.k_scale, info.v_scale)

    engine = AH.HelperEngine(cfg, "cpu", FP8, attend)
    engine.allocate(1, layers, d.n_groups)
    thread = AH.HelperThread(cfg, hw, engine, infos)
    ow.on_empty = lambda: thread.poll_once() > 0
    side = AH.OwnerSide(cfg, d, ow, "cpu")
    return SimpleNamespace(side=side, thread=thread, layers=layers, engine=engine, net=net)


def run_request(cfg, widths, *, split, k_scale=None, seed=1):
    stage = MiniOwnerStage(cfg, k_scale=k_scale)
    g = torch.Generator().manual_seed(seed)
    total = sum(widths)
    x_all = (torch.randn(total, stage.hidden, generator=g)).to(torch.bfloat16)
    rule_o = AH.SplitRule(cfg.min_w, cfg.cap_tokens)
    rule_h = AH.SplitRule(cfg.min_w, cfg.cap_tokens)
    pair = build_pair(cfg, stage, k_scale) if split else None
    outs, rows = [], []
    p = 0
    for w in widths:
        row_h = rule_h.decide(True, ["R"], [p], [w])  # helper (PP0) runs the chunk first
        if pair is not None and row_h is not None:
            pair.thread.push(row_h, {1: pair.layers})
        row = rule_o.decide(True, ["R"], [p], [w])
        rows.append(row)
        outs.append(stage.forward(x_all[p:p + w], p, row if split else None, pair))
        p += w
    return torch.cat(outs), rows, pair


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestBitEqualMiniModel(unittest.TestCase):
    def _check(self, cfg, widths, k_scale=None):
        ref, _, _ = run_request(cfg, widths, split=False, k_scale=k_scale)
        got, rows, pair = run_request(cfg, widths, split=True, k_scale=k_scale)
        self.assertTrue(torch.equal(ref, got), msg=f"max diff {(ref.float() - got.float()).abs().max()}")
        return rows, pair

    def test_one_group_three_chunks(self):
        rows, pair = self._check(mini_cfg(n_layers=2, n_groups=1), [64, 64, 48])
        self.assertTrue(all(r is not None for r in rows))
        self.assertEqual(pair.thread.stats["layers"], 3 * 2)
        self.assertEqual(pair.side.stats["layers"], 3 * 2)
        self.assertEqual(pair.thread.pending(), 0)

    def test_two_groups_all_layers(self):
        rows, pair = self._check(mini_cfg(n_layers=3, n_groups=2), [40, 64, 64])
        self.assertEqual(pair.thread.stats["layers"], 3 * 3)

    def test_with_kv_scale_and_inplace_division(self):
        self._check(mini_cfg(n_layers=2, n_groups=1), [64, 64], k_scale=0.5)

    def test_short_chunk_ends_the_split_owner_computes_all(self):
        rows, pair = self._check(mini_cfg(n_layers=2, n_groups=1), [64, 16, 64])
        self.assertEqual([r is not None for r in rows], [True, False, False])
        self.assertEqual(pair.thread.stats["layers"], 2)

    def test_mirror_holds_the_pool_bytes(self):
        cfg = mini_cfg(n_layers=2, n_groups=1)
        stage = MiniOwnerStage(cfg)
        pair = build_pair(cfg, stage, None)
        rule = AH.SplitRule(cfg.min_w, cfg.cap_tokens)
        x = torch.randn(64, stage.hidden).to(torch.bfloat16)
        row = rule.decide(True, ["R"], [0], [64])
        pair.thread.push(row, {1: pair.layers})
        stage.forward(x, 0, row, pair)
        for lid in pair.layers:
            mk, mv = pair.engine.mirror[(1, lid)]
            pk, pv = stage.pool[lid]
            self.assertTrue(torch.equal(mk[:64].view(torch.uint8), pk[:64, 3:].view(torch.uint8)))
            self.assertTrue(torch.equal(mv[:64].view(torch.uint8), pv[:64, 3:].view(torch.uint8)))


class TestRanksAgree(unittest.TestCase):
    def test_rule_sequence(self):
        r = AH.SplitRule(min_w=1024, cap_tokens=10000)
        self.assertIsNone(r.decide(True, None, None, None))                   # capture
        a = r.decide(True, ["A"], [0], [2048])
        self.assertEqual((a.p, a.w, a.chunk_no), (0, 2048, 0))
        b = r.decide(True, ["A"], [2048], [2048])
        self.assertEqual((b.p, b.chunk_no), (2048, 1))
        self.assertIsNone(r.decide(True, ["A", "B"], [4096, 0], [1024, 1024]))  # bs 2
        self.assertIsNone(r.decide(True, ["A"], [5120], [2048]))              # lost
        c = r.decide(True, ["C"], [0], [4096])
        self.assertEqual(c.rid, "C")
        self.assertIsNone(r.decide(True, ["C"], [4096], [8000]))              # past cap
        self.assertIsNone(r.decide(True, ["C"], [12096], [2048]))
        self.assertIsNone(r.decide(True, ["D"], [300], [2048]))               # warm prefix
        self.assertIsNone(r.decide(False, ["E"], [0], [2048]))                # not plain extend
        e = r.decide(True, ["E"], [0], [2048])
        r.reset()                                                              # KV release
        self.assertIsNone(r.decide(True, ["E"], [2048], [2048]))

    def test_two_ranks_on_one_sequence_decide_the_same(self):
        import random

        rnd = random.Random(7)
        a_rule, b_rule = AH.SplitRule(1024, 1 << 20), AH.SplitRule(1024, 1 << 20)
        pos, n_split = {}, 0
        for _ in range(4000):
            rid = rnd.choice("ABCD")
            bs = 1 if rnd.random() < 0.8 else 2
            w = rnd.choice([256, 512, 1024, 2048, 2048])
            p = pos.get(rid, 0) if rnd.random() < 0.9 else rnd.choice([0, 128])
            pos[rid] = p + w
            rids = [rid] if bs == 1 else [rid, "Z"]
            pre, ext = ([p] if bs == 1 else [p, 0]), ([w] if bs == 1 else [w, 64])
            a = a_rule.decide(True, rids, pre, ext)
            b = b_rule.decide(True, rids, pre, ext)
            self.assertEqual(a, b)
            n_split += a is not None
        self.assertGreater(n_split, 100)

    def test_a_rank_blind_to_graph_forwards_diverges(self):
        """Why the rule runs in ModelRunner.forward (graph replays included)
        and not in the model: a request that rewinds (retract + re-prefill in
        graph-sized chunks) back to exactly the old next_pos would be split by
        a rank that never saw the rewind -- on a mirror it did not refill."""
        seen, blind = AH.SplitRule(1024, 1 << 20), AH.SplitRule(1024, 1 << 20)
        seq = [(0, 1024), (0, 512), (512, 512), (1024, 2048)]
        out = []
        for p, w in seq:
            a = seen.decide(True, ["D"], [p], [w])
            b = None if w <= 512 else blind.decide(True, ["D"], [p], [w])
            out.append((a is not None, b is not None))
        self.assertEqual(out[-1], (False, True))
        src = open(os.path.join(SRT, "model_executor", "model_runner.py")).read()
        body = src[src.index("    def forward(\n        self,\n        forward_batch: ForwardBatch,"):]
        body = body[: body.index("\n    def ", 10)]
        self.assertIn("_ah_rt.on_forward(forward_batch)", body)
        self.assertNotIn("on_forward", open(os.path.join(SRT, "models", "qwen3_5.py")).read())

    def test_header_mismatch_is_a_named_crash_stop(self):
        cfg = mini_cfg()
        stage = MiniOwnerStage(cfg)
        pair = build_pair(cfg, stage, None)
        want = AH.ChunkRow("R", AH.rid_hash("R"), 0, 64, 0)
        pair.thread.push(AH.ChunkRow("R", AH.rid_hash("R"), 0, 48, 0), {1: pair.layers})
        x = torch.randn(64, stage.hidden).to(torch.bfloat16)
        with self.assertRaises(RuntimeError) as cm:
            stage.forward(x, 0, want, pair)
        self.assertIn("RANKS DISAGREE", str(cm.exception))
        self.assertIn("w: got 64 want 48", str(cm.exception))

    def test_an_owner_that_never_sends_is_a_named_stall(self):
        cfg = mini_cfg()
        stage = MiniOwnerStage(cfg)
        pair = build_pair(cfg, stage, None)
        pair.thread.push(AH.ChunkRow("R", 1, 0, 64, 0), {1: pair.layers})
        old = AH.HELPER_STALL_S
        AH.HELPER_STALL_S = 0.0
        try:
            self.assertEqual(pair.thread.poll_once(), 0)
            with self.assertRaises(RuntimeError) as cm:
                import time; time.sleep(0.01)
                pair.thread.poll_once()
        finally:
            AH.HELPER_STALL_S = old
        self.assertIn("helper stall", str(cm.exception))

    def test_kv_release_drops_jobs_and_resets(self):
        rt = AH.AHRuntime(mini_cfg(), pp_rank=0, pp_size=2)
        rt.rule.decide(True, ["R"], [0], [64])
        rt.on_kv_release()
        self.assertIsNone(rt.rule.active)
        self.assertEqual(rt.role, "helper")


class TestSpecAndPost(unittest.TestCase):
    def test_parse_and_refuse(self):
        self.assertEqual(AH.parse_spec("off"), ())
        self.assertEqual(AH.parse_spec("2:0:2:1,1:0:1:1"),
                         (AH.Delegation(2, 0, 2, 1), AH.Delegation(1, 0, 1, 1)))
        V = lambda s, **kw: AH.validate(AH.parse_spec(s), pp_size=3, num_kv_heads=4, **kw)
        V("2:0:2:1,1:0:1:1")
        for bad, word in (("0:1:1:1", "UPSTREAM"), ("1:0:1:4", "n_groups"), ("1:0:0:1", "n_layers"),
                          ("2:1:1:1,1:0:1:1", "owner AND helper"), ("1:0:1:1,1:0:1:1", "twice"),
                          ("3:0:1:1", "outside")):
            with self.assertRaises(AH.AHSpecError) as cm:
                V(bad)
            self.assertIn(word, str(cm.exception))
        with self.assertRaises(AH.AHSpecError):
            V("1:0:5:1", owner_attn_layers={1: 3, 2: 3})
        with self.assertRaises(AH.AHSpecError):
            AH.parse_spec("1:0:1")

    def test_env_round_trip(self):
        cfg = AH.AHConfig(AH.parse_spec("2:0:2:1,1:0:1:1"), 1024, 131072, 2048)
        self.assertEqual(AH.AHConfig.from_env(cfg.to_env()), cfg)
        self.assertEqual(AH.config_from_env({AH.ENV: cfg.to_env()}), cfg)

    def test_post_is_the_mirror_plus_named_terms(self):
        cfg = AH.AHConfig(AH.parse_spec("2:0:2:1,1:0:1:1"), 1024, 262144, 2048)
        mirror = 3 * 2 * 256 * 262144 / 2**20  # 3 group-layers, K+V, fp8
        self.assertEqual(mirror, 384.0)
        pay, out = AH.message_bytes(1, 2048, 256, 6)
        self.assertEqual(pay, (32 + 2048 * 8 * 256) * 2)
        helper = AH.stage_post_mib(cfg, 0)
        self.assertAlmostEqual(helper, 384.0 + AH.HELPER_FLOAT_WS_MIB + 2 * AH.INT_WS_MIB + 2 * (pay + out) / 2**20)
        owner = AH.stage_post_mib(cfg, 1)
        self.assertAlmostEqual(owner, AH.INT_WS_MIB + (pay + out + 2048 * 18 * 256 * 2) / 2**20)
        self.assertEqual(AH.stage_post_vector(None, 3), ())
        self.assertEqual(len(AH.stage_post_vector(cfg, 3)), 3)
        self.assertGreaterEqual(cfg.p2p_bytes(), max(pay, out))
        self.assertEqual(cfg.p2p_bytes() % 4096, 0)

    def test_payload_round_trip(self):
        w, G, D, gs = 5, 2, 16, 3
        q = torch.randn(w, gs * G, D).to(torch.bfloat16)
        k = torch.randn(w, G, D).to(torch.bfloat16)
        v = torch.randn(w, G, D).to(torch.bfloat16)
        hdr = torch.tensor([AH.MAGIC, 7, 9, 0, w, 0, G, 0], dtype=torch.int64)
        buf = AH.pack_payload(hdr, q, k, v)
        self.assertEqual(buf.numel() * 2, AH.message_bytes(G, w, D, gs)[0])
        h2, q2, k2, v2 = AH.unpack_payload(buf, w, G, D, gs)
        for a, b in ((hdr, h2), (q, q2), (k, k2), (v, v2)):
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(b.is_contiguous())


class TestPlannerPost(unittest.TestCase):
    def _model(self, **kw):
        from sglang.srt.planner.pp_cut import PhasePoolModel

        return PhasePoolModel(free_mib=(28000.0, 16000.0, 16000.0), weight_mib_per_layer=400.0,
                              kv_mib_per_token_per_attn_layer=2048 / 2**20,
                              arming_floor_mib=(1229.0, 1229.0, 1229.0), **kw)

    def test_the_planner_charges_exactly_the_vector(self):
        from sglang.srt.planner.pp_cut import _stage_free_after_residency

        cfg = AH.AHConfig(AH.parse_spec("2:0:2:1,1:0:1:1"), 1024, 262144, 2048)
        vec = AH.stage_post_vector(cfg, 3)
        counts, attn = (42, 11, 11), (11, 3, 2)
        off = _stage_free_after_residency(counts, attn, self._model())
        on = _stage_free_after_residency(counts, attn, self._model(attn_head_split_mib=vec))
        for a, b, c in zip(off, on, vec):
            self.assertAlmostEqual(a - b, c)
        with self.assertRaises(ValueError):
            _stage_free_after_residency(counts, attn, self._model(attn_head_split_mib=(1.0,)))

    def test_the_runtime_post_is_named_in_the_model(self):
        from sglang.srt.planner.pp_cut import PhasePoolModel

        names = {n for n, _ in PhasePoolModel.RUNTIME_BUDGET_POSTS}
        self.assertIn(AH.POST_NAME, names)
        src = open(os.path.join(SRT, "model_executor", "model_runner_kv_cache_mixin.py")).read()
        self.assertIn(f'budget_posts.append(("{AH.POST_NAME}"', src)


class TestOffIsIdentical(unittest.TestCase):
    def test_no_env_no_config_no_runtime(self):
        self.assertIsNone(AH.config_from_env({}))
        self.assertIsNone(AH.config_from_env({AH.ENV: "off"}))
        AH.reset_for_tests()
        self.assertIsNone(AH.runtime())
        self.assertFalse(AH.owner_split_now(3))
        old = os.environ.pop(AH.ENV, None)
        try:
            self.assertIsNone(AH.install(SimpleNamespace(is_draft_worker=False)))
            from sglang.srt.model_executor.model_runner_kv_cache_mixin import _attn_head_split_post_mib

            self.assertEqual(_attn_head_split_post_mib(SimpleNamespace(pp_rank=0)), 0.0)
        finally:
            if old is not None:
                os.environ[AH.ENV] = old

    def test_model_hooks_default_off(self):
        src = open(os.path.join(SRT, "models", "qwen3_5.py")).read()
        self.assertEqual(len(re.findall(r"^\s+_ah_split = False$", src, re.M)), 1)
        # the stock call is still the else-branch, unchanged
        self.assertIn("            attn_output = self.attn(q, k, v, forward_batch)\n", src)


class TestLauncher(unittest.TestCase):
    def setUp(self):
        from sglang.srt.weg2 import launcher as L

        self.L = L
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "config.json"), "w") as f:
            json.dump({"text_config": {"num_attention_heads": 24, "num_key_value_heads": 4,
                                       "head_dim": 256, "hidden_size": 5120}}, f)

    def _ns(self, *extra):
        return self.L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--model", self.tmp] + list(extra))

    def test_default_off(self):
        ns = self._ns()
        self.assertEqual(ns.p_attn_head_split, "off")
        cfg = self.L.p_attn_head_split_cfg(ns, 3, 2048, ns.model)
        self.assertIsNone(cfg)
        self.assertEqual(self.L.p_attn_head_split_env(cfg), {})
        self.assertEqual(self.L._p_ah_post_vector(ns, 3, 2048, ns.model), ())

    def test_on_builds_the_one_config(self):
        ns = self._ns("--p-attn-head-split", "2:0:2:1,1:0:1:1", "--max-kv-per-request", "131072")
        cfg = self.L.p_attn_head_split_cfg(ns, 3, 2048, ns.model)
        self.assertEqual((cfg.gqa, cfg.num_kv_heads, cfg.head_dim, cfg.cap_tokens, cfg.min_w),
                         (6, 4, 256, 131072, 1024))
        env = self.L.p_attn_head_split_env(cfg)
        self.assertEqual(AH.config_from_env(env), cfg)
        self.assertEqual(self.L._p_ah_post_vector(ns, 3, 2048, ns.model), AH.stage_post_vector(cfg, 3))
        self.assertIn("P-ATTN-HEAD-SPLIT", self.L.p_attn_head_split_line(cfg, 3))

    def test_refusals(self):
        for extra, word in ((("--p-attn-head-split", "0:1:1:1"), "UPSTREAM"),
                            (("--p-attn-head-split", "1:0:1:1", "--p-attn-head-split-min-w", "4096"), "never fire")):
            with self.assertRaises(SystemExit) as cm:
                self.L.p_attn_head_split_cfg(self._ns(*extra), 3, 2048, self.tmp)
            self.assertIn(word, str(cm.exception.code))


if __name__ == "__main__":
    unittest.main()
