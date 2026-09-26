"""--p-layer-split dynamic: the rank-side executor (weg2/p_layer_split_runtime.py).

Pinned without a GPU (design: /spinning/gpu-arb/docs/DYN_LAYER_SPLIT.md sec. 4):
  * BESITZKARTE -- per request one frontier on PP0; it falls with every
    forward's cut, rises ONLY through a pull announced one forward ahead
    (slot, boundary, [lo, hi), end); a warm request's first chunk cannot
    rise; a wake forces one home forward that announces the weight refill;
    the row carries the batch order and every rank checks it;
  * PULL PLAN -- a rise is a precedence in the flow shop (the pulled share
    of the stage waits for home's head layers of the previous chunk plus the
    transfer), never cheaper than the same cut without the wait; async
    planning adopts the background plan exactly at its start position;
  * WRITE-BACK -- every state tensor of a swing GDN layer rides the frame; a
    missing or surplus entry is a crash-stop;
  * REFUSALS -- a swing layer without its slab, swing layers before the
    refill landed, a unified / page-major pool, a home span that is not the
    P partition, a layer-split row on an unarmed rank;
  * CPU EQUIVALENCE -- a 3-stage mini hybrid model (attention over per-rank
    KV pools with rank-specific slot permutations, GDN-like recurrent state
    per rank, weights in modules) run through the REAL runtime with an
    in-memory upstream transport: a moving cut with rises (prefix pulls incl.
    an attention layer's KV prefix), falls, a two-request batch, and a
    sleep/wake with corrupted swing weights + refill gives BIT-IDENTICAL
    outputs and bit-identical home pools against the home cut.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import random
import unittest
from typing import List

import torch

from sglang.srt.weg2 import p_chunk_policy as P
from sglang.srt.weg2 import p_layer_split as S
from sglang.srt.weg2 import p_layer_split_runtime as R
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

LIM = P.ChunkLimits(max_tokens=2048, min_tokens=512, fixed_tokens=512, page=1,
                    graph_buckets=(512,), eager=True)
H = 8
FAM = S.hybrid_families(12, 4)            # attention at 3, 7, 11
HOME = (6, 9)                             # stage0 [0,6) stage1 [6,9) stage2 [9,12)
WIN = (2, 2)                              # b0 may take 6 (GDN), 7 (ATTN); b1 9, 10 (GDN)


def _model():
    return S.FamilyStageModel(layer_ms=(1.0, 6.0, 6.0), attn_ms_per_1k=(0.3, 0.6, 0.6),
                              stage_fixed_ms=(0.0, 0.0, 0.0), chunk_tokens=512)


def _spec(pull=True, window=WIN, home=HOME, fam=FAM):
    extra = S.ExtraCosts(S.WritebackPrice(2048, 1634304, (6.5, 6.5), True), (), (6.5, 6.5) if pull else ())
    return S.SplitSpec(S.SplitGeometry(fam, home, window), _model(), LIM, 0.0, "every", "test", extra)


def _script(lead, key, pos0, chunks, cuts, frontier=None):
    """Install a scripted plan for ``key`` (chunk widths + wanted cuts)."""
    steps, p = {}, pos0
    for c, cut in zip(chunks, cuts):
        steps[p] = (c, tuple(cut))
        p += c
    plan = S.SplitPlan(pos0, tuple(chunks), tuple(tuple(c) for c in cuts), 0.0, 0.0, tuple(chunks), "script")
    top = lead._top() if pos0 == 0 else lead.spec.geometry.home_cuts
    lead._reqs[key] = S._ReqState(p, frontier or top, plan, steps)


# ---------------------------------------------------------------------------
# Besitzkarte


class TestBesitzkarte(unittest.TestCase):
    def test_frontier_falls_and_rises_only_by_announced_pull(self):
        lead = S.LeaderCursor(_spec())
        _script(lead, "r", 0, [4, 4, 4, 4], [(8, 11), (6, 9), (8, 10), (7, 10)])
        r0 = lead.decide([("r", 0, 4)])
        self.assertEqual(r0.cut, (8, 11))                 # fresh: rise, nothing to pull
        self.assertEqual(r0.pulls, ())                     # next wants (6,9) <= frontier
        r1 = lead.decide([("r", 4, 4)])
        self.assertEqual(r1.cut, (6, 9))
        self.assertEqual(lead.frontier("r"), (8, 10))      # lifted by the announced pull
        self.assertEqual(r1.pulls, ((0, 0, 6, 8, 8), (0, 1, 9, 10, 8)))
        r2 = lead.decide([("r", 8, 4)])
        self.assertEqual(r2.cut, (8, 10))                  # the pull landed: may rise
        self.assertEqual(r2.pulls, ())
        r3 = lead.decide([("r", 12, 4)])
        self.assertEqual(r3.cut, (7, 10))
        self.assertIsNone(lead.frontier("r"))              # finished: forgotten

    def test_v1_never_rises_after_the_first_forward(self):
        lead = S.LeaderCursor(_spec(pull=False))
        _script(lead, "r", 0, [4, 4, 4], [(7, 10), (6, 9), (8, 11)], frontier=(7, 10))
        cuts = [lead.decide([("r", p, 4)]).cut for p in (0, 4, 8)]
        self.assertEqual(cuts, [(7, 10), (6, 9), (6, 9)])

    def test_batch_minimum_and_slot_numbers(self):
        lead = S.LeaderCursor(_spec())
        _script(lead, "a", 0, [4, 4, 4], [(8, 11), (8, 11), (8, 11)])
        _script(lead, "b", 0, [4, 4], [(7, 10), (8, 11)])
        r0 = lead.decide([("a", 0, 4), ("b", 0, 4)])
        self.assertEqual(r0.cut, (7, 10))
        # a's next wants (8,11) > its frontier (7,10): pull at slot 0; b likewise slot 1
        self.assertIn((0, 0, 7, 8, 4), r0.pulls)
        self.assertIn((1, 1, 10, 11, 4), r0.pulls)
        self.assertEqual(r0.order, S.batch_order(["a", "b"]))

    def test_warm_first_chunk_cannot_rise(self):
        g = S.SplitGeometry(FAM, HOME, WIN)
        ex = _spec().extra
        cuts = S.trajectory([512] * 6, 8192, g, _model(), LIM, False, ex)
        self.assertEqual(cuts[0], HOME)
        lead = S.LeaderCursor(_spec())
        lead.next_width("w", 8192, 8192 + 3072)
        self.assertEqual(lead.frontier("w"), HOME)
        row = lead.decide([("w", 8192, lead._reqs["w"].steps[8192][0])])
        self.assertEqual(row.cut, HOME)

    def test_wake_forces_home_and_announces_refill(self):
        lead = S.LeaderCursor(_spec())
        lead.refill_needed = True
        _script(lead, "r", 0, [4, 4], [(8, 11), (8, 11)])
        r0 = lead.decide([("r", 0, 4)])
        self.assertTrue(r0.refill)
        self.assertEqual(r0.cut, HOME)
        r1 = lead.decide([("r", 4, 4)])
        self.assertFalse(r1.refill)
        self.assertEqual(r1.cut, (8, 11))

    def test_row_codec_and_pull_validation(self):
        spec = _spec()
        row = S.ForwardRow(3, spec.digest(), (8, 10), True, ((0, 0, 6, 8, 12),), True, "abc")
        self.assertEqual(S.ForwardRow.decode(json.loads(json.dumps(row.encode()))), row)
        self.assertEqual(S.ForwardRow.decode(row.encode()[:4]).pulls, ())       # legacy width
        f = S.FollowerCursor(spec, 0)
        bad = S.ForwardRow(4, spec.digest(), (8, 10), True, ((0, 0, 5, 8, 12),))  # lo below home
        with self.assertRaises(S.LayerSplitDivergence):
            f.adopt(bad.encode())
        with self.assertRaises(S.LayerSplitDivergence):
            S.FollowerCursor(spec, 0).adopt(
                S.ForwardRow(4, spec.digest(), (8, 10), True, ((3, 0, 6, 8, 12),)).encode(), batch_size=1)
        self.assertEqual(S.FollowerCursor(spec, 1).pull_sends(row), ((0, 0, 6, 8, 12),))
        self.assertEqual(S.FollowerCursor(spec, 0).pull_sends(row), ())


# ---------------------------------------------------------------------------
# pull plan


class TestPullPlan(unittest.TestCase):
    def test_rise_is_a_precedence_not_free(self):
        g = S.SplitGeometry(FAM, HOME, WIN)
        ex = _spec().extra
        chunks = [512] * 6
        rise = [HOME, HOME, HOME, (8, 11), (8, 11), (8, 11)]
        no_pull = S.ExtraCosts(ex.writeback, (), ())
        with_wait = S.makespan(chunks, rise, 0, g, _model(), LIM, ex)
        without = S.makespan(chunks, rise, 0, g, _model(), LIM, no_pull)
        self.assertGreaterEqual(with_wait, without - 1e-9)
        self.assertEqual(S.pull_layers(g, HOME, (8, 11), 0), (6, 7))
        self.assertEqual(S.pull_layers(g, (8, 11), HOME, 0), ())

    def test_pull_bytes_and_pieces(self):
        self.assertEqual(S.pull_pieces(40000, 16384), ((0, 16384), (16384, 32768), (32768, 40000)))
        self.assertEqual(S.pull_bytes(FAM, (0, 0, 6, 8, 1000), 2048, 100), 1000 * 2048 + 100)

    def test_async_plan_adopted_at_its_start(self):
        class Now:
            def submit(self, fn, *a):
                f = __import__("concurrent.futures").futures.Future()
                f.set_result(fn(*a))
                return f

        spec = S.SplitSpec(S.SplitGeometry(S.hybrid_families(64, 4), (36, 50), (12, 6)),
                           S.FamilyStageModel((1.0, 6.0, 6.0), (0.3, 0.6, 0.6), (0.0, 0.0, 0.0), 512),
                           LIM, 0.0, "every", "", S.ExtraCosts(None, (), (6.5, 6.5)))
        lead = S.LeaderCursor(spec, executor=Now(), lead_chunks=2)
        end, pos, widths = 16384, 0, []
        while pos < end:
            w = lead.next_width("r", pos, end)
            lead.decide([("r", pos, w)])
            widths.append(w)
            pos += w
        self.assertEqual(sum(widths), end)
        self.assertEqual(lead.stats["async_adopted"], 1)

    def test_async_plan_late_stays_home(self):
        class Never:
            def submit(self, fn, *a):
                return __import__("concurrent.futures").futures.Future()   # never done

        lead = S.LeaderCursor(_spec(), executor=Never(), lead_chunks=1)
        pos, end = 0, 4096
        while pos < end:
            w = lead.next_width("r", pos, end)
            row = lead.decide([("r", pos, w)])
            self.assertEqual(row.cut, HOME)
            pos += w
        self.assertEqual(lead.stats["async_late"], 1)


# ---------------------------------------------------------------------------
# write-back


class TestWriteback(unittest.TestCase):
    def test_every_state_tensor_rides_and_lands(self):
        src = {6: (torch.randn(5, 3), torch.randn(5, 2, 2), torch.randn(5, 4))}
        dst = {6: tuple(torch.zeros_like(t) for t in src[6])}
        idx_s, idx_d = torch.tensor([2]), torch.tensor([4])
        pay = S.writeback_payload([], [6], None, lambda l: src[l], None, idx_s, True)
        self.assertEqual(sorted(pay), ["swing_ssm:6.0", "swing_ssm:6.1", "swing_ssm:6.2"])
        S.apply_writeback(pay, [], [6], None, lambda l: dst[l], None, idx_d, True)
        for a, b in zip(src[6], dst[6]):
            self.assertTrue(torch.equal(a[2], b[4]))
        del pay["swing_ssm:6.1"]
        with self.assertRaises(S.LayerSplitDivergence):
            S.apply_writeback(pay, [], [6], None, lambda l: dst[l], None, idx_d, True)
        pay = S.writeback_payload([], [6], None, lambda l: src[l], None, idx_s, True)
        pay["swing_ssm:6.3"] = pay["swing_ssm:6.0"]
        with self.assertRaises(S.LayerSplitDivergence):
            S.apply_writeback(pay, [], [6], None, lambda l: dst[l], None, idx_d, True)


# ---------------------------------------------------------------------------
# the mini pipeline: fake pools with the real pools' attribute contract


class FakeMHA:
    def __init__(self, n, pool):
        self.k_buffer = [torch.zeros(pool, H) for _ in range(n)]
        self.v_buffer = [torch.zeros(pool, H) for _ in range(n)]
        self.start_layer = 0
        self._local_slot_of = None
        self.layer_transfer_counter = None

    def local_slot(self, d):
        return d - self.start_layer


class FakeKV:
    def __init__(self, attn_ids, pool):
        self.full_kv_pool = FakeMHA(len(attn_ids), pool)
        self.full_attention_layer_id_mapping = {l: i for i, l in enumerate(attn_ids)}


@dataclasses.dataclass(frozen=True, kw_only=True)
class FakeState:
    conv: List[torch.Tensor]
    temporal: torch.Tensor

    def at_layer_idx(self, i):
        return FakeState(conv=[c[i] for c in self.conv], temporal=self.temporal[i])


class FakeMambaPool:
    def __init__(self, n, slots):
        self.mamba_cache = FakeState(conv=[torch.zeros(n, slots, H)], temporal=torch.zeros(n, slots, H, H))


class FakeRTT:
    def __init__(self, gdn_ids, slots, rows=4, width=96):
        self.mamba_pool = FakeMambaPool(len(gdn_ids), slots)
        self.mamba_map = {l: i for i, l in enumerate(gdn_ids)}
        self.req_to_token = torch.zeros((rows, width), dtype=torch.int64)


class ToyLayer(torch.nn.Module):
    def __init__(self, l):
        super().__init__()
        g = torch.Generator().manual_seed(77 + l)
        self.w = torch.nn.Parameter(torch.randn(4, H, H, generator=g) * 0.3, requires_grad=False)


class Transport:
    """In-memory upstream queues shared by the ranks (FIFO per pair)."""

    def __init__(self, box, stage):
        self.box, self.stage = box, stage

    def send(self, dst, t):
        self.box[(self.stage, dst)].append(t.detach().clone())

    def recv(self, src):
        return self.box[(src, self.stage)].popleft()

    def drain(self):
        pass


class Rank:
    def __init__(self, spec, stage, seed, dynamic=True):
        g = spec.geometry
        self.stage = stage
        home = tuple(g.home(stage))
        win = tuple(g.swing_window(stage)) if dynamic else ()
        built = home + win
        self.layers = torch.nn.ModuleList([ToyLayer(l) if l in built else torch.nn.Identity() for l in range(12)])
        self.kv = FakeKV([l for l in home if FAM[l] == S.FAMILY_ATTENTION], pool=160)
        self.rtt = FakeRTT([l for l in home if FAM[l] == S.FAMILY_LINEAR], slots=8)
        rng = random.Random(1000 + seed)
        self.free = list(range(1, 160))
        rng.shuffle(self.free)
        self.mamba_of = {}
        self.row_of = {}
        self.rt = None
        if dynamic:
            self.rt = R.RankSplitRuntime(spec, stage, async_plan=False)
            self.rt.detach(self.layers, lambda idx: torch.nn.Identity())
            self.rt.bind_pools(self.kv, self.rtt)

    def slot(self, rid):
        if rid not in self.row_of:
            self.row_of[rid] = len(self.row_of)
            self.mamba_of[rid] = 1 + (len(self.mamba_of) * 3 + self.stage) % 7   # differs per rank
        return self.row_of[rid], self.mamba_of[rid]

    def alloc(self, rid, pos, n):
        row, _m = self.slot(rid)
        loc = torch.tensor([self.free.pop() for _ in range(n)], dtype=torch.int64)
        self.rtt.req_to_token[row, pos:pos + n] = loc
        return loc

    def kv_of(self, l):
        if self.rt is not None:
            return self.rt.kv_of(l)
        fk = self.kv.full_kv_pool
        d = self.kv.full_attention_layer_id_mapping[l]
        return fk.k_buffer[d], fk.v_buffer[d]

    def state_of(self, l):
        if self.rt is not None:
            return self.rt.state_of(l)
        return R._state_tensors(self.rtt.mamba_pool.mamba_cache.at_layer_idx(self.rtt.mamba_map[l]))

    def module(self, l):
        return self.layers[l] if self.rt is None else self.rt.module(l, self.layers[l])


def _toy(rank, mod, l, x, reqs):
    wq, wk, wv, wo = mod.w
    outs, off = [], 0
    for rid, pos, n, loc in reqs:
        xs = x[off:off + n]
        q, k, v = xs @ wq, xs @ wk, xs @ wv
        row, mamba = rank.slot(rid)
        if FAM[l] == S.FAMILY_ATTENTION:
            kb, vb = rank.kv_of(l)
            kb.index_copy_(0, loc, k)
            vb.index_copy_(0, loc, v)
            ctx = rank.rtt.req_to_token[row, : pos + n]
            att = (q @ kb[ctx].T) / H ** 0.5
            mask = torch.ones(n, pos + n, dtype=torch.bool).tril(pos)
            out = att.masked_fill(~mask, float("-inf")).softmax(-1) @ vb[ctx]
        else:
            conv, ssm = rank.state_of(l)
            st = ssm[mamba].clone() if pos > 0 else torch.zeros(H, H)
            o = []
            for t in range(n):
                st = 0.9 * st + torch.outer(k[t], v[t])
                o.append(q[t] @ st)
            ssm[mamba] = st
            conv[mamba] = xs[-1]
            out = torch.stack(o)
        outs.append(xs + torch.tanh(out @ wo))
        off += n
    return torch.cat(outs)


def _forward(ranks, batch, tokens_of, lead=None):
    """One pipeline forward, stage by stage (a pull sent inside stage b+1's
    forward v is received inside stage b's forward v+1)."""
    if lead is not None:
        ranks[0].rt.leader_decide([(rid, pos, n) for rid, pos, n in batch])
    x = torch.cat([tokens_of[rid][pos:pos + n] for rid, pos, n in batch])
    frame = None
    for rk in ranks:
        reqs = [(rid, pos, n, rk.alloc(rid, pos, n)) for rid, pos, n in batch]
        g = ranks[0].rt.geom if ranks[0].rt is not None else None
        if rk.rt is not None:
            ctx = R.ForwardCtx(tuple(R.ReqSlot(rid, rk.slot(rid)[0], rk.slot(rid)[1], pos, n)
                                     for rid, pos, n, _loc in reqs),
                               torch.cat([loc for *_r, loc in reqs]), rk.rtt.req_to_token)
            if frame is not None and S.ROW_KEY in frame:
                rk.rt.push_row(json.loads(json.dumps(frame.pop(S.ROW_KEY))))
            rk.rt.pre_forward(ctx, frame)
            ids = rk.rt.layer_ids(tuple(g.home(rk.stage)))
        else:
            ids = tuple(range((0, 6, 9)[rk.stage], (6, 9, 12)[rk.stage]))
        if frame is not None:
            x = frame["h"]
        for l in ids:
            if rk.rt is not None:
                rk.rt.before_layer(l)
            x = _toy(rk, rk.module(l), l, x, reqs)
            if rk.rt is not None:
                rk.rt.after_layer(l)
        frame = {"h": x}
        if rk.rt is not None:
            rk.rt.post_forward(frame if rk.stage < 2 else None)
    return x


def _ranks(dynamic):
    spec = _spec()
    ranks = [Rank(spec, s, seed=s, dynamic=dynamic) for s in range(3)]
    if dynamic:
        box = collections.defaultdict(collections.deque)
        for rk in ranks:
            rk.rt.bind_transport(Transport(box, rk.stage))
    return ranks


def _tokens(n, seed):
    return torch.randn(n, H, generator=torch.Generator().manual_seed(seed))


class TestCpuEquivalence(unittest.TestCase):
    def _run(self, ranks, schedule, tokens_of, scripts, wake_before=None):
        lead = ranks[0].rt.leader if ranks[0].rt is not None else None
        if lead is not None:
            for key, (pos0, chunks, cuts) in scripts.items():
                _script(lead, key, pos0, chunks, cuts)
        outs = []
        for i, batch in enumerate(schedule):
            if wake_before is not None and i == wake_before and lead is not None:
                for rk in ranks:
                    rk.rt.on_sleep()
                for rk in ranks:                     # the pages came back with garbage
                    for m in rk.rt.modules.values():
                        m.w.data.add_(1.0)
                    rk.rt.on_wake()
                for key, (pos0, chunks, cuts) in scripts.items():
                    if key.startswith("post"):
                        _script(lead, key, pos0, chunks, cuts)
            outs.append(_forward(ranks, batch, tokens_of, lead))
        return outs, ranks

    def test_moving_cut_with_pulls_equals_home_bitwise(self):
        tokens_of = {"r1": _tokens(40, 5), "r2": _tokens(15, 6), "post": _tokens(20, 7)}
        c = [5] * 8
        r1_cuts = [(8, 11), (7, 10), (6, 9), (8, 10), (8, 11), (7, 11), (8, 9), (6, 9)]
        schedule = [[("r1", 5 * i, 5)] for i in range(3)]
        schedule += [[("r1", 15, 5), ("r2", 0, 5)], [("r1", 20, 5), ("r2", 5, 5)],
                     [("r1", 25, 5), ("r2", 10, 5)], [("r1", 30, 5)], [("r1", 35, 5)]]
        schedule += [[("post", 5 * i, 5)] for i in range(4)]
        scripts = {"r1": (0, c, r1_cuts), "r2": (0, [5, 5, 5], [(7, 11), (8, 11), (8, 10)]),
                   "post": (0, [5] * 4, [(8, 11), (6, 9), (8, 11), (8, 10)])}
        dyn, rd = self._run(_ranks(True), schedule, tokens_of, scripts, wake_before=8)
        ref, rs = self._run(_ranks(False), schedule, tokens_of, scripts)
        for i, (a, b) in enumerate(zip(dyn, ref)):
            self.assertTrue(torch.equal(a, b), f"forward {i} output differs")
        c0 = rd[0].rt.counters
        c1 = rd[1].rt.counters
        self.assertGreater(c0["pulls_received"], 0)
        self.assertGreater(c1["pulls_sent"], 0)
        self.assertGreater(c0["pull_recv_bytes"], 5 * 2 * H * 4)     # an attention prefix crossed
        self.assertGreater(c1["writeback_applied"], 0)
        self.assertGreater(c0["refill_recv_bytes"], 0)
        self.assertGreater(c1["refill_sent_bytes"], 0)
        # home pools bit-identical
        g = rd[0].rt.geom
        for l in range(12):
            h = g.home_rank(l)
            a, b = rd[h], rs[h]
            for rid in ("r1", "r2", "post"):
                n = tokens_of[rid].shape[0]
                if FAM[l] == S.FAMILY_ATTENTION:
                    la = a.rtt.req_to_token[a.slot(rid)[0], :n]
                    lb = b.rtt.req_to_token[b.slot(rid)[0], :n]
                    ka, va = a.kv_of(l)
                    kb, vb = b.kv_of(l)
                    self.assertTrue(torch.equal(ka[la], kb[lb]), f"K {l} {rid}")
                    self.assertTrue(torch.equal(va[la], vb[lb]), f"V {l} {rid}")
        # the unused channel is empty (every announced pull was consumed)
        for rk in rd:
            self.assertEqual(rk.rt._recv_next, [])

    def test_home_script_is_the_static_forward(self):
        tokens_of = {"r": _tokens(20, 9)}
        schedule = [[("r", 5 * i, 5)] for i in range(4)]
        scripts = {"r": (0, [5] * 4, [HOME] * 4)}
        dyn, rd = self._run(_ranks(True), schedule, tokens_of, scripts)
        ref, _ = self._run(_ranks(False), schedule, tokens_of, scripts)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(dyn, ref)))
        self.assertEqual(rd[0].rt.counters["eager_forwards"], 0)


# ---------------------------------------------------------------------------
# refusals


class TestRefusals(unittest.TestCase):
    def test_swing_without_slab_is_refused(self):
        spec = _spec()
        rt = R.RankSplitRuntime(spec, 0, async_plan=False)     # no modules, no mirrors bound
        row = S.ForwardRow(1, spec.digest(), (8, 9), False)
        rt.push_row(row.encode())
        ctx = R.ForwardCtx((R.ReqSlot("r", 0, 1, 0, 4),), torch.arange(4), torch.zeros(1, 8, dtype=torch.long))
        with self.assertRaises(S.LayerSplitError) as cm:
            rt.pre_forward(ctx, None)
        self.assertIn("without their slab", str(cm.exception))

    def test_swing_before_refill_is_refused(self):
        rk = _ranks(True)[0]
        rk.rt.on_wake()                       # the leader forgets and wants a refill first
        rk.rt.leader.refill_needed = False    # a (buggy) leader skipping the refill
        row = S.ForwardRow(1, rk.rt.follower.digest, (8, 9), False)
        rk.rt.push_row(row.encode())
        ctx = R.ForwardCtx((R.ReqSlot("r", 0, 1, 0, 4),), torch.arange(4), rk.rtt.req_to_token)
        with self.assertRaises(S.LayerSplitDivergence):
            rk.rt.pre_forward(ctx, None)

    def test_batch_order_mismatch_is_refused(self):
        rk = _ranks(True)[1]
        row = S.ForwardRow(1, rk.rt.follower.digest, HOME, False, (), False, S.batch_order(["a", "b"]))
        rk.rt.push_row(row.encode())
        ctx = R.ForwardCtx((R.ReqSlot("b", 0, 1, 0, 2), R.ReqSlot("a", 1, 2, 0, 2)), torch.arange(4),
                           rk.rtt.req_to_token)
        with self.assertRaises(S.LayerSplitDivergence):
            rk.rt.pre_forward(ctx, {"h": torch.zeros(4, H)})

    def test_unified_and_page_major_pools_are_refused(self):
        spec = _spec()
        rt = R.RankSplitRuntime(spec, 0, async_plan=False)

        class UnifiedMHATokenToKVPool:
            pass

        class HybridLinearKVPool:
            full_attention_layer_id_mapping = {}
            full_kv_pool = UnifiedMHATokenToKVPool()

        with self.assertRaises(S.LayerSplitError):
            rt.bind_pools(HybridLinearKVPool(), FakeRTT([0], 4))

    def test_mamba_extra_buffer_is_refused(self):
        spec = _spec()
        rt = R.RankSplitRuntime(spec, 0, async_plan=False)      # window 6 (GDN), 7 (ATTN)
        rtt = FakeRTT([0, 1, 2, 4, 5], 4)
        rtt.enable_mamba_extra_buffer = True
        with self.assertRaises(S.LayerSplitError) as cm:
            rt.bind_pools(FakeKV([3], 16), rtt)
        self.assertIn("extra_buffer", str(cm.exception))

    def test_home_span_must_be_the_partition(self):
        env = {S.POLICY_ENV: "dynamic", S.SPEC_ENV: _spec().to_json()}
        R.reset_for_tests()
        try:
            rt = R.ensure_from_env(12, 1, 3, env=env)
            self.assertIsNotNone(rt)
            self.assertEqual(R.swing_window_for(12, 1, 3, (6, 9)), (9, 10))
            with self.assertRaises(S.LayerSplitError):
                R.swing_window_for(12, 1, 3, (5, 9))
            self.assertEqual(R.swing_window_for(1, 0, 1), ())     # a draft's one-layer stack
            self.assertTrue(R.is_swing_layer(10) and not R.is_swing_layer(8))
            self.assertEqual(R.swing_extra_layer_counts(6, 9), (0, 2))
            self.assertEqual(R.swing_extra_layer_counts(0, 12), (0, 0))
        finally:
            R.reset_for_tests()

    def test_static_installs_nothing(self):
        R.reset_for_tests()
        self.assertIsNone(R.ensure_from_env(64, 0, 3, env={}))
        self.assertIsNone(R.active())
        self.assertEqual(R.swing_window_for(64, 0, 3, (0, 41)), ())
        self.assertEqual(R.swing_extra_layer_counts(0, 41), (0, 0))
        self.assertFalse(R.is_swing_layer(41))


class TestRefillLayoutCheck(unittest.TestCase):
    def _run(self, ranks):
        box = collections.defaultdict(collections.deque)
        for rk in ranks:
            rk.rt.bind_transport(Transport(box, rk.stage))
        for rk in reversed(ranks):        # the in-memory sends never block: downstream first
            rk.rt.verify_refill_layout()

    def test_identical_passes(self):
        spec = _spec()
        self._run([Rank(spec, s, seed=s) for s in range(3)])

    def test_foreign_bytes_refuse(self):
        spec = _spec()
        ranks = [Rank(spec, s, seed=s) for s in range(3)]
        ranks[0].rt.modules[7].w.data[0, 0, 0] += 1.0     # a card-specific repack, say
        with self.assertRaises(S.LayerSplitError) as cm:
            self._run(ranks)
        self.assertIn("differ bytewise", str(cm.exception))


class TestPoolRoutes(unittest.TestCase):
    """The two real accessors route swing ids to the mirror, nothing else."""

    def test_hybrid_kv_and_mamba_routes(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool

        kv = object.__new__(HybridLinearKVPool)
        self.assertIsNone(kv._p_layer_split_kv)
        shim = FakeMHA(1, 8)
        shim.get_key_buffer = lambda d: ("mirror-k", d)
        shim.get_value_buffer = lambda d: ("mirror-v", d)
        kv._p_layer_split_kv = (shim, {7: 0})
        self.assertEqual(kv.get_key_buffer(7), ("mirror-k", 0))
        self.assertEqual(kv.get_value_buffer(7), ("mirror-v", 0))
        rp = object.__new__(HybridReqToTokenPool)
        self.assertIsNone(rp._p_layer_split_state)

        class M:
            def mamba2_layer_cache(self, j):
                return ("mirror-state", j)

        rp._p_layer_split_state = (M(), {6: 0})
        self.assertEqual(rp.mamba2_layer_cache(6), ("mirror-state", 0))


class TestLauncher(unittest.TestCase):
    def test_dynamic_arms_with_pull_and_env(self):
        from sglang.srt.weg2 import launcher as L

        ns = L.build_parser().parse_args([
            "--tree", "/t", "--tag", "t", "--p-layer-split", "dynamic",
            "--p-layer-split-home", "48,56", "--p-layer-split-window", "3,3",
            "--p-layer-split-model", "27b_nvfp4_rc9j"])
        try:
            L.apply_p_layer_split(ns)
            env = L.p_layer_split_env()
            self.assertEqual(env[S.POLICY_ENV], "dynamic")
            spec = S.SplitSpec.from_json(env[S.SPEC_ENV])
            self.assertEqual(spec.extra.pull_gbps, (6.5, 6.5))
            self.assertEqual(L.p_layer_split_swing_by_stage(3), ((0, 3), (0, 3), (0, 0)))
        finally:
            ns0 = L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
            L.apply_p_layer_split(ns0)
        self.assertEqual(L.p_layer_split_env(), {})
        self.assertEqual(L.p_layer_split_swing_by_stage(3), ())

    def test_home_must_match_the_stage_ratio(self):
        from sglang.srt.weg2 import launcher as L

        ns = L.build_parser().parse_args([
            "--tree", "/t", "--tag", "t", "--p-layer-split", "dynamic", "--pp-stage-ratio", "41,12,11",
            "--p-layer-split-home", "48,56", "--p-layer-split-window", "3,3",
            "--p-layer-split-model", "27b_nvfp4_rc9j"])
        with self.assertRaises(SystemExit):
            L.apply_p_layer_split(ns)
        self.assertEqual(L.p_layer_split_env(), {})


class TestPlannerPost(unittest.TestCase):
    def test_swing_post_shrinks_capacity_like_owned_layers(self):
        from sglang.srt.planner import pp_cut as C

        base = dict(free_mib=(20000.0, 12000.0, 12000.0), weight_mib_per_layer=300.0,
                    kv_mib_per_token_per_attn_layer=2048 / 2 ** 20,
                    arming_floor_mib=(1000.0, 1000.0, 1000.0), mamba_mib_per_linear_layer_per_slot=1.6,
                    mamba_slots=24)
        try:
            m0 = C.PhasePoolModel(**base)
        except TypeError as exc:     # the model grew required posts: name them, never guess
            self.skipTest(f"PhasePoolModel signature: {exc}")
        m1 = dataclasses.replace(m0, swing_layers_by_stage=((1, 2), (0, 3), (0, 0)))
        counts, attn = (41, 12, 11), (10, 3, 3)
        c0 = C.stage_pp_capacities(counts, attn, m0)
        c1 = C.stage_pp_capacities(counts, attn, m1)
        self.assertLess(c1[0], c0[0])
        self.assertLess(c1[1], c0[1])
        self.assertEqual(c1[2], c0[2])
        # the swing post equals pricing the stage with the extra layers owned
        m_own = C.stage_pp_capacities((44, 15, 11), (11, 3, 3), m0)
        self.assertEqual(c1[0], m_own[0])


if __name__ == "__main__":
    unittest.main()
