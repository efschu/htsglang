"""formhp-qguard-1028: the DFLASH q guard (selector_q_guard_scatter) as ONE fused Triton launch per round.

py-spy formb165 TP0 bs1 (21:08:13-56Z, 6455 scheduler-thread samples, DSTEP wall 21.44 ms a round): _selector_sampling_accept
13.8 % = 2.97 ms a round (DSTEP a_acc 2.88 ms), of it selector_q_guard_scatter 8.2 % = 1.77 ms -- ~50 eager torch ops
(isfinite, where, sort, two scatters, six counter sums, stack ...), each a dispatch under the GIL, no host sync.

Pinned (CPU; the kernel under TRITON_INTERPRET=1 in a subprocess):
  * repro: with the switch on the guard issues at most one eager op outside the kernel launch (the mask) -- before ~50;
  * fused == eager, bit for bit (dense rows, fallback mask, the six counters) over every fallback cause: non-finite (inf,
    nan), negative, off-sum, token missing / twice / q(x) = 0, duplicate ids, K = 1 / 5 / 8, int32 ids, bf16 q, the
    drafted tokens as the non-contiguous view candidates[:, 1:], a slice of a larger buffer, no counters;
  * FLLIPER_FORMHP_QGUARD_FUSED=0 = the eager guard.
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="base-a-test-cpu")

import ast
import os
import subprocess
import sys
import textwrap

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from flliper.srt.speculative import dflash_utils as du

_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python"))


def _case(B=2, G=3, K=8, V=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(V, generator=g)[:K] for _ in range(B * G)]).view(B, G, K).to(torch.int32)
    q = torch.rand(B, G, K, generator=g)
    q = q / q.sum(-1, keepdim=True)
    cand = torch.zeros(B, G + 1, dtype=torch.int64)
    cand[:, 1:] = ids[..., 0].to(torch.int64)
    buf = torch.zeros(B + 1, G, V)
    return buf[:B], ids, q, cand[:, 1:]


class _Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def test_repro_formb165_the_guard_is_one_launch_not_fifty_eager_ops(monkeypatch):
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    monkeypatch.setenv("FLLIPER_FORMHP_QGUARD_FUSED", "1")
    monkeypatch.setattr(du, "_triton_interpreter_usable", lambda: True)  # late env in this process; the launch is stubbed anyway
    launches = []

    class _Stub:
        def __getitem__(self, grid):
            return lambda *a, **kw: launches.append((grid, kw))

    monkeypatch.setattr(du, "_qguard_kernel", lambda: _Stub())
    dp, ids, q, x = _case()
    stats = torch.zeros(len(du.SELECTOR_Q_GUARD_STATS), dtype=torch.int64)
    with _Count() as c:
        du.selector_q_guard_scatter(draft_probs=dp, candidate_ids=ids, q_rows=q, draft_tokens=x, stats=stats)
    assert len(launches) == 1 and launches[0][0] == (dp.shape[0] * dp.shape[1],), launches
    assert len(c.ops) <= 1, c.ops  # before: ~50 eager ops a round on the scheduler thread


def test_switch_off_is_the_eager_guard(monkeypatch):
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    monkeypatch.setenv("FLLIPER_FORMHP_QGUARD_FUSED", "0")
    monkeypatch.setattr(du, "_qguard_kernel", lambda: (_ for _ in ()).throw(AssertionError("fused path taken")))
    dp, ids, q, x = _case()
    with _Count() as c:
        du.selector_q_guard_scatter(draft_probs=dp, candidate_ids=ids, q_rows=q, draft_tokens=x)
    assert len(c.ops) > 10


_SUB = textwrap.dedent(
    """
    import torch
    from flliper.srt.speculative import dflash_utils as du
    def case(B, G, K, V, seed, qdt):
        g = torch.Generator().manual_seed(seed)
        ids = torch.stack([torch.randperm(V, generator=g)[:K] for _ in range(B * G)]).view(B, G, K).to(torch.int32)
        q = torch.rand(B, G, K, generator=g)
        q = (q / q.sum(-1, keepdim=True))
        cand = torch.zeros(B, G + 1, dtype=torch.int64)
        pick = torch.randint(0, K, (B, G), generator=g)
        cand[:, 1:] = torch.gather(ids.to(torch.int64), -1, pick.unsqueeze(-1)).squeeze(-1)
        s = 0
        def slot():
            nonlocal s
            s += 1
            return divmod(s - 1, G)
        if B * G >= 9 and K >= 2:
            b, j = slot(); q[b, j, 0] = float("inf")                       # non-finite
            b, j = slot(); q[b, j, K - 1] = float("nan")                   # non-finite
            b, j = slot(); q[b, j, 0] = -q[b, j, 0]                        # negative (and off-sum)
            b, j = slot(); q[b, j] = q[b, j] * 1.5                         # off-sum
            b, j = slot(); cand[b, 1 + j] = V + 3 if False else int(ids[b, j].max()) + V  # token missing
            b, j = slot(); ids[b, j, 1] = ids[b, j, 0]; cand[b, 1 + j] = int(ids[b, j, 0])  # token twice = duplicate
            b, j = slot(); m = (ids[b, j].to(torch.int64) == cand[b, 1 + j]); q[b, j][m] = 0.0; q[b, j] = q[b, j] / q[b, j].sum()  # q(x) = 0
            b, j = slot(); ids[b, j, K - 1] = ids[b, j, 0] if int(cand[b, 1 + j]) != int(ids[b, j, 0]) else ids[b, j, 1]  # duplicate
        q = q.to(qdt)
        return ids, q, cand, V
    res = []
    for (B, G, K, V, seed, qdt, with_stats) in [(3, 4, 8, 64, 1, torch.float32, True), (3, 4, 5, 64, 2, torch.bfloat16, True),
                                                (2, 7, 1, 32, 3, torch.float32, True), (4, 3, 8, 128, 4, torch.float32, False),
                                                (1, 2, 8, 64, 5, torch.float32, True)]:
        ids, q, cand, Vv = case(B, G, K, V, seed, qdt)
        V2 = Vv + int(cand.max()) + 1
        out = []
        for fused in (False, True):
            buf = torch.zeros(B + 2, G, V2)
            dp = buf[:B]
            st = torch.zeros(len(du.SELECTOR_Q_GUARD_STATS), dtype=torch.int64) if with_stats else None
            if fused:
                assert du._qguard_fused_on(dp, ids, q, cand[:, 1:], st)
                fb = du._selector_q_guard_scatter_fused(draft_probs=dp, candidate_ids=ids, q_rows=q, draft_tokens=cand[:, 1:], stats=st)
            else:
                fb = du._selector_q_guard_scatter_torch(draft_probs=dp, candidate_ids=ids, q_rows=q, draft_tokens=cand[:, 1:], stats=st)
            out.append((buf.clone(), fb.clone(), None if st is None else st.clone()))
        (b0, f0, s0), (b1, f1, s1) = out
        res.append(dict(dense=bool(torch.equal(b0, b1)), mask=bool(torch.equal(f0, f1)),
                        stats=(s0 is None and s1 is None) or bool(torch.equal(s0, s1)),
                        fallbacks=int(f0.sum()), counters=None if s0 is None else s0.tolist()))
    print("RES", res)
    """
)


def test_fused_equals_eager_bit_for_bit_over_every_fallback_cause():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=_PY, FLLIPER_FORMHP_QGUARD_FUSED="1")
    p = subprocess.run([sys.executable, "-c", _SUB], env=env, capture_output=True, text=True, timeout=900)
    out = [l for l in p.stdout.splitlines() if l.startswith("RES ")]
    assert out, p.stdout[-2000:] + p.stderr[-3000:]
    res = ast.literal_eval(out[0][4:])
    assert all(r["dense"] and r["mask"] and r["stats"] for r in res), res
    assert res[0]["fallbacks"] >= 8 and res[0]["counters"][2] == 2, res  # every cause was exercised (2 non-finite rows)
