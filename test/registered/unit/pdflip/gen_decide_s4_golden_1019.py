"""Generator of ``decide_s4_compat_golden_1019.json``: run it against the S1 tree (6bba3b5901 phase_policy, BEFORE S2a) --
the golden is what ``set_verdict`` / ``classify_arrival`` answered with section-4 arguments only. Usage:
``PYTHONPATH=<S1 tree>/python python gen_decide_s4_golden_1019.py <out.json>``."""
import json
import random
import sys

from flliper.srt.pdflip import phase_policy as pp

X, HALF, T_HALF = 6000, 0.5, 4.5


def rows():
    rnd = random.Random(1019)
    out = []
    # SOLL section 5 examples 1-8 are in test_pdflip_scheduler_soll_1017; here a dense grid of section-4 calls
    for i in range(600):
        n = rnd.randint(0, 4)
        mem = [(f"r{j}", rnd.choice([100, 2999, 3000, 3500, 5999, 6000, 9000, rnd.randint(1, 12000)]), round(rnd.uniform(0, 5), 2),
                rnd.random() < 0.1) for j in range(n)]
        kw = dict(x_tokens=rnd.choice([4096, 6000, 8000]), half_frac=0.5, t_half_s=T_HALF, now=round(rnd.uniform(0, 12), 2),
                  t_open=rnd.choice([None, 0.0, 1.0]), d_decoding=rnd.random() < 0.6, seats_free=rnd.randint(0, 7),
                  d_prefill_running_uncached=rnd.choice([0, 0, 1500, 4000]), new_since_last=rnd.randint(0, 2))
        sv = pp.set_verdict([pp.SetMember(r, u, t, p) for r, u, t, p in mem], **kw)
        out.append(dict(members=mem, kw=kw, out=[sv.verdict, sv.why, round(sv.left_s, 6), sv.s_tokens, sv.half_x]))
    cls = []
    for _ in range(300):
        u, x, po = rnd.randint(0, 14000), rnd.choice([4096, 6000, 8000]), rnd.random() < 0.1
        cls.append([u, x, po, pp.classify_arrival(u, x, HALF, po)])
    return dict(set_verdict=out, classify=cls)


if __name__ == "__main__":
    json.dump(rows(), open(sys.argv[1], "w"))
