"""EZ-S1 review Auflage 8: LOG REPLAY of the arrival classes (plan R.6, R-1 and R-2).

    python desk_replay_ez_s1_1019.py <front.log> [<front.log> ...]

Reads the front log(s) of a boot and replays, WITHOUT any socket, the class of every arrival the log names, through the new
ladder (``decision_numbers.class_basis`` + ``phase_policy.classify_arrival``):

* R-2 (rc15 ``..._071429``, priced front, live probe): every ``PDFLIP SET-VERDICT ... members=rid:u ... X= halfX= classes=``
  line carries the number the old code classified on. The replay forms the class again from the SAME number as a FRESH exact
  ``u_d`` (the normal path) and compares it with the logged class: the normal path must be unchanged (0 differences).
* R-1 (rc16 ``..._121629``, every probe dead): every ``PDFLIP D-PROBE-UNPRICED rid= n= n_src= p_uncached=`` line is an arrival
  WITHOUT an exact number. The replay runs the ladder on ``p_uncached`` (``u_p`` only downwards, media without D ids no
  bound) and prints the census: MEASURE vs kurz-on-u_p, and how many the OLD code classified on ``u_p`` / the estimate as
  halb or big (the 140-474 class, the defect list of S1).
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import decision_numbers as dn  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402

SV = re.compile(r"PDFLIP SET-VERDICT epoch=\d+ set=\d+ n=\d+ members=(?P<m>\S+) S=\d+ X=(?P<x>\d+) halfX=(?P<hx>\d+) "
                r"classes=(?P<c>\S+) ")
UP = re.compile(r"PDFLIP D-PROBE-UNPRICED rid=(?P<rid>\S+) n=(?P<n>\d+) n_src=(?P<src>\S+) p_uncached=(?P<up>\S+)")
MEM = re.compile(r"(?P<rid>[^:,]+):(?P<u>\d+)")


def replay_priced(path):
    diff, n, cls = [], 0, collections.Counter()
    for line in open(path, errors="replace"):
        m = SV.search(line)
        if not m:
            continue
        x, hx = int(m["x"]), int(m["hx"])
        mem = [(a["rid"], int(a["u"])) for a in MEM.finditer(m["m"])]
        old = m["c"].split(",")
        if len(mem) != len(old):
            continue
        for (rid, u), oc in zip(mem, old):
            n += 1
            b = dn.class_basis(u_d=dn.Exact(u, "d_probe", 0.0), u_p=None, n_from_d_ids=True, x_tokens=x, half_x=hx, now=0.0)
            new = b.klass
            cls[new] += 1
            if new != oc and not (oc == "p_only" and new in (dn.CLASS_KURZ, dn.CLASS_HALB, dn.CLASS_GROSS)):
                diff.append((rid, u, x, oc, new))
    return n, cls, diff


def replay_unpriced(path):
    census, old_up_halb_or_big, last_x = collections.Counter(), 0, 4096
    rows = 0
    for line in open(path, errors="replace"):
        m = SV.search(line)
        if m:
            last_x = int(m["x"])
        u = UP.search(line)
        if not u:
            continue
        rows += 1
        x = last_x
        hx = pp.half_x_of(x, 0.5)
        up = -1 if u["up"] == "unknown" else int(u["up"])
        text_ids = u["src"] == "front_ids"
        b = dn.class_basis(u_d=None, u_p=dn.exact_of(up if up >= 0 else None, dn.SRC_P_STORE), n_from_d_ids=text_ids,
                           x_tokens=x, half_x=hx, now=0.0)
        census[b.kind] += 1
        if up >= hx:
            old_up_halb_or_big += 1      # the old code took the upper bound u_p as the class number
    return rows, census, old_up_halb_or_big


def main(paths):
    for p in paths:
        print("==", os.path.basename(p))
        n, cls, diff = replay_priced(p)
        print("  priced SET-VERDICT members replayed: %d  classes(new)=%s  differences vs the logged class: %d"
              % (n, dict(cls), len(diff)))
        for d in diff[:10]:
            print("   DIFF rid=%s u=%d X=%d logged=%s new=%s" % d)
        rows, census, oldbig = replay_unpriced(p)
        print("  D-PROBE-UNPRICED arrivals: %d  new ladder=%s  of them with u_p >= halfX (old code: class on the upper bound): %d"
              % (rows, dict(census), oldbig))


if __name__ == "__main__":
    main(sys.argv[1:])
