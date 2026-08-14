#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#485 -- is the seam threshold REACHABLE? The three probes behind the verdict.

``seam_target_485.py`` answers "which term binds". This answers "can any
mechanism get under it", by pricing each mechanism against the decomposition
in ``ANALYSE_SEAM_485.md`` §2 and re-solving.

Subcommands:

  ceilings   Bisect, per seam term, the largest value it may take with every
             OTHER term held at exactly zero. The most generous configuration
             that exists, so these are upper bounds on what a mechanism may
             leave behind -- and they are not simultaneously available.

  mechanisms Price mechanisms (a) persistent arena, (b) chunked refill and
             (d) wave exclusivity against the measured terms and re-solve.

  pools      The best-achievable vector as a function of --max-total-tokens.
             The seam is pool-linear (§6), so a sweep that holds it fixed --
             as #584's L3 did -- measures a model in which the pool cannot
             help. This couples them.

Usage: seam_feasibility_485.py {ceilings|mechanisms|pools} [census_dir]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seam_target_485 as S  # noqa: E402

DEFAULT_CENSUS = "/spinning/evidence-631/m485/m0/census"
CERT_POOL = 280000

ORDER = [
    (0, "SEAM_TP_TO_PP"), (0, "SEAM_PP_TO_TP"),
    (1, "SEAM_TP_TO_PP"), (1, "SEAM_PP_TO_TP"),
    (2, "SEAM_TP_TO_PP"), (2, "SEAM_PP_TO_TP"),
]
#: As measured on the governing M0 census (cut 28,20,16, pool 620000).
MEASURED = {
    (0, "SEAM_TP_TO_PP"): 1838, (0, "SEAM_PP_TO_TP"): 1302,
    (1, "SEAM_TP_TO_PP"): 490, (1, "SEAM_PP_TO_TP"): 164,
    (2, "SEAM_TP_TO_PP"): 408, (2, "SEAM_PP_TO_TP"): 256,
}
#: Pool-borne content held at the trough (wave + kv_stage), ANALYSE §2/§2d.
POOL_BORNE = {
    (0, "SEAM_TP_TO_PP"): 1670, (0, "SEAM_PP_TO_TP"): 1992,
    (1, "SEAM_TP_TO_PP"): 450, (1, "SEAM_PP_TO_TP"): 584,
    (2, "SEAM_TP_TO_PP"): 326, (2, "SEAM_PP_TO_TP"): 770,
}
#: Arena tail at the CANDIDATE cut 29,19,16 (+128 MiB checksum), ANALYSE §2c.
#: This is what mechanism (b) chunked refill removes, at zero at-rest cost.
ARENA_AT_CANDIDATE_CUT = {
    (0, "SEAM_TP_TO_PP"): 282, (0, "SEAM_PP_TO_TP"): 0,
    (1, "SEAM_TP_TO_PP"): 280, (1, "SEAM_PP_TO_TP"): 0,
    (2, "SEAM_TP_TO_PP"): 1564, (2, "SEAM_PP_TO_TP"): 0,
}
#: Arena tail as MEASURED on M0, i.e. what a mechanism would recover on the
#: census as it stands rather than at the cut the gate would admit.
ARENA_MEASURED = {
    (0, "SEAM_TP_TO_PP"): 0, (0, "SEAM_PP_TO_TP"): 0,
    (1, "SEAM_TP_TO_PP"): 594, (1, "SEAM_PP_TO_TP"): 0,
    (2, "SEAM_TP_TO_PP"): 1564, (2, "SEAM_PP_TO_TP"): 0,
}
#: The rank-0 entry deficit -- the rank already below its census baseline when
#: the flip starts. No seam mechanism reaches it (ANALYSE §5e).
ENTRY_DEFICIT_R0_TP = 169
#: memory_pool.py:2587-2600: the irreducible ONE DESTINATION LAYER transient on
#: the tp_to_pp leg. Cross-checks against the solver's own KV term to 0.6 MiB.
MIB_PER_1000_TOKENS_PER_LAYER = 1.953
MEASURED_POOL = 620000


def solve_vector(census, vals, pool, tag):
    ov = dict(zip(ORDER, [float(v) for v in vals]))
    dst = f"/tmp/seam485_feas/{tag}"
    S.write_census(census, dst, ov)
    S.BASE_BUDGET = (31400, 19300, 19300)
    S.POOL = pool
    ok, detail = S.solve(dst)
    return ok, detail


def one_layer_floor(pool):
    return MIB_PER_1000_TOKENS_PER_LAYER * pool / 1000.0


def cmd_ceilings(census):
    print(f"INDIVIDUAL CEILING per seam term, every OTHER term at zero, "
          f"pool={CERT_POOL}")
    print(f"  {'term':<26}{'measured':>10}{'ceiling':>10}{'cut needed':>12}")
    n = [0]

    def test(key, v):
        n[0] += 1
        vals = [v if k == key else 0 for k in ORDER]
        ok, _ = solve_vector(census, vals, CERT_POOL, f"ceil{n[0]:04d}")
        return ok

    for key in ORDER:
        if not test(key, 0):
            print(f"  {f'rank{key[0]} {key[1]}':<26}{MEASURED[key]:>10}"
                  f"{'<0':>10}{'impossible':>12}", flush=True)
            continue
        lo, hi = 0, 4000
        while hi - lo > 25:
            mid = (lo + hi) // 2
            if test(key, mid):
                lo = mid
            else:
                hi = mid
        print(f"  {f'rank{key[0]} {key[1]}':<26}{MEASURED[key]:>10}{lo:>10}"
              f"{MEASURED[key] - lo:>12}", flush=True)


def cmd_mechanisms(census):
    cases = [
        ("as measured (control)",
         [MEASURED[k] for k in ORDER]),
        ("(a)/(b) arena removed, on the census as it stands",
         [max(0, MEASURED[k] - ARENA_MEASURED[k]) for k in ORDER]),
    ]
    # At the candidate operating point: pool-rescale, then apply mechanisms.
    f = CERT_POOL / MEASURED_POOL
    rescaled = {k: max(0.0, MEASURED[k] - POOL_BORNE[k] * (1 - f))
                for k in ORDER}
    e1 = dict(rescaled)
    e1[(0, "SEAM_TP_TO_PP")] += ARENA_AT_CANDIDATE_CUT[(0, "SEAM_TP_TO_PP")]
    cases.append(("E1 re-derived at pool 280000, cut 29,19,16, no mechanism",
                  [round(e1[k]) for k in ORDER]))
    e2 = {k: max(0.0, e1[k] - ARENA_AT_CANDIDATE_CUT[k]) for k in ORDER}
    cases.append(("E2 + (b) chunked refill removes the arena term",
                  [round(e2[k]) for k in ORDER]))
    e3 = dict(e2)
    e3[(0, "SEAM_TP_TO_PP")] = ENTRY_DEFICIT_R0_TP + one_layer_floor(CERT_POOL)
    cases.append(("E3 + (d) wave AT ITS DOCUMENTED FLOOR -- best achievable",
                  [round(e3[k]) for k in ORDER]))
    for i, (name, vals) in enumerate(cases):
        ok, detail = solve_vector(census, vals, CERT_POOL, f"mech{i:03d}")
        print(f"  {'ADMIT ' if ok else 'refuse'}  {tuple(vals)!s:<38} {name}",
              flush=True)


def cmd_pools(census):
    print("BEST-ACHIEVABLE vector by pool -- every mechanism perfect, the wave "
          "sitting exactly on its documented one-layer floor.")
    for pool in (280000, 240000, 200000, 160000, 120000, 80000):
        f = pool / MEASURED_POOL
        floor = one_layer_floor(pool)
        v = {}
        for k in ORDER:
            t = max(0.0, MEASURED[k] - POOL_BORNE[k] * (1 - f))
            v[k] = max(0.0, t - ARENA_AT_CANDIDATE_CUT[k])
        v[(0, "SEAM_TP_TO_PP")] = max(
            v[(0, "SEAM_TP_TO_PP")], ENTRY_DEFICIT_R0_TP + floor)
        vals = [round(v[k]) for k in ORDER]
        ok, _ = solve_vector(census, vals, pool, f"pool{pool}")
        print(f"  {'ADMIT ' if ok else 'refuse'}  pool={pool:<7} "
              f"{tuple(vals)!s:<32} [1-layer floor {floor:.0f}]", flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "mechanisms"
    census = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CENSUS
    {"ceilings": cmd_ceilings,
     "mechanisms": cmd_mechanisms,
     "pools": cmd_pools}[cmd](census)
    return 0


if __name__ == "__main__":
    sys.exit(main())
