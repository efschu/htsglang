"""Build the #354 phase-optimal TPS table from the measured points.

Plain tokens per second on both axes -- prefill tok/s as the server counted
the prompt tokens, decode tok/s as the scheduler's own tick rate (the
windowed quantity; the client rate is carried as the second opinion).
Percentages are shown only against the reused noise floors, and a delta
inside its floor is printed as "within noise", never as a number to act on.
"""

import json

OUT = "/spinning/gpu-battery-results/2026-07-31_354_phase_optimal"
FLOOR_PREFILL = {1: 2.71, 8: 3.18}
FLOOR_PREFILL_DEFAULT = 3.18
FLOOR_DECODE = 2.72

# #327 reference arms, quoted, not re-measured
# (2026-07-31_327_int8_ab/tabelle_327.txt).
QUOTED = {
    ("fp8", "prefill", 1): 1310.7,
    ("fp8", "prefill", 8): 1134.5,
    ("int8", "prefill", 1): 1657.5,
    ("int8", "prefill", 8): 1396.1,
    ("fp8", "decode", 1): 94.67,
    ("fp8", "decode", 8): 379.88,
    ("int8", "decode", 1): 124.07,
    ("int8", "decode", 8): 431.1,
}


def load():
    pre, dec = {}, {}
    for r in (json.loads(x) for x in open(f"{OUT}/punkte.jsonl")):
        pre[(r["arm"], r["sessions"])] = r["prefill"]["prefill_tok_s"]
    for r in (json.loads(x) for x in open(f"{OUT}/decode_punkte.jsonl")):
        dec[(r["arm"], r["bs"])] = (
            r["tick_gen_tok_s_median"],
            r["klient_tok_s"],
            r["tick_ms_pro_verify"],
            r["tick_cuda_graph"],
        )
    return pre, dec


def verdict(a, b, floor):
    if a is None or b is None:
        return ""
    d = (b / a - 1.0) * 100.0
    return "within noise" if abs(d) < floor else f"{d:+.1f}%"


def main():
    pre, dec = load()
    L = []
    L.append("#354 PHASE-OPTIMAL TPS -- Qwen3.6-27B, TP=3 uneven (5090 + 2x 3080)")
    L.append("All figures are tokens per second. Higher is better everywhere.")
    L.append("")
    L.append(
        "Planner-solved vectors (--rank-tp-ratio auto-performance "
        "--rank-perf-tune enc):"
    )
    L.append(
        "  FP8  lanes 568.5 : 58.4 : 59.1 TFLOPS (9.73 : 1.00 : 1.01)"
        "  -> prefill-optimal --rank-mlp-ratio 16,1,1  (MLP units 121/8/7,"
        " predicted +22.9%)"
    )
    L.append(
        "  INT8 lanes 676.7 : 183.8 : 164.8 TFLOPS (3.68 : 1.00 : 0.90)"
        "  -> prefill-optimal --rank-mlp-ratio 10,1,1  (MLP units 113/12/11,"
        " predicted +9.1%)"
    )
    L.append(
        "  decode-optimal for BOTH formats = the plain VRAM-auto split "
        "(no MLP vector); #265 finding, unchanged."
    )
    L.append("")
    L.append("PREFILL tok/s")
    L.append(
        f"{'s':>3} {'FP8 auto':>12} {'FP8 quoted':>11} "
        f"{'FP8 16,1,1':>11} {'delta':>13} | "
        f"{'INT8 auto':>10} {'INT8 quoted':>12} {'INT8 10,1,1':>12} "
        f"{'delta':>13}"
    )
    for s in range(1, 9):
        fa, fp = pre.get(("fp8_auto", s)), pre.get(("fp8_prefopt", s))
        ia, ip = pre.get(("int8_auto", s)), pre.get(("int8_prefopt", s))
        fl = FLOOR_PREFILL.get(s, FLOOR_PREFILL_DEFAULT)
        fq = QUOTED.get(("fp8", "prefill", s))
        iq = QUOTED.get(("int8", "prefill", s))
        L.append(
            f"{s:>3} {fa:>12.1f} {(f'{fq:.1f}' if fq else '-'):>11} "
            f"{fp:>11.1f} {verdict(fa, fp, fl):>13} | "
            f"{ia:>10.1f} {(f'{iq:.1f}' if iq else '-'):>12} "
            f"{ip:>12.1f} {verdict(ia, ip, fl):>13}"
        )
    L.append("")
    L.append("DECODE tok/s (scheduler tick rate, median over the window)")
    L.append(
        f"{'bs':>3} {'FP8 auto':>12} {'FP8 quoted':>11} "
        f"{'FP8 16,1,1':>11} {'delta':>13} | "
        f"{'INT8 auto':>10} {'INT8 quoted':>12} {'INT8 10,1,1':>12} "
        f"{'delta':>13}"
    )
    for b in range(1, 9):
        fa = dec.get(("fp8_auto", b), (None,) * 4)[0]
        fp = dec.get(("fp8_prefopt", b), (None,) * 4)[0]
        ia = dec.get(("int8_auto", b), (None,) * 4)[0]
        ip = dec.get(("int8_prefopt", b), (None,) * 4)[0]
        fq = QUOTED.get(("fp8", "decode", b))
        iq = QUOTED.get(("int8", "decode", b))
        L.append(
            f"{b:>3} {fa:>12.1f} {(f'{fq:.1f}' if fq else '-'):>11} "
            f"{fp:>11.1f} {verdict(fa, fp, FLOOR_DECODE):>13} | "
            f"{ia:>10.1f} {(f'{iq:.1f}' if iq else '-'):>12} "
            f"{ip:>12.1f} {verdict(ia, ip, FLOOR_DECODE):>13}"
        )
    L.append("")
    L.append(
        "THE PHASE-OPTIMAL RECIPE (prefill column from the concentrated "
        "boot, decode column from the auto boot):"
    )
    L.append(f"{'point':>10} {'FP8':>10} {'INT8':>10}")
    for s in range(1, 9):
        L.append(
            f"{'prefill s=' + str(s):>10} "
            f"{pre[('fp8_prefopt', s)]:>10.1f} "
            f"{pre[('int8_prefopt', s)]:>10.1f}"
        )
    for b in range(1, 9):
        L.append(
            f"{'decode bs=' + str(b):>10} "
            f"{dec[('fp8_auto', b)][0]:>10.1f} "
            f"{dec[('int8_auto', b)][0]:>10.1f}"
        )
    L.append("")
    L.append(
        "Client-side decode rate (independent second opinion, tok/s) and "
        "CUDA-graph flag per point:"
    )
    for arm in ("fp8_auto", "fp8_prefopt", "int8_auto", "int8_prefopt"):
        row = ", ".join(
            f"bs{b} {dec[(arm, b)][1]:.1f}/{'G' if dec[(arm, b)][3] else 'g'}"
            for b in range(1, 9)
        )
        L.append(f"  {arm:>13}: {row}")
    L.append("")
    L.append(
        "Noise floors reused, not re-measured: prefill s=1 2.71%, "
        "prefill s>=2 3.18%, decode 2.72%."
    )
    txt = "\n".join(L)
    print(txt)
    with open(f"{OUT}/tabelle_354.txt", "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
