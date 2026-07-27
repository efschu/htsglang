#!/usr/bin/env python3
"""Fit the decode-knee model against the measured MLP-split campaign."""
import json, sys, itertools, math
sys.path.insert(0, "/spinning/wt-knee-guard/python")
from sglang.srt.uneven_perf import PlanInputs, PerfCostModel

D = "/spinning/wt-knee-guard/bench216/logs"
MODEL = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-FP8"
BASE = [28447, 16320, 16320]           # budgets this campaign actually booted
MEMBW = [1664.1, 718.2, 718.2]
ARMS = {"base_A": None, "base_B": None, "m611_A": [6,1,1], "m611_B": [6,1,1],
        "m311_A": [3,1,1], "m411_A": [4,1,1]}

pi = PlanInputs(tp_size=3, model_path=MODEL, kv_cache_dtype="fp8_e5m2",
                speculative_algorithm="NEXTN", speculative_num_draft_tokens=4,
                rank_gpu_id=[0,1,2], effective_vram_mib=BASE, rank_tp_ratio=BASE)
m = PerfCostModel(pi, BASE, BASE)

def load(arm):
    try:
        return json.load(open(f"{D}/{arm}.json"))
    except FileNotFoundError:
        return None

data = {a: load(a) for a in ARMS}
have = {a: d for a, d in data.items() if d}
print(f"arms measured: {sorted(have)}\n")

# ---------- noise floor: boot-to-boot A vs A --------------------------
print("=== NOISE FLOOR (boot-to-boot, same arm) ===")
for a, b in (("base_A","base_B"), ("m611_A","m611_B")):
    if a in have and b in have:
        for L in have[a]["decode"]:
            x, y = have[a]["decode"][L]["step_ms"], have[b]["decode"][L]["step_ms"]
            print(f"  decode ctx={L:>6}: {a} {x:8.3f} vs {b} {y:8.3f} ms "
                  f"-> {abs(x-y)/((x+y)/2)*100:5.2f} %")
        for L in have[a]["prefill"]:
            x, y = have[a]["prefill"][L]["median_ms"], have[b]["prefill"][L]["median_ms"]
            print(f"  prefill L={L:>6}: {a} {x:9.1f} vs {b} {y:9.1f} ms "
                  f"-> {abs(x-y)/((x+y)/2)*100:5.2f} %")
print()

def avg(arm_list, kind, L):
    v = [have[a][kind][L]["step_ms" if kind=="decode" else "median_ms"]
         for a in arm_list if a in have]
    return sum(v)/len(v) if v else None

GROUPS = {"base": ["base_A","base_B"], "3,1,1": ["m311_A"],
          "4,1,1": ["m411_A"], "6,1,1": ["m611_A","m611_B"]}
VEC = {"base": BASE, "3,1,1": [3,1,1], "4,1,1": [4,1,1], "6,1,1": [6,1,1]}

# ---------- decode vs base -------------------------------------------
print("=== DECODE (ms per output token) ===")
dec = {}
for g, arms in GROUPS.items():
    row = {L: avg(arms, "decode", L) for L in have[list(have)[0]]["decode"]}
    dec[g] = row
b = dec["base"]
for g, row in dec.items():
    s = "  ".join(f"ctx{L}={row[L]:7.3f} ms ({row[L]/b[L]-1:+6.2%})"
                  for L in row if row[L] and b[L])
    print(f"  {g:>6}: {s}")
print()

# ---------- decode on natural text (primary: normal acceptance) -------
print("=== DECODE on natural text (ms per output token, and per spec step) ===")
dect = {}
for g, arms in GROUPS.items():
    row = {}
    for a in arms:
        for L, d in (have.get(a, {}).get("decode_text") or {}).items():
            row.setdefault(L, []).append(d)
    dect[g] = row
if dect.get("base"):
    for L in sorted(dect["base"], key=int):
        bt = sum(d["step_ms"] for d in dect["base"][L]) / len(dect["base"][L])
        for g in dect:
            if not dect[g].get(L):
                continue
            v = [d["step_ms"] for d in dect[g][L]]
            t = sum(v) / len(v)
            acc = [x for d in dect[g][L] for x in (d.get("spec_accept_length") or [])]
            am = (sum(acc) / len(acc)) if acc else None
            step = f"  step={t*am:7.3f} ms/spec-step (accept {am:.2f})" if am else ""
            print(f"  ctx~{L:>6} {g:>6}: {t:7.3f} ms/tok "
                  f"({t/bt-1:+6.2%} vs base){step}")
    print()

# ---------- prefill vs base ------------------------------------------
print("=== PREFILL (ms, max_new_tokens=1, cache-miss) ===")
pre = {g: {L: avg(a, "prefill", L) for L in have[list(have)[0]]["prefill"]}
       for g, a in GROUPS.items()}
Ls = sorted(pre["base"], key=int)
print("     arm |" + "".join(f"{L:>10}" for L in Ls))
for g in pre:
    print(f"  {g:>6} |" + "".join(f"{pre[g][L]:10.1f}" if pre[g][L] else f"{'-':>10}" for L in Ls))
print("  gain vs base:")
for g in pre:
    if g == "base": continue
    print(f"  {g:>6} |" + "".join(
        f"{(pre['base'][L]/pre[g][L]-1)*100:+9.1f}%" if pre[g][L] and pre['base'][L] else f"{'-':>10}"
        for L in Ls))
print()

# ---------- crossover: where does 6,1,1 stop paying? ------------------
# Prefill saving is per PROMPT token (slope difference of the cache-miss
# prefill line); decode cost is per OUTPUT token (per-spec-step difference
# divided by the acceptance the base arm shows at that depth). The break-even
# is the prompt:output ratio at which the two cancel.
print("=== CROSSOVER ===")
def slope(g):
    Ls_i = sorted((int(L) for L in pre[g] if pre[g][L]))
    if len(Ls_i) < 2: return None, None
    lo, hi = Ls_i[0], Ls_i[-1]
    a = (pre[g][str(hi)] - pre[g][str(lo)]) / (hi - lo)
    return a, pre[g][str(lo)] - a*lo
for g in GROUPS:
    if g == "base" or not pre.get(g): continue
    sb, ib = slope("base"); sg, ig = slope(g)
    if sb is None or sg is None: continue
    save = sb - sg                      # ms saved per prompt token
    # decode: ms per spec step, averaged over the measured depths
    def stepms(gg):
        v = []
        for L, ds in (dect.get(gg) or {}).items():
            for d in ds:
                acc = d.get("spec_accept_length") or []
                if acc: v.append(d["step_ms"] * (sum(acc)/len(acc)))
        return sum(v)/len(v) if v else None
    def accept(gg):
        v = [x for L, ds in (dect.get(gg) or {}).items() for d in ds
             for x in (d.get("spec_accept_length") or [])]
        return sum(v)/len(v) if v else None
    sb_ms, sg_ms, acc = stepms("base"), stepms(g), accept("base")
    print(f"  {g}: prefill slope {sb:.4f} -> {sg:.4f} ms/prompt-token "
          f"(saves {save:.4f} ms/tok, {save/sb:+.1%})")
    if sb_ms and sg_ms and acc:
        dstep = sg_ms - sb_ms
        cost_tok = dstep / acc
        print(f"      decode {sb_ms:.2f} -> {sg_ms:.2f} ms/spec-step "
              f"({dstep:+.2f} ms, {dstep/sb_ms:+.1%}); at accept {acc:.2f} "
              f"that is {cost_tok:+.4f} ms per OUTPUT token")
        if save > 0 and cost_tok > 0:
            print(f"      BREAK-EVEN: prompt/output ratio = "
                  f"{cost_tok/save:.1f} : 1")
            for n in (64, 128, 256, 512, 1024):
                print(f"         {n:>5} output tokens -> pays off above a "
                      f"{cost_tok/save*n:,.0f}-token prompt")
        elif save <= 0:
            print("      no prefill gain -- never pays off")
        else:
            print("      no decode cost measured -- pays off everywhere")
print()

# ---------- what the SHIPPED model predicted --------------------------
print("=== SHIPPED MODEL vs MEASUREMENT ===")
def proxy(vec, bw):
    s = m.streamed_bytes(list(vec))
    return max(x/b for x, b in zip(s, bw))
pb = proxy(BASE, MEMBW)
print(f"  {'vec':>7} {'share_r0':>9} {'knee_ok':>8} {'predicted':>10} {'measured':>10}")
for g, v in VEC.items():
    cand = m.streamed_bytes(list(v)); tot = sum(cand)
    ok, _ = m.decode_knee_detail(list(v), MEMBW)
    pred = proxy(v, MEMBW)/pb - 1
    meas = (dec[g][Ls_d]/b[Ls_d] - 1) if (Ls_d := sorted(b, key=int)[0]) and dec[g][Ls_d] else None
    print(f"  {g:>7} {cand[0]/tot:>8.2%} {str(ok):>8} {pred:>+9.1%} "
          + (f"{meas:>+9.1%}" if meas is not None else f"{'-':>10}"))
print()

# ---------- refit: effective bandwidth --------------------------------
# Fitted on ms per SPEC STEP from the natural-text arm. Random-token prompts
# drive the model into a degenerate output mode whose acceptance rate swamps
# the weight term, so that arm is not usable for calibration.
# Model: t(v) = C + max_r bytes_r(v) / membw_r**beta  (beta=1 = shipped).
print("=== REFIT on natural-text ms/spec-step ===")
def stepms_g(g):
    v = []
    for L, ds in (dect.get(g) or {}).items():
        for d in ds:
            acc = d.get("spec_accept_length") or []
            if acc:
                v.append(d["step_ms"] * (sum(acc) / len(acc)))
    return sum(v) / len(v) if v else None

pts = [(VEC[g], stepms_g(g)) for g in VEC if stepms_g(g)]
print("  points: " + ", ".join(
    f"{g}={stepms_g(g):.2f} ms" for g in VEC if stepms_g(g)))
best = None
for bi in range(0, 121):
    beta = bi / 100
    bw = [x ** beta for x in MEMBW]
    xs = [max(s0 / b for s0, b in zip(m.streamed_bytes(list(v)), bw))
          for v, _ in pts]
    ys = [t for _, t in pts]
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        continue
    A = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    C = my - A * mx
    if A <= 0 or C <= 0:
        continue                 # weight term must cost time; constant >= 0
    rms = math.sqrt(sum((C + A * x - y) ** 2 for x, y in zip(xs, ys)) / n)
    if best is None or rms < best[0]:
        best = (rms, beta, A, C, xs, ys)
rms, beta, A, C, xs, ys = best
bw_eff = [x ** beta for x in MEMBW]; se = sum(bw_eff)
print(f"  best beta = {beta:.2f}   (beta=1.00 = shipped peak-bandwidth model)")
print(f"  fit rms   = {rms:.3f} ms over {len(xs)} points; constant C = {C:.2f} ms")
print(f"  effective bandwidth share = {[f'{x/se:.2%}' for x in bw_eff]}")
print(f"  peak bandwidth share      = {[f'{x/sum(MEMBW):.2%}' for x in MEMBW]}")
for (v, t), x in zip(pts, xs):
    lbl = "base" if v is BASE else ",".join(map(str, v))
    print(f"    {lbl:>7}: measured {t:7.3f} ms   fitted {C + A*x:7.3f} ms")
print()
print("  knee verdict under the refit (share vs EFFECTIVE bandwidth share):")
for g, v in VEC.items():
    cand = m.streamed_bytes(list(v)); tot = sum(cand)
    meas = stepms_g(g) / stepms_g("base") - 1 if stepms_g(g) else None
    print(f"    {g:>7}: share_r0 {cand[0]/tot:6.2%} vs eff ceiling "
          f"{bw_eff[0]/se:6.2%} -> "
          f"{'OK' if cand[0]/tot <= bw_eff[0]/se else 'REJECT':>6}"
          + (f"   (measured {meas:+.1%})" if meas is not None else ""))
