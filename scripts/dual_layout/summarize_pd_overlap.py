#!/usr/bin/env python3
"""Summarize a run_pd_overlap.sh output dir: shares, E, decode latency."""
import glob, json, os, sys
d = sys.argv[1]
def L(p):
    try: return json.load(open(p))
    except Exception: return None
for base in sorted({f.rsplit('.', 2)[0] for f in glob.glob(d + '/card*.json')}):
    sd, sp = L(base + '.solo_decode.json'), L(base + '.solo_prefill.json')
    if not sd or not sp: print('missing solo', base); continue
    D0, P0 = sd['decode_steps_per_s'], sp['prefill_tflops']
    print(f"\n{os.path.basename(base)}  solo decode {D0:.1f} step/s p50 {sd['decode_p50_ms']:.2f} ms p99 {sd['decode_p99_ms']:.2f} | solo prefill {P0:.1f} TFLOP/s")
    arms = [('inproc', L(base+'.inproc.json'), None), ('inproc_eqprio', L(base+'.inproc_eqprio.json'), None)]
    for a in ('2proc', '2proc_mps', '2proc_mps_p50'):
        arms.append((a, L(f"{base}.{a}_decode.json"), L(f"{base}.{a}_prefill.json")))
    for name, x, y in arms:
        if x is None: print(f"  {name:14s} missing"); continue
        y = y or x
        if 'decode_steps_per_s' not in x or 'prefill_tflops' not in y: print(f"  {name:14s} incomplete"); continue
        a_, b_ = x['decode_steps_per_s'] / D0, y['prefill_tflops'] / P0
        print(f"  {name:14s} share_dec {a_:.3f} share_pre {b_:.3f}  E {a_+b_:.3f}  dec p50 {x['decode_p50_ms']:.2f} p99 {x['decode_p99_ms']:.2f} max {x['decode_max_ms']:.1f} ms | pre {y['prefill_tflops']:.1f} TF")
