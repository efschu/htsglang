# #753 — the gapped forward passes the acceptance probe on the current tip

GPU window 2026-08-18 22:16-22:43Z, rig CT999, boots on `feat/v7-gapped-base`
(tip `9886bddb01` plus the #763 fix). Serving restored and verified afterwards.

## Result

The gapped layer set with the crossing wire answers the #753 determined-answer
probe **correctly**, and does not degenerate on longer generations.

| arm | layout | answer to "The capital of France is" (temp 0, seed 735000001) |
|---|---|---|
| reference | contiguous PP=3, `--pp-stage-ratio 14,10,8` | `' Paris.\nThe capital'` |
| gapped | `SGLANG_PP_LAYER_SET` 48/8/8 + `SGLANG_PP_CROSSING_WIRE=1` | `' Paris.\nThe capital'` |
| gapped + CUDA graphs | same, graphs enabled | `' Paris.\nThe capital'` |

40-token continuation on the gapped arm runs clean through Berlin/Rome/Madrid/
Lisbon, and a free-prose prompt returns a correct Rayleigh-scattering answer.
#753 recorded `'\n\n'` for the first token and "longer generations degenerate
into a repeated token"; neither reproduces here.

The gapped layout was genuinely active, not silently ignored: the wire logged
`crossing wire rank 0: 16 send(s), 15 receive(s)`, `rank 1: 8/8`, `rank 2: 7/8`
— the 31 crossings the #735 step-2 ticket derived.

## What this does and does not license

It does NOT license lifting the `SGLANG_PP_GAPPED_ALLOW_KNOWN_WRONG` gate. The
arms above are a PLAIN PP=3: no phase flip, no speculation, `--context-length
8192`, `--chunked-prefill-size 512`, `--disable-overlap-schedule`, on
`Qwen3.8-27B-INT8-yarn1.5`. The #753 repro ran the composite argv, which carries
the flip and spec. So the finding narrows the defect rather than clearing it:
the gapped forward is correct on its own, and whatever produced `'\n\n'` lives
in the interaction with flip and/or spec, not in the layer set or the crossing
wire. CUDA graphs were tested explicitly and are NOT the interacting feature.

Next step before the gate can move: re-run these two arms with the flip vector
and NEXTN spec enabled, one variable at a time.

## Layer-norm trace: two instrument defects found and fixed

`SGLANG_LAYER_NORM_TRACE` as shipped traces the FIRST forward only, and that
forward is the KV memory-profiling dummy. On a PP layout its later stages
receive zeros, so the trace recorded `h=0.000000 r=0.000000` for all 36 layers
past stage 0 — it could not localize anything. `SGLANG_LAYER_NORM_TRACE_PASSES`
now selects how many passes per layer to record, so the probe prompt gets its
own pass.

Extending it that way immediately exposed the second defect: a norm is an
`.item()`, i.e. a device-to-host sync, which is illegal inside a CUDA graph
capture. Once the trace outlived the profiling forward it landed inside capture
and took the boot down with `cudaErrorStreamCaptureInvalidated`. The trace now
skips captured passes.

With both fixed, the contiguous reference produced a clean 64-layer table. The
gapped table is NOT directly diffable against it: the two arms batched the same
prompt differently (`rows=60` vs `rows=329`), and the instrument norms row 0
only, so equal-looking rows are not guaranteed to be the same token. Reported
deviations reached 5% on 21 layers, which on that basis is not evidence of
anything; the end-to-end acceptance above is the load-bearing result.

Reference table kept at `/spinning/evidence-665-f1/trace_ref_pass0.txt`, gapped
at `trace_gapped_pass0.txt`, boots at `boot_753_ref.sh` / `boot_753_gapped.sh` /
`boot_753_gapped_graphs.sh`.
