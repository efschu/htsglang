# WINDOW TICKET 735 STEP 2 — the gapped family layout (48 GDN -> 5090)

Pre-built at the desk so it fires the moment the harvest boot writes
COMP4_ACCEPTED. Do not boot before the preconditions table is all-green.
Owner of the boot: F4-r5. Runner: `run_window_ladder.sh --arm III`.

## 1. The exact configuration (derived LIVE, not copied)

Base: the harvest composite argv (`argv_735_composite.txt` shape) on the
then-current composite tip. Deltas, exactly two env additions:

```
SGLANG_PP_LAYER_SET="0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30,32-34,36-38,40-42,44-46,48-50,52-54,56-58,60-62;3,7,11,15,19,23,27,31;35,39,43,47,51,55,59,63"
SGLANG_PP_CROSSING_WIRE=1
```

Derivation (re-run today through the REAL modules, `tools`-free one-liner
in the ticket commit): FA mask = every 4th layer (16 layers); stage 0
(5090) = all 48 GDN layers; FA split 8/8 per DESIGN_pp_layer_set §5 with
the TERMINAL-LAYER LEVER — the x4-linked 3080 (stage 2) holds the half
containing layer 63, so the costliest link is owed one fewer return.
`parse_pp_layer_sets` accepts the string (counts [48, 8, 8]);
`crossing_schedule` on the parsed sets emits EXACTLY 31 crossings:
per-pair (0->1)=8, (0->2)=8, (1->0)=8, (2->0)=7 — 16 out, 15 back,
layer 63 terminal. The 32nd movement is the terminal output to `lm_head`
(on the 5090 per the full plan), not an inter-layer crossing.

`--pp-stage-ratio`/`--pp-attn-stage-ratio` stay AS-IS in the argv: the
SET overrides ownership; the contiguous ratios behind it are inert under
an explicit set (step 1 proved the set mechanism identity-clean).
Flip: unchanged (vector 32,16,16). The flip's TP stack resolves the
layer set to None via the scope fold (#754-in-#753), so env-set x flip
is legal ONLY once #753 is on the tip — which is also what provides the
wire. One precondition, two reasons.

## 2. Acceptances (stop at first failure)

| # | acceptance | readout | pass |
| --- | --- | --- | --- |
| A | gapped set accepted | boot reaches ready; no PPLayerSetError | ready |
| B | crossing count | log_routing / wire counters | == 31 EXACTLY; 30 or 32 = wrong map, abort the window |
| C | text identity | same seed+prompt greedy vs the step-1 contiguous boot | generated text matches (DESIGN §9.1 step-2 acceptance); GDN prefill nondeterminism >=~109 tokens applies — keep probes short |
| D | refusals fire (§9.2, once each, deliberately) | gapped set WITHOUT the wire env -> PPLayerSetError naming contiguity; set on a non-converted model -> RuntimeError | both fire loudly |
| E | throughput vs step-1 | soak driver as load, SAME probe set; across-a-flip cached-token acceptance per the #706 determination (9875bafa95) — NOT absolute hit rate | reported with A-vs-A floor; no target invented — this boot MEASURES the family layout for the first time |
| F | corridor | NVML free per card under load | above OOM floor; ~1024 target advisory |
| G | no #737 wedge | rank-local ACK drain lines present, no ack-wait hang | no wedge across >=1 flip each way |
| H | GDN slot ceiling readout | boot ledger: mamba slots on the 5090 | record slots vs the #735 ladder (30/23/20 nominal-corrected); with #727 arm C absent expect the bf16-lm_head ~7-slot column — RECORD, don't fail |

## 3. Preconditions (status at ticket-writing, 2026-08-18)

| item | status |
| --- | --- |
| COMP4_ACCEPTED marker | GATE — written by F4-r5 after the harvest acceptances |
| feat/753 wire + gapped-refusal lift + #754 fold | **ABSENT from composite tip b7e6a4110b — THE blocking precondition** (lands by its owner; provides both the transport and the flip-scope legality) |
| #757 fold (armed liveness) | PRESENT (5e2c121595 merged at f8368f7208) |
| #756 local_slot family | PRESENT (comp4 ancestry) |
| #760 hicache binding refusal | PRESENT (e4fe5e28a8) |
| #706 remainder determination | PRESENT (9875bafa95) — its across-a-flip acceptance form is row E |
| #727 arm C artifact | OPTIONAL, both variants priced: WITH `vocabint8-both` the 5090 ladder reads 20-23 slots (NVML-corrected); WITHOUT it ~7 slots (bf16 lm_head) — the boot is legal either way, row H records which world it ran in |
| host ledger | 16G lane floor respected; #738 caveat: the load-time page-cache spike is the MMAP path — the drop flag + direct-io flags of the composite argv stay ON |
| hicache disk tier + interval 8192 | as the composite argv (ARM II semantics ride along; anchor emitters now in-tree) |

## 4. Runner

`run_window_ladder.sh --arm III --log <bootlog>`: rows A/B/D/G/H are
log-greps (crossing count parsed exactly, refusal lines counted), C and E
are operator-phase rows (need the step-1 reference boot's outputs at
hand). Mock-smoked against fixtures in both directions (31 passes,
30 fails; refusal present passes, absent fails).
