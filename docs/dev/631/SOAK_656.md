# SOAK 656 — the long-soak MTTF proof, and the ~280k band

Merged line `567ba68023` (`integration/r2` == `feat/route-a-631` == `merge/r9-batch`).
Two boots, argv-identical on every load-bearing flag (`context_length=393216`,
`max_running_requests=4`, `chunked_prefill_size=512`, `pp_stage_ratio=[14,10,8]`,
`rank_gpu_memory_mib=[31583,15750,18205]`, `phase_flip_tp_vector='30,16,18'`,
`SGLANG_PHASE_FLIP_SEAM_MARGIN_MIB=384`), so the two runs differ in the LOAD MIX
and in nothing else. Evidence: `/spinning/evidence-631/soak-656/`.

## 0. The bar, fixed before the numbers existed
`METHOD.md`, written before the first probe. With 0 failures in N cutovers against
the acceptance's observed p0 = 1/320: 90 % needs 736, **95 % needs 957**, 99 % needs
1471 — reproducing HANDOFF_MERGE_R9 §12.1 exactly.

## 1. VERDICT

| task | verdict |
|---|---|
| long soak / MTTF | **PASS.** 1134 clean cutovers, 0 failures of any class |
| ~280k band bisect | **DID NOT REPRODUCE.** 14 of 14 cold probes exact |
| sustained bs=4 leg | ran, see §5 |
| defect found | **YES** — the live slot set is never agreed; red test on `feat/soak-fixes-656` |

### The MTTF statement, precisely
**1134 completed cutovers with zero abandons, zero divergences, zero tracebacks,
zero SIGQUIT, zero KvReshardError, zero CANNOT FUND, and zero corridor breaches.**

* An instance with the acceptance's failure rate (1 in 320) produces a run this
  clean with probability **2.87 %** — so this run **rejects that rate at 97.1 %
  confidence**, above the 95 % bar the task named.
* Assumption-free companion (rule of three): 0 failures in 1134 bounds the true
  per-cutover failure probability at **< 1 in 378 with 95 % confidence**.
* What it does NOT say: the rate is not proven to be zero, and 1134 does not reach
  the 99 % bar (1471).

## 2. BOOT 2 — the recipe-faithful run (19:27-22:02Z)
Load: mixed bs1-4, rungs to 64k, a 128k deep prefill every 5th cycle, 91 cycles,
plus the router leg through 30099 concurrently. **No deep probes** — the acceptance
recipe and nothing else.

| | |
|---|---|
| cutovers | **1134** (3402 rank lines / 3) |
| KvReshardError / tracebacks / SIGQUIT / CANNOT FUND | **0 / 0 / 0 / 0** |
| FLIP ABANDONED / wire frame divergence | **0 / 0** |
| flips DELAYED / margin YIELDED | **0 / 0** |
| corridor | **92752 samples/card, 154.9 min, minima 1264 / 2640 / 1390 MiB, 0 below the 1024 law** |
| corridor percentiles | gpu0 p1 1378 median 1870; gpu1 p1 3453 median 4325; gpu2 p1 1646 median 2072 |
| pool | 586186 tokens, derived and uncapped |
| graphs + spec | 1080 graph-covered decode batches; draft verify graph captured on all 3 ranks; accept len 2.4-2.8 |
| router leg | 325 of 325 HTTP 200, 0 failures |
| spill rungs | 6801 |

**Stated because it is the honest half:** on this boot **neither** R2 mechanism
fired — 0 cap-agreement moves, 0 corridor-bounded recoveries, 0 purity stand-downs,
0 yields withheld. Boot 2 is a **no-failure** result, not a proof that the
mechanisms work. Their positive proof is boot 1.

## 3. BOOT 1 — where both mechanisms fired, and what it cost
Full detail in `BOOT1_VERDICT.md`. 194 cutovers, 0 KvReshardError / tracebacks /
SIGQUIT, corridor 33280 samples/card with minima 1028 / 2195 / 1094 and **0 below
the law**. This boot carried the band bisect CONCURRENTLY (coordinator direction);
those 260k-290k probes are the KV pressure the acceptance recipe does not contain.

**Both mechanisms R2 could not exercise fired here, on metal:**

1. **Cap agreement levelled the group.** 19:04:11Z PP1 came back corridor-bounded at
   569479 of 584934 rows (15455 short); 15 s later PP0 and PP2 each logged
   `cap agreement moved this rank -15455 exposed rows`. At-arm the three censuses were
   byte-identical **including free-list head order** (`[349798 ... 349809]` on all
   three) while PP1 carried `unaccounted=15967` against its peers' 512.
   Evidence: `EVENT_cap_agreement_1.txt`.
2. **The ballot caught a real divergence and the valve kept serving.** 19:16:48Z all
   three ranks framed a `wire frame divergence` and ABANDONED the flip rather than
   send mismatched frames — the exact C22 failure that killed the #656 acceptance
   with a "checksum mismatch" that was never a checksum. Backoff 6/12/24/48 s, then a
   three-rank purity stand-down: decode ran in the layout the instance was in.
   `/health` 200 and real generations throughout; the 128k prefill and a 280k probe
   both completed after the stand-down. Evidence: `EVENT_ballot_divergence.txt`.

**The cost:** flipping never resumed. The policy re-probes on a 60 s cap and refuses
every time. Draining to zero requests did not clear it; neither did `/flush_cache`.
So the C22 fix converts an instance-killing crash into a **bounded, announced,
serving-preserving livelock**.

## 4. THE DEFECT: the live slot set is never agreed
`ROOTCAUSE_liveslot_divergence.md`, `EVENT_pp1_outlier.txt`.

**PP1 was the outlier in 4 of 4 divergence episodes; PP0 and PP2 always agreed with
each other. PP1 was also the only rank that took corridor-bounded recoveries (7 of 7).**

```
19:16:48 PP0/PP2 mine=1538433496   PP1 mine=2029216398
19:16:54 PP0/PP2 mine=2068034637   PP1 mine=1588424085
19:17:31 PP0/PP2 mine=2032529267   PP1 mine=1488169107
19:21:20 PP0/PP2 mine=161620523    PP1 mine=1983122301
```

The digest hashes `torch.unique(live_slots)` — KV pool ROW ids, the radix tree's
`all_values_flatten()` union every live request's `req_to_token` rows
(`phase_flip_runtime.py:3399-3480`, `:630-745`). `torch.unique` sorts, so the digest
is **order-insensitive**: the sets differ in CONTENT. Each rank samples it
independently at `:4831` with **no collective over the content**; arrival and abort
are guarded, **finish/EOS and retraction are not**.

**The code already diagnosed this class** at `phase_flip_runtime.py:4407-4416`: a rank
bounded by the corridor "comes back from a phase with fewer rows than its peers, its
live-slot enumeration differs by exactly that many, and the frame ballot below refuses
every subsequent flip". The fix made the row COUNT agree before framing. **Boot 1 shows
that is not sufficient** — the cap agreement fired (-15455, then -77) and the SET still
diverged, because levelling how many rows a rank exposes does not make the ranks hold
the SAME rows.

Latent trap found while reading: `blocking_guards` is only ever appended to, never
cleared, so at `DEFAULT_SEAM_ABANDON_CAP = 8` group abandons the direction is refused
for the rest of the boot with no path back short of a restart. Boot 1 reached 4.

**Red-first test committed** (`feat/soak-fixes-656`, `56f79459fc`), NOT merged:
`test/registered/scheduler/test_flip_live_slot_agreement_656.py` — 1 failed (on the
assertion, reproducing the production message), 3 passed; with the sibling suites
1 failed / 79 passed. The fix it points at is an agree-before-frame step over the live
slot set at `_execute()`, mirroring where the row-count agreement already sits.

## 5. THE ~280k BAND — it did not reproduce
Unique filler, `/flush_cache` before every probe, `cached` asserted 0 on every one.

| depth | probes | exact | empty | rate |
|---|---|---|---|---|
| 260016 | 2 | 2 | 0 | 0 % |
| 270016 | 2 | 2 | 0 | 0 % |
| 280016 (rounds 1-2, measured 279996) | 2 | 2 | 0 | 0 % |
| 290016 | 2 | 2 | 0 | 0 % |
| **280016 exact historical construction (reps 14555)** | **6** | **6** | **0** | **0 %** |
| **total** | **14** | **14** | **0** | **0 %** |

Round 3 reproduces R2's failing probe byte-for-byte (same template, same fixed-width
tag, reps 14555 → `prompt_tokens=280016` measured on every probe).

**Two readings the data kills outright:**
* *"The first request past `max_position_embeddings` fails."* b02 at 270016 was this
  boot's first request past 262144 and it was EXACT.
* *"~280k is a band on this line."* 8 cold probes at ~280k, all exact.

**Rate statement.** The historical rate was ~5 of 7 (0.71) at ~280k. 8 exact probes
give P(all exact | p=0.71) = 5e-5, so that rate is excluded. The assumption-free
bound: 0 failures in 8 puts the true rate at **< 37.5 % with 95 % confidence**, and
any rate ≥ 31 % is rejected. A rate of, say, 5 % is NOT excluded by 8 probes and would
need ~60.

**Mechanism** (`CODEREAD_SUSPECTS.txt`). Excluded by code evidence, not argument: no
position clamp/wrap/modulo at `max_position_embeddings` anywhere in the RoPE stack;
text-only mrope position ids are a plain `arange`; every position- and
frequency-carrying tensor is float32; the YaRN ramp/mscale is a pure function of
position and so cannot be intermittent at a fixed depth. The lazy RoPE cache is
default-off AND this boot proves it directly — every rank logged
`393600x64 float32 96.1 MiB reserved / 96.1 MiB written EAGER ... (0 lazy)`, so all
393600 rows exist at boot and no growth path is reachable.

**Named suspects that remain open**, for the register: the GDN/mamba chunked-scan
boundary across chunked prefill, and attention-backend / CUDA-graph / page-table
bucket sizing at these depths. Neither was examined by any shift, and this run cannot
close them because it has no failure to attribute.

## 6. HONEST NOTES
* Boot 1's stall and boot 2's clean sheet differ in the load mix, not the code. The
  deep probes drove PP1's recoveries; without them there were 0 recoveries in 1134
  cutovers. So "the recipe stops flipping after 194 cutovers" is NOT supported —
  what is supported is that **deep-prefill pressure on one rank reaches the divergence,
  and once there the instance does not flip again**.
* The round-3 record-writer crashed on my own `int("280016x")`; the probes had already
  run and their raw responses were on disk, so the results were recovered from
  `out/bb_b0*.json` into `bisect_r3_recovered.json`. The recovery re-reads
  `prompt_tokens` and `cached_tokens` from the server's own replies.
* Corridor sampling covers both boots continuously at 100 ms; no sample on any card in
  either boot went below 1024 MiB.
