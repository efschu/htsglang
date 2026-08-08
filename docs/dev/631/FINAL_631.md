# Route A (#631): PP=3 prefill, live flip, TP=3 decode with NEXTN

Final record for the standing Route A strand. One instance, three ranks,
a phase-layout flip on the same cards — not PD, not two groups.

**Target, met:** Qwen3.6-27B INT8-W8A8 on the main rig serves with PP=3
during prefill, reshards live to TP=3 for decode, with NEXTN speculation
running in the TP decode phase, and flips back.

Tree: `feat/route-a-631`. Acceptance boot: 22 (2026-08-08), commit
`6710954167`. Test family: `scripts/run_631_flip_family.sh`, 261/261.

---

## 1. The acceptance numbers

Rig: 2x RTX 3080 20 GB + 1x RTX 5090 32 GB, no P2P/NVLink. Model
Qwen3.6-27B INT8-W8A8, KV fp8_e4m3, context 65536. PP stage ratio 2,1,1;
flip vector 30,17,17. NEXTN with `--speculative-num-steps 3
--speculative-eagle-topk 1 --speculative-num-draft-tokens 4`.
Cards resolved by NVML UUID, 5090 first — never by index.

### Prefill, PP=3 phase

Uncached random `input_ids` (prefix caching cannot contaminate a repeat),
`max_new_tokens=1` to isolate prefill, warm-up draw discarded, 3 kept
draws, median reported.

| input tokens | median ms | tok/s | spread of 3 draws |
|---|---|---|---|
| 2048 | 472.5 | 4335 | 471.1 / 472.5 / 472.7 |
| 8192 | 1101.2 | 7439 | 1099.7 / 1101.2 / 1102.9 |
| 32768 | 4621.8 | 7090 | 4604.8 / 4621.8 / 4637.2 |

Reproduces the earlier rung-a measurement (473 / 1108 / 4682 ms) within
its own A-vs-A noise floor, on a different boot and with speculation
configured — so the spec slice costs the prefill phase nothing, which is
what confining the draft worker to the TP phase was for.

### Reshard (the flip itself), per rank

Server-side `PHASE-FLIP DONE` records. Ranks are PP0 (5090), PP1, PP2.

| direction | PP0 | PP1 | PP2 | live slots | bytes moved/rank |
|---|---|---|---|---|---|
| pp_to_tp (epoch 5) | 1087 ms | 1563 ms | 837 ms | 0 | 0 |
| tp_to_pp (epoch 4) | 1242 ms | 1345 ms | 982 ms | 1031 | 6.4–8.5 MiB |

The KV movement itself is ~5 ms (read 1.3–1.9, exchange 2.1–4.2, write
1.5–2.7). The remaining ~1 s is the weights-arena refill plus cutover —
so the flip cost is dominated by weights, not by KV, and does not grow
with the live set in this range. Client-observed commit-from-arm: 1586 ms
(arm RPC returns in 1.7 ms; the flip commits at the next quiescent
boundary, so the arm round trip is not the reshard duration).

### Decode, TP=3 phase, NEXTN live

`ignore_eos` so the token count is the independent variable; TTFT from a
streamed request (first token's arrival), decode rate excluding the
prefill-produced first token.

| prompt | generated | TTFT | decode tok/s |
|---|---|---|---|
| natural, 40 tok | 512 | 100.0 ms | 78.3 |
| random ids, 512 tok | 512 | 295.3 ms | 79.0 |
| natural, 40 tok | 256 | 298.7 ms | 106.1 |

**Speculation accept length: 2.23 – 3.80** (accept rate 0.41 – 0.93),
across 33 reporting intervals, with `cuda graph: True` in the decode
phase. Best sustained sample 3.80 at rate 0.93.

The quantity is the scheduler's own
`spec_num_accept_tokens / spec_num_forward_ct` — not the rolling
`spec_ema_accept_len`, which is a different number. The spread tracks
output content, exactly as this rig's benchmark notes predict; a single
accept-length figure without its content axis would be a fiction.

### Corridor (NVML free ≥ 1024 MiB per card, continuous, 100 ms sampling)

At `RANK_MIB 16400,10750,10750` the steady serving state held **≥ 2064
MiB free on every card** through the whole acceptance window (flips,
ladder, decode). The time-series minimum over the full session, however,
touched **961 MiB on the 5090 and 1004 MiB on a 3080** — three samples,
at two instants, simultaneous on all three cards: a BOOT peak, when both
stacks, the arena and the pinned images are momentarily all live. The
rule is continuous, so this counts as a breach and the budget was trimmed
rather than argued with.

**Confirmed at `RANK_MIB 16150,10550,10550` (boot 23):** minimum free,
sampled at 100 ms from before the boot through the flip and a 512-token
speculating decode, is **2338 / 1221 / 1130 MiB** — every card above 1024
at its own peak. The flip and speculation are unaffected at the trimmed
budget: flip 903 / 1154 / 1627 ms per rank, decode TTFT 101.2 ms,
75.8 tok/s, accept length 2.62. This is the configuration to run.

---

## 2. What had to be built

### Speculation as a property of the PHASE

The flip stack had zero speculation plumbing, and `pp_size > 1` refused
`speculative_algorithm` outright in two places. The constraint behind
that refusal is real: no draft worker has a PP form — the constructors
take no `pp_rank`. So the rule is not waived, it is enforced by
construction:

- the scheduler boots with `spec_algorithm` NONE and no draft worker, and
  keeps the configured algorithm aside as `flip_spec_algorithm`, so every
  spec-keyed branch in the PP phase takes the no-speculation path;
- `phase_flip_boot` builds the draft worker on the TP stack, inside the
  flip scope with the TP server args published (a draft built against the
  target's PP geometry would shard its heads for the wrong topology), and
  shares the TP stack's request pool and KV allocator exactly as
  `Scheduler.init_memory_pools` does — so draft KV is sized rank-locally
  inside the already-profiled TP budget;
- the cutover arms it with the stack it targets and disarms it on the
  return trip; `verify_flip_cutover` refuses a half-armed cutover in
  either direction.

Still refused, deliberately: `ngram` (its external corpus manager is
wired to the tokenizer channel and is not on the cutover rebuild list)
and `--speculative-draft-placement solo` (shadow-rank identity is not
modelled across a flip, where every rank changes topology).

### Bounded park, group-agreed abandon

An armed flip withholds new work so the in-flight state drains — that is
what lets it interpose *between* a request's prefill and its decode
instead of only after every stream finishes. Unbounded, it would hold the
requests of a rank that never reaches quiescence forever. A rank armed
past the deadline now joins the reduction carrying `expired`, and every
participating rank abandons on the reduced maximum. The **flip** is
abandoned, loudly; the parked requests are never touched, which is why
the abort path returns rather than raises.

The same principle then had to be applied to the pool-fit bound (§3).

---

## 3. The defect families this uncovered

Two families produced nine of the boot failures between them. Both are
worth naming, because both will produce more.

### "Rank-local state feeds a group collective"

Every wedge and half-state in this work was one rank reading something
local and acting on it where the group had to agree. The fixes all take
the same shape — put the verdict in the consensus payload:

- the park deadline (`expired` in the payload, abandon on the max);
- the pool-fit bound, which used to **raise**: rank-local sizes, so one
  rank raising while a peer proceeded would half-flip the group, and the
  raise killed a healthy server that was serving fine in its current
  phase. Nothing is mutated at that point, so the verdict is now reduced
  and the abandon is unanimous. Observed working on metal.

### "`is_draft_worker` is a construction gate, not an identity"

`is_draft_worker` means "build me as a secondary runner: no distributed
re-init, no process-global installs". It has **three** producers — a
speculative draft worker, the #274 dual-group lane, and the #631
phase-flip TP stack — and only the first holds draft weights. Sites that
mean draft-*ness* were asking the construction gate, and got the wrong
answer for the other two.

The codebase had already made this distinction once, for pools
(`is_draft_pool_worker`, whose docstring tells pool sites never to ask
`is_draft_worker` directly). It is now stated once for the general case
as `is_draft_model_runner` and used at all 25 draft/target decision sites
in the runner files plus the KV pool-config branch. The substitution is
behaviour-preserving for the other two producers — for them the flags are
equal by definition — and for the flip stack it is the fix: it reads as
the target it is.

These were unreachable before this feature, because speculation and the
flip were mutually refused. That is why they had not been found.

### Instrument defects

The collective census was counting collectives on `world_size == 1`
groups, which short-circuit without touching the wire — so it called a
correct PP=3/TP=1 configuration a desync (`tp.all_gather: counts
[536, 1096, 1096]`, the stage ratio giving rank 0 a different local layer
count, exactly as it should). And its own periodic comparison is a
blocking collective on a group that also carries payload traffic: fired
at drifted per-rank rounds it mispairs the FIFO, so the instrument seeded
the class of wedge it exists to explain. The round now rides in its
payload and a drift stands the comparison down permanently, leaving the
wedge-proof local dump armed.

### Latent bugs made ordinary

- Flipping with an **empty live set** crashed every rank: the byte view
  inferred its row width with `view(n, -1)`, ambiguous at n == 0. An idle
  server — or one whose cache was just flushed to make room for the flip
  — is the most ordinary case there is.
- The tokenizer manager indexed `spec_verify_ct` without the length check
  its sibling list already had; a spec-configured instance whose PP phase
  has no draft worker emits that field empty.
- `batch_result_processor` caches the workers and is on the decode hot
  path; built at boot, it had the boot-phase (non-spec) pair, so the
  first post-flip decode asked a `TpModelWorker` for
  `on_verify_complete_cpu`.

---

## 4. Operating notes

**Boot** (`scripts/route_a_631_flip_boot.sh`), with speculation:

```
RANK_MIB=16150,10550,10550 bash scripts/route_a_631_flip_boot.sh \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

`RANK_MIB` is trimmed from the 16400,10750,10750 used for the acceptance
run, against the boot-peak corridor breach in §1, and the trimmed values
are corridor-confirmed under load (boot 23). The acceptance ladder and
flip figures above were taken at the higher budget; the trim costs pool
size, not correctness, and was re-verified to flip and speculate.

**Flipping under a warm cache.** The flip moves the live set, which is
the radix tree's resident rows unioned with the parked requests'. A
prefill ladder leaves thousands of cached rows that the *target* pool
must also hold, and the TP pool is smaller than the PP pool — more so now
that a draft-KV allocation shares its budget. When it does not fit, the
flip abandons cleanly and serving continues; to make it fit, flush first:

```
curl "http://127.0.0.1:30023/flush_cache?timeout=60"
```

The `timeout` matters. Without it the flush is refused outright while any
request is running or waiting, silently leaving the rows resident.

**Measuring.** `scripts/route_a_631_acceptance.py full` drives the whole
sequence and emits one json record. TTFT comes from a streamed request,
decode rate excludes the prefill-produced first token, `ignore_eos` pins
the token count, and prefill is measured in the PP phase with decode in
the TP phase — a run reporting both from one phase has not measured
Route A.

---

## 5. Residuals, named

- **Per-request spec metrics are not populated on this path.**
  `meta_info["spec_accept_length"]` comes back null for post-flip decodes
  even while the scheduler's own per-interval accept length is correct
  and non-zero. The authoritative number is therefore the scheduler
  metric, and the acceptance figures above use it. Worth plumbing, but it
  is an observability gap, not a correctness one.
- ~~Boot-peak corridor.~~ Closed: trimmed budget confirmed under load at
  2338 / 1221 / 1130 MiB minimum free (§1).
- **Scheduler token/memory info is not re-read at cutover.**
  `max_total_num_tokens` and friends are read from the model worker at
  boot and kept. The two stacks' pools differ. It has not misbehaved, and
  it is not part of this acceptance, but it is an assumption rather than
  a verified invariant.
- Deferred by standing decision: #636 handover preconditions doc, #652
  (the 5090's CUDA context sees only 19.58 GiB total), cold-image disk
  spill design.

## 6. Evidence

`/tmp/route-a-631/`: `acceptance_w3.json` (prefill ladder + both flips),
`tp_decode_spec.json`, `tp_decode_natural.json`,
`flip_and_spec_stats.json` (per-rank flip records, accept-length
samples), `corridor_w3_stream.csv` (100 ms NVML free), and the per-boot
server logs `server.log.boot*` — each named for the defect it found.
