# #656 HANDOFF v22 — successor 19

Written 2026-08-10, tree `/spinning/wt-631-routea`, branch
`feat/route-a-631`.

Read this before HANDOFF_661. 661's ship number (460000) is retired here,
and the agent-traffic blocker it left open is closed — the router was never
the defect.

---

## 1. MY ERRORS, ranked — read these before my results

**0. I ran a confounded A/B and then acted on it.** I measured the corridor
at pool 460000 under one set of live agent agents, then at 380000 under a
*different* set, saw the second reading was lower, and concluded "the pool
does not control the corridor — `RANK_MIB` does". The two windows had
different loads, so the comparison established nothing. Acting on it cost
two boots and ~25 minutes: I lowered `RANK_MIB` to 30500/16550/16500 and
made the rig **worse** (852 MiB free on the 5090 *at idle*, already under
the floor). The same-load-floor law exists precisely for this, and I know
it — I skipped it because the second reading felt like a trend. **Two
measurements under two different loads are not an A/B, they are two
anecdotes.** The correct move was to hold the load fixed and change one
variable, which is what finally worked (340000 on the proven `RANK_MIB`).

**1. I dispatched agent traffic and then rebooted the server underneath
it.** Three qwen agents died mid-task with `502 router could not reach the
local endpoint`, losing their partial work. Agent traffic is a *dependency*
on a stable instance, not a background decoration. Dispatch it only against
an instance you are not about to restart, and if you must reboot, expect to
re-issue every agent.

**2. I let a subagent run Python against the live tree during a corridor
measurement.** The Explore agent executed `derive_pp_layer_split` to verify
a layer split; importing that module can initialise CUDA and place a
~300–500 MiB context on each card, which is exactly the magnitude of the
drop I was trying to explain. I did check (`nvidia-smi` showed only the
three scheduler PIDs, so it was clean) — but I checked *after* the fact.
Every read-only briefing from #3 onward carries "do NOT run python that
imports torch or touches a GPU". It should have been in briefing #1. A
measurement rig has to be defended in the briefing, not audited afterwards.

---

## 2. THE AGENT-TRAFFIC LANE: closed, and the router was innocent

This was the gate blocking green since HANDOFF_661 §9. Full evidence in
`PROD_BRINGUP_BENCH.md` §"SUCCESSOR 19 / 1".

A direct probe through the router works and lands on this instance:

    router  POST /v1/messages -> local (model=Qwen3.6-27B)
    serving [01:08:13] "POST /v1/messages HTTP/1.1" 200 OK

**The defect is the dispatch, not the plumbing.** The `qwen` /
`local-model` / `local-model-think` agent definitions carry
`model: Qwen3.6-27B`, and the router routes on an exact match of that
string. **Passing an explicit `model:` argument to the Agent tool overrides
the frontmatter**, so the request leaves as `claude-*` and goes upstream to
Anthropic. Control pair, same task, one minute apart:

| dispatch | router verdict |
|---|---|
| `subagent_type: qwen`, no model argument | `-> local (model=Qwen3.6-27B)` |
| `subagent_type: qwen`, `model: haiku` | `-> upstream (model=claude-haiku-…)` |

**The trap is that a standing memory law causes it.** `agent-modellwahl`
says "model IMMER explizit". Applied to a local-model agent that rule
silently sends the work to Anthropic and produces exactly successor 18's
symptom: agents that run correctly while the serving log stays empty.
**For qwen/local-model agents, pass no `model` argument.**

## 3. 460000 IS RETIRED. The ship number under agent traffic is 340000

Successor 18 established 460000 on a synthetic soak with **no agent
traffic**, and the user's green criterion *requires* agent traffic. Adding
it breaks the corridor on all three cards:

| pool | load | gpu0 | gpu1 (5090) | gpu2 | verdict |
|---|---|---|---|---|---|
| 460000 | soak only (s18) | 1397 | 1354 | 1451 | 0 breaches |
| 460000 | soak + agent traffic | **591** | **270** | **621** | all breach |
| 340000 | soak + agent traffic | **1733** | **1886** | **1551** | 0 breaches |

Floor is 1024 MiB. **A capacity number is only real under the load the
acceptance criterion names**, and this is the second time in this chain that
a rung was climbed under a load lighter than the bar (s18 caught the same
error going from bs=1 to bs=4; this is the bs=4→agent-traffic instance of
it).

Two properties worth carrying:

* **Free memory swings by GiB across a flip** (the KV backing
  release/restore leg), so a corridor minimum is meaningless without the
  phase it was taken in.
* **A flat corridor reading can mean an idle server.** The first 90 s of the
  460000 window read a rock-steady 1393/1352/1447 — that was the soak
  spinning up, not the corridor passing. Flatness is the tell.

340000 still leaves 527–862 MiB above the floor, so it is not the edge; the
edge is nearer 400000 and was not chased because a proven number beat an
optimal one with the clock where it was.

## 4. FULL KV >600k: the honest closure the spec asks for

The user's spec item 6 orders: at the phase change, spill everything cold of
the inactive layout — **including the PP weight shards** — to system RAM.
Successor 18 closed >600k costing only the draft assets, so this was the
right thing to re-open.

**The ordered mechanic is already implemented and already banked.**
`weights_arena.py:2-8`, `phase_flip_boot.py:477-547`: both layouts'
parameters live one at a time in ONE arena per rank sized
`max(pp_bytes, tp_bytes)`, and a flip rewrites its contents from the other
layout's **pinned host image**. The inactive layout's weights are already in
system RAM — at a cost of 59.75 GiB of pinned host memory already being
paid. Measured arena sizes:

| rank | PP layout | TP layout | arena = max | idle tail | tail freeable in |
|---|---|---|---|---|---|
| 0 (5090) | 14936 MiB | 13163 MiB | 14936 MiB | 1773 MiB | TP phase (binding) |
| 1 (3080) | 6690 MiB | 7924 MiB | 7924 MiB | 1234 MiB | PP phase (wrong side) |
| 2 (3080) | 9115 MiB | 7924 MiB | 9115 MiB | 1191 MiB | TP phase (binding) |

So the residue the spec's mechanic could still reclaim is the arena's **idle
tail**, not the shards — and unlike the draft-asset rung, two of three ranks
free it in the phase that binds.

**It still cannot fund >600k, for a structural reason rather than an
arithmetic one: the KV pool is sized ONCE at boot from a single
`mem_get_info` reading, and every runtime grow path is closed.**
`_profile_available_bytes` (`model_runner_kv_cache_mixin.py:608-700`) never
re-measures; `--max-total-tokens` is a `min()` cap that can only lower
(`:4406-4427`); `post_capture_resize_kv_pool` is off by default, shrink-only,
and excluded for the TP stack; `runtime_set_backing_tokens` is bounded by the
boot VA reservation; and `phase_flip_boot.py:636-650` refuses boot unless
`tp_capacity >= pp_capacity` because one allocator serves both layouts for
process life.

**Therefore no runtime spill of any asset can add KV capacity — freed bytes
become idle card slack.** This is the third handoff to arrive here.
**The framing itself is the defect: "can we spill at runtime to grow the
pool" is unanswerable, and the answerable question is "what pool can the rig
BOOT with and still hold the corridor".** Under the load the acceptance
criterion names, that answer is 340000 — so >600k is further away than
successor 18's negative suggested, not closer.

`phase_flip_spill.py` is written but **not wired**: `get_spill_ladder` has no
caller outside its own module and `phase_flip_spill_depth` is not a
`ServerArgs` field. Its docstring describes `_cutover` step-7b call sites
that do not exist.

## 5. THE ONE UNTRIED LEVER: a 12.1 % capacity tax from a vector mismatch

Per-rank KV bytes obey one expression, validated exactly against **three**
pool sizes (540000, 460000, 380000, 340000 — all four predicted to the
reported decimal):

    per-rank KV bytes = T x 32 KiB x max(layer_share_i, token_share_i)

The `max` is there because one arena backs both layouts. The ship config runs
`pp_layer_ratio [32,16,16]` (0.500/0.250/0.250) against token vector
`28,26,20` (0.378/0.351/0.270), so the rig pays

    max = (0.500, 0.351, 0.270),  sum = 1.121

— **112.1 % of one KV cache to store one KV cache.** Cross-check: 460000 x
32 KiB x 1.121 = 15.74 GiB, and the three measured arenas sum to
7.02 + 4.93 + 3.79 = 15.74 GiB exactly.

Aligning the two vectors drives the sum toward 1.000. **This is a boot-flag
change, not a feature** — the cheapest capacity lever in the whole chain and
the only one still untried.

**Two warnings before someone spends a shift on it.**

1. **It is in direct tension with PP compute balance.** `pp_layer_ratio` sets
   both the PP stage's compute share and its KV share. HANDOFF_661 §8 records
   the 5090 drawing ~250 of 400 W in PP prefill, which argues for giving it
   *more* layers; capacity argues for *fewer*, because its layer share (0.500)
   exceeds its token share (0.378) and is therefore the binding term. **The
   5090 cannot be both the compute-heavy PP stage and the capacity-efficient
   one.**
2. **The layer split also moves the weight arena, and that partly cancels the
   gain.** Moving rank0 32→24 layers frees ~1.77 GiB of arena there but adds
   ~1.31 GiB on rank1, and the 3080s are the cards with no slack. I costed
   `[24,23,17]` and it *breaks* both 3080s while giving the 5090 3.5 GiB it
   does not need. **Any candidate split must be costed on weights AND KV
   together, per card** — the KV-only arithmetic looks like a free win and
   is not.

## 6. State at handover

* HEAD as committed this shift; suite status recorded in the commit message.
* Serving UP on 30030, boot `06ed5afc25`, `RANK_MIB=31800,17400,17450`,
  `MAX_TOTAL_TOKENS=340000`, CTX 393216, purity strict, both fairness
  windows on, POLICY=auto.
* Boot replay: `/tmp/s18_boot.sh` (pass `RANK_MIB` and `MAX_TOTAL_TOKENS`).
  Note its defaults differ from `route_a_631_prod_boot.sh`, whose own
  defaults are NOT the proven config.
* **Boot provenance caveat:** the banner records the last *commit* plus
  `dirty_files=N`. Successor 18's run showed `commit=1c7d3c3e0b` while
  actually running the uncommitted self-merge fix — the banner alone cannot
  name the patch level, `dirty_files` is load-bearing, and the way to
  confirm a fix is live is to grep the log for its own log line
  (`SELF-MERGE REFUSED` fired 494106 times, which is how I confirmed it).

## 7. Next steps, in order

1. **Judge the green run** (armed 01:40:33Z, 65 min, T=340000, with real
   agent traffic). Evidence must show both layouts visited, prefill only in
   PP, decode only in TP, graphs live, accept-len, the pool number, the
   corridor time-series minimum, and agent tasks completed through 30099.
2. **Re-establish the capacity edge under agent traffic.** 340000 holds with
   527–862 MiB of margin; the edge is nearer 400000. One boot, same load.
3. **The alignment lever (§5)**, costed on weights *and* KV per card.
4. **Graph A/Bs (spec item 8), still unmeasured** — I did not reach them.
   NEXTN draft graphs (`SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH`), DFLASH x
   graphs. PP-prefill graphs are ANSWERED as impossible without a design
   change (HANDOFF_661 §7: the PP stack is eager by construction).
5. **Put the full-KV physics to the user** (§4). The requirement is written
   against 669440, which our own bench calls a TP-only modelled ceiling.
