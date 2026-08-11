# HANDOFF 684 — #656 / #631 Route A, successor 40 (queue items #658, #661, #660)

The shift that joined the #330 budget dial to the corridor guard. The join is
smaller than the brief expected and points the OTHER way: the dial cannot be
the guard's actuator, so the guard became the dial's. Errors first.

---

## 0. THE ONE-LINE STATE

**C18 is resolved: the corridor law is now a TERM in the dial's own floor, and
an external budget REDUCTION spends the guard's relief ladder before the
dial's capacity arithmetic is asked for the residual.** The direction is
inverted from the brief because the dial disqualifies itself as relief on two
measured axes: it commits only at a fully-idle group boundary, and its answer
to "give memory back" was to flush the radix cache and shrink the pool.

---

## 1. ERRORS FIRST

### 1a. THE BRIEF'S DIRECTION IS THE WRONG ONE, AND THE DIAL SAYS SO ITSELF

The brief reads "the built runtime per-card VRAM budget dial currently
bypasses the pressure controller. Wire it: an external budget REDUCTION is
treated as CorridorGuard pressure". The first half is true, the second half is
what shipped — but the obvious reading of "wire it", namely *register the dial
as a relief provider so the guard can spend it*, is a mistake, and the reasons
are in the dial's own code:

* **It has no bounded latency.** `KvCapacityRuntime` commits at a consensus
  boundary gated on `ready_fn=scheduler.is_fully_idle` (`vram_dial.py:1222`,
  gate at `:766-777`). A guard arming inside the flip's no-return region, or
  at prefill admission, needs bytes NOW. Under load "the next fully-idle
  boundary" is unbounded.
* **Its actuator is the thing the standing rule forbids as a first resort.**
  A shrink calls `tree_cache.reset()`, `req_to_token_pool.clear()` and
  `token_to_kv_pool_allocator.resize()` (`vram_dial.py:971-980`) — it throws
  away the prefix cache and makes the pool smaller. "Never a smaller pool as
  the fix" is a standing law, and the dial had it as its DEFAULT answer.

> Registered as **law 13**: *name the LATENCY class of a relief mechanism
> before calling it relief.* A mechanism that returns memory EVENTUALLY cannot
> fund an allocation happening NOW, and the two must not sit in one ladder.

So the wiring runs the other way. The dial does not become a provider; the
dial becomes a CALLER. That is also what makes the guard the one authority
for co-residency (the #584 line): an external tenant lowering this instance's
budget now spends the same ladder, in the same tier order, that a flip seam
and a prefill admission spend.

### 1b. THE DECISION SIDE WAS ALREADY GROUP-UNIFORM — THE BRIEF'S N32/N39 WARNING DOES NOT BITE HERE

The brief carries N39's law forward ("no rank-local clocks on replicated
state") and asks for the MIN pattern to be applied. It is already there, and
this matters because building a second consensus would have been the error:

* the dial request arrives **broadcast** — `POST /vram_budget`
  (`http_server.py:1258`) → `TokenizerManager.vram_budget`
  (`tokenizer_control_mixin.py:331`) → every scheduler
  (`scheduler.py:1970`, handler `:6699`), and every rank recomputes identical
  budgets (`vram_dial.py:470-477`);
* capacity commits behind a packed **MIN reduction** with an epoch check and
  an op_seq skew hold that raises loudly rather than hanging
  (`vram_dial.py:722-777`).

Nothing in this shift runs on a clock. What the relief ladder spends is
rank-local PAYLOAD (the allocator's cached blocks, a carrier's pages), never
replicated scheduler state; the KV pool still moves only through the existing
MIN-reduced boundary. That is the same split `CorridorGuard` already draws
between its rank-local providers and `collective_kv_backing_relief`.

### 1c. C23: TWO INDEPENDENT DRIVERS OF ONE VMM RELEASE PRIMITIVE, STILL UNRECONCILED

`pool.runtime_set_backing_rows` — the only call that hands KV pages back to
the driver — has two callers that do not know about each other:

| driver | site | floor | cadence | protocol |
|---|---|---|---|---|
| `KvBackingRelief` | `kv_backing_relief.py:818`, `:941` | the corridor law (`:921`) | the flip seam | MAX reduction |
| `KvCapacityRuntime` | `vram_dial.py:986` | the budget vector | fully-idle boundary | MIN reduction |

This shift wired the DECISIONS together (the dial's floor now carries the law;
a cut spends the guard first). The two ACTUATORS still race on the same pool.
Nothing has been observed breaking and nothing has been proven safe: the dial
is off in the ship config, so they have never run together under load.
**Reopen trigger: the first boot that ships `--enable-vram-dial`.** The
question to answer then is who owns `backed_rows` when both want to move it in
one round.

---

## 2. WHAT SHIPPED (#658)

`52617c9acb`, two joins, both minimal, both default-inert.

**1. The law is a term in the dial's floor (C18).**
`corridor_law_floor_bytes()` imports `DEFAULT_FLOOR_MIB` from the guard rather
than restating 1024 — a second literal is a second source of truth — and
`_measure_local_floor_bytes` adds it to `used - backed`. Every consumer
inherits the reserve: the unit cap (`_unit_cap_for_rank`), the budget rows,
the minimum viable budget, the effective ceiling, and the below-floor refusal.

**The boot state is bit-identical, by construction.** The natural budget is
`floor + backed`, so raising the floor raises the natural budget by the same
amount and the spendable `budget - floor` does not move. The term bites only
when a caller names an ABSOLUTE budget — exactly the moment an external tenant
is taking bytes off this card and the law is what protects serving.

**2. A reduction is corridor pressure.** `apply_budget_request` now calls
`_relieve_for_reduction`, which spends `guard.ensure_headroom(D)`.
`ensure_headroom` is the exact equivalent of the situation: a budget reduction
of D bytes means someone else is about to take D bytes of this card, which is
precisely the question the gate answers. It is deliberately NOT
`lend_to_level` (bounded by the water-fill objective and capped below the host
tier — right for levelling, wrong for a tenant already promised the bytes),
and `refusal_is_fatal` is FALSE (unlike the pp->tp leg this caller has a
survivable path, so it may not force host RAM onto an unlevel fleet).

The hook is injected (`relief_fn`) like `commit_fn` and `ledger_report_fn`, so
the runtime stays a pure-math object. `None` is supported and a raising ladder
is swallowed: an external tenant asking for its own memory must never fail on
an optimisation.

### 2a. SPEC ITEM 13 IS SATISFIED STRUCTURALLY, NOT BY NEW CODE

The brief requires that restored sessions return to CUDA graphs. In this lane
the graphs are never LEFT: the dial changes physical backing behind a stable
VA and the store bound is baked at capture to cover the grown ceiling
(`graph_safe_store_bound`, `memory_pool.py:131`; enforced post-commit by
`verify_pool_reached_capacity`, `vram_dial.py:993-1038`, citing #352). The
dial states it outright at `vram_dial.py:24-25`: "No tensor moves, no
CUDA-graph re-capture." Covered by `TestGraphSafeStoreBound` (5 tests, green).

Building a re-capture would therefore have been building a second mechanism
for a problem the address stability already solves. **Its reopen trigger:** a
payload the RELIEF ladder spills is a different question from the dial's own
backing, and the drafter's restore is currently owned by the flip cycle
(`phase_flip_spill.py:735`, `:1054`), not by the dial.

---

## 3. THE EVIDENCE

### 3a. DESK

| axis | result |
|---|---|
| new fixture `test_vram_dial_corridor_658.py` | 8 passed |
| red first | yes — import error, then the ladder assertion |
| can-fail proof | disabling the hook fails exactly `test_a_reduction_asks_the_relief_ladder_first`, nothing else |
| existing dial + corridor suites | 96 passed (`test_vram_dial`, `test_corridor_guard_631`, `test_corridor_admission_631`, `test_kv_vmm_dial`) |
| #631 flip family suite | **1076 passed / 0 failed** — the inherited baseline, unmoved |
| ruff / codespell | clean |

### 3b. METAL: THE DIAL CANNOT BOOT ON THE ROUTE-A SHIP RECIPE AT ALL

**This is the shift's second finding, and it was only obtainable by trying.**
The boot was attempted at 23:31Z on the ship argv plus `--enable-vram-dial`
and `--enable-dynamic-chunking`. All three ranks raised the same error before
serving ever came up:

    RuntimeError: --enable-vram-dial: no boot capacity plan was recorded; the
    uneven-DCP token sizing path did not run for this configuration. The dial
    requires weighted uneven DCP (see DESIGN_330 section 7).

It is a REFUSAL, not a crash, and it is correct. The gate is at
`model_runner_kv_cache_mixin.py:3851-3858`; the plan it looks for is recorded
in exactly two places, `:4494-4511` (with `--kv-reshard-vectors`) and
`:4545-4553` (without), **both inside the #297 fitted-ceiling token-ratio
sizing branch** — the uneven-DCP path. The Route-A boot is PP=3 with
`dcp_size == 1`, so it sizes the pool by layer-bound PP geometry and never
enters that branch. No plan, no dial.

**What this means for #658, stated exactly.** The wiring is built, tested and
correct; it is also UNEXERCISED on metal, and will stay so on this recipe
until the dial's capacity model learns a PP layout. The desk half is not
weakened by this — C18's divergence was always a latent bug awaiting the
dial's first boot — but nobody should read this handoff as "the join was
proven under load". It was not.

**It also settles C23's reopen trigger.** "The first boot that ships
`--enable-vram-dial`" cannot happen on the ship recipe as it stands, so the
two-drivers race is further away than it looked, and the prerequisite work is
in #330's sizing path, not in the corridor lane.

**The cheapest next step, for whoever takes this:** the dial needs a boot
capacity plan for a PP layout — a single-vector plan (the `:4545` shape, one
entry keyed by the boot vector) is probably all it needs, since the PP phase
has nothing to re-raise ACROSS vectors. That is a #330 change, small, and it
is the one thing standing between #658's desk half and its metal proof.

---

## 4. #661, THE CHUNK REST

**The INFO engagement line already existed.** `_log_dynamic_chunk_engagement`
(`scheduler.py:4961`) is edge-triggered on a change of width and landed in
`88508d8cc6`, an earlier shift. It had never fired in any evidence file in
`/spinning/evidence-631/`, which is consistent rather than surprising: the arm
needs `--enable-dynamic-chunking`, which the ship config does not carry.

### 4a. THE STATIC ARM, MEASURED ON THE SHIPPED INSTANCE

Run on the ship config before any of this shift's boots, under real mixed
load, with `s31_chunk_ab.py`'s own A-vs-A floor:

| arm | prefill | noise floor |
|---|---|---|
| `static512` (ship config) | **1687 tok/s** (median of two passes: 1777 / 1598) | **10.0%** |

Corridor over that leg: **5643 samples at 100 ms, 0 breaches**, per-card
minimum free 1559 / 2360 / 1977 MiB.
Evidence: `/spinning/evidence-631/s40/phaseA/`.

### 4b. THE DYNAMIC ARM CANNOT BE MEASURED, BECAUSE IT DEADLOCKS THE BOOT

**This is the shift's third finding and the most consequential one.**
`--enable-dynamic-chunking` on the PP=3 Route-A recipe **wedges the boot**.
The HTTP port never opens. Reproduced once, diagnosed on all three ranks with
py-spy, and root-caused to a line pair:

* `Scheduler.init_chunked_prefill` (`scheduler.py:1535-1543`) wraps
  `profile_and_init_predictor()` in a **rank-local `try/except`** that sets
  `enable_dynamic_chunking = False` and continues;
* `profile_and_init_predictor` (`scheduler_pp_mixin.py`) profiles on PP0 only
  and ends in a **collective** every rank enters unconditionally,
  `pp_group.broadcast_object_list(data_to_sync, src=0)`.

On this rig PP0's profiling raised, in the log verbatim:

    [PP0] [PP Dynamic Chunk] Failed to profile prefill latency:
    alloc_req_slots runs out of memory. Please set a smaller number for
    `--max-running-requests`. req_to_token_pool.available_size()=4

The profile builds up to 128 requests while `--max-running-requests` is 4. So
PP0 caught its own failure, disabled the arm **for itself**, and walked into
the event loop; PP1 and PP2 blocked in the broadcast waiting for a src that
had already left. The three stacks, taken live:

| rank | where it was |
|---|---|
| PP0 | `_pp_commit_comm_work` — already in the event loop |
| PP1, PP2 | `broadcast_object_list` ← `profile_and_init_predictor` ← `init_chunked_prefill` ← `Scheduler.__init__` |

Compare a healthy boot: N39's restore went from "Tree cache initialized" to
"Application startup complete" in **2 seconds**. This one sat for **6+
minutes** with no output before I killed it. That gap is how to tell this
wedge from a slow boot.

> This is register law 12's sibling and it deserves naming: **a rank-local
> `except` around a collective is a deadlock waiting for the one rank that
> took a different branch.** The corridor lane already knows this — it is
> exactly why `KvBackingRelief` is built but deliberately NOT registered as a
> guard provider (`phase_flip_spill.py:1468-1477`). The dynamic-chunking path
> did not know it.

**FIXED IN THIS SHIFT** (`scheduler_pp_mixin.py`): PP0's profiling is caught
on the rank that can have it and published as DATA — an empty sample set.
Every rank still enters the broadcast, every rank receives the same empty
lists, and every rank raises the same error AFTER the collective, so the
caller's `except` disables dynamic chunking on ALL ranks, which is what it
always meant to do. The empty set is also refused before the fit, because
fitting nothing would dress a guess as a measurement.

Tests: `test_dynamic_chunk_profile_661.py`, 3 tests. The load-bearing one is a
real deadlock falsifier — the fake `broadcast_object_list` is a
`threading.Barrier` with a timeout, so a rank that never arrives fails by
timing out, exactly as it presents in production. **Can-fail proven: all 3
fail against the pre-fix file, all 3 pass with it.**

### 4c. THE VERDICT ON #617, BY THE RULE FIXED IN ADVANCE

The brief's rule: *if the dynamic arm does not beat or tie 512 without
corridor cost, book the numbers, keep 512, close #617 with evidence.*

The arm cannot be booted on this recipe at all, and now that the deadlock is
fixed it will *self-disable* on this recipe rather than run — because the
profile's 128 requests still exceed `--max-running-requests 4`, which is the
underlying condition and is NOT fixed here (fixing it means making the
profile respect the request-slot budget, a separate change).

**So: KEEP 512.** 512 remains a measured interior optimum at 1687 tok/s with a
10.0% floor and 0 breaches. #617's dynamic arm is not merely unproven on this
rig, it is unreachable without a second change. Booked with evidence.

**What a successor would need to actually run the A/B**, in order: (1) make
the profile size itself against `max_running_requests` (or profile with a
single slot, serially); (2) re-boot with the arm; (3) confirm the engagement
line fires; (4) only then compare against 4a's static number.

---

## 5. #660, THE PRE-ARENA FALSIFIER BUNDLE

Two of the three are settled with structural verdicts. The third is not, and
is handed on honestly rather than guessed.

### 5a. #646 — the flat-half-split mispairing: REAL ARITHMETIC, UNREACHABLE LANE

The pattern the issue names exists, and it is genuinely wrong when a draft
pool is present:

* `staging_handler.py:763` computes `num_kv_layers = len(kv_item_lens) // 2`,
  i.e. it assumes the buffer list is exactly `[K per layer, V per layer]`;
* `prefill.py:189-193` (and `decode.py:463-467`) **append the draft pool's
  buffers to that same list** when speculative decoding is on.

So with a draft pool the list is `[K…, V…, draft_K…, draft_V…]` and the `// 2`
no longer names the target pool's layer count — it lands in the draft region.
That is the mispairing, and it would feed a wrong `num_kv_layers` into
`compute_staging_layout`.

**It is unreachable in the ship recipe, and the reason is structural, not
incidental:** the whole file lives under `srt/disaggregation/` and runs only
in the PD lane. The ship argv carries no `--disaggregation-mode`, so the
staging handler is never constructed. Reachability condition, stated so it can
be tested rather than believed: `--disaggregation-mode != null` **and**
speculative decoding on.

**Verdict: PIN TEST, not a PRIO fix.** Reopen trigger: the first PD boot that
also enables spec. (Note the dial refuses PD outright,
`vram_dial.py:1048-1050`, so this cannot combine with #658's lane either.)

### 5b. #648 — the same pattern in the mori backend: UNREACHABLE ON THIS RIG

Same shape, two sites: `mori/conn.py:698` (`num_local_layers = len(src_descs)
// 2`) and `:704` (`dst_total_layers = len(dst_mem_descs) // 2`).

`import mori` raises `ModuleNotFoundError` in this rig's venv — the backend
is not installed — and it is a PD transfer backend besides, so it inherits
5a's gate as well. **Structural verdict: unreachable here.** Recorded with
file:line so the next person does not re-derive it; the fix, if the backend
is ever installed, is the same one 5a needs.

### 5c. #636 — the four handover preconditions: NOT SETTLED, HANDED ON

This one was delegated to a qwen lane and the lane died with the server when
I stopped serving (see §6). It is the only item of the three that needs
reading rather than grepping, because the four preconditions are prose in
`DESIGN_631b_draft_kv_wiring.md:210`/`:270`/`:286` and `HANDOFF_656.md:363`,
not a code pattern. **Do not assume the exclusive arena retired them** — that
is the hypothesis to test, not the finding. Nothing was concluded here.

---

## 6. PROCESS NOTES

* **`ARGV_SET` could not express a bare flag.** `s33_boot_from_capture.sh`
  appended `["--flag", ""]` for a `store_true` option, handing argparse an
  empty positional. Fixed in this shift; it fails in a way that reads like a
  config error rather than a quoting one, which is how it would have cost the
  next shift a boot.
* **A graceful stop really does take ~10 minutes**, as HANDOFF_683 warned.
  `/health` goes 503 within seconds while the three scheduler children keep
  the VRAM for minutes. `py-spy dump` on the parent shows the main thread
  idle in uvicorn's loop throughout — that is the drain, not a wedge, and the
  distinguishing signal is the 503 plus live children.
