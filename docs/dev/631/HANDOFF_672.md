# HANDOFF 672 — #656 / #631 Route A, successor 29

Predecessor: HANDOFF_671 (successor 28). Read its section 1 for the
item-8 verdict and the inert-flag family; this file does not repeat them.

---

## 0. THE ONE-LINE VERSION

**Rung 2 is built, wired, tested and RUNNING on metal: the drafter's
pages leave the card during PP and come back behind the same virtual
addresses, so the draft CUDA graphs stay valid.** And the rung is worth
**about a sixth of what HANDOFF_671 priced it at**, because most of what
that estimate counted belongs to the target model, not the drafter.

---

## 1. ERRORS FIRST

### 1a. The 1925 MiB payload was never the drafter's to give

HANDOFF_671 section 4 priced Direction A at **1925 MiB/rank**, read off
the boot's `Load weight end ... mem usage=` delta. The first depth=draft
boot refused at 21:02:04Z:

    'lm_head.weight' views 848035840 bytes of a 14137090816-byte storage;
    a partial view would smuggle unowned bytes into the arena (V1 scope)

14137090816 B is 13481 MiB — the **target model's weights arena**. Two
draft parameters, `model.embed_tokens.weight` and `lm_head.weight`, 809
MiB each, are VIEWS into it. The drafter shares the embedding and the
output head. Releasing them would have freed pages the target still
reads, during the phase where the target is the only thing running.

Exclusively-owned payload, measured at boot:

| rank | card | payload | vs the 1925 MiB estimate |
|---|---|---|---|
| PP0 | 5090 | **439.1 MiB** | -77% |
| PP1 | 3080 | **285.5 MiB** | -85% |
| PP2 | 3080 | **285.5 MiB** | -85% |

The binding cards are the two 3080s, so **the rung is worth 285 MiB
where it matters**.

**A memory-usage delta is not a payload.** It counts every byte that
appeared while a thing loaded, including bytes it merely aliases. Price
a spill from what the payload EXCLUSIVELY owns. The good news is that
this failure mode is now self-reporting: the carrier logs every excluded
parameter by name and size at boot.

This is the seventh capacity headline in this chain that did not survive
contact, and the **first to die before being claimed** — the arena's
V1-scope check refused the boot instead of letting a shared view be
packed and later released.

### 1b. `pgrep -f` matched my own shell, twice, in one shift

My reboot script selected the server with
`pgrep -f "sglang.launch_server.*--port 30030"`. That matches any process
whose command line CONTAINS the text — including the shell running the
pgrep. It captured a bash wrapper's **5-entry argv** as if it were the
server's 58-entry argv, killed that wrapper, and relaunched a stub.

Separately, a health-wait loop I wrote used `! pgrep -f "...30030"` in
its exit condition. It matched itself, so the condition could never
become true and the loop ran forever — and then poisoned every later
`pgrep`, which is why a subsequent boot refused with "a serving instance
is already running" when none was.

This is the **`pkill -f` self-match family the brief forbids**, twelfth
and thirteenth occurrences. Selection is structural now: argv[0] is a
python interpreter, argv[2] is exactly `sglang.launch_server`, and
`--port <PORT>` are adjacent argv entries. **Never select a process with
a regex over the whole command line. A shell that quotes the pattern is
not the process you want.**

### 1c. My verification allocated more than the thing being verified

The metal probe's first checksum was `span.to(torch.int64).sum()`, which
inflates a 192 MiB span into a 2 GiB allocation and OOMed on a card that
was also serving. Replaced by sparse boundary sampling. **A memory test
must not be the largest allocation in the test.**

### 1d. Two instrument defects that manufactured a false negative

The probe first reported "regained 80 MiB of 192" and looked like a
partial release. It was not: free memory at the end of the run was 112
MiB below free at the start *with the arena already closed* — the live
instance took that memory mid-probe, and 192 − 112 = 80. **A single
before/after pair on a shared card measures the carrier plus everything
else that moved.** Fixed by cycling three times and taking the best
regain: other processes can only make the observed regain smaller.

Then the pass/fail line compared a MiB delta against a BYTE payload and
printed `FAIL ... regained 192 MiB of 192`. A check that cannot pass is
not a check.

### 1e. Two CPU tests failed for reasons unrelated to what they test

The carrier called `snapshot_and_free(pin=True)`; pinned host memory
needs CUDA, and the family runner runs CPU tests with no GPU visible. And
`_draft_restore_bytes` read `self._census_scheduler` outside its guard,
so unit stubs built with `object.__new__` raised. Both fixed (pin only
when the payload is on CUDA; `getattr` with a default). **23 failures in
one suite run, none of them about the feature.**

---

## 2. WHAT IS BUILT

`VmmDraftWeightCarrier` in `phase_flip_spill.py`. The blocker for four
shifts was never the bytes, it was the **addresses**: the TP decode CUDA
graphs bake the drafter's parameter pointers at capture, and the dead
code restored into a freshly allocated arena, which moves them.

`KvVmmArena` splits what a normal allocation fuses — the virtual range is
reserved once and freed only at close; the physical pages underneath are
mapped and unmapped freely. So:

    spill    = decommit_range(offset, 0)   pages to the DRIVER, NVML free
                                           rises, data_ptr()s unchanged
    restore  = commit_range + arena_refill same addresses, one H2D copy,
                                           checksum verified on device

Parameters are bound onto the carrier **once**, at boot, and never
rebound. That is what makes item 8's "draft graphs stay ON" verdict and
the spill compatible at all.

Wiring, all checkable:

* **Boot** packs the carrier strictly between `build_flip_draft_worker`
  and `draft_worker.init_cuda_graphs()`, and a **pin AFTER capture**
  refuses the boot if any draft parameter landed outside the reservation.
  That corruption has no runtime symptom — wrong draft logits and a
  decaying accept rate — so it is an assertion or it is nothing.
* **Hooks in `_cutover`**, not the pre-wave site 671 proposed. On the PP
  leg `scheduler.draft_worker` is already None there, so the drafter is
  unreachable by construction. The corridor gain is unaffected because
  the binding minimum is a PP-PHASE minimum, not a seam one.
* **Affordability**: `_staging_bytes` returns
  `max(wave_peak, draft_restore_bytes)` on pp→tp, so the commit that
  could die in the no-return region is priced before the flip commits.
  **max(), not sum()** — the two peaks do not coexist, and summing would
  abandon flips that fit against a record of 0 abandons in 402.
* `IMPLEMENTED_DEPTH` 1 → 2. Rung 3 stays refused with the real reason: a
  captured graph cannot be refilled from a host image, only re-CAPTURED,
  and item 8 priced those graphs at 41% of decode.
* `depth>=2` refuses non-strict purity **at boot** — between spill and
  restore the parameters address unbacked memory, sound only because
  strict purity forbids decode in PP.

## 3. IT RUNS

Boot 21:08Z, POLICY=auto, strict purity, MTP on, decode/verify AND draft
graphs on, pool 500000, `--rank-gpu-memory-mib 31800,14000,15600`.

    carrier installed: 12 params, 439.1 / 285.5 / 285.5 MiB
    carrier pin OK after graph capture            (all three ranks)
    The server is fired up and ready to roll!
    rung 2 SPILLED  440.0 / 286.0 / 286.0 MiB to the driver,
                    parameter addresses UNCHANGED
    rung 2 RESTORED behind the SAME addresses, checksum verified

24 flips, 0 abandons, cycling continuously under load. A real request
returned HTTP 200 in 4.17 s with coherent on-topic output — the drafter
produces valid tokens after its pages were released and re-committed.

N28's warning was heeded: a boot that reaches READY is not a boot that
works, so this one was loaded before it was believed.

## 4. WHAT IT BUYS — MEASURED UNDER LOAD

7196 samples at 100 ms over 12.0 min, 4 streams, ctx 150k, pool 500000,
174 flips, **0 abandons**, occupancy 410142/500000 (82%). Same load and
sampler as HANDOFF_671 section 3, so the rows compare directly.

| card | PP min: cache -> draft | delta | TP min | binding |
|---|---|---|---|---|
| 0 (3080) | 896 -> **1496** | **+600** | 1216 | PP -> **TP** |
| 1 (5090) | 3345 -> **4149** | **+804** | 3029 | TP |
| 2 (3080) | 1210 -> **1766** | **+556** | 1414 | PP -> **TP** |

    per-card MINIMUM free 1196 / 2942 / 1396 MiB, floor 1024
    CORRIDOR HELD: True

* **The pool-500000 breach is closed.** Card 0 was 128 MiB below the
  floor in PP; it is now 472 above it.
* **671's warning came true exactly: the binding phase MOVED to TP, on
  all three cards.** Every card now binds where the drafter is resident
  by design and this rung is worth nothing. **Price the next spill
  against the TP row.**
* **The PP gain is roughly DOUBLE the payload** (+556..+804 MiB against
  285.5..439.1 MiB). The carrier also replaces ~2 GB of scattered
  per-tensor storages with one arena block, so the allocator's
  fragmentation residue goes with it. Do not credit the whole delta to
  the spill — part is a one-off layout change.
* **The worst reading of the run is now the SEAM** (1196 on card 0,
  +172), not either phase.

## 5. THE SPEC MOVED UNDER THIS WORK — items 11-14

The user issued a design escalation mid-shift (now in
`flip-setup-kapazitaets-spec.md`):

11. **Dynamic residency per (phase, load)**, PP and TP independently:
    everything not needed at the current load is wave-spilled. At bs1 the
    idle slots' mamba/GDN states go, in BOTH phases.
12. **There is no fixed "max KV".** KV is itself a spill class into host
    RAM; with 4 slots × 262144 there is a ~1M-token universe and VRAM
    holds only the working set. **The single-600k-number ladder is no
    longer the target** — the residency machinery is. 669440 stays as the
    sanity baseline.
13. **Graphs rule**: fully resident sessions MUST run with graphs;
    partially resident MAY run eager, but must return to graphs once
    fully restored.
14. **Acceptance load = REAL agent tasks over router 30099.**
    `s26_fill_load.py` is calibration only, no longer an acceptance
    carrier.

The carrier's allocation is injectable (`allocate_carrier_tensor`)
precisely so the payload classes behind the drafter — idle-slot GDN
states, cold layout bytes, session KV — are further rungs of this same
ladder rather than parallel machinery. Item 13 is the same argument the
carrier already makes: VA stability is what lets a graph survive a
release.

### 5b. ITEM 15 ARRIVED LATE AND IT RETIRES THE POOL LADDER

I had only items 11-14 in my brief; **item 15 was added to the spec at
20:48Z** and I read it only after the coordinator named it. It changes
the program:

> "AUFFUELLEN BIS ZUR GRENZE statt Layout-Kapazitaetstests ... KEINE
> statischen Pool-Treppen-Messlaeufe mehr ... Residenz regelt ein
> Laufzeit-Druckregler. Draft-Layer werden im PP-Prefill gar nicht erst
> resident gehalten."

* **No more static pool ladders.** Stepping the pool toward a fixed
  600000 is explicitly rejected. Cards are kept filled NEAR the 1024
  floor and a **runtime pressure controller** spills KV/cold as a card
  approaches it.
* Build requirements the user confirmed against operator objections:
  **(a) SPILL-BEFORE-ALLOC** — the check happens AT the allocation
  (`free - X >= 1024`, else spill synchronously first), NOT reactive
  threshold observation; **(b) two watermarks** against thrashing (free
  up to floor+delta); **(c)** when everything resident is hot, **kvso
  keeps computing over the host tier — the price is tempo, never a
  corridor breach**.
* **"Draft layers are not kept resident in PP prefill at all"** is
  precisely what rung 2 now does, so this shift is on item 15's path.

**Existing machinery to reuse, not rebuild** (surveyed, not yet wired):
`kv_pressure_ladder` (#287) + `managers/kv_pressure_runtime.py` (already
a rank-uniform, consensus-driven pressure driver with an
`on_pressure_boundary` planner hook), `kv_ladder_auto.py`,
`gdn_slot_runtime.py` (#364 idle-slot vacate), and kvso for (c). The
gap versus item 15 is that these are **boundary/threshold** driven,
while 15a demands the check sit **at the allocation**.

### 5c. ITEM 15a/15b: `CorridorGuard` IS BUILT (not yet wired)

`managers/corridor_guard.py`, 14 hermetic tests. A gate at the CALL SITE
rather than a round-boundary observer:

* arms on `free - want < floor` — against the allocation about to happen,
  not the current reading. A guard that only reads current free passes
  immediately before a breach, and that case is a test.
* frees to `floor + delta` (15b). Freeing exactly to the floor guarantees
  the next allocation spills again; the no-respill consequence is pinned.
* spends providers **cheapest first** (the reclaim-ordering law).
* **re-probes the driver after every provider** instead of trusting its
  return value — a provider that hands bytes to torch's cache has freed
  nothing NVML can see, and this chain has credited such a release before.
* **refuses** when providers are exhausted. Allocating anyway would
  launder a breach as a passed check.
* 15c (kvso host-tier continuation) is expressed as the most EXPENSIVE
  provider, not a special case, so it is reached last and still never
  breaches.

`draft_carrier_provider` adapts rung 2's carrier as the first provider.

**Not wired into a production allocation site.** That wiring wants its
own commit and its own metal supervision, and the obvious first target is
named below.

## 6. NOT DONE

* **Wiring `CorridorGuard` into the seam.** The measurement says where:
  the tightest instant of the whole cycle is now the **seam** (1196 MiB,
  +172 above the floor), and the seam's `commit_range` is the allocation
  that has already killed this instance once. Register the draft carrier
  and the GDN idle slots as providers, call `ensure_headroom` before the
  commit, and route a refusal into the existing unanimous abandon.
* **The rest of the pressure controller (item 15).** It supersedes the
  pool ladder. Measured starting point: after rung 2 every card binds in
  **TP**, so the first real payload is a TP-phase asset, not a PP one.
* Remaining item-8 arms: DFLASH × graphs, PP-prefill-graphs, chunk A/B,
  5090 stage-imbalance A/B.
* Threshold-purity arm (item 10) — note it is now REFUSED with depth>=2;
  measure it at depth<=1 or teach the carrier to stay resident.
* Final all-axes acceptance (item 2), now under item 14's real-agent
  requirement. YaRN >262k leg (item 4).

## 7. IF YOU DO ONE THING

Build **item 15a, spill-before-alloc**, on top of
`managers/kv_pressure_runtime.py` rather than starting a new controller.
The measured facts that should shape it:

* Every card now binds in **TP** (1216 / 3029 / 1414). A PP-phase payload
  buys nothing more; the drafter is done.
* The tightest instant in the whole cycle is the **seam** (1196), so the
  allocation the controller most needs to guard is the seam's
  `commit_range`, which is also the one that has killed this instance
  before.
* `VmmDraftWeightCarrier` already solves the hard parts for any payload
  class — VA stability so graphs survive a release, the affordability
  gate, and a post-capture pin. Reuse it via `allocate_carrier_tensor`;
  do not write a second carrier.

And do **not** run another static pool ladder. Item 15 rejects it in
those words.

## 8. A PROCESS NOTE THAT COST ME A ROUND TRIP

I ended a turn to wait for a 12-minute sampler, on the theory that an
armed monitor would resume me. It does not, reliably; the operator's
re-poke does, and that is a full round trip of latency. **Never end a
turn to wait.** Poll with bounded `until ...; do sleep 3; done` loops
under 60 s inside the same turn, or do desk work meanwhile. This is
occurrence 11+ of that trap in this chain.

Related: my brief carried spec items 11-14 but the user had added **item
15**. Re-read `flip-setup-kapazitaets-spec.md` at the START of a shift
rather than trusting the briefing's copy — the spec is the law, the
briefing is a snapshot of it.
