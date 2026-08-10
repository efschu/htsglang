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

## 4. WHAT THAT BUYS, AND WHAT IT DOES NOT

* **It closes the pool-500000 corridor breach.** Card 0's PP minimum was
  896 MiB against the 1024 floor, a 128 MiB deficit; +285 MiB clears it.
* **It does NOT fund 600000.** That needed **+1140 MiB in PP** on the
  binding rank. 285 MiB is a quarter of it. HANDOFF_671's "A+B clears by
  39 MiB" rested on the 1925 figure and **does not survive** — and B (the
  arena tail, 319/220/1191 MiB) pays the TP phase, not PP.

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

## 6. NOT DONE

* The loaded PP/TP corridor split at depth=draft was **in flight** when
  this file was written; read `/spinning/evidence-631/s29/loaded_phase/`
  and the bench section 2p.5 if it landed.
* The pool raise. Given section 4, the next bytes are not in the
  drafter — they are in items 11/12 (idle-slot GDN states at bs<4, and
  KV itself). **Re-measure which phase binds before pricing any of it**;
  671 section 3 is emphatic and it was right.
* Remaining item-8 arms: DFLASH × graphs, PP-prefill-graphs, chunk A/B,
  5090 stage-imbalance A/B.
* Threshold-purity arm (item 10) — note it is now REFUSED with depth>=2;
  measure it at depth<=1 or teach the carrier to stay resident.
* Final all-axes acceptance (item 2), now under item 14's real-agent
  requirement. YaRN >262k leg (item 4).

## 7. IF YOU DO ONE THING

Do **not** spend another shift on the drafter — it is done and it is
small. Take the next payload class from spec item 11: at bs1 three of the
four mamba/GDN slots are idle, and those states are far larger than 285
MiB. Reuse `VmmDraftWeightCarrier`'s mechanics through
`allocate_carrier_tensor`; the hard parts (VA stability, the
affordability gate, the pin) are solved and tested.
