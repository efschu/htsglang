# HANDOFF: #656 KV-UNIVERSE R1 — four tickets, errors first

Branch `feat/kv-universe-656`, cut from `origin/integration/r2` and moved
forward onto **Merge R6 `481411ac6b`**. Comparison baseline for the code arm
is the frozen `/spinning/wt-merge-r6-base` tree (R6 kept it at the pre-merge
tree `598f570ba4`); the metal arm compares against the live ship capture
`/spinning/evidence-631/s485/ship_argv.txt`.

Evidence: `/spinning/evidence-631/kvuniverse-r1/` (RESULTS.md, boot logs,
generations). GPU window 22:11Z–22:30Z, released with the heartbeat stopped
first. Serving downtime ~17 min; ship restored and verified with three real
generations.

---

## 0. ERRORS FIRST — three premises this shift falsified

**E1. "Remove the `max_total_num_tokens` cap" is falsified ON METAL.**
Boot A set `--max-total-tokens 100000000`. The boot DIED:

```
cuMemCreate: committing 8388608 bytes would leave 116 MiB free, below the
1024 MiB corridor law floor ... would leave -4 MiB free
cuMemCreate: 8388608 bytes refused by the driver
RuntimeError: cuMemCreate failed: <CUresult.CUDA_ERROR_OUT_OF_MEMORY: 2>
```

The cap is **not** conservatism and not a soft flag that can simply be
deleted. Uncapped, the TP-phase pool sizes itself to its *own* budget and
eats the VRAM the PP pool needs. The order "the cap is to be REMOVED, not
justified" cannot be executed as a flag change; it requires the TP pool to be
**sized to the PP id space** in code. That code is NOT written (see §5).

**E2. "Move layers onto the 5090 to raise KV" is falsified ON METAL.**
Boot B set `--pp-stage-ratio 18,7,7`. PP0 died:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 810.00 MiB.
GPU 0 has a total capacity of 31.34 GiB of which 421.75 MiB is free.
```

Two facts behind it, both worth carrying forward:

* **torch gpu0 IS the 5090** (FASTEST_FIRST ordering); NVML index 1 is the
  5090. The rank→card mapping is NOT NVML order. Any reasoning that assumes
  `--rank-gpu-id 0,1,2` follows `nvidia-smi` order is wrong.
* **rank0 was already at its physical ceiling**: budget 31800 MiB of a 32607
  MiB card leaves 807 MiB — *below* the 1024 MiB corridor floor. The 5090 has
  no room to receive layers.

**E3. HANDOFF_662 §5's arithmetic for `[24,23,17]` used the wrong cost model.**
662 costed a layer move as *additive* on the receiving rank ("frees ~1.77 GiB
on rank0 but adds ~1.31 GiB on rank1"). The weight arena is a **max over
layouts**, not a sum — `phase_flip_boot.py`:

```python
arena_total = max(layout_pp.total_bytes, layout_tp.total_bytes)
```

So a rank whose TP weight share already exceeds its new PP layer share pays
**nothing** for received layers. Measured on the ship boot: PP weights
13482.18 / 8144.00 / 9114.95 MiB against TP weights 13692.29 / 7659.52 /
7659.52 — ranks 1 and 2 sit at their PP term, rank 0 at its TP term. 662's
conclusion (don't do it) still held for rank0, but for the reason in E2, not
the reason 662 gave.

---

## 1. T1 — PP-phase ceiling: WHAT BINDS IT, and the fix

**The binding constraint, named with numbers.** The pool is a **min-reduction
across ranks**. Boot A lifted the cap so every rank reports its true local
capacity under the ship vector `14,10,8`:

| rank | true local capacity | stranded |
|---|---|---|
| PP0 | 1252026 | 748076 |
| PP1 | **503950** | 0 — **THIS RANK BINDS** |
| PP2 | 727592 | 223642 |

The rig was serving at **503950 tokens while 971718 token-slots were
physically present and unusable** on the other two ranks.

**Why nobody could see it:** the ship cap of 620000 made PP0 and PP2 each
report exactly `620000`. The cap did not lower the pool (the min was below
it) — it **masked the imbalance that would have told you where the capacity
went.** That is the real cost of the cap, and it is why T1 and T2 are one
ticket.

**Why the token vector alone cannot fix it.** Per-rank KV cost is
`T x 32 KiB x max(layer_share, token_share)` (HANDOFF_662 §5, HANDOFF_663).
rank1's cost is **floored by its layer share**, so lowering only
`SGLANG_UNEVEN_TOKEN_VECTOR` buys nothing on the binding rank. Either the
layers move (E2 says they cannot move onto the 5090) or the budget changes.

**The fix, metal-proven (boot C).** The stranded resource is not the layer
ratio at all: `--rank-gpu-memory-mib 31800,14000,15600` gave the two RTX 3080s
**14000 and 15600 MiB out of 20480 each — 11.3 GiB of VRAM the engine was
never handed.** Boot C keeps the ship vector `14,10,8` untouched and only
rebudgets to `31500,16000,16000` (rank0 down 300 MiB to buy back corridor
headroom on the 5090):

| quantity | ship | boot C |
|---|---|---|
| pool / id space | 503950 | **620000 (+23.0 %)** |
| ranks stranding capacity | 2 of 3 (971718 tok) | **0 of 3** |
| binding rank | PP1 | none — the CAP binds |
| tracebacks | — | 0 |
| server | — | fired up 22:23:52 |
| real generations | 3 (ship baseline) | **3**, accept 2.29–2.67 |
| NVML free, idle | — | 1879 / 3460 / 3261 MiB |

Every rank now reports `THIS RANK BINDS; 0 stranded`. 620000 is the cap, not
a rank — **the PP phase is no longer the ceiling.**

**Is 620000 the optimum? No, and it is not claimed to be.** 855 / 2436 / 2237
MiB remain above the 1024 corridor floor, so more is backable. The next
binding thing is the TP-phase pool (§2), not the PP layout.

---

## 2. T2 — the 500000/620000 cap

**Where it is set:** nowhere in `python/sglang`. It is the boot flag
`--max-total-tokens` (`server_args.py:1151`, default `None`), passed through
verbatim to `MemoryPoolConfig.max_total_num_tokens` and written as `self.size`
at `mem_cache/memory_pool.py:2481`. No `min()`, no clamp, no assertion.

**Is it structural? No.** `ReqToTokenPool.req_to_token` is
`dtype=torch.int32` (`memory_pool.py:373-375`) — ~2.1e9 addressable, four
orders of magnitude above any pool discussed here. **There is no integer
width or allocator row limit to widen.** The order to widen one has no target.

**So why can it not just be deleted?** Because of E1: removing it OOMs the
boot. The documented reason is confirmed by the boot A failure —
`PROD_BRINGUP_BENCH.md:642-663`: the TP layout can only address ids the PP
allocator issues, so an uncapped TP pool allocates VRAM it can never address
and starves the PP pool.

**Verdict: NOT DELIVERED as ordered, and the order as written is not
executable.** The honest fix is not "remove the number" but "stop there being
a number": size the TP-phase pool to the PP id space, so the cap is a
*derived* quantity rather than an operator guess, and it rises automatically
whenever T1 raises the id space. That change is specified in §5 and is not
written. What IS delivered is that the cap no longer *masks* anything: after
boot C every rank reports zero stranded, so the cap's damage is now bounded
and visible.

---

## 3. T3 — YaRN toward 1M, and the bug in the way

**The zero-upfront-cost falsifier ran first, as ordered, and it found a
blocker that is not a memory cost at all.**

Structures scaling with the context ceiling, 393216 → 1048576:

| site | scales with | Δ at mrr=4 |
|---|---|---|
| `memory_pool.py:373` `req_to_token` | ceiling × (max_running_requests+1), int32 | **12.5 MiB** |
| `utils/common.py:5049` RoPE `cos_sin_cache` eager reserve | ceiling × rotary_dim, fp32, per instance | **~320 MiB @ rotary_dim 128** |
| `flashinfer_backend.py:1784` decode-graph kv_indices | ceiling × max_num_tokens | ~10 MiB |
| `flashinfer_backend.py:7741` MTP draft kv_indices | ceiling × topk × steps | ~5 MiB |
| `flashinfer_backend.py:1978` full-CG prefill kv_indices | ceiling × num_slots | up to ~4.2 GB — **conditional, not in the ship graph mode** |

The named prime suspect (`req_to_token`) is **not** the cost. The dominant
one is `reserve_rope_cache_for_long_sequences`, called unconditionally from
`model_runner.py:2496` before graph capture — it pre-expands the cos/sin cache
to the *full ceiling* at boot. It defeats laziness deliberately: growing the
cache after capture would reallocate a tensor whose address the graphs have
baked in (the same silent-corruption class as `phase_flip_boot.py` §4c).

**The blocker found on the way — a silent correctness bug, now FIXED.**
`YaRNScalingRotaryEmbedding` builds its cache with
`_compute_inv_freq(self.scaling_factor)` and `* self.mscale`, but does not
override `_ensure_cos_sin_cache_length`. The inherited base growth path used
`self._compute_inv_freq(self.base)` — **passing the RoPE theta where a
scaling factor is expected — and applied no mscale at all.** Every appended
row carried different frequencies and the wrong amplitude from the rows before
it. No exception; just wrong attention past the boot cache length.

Three classes share the defect: `YaRNScalingRotaryEmbedding` (yarn.py),
`DeepseekScalingRotaryEmbedding` (rope_variant.py), `YaRNScalingMRotaryEmbedding`
(mrope.py).

It fires **today**, latently: the boot cache is
`max_position_embeddings × scaling_factor` = 393216 rows, while
`reserve_rope_cache_for_long_sequences` asks for
`ceiling + steps×draft×2 + 256` aligned to 128 = 393600 — so ~384 rows are
already appended through the broken path at every boot. They are unreachable
only because `context_len` caps positions at 393215. **Raising the ceiling is
exactly the change that makes them reachable**, which is why this had to be
fixed before any 1M work, not after.

Fix: an overridable hook pair on the base class
(`_cos_sin_cache_inv_freq`, `_cos_sin_cache_row_scale`), overridden in all
three scaled variants so the growth path extends with the parameters the
cache was built with.

Test `test/srt/test_yarn_rope_cache_growth.py`, three cases (formula match,
seam continuity, mscale applied). **Can-fail proof, no `git stash`:**

| arm | tree | result |
|---|---|---|
| fixed | `/spinning/wt-kv-universe/python` | **3 passed** |
| unfixed | `/spinning/wt-merge-r6-base/python` (hook absent, verified) | **3 failed** |

Import path was printed in both arms — this is not the worktree-PYTHONPATH
false-red.

**NOT DELIVERED for T3:** no 1M boot, no corridor sample at 1M, no
admitted-at-1M session, no decode past 393216. The measured Δ above is from
code arithmetic, not from a boot. The lazy/VA-stable conversion of the RoPE
reserve (the actual "zero cost until used" mechanism, §5) is not written.

---

## 4. T4 — PP-after-spill composition

**NOT STARTED.** No falsifier constructed, no verdict. The window went to T1's
three boots and the ship restore. Nothing is claimed about it. The specific
hazard remains as briefed: the spilled session's `req_to_token` sentinel rows
and host-tier regions crossing a KV reshard, with silent corruption as the
failure class to hunt.

---

## 5. THE WORK THIS SHIFT SPECIFIES BUT DID NOT WRITE

1. **Size the TP pool to the PP id space.** Removes the `--max-total-tokens`
   guess entirely and makes T2's order executable. Boot A is the proof it is
   needed; without it, no cap change is safe.
2. **VA-stable RoPE cache reserve.** Reserve virtual address space for the
   full ceiling and commit pages on use, so the address graphs bake stays
   fixed while the cost stays proportional to what is used. The machinery
   already exists in-tree: `VmmWeightsArenaCarrier` / the RUNG 3 reservation
   in `phase_flip_boot.py`. This is what makes "1M grantable at zero upfront
   cost" true rather than asserted.
3. **Re-solve the vector after rebudgeting.** Boot C left 855/2436/2237 MiB
   above the corridor floor. With the 3080s finally holding their own cards
   the optimal stage vector moves (desk estimate `15,9,8`), but that must be
   re-measured against boot-C-class budgets, not against the ship numbers.
4. **Corridor under sustained load.** Boot C was sampled at idle and after
   three generations, NOT as a continuous 100 ms minimum under load. The
   corridor law is a continuous-minimum law; boot C is not yet a corridor
   acceptance.

---

# R3 (2026-08-13): the seam is a sizing post, and the policy can now hear "no"

Branch `feat/kv-universe-656`. Evidence
`/spinning/evidence-631/kvuniverse-r2/` (RESULTS.md, boot_f/g/h/i/i2/j logs,
corridor CSVs, load transcripts). Window 23:45Z-06:15Z; serving stopped and
restored by this session, verified with three real generations + health 200.

## 6. ERRORS FIRST -- four premises this shift falsified, three of them mine

**E6. The seam measurement read ZERO in the phase it was taken in.** Boot F:
`floor 0 MiB` on every rank, against boot E's refusal naming 464 MiB. Both of
the runtime's accessors are STATE readings -- `pending_tail_bytes` is
`want - committed`, `pending_restore_bytes` is 0 unless CURRENTLY spilled --
and at the first round the instance sits in PP with the arena committed and
nothing spilled. The 464 MiB is what rung 3 releases on ENTERING TP. **A
requirement measured in the phase that does not pay it reads zero.**

**E7. `src.num_rows` is not the id space.** Same rank, same 1396 MiB of slack:
2360.7 B/row on `pp_to_tp`, 5393.8 on `tp_to_pp`, because under the TP layout
`num_rows` is the rank's token SHARE (rank2: 170793 against 683151). Any
per-token coefficient must be normalised by the GLOBAL id space, which is what
the sizer's `T` means.

**E8. A sizing correction keyed to the post-capture path never fires on this
config.** Boot H applied nothing and logged nothing: the pool came back at
683150 unchanged. The ship pool is decided PRE-capture. The fix is not just
the hook location (`_config_from_budget`, the one funnel all three paths
reach) but the SHAPE of the term: the pre-capture path has no "headroom"
quantity to subtract from, so the correction is now anchored on a MEASURED
position -- bytes actually spendable above the corridor law at a known id
space -- and needs no model of the activation reserve, capture peak, arena, TP
stack or carve-out, because all of them were already resident when it was
measured.

**E9. An edit deleted the measurement half and every unit test still passed.**
Boot I died on `ImportError: cannot import name 'measure_and_record'`. The
tests covered the arithmetic, which survived; nothing imported the deleted
names the way the CALLERS import them. Pinned now by
`test_every_symbol_the_callers_import_exists`, one assertion per real call
site.

## 7. T1 -- the sizer works, and the honest pool is SMALLER than 620000

The measurement now reproduces the runtime's refusal from the other side of
the code. Boot G at 683150:

| rank | seam needs at rest | spendable above the law | verdict |
|---|---|---|---|
| 0 | 455 MiB | 2002 MiB | fundable |
| 1 | 484 MiB | **8 MiB** | cannot fund its own flip |
| 2 | **1455 MiB** | 524 MiB | cannot fund its own flip |

rank2's 1455 MiB is the tail boot E logged as `rung 3 released 1436.0 MiB`.
The ERROR line naming this fires BEFORE the instance wedges -- boot G then did
wedge (empty `/generate` at 85 s), which is the self-diagnosis boot E lacked.

**Boot J (acceptance, commit `5a4ff94208`, no `--max-total-tokens`):** pool
**563974**, **24 completed cutovers, 0 abandons, 0 refusals, 0 tracebacks**,
real generations with speculation (accept 2.46-3.56), a **64001-token
prefill**, and a continuous 100 ms corridor minimum under load of
**1634 / 2845 / 1804 MiB with 0 breaches**.

Per-rank allowed id space: 712159 / 634396 / **563974** -- rank2 binds.

**563974 is BELOW the 620000 serving-proven default, so the quarantine STAYS
and the task's "land above 620000" is NOT met.** This is not the sizer being
conservative: it is the sizer pricing a cost boot E was paying with a wedge.
At budgets `31583,15750,18205` rank2 is handed 18205 MiB and keeps only 524
MiB spendable, while its arena tail alone needs 1455.

**THE NEXT STEP IS A BUDGET RE-SOLVE, NOT A CAP CHANGE.** The budget vector
was solved (r2) against the corridor law ALONE. It must be re-solved against
`corridor + seam floor` per rank, where the seam floor is now a measured
number on disk: 455 / 484 / 1455 MiB. rank2 is over-budgeted for a card that
has to carry a 1455 MiB arena tail; moving budget from rank2 to rank0 (which
has 2002 MiB spare above its own seam) is the obvious first candidate, and
unlike every previous vector guess it can be checked against a recorded
per-rank requirement before a boot is spent on it.

## 8. T2 -- the control layer, proven on metal

Boot G held the exact boot-E condition and produced **5 arms, 17 arm refusals,
9 abandons, 3 completed cutovers** against boot E's **179 arms, 0 cutovers**.
Three events are now distinct in the policy: ARMED is provisional, ACCEPTED is
not COMPLETED (`arm()` returns True for the first `SEAM_ABANDON_CAP` attempts
of an unfundable configuration), and only a cutover retires the attempt.

## 9. Carry forward

* The 620000 quarantine stays. Removing it needs a budget vector whose seam
  floors fit, not a code change.
* `flips_completed` is the metric. Arm counts read boot E as healthy.
* T3 (YaRN 1M) and T4 (PP-after-spill) carry no metal from this shift. T3's
  RoPE cache-growth fix is still on the branch, can-fail proven, unbooted.
* T5 is closed: the additive weight-arena model was never implemented in code
  or tests -- prose only, corrected in place in HANDOFF_662.md.

## 10. #364 idle-vacate: BUILT, NOT ENGAGED (found 2026-08-13, not fixed)

Ordered mid-shift: verify that idle GDN/Mamba state (~147 MiB per slot,
~440 MiB for three idle slots) vacates into the KV pool, and that the sizer
counts it as reclaimable mass.

**(a) It does not fire.** Zero occurrences of `vacat` and zero of
`resident state slots` across boot_f, boot_g and boot_j. The gate is
`if self.server_args.gdn_resident_state_slots is not None:` in the
scheduler's between-tick block, and `--gdn-resident-state-slots` is in
NEITHER the ship argv capture nor the uncapped flip argv. Unset flag ->
`gdn_slot_executor` stays None -> the ladder never runs. The machinery is
built and correct-looking; it is simply not switched on for this path.

**(b) The sizer does not count it, and currently must not.** The mamba
reservation is its own budget post (`MAMBA_BUDGET_POST`) subtracted before
KV, and the seam reserve's measured anchor (`have_bytes`) is read with every
slot reserved. So vacatable bytes are subtracted as a fixed post AND absent
from the spendable measurement -- counted as unavailable twice, reclaimable
never. Making the sizer bank them while the vacate is off would size a pool
against memory nothing releases.

**(c) The bs1 leg is NOT proven.** No boot with the flag, no vacated-byte
count, no wake-on-slot-2 determined-answer probe. Nothing is claimed.

**The order of work is (a) then (b), and it is not optional.** ~440 MiB is
the same order as the seam shortfalls that bind this rig (rank1 short 476
MiB, rank2 short 931 MiB at boot G), so this is a live candidate for the
budget re-solve in section 7 -- but only once the vacate is engaged and its
freed bytes are COUNTED in a log. Until then the seam floors 455/484/1455
MiB stand as measured, against slots that stay reserved.

# R4 (2026-08-13): the vacate is engaged, and the wall is the layout, not the budget

Branch `feat/kv-universe-656`. Evidence `/spinning/evidence-631/kvuniverse-r3/`
(RESULTS.md, boot_k1/k2/k3 logs, corridor CSVs, load transcripts). Window
06:22Z-08:0xZ; serving stopped and restored by this session (res-r5 script,
only SELF and LOG changed), verified health 200 + three real generations.

## 11. ERRORS FIRST -- four premises this shift falsified, two of them R3's

**E10. The idle-vacate credit is ~4x smaller than R3 priced it, and correctly
so.** `--gdn-resident-state-slots 4` against a profiled 12 banks
0.26/0.18/0.15 GB per stack, not ~440 MiB per rank. The cap shrinks
`ssm_state`+`conv_state`; it does NOT shrink
`intermediate_ssm_state_cache` (0.62/0.44/0.35 GB, the largest component),
which is sized from `max_running_requests x draft_tokens` -- from ADMISSION,
which slice 3 deliberately decouples from the cap.

**E11. kvso is refused on the flip path at parse time**: `--enable-kv-session-
offload (S1) supports single-node pure TP/DCP only (pp_size=3)`, with no
opt-in env beside the `KVSO_ALLOW_SPEC`/`KVSO_ALLOW_HICACHE` pair. kvso is the
ladder's named standing population (`live_offload_reqs`), so on a PP=3 flip
instance the runtime vacate has nothing to act on. R3's entry 57 read this as
"the flag is not in the argv"; the flag is necessary but NOT sufficient.

**E12. The budget vector is inert for the binding rank.** `have_bytes` is
`torch.cuda.mem_get_info()[0] - law` -- PHYSICAL free VRAM. rank2's pool is
capped by its seam, not its budget (4406 MiB used of an 8438 MiB KV budget),
so raising `--rank-gpu-memory-mib` allocates nothing and frees nothing.
**R3's carry-forward "the next step is a BUDGET re-solve" is wrong**, and the
task's own fallback clause applies: the why is arithmetic, in section 13.

**E13. The seam solver targets equality with the floor, so it sizes a boot
with zero margin.** K3 derived 610942, then re-measured rank2 at 1430 MiB
against its 1455 MiB floor and logged `CANNOT FUND ITS OWN FLIP`. All 30
flips completed anyway -- the line is conservative -- but the sizer books no
margin, and the flip gate separately wants 512 MiB of C20 entry margin that
the sizer never reserves.

## 12. T1 -- ENGAGED, MEASURED, BANKED; wake-correctness NOT proven

Engaging is argv-only; the banking route is boot-time and automatic (capped
`max_mamba_cache_size` -> smaller `MAMBA_BUDGET_POST` -> the physical-free
`have` anchor rises by the same bytes). No sizer change was written or needed.

* The cap fires on all three ranks with a byte-naming log line, and
  `GDN-SLOT-LADDER armed: 4 resident state slots` on all three.
* Credit MEASURED at an identical pinned pool (563974) against boot J:
  have_m **3744->4306 / 1224->1574 / 1514->1822 MiB** (+562/+350/+308).
* Boot K1 serves under the cap: 30 cutovers (15/15), 0 abandons, 0 refusals,
  0 tracebacks, 64001-token prefill, corridor min 2082/3341/2128, 0 breaches.
* The runtime vacate NEVER fired and structurally cannot here (E11), so
  **wake-correctness is not claimed**. The banked bytes do not depend on it
  (slice 1 is boot-time); slice 3's admission decoupling DOES -- see risk.

**RISK, single-sourced, not proven.** `session_admission_slots(...,
vacate_available=_resident_cap is not None)` keys "the overflow has a backer"
on THE FLAG, not on the ladder's reachable inventory. Cap 4 + ratio 2 admits
4 requests against 2 requests' worth of slots, and actives are never vacate
victims. Falsifier: a genuine bs=4 concurrent load on a capped flip boot.
Expect a stall, not an OOM -- untested. Fix: derive `vacate_available` from
inventory reachability.

## 13. T2 -- the honest optimum at this layout is 610942, and the lever is the layout

Boot K3, derived with no operator number: **pool 610942** (up from 563974,
+8.3%), per-rank allowed 754642/675579/610942, **30 completed cutovers, 0
abandons, 0 refusals, 0 tracebacks**, 64001-token prefill, corridor
**1510/2715/1994, 0 breaches**. **610942 < 620000: the quarantine STAYS and
was not removed.**

The pool is bounded by rank2's physical free memory:

    free at rest 2846  -  seam floor 1455  -  corridor law 1024  =  367 MiB
    367 MiB / 8192 B per token  =  46976 tokens   ->  563974 + 46976 = 610950

against the sizer's derived 610942. No budget term appears. rank2 holds
4032 MiB of unusable KV-budget slack.

Reaching 620000 needs **438 MiB more free memory on rank2**. Levers, by size:

1. **rank2's arena tail, 1455 MiB** (rank0 455, rank1 484) --
   `max(0, pp_bytes - tp_bytes)` over rank2's two layouts, i.e. a function of
   `--pp-stage-ratio 14,10,8` vs `--phase-flip-tp-vector 32,16,16`. THIS IS
   THE LEVER. It also lowers rank2's 8192 B/token cell. This is the user law
   "PP layout/budgets follow the KV target" taken literally.
2. cap 4->2 buys ~77 MiB on rank2; 4->1 ~115 MiB. Both cost concurrency.
3. `--max-running-requests 4->2` halves `intermediate_ssm_state_cache`
   (~179 MiB on rank2), changes the fingerprint and the serving contract.

2+3 together still fall short of 438 MiB. Next shift should re-solve the
STAGE RATIO against the measured per-rank arena tails, and must re-measure
the seam per layout -- `pp_stage_ratio` is NOT in the record fingerprint.

## 14. T4 -- answered without a boot: REFUSAL at parse time

The composition "spilled kvso session across a PP-prefill flip" cannot be
constructed on this configuration: kvso refuses `pp_size>1` before the server
starts (E11). Verdict is refusal -- not crash, not silent corruption -- and it
needs no new regression test; the existing validation is explicit and has no
override. Consequence: the #549 GDN-vacate-x-kvso fixes are unreachable on the
flip path and are exercised only under pure TP/DCP.

## 15. Carry forward

* The 620000 quarantine stays. It is now bounded by a MEASURED physical
  quantity (rank2's free memory vs its 1455 MiB arena tail), not by a guess.
* T3 (YaRN 1M) still carries no metal. RoPE cache-growth fix still unbooted.
* The seam record fingerprint omits `gdn_resident_state_slots`,
  `enable_kv_session_offload`, `pp_stage_ratio`, `phase_flip_tp_vector` and
  `max_total_tokens`. Any layout experiment MUST account for that.

# R4 continuation: the quarantine is deleted, and the tail was a weights term

## 16. THE ARENA TAIL IS `PP_weights - TP_weights`, confirmed three ways

R4's first half named rank2's 1455 MiB seam floor as "the layout" without
saying which knob. It is this, and the runtime says so itself on boot L3:

    rung 3 released 924.0 MiB of weights-arena tail (TP layout needs 8188.4 of 9115.0 MiB)  <- rank2
    rung 3 released 466.0 MiB (TP needs 7659.5 of 8144.0)                                   <- rank1
    rung 3 released 300.0 MiB (TP needs 13163.5 of 13482.2)                                 <- rank0

which reproduces register 46's measured PP/TP weight vectors exactly, and
reproduces the measured floors 0/484/1455 under the old vector.

**The cheap knob is `--phase-flip-tp-vector`, not `--pp-stage-ratio`.** Raising
the binding rank's TP share shrinks `max(0, PP - TP)` without moving `have`,
because `have` is measured at rest in the PP phase, where rung 3 has already
released the TP arena. Moving PP layers instead costs the receiving rank
~1304.9 MiB of `have` per stage unit (its PP weights AND its KV both grow),
and modelling that is what made the (16,10,6) and (14,9,7) candidates come
out WORSE than the status quo.

## 17. T2 CLOSED -- quarantine deleted, 648388 derived and served

`--phase-flip-tp-vector 30,16,18` drops rank2's tail 1455 -> 927 MiB.
Boot L3 (commit `9fc98e8649`, constant removed): derived pool **648388**,
**+28388 above the 620000 it replaced**, no capping warning, **24 completed
cutovers (12/12), 0 abandons, 0 refusals, 0 tracebacks, 0 CANNOT FUND**,
64001-token prefill, accept 2.46-2.67, corridor **1128/2567/1664, 0 breaches**
over 5991 samples.

Every rank re-measured ABOVE its floor at the derived pool -- +2873 / +224 /
+173 MiB -- against boot K3's -25 MiB. That is the margin term working.

**The arithmetic now predicts the sizer**: L2 predicted 651398 vs derived
651498, L3 predicted 648288 vs derived **648388**. 100 tokens both times.
Future layout work should predict first and spend the boot on confirmation.

**What replaced the constant** is the mechanism, not another number:
seam_reserve_enabled() defaults True, so a pool sized as "VRAM minus corridor"
with nothing left for the seam cannot be built by accident; the solver holds
a margin; the policy counts refusals. The gate test asserts those, not 620000.

**Careful with boot L2**: it ran CLAMPED at 620000 (pre-removal commit), so
its numbers prove the LAYOUT, not the removal. L3 proves the removal.

## 18. Open, and honest

* **The bs=4 falsifier came back BENIGN.** Four concurrent sessions on a
  4-slot pool all completed correctly; concurrency degraded to 3 with 1
  queued, mamba usage 1.00, 6x "mamba slot pool exhausted ... skipping this
  cache insert", 0 OOM. `vacate_available` is optimistic but the scheduler's
  own slot check gates admission first, so the cost is throughput and state
  caching, not correctness. Fix is documentation: "sessions beyond the cap run
  with a vacated (host-blob) state" is FALSE on a PP flip boot -- they WAIT.
* **T4's refusal must not be read as the spill requirement being unmet.** The
  phase-flip spill ladder fires 18x per direction on flip boots (rung 1 cache,
  rung 2 draft weights, rung 3 arena tail) and is what makes the seam
  affordable. Only the kvso SESSION KV tail on the host is unavailable under
  PP, and with it #549's fixes.
* **Corridor is now genuinely tight**: 1128 MiB minimum on GPU0, 104 above the
  law, 0 breaches. That is the user law's "frei nahe 1024" rather than slack
  to spend. A further pool raise needs freed bytes, not a bigger number.
* **The ship env sets `SGLANG_CORRIDOR_FLOOR_MIB=1536`**, so the flip GATE
  arms against 1536 while the corridor VERDICT is read against 1024. Both
  statements are true at once and the runtime logs the distinction; do not
  "fix" one into the other.
* **T3 (YaRN 1M) still carries no metal.** Note it now costs a cold seam
  record: `context_length` is in the fingerprint, so a 1048576 boot orphans
  the 393216 record and its first boot sizes uncorrected. Pin
  `--max-total-tokens` on that first boot exactly as L1 did.

# R4 T3: the 1M ceiling reached, priced, and proven correct past 393216

## 19. Reaching 1048576 -- the route, and two dead ends worth not repeating

The checkpoint carries `rope_parameters {rope_type: yarn, factor: 1.5}` over
`max_position_embeddings 262144`, and the convention is derived = max_pos x
factor. So:

* `--context-length 1048576` alone is refused. Its suggested
  `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` is register 47's hazard --
  positions past the cache -- and was deliberately NOT used.
* `--json-model-override-args` merges SHALLOWLY at the top level: a nested
  `{"text_config": {"rope_parameters": ...}}` replaces the whole text_config
  and the boot dies on a missing `max_position_embeddings`.
* Adding `original_max_position_embeddings` (demanded by the transformers
  YaRN validator on that path) collapses the derived ceiling to 262144,
  because it marks max_pos as ALREADY scaled.

Route that works: a symlink checkpoint with one edited config.json carrying
`factor: 4.0` and nothing else. 262144 x 4.0 = 1048576, derived legitimately.
`/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8-yarn4.0`,
7.5 K of symlinks.

## 20. "Zero upfront cost" is FALSE: ~440 MiB per rank, and it is eager

At an identical pinned pool with nothing using the context, `have` fell by
**exactly 440 MiB on every rank** (3312->2872, 708->268, 1100->660). Equal
across ranks = a replicated per-position structure. `head_dim 256 x
partial_rotary 0.25` -> 64 rotary dims, cos+sin 128 values/row, fp32 512
B/row, x 655360 new rows = **320 MiB**, the predicted figure; the remaining
~120 MiB is other per-position state and is not separately attributed.

It is already PRICED, though, because `have` is a physical-free reading: the
sizer sees it and lowers the pool without being told. **The 1M ceiling costs
58652 tokens of pool** -- derived 589736 at 1048576 against 648388 at 393216,
-9.0%. Making the reserve LAZY is the remaining work if that 9% is wanted
back; nothing is broken without it.

## 21. Positions past 393216 decode CORRECTLY -- the RoPE fix, on metal

Determined-answer probe: 400030 prompt tokens, a planted secret at the end,
answered **BANANA47**, finish=stop, 330 s. Register 47 says raising the
ceiling is exactly what makes the corrupt appended rows reachable. Raised,
reached, correct.

## 22. M1's wedge is the thesis, restated by the runtime

M1 pinned the 393216-era pool (648388) at 1048576 ctx. Both small ranks
logged `CANNOT FUND ITS OWN FLIP` AT BOOT, and the instance later held in TP:
`tp_to_pp refused 14 times and treated as unfundable ... re-probing in 13.8s`,
with 0 tracebacks and every process alive.

Read it as a result, not an accident: **pinning a pool the seam cannot fund
is the failure mode; letting the sizer derive is the fix.** The same
configuration, unpinned (M2), gave 589736 with 132 completed cutovers
(66/66), 0 abandons, 0 refusals, 0 tracebacks, corridor 1256/3099/1360, 0
breaches. And unlike boot E, the wedge announced itself twice at startup and
then backed off on a clock instead of storming.

## 23. OPEN and unattributed: A-vs-A is not byte-identical at 1M

Two identical temperature-0 requests on M2 returned different text, and took
2.79 s and 9.66 s -- served in DIFFERENT PHASES. The rig also carries a
documented upstream GDN prefill nondeterminism beyond ~109 tokens.

**No causal claim either way.** The attributing control -- the same A-vs-A on
the 393216 config -- was not run. Next shift: A-vs-A on L3's config FIRST,
then M2, then compare. Do not record "YaRN 4.0 broke determinism" until that
control exists; this evidence does not support it, nor its denial.

# R6: the lazy RoPE reserve, and the cgroup that was killing serving

## 24. ERRORS FIRST -- four premises falsified, two of them mine

**E18. Serving was not crashing. It was being SIGTERMed by its own cgroup.**
Agent sessions run in `/system.slice/claude.service`; `setsid` detaches the
SESSION, not the CGROUP; so a server booted from an agent shell is a member of
that unit and dies with every restart of it. The 09:04:59 instance drained
mid-probe and `claude.service` came up at 09:05:07 -- eight seconds later. The
peer's fresh restore and all three of its schedulers read
`0::/system.slice/claude.service` when checked directly. Fixed for every
capture-replay boot in `scripts/s33_boot_from_capture.sh` (transient
`systemd-run --scope`), acceptance printed by the boot itself and re-checkable
with `cat /proc/<serving-pid>/cgroup`. Runbook section 4.1 carries it.

**E19. The class this rig builds is `YaRNScalingMRotaryEmbedding`, not
`YaRNScalingRotaryEmbedding`.** A desk-read said the latter; the boot said
otherwise, and the lazy allowlist DECLINED it and ran fully eager. That is the
allowlist working: register 47's failure mode is silent, so an unverified
growth hook must not be trusted. Consequence beyond the class name: M-RoPE
position ids are NOT bounded by the sequence length, so the host-side
`seq_lens` bound the batch hook uses needed an explicit multimodal branch.

**E20. The per-batch hook runs inside CUDA graph capture.** The eagle draft
worker enters `ModelRunner.forward` from within `torch.cuda.graph`, where any
device read is `cudaErrorStreamCaptureUnsupported`; the boot then dies in
`capture_end`, nowhere near the read. The capture guard must be the hook's
first statement.

**E21. `have` does not move when the RoPE cache does.** Same tree, same argv,
warm records both: eager 1M measured 3536/708/1106 MiB and lazy 1M measured
3535/708/1106 MiB, with 240 MiB per rank fewer bytes written. The eager cache
comes out of the torch caching allocator's existing arena, which `have` --
a driver-free reading -- already counted. **So register 69's "440 MiB per
rank" does not reproduce**: see section 26.

## 25. MERGE READINESS -- what the R7 merge shift must know

`origin/integration/r2` has NOT moved: it is exactly `481411ac6b`, and this
branch is 0 commits behind it, so **no rebase is required**. The frozen
baseline `/spinning/wt-merge-r6-base` is `598f570ba4`, which is the merge base.

Suite, run in both trees on the six phase-flip files they share (boot, plan,
protocol, resident carry, round cadence, runtime): **189 passed in each**, so
the branch adds no red to the shared suite. Branch total over the flip family
plus the seam, gate and RoPE suites: **239 passed** before this shift's lazy
work, plus 16 new lazy cases.

What changed that a merger must carry:

* **`seam_reserve_enabled()` defaults True** (from R4). A pool sized as "VRAM
  minus corridor" with nothing left for the seam can no longer be built by
  accident, and the gate test asserts the MECHANISM, not the number 620000.
* **Seam fingerprint fields**: `gdn_resident_state_slots`,
  `enable_kv_session_offload`, `pp_stage_ratio`, `phase_flip_tp_vector`, keyed
  only when non-default so digests on a defaults rig stay valid. A record
  written before this widening is not comparable across those flags.
* **Solver margin, 192 MiB**, taken off the MEASURED position so it binds in
  both regimes. Every rank now re-measures above its floor.
* **`SGLANG_ROPE_LAZY_CACHE` defaults OFF** and is the only switch that
  changes when RoPE memory is spent. Nothing in the default path changes: with
  it unset, `install()` returns None on its first line and every call site
  lands on the code it landed on before.
* **`scripts/s33_boot_from_capture.sh` now launches inside a transient systemd
  scope.** This affects EVERY capture-replay boot including the sanctioned
  restore. If a deployment has no systemd (a container without it), the script
  warns and behaves exactly as before.
* **`_build_cos_sin_rows` is now the single row builder** for both the growth
  path and the lazy fill. Any new scaled RoPE subclass must override
  `_cos_sin_cache_inv_freq`, `_cos_sin_cache_row_scale` and (if its cache is
  longer than `max_position_embeddings`) `_cos_sin_cache_rows` -- and must be
  added to the lazy allowlist explicitly, or it silently stays eager, which is
  the safe direction.

## 26. THE 1M CEILING'S PRICE IS NOT THE ROPE CACHE (register 69 reversed)

R4 priced the ceiling at 440 MiB per rank by comparing `have` on a 393216 boot
against a 1048576 boot. R6 built the lazy reserve that was supposed to buy
those bytes back, engaged it on metal, and measured that **it buys nothing**:

| boot (all 1048576 ctx, same tree, same argv) | RoPE written per rank | have per rank |
|---|---|---|
| N0 eager  | 400.1 MiB | 3536 / 708 / 1106 |
| N2 lazy   | 160.1 MiB | 3535 / 708 / 1106 |

240 MiB per rank fewer bytes written, and `have` is unchanged. The mechanism
works -- the boot prints `256.1 MiB reserved / 16.0 MiB written LAZY` and the
managed reservation itself commits 0 MiB until touched -- but the eager cache
was coming out of the torch caching allocator's ALREADY-RESERVED arena, which
a driver-free reading cannot see. `have` is a driver-free reading. So the
sizer never paid for the eager cache, and freeing it gives the sizer nothing.

Two further consequences, both of which the next shift should treat as open:

* **Register 69's 440 MiB is not attributable to the ceiling.** R4's pair also
  crossed a COLD/WARM seam-record boundary (M1's log says `seam reserve is
  COLD`) and a different `--rank-gpu-memory-mib` vector, and this shift's
  warm-vs-warm 1M eager boot measures the same `have` as R4's warm 393216 boot
  to within 6 MiB on two ranks. A difference measured under three uncontrolled
  variables is a question, not a result -- the same law entry 72 was written
  for.
* **The ceiling still costs pool** (593264 at 1048576 against 648388 at
  393216), so SOMETHING scales with `context_length`. It is not the cos/sin
  cache. Candidates not yet separated: per-request structures sized by
  `context_len` (`req_to_token`), attention-backend workspaces, and the fact
  that the two boots also differ in their budget vector. Whoever takes this
  next should change ONE of those at a time.

**The lazy reserve stays, at default OFF.** It is correct, it is proven on
metal past the old ceiling (section 27), and it is the only mechanism that
makes a ceiling cheap when the allocator is NOT already holding the bytes --
but on this rig, today, it is not worth turning on for pool reasons.

## 27. WHAT THE LAZY RESERVE STILL NEEDS BEFORE IT MAY BE TURNED ON

The mechanism is built, unit-proven and engaged on metal, and it is DEFAULT
OFF for a reason this shift measured rather than guessed: on the lazy boot the
corridor was breached under a deep prefill -- continuous 100 ms minimum
**692 / 1785 / 1174 MiB** over 3958 samples, **19 samples under the 1024 MiB
law**, where the eager boots of this shift stayed above it.

The mechanism of that breach is inherent to the design, not a bug in it:

* An EAGER cache is allocated at boot, so the sizer sees the bytes gone (via
  `have`) and sizes the pool around them.
* A LAZY cache commits its pages when a sequence walks into new rows, i.e.
  during serving, in 128 MiB blocks -- and nothing prices that.

So a lazy reserve is only sound if the SOLVER charges for the rows a session
can still reach. The honest term is bounded and cheap to compute:

    reachable_rope_bytes = min(context_length, pool_tokens) * row_bytes
                           - already_filled_bytes

subtracted from `have` before sizing, exactly where the seam's own terms are
subtracted. `row_bytes` is `rotary_dim * 4` per lazy cache and the boot
already prints it. With that term in place the pool is smaller by the amount
the corridor was losing, which is the correct trade and the one the user law
asks for; without it, laziness is a corridor risk dressed as a saving.

**Do not turn `SGLANG_ROPE_LAZY_CACHE` on before that term exists**, and note
that on this rig the term would cancel the entire saving (section 26), so the
first question for the next shift is whether the feature is worth having HERE
at all -- as opposed to on a rig where the allocator is not already sitting on
the bytes.

## 28. T2 CLOSED and T3 CLOSED

**T2, the control R4 left open: R4's differing pair is NOT reproducible.** At
393216, on the ship, a matched 2x2 (short prompt below the ~109-token GDN
threshold, long prompt above it) x 5 repeats returned **10/10 byte-identical**
replies, every one of them served across a phase flip (two cutover lines per
rank inside each request's own log span). At 1048576, two valid observations
were byte-identical as well; the third died with the instance. So "YaRN 4.0
broke determinism" is refuted for this shape rather than merely unsupported --
and register 72's pair, which crossed a phase boundary AND a 7-second latency
gap, stays unexplained but unreproduced.

**T3: no shortfall, so the margin term is untouched.** 16 minutes, 20 cycles
of bs=4 concurrent decode + 64001-token prefill + 32k prefill + bs=4 on 4k +
a short pair, on the restored ship: 240 completions, 0 errors, 0 tracebacks,
168 completed cutovers, health 200 throughout, corridor continuous minimum
**1516 / 2701 / 1820 MiB and 0 breaches**. The qualification that matters: this
is the SHIP configuration (32,16,16 at 393216, pool pinned 620000), not L3's
(30,16,18, derived 648388) whose 1128 MiB minimum raised the question. The
production instance holds 492 MiB above the law under sustained load; L3's
tighter figure would need an L3 boot to re-measure, and that boot is the one
piece of T3 this shift did not buy.

## 29. WHAT THE NEXT SHIFT SHOULD DO FIRST

1. **Do not turn on `SGLANG_ROPE_LAZY_CACHE`.** It is wrong at depth (register
   77) and its guard cannot see the failure. The reproducer is bounded and
   cheap: 1M ceiling, planted-answer probe at 250026 (passes) and 390026
   (fails, one empty token).
2. **Write the guard that WOULD have caught it**: read a filled row back off
   the device and compare it against `_build_cos_sin_rows` for the same index.
   Content, not indices. Run it at 390k, not at 4k.
3. **Then decide whether the feature is worth having on THIS rig at all**
   (section 26): the eager cache costs the sizer nothing here, so laziness has
   nothing to win back, and it has a corridor breach to lose.
4. **The open question the ceiling actually poses** is unanswered: 648388 at
   393216 against 593264 at 1048576, and it is NOT the cos/sin cache. Change
   one variable at a time -- the budget vector differs between those two boots
   as well.

---

## SUPERSEDED IN PART BY THE #656 REMEDIATION SHIFT (2026-08-13)

See `HANDOFF_REMEDIATION_656.md`. Three items above are now moved on:

* **The solver margin is no longer one constant.** Section "Solver margin,
  192 MiB" still describes the MEASUREMENT ERROR BAR, and that term is
  unchanged. A second, per-rank term joins it: the deepest CORRIDOR SHORTFALL
  the runtime has actually observed on that rank, written into the seam record
  and read by the next boot of the same configuration. It is ZERO until a
  breach is measured, so a rig that has never breached sizes exactly as
  before. `seam_margin_bytes(reserve)`, `record_corridor_shortfall()`.

* **L3's corridor figure did not survive an hour, and the reason is not
  sizing.** The acceptance measured 886 MiB on the binding card, 138 below the
  law, in five episodes — and all five begin 2-4 s after a C20 seam-entry
  margin YIELD on the rank that owns that card. The remedy is 138 MiB, not the
  200-300 MiB the acceptance guessed, and the rank is NOT rank0: by register
  C21's card rule rank0 is the 5090, which never came near the law.

* **The empty-token probe at 390026 quoted in item 1 above is CONFOUNDED the
  same way the acceptance's bracket was.** Cold and cached deep prefills were
  compared without separating them. The full cold-probe table is in
  `HANDOFF_REMEDIATION_656.md` section 4: 280026 has failed 2 of 3 cold
  attempts while 250026, 300026 and 338916 are 4 of 4 exact. Do not read the
  390026 point as a lazy-RoPE property — the same signature appears with the
  lazy cache OFF.
