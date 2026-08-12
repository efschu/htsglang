# HANDOFF 690 — #656 / #631 Route A, successor 46

C29 and C30 are both fixed, red-first, and the ship config is running with
both in the boot. **The headline is not the two fixes — it is that C30's
booked diagnosis was wrong in a way that would have made the "fix" actively
harmful, and that the #559 backlog's two large branches must NOT be merged.**
Errors first.

---

## 1. ERRORS FIRST

### 1a. C30 WAS BOOKED WRONG, AND BUILDING WHAT THE REGISTER ASKED FOR WOULD HAVE DAMAGED THE TREE

HANDOFF_689 §4 and the register booked C30 as
"`--enable-deterministic-inference` and `--enable-kv-session-offload` cannot
both be on", and my own brief inherited that and told me to "wire a LOUD
boot-time refusal naming both flags". **That refusal would have been a
defect.** It would have rejected a configuration that works, and left the
actual trap armed for every boot that does not use kv-session-offload at all.

The real mechanism, verified line by line in the source:

    trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size
    if truncation_align_size is not None:
        if trunc_len < truncation_align_size:
            return AddReqResult.OTHER

`rem_chunk_tokens` is bounded above by `--chunked-prefill-size`, so a chunk
budget below one alignment unit refuses **every** request longer than the
budget, forever. The admission loop `break`s on any non-CONTINUE verdict, so
one such request at the head of the FCFS queue blocks the queue behind it,
`can_run_list` stays empty, and no batch is ever built. The wedged boot had
`--chunked-prefill-size 256` against an alignment of 4096.

kv-session-offload's ONLY role is that it forces the flashinfer backend. The
refusing predicate is a **third variable**. The same two flags with
`--chunked-prefill-size 4096` serve normally.

**Two sub-claims in the C30 entry are falsified by its own evidence files:**

* **there is no rank divergence.** `probe_v8.log` prints the #583 collective
  census from both ranks throughout the wedge and the two are IDENTICAL
  (`all_reduce 1862x` each). It froze because no forward ran. `num_queue_reqs
  0` is a 30 s idle gauge sampled after the client had already disconnected.
  The py-spy dump does not show `schedule_policy.py:1255` at all.
* **the silently disabled radix cache is a red herring.** `ChunkCache`
  inherits `evictable_size() -> 0` (never None, never raising) and
  `_tree_evictable_size()` already tolerates it. It costs restore headroom; it
  does not gate admission.

**The lesson worth carrying, because this is the third time on this line:** a
booked defect's ATTRIBUTION is not evidence just because the measurement was
real. N45's one-variable attribution ("same boot minus those two flags served
in 0.36 s") was correct and reproducible, and the conclusion drawn from it was
still wrong — removing the two flags also removed the alignment, so the
experiment moved two things at once without anyone noticing. Re-derive the
mechanism from source before building the refusal a register entry asks for.

### 1b. THE BYTE-IDENTITY DOOR IS NOT CLOSED — HANDOFF_689 §1a IS TOO PESSIMISTIC

HANDOFF_689 concluded byte-identity across a park "is therefore not obtainable
on this rig until deterministic inference and kv-session-offload can coexist"
and told the next shift not to spend time on flag variations. **They can
coexist.** What the wedged boot lacked was a chunk budget of at least the
alignment size.

**What would prove it, for whoever picks it up:** boot the probe with
`--enable-deterministic-inference --attention-backend flashinfer
--enable-kv-session-offload` AND `--chunked-prefill-size 4096` (>= the 4096
alignment), then re-run `park_complete_proof2.py`'s cohort. With determinism
live, the CONTROL arm should stop diverging from the quiescent reference; once
the control holds, a parked-arm mismatch becomes ATTRIBUTABLE instead of the
current "this run separates nothing" verdict.

The cost that made that boot unattractive is still real and is not solved:
flashinfer under deterministic mode disables the radix cache, so restore
headroom shrinks — which makes C29's margin sizing matter MORE, not less. Set
the restore margin explicitly on that boot.

**I did not run this.** It needs a GPU window of its own and my shift spent
its window on the ship-config confirmation, which the merges and fixes
required first.

### 1c. THE TWO LARGE #559 BRANCHES MUST NOT BE MERGED — N45's TRIAGE IS SUPERSEDED HERE

HANDOFF_689 §8 recommended cherry-picking `bugfix/pd-mamba-conv-state-transfer`
and treating `docs/features-vs-upstream` as a policy decision to extract
`FEATURES_VS_UPSTREAM.md`. **Measurement says neither is needed, and merging
either would be a large regression.** Both branches share the same stale base
and produce the identical signature against the current line:

    766 files changed, 7024 insertions(+), 338354 deletions(-)

That is 338k lines of work that exists on the integration line and NOT on
these branches. They are behind, not ahead.

* `bugfix/pd-mamba-conv-state-transfer` — its namesake fix `864edb8cb3` is
  **already on the line in substance**. `get_conv_transfer_segments` is at
  `mem_cache/memory_pool.py:1261` (and `:3797`) with the fix commit's exact
  docstring; `_send_mamba_state_slice` and `disaggregation/local_proxy.py` are
  present too. The commit is not an ancestor of HEAD, so it was re-landed by
  another route. Diffing only the four files the fix touches gives 70
  insertions against 1146 deletions — i.e. the branch would REMOVE work.
  **No merge, no cherry-pick. Recommend closing the branch.**
* `docs/features-vs-upstream` — even the one file the triage proposed
  extracting is superseded: `FEATURES_VS_UPSTREAM.md` is 1477 lines on the
  line versus 91 on the branch. I sampled five of its "unique" reference URLs
  (`2505.15141`, `HexGen`, `2503.20552`, `deterministic_inference.md`,
  `ktransformers`) and **all five are already present** on the line; the
  branch's lines differ only in formatting. **Nothing to salvage. Recommend
  closing the branch.**

I did not delete either branch — deletions are not mine to make unilaterally.
They are staged for a decision, with the measurement above as the basis.

### 1d. WHAT I DID NOT DO

* **Spec item 13's second half** (a restore where the restored session is the
  last one alive, to turn `restore_graph_evidence.py` from WEAK to STRONG) —
  untouched. It was #2 on N45's list; C29/C30 and the merge backlog consumed
  the shift. C29 makes it easier, not harder: the gate is now reachable on a
  small pool without the config workaround.
* **The `used:park:file` gauge reading 0 against 33 files on disk** — still
  noticed, still not investigated.
* **The two latent silent-drop siblings** (`kv_session_offload.py:4520-4524`
  and `:4931-4934`) — still deliberately unpatched, for N45's reason.
* The two documentation inconsistencies found during the merge and left alone
  because neither number can be re-derived from the tree: the audit
  memoisation speed-up is quoted as ~35 min in two files and ~50 min in
  `detC.py`'s own comment; and `AUDIT_421_UNWIRED.md` §B.8 says "three of the
  eight pins are retired" while naming four classes.

---

## 2. C29 — THE RESTORE MARGIN IS NOW SIZED AGAINST THE POOL

Commit `948c53e6da`. `resolve_restore_margin_tokens()` judges the margin at
manager init, **after** the draft-scratch carve-out, because that carve
permanently shrinks `allocator.size` and the margin must be judged against the
pool the gate will actually see.

Who chose the value decides the outcome — this is the part that matters:

| case | outcome |
|---|---|
| explicitly chosen, unsatisfiable | **REFUSED** at startup, numbers named |
| left at the SHIPPED DEFAULT | **clamped** to half the pool, logged at ERROR |
| `SGLANG_KVSO_RESTORE_MARGIN_FORCE=1` | honoured verbatim, still reported |
| satisfiable but above half the pool | value untouched, warning names the tail |

An explicit unsatisfiable margin gets the same treatment
`mtp_resident_reservation_error` already gives an unsatisfiable scratch
reservation, because both wedge silently. The shipped default is CLAMPED
rather than refused because the operator did not choose it, and refusing on a
shipped constant would turn every small-pool boot into a hard failure. The
default value is READ from the `ServerArgs` dataclass, not restated — the C18
rule; a drifted copy would silently move a boot from the clamp branch to the
refusal branch.

**Inert on the ship config:** `(512552, 4096)` resolves to 4096 with no log.

**Red-first, executed by reverse-applying the patch**, and the red is
BEHAVIOURAL rather than an import error: "the manager would run this
4096-token pool with margin 4096, at which no spilled session can EVER be
restored". The test's first half pins the MECHANISM and passes on both trees,
so it survives the fix as a regression guard; a control assertion proves the
fixture can open the same gate at a small margin, so a False cannot come from
the fixture.

---

## 3. C30 — THE GUARD, AND WHERE IT HAD TO GO

Commit `6a751b0adb`. `truncation_align_admission_error()` lives in
`schedule_policy.py`, co-located with the trap it protects so a future edit to
the trap sees the guard beside it, and is called from
`Scheduler.init_deterministic_inference_config`.

**It is called after the lcm, and that placement is the whole design.** The
alignment has TWO independent sources and either alone arms the trap:

* `--enable-deterministic-inference` on flashinfer/triton (align 4096 by
  default, from `SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE` /
  `SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE`);
* `--mamba-checkpoint-interval`, which sets the alignment on its own with **no
  deterministic inference anywhere**, and is lcm-ed into it when both are set.

`server_args` cannot see the lcm without restating it, so the check belongs in
the scheduler. A refusal worded as a flag-pair exclusion would have missed the
mamba source entirely.

**Side-finding, fixed in the same commit:** `--mamba-checkpoint-interval`'s own
help recommended "2048", which silently wedges any boot whose
`--chunked-prefill-size` is below 2048 — **including this rig's ship config at
512**. The help now states the constraint and the guard enforces it.

**Red-first, split so both halves are visible on the unfixed tree.** The guard
import is deliberately INSIDE the guard tests so collection still succeeds
without it. Reverse-applying: the MECHANISM test PASSES (the trap is real,
driven through the real `add_one_req`) while the WIRING test fails
behaviourally — "the scheduler accepted `--enable-deterministic-inference` on
flashinfer with `--chunked-prefill-size 256`: that instance boots, reports
ready and then admits nothing at all".

**One trap worth inheriting:** the mechanism test measures `can_run_list`, not
`add_one_req`'s return value. A successful chunked admission consumes the
whole chunk budget, so `budget_state()` reports `OTHER` on the way out of a
SUCCESS exactly as the refusal does. An earlier draft asserted on the return
value; its own control assertion caught it.

---

## 4. #559 MERGE BACKLOG — FOUR MERGED, TWO REFUSED ON MEASUREMENT

Suite green after EACH merge, pushed after each. Never batch-merged.

| # | branch | outcome |
|---|---|---|
| 1 | `chore/stash-rescue-432` | merged clean, `38a73e7b6f` |
| 2 | `docs/433-int8-prefill-vector` | 1 conflict resolved, `31bcfbfc6a` |
| 3 | `docs/pool-audit-2026-08-04` | 2 conflicts resolved, `124fac2f24` |
| 4 | `audit/unwired-sweep-421` | 4 conflicts resolved, `c78cc71442` |
| 5 | `bugfix/pd-mamba-conv-state-transfer` | **NOT merged — superseded, see §1c** |
| 6 | `docs/features-vs-upstream` | **NOT merged — superseded, see §1c** |

**Three stale citations were caught and corrected while resolving**, which is
the part of this work that generalises: in a docs merge the two sides often
BOTH cite something wrong, and picking a side ships a falsehood either way.

* `FEATURE_CATALOG.md` cited `environ.py:1776` for `SGLANG_GGUF_MXFP4_REPACK`;
  the pool-audit branch "corrected" it to `:1803`. Both stale — the symbol is
  at **`:1897`**. Re-derived and the drift recorded in the text.
* a retirement note cited `test/registered/unit/moe/test_cold_tier_wiring_394.py`;
  the file is at `test/registered/unit/layers/moe/...`.
* `AUDIT_421_UNWIRED.md` §B.9 claimed the #363/S8 planner `solve_fn` seam is
  unbound and cited a ratchet test. Task **#578 bound it**
  (`managers/regime_runtime.py:912`), and the cited test does not exist — it
  was inverted into
  `test_regime_act.py::TestPlannerFeed::test_the_seam_is_bound_and_refuses_without_measurement`.
  Both re-verified against this tree.

**The conflicted TEST file resolved without losing coverage.**
`test_unwired_features_421.py`: the branch carries four pin classes that the
integration line deliberately RETIRED because the features got wired
(#428/#394/#286), each replaced by a named positive test. Verified in the tree
rather than trusted from the notes — `cold_tier_fetch` is imported at
`fused_moe_triton/layer.py:1447`, `memtier` at
`short_term_offload_register.py:113-114`, and all five replacement files exist.
So the branch's 104 unique lines are superseded content, not dropped
assertions.

---

## 5. SUITE, AND A COUNT THAT MOVED ON PURPOSE

**#631 flip family: 1095 passed / 0 failed.** The count moves **1092 -> 1095**
because I ADDED `test_truncation_align_admission_656.py` to the canonical list
in `scripts/run_631_flip_family.sh`. This tree has had three under-collections
on that list and an admission-path test must not be orphaned. Do not read the
delta as a regression.

**The kvso family is NOT on that list and never was** — it is a separate
family and N45 ran it separately too. Run it explicitly:

    PYTHONPATH=/spinning/wt-631-routea/python /spinning/htsglang-gpu/.venv/bin/python -m pytest \
      test/registered/unit/test_kv_session_offload_unit.py \
      test/registered/unit/test_kv_spill_destination_unit.py \
      test/registered/unit/test_kv_spill_budget_unit.py \
      test/registered/unit/managers/test_prefill_adder.py -q

That set: **217 passed / 0 failed.** ruff: identical error count to baseline on
every touched source file (0 new), clean on both new test files. codespell
clean.

---

## 5b. THE CONFIRMATION WINDOW — 0 BREACHES ON BOTH INSTRUMENTS

22 min on the ship config at boot commit `c78cc71442`, canonical harness
(`scripts/s24_green_run.sh`, the same recipe N43 used, so the two are
comparable axis by axis) plus real agent traffic reading this tree's own
source through the router. Full extract:
`/spinning/evidence-631/s46/WINDOW_EXTRACT.txt`.

| axis | N43 (baseline) | N46 |
|---|---|---|
| corridor breaches | 0 | **0** (10214 samples) |
| gpu0_free MIN | 1141 | **1435** |
| gpu1_free MIN | 1624 | **2388** |
| gpu2_free MIN | 1241 | **1713** |
| deepest seam-census trough | 1024 | **1434** (306 troughs) |
| seam-census breaches | 0 | **0** |
| census CORRIDOR LAW BROKEN | 0 | **0** |
| seam PREDICTS A SUB-LAW TROUGH | 0 | **0** |
| FLIP ABANDONED | — | **0** (306 flips) |
| tracebacks / CUDA errors | 0 | **0** |
| soak | ok=104 err=0 | **ok=54 err=0** (22 min vs 31) |

**Judged on the seam census, not only on NVML**, per the standing law. Every
minimum is ABOVE N43's, and the deepest seam trough clears the 1024 law by
410 MiB where N43's touched it exactly. No axis regressed. Long-context
ladder reached 111405 input tokens.

**C29 and C30 were confirmed INERT here on metal, not by argument:**
`restore-margin clamp/refusal lines: 0` and `truncation-align refusals: 0`
across the whole boot. The C30 guard executes at every rank's scheduler init
on every boot, so a wrong predicate would have refused the boot outright —
that it served 306 flips and 393 decode batches is the proof it is correct
for this config. Note the inverse relationship to C30's own signature: the
wedged boot had **zero** `Decode batch` lines; this one has hundreds.

Two caveats stated so nobody over-reads the table. The window is 22 min
against N43's 31, so the soak totals are not directly comparable (the rates
are). And `gpu0`'s 1435 MiB minimum is below this boot's OWN configured
`SGLANG_CORRIDOR_FLOOR_MIB=1536` by 101 MiB — the same undershoot N43 (1141)
and N45's probe (1381) both had. The USER LAW at 1024 is what holds; the
boot's self-configured floor has been undershot on every window in this chain
and is not a new symptom.

## 6. THE RIG AS I LEAVE IT

* **Serving is UP on 30030, ship config, boot commit `0fbbf9bfdc` = HEAD**,
  booted by me with `setsid` from
  `/spinning/evidence-631/s43/boot_ship_30030.sh` after I verified that
  script's flag set against the LIVE `/proc/<pid>/cmdline` before stopping
  anything. I stopped it twice; I brought it back both times. Nobody owes a
  restore.
* **The rig runs exactly HEAD, deliberately.** The confirmation window ran on
  `c78cc71442`; the dynamic-chunking follow-up (`0fbbf9bfdc`) landed after it
  and touches python, so I rebooted rather than leave the running instance a
  commit behind what is pushed (Patchstand-vor-Last). The follow-up is
  therefore covered by the suite and by a clean boot + real generation, but
  NOT by the 22-min window — it is inert for this config (no dynamic
  chunking, no alignment source) and the boot proves the guard path executes
  without refusing.
* Verified with a **real generation** on both boots ("42", then "shipped"),
  not health alone. Pool **503950**, matching N43's window. Corridor after the
  final restore: free **1853 / 3458 / 3213 MiB**, all above 1024.
* **The ship config now runs with C29 and C30 in the tree, and the C30 guard
  executes at every rank's scheduler init on every boot.** That it booted at
  all is the on-metal proof the guard is correctly inert here — a wrong
  predicate would have refused the boot outright.
* Router 30099 untouched. All three cards were at 3 MiB before the reboot.
* Operational note, confirmed again: `pgrep -f "sglang.launch_server"` MATCHES
  YOUR OWN SHELL because the pattern sits in your command line. I briefly read
  my own bash process as a surviving server. Filter with
  `ps -eo pid,cmd | grep launch_server | grep -v "bash -c"`, and confirm the
  stop with `ss -tlnp | grep 30030` plus `nvidia-smi`, not with pgrep.

---

## 7. WHAT THE NEXT SHIFT SHOULD DO, IN ORDER

1. **The byte-identity proof (§1b)** — it is now unblocked and the exact boot
   recipe is written down. This is the highest-value open item on the line.
2. **Spec item 13's second half** — a run where the restored session is the
   last one alive. C29 makes the gate reachable without a config workaround.
3. **Close branches 5 and 6** of the #559 backlog on the §1c measurement, or
   overrule it with a counter-measurement. Do not merge them.
4. The `used:park:file` gauge reading 0 against 33 blobs on disk.
