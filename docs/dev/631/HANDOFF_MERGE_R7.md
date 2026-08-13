# HANDOFF MERGE-R7 — two merges, zero conflicts, and a label collision twice the size it was briefed as

Shift `656-merge-r7`. Worktree `/spinning/wt-merge-r7`, branch `merge/r7-batch`,
based on `origin/feat/route-a-631` at `481411ac6b` (the MERGE-R6 tip both lines
carried). Frozen pre-merge baseline: `/spinning/wt-merge-r7-base`, detached at
`481411ac6b`, clean tree. Evidence and logs: `/spinning/evidence-631/merge-r7/`.

Final tip on both lines: **`2496e4e84c`**, `ls-remote` verified after every push.

ERRORS FIRST.

---

## 1. The C-label collision is EIGHT labels, not four — and it merged silently

The briefing named a numbering collision between the two source branches'
register entries and expected it at `C23`–`C26`. It is `C23`–`C30`, and the
reason the extra four are easy to miss is the reason they were dangerous.

`feat/vram-flightrecorder-605` appends its eight entries as **table rows** at
the top of `docs/dev/631/CONTRADICTIONS_REGISTER.md`. The incumbents are not
rows:

| arriving label | incumbent at `481411ac6b` | incumbent form | body citations |
|---|---|---|---|
| `C23` | `SGLANG_UNEVEN_TOKEN_VECTOR` is not the PP stage ratio | `### C23` section | 3 |
| `C24` | the #695 recipe's rank discovery could never match | `### C24` section | 2 |
| `C25` | the #695 exact-size pin does not cost flip latency | `### C25` section | 2 |
| `C26` | #644's residual is untrimmed allocator arena | `### C26` section | 5 |
| `C27` | the corridor law enforced by nothing where it is spent | prose entry | 4 |
| `C28` | the park-completion defect (#659) | prose entry | **9** |
| `C29` | the restore-margin default token count | prose entry | 3 |
| `C30` | deterministic-inference vs kv-session-offload | prose entry | 6 |

A table row and a `###` heading never touch the same bytes, so **git merged all
eight without a murmur** and no conflict marker was ever raised. Grepping the
headings — the obvious check, and the one MERGE-R6 §7 used legitimately when
only one branch touched the file — reports `C1 C7 C23 C24 C25 C26 C31 C32 C42`
identically on all three refs and looks clean. The collision is only visible by
grepping the **body** for bare `C<n>` references, which is how `C27`–`C30`
surfaced: `C28` alone is cited nine times.

Left alone, the merged file would have carried eight identifiers each denoting
two different contradictions, in the one document whose entire purpose is that
a claim can be looked up and checked. That is the failure mode this register
exists to prevent, arriving inside the register itself.

**Resolved by renaming the arriving side**, which has no inbound references in
this file: `C23`→`C605-1` … `C30`→`C605-8`, in arrival order, in commit
`2496e4e84c`. Chosen over renumbering into free space because the `C`-namespace
is saturated and interleaved (rows `C1`–`C22`, sections `C1 C7 C23`–`C26 C31
C32 C42`, prose `C27`–`C30` and up), so any plain number is a future collision;
`C605-*` cannot be one and names its own ticket. Verified programmatically that
**all eight row bodies are byte-identical apart from the label** — nothing
dropped, merged or reworded, and no incumbent touched. A note above the table
records the mapping, and both #605 handoffs, which name their labels in prose
(`Recorded as C23`, `C27–C30 appended`), carry a pointer to the shipped ones.

## 2. Four of the kv-universe branch's own tests cannot pass on the merge protocol

`test/srt/test_phase_flip_serving_proof_gate.py` — the file that pins the
quarantine-constant removal, i.e. the central claim of merge step 1 — fails all
four of its cases under the canonical CPU-only desk protocol with
`RuntimeError: No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS) or platform
plugin is available`.

**Inherited, not introduced**, and proven so rather than assumed: the same arm
run against the untouched source worktree `/spinning/wt-kv-universe` returns the
identical `4 failed, 56 passed`. The file needs a visible device.

Two consequences the next shift should hold:

* The branch handoff's "**239 passed**" was necessarily measured with a device
  visible. It is not reproducible under `CUDA_VISIBLE_DEVICES=99`, so the two
  numbers are not comparable and neither is wrong.
* **The gate that asserts the quarantine constant is gone did not run in this
  merge.** The constant's removal is verified here by reading and by the rest of
  the seam suite, not by that file. A merge shift does not take GPUs (§6), so
  closing this needs either a device-requiring arm run in someone's GPU window
  or a `skipif` marker so the file's status is explicit instead of red.

## 3. The tree is formatter-dirty on both sides of this merge

`black` 26.1.0 — the version pinned in `.pre-commit-config.yaml` — over the 37
touched `.py` files: **9 dirty before, 9 dirty after**. The equal count hides
that the *sets differ*:

* **Fixed by this merge**: `server_args.py`, `model_runner.py`,
  `kv_vmm_backing.py` were dirty at `481411ac6b` and are clean now.
* **Newly dirty**: three added test files —
  `test_seam_fingerprint_and_margin_656.py`, `test_flight_recorder_605.py`,
  `test_phase_flip_serving_proof_gate.py`.
* **Dirty on both sides**: `phase_flip_boot.py`, `phase_flip_runtime.py`,
  `phase_policy.py`, `scheduler.py`, `model_runner_kv_cache_mixin.py`,
  `uneven_perf.py`.

This also **reverses the obvious reading of the #605 branch's `server_args.py`
diff**, which looks like gratuitous reformat churn against unrelated regions
(assert-wrapping, line joins). It is not churn: it is a black 26.1.0 pass that
*corrects* pre-existing drift. The tempting cleanup — reverting those hunks to
keep the diff tight — would have re-broken the file.

Recorded and **not fixed**: a merge shift does not reformat, and R6 set the
precedent of recording lint deltas rather than acting on them. The standing
defect is that **the pinned pre-commit hook is evidently not running on this
line** — six files have been dirty across at least two merge rounds.

`ruff` over the same 37 files: **473 → 472**, one error removed
(`phase_policy.py`), none added, and all **15 new modules are ruff-clean**.
`codespell` over 44 touched `.py`/`.md`/`.sh`: **9 → 11**; both new hits are the
word `unparseable` (a valid variant). The pre-existing `schedul` hit in
`CONTRADICTIONS_REGISTER.md` is the deliberate 15-character `TASK_COMM_LEN`
truncation from MERGE-R5 and **must not be "fixed"**.

## 4. The stale local ref trap is STILL armed

Unchanged from MERGE-R6 §1 and re-verified at this shift's start:

| ref | SHA at shift start |
|---|---|
| `origin/feat/route-a-631` | `481411ac6b` |
| `origin/integration/r2` | `481411ac6b` |
| **local** `feat/route-a-631` | **`0ae49fafb4`** — the MERGE-**R4** base, now THREE rounds stale |

It is checked out in `/spinning/wt-631-routea`, which belongs to another strand,
so git will not move it and forcing it would move another session's working tree
out from under it. This shift did what R6 did: **every merge source named as
`origin/<branch>`, every push as `git push origin HEAD:refs/heads/<line>`,
`ls-remote` compared against `git rev-parse HEAD` after each.** The stale ref was
never an input to anything. Do not "fix" it from a merge shift.

`local integration/r2` was at `481411ac6b` (checked out nowhere) and is left to
the owner to fast-forward; this shift did not touch local refs at all.

## 5. Nothing on GPU, on purpose

Every suite run used `CUDA_VISIBLE_DEVICES=99`. Serving on 30030 was not
touched, no GPU arbitration window was claimed, port 30099 was not touched, no
`pkill` was used, and `git stash` was never invoked. Merges are CPU-side.
The cost of that discipline is §2, and it is the right trade.

---

## 6. WHAT MERGED

A path-overlap check across both source branches ran **before** the first merge.
Unlike R6's three branches, these two are **not** disjoint: 28 + 18 files with
**two in common** — `docs/dev/631/CONTRADICTIONS_REGISTER.md` and
`python/sglang/srt/server_args.py`. Both merged without conflict anyway, and
both were verified present on both sides afterwards rather than assumed (§8).

| step | source | at | resulting tip | conflicts |
|---|---|---|---|---|
| 1 | `feat/kv-universe-656` | `4157aad9cf` | `70302cfb4f` (`--no-ff`) | **0** |
| 2 | `feat/vram-flightrecorder-605` | `848ccb696d` | `a331063df0` (`--no-ff`) | **0** |
| 3 | register C-label resolution (§1), docs only | — | **`2496e4e84c`** | — |

Each step's suite was green and pushed to **both lines** before the next merge
was started; nothing was batched. Total delta against `481411ac6b`: **44 files,
7272 insertions, 254 deletions**. Author on every commit
`efschu <efschu@users.noreply.github.com>`, no trailers.

### Step 1 — `feat/kv-universe-656`, the capacity work

The **620000-token quarantine constant is DELETED**, not raised, and what
replaces it is a mechanism rather than a number: `phase_flip_seam_reserve` sizes
the pool so every rank can fund its own flip seam, from a position measured on
the previous boot. `seam_reserve_enabled()` **defaults True**, so a pool sized
as "VRAM minus corridor" with nothing left for the seam cannot be built by
accident. The solver targets equality and therefore carries a margin
(`ENV_MARGIN_MIB`, 192 MiB) taken off the *measured* position. Also carried: the
phase-policy control-loop fix (a refused arm no longer commits the dwell clock,
so an unfundable seam becomes a bounded stand-down instead of a silent
livelock), #364 idle-vacate engagement, the YaRN 1M ceiling with the RoPE
growth-path corruption fix, the widened seam fingerprint
(`gdn_resident_state_slots`, `enable_kv_session_offload`, `pp_stage_ratio`,
`phase_flip_tp_vector`, keyed only when non-default), and the cgroup-escape boot
fix in `scripts/s33_boot_from_capture.sh` (transient `systemd-run --scope` —
this affects **every** capture-replay boot including the sanctioned restore).

**Lazy RoPE ships BUILT AND OFF, as briefed and as verified.**
`SGLANG_ROPE_LAZY_CACHE = EnvBool(False)` at `environ.py:1531`, and checked at
runtime after the merge in the merged tree: resolves `False`, `is_set()` False.
It was falsified on metal — it buys 240 MiB/rank of *written* bytes and **zero
pool**, because the eager cache came out of the allocator arena `have` already
counted, and it breached the 1024 MiB corridor under deep prefill (19 samples
below the law).

### Step 2 — `feat/vram-flightrecorder-605`, the #605 recorder

Per-boot attribution with **drift 0**, achieved by grouping marks **by PID**:
under PP the TP rank is 0 in all three processes, so rank-grouping merged three
cards into one timeline and billed a 3080's context to the 5090. The commit
watermark becomes a recorded post via `KvVmmArena.arena_census()` (a
`WeakValueDictionary` — observing an arena must not keep it alive). Continuous
under-load mode, and the ship-boot ledger dump: `ServerArgs.__post_init__` now
builds a modelled ledger **for the record**, gated on the recorder already being
armed and **discarded immediately**, because the ship config pins
`--rank-gpu-memory-mib`, which takes the pin path, which skips the planner — so
fourteen boots of measured marks had sat beside zero modelled counterparts and
`reconcile.py` had nothing to compare against. Flag-gated throughout: with
`SGLANG_VRAM_FLIGHT_DIR` unset, `_dump_observation_ledger` returns on its first
line and the boot is byte-identical.

---

## 7. SUITE — every failure count identical to baseline, at every step

Baseline is the frozen worktree `/spinning/wt-merge-r7-base` at `481411ac6b`,
measured in `/spinning/wt-merge-r7` before the first merge. Same interpreter
(`/spinning/htsglang-gpu/.venv/bin/python`), `PYTHONPATH=<worktree>/python`,
`CUDA_VISIBLE_DEVICES=99`, `pytest --color=no`, one directory per invocation so
a truncation is isolated to its directory. **The baseline's counts are identical
to MERGE-R6's post-step-3 column**, so the chain back to R6's frozen
`/spinning/wt-merge-r6-base` @ `598f570ba4` is continuous and this shift is
comparable to the previous three.

| suite | BASE `481411ac6b` | after step 1 | after step 2 |
|---|---|---|---|
| #631 flip family (canonical script) | **1116 passed** | 1116 passed | **1116 passed** |
| `unit/managers` | 9F 1357P 18S | 9F 1357P 18S | 9F 1357P 18S |
| `unit/mem_ledger` | 1F 359P | 1F 359P | 1F **379P** |
| `unit/model_executor` | 15F 594P | 15F 594P | 15F 594P |
| `unit/server_args` | 615P | 615P | 615P |
| `unit/turnkey` | 116P | 116P | 116P |
| `unit/utils` | 46F 348P 4S | 46F 348P 4S | 46F 348P 4S |
| `unit/docker` | 4P | 4P | 4P |
| **new-656 arm** | *(files absent)* | **4F 56P** | 4F 56P |

**Every failure count is identical on both sides at every step.** The single
`mem_ledger` failure is the same inherited
`test_communicator_group_contract_612.py` case before and after, checked by
name, not by count. Pre-existing failure sets inherited unchanged: 46
`unit/utils`, 15 `unit/model_executor`, 9 `unit/managers`, 1 `unit/mem_ledger`.

**+80 tests collected across both merges, 0 new failures**: +60 in the new-656
arm at step 1 (56 pass, 4 device-required — §2), +20 `mem_ledger` at step 2 (the
three new #605 test files).

### The new-656 arm exists because the canonical family list is explicit

`scripts/run_631_flip_family.sh` names its files one by one and its own header
records **three** historical under-collections from assuming a glob covered it.
The kv-universe branch adds six test files and **does not extend that list**:
three under `test/registered/scheduler/` and three under `test/srt/`. They would
have been collected by nothing in the R6 protocol, and the flip-family count
would have sat at 1116 looking green. They were run as an explicit separate arm
(`/spinning/evidence-631/merge-r7/run_new656.sh`) instead. **Follow-up for the
branch owner: those six files belong in the family list.**

---

## 8. REGISTER UNION — verified, nothing lost

`docs/dev/631/CONTRADICTIONS_REGISTER.md` is touched by **both** branches, so
unlike R6 there was a genuine union to reconcile and a real last-writer-wins
risk. Verified across the whole R7 range `481411ac6b..2496e4e84c`:

| check | result |
|---|---|
| deleted lines in the register | **0** (`git diff --numstat` reports `542  0`) |
| added lines | **542** = 518 (kv-universe) + 8 (flight recorder) + 16 (§1 mapping note) |
| file length | 1866 → **2409** |
| `### C<n>` entry headings | 9 → **9**, byte-identical and in the same order |
| numbered `## <n>.` headings | 0 → **26**, running `## 52` → `## 77` |
| `C605-*` rows | 0 → **8**, bodies byte-identical to the arriving `C23`–`C30` |

Append-only and monotone: **zero deletions against the base after the rename**,
because the renamed rows are lines that did not exist at `481411ac6b` at all.
Every pre-existing entry is still present. The kv-universe additions run
`## 52` → `## 77` (through the R4 and R6 blocks: the seam sizer, the margin, the
fingerprint, the 1M ceiling, the cgroup trap, and register 77 — the lazy
reserve's own falsification). The eight #605 rows ship as `C605-1`–`C605-8` per
§1.

Both sides were also verified present in the other shared file,
`server_args.py`, by marker rather than by trusting a clean merge: the
kv-universe seam/vacate comment blocks and the #605 `_dump_observation_ledger`
are all there.

---

## 9. STATE AT HANDOVER

- **Both lines at the same SHA** — `2496e4e84c`, `ls-remote`-verified against
  local `HEAD` after every push. Pushed to **`origin` = the efschu fork only**;
  `upstream` was never a push target.
- Working branch `merge/r7-batch` in `/spinning/wt-merge-r7` — same SHA, kept.
- New frozen baseline `/spinning/wt-merge-r7-base` at `481411ac6b`, detached,
  clean — **kept deliberately** so R8 can diff against the tree R7 measured
  against. R6's `/spinning/wt-merge-r6-base` @ `598f570ba4` is also still there.
- **Serving, GPUs, arbitration, port 30099: untouched.** No boot, no window, no
  `pkill`, no `git stash`. Nothing under `/etc` modified.
- Local refs untouched, including the stale `feat/route-a-631` (§4).

## 10. REMAINING UNMERGED BRANCHES

78 local branches are not merged into the tip — unchanged in count from R6,
because both branches R7 merged were already counted there under their own
names. The ones recent enough to be live work, newest first:

| date | SHA | branch |
|---|---|---|
| 2026-08-12 | `d38bb6df32` | `trial/cumulative` |
| 2026-08-09 | `982b6434ce` | `feat/route-a-631-resume-gate` |
| 2026-08-09 | `00a1c50fcb` | `feat/gguf-q4-bringup-651` |
| 2026-08-08 | `27f3bf7996` | `fix/collective-stream-622` |
| 2026-08-08 | `18370879e3` | `integration/r3-probe-next2` |
| 2026-08-07 | `b851df7626` | `feat/dual-group-631` |

(`backup/pre-email-fix-s13` and `backup/pre-deps-strip-s13` are backups, not
merge candidates.) The three standing Claude strands own
`feat/gguf-q4-bringup-651` (#651), `fix/collective-stream-622` (#622/#649) and
the Route-A line; none was asked to be merged this shift, and none was.

## 11. OPEN FOLLOW-UPS CARRIED FROM BOTH BRANCH HANDOFFS

Carried verbatim in substance, none of it addressed by this shift:

1. **Lazy RoPE's root cause is OPEN.** The reserve is correct under test and
   wrong on metal past a depth: the planted-answer probe passes at 250026 and
   fails at 390026 with one empty token, and its guard cannot see the failure.
   The guard that *would* have caught it is specified and unwritten — read a
   filled row back off the device and compare it against `_build_cos_sin_rows`
   for the same index, **content, not indices, at 390k rather than 4k**. Ships
   default OFF; do not enable before that guard exists and before the solver
   charges for reachable rows
   (`min(context_length, pool_tokens) * row_bytes - already_filled`).
2. **The 1M pool cost is UNATTRIBUTED.** 648388 tokens at 393216 against 593264
   at 1048576, and register 26 proves it is **not** the cos/sin cache. Candidates
   not yet separated: `req_to_token` and other `context_len`-sized per-request
   structures, attention-backend workspaces — and the two boots also differ in
   their budget vector. Change **one** variable at a time.
3. **The `KvReshardError` checksum incident (register C22) stands.** #657's
   corridor steering decides correctly and group-uniformly, but its application
   re-sorts on a rank-local 1 s clock; three ranks re-sorted at three different
   instants and a `pp->tp` cutover died on `payload checksum mismatch — refusing
   to scatter`. A pure function applied on a private clock is not a
   group-uniform mutation of replicated state. Not actuated, not fixed.
4. **L3's 1128 MiB corridor is UN-RE-MEASURED.** T3's 16-minute sustained run
   (240 completions, 168 cutovers, 0 breaches, minimum 1516/2701/1820 MiB) was
   the **ship** configuration (32,16,16 at 393216, pool pinned 620000), not L3's
   (30,16,18, derived 648388) whose 1128 MiB minimum — 104 MiB above the law —
   raised the question in the first place. That boot was not bought.
5. **`corridor_trace.py` has no production call site.** Tested and ready; its
   natural home is the scheduler, which belongs to the #656 shift. The
   out-of-process sampler covers the law's own quantity today.
6. **`reconcile.py` has still never RUN** against a ship boot. Both halves now
   exist — the modelled ledger (step 2) and the measured posts — and nobody has
   yet put them in one table on the same boot. **This is the next #605 payout and
   it needs no GPU.**
7. The driver-unattributed residual band, **164–276 MiB per card**, is the only
   non-deterministic term left and is irreducible with NVML.

## 12. NEXT, IN ORDER

1. **§6 item 6 first**: run `reconcile.py` on a ship boot. No GPU, both halves
   present, and it is the payout #605 was built for.
2. **Get the six new kv-universe test files into
   `scripts/run_631_flip_family.sh`** (§7). Right now they are covered only by a
   private arm this shift wrote; the next merge shift will not know to run it.
3. **Decide `test_phase_flip_serving_proof_gate.py`'s status** (§2): either a
   `skipif` for no-accelerator so the CPU protocol reports it honestly, or a
   device-visible arm inside someone's GPU window.
4. **Run the pinned pre-commit black over the six persistently dirty files**
   (§3) as a standalone formatting commit, outside a merge shift.
5. Carried unchanged from MERGE-R6 §10 and still unaddressed: #363's stage
   actuator is desk code on the line (gated off, ticket
   `docs/dev/363/TICKET_363_STAGE_CLOCK.md` unrun — pre-step P2 decides whether
   the 5 % enter watermark clears this rig's noise floor); the first real
   `docker build` is the #384 gate's first test; the stale local
   `feat/route-a-631` ref needs its worktree owner; the #695 census lines still
   need a PP-unique rank identity; `route_a_631_prod_boot.sh` still diverges
   from the ship capture in seven flags; and `s33_boot_from_capture.sh` still
   waits on a flat 2000 MiB while the ship budget leaves ~289 MiB of slack.
