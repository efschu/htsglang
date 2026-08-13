# HANDOFF MERGE-R9 — one merge that could not conflict, and a formatter that has stopped being enforced

Shift `656-merge-r9`. Worktree `/spinning/wt-merge-r9`, branch `merge/r9-batch`,
based on `origin/feat/route-a-631` at `6d169c04ab` (the MERGE-R8 tip both lines
carried). Frozen pre-merge baseline: `/spinning/wt-merge-r9-base`, detached at
`6d169c04ab`, clean tree. Evidence and logs: `/spinning/evidence-631/merge-r9/`.

Merged: `feat/acceptance-remediation-656` @ `faa835977b` — the branch that took
the #656 formal acceptance to **7 of 7**.

**Both lines are at `4f0b4f0bcb`** (merge `9dfae34eda` + the step-2 list patch),
`ls-remote` verified against `git rev-parse HEAD` after every push. This handoff
cannot name the commit that contains it, so the actual tip is the single
docs-only commit sitting directly on top of `4f0b4f0bcb` — the convention R7 and
R8 used.

ERRORS FIRST.

---

## 1. THE PINNED `black` IS NOT RUNNING, AND THIS BRANCH MADE IT WORSE BY SEVEN FILES

R8 §10 recorded five persistently dirty files and closed with *"run the pinned
pre-commit `black` over them as a standalone formatting commit, outside a merge
shift."* Nobody did, and the arriving branch was written without the hook:

| black 26.1.0 | dirty | of |
|---|---|---|
| the 13 pre-existing `.py` this branch modifies, at **base** | **7** | 13 |
| the same 13, at **final** | **11** | 13 |
| the **7 new** `.py` this branch adds | **3** | 7 |

**Four pre-existing files were newly dirtied** — `phase_flip_seam_census.py`,
`phase_flip_seam_reserve.py`, `kv_vmm_backing.py`, `scheduler.py` — and three of
the seven new files land dirty. R8's five are all still dirty; `scheduler.py`,
which R8's merge had *fixed*, is dirty again.

That is the whole story in one line: **R8 arrived with the hook run and R9
arrived without it, so the fix R8 bought was undone in one round.** A formatting
standard that survives only when an author happens to run it is not enforced,
and this is the second consecutive shift to write that sentence.

**Not fixed here, deliberately** — reformatting 14 files inside a merge shift
destroys the property that makes the suite comparison meaningful (every diff in
this merge is attributable to the branch). It is a standalone commit, and it is
now overdue rather than merely open.

### The measurement was wrong the first time, and the reason generalises

The first pass of this table reported **0 dirty of 20** and it was an artefact:
`black --check -q FILES 2>&1 | tail -5; echo rc=$?` captures **`tail`'s** exit
code, not black's, and `-q` suppresses the per-file lines that would have
contradicted it. A clean-looking `rc=0` was produced by a pipeline that never
consulted the tool. Re-measured by reading black's own `would reformat` lines
and its `N files would be reformatted` trailer. Recorded because a lint table is
exactly the kind of low-stakes check nobody re-derives, and this one was inverted.

## 2. A SEVENTH UNDER-COLLECTION — six of seven new test files were listed

The arriving branch added **seven** test files and extended the explicit list in
`scripts/run_631_flip_family.sh` by **six**.
`test/registered/unit/managers/test_frame_component_ballot_656.py` was the one
left out.

This is the fifth consecutive round in which the explicit family list has been
short, and the branch had itself fixed the previous instance in the same commit
that created this one.

**It is milder than R8 §1-2 and the difference is worth naming.** R8's orphan
sat in `test/registered/unit/mem_cache/`, a directory no canonical arm collects
at all, so it was invisible. This one sits in `test/registered/unit/managers/`,
which **is** a canonical arm — so it ran, it was green, and nothing was red
anywhere. What it missed was the *canonical family sweep*, the list that is
supposed to be the single answer to "what covers the flip".

**It is not a trivial file.** It pins register row **C22-d**: three ranks whose
`POOL CENSUS` lines are identical in every field (`size=579870 free=278572
cached=300034 unaccounted=1264`) framed different payloads anyway, six rounds
running, because `KvRowCap.release` restores withheld ids with `torch.sort`
while `_apply` preserves eviction order — **identical membership, different
order, and the allocator takes from the front.** A single digest detects that
and cannot attribute it; this file pins the three-part
(slots / waves / geometry) digest that names the diverging term.

Folded in as merge step 2 (§4), for the same reason R8 folded in the previous
six: a private arm is not a canonical list.

## 3. THIS MERGE COULD NOT HAVE CONFLICTED, AND THAT IS A FACT ABOUT THE INPUT

`6d169c04ab` is an **ancestor** of `faa835977b` — the remediation shift rebased
onto the R8 tip before its metal leg and never diverged after. So:

* the merge is fast-forwardable; `--no-ff` was used anyway, per canon;
* **the merge commit's tree is byte-identical to the source branch's tree**
  (`03af24a437` on both sides, checked by `rev-parse`);
* zero conflicts was **structural, not an outcome**.

Stated plainly so nobody reads "0 conflicts" here as the same evidence it was in
R4-R8, where the sources genuinely diverged. There was no resolution to review,
and the register union-merge (§5) was satisfied trivially — it was verified by
count anyway, because a trivially-satisfied check that nobody ran is not a check.

The one thing this *does* buy: the merged tree is provably the tree the branch's
own 7-of-7 acceptance ran against, modulo step 2's list patch.

---

## 4. WHAT MERGED

| step | source | at | resulting tip | conflicts |
|---|---|---|---|---|
| 1 | `feat/acceptance-remediation-656` | `faa835977b` | `9dfae34eda` (`--no-ff`) | **0** (§3) |
| 2 | component-ballot family-list patch (§2) | — | **`4f0b4f0bcb`** | — |

27 files, +4522 / −103. Each step's suite was green and pushed to **both lines**
before the next was started; nothing was batched. Author on every commit `efschu
<efschu@users.noreply.github.com>`, no trailers.

The branch carries 20 commits and four mechanisms. In the order they were found,
because each one falsified the previous shift's story:

**(a) C22's root cause, and the misreport that hid it.** The acceptance died on
`KvReshardError: payload checksum mismatch`, and the log carried its own
falsifier: a `uint8` sum over N bytes lies in `[0, 255N]`, and the two `sender`
fields were `4626949667419791296` (needs an 18-petabyte payload) and
`-4450328002521349435` (**negative**). **Neither was ever a checksum**, so the
data was never what differed. The per-peer payload length is a product of
rank-local terms and **nothing on the wire carries a length**; the receiver's
size check compares `payload.numel()` against `incoming_nbytes[peer]` — a buffer
it allocated itself at that size — and is **vacuous by construction**. This
retracts MERGE-R7's attribution to #657's corridor steering, which was not even
enabled on the acceptance boot (`SGLANG_CORRIDOR_REBALANCE=0`) — a *trigger*
mistaken for a *cause*.

**(b) The frame ballot, and why it was not enough.** `_frame_digest` rides the
`[x, -x]` MIN pair on the collective the round already runs, so divergent ranks
**abandon unanimously before a byte moves**. It fired on metal at 14:46:39Z:
PP0/PP2 framed `1545804850`, PP1 framed `1237458399`, no `KvReshardError`, no
SIGQUIT, the process lived. And then the honest half: a **persistent**
divergence starves the `pp->tp` leg, and R1's instance **wedged** — alive, KV
intact, `/health` **503**, no tokens. Detection converted a crash into a wedge,
which is better and is not "serving continues".

**(c) The mechanism: the shrink is collective and the recovery was not.**
Comparing all 1980 `POOL CENSUS` lines rank by rank gives 7 divergent events in
two episodes, both opening at a `tp_to_pp` post-cutover, and two log lines in
33 minutes say what ran there — both on PP1: `KV-BACKING recovered to 544077 of
578390 rows (corridor-bounded)` and `... 537986 of 578390`. `578390−544077 =
34313` and `578390−537986 = 40404` — **both census deltas, to the row**.
`shrink` is decided once for the group by a MIN reduction; `recover()` is
bounded by **this rank's own** distance from the corridor law, so the binding
card stays capped where its peers do not, enumerates a different id space, and
frames a different payload — persistently, because a cap does not clear itself.
**A refusal may be decided locally, a capacity may not.** Closed by
`cap_proposal` / `collective_cap_target` / `reconcile_to`, riding the KV rung's
existing reduction widened 4 → 8 fields. Metal 16:00:30Z: 579870 / 579722 /
578606 levelled to 578606 **in one round**.

**(d) The wedge closure, and the corridor.** A controller that cannot flip must
still serve: strict purity is a **throughput** rule, never a correctness one
(its own `threshold:<n>` and `off` modes run decode in the PP layout as
supported configurations), so the prohibition on the stuck class is lifted,
keyed on the stuck direction, logged once, self-clearing on the next committed
flip. Metal 16:01:24Z: all three ranks relaxed purity for prefill
simultaneously and the instance kept serving — `/health` 200, tokens flowing —
where the same state was 503-with-no-tokens one boot earlier. That in turn
unblocked the corridor: the gate had long measured its worst in-cutover draw and
refused to act on it *because* "refusing `pp->tp` starves decode outright", and
that premise is now false, so the **C20 yield is withheld** when this rank's own
measured draw predicts a sub-law trough. With the seam margin re-derived
(192 error bar + 138 measured shortfall + 38 measured spread → 384 MiB, pool
597106 → 578390 tokens), boot_v3 recorded **0 of 33648 samples below the law**.

Also carried: the mamba-floor acceptance specimen tests, `ACCEPTANCE_656_R2.md`,
and the `HANDOFF_KV_UNIVERSE` / `HANDOFF_LEDGER_RECONCILE` union updates.

### Two defects the branch put in and metal took back out

Recorded because they are in the merged history and the reasons generalise.
**(a)** `cap_proposal` first offered to **grow** to what `recover` would be
allowed to commit, which on the `pp_to_tp` leg hands back exactly the rows the
collective shrink had just taken to fund the seam — `cuMemCreate failed:
CUDA_ERROR_OUT_OF_MEMORY`, rank 0 driven to **3 MiB free**. The agreement is now
strictly non-allocating. *A fake pool grows whenever the model says the bytes are
there, and the model was the thing that was wrong.* **(b)** The valve was keyed
on `_seam_abandons_in_a_row`, which the seam **backoff** freezes by declining
arms without entering the seam — so the counter sat at 3 while the policy logged
`arm refused (7 in a row)`. It now also reads the policy's `arm_refusals` streak.

---

## 5. SUITE — every failure count and every failure SET identical, at every step

Same interpreter (`/spinning/htsglang-gpu/.venv/bin/python`),
`PYTHONPATH=<worktree>/python`, `CUDA_VISIBLE_DEVICES=99`, `pytest --color=no`,
one directory per invocation. Runner: `/spinning/evidence-631/merge-r9/arms.sh`.

| suite | BASE `6d169c04ab` | after step 1 | after step 2 |
|---|---|---|---|
| #631 flip family (canonical script) | **1169P 7S** | **1237P 7S** | **1247P 7S** |
| `unit/managers` | 9F 1357P 18S | 9F **1425P** 18S | — |
| `unit/mem_ledger` | 1F 437P | 1F 437P | — |
| `unit/model_executor` | 15F 594P | 15F 594P | — |
| `unit/server_args` | 615P | 615P | — |
| `unit/turnkey` | 116P | 116P | — |
| `unit/utils` | 46F 348P 4S | 46F 348P 4S | — |
| `unit/docker` | 4P | 4P | — |

Step 2 touches only `scripts/run_631_flip_family.sh`, which no arm imports, so
only the family arm was re-run.

**Continuity, the check R8 specified.** `/spinning/wt-merge-r8-base` is at
`cd71ec34ce`, R8's *pre*-merge tree, and is therefore the wrong tree to diff
against. The check used instead is that **this shift's BASE column reproduces
R8's post-step-2 column exactly** — and it does, on every one of the eight arms,
including the flip family at 1169P 7S. The chain back through R8's, R7's and
R6's frozen bases is unbroken.

**Failure SETS diffed by name, not merely counted**, for all four red arms:

| arm | names at base | names at step 1 | diff |
|---|---|---|---|
| `managers` | 9 | 9 | **identical** |
| `model_executor` | 15 | 15 | **identical** |
| `utils` | 46 (44 `FAILED` + 2 `SUBFAILED`) | 46 | **identical** |
| `mem_ledger` | 1 | 1 | **identical** |

`utils` needed a second pass: a `^FAILED ` grep finds only **44** of its 46,
because two are emitted as `SUBFAILED(module=...)` lines by
`test_capability_vendor_gates.py::test_cutedsl_blackwell_gates` (the
`gdn_cutedsl` and `kda_cutedsl` subtests). A name diff that matched the count
would have silently skipped them; the diff above matches all three prefixes and
covers **46 of 46**.

Inherited failure sets carried unchanged: 46 `unit/utils`, 15
`unit/model_executor`, 9 `unit/managers`, 1 `unit/mem_ledger` — **none grew**.
The single `mem_ledger` failure is the same
`test_communicator_group_contract_612::test_no_runtime_group_is_missing_from_the_declaration`
throughout, checked by name (§8.6).

**+68 tests, 0 new failures.** `managers` 1357 → 1425 and the family 1169 → 1237
are the same +68, which is the arithmetic closing: the branch's six listed new
files are one `scheduler/` file plus five `managers/` files, and the family and
the `managers` arm each collect their own side of that set. Step 2's +10 are
exactly the ten cases in `test_frame_component_ballot_656.py` (10 passed
standalone), the seventh file (§2).

**The branch's own claimed family run was 1237P 7S and it reproduces exactly in
the merged tree.**

## 6. REGISTER — union-merge verified by count, 0 deletions, `C605-*` intact

| check | result |
|---|---|
| deleted lines in `CONTRADICTIONS_REGISTER.md` | **0** (`git diff --numstat` reports `7  0`) |
| added lines | **7**, one row each |
| file length | 2417 → **2424** |
| `C605-1`…`C605-17` occurrence counts | **identical at base and at final**, byte for byte |
| duplicate row labels anywhere in the file | **none** (46 rows, all unique) |

Verified by counting every `C605-n` occurrence rather than by eyeballing
headings — R7's lesson, kept. The seven new rows are **C22-b, C22-c, C22-d,
C22-e, C21-b, C21-c, C-B1**, each present exactly once. `C22-c` and `C22-d`
supersede readings this register itself carried; `C-B1` **retracts an evidence
base** (see §8.2).

The `schedul` codespell hit at line 1811 is the deliberate 15-character
`TASK_COMM_LEN` truncation from MERGE-R5 and **must not be "fixed"**.

## 7. LINT DELTAS — recorded, not acted on

| tool | BASE | FINAL | note |
|---|---|---|---|
| `black` 26.1.0 | 7 dirty of the 13 pre-existing touched `.py` | **11 of 13**, plus **3 of the 7 new** | **§1 — this merge makes it worse** |
| `ruff` 0.15.1 | 97 errors (13 files) | 97 (same 13); **all 7 new files clean** | no delta |
| `codespell` | 7 hits | **8** | one new: a second `unparseable` in `phase_flip_seam_reserve.py:219` |

`ruff` is the one clean story here: the seven new files pass with zero findings
and the thirteen modified files gained nothing.

## 8. THE STALE LOCAL REF TRAP IS STILL ARMED — now FIVE rounds behind

| ref | SHA at shift start |
|---|---|
| `origin/feat/route-a-631` | `6d169c04ab` |
| `origin/integration/r2` | `6d169c04ab` |
| **local** `feat/route-a-631` | **`0ae49fafb4`** — the MERGE-**R4** base, now FIVE rounds stale |
| local `integration/r2` | `481411ac6b` — the R7 base, four rounds stale |

`feat/route-a-631` is checked out in `/spinning/wt-631-routea`, which belongs to
another strand; git will not move it and forcing it would move another session's
working tree out from under it. This shift did what R6, R7 and R8 did: **every
merge source named as `origin/<branch>`, every push as `git push origin
HEAD:refs/heads/<line>`, `ls-remote` compared against `git rev-parse HEAD` after
each.** The stale refs were never an input to anything. **Do not "fix" them from
a merge shift.**

## 9. NOTHING ON GPU, ON PURPOSE

Every suite run used `CUDA_VISIBLE_DEVICES=99`. Serving on 30030 was not
touched, no GPU arbitration window was claimed, port 30099 was not touched, no
`pkill` was used, and `git stash` was never invoked. Nothing under `/etc`
modified. Pushed to **`origin` = the efschu fork only**; `upstream` was never a
push target.

Unlike R7 and R8, **no arriving test needed a device this round** — the
branch's seven new files all run CPU-only and all pass. R8 §1's orphan,
`test/registered/unit/mem_cache/test_gdn_resident_cap_floor_656.py`, is
unchanged by this merge and still 7F on the desk, still in a directory nothing
collects (§10.8).

## 10. STATE AT HANDOVER

- **Both lines at the same SHA** — `4f0b4f0bcb` plus this handoff commit on top,
  `ls-remote`-verified against local `HEAD` after every push.
- Working branch `merge/r9-batch` in `/spinning/wt-merge-r9` — same SHA, kept.
- New frozen baseline `/spinning/wt-merge-r9-base` at `6d169c04ab`, detached,
  clean — **kept deliberately** so R10 can diff against the tree R9 measured
  against. R8's, R7's and R6's frozen bases are also still there.
- Local refs untouched, including the stale ones (§8).

## 11. REMAINING UNMERGED BRANCHES

**78** local branches are not merged into the tip (79 at R8; one merged this
round). The ones recent enough to be live work, newest first:

| date | SHA | branch |
|---|---|---|
| 2026-08-12 | `d38bb6df32` | `trial/cumulative` |
| 2026-08-09 | `982b6434ce` | `feat/route-a-631-resume-gate` |
| 2026-08-09 | `00a1c50fcb` | `feat/gguf-q4-bringup-651` |
| 2026-08-08 | `27f3bf7996` | `fix/collective-stream-622` |
| 2026-08-08 | `18370879e3` | `integration/r3-probe-next2` |
| 2026-08-07 | `b851df7626` | `feat/dual-group-631` |
| 2026-08-06 | `cc2e03da59` | `serving/530-plus-603b` |

(`backup/pre-email-fix-s13` and `backup/pre-deps-strip-s13` are backups, not
merge candidates.) The three standing Claude strands own
`feat/gguf-q4-bringup-651` (#651), `fix/collective-stream-622` (#622/#649) and
the Route-A line; none was asked to be merged this shift, and none was.

## 12. CARRIED FOLLOW-UPS

Carried, none of it addressed by this shift.

1. **The long-soak MTTF proof is outstanding, and boot_v3 is not it.** The
   acceptance re-run took **364 cutovers with 0 abandons**, and at the
   acceptance's observed 1-in-320 rate a clean run of that length happens about
   **32 % of the time on an instance where nothing was fixed**. So boot_v3 is a
   **no-regression** result on C22, not positive proof. The positive proofs are
   boot_v2's two events (the levelling at 16:00:30Z, the valve at 16:01:24Z) and
   the desk red arms. A 95 %-level metal claim needs **~957 cutovers** (~3 h at
   this rate); 99 % needs ~1471. The counter is already in the log
   (`PHASE-FLIP cutover complete`, once per rank — divide by 3). **Fold the long
   soak into the next acceptance rather than treating 320-364 as a standard.**
2. **The ~280k empty-completion band needs its bisect — ~20 min, fully
   specified.** Pooled across four boots: **~5 of 7 cold probes empty at ~280k,
   0 of 12 at 240016 / 250026 / 300016 / 300026 / 332532 / 338916.** A narrow
   failing **band** above `max_position_embeddings` (262144) — not a ceiling,
   since 300k, 332k and 338k all pass — and intermittent within it, so a **rate**
   and not a threshold. `p02` (unique filler, `cached=0`, 280016, `completion=1`)
   killed the "repetitive content is fragile" confound. Cached prefills in the
   band are clean, 4 of 4. **Run: 260k / 270k / 280k / 290k, unique filler, cold,
   `cached` asserted 0** — about 5 probes. Note `C-B1`: the load driver's
   one-token completions at 4001/16001/32001/64001 were **the model emitting EOS
   on a prompt that asks nothing**, 100 % uniform on both boots, and were never
   part of this defect — most of the original evidence base was noise.
3. **The seam cost model is still ~3.8x low on the binding rank** — 484 MiB
   modelled against ~1830 MiB drawn. Unchanged by this shift; what changed is
   that the gate now acts on a **measured** draw instead of the model, so the
   model's error costs delays rather than breaches. The next measurement to buy
   is `scripts/s38_seam_price_vs_draw.py` against a boot of this configuration,
   which gives the seam-**scoped** price instead of a 9 s-window upper bound.
4. **The arena tail is priced with `max()` and the measurement says `+`.**
   `_staging_bytes` returns `max(wave_peak, draft_restore, arena_tail)` on the
   reasoning that the peaks belong to different legs; the stage walk shows the
   `weights_refill` commit happening with ~1214 MiB of seam state still
   outstanding (entry 2464, refill at 1250), so on the `tp_to_pp` leg the arena
   tail and the wave peak **do coexist**. Deliberately not changed — it widens
   the entry requirement on every seam. Measured, not guessed.
5. **kvso under PP — the spec-compliance note stands.**
   `--enable-kv-session-offload` is **refused outright** under `pp_size>1`
   (`server_args.py:7405`), so no argv produces vacate lines on a flip boot and
   the spec's "bs2-4 reserves including unused mamba states are spilled during
   bs1 time" is **structurally unreachable** there. The flip pressure ladder
   (cache / draft weights / weights-arena tail) is real spilling but it is
   **flip-seam spilling, not the idle-session mamba vacate the spec asks for**,
   and reporting the rung count as satisfying that axis would be the exact
   substitution the register exists to prevent. The dependency is a **sourcing**
   one, not storage — `GdnStateStore` is an interface — so the route is a
   PP-safe idle-session source, **not** lifting the PP refusal, whose stated
   reason (host pool rows sized from the boot vector) is real.
6. **`test_communicator_group_contract_612` — the one inherited `mem_ledger`
   failure — remains LOAD-BEARING.** `parallel_state` builds `flip_dcp`,
   `flip_pp` and `flip_tp`; `RUNTIME_COMMUNICATOR_GROUPS` does not declare them,
   and those undeclared groups allocate communicator buffers **inside** the
   `nccl_init_begin`/`nccl_init_end` gap, so the first boot to measure
   `TERM_NCCL_BUFFERS` will measure them while the term's signature does not know
   they exist. Fixing the declaration is a **prerequisite for trusting the NCCL
   term**. Note in passing: this merge introduces **no new communicator** — the
   ballot widens an existing `_collective_min` payload from 3 to 5 elements, and
   the cap agreement widens the KV rung's existing reduction from 4 to 8 fields,
   both on groups the round already reduces on.
7. **NIXL still blocks collection of `test/registered/unit/mem_cache/`.** One
   unconditional module-scope `ImportError` in
   `test_hicache_nixl_storage.py` takes the **whole directory** down at
   collection time, which is why no canonical arm covers it — and therefore why
   R8 §1's `test_gdn_resident_cap_floor_656.py`, the only gate on an
   instance-killing fix, is red in a place nothing looks. Unchanged this round.
8. Carried unchanged from R8 §13 and R7 §11-12, still open: the **first cutover
   of a boot has no measured draw** so the yield-withholding cannot protect it
   (`_seam_draw_max` is 0 until this rank has seen a cutover, and an unmeasured
   bucket is never a licence to invent a number); **the wire still carries no
   length** and the receiver's size check remains vacuous by construction (a
   16-byte framed header is the obvious follow-up, deliberately not taken because
   it changes a wire format pinned by
   `test_streamed_pack_equals_the_concatenation_reference`); **the weights row
   does not close under PP** (488 / 416 / 581 MiB, a 39 % spread); **the ledger
   residuum stays large** because it sums PEAK and STEADY-STATE terms into one
   total, which is a taxonomy change; **lazy RoPE's root cause** (ships default
   OFF; do not enable before the read-back guard exists) — note the acceptance
   **exonerated** it for the empty completion, since the probes ran at the
   shipped eager default and saw the same signature; the **1M pool cost is
   unattributed**; **L3's 1128 MiB corridor is un-re-measured**; the
   driver-unattributed residual band **164-276 MiB per card**; #363's stage
   actuator is desk code on the line; the first real `docker build` is the #384
   gate's first test; the #695 census lines still need a PP-unique rank identity;
   `route_a_631_prod_boot.sh` still diverges from the ship capture in seven flags.

## 13. NEXT, IN ORDER

1. **Run the pinned pre-commit `black`** over the 14 dirty files as a standalone
   formatting commit, outside a merge shift (§1), and find out why the hook is
   not firing for branch authors — the round-trip R8→R9 shows a one-off cleanup
   does not hold.
2. **Bisect the ~280k band** (§12.2). It is ~5 probes and ~20 minutes, it is
   fully specified, and it is the cheapest open item on the list by a wide
   margin. Pin the flip policy (no auto-flip) if the intent is to measure the
   completion and not the seam.
3. **Fix the `mem_cache` collection break** (§12.7) — one unimportable NIXL
   module hides an entire directory, and with it the only gate on R8's
   instance-killing mamba-floor fix.
4. **Declare `flip_dcp`/`flip_pp`/`flip_tp` in `RUNTIME_COMMUNICATOR_GROUPS`**
   (§12.6) *before* the first boot that measures the NCCL term.
5. **Buy the long soak** (§12.1). ~957 cutovers is the number; anything shorter
   is a no-regression result and should be reported as one.
