# HANDOFF MERGE-R10 — one defect that took three fixes, and the third was mine

Shift `656-liveslot-fix`. Worktree `/spinning/wt-merge-r10`, branch
`merge/r10-batch`, based on `origin/feat/route-a-631` at `567ba68023` (the
MERGE-R9 tip both lines carried). Frozen pre-merge baseline
`/spinning/wt-merge-r10-base`, detached at `567ba68023`, clean tree. Evidence:
`/spinning/evidence-631/merge-r10/` (suite) and
`/spinning/evidence-631/liveslot-656/` (metal).

Merged: `feat/soak-fixes-656` — the soak shift's red test, plus this shift's
three fixes for the defect it pinned.

**Both lines are at `cb8da83774`**, `ls-remote` verified against
`git rev-parse HEAD` after every push. This handoff cannot name the commit that
contains it, so the actual tip is the single docs-only commit sitting directly
on top of `cb8da83774` — the convention R7, R8 and R9 used.

ERRORS FIRST.

---

## 1. I BROKE THE CORRIDOR LAW AND THE METAL CAUGHT IT

The C22-d fix (§3) made `KvRowCap.sort_free_lists` run **every seam round on
every rank** instead of only on the rare rounds the cap agreement moved a rank.
That function did:

```python
setattr(self._alloc, name, torch.sort(pages).values)
```

On a **device** tensor `torch.sort` allocates a values tensor AND an indices
tensor. The free list is one int64 per row — 4.7 MiB at this rig's 586642 rows
— so ~14 MiB transient per call and up to ~34 MiB across the normalise and
release paths, taken **at the seam**, which is exactly where this rig's
corridor is tightest.

| boot | code | gpu0 continuous min | samples < 1024 |
|---|---|---|---|
| soak boot 1 | before | 1028 MiB | **0** of 33459 |
| soak boot 2 | before | 1084 MiB | **0** of 125753 |
| leg 2 | C22-d | 978 MiB | **2** |
| leg 3 | C22-d + C22-e | 990 MiB | **4** |

`1024 - 990 = 34`, the transient to the MiB. The corridor law is a hard user
limit; a mechanism that is correct about ids may not pay for itself in device
memory.

Fixed in the same shift, not deferred: the sort now runs on the **host** and
writes back with `copy_` into the same storage, so the device allocates
nothing, with an equality guard that skips the write-back entirely when the
list is already ascending. `release`, where the merge genuinely changes the
tensor SIZE, does its `cat` and `sort` on the host too — one device allocation
instead of four. Re-verified on metal (§5, leg 4).

**The general shape, recorded as register row 84 because this corpus keeps
meeting it:** a correctness fix was judged on what it DECIDED and not on what
it ALLOCATED, and it was moved from a rare path to a hot one in the same
change. **When a mechanism's frequency changes, its cost has to be re-measured
at the new frequency** — the old measurement was taken under the old one.

## 2. THE SIXTH CONSECUTIVE SHORT FAMILY LIST

`test/registered/scheduler/test_flip_live_slot_agreement_656.py` arrived with
the soak shift and `scripts/run_631_flip_family.sh` was not extended in that
commit. R9 §2 wrote the fifth instance; R8 §§1–2 the ones before it.

Milder than R8's orphan, which sat in a directory no arm collects at all: this
file lives in `test/registered/scheduler/`, so it ran and was green in the
scheduler arm. What it missed is the **canonical family sweep** — the list that
is supposed to be the single answer to "what covers the flip". A private arm is
not a canonical list. Fixed here, in the same batch as the fix it pins.

## 3. THE DEFECT, AND WHY IT TOOK THREE FIXES

The soak shift left a red test and a named class: the live slot SET is never
agreed between ranks, only the row COUNT. Boot `boot_m3`: after 194 clean
cutovers every `pp_to_tp` flip was abandoned with `THE DIVERGING TERM IS: the
live slot set`, PP1 the outlier in 4 of 4 episodes and the only rank with
corridor-bounded recoveries (7 of 7).

### (a) C22-d, the trigger — named from the CODE, not from the log

`KvRowCap.release` **sorts** the free list; `engage` preserves eviction order.
A corridor-bounded `recover()` therefore reorders the recovering rank's free
list while peers that never shrank keep theirs in eviction order. The only
re-normalisation is `reconcile_to`'s trailing sort — and `collective_cap_target`
returns `None`, skipping it, **precisely when the group's exposed counts already
AGREE**. The allocator takes from the FRONT, so from that moment the ranks hand
different physical rows to the same logical token: identical cardinality,
different set, identical `POOL CENSUS`.

That is why the trigger is **narrower than "a rank came back short"**, which is
the question the soak shift left open. Soak leg C's single 23199-row recovery
left the counts UNEQUAL, so the next shrink levelled the group by construction
and opened no divergence. Boot 1 took seven recoveries, and it is the ones that
left the counts EQUAL that opened the wedge.

Closed in two halves, because prevention cannot undo rows already handed to
live requests and cached in the radix tree — which is why draining to zero
requests and `/flush_cache` both failed to clear boot 1:

* **Source:** `normalize_free_lists` runs unconditionally at the KV rung, every
  rank, every seam round, whatever the cap agreement decides to do. No
  collective, no bytes moved.
* **Recovery:** the rung's existing reduction widens **8 → 12 fields** (the move
  R2 made 4 → 8) to carry a membership digest, the group's row extent and the
  group's MINIMUM BACKED rows. On disagreement every rank adopts the group
  **UNION**, computed on the same MIN channel because `MIN(-p)` is `-OR(p)`.

A union never removes a row from the rank holding it, so no request loses
context and no backing is released — a rank-0-authoritative broadcast or an
intersection would drop a peer's live rows at the seam.
`reshard_plan.rows_of` is `(slot // s) * ratio + (slot % s - lo)`, a pure
function of the SLOT ID, so no already-agreed row changes destination. It costs
nothing when the ranks agree: soak boot 2's 1134 consecutive cutovers return at
the first branch, and that is pinned by a test asserting the agreeing path
enters no collective at all.

**The bound that is not negotiable:** a row at or above a rank's backed count
is unmapped there, and the mover reading it is `cudaErrorIllegalAddress`, which
kills every rank rather than raising. A union reaching the group's minimum
backing is REFUSED into the existing unanimous abandon, named.

### (b) C22-e — metal proved C22-d necessary and NOT sufficient

This is the part a desk could not have produced. On metal, C22-d behaved
exactly as designed — it detected the divergence and **safely refused** — and
that refusal exposed the real root one rung down:

```
05:40 PP2  proposal: current=585390 floor=585903 (max_live=585390)
05:40 PP1  proposal: current=546236 floor=546749 (max_live=546236)
05:44 all  FLIP ABANDONED ... the group's union reaches row 585390 and the
           poorest rank has only 546236 rows BACKED
```

`recover()` is bounded by each rank's own distance from the corridor law —
correctly, the grow being an allocation on the card that needed relieving — so
the corridor-bounded rank comes back lower than its peers and **the group is
left with two different ID SPACES**. The peers then hand out ids the poorest
rank cannot map, and from there BOTH repairs refuse — **and both refusals are
right**. `collective_cap_target` declines because `max_floor > capable`: it may
not withhold ids a peer's request is using. C22-d's union declines because
framing row 585390 on a rank backed to 546236 is a read of unmapped memory.

Two correct refusals and a flip that never happens again: that boot ran to the
announced 9-refusal livelock with `/health` 200 and every request intact. The
C22 valve held throughout — this was never a crash.

So the divergence is prevented **where it is created**. `recover_kv_backing`
now reduces `[backed, -backed]` on the `tp->pp` post-cutover hook — a point
every rank reaches exactly once per unanimous cutover — and every rank caps its
allocator to the group's MINIMUM backing. No pages released, no memory
committed: an id decision, resting on the cap agreement's own argument that
under pure PP rows above the group minimum could never have been admitted
against.

`level_recovery_to` differs from `reconcile_to` in the one way that matters: it
**remembers the reservation** before capping below it. `reconcile_to` clears
`_rows_at_boot` when the level reaches the ceiling it knows about, a fully
recovered rank has already cleared it, and `recover()` returns 0 immediately on
a None — so levelling with `reconcile_to` alone would make the group level a
**ratchet**, the one outcome the standing rule forbids. Pinned by its own test.

### (c) The corridor regression, §1.

## 4. THE RED TEST IS GREEN, AND THE ASSERTION MOVED — DELIBERATELY

The soak shift's red test asserted that three ranks' digests collide, computed
by **re-deriving** them from the three RAW per-rank views. No agreement step can
satisfy that: the fixture builds three genuinely different sets and no fix
mutates `live_slots_fn` retroactively. As a statement of the goal it was
unreachable; as a statement of the defect it measured the fixture rather than
the runtime.

What the ballot votes on is the set each rank **frames**, which the runtime now
records (`last_framed_slots_digest`). Asking THAT is the same sentence the old
docstring wrote — *"every rank must frame the SAME digest — i.e. the flip
completes"* — against the value that sentence was about, and it is **strictly
harder** than the cutover assertions beside it: a fix that cut over while
framing different sets would pass those and fail this. Added alongside: every
rank's framed set must be a superset of its own live rows, which is the
property the alternative repairs violate.

The file's second test asserted today's behaviour (two abandons, no cutover)
and its own docstring said the numbers *should* change once set-agreement
lands. It now runs three **alternating** legs — the same direction twice is a
no-op once the flip succeeds — and asserts a cutover on every one, with the
agreement firing on both legs.

**Can-fail arms** (an instrument that cannot fail has certified nothing — see
the deliberately removed `torch.unique` belt in `KvRowCap._apply`):

* the agreement disarmed reproduces the metal signature on the identical
  fixture;
* a union reaching past the group backing refuses and names the numbers;
* an agreeing ballot enters **no collective**;
* the ballot still abandons a **wave-partition** divergence with nothing
  stubbed — a term the union does not touch — and attributes it correctly.

**Four sibling arms in `test_flip_frame_agreement_656.py` and
`test_phase_flip_runtime.py` planted set divergences and expected an abandon.**
The agreement now repairs those. They were not weakened: each now disarms the
agreement **explicitly** (that is what makes them can-fail arms — they must
reproduce the state the protection is absent in), and the repaired behaviour is
pinned by new tests of its own.

## 5. METAL — the mechanism fired, and the counterfactual is on the same rig

Four legs, all on `feat/soak-fixes-656`, argv identical to the soak's boot and
therefore to the R2 acceptance `boot_v3`. Code is the only difference between
them and the soak boots. Evidence `/spinning/evidence-631/liveslot-656/`.

**Leg 3 (C22-d + C22-e), the functional proof.** Every corridor-bounded
recovery immediately followed by a levelling on all three ranks:

```
05:57:18 PP1 recovered 581333 of 586642 (corridor-bounded)
05:57:18 PP1 backs 581333, poorest backs 581333, capped at 581333 (+0)
05:57:18 PP0 backs 586642, poorest backs 581333, capped at 581333 (-5309)
05:57:18 PP2 backs 586642, poorest backs 581333, capped at 581333 (-5309)
```

| | leg 3 |
|---|---|
| cutovers | **56** |
| corridor-bounded recoveries | **9** |
| recovery levellings | **8** |
| wire frame divergences / FLIP ABANDONED | **0 / 0** |
| union refusals | **0** |
| KvReshardError / tracebacks / SIGQUIT / CANNOT FUND | **0 / 0 / 0 / 0** |

**The counterfactual is leg 2, on the same rig and the same argv**, with C22-d
but without C22-e: the same recovery shape produced 4 divergence episodes, 6
abandons and a permanent livelock. That is the comparison that makes leg 3 a
proof rather than a quiet run — and it is why C22-e exists.

**Leg 4 re-verifies the corridor** after the §1 fix, on the code both lines now
carry, over **46 minutes**:

| | leg 4 |
|---|---|
| cutovers | **40** |
| corridor-bounded recoveries / levellings | **6 / 5** |
| wire frame divergences / FLIP ABANDONED / union refusals | **0 / 0 / 0** |
| KvReshardError / tracebacks / SIGQUIT / CANNOT FUND | **0 / 0 / 0 / 0** |
| corridor | **27907 samples/card, minima 1042 / 2135 / 1076 MiB, 0 below the 1024 law** |
| requests | 36, **0 non-JSON, 0 failures** |

Leg 4 also shows the levelling is not rank-specific — PP2 is the poorest in one
episode and PP1 in another:

```
PP1 backs 577867, poorest backs 577115, capped at 577115 (-752)
PP2 backs 577115, poorest backs 577115, capped at 577115 (+0)
PP0 backs 585846, poorest backs 577115, capped at 577115 (-8731)
```

**Leg 4's boot stamp is one commit stale**, and the evidence directory says so
in `PATCHSTATE_leg4.txt`: the boot stamped `79eac07608` and ran the content of
`1fd4d514e6`, proven by a clean tree and an empty `git diff HEAD -- python/`
taken **while the server was still importing those files**. The honest sentence
is "leg 4 ran the content of `1fd4d514e6`", not "leg 4 booted at `1fd4d514e6`".
The standing rule wants load against a NAMED commit; this leg satisfies its
substance and not its form.

Full detail: `/spinning/evidence-631/liveslot-656/SOAK_C22DE.md`, carried into
the repo record beside it.

**Honest note.** The union REPAIR path (`live slot SET agreed by union`) has
fired 0 times on metal, because with C22-e the divergence no longer forms — the
source-half and the levelled recovery prevent it. Its proof is the hermetic
suite, including the can-fail arm. That is a **prevention** result, not a
demonstration of the repair on hardware, and it should be read that way.

## 6. SUITE — every arm, every failure SET, name-identical at every step

Same interpreter (`/spinning/htsglang-gpu/.venv/bin/python`),
`PYTHONPATH=<worktree>/python`, `CUDA_VISIBLE_DEVICES=99`, `pytest --color=no`,
one directory per invocation. Runner
`/spinning/evidence-631/merge-r10/arms.sh`.

| suite | BASE `567ba68023` | step 1 (C22-d) | step 2 (C22-e) | step 3 (corridor) |
|---|---|---|---|---|
| #631 flip family (canonical script) | **1247P 7S** | **1260P 7S** | **1264P 7S** | **1268P 7S** |
| `unit/managers` | 9F 1425P 18S | — | 9F 1429P 18S | 9F **1433P** 18S |
| `unit/mem_ledger` | 1F 437P | — | 1F 437P | 1F 437P |
| `unit/model_executor` | 15F 594P | — | 15F 594P | 15F 594P |
| `unit/server_args` | 615P | — | 615P | 615P |
| `unit/turnkey` | 116P | — | 116P | 116P |
| `unit/utils` | 46F 348P 4S | — | 46F 348P 4S | 46F 348P 4S |
| `unit/docker` | 4P | — | 4P | 4P |

**Continuity, the check R8 specified and R9 repeated.** This shift's BASE column
reproduces R9's post-step-2 column **exactly** on all eight arms, including the
flip family at 1247P 7S and `managers` at 9F 1425P 18S. The chain back through
R9's, R8's, R7's and R6's frozen bases is unbroken.

**Failure SETS diffed by name, not merely counted**, for all four red arms, at
every step:

| arm | names at base | at step 2 | at step 3 | diff |
|---|---|---|---|---|
| `managers` | 10 | 10 | 10 | **identical** |
| `model_executor` | 15 | 15 | 15 | **identical** |
| `utils` | 46 | 46 | 46 | **identical** |
| `mem_ledger` | 1 | 1 | 1 | **identical** |

R9 §5 recorded that `utils` needs a second pass because two of its 46 are
emitted as `SUBFAILED(module=...)` by
`test_capability_vendor_gates.py::test_cutedsl_blackwell_gates`. **That is not
the only arm with the problem.** `managers` also carries a `SUBFAILED` line —
`SUBFAILED(loop='event_loop_pp')` — so a `^FAILED ` grep finds 8 of its 10
lines. The grep used here matches all three prefixes (`FAILED`, `ERROR`,
`SUBFAILED`) on every arm and therefore covers 10 of 10, 15 of 15, 46 of 46 and
1 of 1. **A name diff scoped to `^FAILED` under-reports at least two arms, not
one.**

The separate `scheduler` + `unit/managers` arm used during development: base
`811c1c6bf1` 21F 1905P 19S 25E → 20F 1923P 19S 25E, **0 new names, exactly one
gone** (the red test), identical across all three fix steps.

**+21 tests, 0 new failures.** The family's 1247 → 1268 is: **+10** for the
newly-collected `test_flip_live_slot_agreement_656.py` (§2 — it existed and was
never swept), **+2** frame-agreement arms, **+1** abort-deferral arm, **+4**
C22-e arms, **+4** normalisation arms.

## 7. REGISTER — union-merge verified, 0 deletions, `C605-*` intact

| check | result |
|---|---|
| deleted lines in `CONTRADICTIONS_REGISTER.md` | **0** (`git diff --numstat` reports `209  0`) |
| added lines | **209** |
| file length | 2424 → **2633** |
| `C605-1`…`C605-17` occurrence counts | **identical at base and at final**, byte for byte |
| duplicate row labels anywhere in the file | **none** |

Rows added: **82** (C22-d closed, the trigger named from the code), **83**
(C22-e, the recovery creates the divergence), **84** (the corridor regression
and its general shape). Rows 78–81 arrived with the soak branch.

The step-2 merge conflicted in this file — row 82 on the merge branch against
row 83 on the feature branch — and was resolved as a **union**, both rows kept
in order, which is the canon rule and is why the deletion count is 0.

## 8. LINT — no delta on any gate

| gate | base | final |
|---|---|---|
| `ruff` | 3 errors on the touched files | **3**, same names |
| `codespell` | 4 hits | **4**, same |
| `black` | 7 of 9 touched files dirty | **7**, the same 7 |

R9 §1 recorded that `black` is not enforced and that R9 made it worse by seven
files. **This shift does not extend that**: the dirty set is unchanged, base to
tip. The standalone formatting commit R8 asked for is still overdue and still
does not belong inside a merge shift.

## 9. WHAT TOUCHED THE GPU, AND WHAT DID NOT

Unlike R9, this shift **did** claim a GPU window — the task required a metal
proof. `/spinning/gpu-arb/holder` held by `656-liveslot-fix`; heartbeat in its
own systemd scope; serving on 30030 stopped for the window and restored by this
shift (§10). Every **suite** run used `CUDA_VISIBLE_DEVICES=99`.

**One self-inflicted error worth recording.** The first suite run was launched
without `CUDA_VISIBLE_DEVICES=99` and put 702 MiB on the 5090, which made the
first boot attempt fail its budget check by 0.21 GiB. The boot's error message
was exactly right and named the cause. Cost: one boot attempt. **The canon's
`CUDA_VISIBLE_DEVICES=99` is not hygiene, it is a precondition for any boot
sharing the box.**

Port 30099 was never touched as a process. No `pkill`. `git stash` never
invoked. Nothing under `/etc` modified. Pushed to **`origin` = the efschu fork
only**; `upstream` was never a push target.

**Long-running helpers run in their own systemd scopes** (`hb-…`, `cs…`,
`load…`, `deep…`, `boot-…`). This shift's predecessor process was killed
mid-window by a service restart and its heartbeat, corridor sampler and load
driver died with it, leaving the instance idle for 5.7 h inside a held window.
The boot survived because it was in a scope; nothing else was. **Everything
that must outlive the agent goes in a scope, not just the server.**

## 10. STATE AT HANDOVER

- **Both lines at the same SHA** — `cb8da83774` plus this handoff commit on
  top, `ls-remote`-verified against local `HEAD` after every push.
- Working branch `merge/r10-batch` in `/spinning/wt-merge-r10` — same SHA, kept.
- Feature branch `feat/soak-fixes-656` at `1fd4d514e6`, pushed, kept.
- New frozen baseline `/spinning/wt-merge-r10-base` at `567ba68023`, detached,
  clean — **kept deliberately** so R11 can diff against the tree R10 measured
  against. R9's, R8's, R7's and R6's frozen bases are also still there.
- `/spinning/wt-656-liveslot-base` at `811c1c6bf1`, the branch-level baseline
  for the development arm, kept.
- Local refs untouched, including the stale ones (R9 §8 — **do not "fix" them
  from a merge shift**).

## 11. WHAT THE NEXT SHIFT SHOULD PICK UP

1. **The union repair has never run on metal** (§5). C22-e prevents the
   divergence, so the repair is dead code on the happy path by design. If it
   should be proven on hardware, it needs a deliberate fault injection, and
   that hook does not exist.
2. **`blocking_guards` is append-only** and at `DEFAULT_SEAM_ABANDON_CAP = 8`
   group abandons the direction is refused for the rest of the boot with no
   path back short of a restart. The soak shift found it; leg 2 reached it. Now
   that both refusal paths are correct-by-design rather than bugs, a boot that
   legitimately refuses 8 times still ends its own flipping.
3. **The ~280k band** remains unreproduced (register 81); the GDN/mamba
   chunked-scan boundary and attention-backend/CUDA-graph bucket sizing at
   those depths are still unexamined by any shift.
4. **The standalone `black` commit** (R8 §10, R9 §1) is three rounds overdue.
