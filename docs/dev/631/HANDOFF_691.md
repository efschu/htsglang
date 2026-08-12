# HANDOFF 691 — #656 / #631 Route A, successor 47

Three items, all three landed, and **two of the three contradicted the brief
that ordered them.** Errors first.

---

## 1. ERRORS FIRST

### 1a. THE AUTHORIZED FAST-FORWARD WOULD HAVE DESTROYED THREE OF THE USER'S COMMITS

My brief said, as operator-authorized fact: *"N46 verified integration/r2 is a
strict ancestor of the current line — the fast-forward is available and
unblocked. Execute it."* **That was true of the LOCAL ref and false of the
remote.** N46 measured `integration/r2` with `git rev-list` against the local
branch, which sat at the merge-base `aac5547527`. `origin/integration/r2` was
at `64f24b92fb` and carried **three README-only commits from 2026-08-01**
(`179f1be25d`, `b72450d958`, `64f24b92fb`) that the line had never had.

A fast-forward of `integration/r2` onto the line — the literal instruction —
would have required `--force` against the remote and would have **discarded
all three**. What caught it was the push rejection, i.e. git's own safety, not
my reading. **A ref name is not a ref. Measure `origin/<branch>`, not
`<branch>`, before calling anything a fast-forward.**

Merged instead: `e16fad731a`. Conflict-free **by measurement, not by luck** —
the remote side touches `README.md` and nothing else (`+91/-0`), and the line
had not touched `README.md` since the merge-base (identical blob
`1f6da3f96f`), so exactly one side changed the one changed file. Unlike the two
refused #559 branches this content is **not** superseded: the Docker usage
guide, the fork CLI flag reference and the not-yet-in-image list are absent
from the line's README (`HEAD` had 0 hits for `ghcr.io`, `docker run`,
`rank-auto-reserve-mib`, `NCCL 2.30.7`; the remote had them all).

Suite **1095 passed / 0 failed** on the integration ref before the merge and
again after it. Both refs pushed non-forced, now `e16fad731a`.

The two refused #559 branches (`bugfix/pd-mamba-conv-state-transfer`,
`docs/features-vs-upstream`) are **closed by refusal on N46's measurement** and
were NOT deleted — branch deletion stays with the user.

### 1b. BYTE-IDENTITY: THE INSTRUMENT WAS THE DEFECT FOR THREE SHIFTS, AND THE ANSWER IS THE OPPOSITE OF THE ONE ASKED FOR

Booked as **C31 + law 21**. Two separate errors here, and the second is worse
than the first.

**First: the flags coexist, on metal.** `--enable-deterministic-inference` +
`--enable-kv-session-offload` + flashinfer at `--chunked-prefill-size 4096`
answered a trivial `/generate` in **0.378 s** where v8 hung 55 s. HANDOFF_689's
*"byte-identity is therefore not obtainable on this rig"* is falsified and
N46's corrected C30 diagnosis is confirmed on metal.

**Second, and this is the real finding: the shift was asked to PROVE
byte-identity across a park. The metal says byte-identity is BROKEN by the KV
spill round trip.**

| run | pressure | spilled | result |
|---|---|---|---|
| floor x3 | quiescent | none | 3/3 identical `89f0c7e305c1ceab` |
| `s47c` | 4 concurrent, NO fill | **none** | **4/4 identical**, cohort-0 included |
| `s47` A | 4 + 3 fill | cohort-0 only | 3/3 others identical; cohort-0 `b25a8fc1` 1673 ch |
| `s47b` B | 4 + 3 fill | cohort-0 only | 3/3 others identical; cohort-0 `81e02f5e` 1706 ch |
| `s47f` F | 8 + 5 fill | cohort-0 only | **7/8 identical**; cohort-0 `eb0c3615` 1639 ch |

**17 never-spilled generations byte-identical across 5 runs. All 3 spilled
generations divergent — from the reference and from each other.** The spill
path does not shift the output, it reintroduces nondeterminism into a boot
whose entire purpose was to remove it.

**The `s47c` falsifier is the load-bearing run.** cohort-0 is also the
first-submitted, so "position 0 is special" was live. Remove only the fill
pressure — same cohort, same concurrency, same prompt — and cohort-0 does not
spill and IS identical. Position, concurrency and batch composition are
excluded **by measurement**.

**Why three shifts missed it, which is law 21.** The defect needs determinism
live AND a spill to occur; no earlier run had both. s45 booked "this run
separates nothing" (all four digests distinct), HANDOFF_689 escalated that to a
claim about the RIG, and the next shift was told not to try flag variations.
**A null from an instrument with no demonstrated floor is a fact about the
instrument.** The check that broke it open cost 55 s: three quiescent repeats
of the reference prompt, required to match, BEFORE asking the driver to judge
anything.

**A second instrument defect — FOUND AND FIXED THIS SHIFT.**
`park_complete_proof2.py` exited **3 ("NOTHING PARKED")** on every run that
contains the finding. Its arm assignment keys only on file-tier
`PARK commit rid=` (`PARK_RE`, `:166`), so a host-tier spiller is filed into
the **CONTROL** arm, where it sets `control_arm_identical_to_reference = False`
and makes a clean control look contaminated. Wrong in both directions at once:
the round-trip arm is empty AND the control arm is poisoned by the very session
that belongs in the other arm.

`park_complete_proof3.py` (in `/spinning/evidence-631/s47/`) assigns arms from
the **union of both exits** — `SPILL(partial)` / `first spill tick` /
`WAVE-BACK` / `spilled session ... finished on host` alongside the file-tier
park family — and reports the tier per rid.

**Validated by replay against the recorded log, not by inspection.** Run
through the real `probe_v10.log`, v3 puts `cohort-0` in the round-trip arm and
leaves a clean three-member control, and its verdict logic on run A's recorded
digests returns:

    round-trip arm : ['s47-cohort-0']              identical = False
    control arm    : [cohort-1, cohort-2, cohort-3] identical = True
    -> EXIT 1 DEFECT: parked arm differs while the control arm MATCHES

**Same recorded run: v2 said exit 3 "no claim can be made", v3 says exit 1
"the round trip is corrupting the session".** The driver's own verdict logic,
once the arms are assigned correctly, reaches the C31 conclusion independently.
v3 has NOT yet been run live against a server — only its scanner and verdict
paths are executed, which are exactly the parts that changed.

**Not closed:** the FILE-tier park arm. The host tier is sized to hold **one**
full-context region — proved by the refusal at
`--kv-session-offload-host-ram-gib 0.12`: *"cannot hold even ONE full-context
session ... 3933 tokens < 32770"* — so one spill fills it and no second
concurrent spiller appeared even at COHORT=8. Chunk 4096 buys determinism but
costs the concurrency that reached the file tier at chunk 256. **The way
through is a bigger pool** (raise `--max-total-tokens` well above 4096 so chunk
4096 still admits several sessions), not a smaller host tier.

**THE MECHANISM IS NO LONGER UNKNOWN — localized from source the same shift and
verified line by line (full detail in C31).** It is not the round trip:

* **`kv_session_offload.py` has ZERO references to
  `enable_deterministic_inference` / `fixed_split_size` / `split_tile`**
  (`grep -c` = 0). The feature was never wired for determinism.
* The resident path pins `fixed_split_size=self.decode_split_tile_size`
  (`flashinfer_backend.py:1668`); the spill path's `w.plan(...)` calls
  (`:3731`, `:3762`) pass it **not at all**, falling back to heuristic split-k.
* A spilled session's attention is a **chain of partial decodes folded by
  `_safe_merge_state` (`:3873`), a sequential non-associative LSE merge** — a
  different reduction tree from the resident decode, on byte-identical KV.
* **The fold's shape is chosen by a timing query**, which is why the three
  spilled runs differ from EACH OTHER:
  `copy_inflight = not self.backend._sess_wave_done.query()`
  (`kv_session_offload.py:4983`), a CUDA event progress probe, plus the live
  allocator free list (`:4918`). Neither depends on the token sequence.
* The path never claimed bit-exactness: its own self-test passes at
  `rel < 5e-2` (`:4282`).

Ruled out from source: fp8 round trip (host pool inherits `store_dtype`=uint8,
indexed byte memcpy), recompute (target never re-forwarded; draft backfill
gated off at `spec=False`), mamba/GDN (never spilled).

**So the honest framing is not "a bug in the spill path".** kv-session-offload
and deterministic inference are **semantically incompatible as currently
built** — the spill path's design IS a runtime-adaptive decomposition. Making
them compose means pinning the boundary and the split-k, at an unpriced
performance cost. That is a product decision, not a bug fix.

**BOOKED AS A REFUSAL PAIR, and the direction matters:**

> **REFUSED (guarantee):** a session that went through a kvso spill is
> **excluded from any determinism guarantee**. The **#412 determinism-
> certificate mode must NAME this exclusion in the certificate** — written
> onto the #412 row of `ROADMAP_456_matrix_execution.md` *before* the mode is
> built, so it cannot ship a claim it cannot honour.
>
> **NOT REFUSED (boot):** the flag pair. The two flags boot and serve
> correctly together at a sufficient chunk budget; most sessions never spill.

Getting that split backwards is a documented failure on this line — C30 records
a shift told to "wire a LOUD boot-time refusal naming both flags", which would
have rejected a working config while leaving the real trap armed. **The object
of refusal is the CLAIM, not the configuration.**

Operational gap worth knowing today: **spilling is silent to the client.**
There is no per-response marker distinguishing an answer that came back through
the host tier from one that never left the device, so a caller cannot tell a
guaranteed answer from an excluded one. #412 would need that signal; the
`PARK/SPILL rid=` records already carry what it needs.

### 1c. THE #330 DIAL BRIEF CONTAINED TWO FALSE PREMISES

Booked as **C32**. Item 3 passed (`PROOF_EXIT=0`) but two of its four
acceptance criteria were unmeasurable as written.

* **"the guard calls dial (C18)"** — INVERTED. The **dial calls the guard**:
  `apply_budget_request` -> `_relieve_for_reduction` (`vram_dial.py:598-599`)
  -> `_corridor_relief._relief` (`:1106-1108`) -> `CorridorGuard.ensure_headroom`.
  `HANDOFF_684:4-5` already said so.
* **"sessions return to graphs"** — FALSE PREMISE, unmeasurable. The graphs are
  **never left**: the dial commits pages behind a stable VA reservation, *"No
  tensor moves, no CUDA-graph re-capture"* (`vram_dial.py:23-25`). No such log
  line exists or can. Measured the converse instead.
* **"the dial refuses PP=3 boots"** — imprecise in a way that matters. There is
  no `pp_size` check anywhere in the dial. It refuses on
  `uneven_dcp_active(dcp_size)` (`:1291`), and `dcp_size = tp_size`, so the
  ship config fails because **TP=1 per stage leaves no DCP axis** — not because
  PP is PP.

---

## 2. #330 — FIRST DIAL BOOT ON METAL, EVER

No file under `/spinning/evidence-631/` had ever contained `VRAM-DIAL` before
this shift. Boot: `/spinning/evidence-631/s47/boot_dial_tp3.sh` (TP=3, weighted
uneven DCP, vector `[30, 17, 17]`, port 30043).

| axis | measurement |
|---|---|
| dial-down under load, HTTP latency | **2.0 ms** (synchronous arming) |
| ladder asked before capacity arithmetic | **YES** |
| commit | `VRAM-DIAL DONE SHRINK 327760 -> 69824` |
| pages to the driver | rank 0 `released 4000.0 MiB`; ranks 1/2 `128.0 MiB` |
| NVML cross-check, out of band | target free **8132 -> 11584 MiB** |
| raise | C `69824 -> 327744`, real generation OK after it |
| corridor | **202 samples, 0 breaches**, min 3485 / 7584 / 4239 vs law 1024 |

**The ladder is ordered correctly and yields nothing:** *"the corridor relief
ladder returned 0 MiB now; the residual is funded by the capacity arithmetic."*
Both providers registered (`allocator-cache[local]`, `draft-weights[rebalance]`)
and the full 4096 MiB still came from the arithmetic — `draft-weights` returns 0
outside `PHASE_PP`, and **no `RELIEF_PARK` or `RELIEF_HOST` provider is
registered anywhere in the tree**. Ordering proven; effect not. Do not trust the
dial's own docstring (`vram_dial.py:626-631`), which claims a
"local -> park -> host" ladder of which two tiers do not exist.

**NEW RISK (C32, open): one card's budget is a GLOBAL lever.** Cutting one
rank by 4096 MiB collapsed the ceiling **327760 -> 69824, i.e. 79% of global KV
for 12% of one card's budget** — global `max_total_num_tokens` is a min-reduce
over per-rank (capacity/ratio) units, so the dialed rank binds everyone. Ranks
1 and 2 lost the same 79% without being dialed. Correct arithmetic, restored on
the raise, but any "dial one card to lend a co-tenant VRAM" model is off by ~6x
here. Not in DESIGN_330.

**dial x flip composition remains structurally blocked on this rig.** C23's
two-actuator race needs **TP>=2 per PP stage** + `--enable-phase-flip` +
`--enable-vram-dial`; three cards can host that only as PP=1.

---

## 3. THE RIG AS I LEAVE IT

* **Serving is UP on 30030, ship config**, restored by me with `setsid` from
  `/spinning/evidence-631/s43/boot_ship_30030.sh` after verifying that script's
  flag set against the live `/proc/<pid>/cmdline`: **31 flags, zero
  differences**. I stopped serving once (05:58Z) and brought it back. Nobody
  owes a restore.
* Verified with a **real generation**, not health alone.
* Router 30099 untouched throughout. All three cards were at 3 MiB between
  every boot.
* Operational trap worth inheriting: **`cd X && cmd &` backgrounds the WHOLE
  list**, so the `cd` happens in the background subshell and every follow-up
  command in that same call runs in the ORIGINAL cwd. It cost me one bogus
  measurement (I "verified" the merged README against `/spinning/shvllm`'s
  README and read 110 lines / 0 markers). Use absolute paths in every
  post-launch check.

---

## 4. WHAT THE NEXT SHIFT SHOULD DO, IN ORDER

1. **CONFIRM C31's mechanism on metal — it is already localized, so this is a
   confirmation, not a hunt.** No code changes needed, two existing knobs:
   set `--kv-session-offload-wave-back-min-free-tokens` above the pool size to
   freeze the boundary at 0 for the whole episode. If the spilled outputs
   become identical **to each other** while still differing from the reference,
   the timing-gated boundary is confirmed as the run-to-run variable; if they
   still all differ, the unpinned split-k is also live. Then raise
   `--kv-session-offload-block-size` above the context so the merge chain
   collapses to a single partial. Free extra signal with no generation at all:
   boot once with `KVSO_ATTN_SELFTEST=1` — `_sess_attn_selftest`
   (`flashinfer_backend.py:4123`) already prints per-block-count twin-vs-
   monolithic deltas on byte-identical KV.
2. **Run `park_complete_proof3.py` live** (§1b). Its scanner and verdict paths
   are validated by replay against the recorded log, but it has not driven a
   live server yet, so the live path is the one thing still unexecuted.
3. **The FILE-tier park arm**, with `--max-total-tokens` raised so chunk 4096
   still admits several concurrent sessions.
4. **C32's min-reduce risk**: check whether admission can run against the stale
   ceiling during the shrink gap.
5. `used:park:file` gauge reading 0 against files on disk — still open, but
   with one new data point from this shift that narrows it: my runs produced
   **0 park files AND gauge 0, i.e. they AGREED**. The gauge is therefore not
   unconditionally broken; the disagreement s45 saw (33 files, gauge 0) needs
   the file tier to actually engage. Whoever picks this up should reproduce
   with a run that reaches the file tier (see the pool-size note in 1b) rather
   than re-deriving it from s45's log.
