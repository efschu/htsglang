# C22-d / C22-e METAL PROOF — the live slot SET, and what it took to close it

Shift `656-liveslot-fix`, 2026-08-13 23:30Z → 2026-08-14 07:33Z. Branch
`feat/soak-fixes-656`, argv identical on every load-bearing flag to the soak
shift's boots and therefore to the R2 acceptance `boot_v3`
(`context_length=393216`, `max_running_requests=4`, `chunked_prefill_size=512`,
`pp_stage_ratio=[14,10,8]`, `rank_gpu_memory_mib=[31583,15750,18205]`,
`phase_flip_tp_vector='30,16,18'`, `SGLANG_PHASE_FLIP_SEAM_MARGIN_MIB=384`).
**Code is the only difference between these legs and the soak's.**

## 0. VERDICT

| claim | verdict |
|---|---|
| the red test green for the right reason | **PASS** — and the assertion moved onto the framed digest, §4 |
| 0 divergences over 45+ min with the ballot armed | **PASS** — leg 4, 46 min, 0 of 40 cutovers |
| at least one exercised recovery path showing the set levels | **PASS** — 6 corridor-bounded recoveries, 5 levellings, leg 4 |
| corridor law holds (0 samples < 1024 MiB) | **PASS on leg 4** — and **FAILED on legs 2 and 3, by my own regression**, §3 |
| the union REPAIR demonstrated on metal | **NO** — it has never fired, by design, §5 |

## 1. THE LEGS

| leg | code | window | cutovers | recoveries | levellings | divergences | abandons | gpu0 corridor min | <1024 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | C22-d | 23:34–23:39Z + 5.7 h idle | 10 | 2 | — | **0** | **0** | 1036 | 0 |
| 2 | C22-d | 05:25–05:44Z | 24 | 4 | — | **12 lines / 4 episodes** | **6** | 978 | **2** |
| 3 | C22-d + C22-e | 05:53–06:38Z | 56 | 9 | 8 | **0** | **0** | 990 | **4** |
| 4 | + corridor fix | 06:46–07:33Z (**46 min**) | 40 | 6 | 5 | **0** | **0** | **1042** | **0** |

Leg 4 corridor: **27907 samples/card at 100 ms, minima 1042 / 2135 / 1076 MiB,
0 below the 1024 law**; p1 1330 / 2925 / 1610, median 1930 / 3319 / 1820.
36 requests, 0 non-JSON, 0 failures. 0 KvReshardError, 0 tracebacks,
0 SIGQUIT, 0 CANNOT FUND, 0 purity stand-downs.

## 2. THE MECHANISM, AND THE COUNTERFACTUAL ON THE SAME RIG

**Leg 2 is the counterfactual and it is the reason C22-e exists.** With C22-d
alone, a corridor-bounded recovery left the group with two id spaces:

```
05:40 PP2  proposal: current=585390 floor=585903 (max_live=585390)
05:40 PP1  proposal: current=546236 floor=546749 (max_live=546236)
05:44 all  FLIP ABANDONED ... the group's union reaches row 585390 and the
           poorest rank has only 546236 rows BACKED
```

Four divergence episodes, six abandons, then the announced 9-refusal livelock —
`/health` 200, every request intact, serving never dropped. **Both refusals in
that trace are correct**: `collective_cap_target` may not withhold ids a peer's
request is using, and C22-d's union may not frame a row above a rank's backing
(that is `cudaErrorIllegalAddress`, which kills every rank rather than raising).
Two right answers and a flip that never happens again.

**Legs 3 and 4, with C22-e, on the same rig and argv.** Every corridor-bounded
recovery is immediately followed by a levelling on all three ranks:

```
05:57:18 PP1 recovered 581333 of 586642 (corridor-bounded)
05:57:18 PP1 backs 581333, poorest backs 581333, capped at 581333 (+0)
05:57:18 PP0 backs 586642, poorest backs 581333, capped at 581333 (-5309)
05:57:18 PP2 backs 586642, poorest backs 581333, capped at 581333 (-5309)
```

and leg 4 shows it is not rank-specific — PP2 is the poorest in one episode and
PP1 in another:

```
PP1 backs 577867, poorest backs 577115, capped at 577115 (-752)
PP2 backs 577115, poorest backs 577115, capped at 577115 (+0)
PP0 backs 585846, poorest backs 577115, capped at 577115 (-8731)
```

No divergence follows any of the 15 recoveries across legs 3 and 4.

## 3. THE CORRIDOR REGRESSION WAS MINE, AND LEGS 2 AND 3 BREACHED THE LAW

Stated first among the failures because it is the one a reader would otherwise
have to find in the CSVs.

C22-d made `KvRowCap.sort_free_lists` run **every seam round on every rank**
instead of only when the cap agreement moved a rank. It did
`setattr(alloc, name, torch.sort(pages).values)`, and on a **device** tensor
`torch.sort` allocates values AND indices — 4.7 MiB per copy at 586642 rows,
~14 MiB per call, up to ~34 MiB across the normalise and release paths, taken at
the seam, where this rig's corridor is tightest.

| boot | code | gpu0 min | samples < 1024 |
|---|---|---|---|
| soak boot 1 | before | 1028 | 0 of 33459 |
| soak boot 2 | before | 1084 | 0 of 125753 |
| **leg 2** | C22-d | **978** | **2** |
| **leg 3** | C22-d + C22-e | **990** | **4** |
| leg 4 | + fix | 1042 | 0 of 27907 |

`1024 - 990 = 34` — the transient to the MiB. Fixed in-shift: the sort runs on
the host and writes back with `copy_` into the same storage (zero device
allocation), with an equality guard that skips the write-back when the list is
already ascending; `release`, where the merge changes the tensor SIZE, does its
`cat` and `sort` on the host too — one device allocation instead of four.

**The general shape** (register row 84): a correctness fix was judged on what it
DECIDED and not on what it ALLOCATED, and was moved from a rare path to a hot
one in the same change. When a mechanism's frequency changes, its cost has to be
re-measured at the new frequency.

## 4. THE RED TEST, AND WHY ITS ASSERTION MOVED

The soak shift's assertion re-derived the three digests from the RAW per-rank
views and required them to collide. **No agreement step can satisfy that** — the
fixture builds three genuinely different sets and no fix rewrites
`live_slots_fn` retroactively. It measured the fixture, not the runtime.

The ballot votes on the set each rank **frames**, now recorded as
`last_framed_slots_digest`. Asserting THAT is the same sentence the old
docstring wrote — *"every rank must frame the SAME digest — i.e. the flip
completes"* — against the value that sentence was about, and it is **strictly
harder** than the cutover assertions beside it. Added with it: every rank's
framed set must be a superset of its own live rows, the property the alternative
repairs (rank-0 broadcast, intersection) violate.

Can-fail arms, because an instrument that cannot fail has certified nothing:
the agreement disarmed reproduces the metal signature on the identical fixture;
a union past the group backing refuses and names the numbers; an agreeing ballot
enters **no collective**; and the ballot still abandons a **wave-partition**
divergence — a term the union does not touch — with nothing stubbed, attributing
it correctly.

## 5. WHAT THIS DOES NOT PROVE

* **The union repair has never run on metal.** `live slot SET agreed by union`
  is 0 across all four legs. With C22-e the divergence no longer forms, so the
  repair is unreachable on the happy path *by design*. Its proof is the hermetic
  suite. This is a **prevention** result; read the 0 as "never needed", not as
  "demonstrated".
* **45 min is not an MTTF claim.** The soak shift's 1134 clean cutovers stand as
  the MTTF evidence; leg 4's 40 cutovers are a mechanism proof, not a rate.
* **Leg 1's 5.7 h are idle time**, not soak: this shift's process was killed by
  a service restart at 23:38Z and its corridor sampler, load driver and
  heartbeat died with it. The boot survived (own systemd scope) but served no
  load. Only the 4.5 active minutes of leg 1 are evidence.
* **Leg 4's boot stamp is one commit stale.** See `PATCHSTATE_leg4.txt`: the
  boot stamped `79eac07608` and ran the content of `1fd4d514e6`, proven by a
  clean tree and an empty `git diff HEAD -- python/` taken while the server was
  still importing those files. The honest sentence is "ran the content of
  `1fd4d514e6`", not "booted at `1fd4d514e6`".

## 6. FILES

`boot_c22d.log` (legs 1–2), `boot_e.log` (leg 3), `boot_f.log` (leg 4);
`corridor.leg1.csv`, `corridor.leg2.csv`, `corridor.leg3.csv`, `corridor.csv`
(leg 4); `EXTRACT_e.txt`, `EXTRACT_f.txt`; `load.*.log`, `deep.*`;
`PATCHSTATE_leg4.txt`.
