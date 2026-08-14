# TICKET 363 — stage-clock window VERDICT

**FAIL.** The window did not produce the measurement the ticket asks for, and
it could not have. Two independent blockers, both proven before any arm was
judged, and neither of them a property of the rig.

Shift R11+363, 2026-08-14. Branch `chore/ticket-363-window` rebased onto the
MERGE-R11 tip `a0db672d5f` (both lines) → `5ddb638aa7`; fixes on
`feat/363-fixes`. Evidence `/spinning/evidence-363/`. Window `363-stage-clock`,
claimed 08:13Z.

---

## 1. THE ANSWER TO THE QUESTION ASKED

> *does `act` deliver measurable ms/round improvement vs `observe` vs `off`,
> with spreads and floors?*

**Unanswerable on this line, and the reason is not "we ran out of window".**
`act` has nothing to actuate: the stage table contains exactly one stage — the
one the server booted on — so the stage clock can never propose a move. The
arms `off`, `observe` and `act` would have produced three traces of the same
configuration doing the same thing, and the ms/round difference between them
would have measured boot-to-boot noise and nothing else.

The runsheet's own §4.2 F2 names this outcome in advance: without an actuator
`act` is *"an expensive 'observe' under a misleading name"*. Running B1/B2/B3
would have produced exactly that, so they were not run. **Three boots were
spent, not seven.**

## 2. BLOCKER ONE — the axis has zero flip targets. Metal, then hermetic.

**On metal** (`/spinning/evidence-363/g3a/boot.log:693-708`), every rank:

```
REGIME-OBSERVE stage feed: prefill_heavy: the planner could not solve 'enc'
  (PlannerFeedUnavailable('no card probe on disk — the solver has no
   per-card rates and will not invent any'))
   ... same for decode_heavy ('dec') and kv_pressure ('maxkv')
REGIME-OBSERVE stage table: 1 stage(s), 1 reachable at runtime,
                            0 flip target(s), booted on 'booted'
REGIME-OBSERVE round 8: ... would flip to nothing -- already on stage 'booted'
```

So criterion **A1** (`stage_clock_proposals > 0` **and** `actuations > 0`) is
unreachable **by construction**, not by tuning.

**And the obvious repair makes it worse.** "No card probe on disk" invites
"then run the card probe". Tested hermetically on this line: feed
`build_stage_table` a planner-shaped candidate — `unmeasured=True` with
placeholder zeros, which is what the docstring says the planner returns — and
it does not build a two-stage table, it **raises**:

```
RegimeError: stage table refused (#578): the planner solved 1 stage(s) --
solved-enc -- but they carry no measurement. Each needs measured_gain_pct,
measured_band_pct and flip_cost_s, taken once at stage creation (an A-vs-A run
on the stage's own phase plus an instrumented flip), before #360 will let a
controller select it. The solver cannot predict any of the three. This is NOT
the old 'unfed' state: the feed is bound and working; what is missing is the
measurement pass, which is #584's slice.
```

`regime_runtime.py:1083-1091` catches that and returns **no table at all**
("nothing could be selected even in act mode"). So running the probe would
turn a 1-stage table into **zero** stages. The blocker is not the probe.

**The owner is named by the code: #584.** This is precisely the STOP condition
the runsheet's P4 states — *"If every non-reference stage is unmeasured, STOP —
that is #584's measurement pass, not this ticket."*

## 3. BLOCKER TWO — gate 3 cannot pass, and one of its two blockers never can

Gate 3 was run properly: two boots, identical flags
(`/spinning/evidence-363/GATE3_WORKLOAD_FLAGS.txt`), heavier than the
re-record exactly as P2 asks. Full report
`/spinning/evidence-363/bands-report-r11.{txt,json}`.

| | re-record (2026-08-01) | **this window** |
|---|---|---|
| active boundaries, arm A / B | 41 / 56 | **78 / 86** |
| `enter_prefill` | CLEARS 1.29x | **CLEARS 5.12x** (0.391 vs 0.0763) |
| `enter_decode` | CLEARS 1.14x | **CLEARS 3.14x** (0.474 vs 0.151) |
| `PRESTAGE_SINGLE_PROMPT_TOKENS` | — | **CLEARS 5.13x** (0.367 vs 0.0716) |
| `kv_ascend_mark` 0.85 | UNREACHED, peak 0.165 | **UNREACHED, peak 0.501** |
| `spread_veto_pct` 25.0 | UNREACHED, peak 0.68 % | **UNREACHED, peak 0.407 %** |

**P2 delivered what it promised.** The two thin verdicts are no longer thin —
1.29x → 5.12x and 1.14x → 3.14x — because the workload was made busier
(`--burst 16 --burst-tokens 6000`, the driver's own documented lever for held
tokens) rather than because anything was re-tuned.

**P1 could not be closed, and one half of it is not closable.**

* `kv_ascend_mark = 0.85` peaked at **0.501**, up 3x from 0.165. It is short
  for a structural reason: KV capacity on this boot is **320 640 tokens** and
  `--max-running-requests` is 16, so at ~10.8k prompt tokens per request the
  reachable ceiling is about half the pool. `bands.py` marks this one
  *"INHERITED from #287 … a failure is a finding for #287, not a licence to set
  a second mark"*.
* `spread_veto_pct = 25.0` peaked at **0.407 %** — **61x** below the constant.
  It is not reachable by any workload on a rig whose `auto-performance` planner
  exists to keep per-rank ms equal, and manufacturing a 25 % cross-rank skew
  would be manufacturing the measurement. **More to the point: this constant
  does not exist in the runtime.** `grep -rn 'spread_veto_pct' python/` returns
  **nothing**; the only act-mode interlock on that signal
  (`regime_runtime.py:602-613`) vetoes on `rank_ms_spread_pct is None` — the
  one-boundary-stale veto — and never compares a magnitude to 25.

  So gate 3 currently **blocks on a number that cannot change any runtime
  decision**. `UNREACHED` is in the blocking list (`bands.py:684-691`,
  `passed = not blocking`), and `bands.py` refused to write the evidence, which
  is correct behaviour for its own rules. This is a finding about the GATE, not
  about the rig, and it is not something a measurement window may re-tune: the
  report says in its own words that it *"does not re-tune"*.

**No bootstrap was written.** The runsheet sanctions one for gate 4's
circularity only. Gate 3 is not circular — it is blocked — and a passing
bootstrap next to a failing measurement is exactly what the runsheet's rule 3
forbids. `/spinning/evidence-363/` therefore contains no evidence file, and
`--regime-controller act` would still be refused at parse time, honestly.

## 4. WHAT THE WINDOW DID DELIVER

| ID | result |
|---|---|
| **A1** proposals/actuations | **FAIL — unreachable by construction** (§2) |
| **A2** flips ≤ 4 | vacuously 0 flips; not a result |
| **A3** ms/round compute vs wait | **NOT MEASURED** — the arms would have differed in nothing (§1) |
| **A4** corridor ≥ 1024 MiB | **PASS.** 3 boots, 100 ms NVML FREE sampling, **0 samples below 1024 on any card** |
| **A5** desyncs 0 | 0 desyncs across all three ranks on both gate-3 arms |

**Corridor (A4), the law that holds regardless of the verdict:**

| boot | samples/card | minima (gpu0/1/2) MiB | below 1024 |
|---|---|---|---|
| g3a | 5 929 | 1495 / 3773 / 2605 | **0** |
| g3b | 5 852 | 1495 / 3837 / 2605 | **0** |

Measured with `/spinning/evidence-363/corridor_report.py`, which was **proven
able to fail** by re-running it over the same series with a 900 MiB sample
planted — it caught it. Two earlier awk one-liners in this window printed a
reassuring minimum while comparing strings; that is why this is a script.

Requests: 132 sent, **0 failed**, on each arm.

## 5. TWO DEFECTS THE WINDOW EXPOSED, BOTH FIXED RED-FIRST

On `feat/363-fixes`, based on the rebased window branch.

**(a) `a4e2b50e95` — the preflight check whose only input nothing produces.**
C4 implements P4, the one STOP condition that would have ended this window at
the desk. It asks for `--stage-table`, *"a JSON dump of the boot's stage
table"*. **Nothing in the tree writes that file** — the only occurrence of the
string outside the preflight is its own argparse, and `server_args` has no dump
flag. So C4 could only ever SKIP; it duly skipped, the cards were claimed, and
the blocker was found on metal. The runtime had been printing it in plain text
the whole time. C4 now also reads `--boot-log`, is red against the real log,
and names the chain down to the missing card probe. Smoke **6/6 → 8/8**.

*The second can-fail arm earned its keep immediately:* `_STAGE_ROW_RE` was
anchored at line start, but real lines begin `[ts TPn] REGIME-OBSERVE   stage
…`, so the row parse matched nothing on either input. The zero-target branch
fires first, so the red case still looked correct — dead code certifying
nothing. Only the GREEN fixture exposed it.

**(b) `04fd724bb1` — every rank of a pure-TP boot staged its census through one
tmp file.** The transient census (P5's artefact — the one whose absence makes a
flip REFUSED instead of priced) **did not parse** on arm A: 433 bytes holding a
complete document plus a trailing `.0\n}`, the tail of a longer write left by a
shorter one. Kept at `/spinning/evidence-363/g3a/census/transient_pp0.json`.

`write()` stages through a tmp and `os.replace`s it — atomic, for ONE writer.
The staging path came from the output path alone, and `begin()` takes `pp_rank`
from `model_runner.pp_rank`, which under **pure TP is 0 on every rank**. Three
schedulers opened one `transient_pp0.json.tmp` with mode `"w"` at overlapping
times. Write-tmp-then-rename makes the PUBLISH atomic; it never made the
STAGING exclusive.

*Honest about what reproduced:* the COLLISION is pinned and was reliably red
pre-fix — **23 of 600** concurrent flushes LOST, `os.replace` finding the tmp
already renamed away by a peer. The exact interleaving that produced the
corrupt file did **not** reproduce hermetically (0 of 9 attempts across payload
sizes), so no test claims it does. The corrupt file is the metal evidence; the
shared staging path is the mechanism.

## 6. WHAT THE NEXT SHIFT SHOULD DO, IN ORDER

1. **#584's measurement pass is the gate on this whole ticket.** Until a stage
   carries `measured_gain_pct`, `measured_band_pct` and `flip_cost_s`, the
   stage table cannot hold a second stage and #363 has nothing to measure. No
   amount of window time substitutes for it.
2. **Decide what `spread_veto_pct = 25` is for** (§3). Either wire it to the
   interlock it is named after, or take it out of gate 3's blocking set. As it
   stands gate 3 can never pass on a balanced rig, which makes the act mode
   permanently unreachable for a reason unrelated to the controller.
3. `kv_ascend_mark` is a **#287** question, per `bands.py`'s own note. If it is
   to be reachable here at all, it needs an admission cap above 16 or a smaller
   pool — both of which change the configuration the bands describe.
4. The census writes **one file per `pp_rank`**, so a TP=3 group of three
   different cards publishes a single document describing whichever rank raced
   last. The corruption is fixed; **this** is not, and it is a design question
   for the census's owner rather than a bug to patch in a measurement window.

---

## 7. ADDENDUM — the staging pattern has a family, and two members already do it right

Recorded after the merge suite, docs-only (the convention R7–R11 use for a
handoff commit sitting on measured code).

`grep -rn '\.tmp"' python/sglang/srt/` finds **~19 sites** staging through
`path + ".tmp"`. Two of them already carry the fix this window had to make:

* `distributed/device_communicators/barlink_matrix.py:1745` —
  `p.with_suffix(p.suffix + f".{os.getpid()}.tmp")`, the exact per-process
  staging path;
* `registry/ledger.py:443` — `tempfile` with `dir=`, which is the same idea.

So the correct pattern was already present in this tree, twice, and the census
did not use it. The remaining ~16 sites are **unaudited**: the pattern is only
dangerous where two or more PROCESSES derive the same output path, which for
most planner artefacts is probably a single writer — but "probably" is what
this window just spent a corrupt file learning about. The discriminating
question per site is not "does it stage through a tmp" but **"can two
processes reach it with the same path"**, and under pure TP the degenerate
`pp_rank` is exactly what makes that happen without anyone intending it.

Two further notes on coverage, both instances of this shift's own MERGE-R11 §2:

* `test/registered/unit/planner/` is **not** a canonical arm, so the new
  regression test does not run in the nine-arm sweep. It was run explicitly:
  the planner arm is **2366P 123S** with 2 failures
  (`test_rejected_evidence_pins.py`), and those two are present on the
  untouched R11 tree as well.
* The transient census writes **one file per `pp_rank`**. Under TP=3 that is
  one document describing whichever of three DIFFERENT cards raced last. The
  corruption is fixed; this is not, and it is a design question for the
  census's owner.
