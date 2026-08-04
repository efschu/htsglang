# #541 — Does the local Qwen3.6-27B do hard agentic work better with thinking on?

A/B against the live INT8 boot on `127.0.0.1:30030`. Two arms differing in
exactly one byte sequence on the wire, the model id:

| arm | model id sent | what the router puts on the wire |
|---|---|---|
| A | `Qwen3.6-27B` | `thinking: {"type":"disabled"}` — today's default |
| B | `Qwen3.6-27B-think` | `thinking: {"type":"adaptive"}` — thinking on, model decides how much |

Everything else is identical: same prompt bytes, same tool set
(`Bash Read Grep Glob`), same empty cwd, same budgets, same 600 s cap.

Artefacts, prompts, raw transcripts, thinking blocks and instruments:
`/spinning/gpu-battery-results/2026-08-04_541_thinking_ab/`.

---

## 0. Headline

**The benchmark does not answer its own question, and the reason is measured,
not suspected.** Two findings dominate every arm difference:

1. **The A-vs-A noise floor is larger than any A-vs-B gap.** The same task, in
   the same arm, run back to back: 179 s with a complete answer, then 600 s
   with no answer at all.
2. **Arm B ran handicapped.** Without `preserve_thinking` the rendered prompt
   drops prior-turn reasoning, so the thinking arm re-prefills more per turn
   (#542). Measured here: 48.5 % prefix reuse in arm A against 40.3 % in arm B.
   Arm-B wall times are therefore a serving artefact, not a model signal.

What the battery *does* establish cleanly, because neither problem touches it:

3. **Adaptive thinking doses itself by task shape.** Thinking was 72 % and 56 %
   of generated tokens on the judgement-heavy audit, and 7 % and 9 % on the
   mechanical inventory sweeps. The model spends deliberation where the task is
   about judgement and almost none where it is about enumeration.
4. **Neither arm ever invented a defect.** Every audit error in both arms was
   false-negative (a stale citation waved through as correct). Not one
   fabricated finding, in any run.

The run was stopped after 9 of 16 runs on the user's instruction, to restart
serving with disk hicache and `preserve_thinking` on. The remaining reps were
not run.

**Phase 2 (§9) then ran 3 of a planned 24 runs against that new boot before
being wound down.** It settled exactly one thing, and it is the one that
mattered: with HiCache and `preserve_thinking` the reuse gap between arms
closes (66.9 % vs 62.9 %), so the arm difference is finally readable — and on
the one complete pair, thinking cost **2.5x wall time and 3.7x tokens for
identical, flawless quality**. One pair is not a result; it is the first clean
one.

---

## 1. Mechanism proof, before any measurement

The live boot carries `--reasoning-parser qwen3` (`/get_server_info`), which is
the precondition for a thinking arm. Three bodies, identical apart from the
field under test, straight at 30030:

| `thinking` sent | `stop_reason` | output tokens | blocks returned |
|---|---|---|---|
| field absent | `max_tokens` | 200 (budget) | `thinking` only, no text |
| `{"type":"disabled"}` | `end_turn` | 4 | `text` (`391`) |
| `{"type":"adaptive"}` | `end_turn` | 200 | `thinking` + `text` |

The first row matters beyond this benchmark: the live boot still predates the
#540 absent-means-disabled front fix, so the router's shim is **not** a no-op
here, and any client that omits the field silently gets thinking.

Inside the real harness (`claude -p --output-format stream-json`, identical
prompt): arm A produced **0** thinking tokens, arm B **63**. Thinking tokens
are counted with the served checkpoint's own tokenizer, not estimated from
characters.

---

## 2. Task material — real open work, not retro-eval

Every task is a genuinely open item whose output is usable as work product.
Ground truth is the code, verified by hand for every graded claim.

| id | task | source | ground truth |
|---|---|---|---|
| T1 | inventory `python/sglang/srt/registry/` (10 modules): purpose, public interface, importers | #538 building-block sweep, slice A | AST for the interface names, grep over `python`/`test`/`scripts` for importers |
| T2 | same for `python/sglang/srt/rigmon/` (16 modules) | #538 slice B, disjoint from T1 | as T1 |
| T3 | verify every line-numbered citation in FEATURE_CATALOG §16 | standing catalog-vs-code audit duty; §16 never checked before | 19 citations, each opened by hand: **15 CORRECT, 4 STALE** |
| T4 | establish from code what the translator boot warmup default does, and recommend | #533, open | flag/path/effect read by hand; 12 file:line claims checkable |

---

## 3. Results

`reuse` is `cached_tokens / prompt_tokens` over the run. It is `n/a` for the
first five runs: the cached counter was wired in mid-battery, after #542 made
clear it was needed.

| run | arm | rc | wall s | prompt tok | reuse | gen tok | thinking tok | turns | tools | valid |
|---|---|---|---|---|---|---|---|---|---|---|
| T3-A-r1 | A | 0 | **179** | 536 426 | n/a | 10 014 | 0 | 78 | 63 | yes |
| T3-A-r2 | A | 124 | **600 (DNF)** | 2 579 390 | n/a | 8 201 | 0 | 97 | 78 | yes |
| T3-B-r1 | B | 0 | 229 | 410 360 | n/a | 14 195 | **10 184 (72 %)** | 48 | 32 | yes |
| T3-B-r2 | B | 0 | 248 | 645 890 | n/a | 13 217 | **7 378 (56 %)** | 79 | 54 | yes |
| T1-A-r1 | A | 0 | 193 | 741 055 | n/a | 4 489 | 0 | 45 | 38 | yes |
| T1-B-r1 | B | 0 | 134 | 361 737 | n/a | 3 132 | 222 (7 %) | 31 | 23 | yes |
| T2-A-r1 | A | 0 | 578 | 1 323 330 | **48.5 %** | 8 174 | 0 | 81 | 69 | yes |
| T2-B-r1 | B | 124 | **600 (DNF)** | 1 181 454 | **40.3 %** | 3 756 | 356 (9 %) | 61 | 49 | yes |
| T4-A-r1 | A | 0 | 253 | — | — | 2 308 | 0 | 24 | 17 | yes |

`valid=yes` on every row means the serving identity (PID + process start time)
was unchanged across the run. #532 lost four runs to unnoticed restarts; this
battery lost none.

### Noise floor — the number that governs the rest

| run | arm | task | wall | prompt tokens | outcome |
|---|---|---|---|---|---|
| T3-A-r1 | A | T3 | 179 s | 536 426 | complete, 17/19 correct |
| T3-A-r2 | A | T3 | 600 s | 2 579 390 | **no answer at all** |

Same task, same arm, back to back, nothing between them. The prompt-token cost
differs by 4.8x and the outcome by everything. This reproduces #532's
`FINDING_cache_reuse` bimodality: the run that lost its prefix re-prefills its
whole grown context every turn and spends the budget doing it.

Because this pair exists, **no single-run A-vs-B difference in this battery is
interpretable**. That is the primary result.

---

## 4. Grading, per task

### T3 — catalog §16 audit (19 citations; truth 15 CORRECT / 4 STALE)

| run | verdicts correct | misses |
|---|---|---|
| T3-A-r1 | 17/19 | `uneven_perf.py:2617`, `metrics_reporter.py:1018` waved through as CORRECT |
| T3-A-r2 | — | DNF |
| T3-B-r1 | **18/19** | only `metrics_reporter.py:1018` |
| T3-B-r2 | 15/19 | all four STALE waved through |

All four runs listed all 19 citations, none invented one. **Every error in both
arms is a false negative.** The arm-B spread (18/19 then 15/19) is as wide as
the arm difference, which is the noise floor again.

The finding itself is real audit output. Arm B's best run, verbatim on the one
citation arm A missed:

> `uneven_perf.py:2617` | STALE | Line 2617 is a comment (`resolution is a pure
> function of the CLI.`). The actual `envs.SGLANG_MEASURED_KV_BUDGET.get()`
> check is at **line 2642**

Independently confirmed: the consumption is at `uneven_perf.py:2642`.

### T1 — registry sweep (10 modules)

| run | modules | invented | interface names | importer claims |
|---|---|---|---|---|
| T1-A-r1 | 10/10 | 0 | 94 verified, 0 wrong | 33 verified, 0 wrong |
| T1-B-r1 | 10/10 | 0 | 92 verified, 0 wrong | 32 verified, 0 wrong |

Both arms **factually flawless**. A quality tie; arm B was faster and cheaper.

A grader caveat worth recording, because it nearly produced a false result: the
first version of `grade_sweep.py` flagged 16 importer claims as wrong. Every
one was the grader's bug — it excluded slice-internal importers and never
searched `scripts/`, and its AST pass missed annotated module constants
(`LADDER: Mapping[...] = {...}`). The model was right in all 16 cases. Checked
by opening `arbiter.py:27`, `__main__.py:5`, `scripts/registry/m1_card_window.py:37`
by hand.

### T2 — rigmon sweep (16 modules)

| run | modules | interface names | importer claims |
|---|---|---|---|
| T2-A-r1 | 16/16, 0 invented | 159 verified, 0 wrong | 51/53 (2 cosmetic) |
| T2-B-r1 | DNF — empty answer file | — | — |

The two arm-A deductions are presentation, not fact: a stray
`slice-internal only` phrase appended *after* correct importer paths, and one
module listing itself. Arm A's strongest single result of the battery; arm B
lost the run entirely.

### T4 — #533 warmup analysis

Arm A, 253 s. **All 12 file:line claims opened and confirmed exact.** Verbatim,
the decisive part:

> RECOMMENDATION: FLIP DEFAULT OFF. The code's own measurements — committed
> into the docstring at `server.py:202-213` — show the warmup providing no
> benefit on this rig: the 15 s cold start it was written to fix did not
> reproduce, and the two boots that differ only in this flag bracket the
> no-warmup control rather than beating it.

It also correctly scoped the effect — "Only the TTS backend is exercised — ASR,
MT, and the speaker embedder are not touched" — which is right: the warmup
calls `self.stack.tts.synthesize` at `server.py:262` and nothing else.

Arm B for T4 was never started: the battery driver was stopped after T4-A-r1
returned. T4 therefore has one arm only.

---

## 5. What thinking costs

| run | task shape | thinking tokens | share of generated |
|---|---|---|---|
| T3-B-r1 | judgement audit | 10 184 | 72 % |
| T3-B-r2 | judgement audit | 7 378 | 56 % |
| T2-B-r1 | mechanical inventory | 356 | 9 % |
| T1-B-r1 | mechanical inventory | 222 | 7 % |

This is the cleanest signal in the battery, and neither the noise floor nor the
caching handicap touches it: **adaptive thinking self-doses by task shape**, an
order of magnitude more on judgement than on enumeration. It also means the
token cost of arm B is not a flat surcharge — on the sweep tasks it is nearly
free, on the audit it roughly doubles generation.

---

## 6. The preserve_thinking handicap

#542 established that without `preserve_thinking` the rendered prompt strips
prior-turn reasoning, so the prefix diverges from what was generated and reuse
drops. The two runs where the cached counter was already wired up:

| run | arm | prompt tokens | cached | reuse |
|---|---|---|---|---|
| T2-A-r1 | A | 1 323 330 | 641 856 | **48.5 %** |
| T2-B-r1 | B | 1 181 454 | 476 160 | **40.3 %** |

One pair is not a measurement of the effect size, but it is the right sign and
enough to disqualify arm-B wall times as a model-quality signal.

Whether `chat_template_kwargs.preserve_thinking` can be passed **per request**
through the Anthropic front is **not established**. The front returns 200 for a
body carrying it, but 200 is not evidence: the front builds its own
`ChatCompletionRequest`, and `anthropic/serving.py:705` states the OpenAI front
is the one that reads that knob, so the field may simply be ignored. A
discriminating two-turn reuse probe is written
(`probe_preserve_thinking.py`) but was **not run** — it needs an idle server and
the battery held it. It remains the only way to know whether the per-request
route works at all, independently of the coming server default.

---

## 7. Honest limitations

* **9 of 16 runs.** Stopped on instruction to restart serving. T4 has one arm
  only; T1 and T2 have one rep per arm.
* **The noise floor swamps the comparison.** With an A-vs-A span of
  "solved in 179 s" to "DNF at 600 s", 2 reps per cell cannot separate arms.
  Three or more reps per cell is the minimum for a conclusion.
* **Arm B ran without `preserve_thinking`** and is therefore penalised on wall
  time and prefill by a serving artefact. Its *quality* numbers are unaffected;
  its *speed* numbers must not be quoted.
* **The judge is the operator agent.** Every graded claim was re-verified
  against the tree by hand, and one grader bug that would have produced 16 false
  failures was caught — but the grading was not double-blinded.
* **Cached-token capture is missing for the first five runs.** It was added
  mid-battery, atomically, after #542 arrived.
* **The harness is not the agent-definition path.** Arms were run as
  `claude -p --model <id>` so the entire loop is local. Running them as
  subagents would have put a Claude parent in the loop that could solve the task
  itself, which would destroy the measurement. The
  `local-model-think` agent definition exists and uses the same alias.

---

## 8. Work product adopted

The point of using real tasks: the better arm's output is kept, not discarded.

| task | adopted from | status |
|---|---|---|
| T3 §16 audit | **arm B r1** (18/19) plus the operator's own verification | 4 stale citations confirmed: `uneven_perf.py:2617`→2642, `metrics_reporter.py:1018`→1020, `:1020`→1022, `:962`→964. A real defect report against FEATURE_CATALOG §16, to be fixed on the line. |
| T1 registry inventory | **either arm** (both flawless); arm A r1 kept as the fuller table | usable as the #538 slice-A inventory as-is |
| T2 rigmon inventory | **arm A r1** (arm B DNF) | usable as the #538 slice-B inventory after removing one stray `slice-internal only` phrase and one self-listing |
| T4 #533 warmup | **arm A r1** (arm B incomplete) | analysis stands: flag at `launch.py:112`, only the TTS stage is exercised, recommendation FLIP DEFAULT OFF, all 12 citations exact |

All four are stored with their transcripts under
`/spinning/gpu-battery-results/2026-08-04_541_thinking_ab/runs/`.

---

## 9. Phase 2 — the HiCache + preserve_thinking boot

Serving was restarted between the two phases. Phase 2 ran against a materially
different boot and is **not** poolable with phase 1.

| | phase 1 | phase 2 |
|---|---|---|
| serving identity | pid 1236 | **pid/pgid 115747**, start-jiffies 230066160 |
| disk HiCache | none | 100 GB, `file` backend, `/spinning/hicache` |
| `preserve_thinking` | off | **server default** (`chat_template_default_kwargs`) |
| absent `thinking` field | produced a thinking block (pre-#540) | **== disabled** (#540 live) |

All four config facts were live-verified from `/get_server_info` before launch,
not taken on report. The thinking mechanism was re-proven on the new boot,
because #540 changes what an absent field means:

| `thinking` sent | stop_reason | out tok | blocks |
|---|---|---|---|
| absent | `end_turn` | 4 | `text` |
| `disabled` | `end_turn` | 4 | `text` |
| `adaptive` | `max_tokens` | 150 | `thinking` |

That is the inverse of phase 1's first row: the router's shim is now a genuine
no-op, and arm A is unaffected either way. Both router aliases were re-verified
end to end against the new boot before any run.

Phase 2 was ordered rep-round-robin (every rep sweeps all four tasks, A and B
adjacent) precisely because phase 1's task-major order lost T4's B arm when it
was stopped. It was stopped after 3 of 24 runs on a wind-down instruction.

| run | arm | rc | wall s | prompt | cached | reuse | gen | thinking | turns | tools |
|---|---|---|---|---|---|---|---|---|---|---|
| P2-T1-A-r1 | A | 0 | **197** | 604 542 | 404 662 | **66.9 %** | 3 951 | 0 | 40 | 32 |
| P2-T1-B-r1 | B | 0 | **485** | 1 124 801 | 707 486 | **62.9 %** | 14 675 | 4 758 (32 %) | 70 | 55 |
| P2-T2-A-r1 | A | 124 | 600 (DNF) | 994 863 | 464 379 | 46.7 % | 4 941 | 0 | 59 | 52 |

### The one thing phase 2 settled

**The caching handicap is gone, and the thinking cost is real.** Reuse is now
66.9 % against 62.9 % — a 4-point gap, against phase 1's 8-point gap at a much
lower level (48.5 % vs 40.3 %). With reuse essentially matched, the arm gap on
the T1 pair is no longer attributable to the serving path:

* arm B took **2.5x the wall time** (485 s vs 197 s)
* arm B spent **3.7x the generated tokens** (14 675 vs 3 951)
* arm B took **1.75x the turns** (70 vs 40)
* **for identical, flawless quality** — both arms 10/10 modules, 94 verified
  interface names, zero wrong claims

Only 4 758 of arm B's 14 675 generated tokens are thinking. The other ~9 900
are ordinary output: thinking did not merely add a reasoning surcharge, it made
the agent take more turns and write more, with nothing to show for it on this
task class.

Arm B's behaviour on T1 also changed sharply between boots — phase 1: 134 s /
3 132 generated / 222 thinking tokens; phase 2: 485 s / 14 675 / 4 758. The
plausible mechanism is `preserve_thinking` itself: prior-turn reasoning now
stays in context, which invites more reasoning in each following turn, a
compounding effect. **This is a hypothesis, not a finding** — it rests on one
pair per phase.

### What phase 2 did not settle

* **n = 1 per arm.** Three runs of a planned 24. The phase-1 noise floor
  (179 s vs 600 s DNF, same task, same arm) has not been re-measured on this
  boot and there is no reason to assume it shrank.
* **T2 flipped without any arm involved.** The rigmon sweep succeeded at 578 s
  in phase 1 and DNF'd at 600 s in phase 2, both in arm A. That task sits on
  the budget boundary, and it is further noise-floor evidence, not a
  boot-quality statement.
* **Co-tenancy was live.** Sampling during phase 2 showed foreign traffic on
  the server (`running=2.0`, `queue=1.0`). The runner gates on an idle server
  at run START only, so a co-tenant request mid-run lands in that run's
  `/metrics` delta. Phase 1 had the same exposure with fewer active agents.
* **The inverted `preserve_thinking` probe result** reported by the coordinator
  (true = 6.4 % vs false = 77.2 % reuse on a two-turn probe) was under
  investigation and is unresolved. It does not touch quality grading, and the
  per-run reuse figures above are what would let it be separated later.

### Recorded operational fault

`P2-T2-A-r1` hit its cap and its `run_arm.sh` produced no status file for
roughly two minutes afterwards, while the process itself had already exited.
The status file did appear later and the run was recovered intact from its
860 KB stream transcript. Nothing was lost, but a run is not safe to declare
missing on the absence of its status file alone — the transcript is the
authority.

---

## 10. Follow-up

1. **Finish the phase-2 battery** (3 of 24 runs done): 4 tasks x 2 arms x 3
   reps on the pgid-115747 boot. Prompts, runner, graders and ground truth are
   reusable unchanged; `battery2.sh` carries the rep-round-robin order.
   Re-measure the A-vs-A noise floor on this boot first — the phase-1 figure
   does not transfer.
2. **Run `probe_preserve_thinking.py`** on an idle server to settle whether the
   per-request route through the Anthropic front works, independently of the
   server default.
3. **Fix FEATURE_CATALOG §16**: the four stale citations above.
4. **#533**: the analysis is done and says FLIP DEFAULT OFF. That is a decision
   for the owner, not something this benchmark should apply.
