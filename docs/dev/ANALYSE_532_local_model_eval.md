# #532 — Is the local Qwen3.6-27B-INT8 usable as a subagent worker?

Evaluation of the local serving boot on `127.0.0.1:30030` as a delegated
worker, run before it is given any real role. Two arms: a retro-benchmark
against solved bugs whose root cause is known (Arm A), and the mechanics
workload it would actually be given (Arm B).

Working directory with all artefacts, prompts, raw answers and instruments:
`/spinning/gpu-battery-results/2026-08-04_532_local_model_eval/`.

---

## 0. Vehicle and readiness

Model under test: `Qwen3.6-27B-INT8-W8A8`, TP=3 over 5090 + 2× 3080, uneven TP
(`--rank-tp-ratio auto-performance`, `--rank-perf-tune phase-decode`,
`--rank-auto-reserve-mib 13000,3800,3800`), NEXTN spec (3 steps),
`--kv-cache-dtype fp8_e4m3`, `--reasoning-parser qwen3
--tool-call-parser qwen3_coder`, `--enable-metrics`.

Driver: `/spinning/wt-530-serving/scripts/dev/local_model_agent.sh` — a
separate `claude -p` process bound to that endpoint, tools limited to
`Bash Read Grep`.

The switch-complete gate was both parts, both observed before any run:
`GET /v1/models` reporting the INT8 checkpoint, **and** a real tool round-trip
(the model called Bash in a worktree and returned the first line of the
output). The earlier FP8 boot could not do the second — it had no tool-call
parser — so the round-trip, not the model id, is the actual readiness probe.

---

## 1. Headline

| | |
|---|---|
| **Deep root-cause search in this repository** | **not usable.** 8 of 10 attempted bug runs consumed the entire 600 s budget without producing any answer; the one clean success (b316) was exact on the root and wrong on the fix. |
| **Bounded mechanics with the material supplied** | **usable.** All four Arm-B probes finished in 44-142 s; two are directly usable, two need review. |
| **Verification with tools against a named file** | **the strongest result.** The catalog-vs-code audit was 11/11 correct on independently re-checked claims and produced a real defect report. |

The dividing line is not difficulty of reasoning — the model reasons well when
it has the material. It is **search**: every failure was a run that spent its
whole budget navigating a 57 MB source tree, and every success was a task
where the relevant text was either handed over in the prompt or reachable in a
handful of greps.

---

## 2. Arm A — retro-bug benchmark

### 2.1 Design, and one hole found in it

Eleven solved bugs, each replayed against a throwaway worktree at the commit
BEFORE its fix. Full protocol in `METHOD.md`; the short form:

* The **falsifier is never shown.** That was the first design and it had to be
  abandoned on evidence: in this repository the post-hoc falsifier is
  answer-bearing by construction. Four of them do not even collect against the
  pre-fix tree because they import the helper the fix introduced
  (`ImportError: cannot import name 'moe_uneven_tp_units'` — the missing name
  *is* the answer), and the ones that do collect fail under test names like
  `test_totals_follow_the_cuda_ordinal_not_the_nvml_index`. Both raw captures
  are kept in `symptoms/*.raw.txt` as the evidence for that decision.
* The prompt therefore carries the **original symptom only** — a server
  traceback, a runtime error string, or a red test run. Per-bug provenance is
  in `answers.md`.
* Leakage check per tree: the symbol the fix introduces must be absent (0/6
  checked), and no `docs/` file may describe the bug (one hit, b289, which
  restates the SYMPTOM only).

**The hole, found by the model itself.** `git worktree add <pre-fix> --detach`
shares the repository object store, so every later commit — including the fix
— stays reachable with `git log --all` / `git show`. b525's answer says
outright: *"The fix commit `7fca12de6b` does exactly this"*. All answers were
scanned; **b525 is the only one that used it**. No other answer cites a commit,
and b316's answer proposes a fix that is *not* the one that shipped, which is
positive evidence it did not look. For the remaining runs each worktree's
`.git` file was moved aside (`git log` then reports "not a git repository")
and b525 was re-run against the sanitized tree.

**The re-run settles it.** b525r is the same bug, the same prompt, the same
model and the same 600 s budget, differing only in that `git log` / `git show`
can no longer reach the fix:

| run | history reachable | wall | tokens | outcome |
|-----|-------------------|------|--------|---------|
| b525 | yes | 184 s | 975 047 | exact root, exact mechanism, correct fix |
| b525r | no | 600 s | 2 160 070 | no answer at all |

So b525's success was reading its own fix commit, not diagnosis. It is struck
from the capability evidence; b525r replaces it. This also means the honest
tally for Arm A is **1 real answer in 10 attempts** (b316), plus one exact
mechanism with the wrong fix site (b527).

### 2.2 Results

Token figures are the server-side `/metrics` delta over the run and are
dominated by prefill, not generation: the harness system prompt alone costs
~21.9 k prompt tokens on turn 1, and every follow-up turn re-sends the grown
context.

| bug | class | root hit | mechanism | fix plausible | wall s | total tokens | generated |
|-----|-------|----------|-----------|---------------|--------|--------------|-----------|
| b289 | alignment | — | — | — | 600 (timeout) | 1 758 011 | 12 505 |
| b300 | alignment | — | — | — | 600 (timeout) | 1 938 903 | 9 691 |
| **b316** | alignment | **exact** (`gptq.py:480` vs hunk 477) | half | no (primary) | **468** | 2 191 575 | 13 819 |
| b383 | alignment | — | — | — | 600 (timeout) | 1 911 054 | 23 165 |
| b392 | device order | — | — | — | 600 (timeout) | 3 857 067 | 15 405 |
| b406 | device order | — | — | — | 600 (timeout) | 1 875 218 | 6 992 |
| b525 | test honesty | exact — **but contaminated, see below** | exact | yes | 184 | 975 047 | — |
| b525r | test honesty | — (same bug, history removed) | — | — | 600 (timeout) | 2 160 070 | — |
| **b527** | test honesty | miss (by file) | **exact** | no | **209** | **224 364** | — |
| b501 | silent | invalid — server restarted mid-run | | | 600 | (counter reset) | — |
| b446 | silent | invalid — `API Error: 500` from the server mid-run | | | 371 | (counter reset) | — |

**The silent-wrongness stratum did not produce a usable measurement.** b501
and b446 both ran and both were destroyed by server restarts (a negative token
delta from a counter reset, and an `API Error: 500` at 371 s respectively);
b443 was still waiting on the readiness gate. The serving process was replaced
three times during this evaluation — at 22:09, at ~23:52 and again at ~00:05 —
each time with the same INT8 checkpoint. The readiness gate added after the
first outage caught the later two before they could produce a fake result,
which is the behaviour wanted, but it means this class has **no** evidence in
Arm A and the tier list below must not be read as covering it.

### 2.3 What the two real answers show

**b316 — exact root, wrong lever.** It named
`GPTQMarlinConfig.get_quant_method()` at `gptq.py:480`, three lines from the
hunk the fix changed, and correctly explained why AWQ and FP8 siblings do not
fail. It then attributed the cause to a missing `modules_to_not_convert`
lookup rather than to the missing *shape* guard — and its primary fix would
not repair the reported checkpoint, because the prompt states that checkpoint
carries no ignore list, which is exactly why the real fix judges unsharded
geometry instead. Its own "secondary defense" is on target and is an
independent finding about the tree: `check_marlin_supports_shape()`'s return
value is **discarded** at `gptq_kernels.py:139`.

**b527 — exact mechanism, wrong place to fix.** It reconstructed the entire
causal chain unaided (top-level `import triton`, `@triton.jit` committing at
decoration time, `sys.modules` caching, the sibling test importing first
without the env var). It then proposed restructuring the production kernel
module — dropping the top-level import, decorating lazily through a
`globals()` rewrite — so that a test's environment manipulation would work.
The shipped fix is one `importlib.reload()` in the test. The diagnosis was
right; the engineering judgement about **where** to fix was wrong, and in the
direction that costs the most.

That pattern is consistent across both: **the model's analysis is better than
its judgement about proportionality.** Neither answer could be merged as
given, and neither was worthless.

---

## 3. Arm B — mechanics probes

All four inside budget. Detail and the verification of every claim in
`GRADING_ARMB.md`.

| probe | wall s | total tokens | generated | verdict |
|-------|--------|--------------|-----------|---------|
| B1 translate a real German docstring to English | 99 | 396 388 | 1 907 | usable after rework |
| B2 gate log → structured verdict | 44 | 383 415 | 1 101 | **directly usable** |
| B3 fixture boilerplate after a template | 58 | 177 948 | 1 333 | usable after rework |
| B4 catalog-vs-code consistency check, with tools | 142 | 472 823 | 7 328 | **directly usable** |

**B1** preserved every literal, `file.py:NNN` reference and RST marker, and
kept the heading underlines valid — the hard part. It truncated the last
bullet, added a preamble and a fence against instruction, and mistranslated
"nicht angenommen" (not *assumed*) as "not adopted".

**B2** got every field right and, unprompted, flagged four real defects
visible in the log: unresolved speaker identity on turn 2, turn 2 still in
`translating` state, the ASR rendering "sechs" as "Sex", and a DE-tagged turn
whose source text starts in Spanish. Only defect: a ``` fence around the JSON.

**B3** produced the right structure and assertion shape, and its three
file:line citations were re-checked and are **correct**. Two defects matter:
it labelled the pin `#505-C-04`, an id that exists but belongs to a *different*
posten (`SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB`), and it invented an unsupported
generality about H100 CUDA-context size despite being told not to. A
plausible-looking wrong cross-reference is the most expensive error class this
repo has, so this output needs a review that specifically hunts for it.

**B4** is the strongest result of the evaluation. Asked to list the flags
catalog §6 names, verify them in `server_args.py`, and check the section's six
`server_args.py:NNNN` citations against the tree, it returned five flag
declarations with exact line numbers and reported **all six citations stale**,
with the correct current location for each. **Every one of those eleven claims
was re-verified by hand and is correct** (see `GRADING_ARMB.md` for the table).
It also read the shape of the drift correctly — inconsistent offsets of ~20 to
~80 lines, therefore accumulated file movement rather than one shifted block —
and separated "the prose is right, the anchors are stale". This is a real
defect report against `FEATURE_CATALOG.md` §6 that should be fixed on the line.

---

## 4. Capability tiers, with the evidence for each

### Tier 1 — up to it, delegate with normal review
* **Structured extraction from a supplied log or artefact into a fixed
  schema, including judgement about what looks wrong.** Evidence: B2, every
  field correct, four unprompted real defects, 44 s.
* **Verification of documentation against code with tool access, when the
  target file is named.** Evidence: B4, 11/11 independently re-verified
  claims, and it used grep instead of answering from memory as instructed.
* **Technical DE→EN translation of prose with heavy markup.** Evidence: B1,
  markup fully preserved; needs a completeness check because it truncated.

### Tier 2 — borderline, delegate only with a targeted review
* **Boilerplate after a template.** Evidence: B3 — correct shape and correct
  citations, but a fabricated-by-misattribution ticket id and an invented
  hardware generality. Usable, but the reviewer must be told to hunt
  cross-references, which eats much of the saving.
* **Diagnosis when the relevant file is already named in the task.** Evidence:
  b527's mechanism was exactly right; b316 landed within three lines of the
  real root. Both then chose the wrong fix. Delegate the *analysis*, never the
  *decision about what to change*.

### Tier 3 — not up to it, do not delegate
* **Open-ended root-cause search in this repository.** Evidence: 8 of 10 bug
  runs (b289, b300, b383, b392, b406, b525r and the discarded first attempts)
  consumed the full 600 s and returned the wrapper's "Execution error" — no
  answer at all, at 1.8-3.9 M tokens per failed attempt. The b525/b525r pair
  is the cleanest single datum in the whole evaluation: with the answer
  reachable in git history the model solved it in 184 s; with the history
  removed it did not solve the same bug in 600 s.
* **Deciding the scope of a fix.** Evidence: b527 proposed rewriting a
  production kernel module's import structure to satisfy a test; b316's
  primary fix would not have repaired the reported checkpoint.

### Cost per class (measured, not estimated)
| class | typical wall | typical total tokens |
|-------|--------------|----------------------|
| bounded mechanics (B1-B4) | 44-142 s | 0.18-0.47 M |
| bug diagnosis that succeeds | 184-468 s | 0.22-2.19 M |
| bug diagnosis that fails | 600 s (budget) | 1.8-3.9 M |

Every task also pays a fixed cold start of ~22 k prompt tokens for the harness
system prompt before it reads anything.

### Ladder additions from #541 (2026-08-04, real open tasks, both thinking arms)

New rows measured on real open work rather than retro-bugs, and measured in
BOTH thinking modes (`thinking: disabled` vs `adaptive`). Full write-up:
`BENCH_541_thinking_ab.md`. The graded claims were re-verified against the tree
by hand.

| task class | evidence | tier | note |
|---|---|---|---|
| **Module inventory of a named directory slice** (purpose + public interface + importers, tool-verified) | #538 slice `registry/` 10 modules: BOTH arms 0 wrong out of 94/92 interface names and 33/32 importer claims, 193 s / 134 s. Slice `rigmon/` 16 modules, thinking-off arm: 159 names and 51/53 importer claims correct, 578 s | **Tier 1** | the strongest class yet measured, stronger than B4. It is B4's shape — named target, verify with grep — scaled from one section to a whole directory |
| **Catalog-vs-code citation audit, section not seen before** | #541 T3, FEATURE_CATALOG §16, 19 citations: 17/19, 18/19, 15/19 verdicts correct across three completed runs; one run DNF | **Tier 1**, with the error direction named | every error in every run was FALSE-NEGATIVE — a stale citation waved through. Zero fabricated defects. So the output is safe to act on positively and must not be trusted as an all-clear |
| **Bounded code analysis of an open question with a named subsystem** (find the flag, trace the path, state the effect, recommend) | #541 T4, #533 warmup, thinking-off arm: all 12 file:line claims exact, correct scoping, recommendation grounded in the code's own recorded measurement, 253 s | **Tier 1** — an upgrade | #532 put "diagnosis when the file is named" in Tier 2 because b316/b527 chose the wrong fix. When the task asks for ANALYSIS AND A RECOMMENDATION rather than a patch, that failure mode does not appear. Delegate the analysis; the #532 rule "never the decision about what to change" still holds for edits |

**Thinking on vs off does not move the tier for any class measured.** Quality
was a tie on the inventory sweeps (both arms flawless), a wash on the audit
(the arm-B spread 18/19→15/19 is as wide as the arm gap), and the thinking arm
lost one run outright to a timeout. What thinking demonstrably changes is cost,
and it self-doses: 72 % / 56 % of generated tokens on the judgement audit
against 7 % / 9 % on the mechanical sweeps.

**Two caveats bound all of the above, both measured:** the A-vs-A noise floor
in this battery spanned "solved in 179 s" to "DNF at 600 s" on the same task in
the same arm, and the thinking arm ran without `preserve_thinking`, which costs
it prefix reuse (48.5 % vs 40.3 % on the one pair with the counter wired up).
Arm-B wall times are therefore not model signal. Quality numbers are unaffected
by both.

---

## 5. Radix reuse in the agentic loop

Full write-up with the discriminating test in `FINDING_cache_reuse.md`.

The Anthropic-format response htsglang returns at `POST /v1/messages` carries
**no** `cache_read_input_tokens` — measured 0 on every one of ~150 logged
turns, including turns whose prefix was demonstrably reused. Per-turn reuse is
therefore not readable from the API and had to be taken from
`sglang:cached_tokens_total` / `sglang:prompt_tokens_total`, sampled at 1 Hz
and attributed between consecutive turn ends.

Over 72 attributed follow-up turns: **median 98.7 %**, mean 87.4 %, 14 turns
below 90 %, worst 0.0 % (a full re-prefill of a 167 030-token context).
Cold start per run: 22 618 tokens on b383, 21 856 on a one-word smoke run.

The distribution is bimodal, not noisy. In b392 the collapse recurs roughly
every fifth turn and lands on a *constant* cached figure —
24 256 / 24 320 / 24 384 / 24 448 / 24 576 — while the context grows from
60 878 to 82 201 tokens.

**It is not template or system-prompt variance.** Two consecutive pairs from
the same run were byte-compared on captured request bodies:

| pair | system+tools identical | first diverging block | identical prefix | cause of divergence | reuse |
|------|---|---|---|---|---|
| 44→45 | yes | 158 of 159 | 170 625 chars | `cache_control` marker moved | **99.7 %** |
| 45→46 | yes | 161 of 162 | 171 032 chars | `cache_control` marker moved | **36.4 %** |

Same shape in every respect the request controls, and nothing changes at token
0 — yet 99.7 % in one case and 36.4 % in the other. The cause is therefore not
in the request: it is eviction of the prefix tail. Claude Code does move its
ephemeral `cache_control` breakpoint every turn, which invalidates a real
suffix, but it sits near the end and the 99.7 % row proves that cost is small.

The expensive event is the eviction, and it is the direct reason the Arm-A
runs hit the wall clock: b383 turn 12 re-prefilled 167 030 tokens in a single
144 s turn.

---

## 6. Operating the model as a worker pool

### 6.1 The co-tenancy constraint is real and was measured twice

**Incident (reported by the coordinator):** a ~53 k-token evaluation prefill
starved the translator's MT hop completely between 21:48 and 21:50.

**The reverse, measured here:** once the runner was taught to yield to the
translator, an active conversation blocked the evaluation pool for **14
minutes straight** (b392 attempts 1-14, and again for b406), during which the
pool did zero work. Translator session state over that window went from 0 to 7
turns, so the block was genuine load, not a stale session.

So the two workloads currently exclude each other in both directions. Until
the server's HTTP path is decoupled, that is a hard constraint on any pool
plan, not a tuning parameter.

The gate that was implemented and used for every run after the incident:
yield while any translator session has `idle_s < 180` or a queued turn; never
start while `num_running_reqs > 0`; never start while the server is not
`/health`-200 with a resolvable model. The predicate was spread-checked on
four known-different inputs before use.

### 6.2 Slot recommendation — 4 parallel sessions

The user's plan is a permanent pool of up to 4 parallel tasks. The measured
constraint is not the number of sessions but the **sum of context in flight**:
a single agentic bug run reached 82-171 k tokens per turn and evicted its own
prefix. Four of those at once would thrash the radix cache continuously and
starve the translator whenever it is used.

Recommended mixture, by context weight rather than by task count:

| slot | assign | why |
|------|--------|-----|
| 1 | **one** large-context task (repo-reading, catalog audit, diagnosis-with-named-file) | only one may hold a >50 k-token context at a time; this is the eviction-limited slot |
| 2 | short mechanics (log→verdict, translation, extraction) | 0.18-0.47 M tokens, 44-142 s, no repository crawl |
| 3 | short mechanics | as above |
| 4 | **reserved / drain** | held free as the buffer the translator needs; used only when translator sessions are 0 |

Binding rules:
* **Never two large-context tasks concurrently.** Stagger them; the second
  one's prefix will be evicted by the first, turning a 200 s task into a 600 s
  timeout.
* **When a translator tenant is active** (any session with a fresh turn),
  drop to slots 2-3 only, and defer anything that would prefill more than
  ~25 k tokens. This is the direct mitigation for the 21:48 incident.
* **The pool pauses entirely** when the server must go down for a GPU window
  or when the model is switched. This is enforced, not just documented: the
  runner requires `/health` 200 plus a resolvable model id before every run.
  The gate was added after an outage at 22:09 silently let a run start against
  a dead endpoint and produced a negative token delta from the counter reset —
  that run (b383, first attempt) was discarded and repeated.
* Revisit the whole slot plan once the HTTP decoupling lands; the "reserved"
  slot 4 exists only because of it.

---

## 7. Honest limitations

* **n is small and one run per bug.** 9 bug attempts, of which 2 produced an
  answer and 1 of those was contaminated. Nothing here separates model
  capability from run-to-run variance, and a single timeout is not proof that
  a bug is unsolvable for this model — only that it was not solved within 600 s
  on the first try.
* **The judge is the operator agent** (this one), grading against a key it
  wrote itself from the fix commits. Every Arm-B factual claim was
  independently re-verified against the tree; the Arm-A mechanism judgements
  were not double-blinded.
* **The b525 contamination was found, not designed out.** It is the only
  detected case, but "only detected" is the honest phrasing: the scan looked
  for commit references and git verbs in the answers, which would miss a run
  that read history and did not say so. Only b525 was re-run sanitized; the
  five earlier timeouts (b289, b300, b383, b392, b406) ran with history
  reachable and did not solve their bugs anyway, so for them the hole was
  available and unused — but that is an inference from the outcome, not a
  control.
* **The silent-wrongness stratum (b443, b446, b501) yields no measurement.**
  Two ran and were destroyed by server restarts, one never got past the
  readiness gate. That stratum was a third of the intended design, so the tier
  list rests on alignment, device-order and test-honesty only.
* **The serving process was replaced three times during the evaluation**
  (22:09, ~23:52, ~00:05), each time with the same INT8 checkpoint. Two runs
  were silently corrupted before the readiness gate existed and were
  discarded; two more were caught by it. Any number here that came from a
  `/metrics` delta spanning a restart is marked invalid rather than reported.
* **Symptom provenance is mixed.** b316 and b383 carry verbatim archived
  tracebacks; b289 and b300 carry the original error string with reconstructed
  boot context; b392, b406, b443 and b446 are reconstructed observations from
  the fixes' own can-fail runs. The reconstructed ones are honest about what
  was observed but were not captured live.
* **Sibling precedent was left in the trees** (7 of 11 pre-fix trees already
  contain a solved sibling of the same family). That is the situation a real
  engineer is in, but it makes those bugs easier than a cold one and it is
  recorded per bug in `answers.md` rather than corrected for.
* **The token figures include the harness.** ~22 k prompt tokens per run are
  Claude Code's system prompt, not the task. A different client would move
  every number in the cost table.
