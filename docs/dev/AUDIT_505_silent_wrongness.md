# Audit #505 — silent wrongness: four axes over the standing tree

Desk audit, nothing executed, no GPU held. Base commit `d653405223`
(`origin/integration/r3-probe-next2`), audit branch `docs/silent-wrongness-505`.
No `.py` file was modified by this audit; the only code-adjacent change is the
correction of `FEATURE_CATALOG.md` §14 and §16, which axis D required.

## 1. The occasion

Three defects found on one day, which have a shared shape rather than a shared
location:

- **C1** — a packed draft weight name matched no parameter. The loader logged
  `unexpected weight`, `continue`d, and the draft loaded nothing; the only symptom
  was a speculative accept rate of zero.
- **#501** — a comment asserted *"a decline leaves no partial state"*. The code
  contradicted it and the scheduler died.
- **#449 / #493** — a query-chunk cap existed and was correct, but shipped at a
  desk-picked 2048 MiB above the real peak. It protected nothing for weeks.

All three are the same failure: **the tree contained a true-looking statement — a
log line, a comment, a numeric default — that no mechanism was obliged to make
true.** CLAUDE.md's three standing laws each name one facet (MECHANISM REACH,
REACH INCLUDES PARAMETERS, SUCCESS CLAIMS ARE NOT EVIDENCE). This audit is their
systematic application to the existing code rather than to a new change.

## 2. The four axes

| axis | question | template incident |
|---|---|---|
| **A** (A1 + A2) | which warn-and-continue sites leave state silently WRONG rather than degraded? | C1 |
| **B** | which comment-asserted invariants are pinned by a test, and which does the code contradict? | #501 |
| **C** | which shipped numeric default that exists to BOUND something has ever been shown to bind? | #449 / #493 |
| **D** | do `FEATURE_CATALOG` §14/§16 match their code predicates? | #492 (the reach law's own occasion) |

Axis D uses `AUDIT_500_mechanism_reach.md`'s method directly and closes the coverage
gap #500 declared for itself: *"§14 (dashboard) and §16 (instruments) were not swept
for predicates … That is a stated coverage gap, not a clean bill of health."*

Each axis ran as an independent sweep with its own extraction grep, per-site reading,
and classification. Every row cites `file:line` and quotes the operative code
verbatim; a row that rests on a docstring or comment rather than an executed branch
says so, because that distinction is the point of the exercise.

## 3. What this audit deliberately did not do

Nothing was executed: no test run, no server booted, no GPU touched, no measurement
taken. Every proposed binds-proof in axis C is a proposal, not a result — axis C
produces the backlog of missing evidence, not the evidence. No behaviour was changed;
every finding is a task proposal, and none of them was fixed here.

Per-axis coverage numbers — grep totals against sites actually opened, and the named
surfaces not reached — are stated at the head of each axis section below. They are
reported as gaps, not as clean bills of health.

## 4. Results

| axis | surface enumerated | opened in source | kept as findings | headline |
|---|---|---|---|---|
| A1 — warn-and-continue: loader / spec / MoE | 186 warning sites (+2 silent-shape greps) | 46 | 9 DANGEROUS | the C1 remedy reaches 3 of 27 draft classes and no GGUF boot |
| A2 — warn-and-continue: memory / dist / pool | 410 warning sites + 51 silent shapes | 58 | 6 DANGEROUS + 1 registration | barlink's two fastest transports fall back per rank with no group agreement |
| B — comment invariants | 1345 candidate comments → 154 priority union | 41 | 1 CONTRADICTED, 11 UNPINNED, 10 PINNED | one executed contradiction; #501's own shape is LIVE at this base |
| C — bounding numeric defaults | 281 fork-added flags/envs → 106 bounding-worded | 42 | 71 INERT, 5 BOUND-PROVEN | **zero** shipped bounding defaults have a value-pinning falsifier |
| D — catalog §14/§16 | 13 conditional claims | 13 | 3 bug candidates | "one-click knee-point probe" has no implementation |

Two numbers carry the audit.

**Axis C: 71 of 106 fork bounding defaults (67 %) sit behind a gate that is off in
the served configuration, and of the ~35 that can act, not one has a test that fails
when the default is doubled or removed.** The BOUND-PROVEN table is not empty, but
four of its five rows are evidence class (c) — a note at the constant naming the
measurement it came from — and three of those were measured on a geometry this rig
cannot reach. The fork tests that its guards FIRE; it does not test that its numbers
are RIGHT. `test_retract_decode_fcfs.py:232` is the proof of the pattern: it reads
the default it is testing (`max_retries = envs.SGLANG_RETRACT_SOLO_OOM_MAX_RETRIES.get()`)
and therefore passes for every possible value. That is why #449 could ship inert for
weeks — not an oversight at one posten, a missing instrument class.

**Axis B: the testing culture is not the problem.** 10 of 22 kept invariant-claims
are genuinely pinned, with good tests, and every UNPINNED row sits in a module whose
neighbours are pinned. The #501 shape survives only where an invariant spans **two
functions** (guard here, mutation there) or **two consumers** (publish here, cap
there) — never inside a single well-tested predicate. That localises the remedy: the
gap is not "write more tests", it is "pin the claims that cross a function boundary".

### Calibration — the instruments discriminate

Per the SUCCESS-CLAIMS law an instrument's verdict counts only after it shows it can
tell known-different inputs apart. Three checks:

- Axis B's AST pass, given no file list, **re-found #501 unaided**
  (`kv_session_offload.try_spill`: `free` at `:3358` before the declines at `:3382`,
  `:3394`, `:3411`, `:3434`, under the comment at `:3397`). Fix `1f2b24ef11` is **not
  an ancestor of this base**, so #501 is live on `integration/r3-probe-next2` and the
  merge chain must carry the fix forward.
- Axis B's one CONTRADICTED row was **executed** hermetically
  (`CUDA_VISIBLE_DEVICES=99`, worktree `PYTHONPATH`), not desk-argued — the trace is
  quoted in the axis-B section.
- Axis A1 records four negative results as exemplars (`bar1ep.py:879` byte proof,
  `compressed_tensors.py:1030`, `gguf_registry.py:347`, `eagle_info.py:229`), and
  axis D records two [WIDER] rows. A sweep that returned only findings would be the
  suspect one.

### The strongest possible finding is absent, and that is reported as a result

Axis C looked for "accepted-then-inert with no consumer at all" — the shape audit
#421 exists for, and the most damning verdict available on this axis. **No fork
posten has zero consumers.** Fourteen looked that way on the first pass and all
fourteen resolved; the #236 spill budgets, the most suspicious group, reach their
consumers through a string-built lookup
(`getattr(sa, "kv_session_offload_" + name, …)`, `kv_session_offload.py:1305-1321`)
that a literal grep cannot see. The defect in this tree is not dead knobs. It is
live knobs whose values nothing has ever tested.

## 5. The ten to act on

Ranked by damage x reachability at the served geometry (uneven TP=3 on 5090 + 2x3080,
uneven DCP, NEXTN, barlink, 27B-35B dense / 122B-A10B offloaded / DSV4-Flash GGUF).
Full argument and all supporting citations are in the per-axis sections below; none
of these was fixed here.

**1. #505-C-01 — the client-liveness timeout table is live by default, self-labelled
"Unmeasured", and aborts healthy streams.** `liveness/classes.py:81-96` ships twelve
silence budgets (`LLM_STREAM = 90.0`) under its own comment *"Unmeasured … the
numbers encode an argument about the consumer, not a measurement of the server"*.
Unlike almost everything else in this sweep it is **not behind a feature gate**:
`serving_base.py:119-124` wraps every OpenAI-shaped streaming response, and on expiry
`watchdog.py:336-350` → `stream.py:169-175` calls `abort_request` on a live request.
The clock advances only on bytes accepted by the transport, so a long first-token
latency counts in full. On the standing recipe 90 s is not obviously above the worst
legitimate TTFT, and nobody has checked.
**Stated precondition, not verified by this audit:** whether the chat generator emits
an early role/keep-alive chunk that restarts the clock was NOT checked. That single
question decides whether this is severe or moot, and it is step 0 of the task — the
finding is ranked first on damage-if-real, not on confirmed damage.
*Task:* `check the early-chunk question first, then measure per-class inter-byte gaps and derive the liveness table (LLM_STREAM=90s vs real TTFT)`

> **CORRECTION (#514, 2026-08-03) — this ranking was wrong; step 0 answers it.**
> The stated precondition was resolved at the code and **TTFT is outside the
> budget**, so the "aborts healthy streams" claim does not hold as written.
> The mechanism is not an early role chunk — that one is emitted *inside* the
> `async for content in generate_request(...)` loop (`serving_chat.py:1123-1134`)
> and so cannot restart anything early. It is an ORDERING one level up:
> `_handle_streaming_request` awaits `generator.__anext__()`
> (`serving_chat.py:1012`, `serving_completions.py:202`) and only *then* builds the
> `StreamingResponse`; `serving_base.py:111-119` awaits that handler and passes the
> **result** to `guard_generate_stream`. The watchdog therefore does not exist while
> the request queues and prefills, and `last_progress_at` is set at construction
> (`watchdog.py:204-210`). The 90 s covers the gap BETWEEN chunks — milliseconds at
> decode — not the time to the first one.
> What survives: the number is still unmeasured, and its safety now visibly rests on
> that ordering. #514 therefore did **not** change the default (that would have acted
> on a refuted premise); it pinned the ordering
> (`TimeToFirstTokenIsOutsideTheBudgetTest`, 4 tests, the structural pin proven
> can-fail by removing the pre-pull) and recorded the scope at
> `DEFAULT_TIMEOUT_RATIONALE`. A measurement of real inter-chunk gaps remains
> worthwhile but is no longer urgent, and this finding should not have been ranked
> first. Same lesson as the #500-B1 refutation: an audit finding is a hypothesis
> until its falsifier is red.

**2. #505-B-01 — `offload_movement` reports a completed wave-in as still parked and
never releases the booking (executed, not argued).** The comment at
`offload_movement.py:916-918` says the error path means *"retrieval failed physically.
The item stays PARKED (its bytes are still at the park target)"*. The `try` it guards
does not end at the retrieval (`:910-914`): it continues through `copy_in_tensors`,
a `wait`, and `free_destination`, whose own contract (`:333-335`) is "release the park
destination after a **completed** wave-in". The handler writes `mv.state = STATE_PARKED`
unconditionally (`:919`) and the booking release sits outside the try (`:924`).
Hermetic run, driving the registered test's own harness:
`fail_at=free_destination` → `state=parked booked=1000` after
`[copy_out,wait,wait,copy_in,wait,free_destination]`. Permanent memtier over-booking
while the item is resident, and a retry re-copies from a destination already told to
free. The existing test injects only at `copy_in` — the one point where the comment
is true.
*Task:* `offload_movement: the error path must distinguish a failed retrieval from a failed release, and the test must inject at both`

**3. #505-A1-01 — the C1 remedy reaches 3 of 27 draft classes and no GGUF boot.**
`raise_on_unloaded_draft_parameters` (`weight_utils.py:2032`) is precisely the fix for
the occasion bug and claims to be *"a property of loading A DRAFT, not of one model
class"*. Three lines set its real reach: `if loaded_params is None: return` (`:2058`)
— only `qwen3_5_mtp.py`, `step3p5_mtp.py` and (via super) `gemma4_mtp.py:382` report
a set; the single call site (`loader.py:903`) is inside `DefaultModelLoader`, and
`GGUFModelLoader` discards the return outright (`loader.py:2409`);
`QuantizedRLModelLoader.load_weights_proxy` also returns `None` (`:1095-1102`).
Sharpest instance: `qwen3_next_mtp.py:149` drops the base class's `return loaded_params`
(`qwen3_next.py:1407`) — one missing `return` disables the guard for the class.
*Task:* `make the draft-load completeness check reach every draft class and every loader (GGUF, QuantizedRL)`

**4. #505-A2-05 — barlink `bar1`/`matrix` fall back to gloo per rank with no group
agreement.** `barlink.py:384-400` catches any bring-up exception, warns and returns
`None` for the two transports deliberately excluded from `_NO_FALLBACK` (`:327`).
Nothing reconciles the outcome across the group, so a probe that fails on one card
only (BAR1 peer mapping, `dmabuf_holder`, the byte proof — `barlink_bar1.py:589`,
`:597`, `:4644-4656`) leaves the group split. Direction of failure is reported as
unproven: a failure before `dist.all_gather_object` (`barlink_bar1.py:1953`) desyncs
into a hang, one after it does not. This also collides with the barlink-default order
in CLAUDE.md — a group silently on gloo is a gloo run published as a barlink run.
The fix is one collective and the tree already contains it twice
(`parallel_state.py:975-992`, `model_runner.py:1365-1369`).
*Task:* `reconcile the barlink transport outcome across the group before use`

**5. #505-A2-04 — PD topology fabricates a CUDA→NVML identity mapping.**
`disaggregation/topology.py:421-427` logs *"assuming identical enumeration orders"*
and sets `mapping = {}` — on the rig where CLAUDE.md records that torch order is not
NVML order. Card VRAM totals then land on the wrong ranks and the only downstream
check is warn-only (`:742-743`), so an infeasible plan boots. The function's own
docstring demands `None` + a caller-side report for the sibling NVML failure, and
`barlink_matrix_transport.py:218-224` shows the correct shape (degrade to a NAMED
provenance, never to a fabricated identity).
*Task:* `PD topology must return None when the CUDA→NVML bridge is unavailable`

**6. #505-A1-02 — the spec-kernel-backend collective degrades to a per-rank answer.**
`speculative/eagle_utils.py:193-194`: `except Exception:  # noqa: BLE001 --
single-process/unit-test contexts` / `tp_group = None`, which skips the group
`all_reduce(MIN)` and lets `group_ok = local_ok` stand. The function's own log at
`:220` names the consequence: *"one rank without the native ops puts every rank on
Triton, because verify decides accept counts and a mixture would desync the group
silently."* The comment naming the intended contexts is an asserted invariant with
nothing pinning it.
*Task:* `decide_spec_kernel_backend must not fall back to a per-rank verify kernel when get_tp_group() raises`

**7. #505-D3 — the #152 GitHub result share posts the start command verbatim with no
anonymity gate.** `github_share.build_report` states *"argv is emitted verbatim"*
(`:186`) and does it (`:214`); `webui.share_submit_payload` (`webui.py:4690-4721`)
posts that markdown to a public issue. `github_share.py` contains no `scrub_tree` and
no `assert_anonymized`. The sibling #271 route refuses absolute filesystem paths by
name (`rig_artifact.py:558-588`, `_ABS_PATH_RE` at `:571`) — and a real start command
on this rig contains `/spinning/llm_stuff/...`. Env redaction keys on five NAME
suffixes only (`:89`). Ranked here rather than lower because it is the one surface
that is public by construction, and this account has a leak incident on record.
*Task:* `route the #152 result share through scrub_tree + assert_anonymized`

**8. #505-A2-03 — `try_spill` declines on a rank-LOCAL quantity where its own twin
raises.** `kv_session_offload.py:3426` compares `n_own` — derived from this rank's
owner window under weighted uneven DCP — against the replicated `region_tokens`, warns,
and returns `False`, sending only this rank down the stock-retraction path
(`scheduler.py:4051`). Every neighbouring decline is annotated as resting on
replicated inputs; this one is not. The prefill-spill twin raises on the identical
condition (`:3874-3880`), and the module header states the doctrine:
*"RANK-UNIFORMITY (this fork: divergence == NCCL hang, not a wrong number)"* (`:501`).
*Task:* `try_spill must not decline on a per-rank owned-row count — raise like the twin or min-reduce the verdict`

**9. #505-C-05 — no shipped bounding default in this fork has a value-pinning
falsifier.** 71 of 106 are gated off in the served configuration; of the rest, the
BOUND-PROVEN table has five rows and **zero** of evidence class (a). The instrument
gap is systematic, not per-posten, and it is the direct reason the #449 lesson was
learnable only in hindsight. The remedy is a convention, not a fix list.
*Task:* `add a value-pinning convention: every shipped bounding default gets a test that fails when it is doubled or removed`

**10. #505-A2-01 — the kvso draft-scratch carve-out silently no-ops on hybrid
allocators and then logs success.** `kv_session_offload.py:2204-2207` does
`self.allocator.size -= int(self.mtp_resident_slices)` inside `except Exception: pass`.
On `UnifiedMambaAlloc` / `UnifiedSWAAlloc` — the allocators for the hybrid GDN models
this fork serves — `size` is a property whose setter is literally `pass`
(`multi_ended_allocator.py:1764-1766`, `:1974-1976`), so the write vanishes without an
exception, the slots stay counted in advertised capacity while out of circulation, and
`:2208` logs *"reserved %d draft-read scratch slots"* either way. The invariant the
site's own comment names (`:2198-2203`) is permanently off. Reach qualifier stated
honestly: the arm is behind `KVSO_ALLOW_SPEC=1` (`server_args.py:6580`, audit #500's
B10), so a stock speculative boot does not reach it today.
*Task:* `kvso draft-scratch reservation must verify the pool carve-out took effect`

### Immediately actionable, outside the ten

- **#501 is live at this base.** Fix `1f2b24ef11` is not an ancestor of
  `d653405223`; the running merge chain must carry it forward.
- **#505-C-04** — #449's own default (`SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB = 2048`,
  `environ.py:1638`) is still the desk number the law was written about, and
  `NOTE_449` §5's measurement arm is still titled "BOOT-PENDING, not run". The posten
  that supplied the lesson has not been discharged.
- **#505-C-03** — `SGLANG_MEASURED_KV_BUDGET_SAFETY_MIB = 400` is contradicted by a
  measurement recorded in its own consumer (`model_runner_kv_cache_mixin.py:809-815`:
  10k prefill ~1 GiB, 50k ~2-3.5 GiB on the draft-solo host).
- **#505-C-06** — barlink reads six knobs through raw `os.environ` at import time, so
  their `environ.py` declarations are decorative, `envs.X.set()` after import does
  nothing, and one literal already disagrees with its declaration
  (`barlink_device.py:989` `"4"` vs `environ.py:658` `None`).
- **Upstream, not ours:** `SGLANG_VLM_CACHE_SIZE_MB` is declared `EnvInt(100)`
  (`environ.py:1465`) and read as `os.environ.get(..., "4096")`
  (`disaggregation/encode_server.py:330`) — a 40x drift between declared and effective
  default, identical in `upstream/main`. For the upstream-PR pile.

## 6. Honest coverage — what this audit did NOT sweep

Stated as gaps, not as clean bills of health. A surface listed here was not audited;
nothing below should be read as "checked and fine".

**Nothing was executed except one falsifier.** Axis B ran the `offload_movement`
contradiction hermetically (`CUDA_VISIBLE_DEVICES=99`) and one `vram_dial` rank-count
check. Everything else is desk reading. No server booted, no GPU held, no measurement
taken — so every axis-C "proposed binds-proof" is a proposal.

**Axis A1** — 140 of 186 sites read at grep + 3-6 lines of context rather than opened
(no DANGEROUS row rests on a context-only read). Named gaps: 40 of 46
`layers/quantization/` warnings (the largest single gap in that part), the 94 non-C1
`models/` warnings, `ci_weight_validation.py` (18), `moriep.py`/`deepep.py` (7),
`expert_compute_placement.py`, `moe/utils.py`, `fused_moe_triton_config.py`,
`hibernate.py`.

**Axis A2** — 410 sites read at message level, 50 opened; the silent-shape sweep found
51 hits and opened 8. Named gaps: `mem_cache/storage/{flexkv,eic,hf3fs,nixl,aibrix,
simm,umbp,lmcache}` (~40), `disaggregation/{mooncake,mori,nixl,ascend}` — **79 of the
83 disaggregation hits, and the likeliest place for a second finding of the A2-04
shape** — managers multimodal/tokenizer/detokenizer (~30), NPU/Ascend and CPU paths.
The `except: pass` sweep covered only bare `pass`/`continue`/`return None`; fallbacks
that substitute a *default value* were followed only where a warning already pointed
at them.

**Axis B** — 1345 candidate comments, 154 in the priority union, 41 opened, 22 kept,
1323 triaged out. Not reached: `model_loader` (1 of 41 opened), `memtier` (0 of 21),
`layers/dcp` (0 of 14), all `mem_cache/storage/**` sub-backends, `scheduler.py` beyond
four sites. The grep total came in ~12 % above the figure quoted in the briefing; the
per-directory table in the axis-B section gives the reconciliation.

**Axis C** — 106 fork bounding postens enumerated by AST, **42 opened**, 64 not opened
(listed with reasons in the axis-C section; no verdict implied for those). 519 upstream
`Env*` entries and 457 upstream `ServerArgs` fields were enumerated for the set
difference and then dropped per the standing "our bugs, not all of sglang" rule.
`AUDIT_434_planner_constants.md` covered `uneven_perf.py` + `planner/**` module
constants (764 literals) and touched neither `environ.py` nor `ServerArgs`; the four
overlapping rows are excluded rather than re-reported.

**Axis D** — all 13 conditional claims in §14/§16 resolved, but **Direction 2 for
those sections was not done**: the inverse sweep for capabilities that are wired and
uncatalogued. §14's six catalog lines describe roughly 33,000 lines of fork code
(`planner/webui.py` 14,816, `rigmon/` 9,971, the seven supporting modules ~8,500). The
size of that gap is recorded; its contents are not enumerated. `expert_stats` is the
one §16 item taken from module presence rather than traced to a consumer.

**Not an axis of this audit at all:** upstream sglang code, the `sgl-kernel` C++/CUDA
surface, `tools/`, `scripts/` beyond the two §16 items, the dashboard's client-side
JavaScript, and the video-enhance / TTS / diffusion lanes.

---

# Part A1 — warn-and-continue: loader / weights / spec / MoE offload

## Axis A1 — warning-instead-of-error: loader / weights / spec / MoE-offload

The defect class, from the occasion: a site DETECTS an anomaly, logs a warning,
and CONTINUES, where the resulting state is silently WRONG rather than merely
degraded. The bar for DANGEROUS is a concrete answer to *what exactly would be
wrong, and through which observable would a user not notice it*. Rows that
cannot answer that are BENIGN, and are marked so without argument.

The tree already knows this class well. Three sites in scope are the CORRECT
shape and serve as the fix templates:

- `models/deepseek_v4_dspark.py:868` `_assert_required_params_loaded` — the
  #496-(b) completeness check, raising by name on unwritten parameters.
- `models/deepseek_v4.py:3074-3076` `raise KeyError(_unmatched_gguf_tensor_message(name))`
  — an unmatched tensor is fatal on the GGUF route because the adapter's name
  table already refused everything it did not recognise.
- `layers/moe/token_dispatcher/bar1ep.py:879` `byte_proof()` — a proof whose
  failure makes the dispatcher DECLINE (`raise Bar1EPUnavailable` at `:877`),
  never warn.

Almost every DANGEROUS row below is a place where one of those three patterns
exists elsewhere in the tree and does not reach here.

### Coverage

Grep command as briefed, per directory:

| surface | grep total | opened in source | not opened |
|---|---|---|---|
| `python/sglang/srt/model_loader/` | 56 | 13 | 43 |
| `python/sglang/srt/speculative/` | 30 | 7 | 23 |
| `python/sglang/srt/layers/moe/` | 34 | 7 | 27 |
| **subtotal (briefed dirs)** | **120** | **27** | **93** |
| C1 string family in `models/` + `layers/quantization/` (`unexpected weight`, `not found in params_dict`, `no parameter`, `skipping/ignoring weight`) | 66 | 19 | 47 |
| **total** | **186** | **46** | **140** |

"Opened in source" means I read the executed branch and its consequence. The
other 140 were read at grep + 3-6 lines of context, which is enough to classify
a deprecation notice or a perf-config fallback but not enough to call something
DANGEROUS — so no DANGEROUS row below rests on a context-only read.

Two further greps were run for the shapes that carry NO warning at all
(`except Exception:` + `pass`/`continue`/`return None`, bare `continue` in a
weight loop) across `layers/moe/expert_offload.py`, `breakable_offload.py`,
`cold_tier_fetch.py`, `cold_tier_shm.py`, `offload_capture_gate.py`,
`expert_heat_migration.py`, `resident_fraction.py`, `router.py`,
`layers/quantization/gguf.py` and `speculative/`. Three rows below come from
that grep rather than the warning grep, and they are the worst of the set
because they carry no message at all: `speculative/eagle_utils.py:193-194`
(A1-02), `models/qwen3_5_mtp.py:404-405` (A1-01), and
`speculative/dflash_utils.py:44-47` (A1-05).

**Not reached, named honestly:**

- `model_loader/ci_weight_validation.py` — 18 of the 56 model_loader sites.
  Read at context only. It is a CI cache-hygiene utility whose warnings all
  precede a `return False` / re-download; no serving path consumes it. Not
  swept in depth, and not claimed clean.
- `layers/moe/token_dispatcher/moriep.py` (4 sites) and `deepep.py` (3 sites) —
  context only. NPU/DeepEP dispatchers, not on this rig's path.
- The remaining 94 `logger.warning` sites in `models/` outside the C1 string
  family, and 40 of the 46 in `layers/quantization/`. Only the C1 family was
  swept there, as briefed. A second pass over `layers/quantization/` is the
  largest single gap in this part.
- `layers/moe/expert_compute_placement.py` (4), `layers/moe/utils.py` (6),
  `moe_runner/triton_utils/fused_moe_triton_config.py` (6) — context only; all
  six of the `utils.py` ones are `"X is not initialized, using <default>"`
  getters and all six config ones are perf-config fallbacks.
- `model_loader/hibernate.py` (5) — context only. Every one of them ends
  `-> cold load`, i.e. falls back to the full, correct load. Benign by
  construction.

### Table

| file:line | site (symbol) | classification | what would be silently wrong | fix pattern |
|---|---|---|---|---|
| `model_loader/weight_utils.py:2058` | `raise_on_unloaded_draft_parameters`, `if loaded_params is None:` / `return` | **DANGEROUS** | The #290/#318 guard against a draft that loaded nothing is a no-op for 25 of 27 draft classes (see A1-01). The draft runs on `torch.empty`, accept length pins at ~1.0, and no error is raised anywhere. | (ii) — the guard IS the #496 shape; it needs reach, not redesign. Make the report mandatory for `is_draft_model`, i.e. turn `loaded_params is None` from "skip" into "this draft class does not report — refuse". |
| `model_loader/loader.py:2409` | `GGUFModelLoader.load_model`, `model.load_weights(_timed(iter(weights_iterator)))` | **DANGEROUS** | Return value discarded, so the draft guard at `loader.py:903` never runs on a GGUF boot. A GGUF MTP draft with a name-table defect (the #113 family) loads zero tensors and reports success. | (ii) — capture the return and call `raise_on_unloaded_draft_parameters` here too. |
| `models/qwen3_next_mtp.py:149` | `Qwen3NextForCausalLMMTP.load_weights`, `super().load_weights(weights, is_mtp=True)` | **DANGEROUS** | The base `qwen3_next.py:1407` DOES `return loaded_params`; the MTP wrapper drops it, so this draft reports `None` and the guard returns at `weight_utils.py:2058`. One missing `return` disables the check for the whole class. | (i)+(ii) — `return super().load_weights(...)`. Same one-line audit for every `*_nextn.py` / `*_mtp.py` wrapper. |
| `models/qwen3_5_mtp.py:404-405` | `Qwen3_5ForCausalLMMTP.load_weights`, `if name_mapped not in params_dict:` / `break` | **DANGEROUS** | No warning at all. A local expert whose packed/dense spelling does not match (`.qweight` vs `.weight`) is dropped in total silence, and the checkpoint name is then recorded as loaded at `:437`. The rank-local "not my expert" case and the quant-mismatch case are indistinguishable here. | (ii) — the parameter-side check is the only thing that separates the two cases; ensure it runs (A1-01) and stop polluting the report (`:437`). |
| `models/qwen3_5_mtp.py:437` | same, `loaded_params.add(name)` at loop-body level | DEGRADED-LOUD | Executed unconditionally, including in the `logger.warning_once(...)` skip branch at `:433`, so `loaded_params` mixes checkpoint names with parameter names. It does not defeat the guard (a skipped name is by construction not in `params_dict`) but it makes the report untrustworthy as evidence. | (iii) — the set is the state probe; it must contain only names actually written. |
| `models/gemma4_causal.py:1437-1442` | `Gemma4ForCausalLM.load_weights`, `unloaded_params = params_dict.keys() - loaded_params` → `logging.WARNING: ("Some weights are not initialized from checkpoints", ...)` | **DANGEROUS** | The check is present and correct; its verdict is a log line. A TARGET model with unwritten real parameters serves uninitialised weights. Observable: a WARNING among hundreds of boot lines. Output is fluent text — nothing downstream flags it. This is the same evidence the draft guard RAISES on. | (i) — the `logging.WARNING` bucket (real parameters, as opposed to the INFO/DEBUG buffer buckets, which are correctly informational) must raise, with the same escape env as `SGLANG_ALLOW_UNLOADED_DRAFT_PARAMS`. |
| `models/qwen3_5.py:1593` | `Qwen3_5ForCausalLM.load_weights`, `logger.warning(f"Parameter {name} not found in params_dict")` + `continue` | **DANGEROUS** on the GGUF route | On GGUF, `deepseek_v4.py:3209` argues the case verbatim: every tensor was mapped by an explicit table, so an unmatched name is a mapping defect and the parameter it should have filled stays uninitialised (#391 walls 10-12: "the server came up on uninitialized weights"). Qwen3.5 is the fork's flagship GGUF bring-up and still only warns. Same line at `:1815` (`Qwen3_5MoeForCausalLM`), `:1957`, `:2340` (the two VL classes). | (i) — hoist `_unmatched_gguf_tensor_message` out of `deepseek_v4.py` and raise on `is_gguf` in every loader, not per model. Only `deepseek_v2.py`, `deepseek_v4.py`, `llama4.py`, `deepseek_common/utils.py` have it today. |
| `speculative/eagle_utils.py:193-194` | `decide_spec_kernel_backend`, `except Exception:  # noqa: BLE001 -- single-process/unit-test contexts` / `tp_group = None` | **DANGEROUS** | The function's own log line at `:220` states the hazard: "one rank without the native ops puts every rank on Triton, because verify decides accept counts and a mixture would desync the group silently." The bare except restores exactly the per-rank answer. A rank that takes this arm skips the `all_reduce` MIN, keeps `group_ok = local_ok`, and can end up on a different verify kernel than its peers. Observable: none — accept counts diverge, output stays fluent. The `# single-process/unit-test contexts` comment is an ASSERTED invariant with no test behind it (CLAUDE.md law 3). | (i) — `get_tp_group()` raising inside a serving process must propagate; gate the swallow on an explicit single-process marker instead of on the exception type. |
| `model_loader/loader.py:1257` | `QuantizedRLModelLoader._load_scale_param`, `logger.warning(f"Scale param shape {scale_param.data.shape[-1]} not divisible by {len(shard_names)}")` | **DANGEROUS** | Warns and then falls straight into `offset = 0` and the per-shard write loop using the truncated `rows_per_shard`. The fp8 scale shards land at wrong offsets — every subsequent dequant of that fused linear uses another shard's scale. Observable: none; logits shift, no shape error. | (i) — the condition is already computed; `raise ValueError` on it instead of logging. |
| `model_loader/loader.py:1223` | same, `logger.warning("[QuantizedRL] Scale parameter not found: %s", scale_param_name)` / `return` | **DANGEROUS** | The weight was requantised; its scale keeps its construction value. Dequant with a stale scale is a silent numeric error on an RL weight-update path, where the model is expected to change between rollouts, so drifting output is the EXPECTED signal — the one observable a user would have is preempted. | (i) — a scale name derived mechanically from a param name that is present is not optional; raise. |
| `model_loader/loader.py:1232` | same, `logger.warning("[QuantizedRL] Scale shape mismatch for %s: expected %s, got %s", ...)` | **DANGEROUS** | Skips the `copy_`, same stale-scale consequence as `:1223`. | (i) |
| `model_loader/weight_utils.py:1929` | `kv_cache_scales_loader`, `"Defaulting to KV cache scaling factors = 1.0 for all layers in TP rank %d as an error occurred during loading."` | **DANGEROUS** | Reached from three `except` arms (`FileNotFoundError`, `JSONDecodeError`, bare `Exception`) after the user explicitly supplied a calibration file. With fp8 KV, scale 1.0 means unscaled e4m3 casts: values past the e4m3 range saturate. Observable: attention quality only. The user asked for calibrated scales and silently got none. | (i) — an explicitly supplied path that fails to load must raise. The fallback is only defensible when no path was given. |
| `model_loader/weight_utils.py:1766`, `:1796` | `maybe_remap_kv_scale_name`, `"... but not found the expected name in the model (e.g. {remapped_name}). {scale_name} is not loaded."` → `return None` | **DANGEROUS** | Callers treat `None` as `continue`. Same end state as `:1929` — fp8 KV on default scales — but per tensor and per name, so no single line says "no scales were loaded at all". `print_warning_once` deduplicates. | (ii) — a completeness check on the attention side after load: if the model declares `k_scale`/`v_scale` parameters and none were written, refuse. |
| `speculative/dflash_worker_v2.py:1859` | `_validate_phase1_sampling_support`, `"DFLASH non-greedy verification is unavailable on this build/device; falling back to greedy argmax verification."` | **DANGEROUS** | A function named `_validate_...` that only warns. A request with `temperature`/`top_p` is verified greedily, so the served distribution is not the requested one. Warned ONCE, on `tp_rank == 0` only, at the first non-greedy request — long after boot. Observable: output is plausible and merely more deterministic. | (i) — refuse the request (or refuse the boot when `is_dflash_sampling_verify_available()` is False and sampling is reachable), rather than substituting a different sampler. |
| `speculative/dflash_utils.py:44-47` | `except Exception:` → `top_k_renorm_prob = None; top_p_renorm_prob = None; tree_speculative_sampling_target_only = None` | **DANGEROUS** (root of the row above) | `_DFLASH_SAMPLING_VERIFY_AVAILABLE` stays `False` with NO log of the exception. An `sgl_kernel` import that fails for an unrelated reason — an ABI/wheel mismatch, the known dual-dist wheel trap — is indistinguishable from "this device has no kernel", and silently converts the whole server to greedy verification. | (iii) — log the exception at ERROR with the resolved wheel path; the availability flag is a state claim and needs a probe, not a swallowed import. |
| `speculative/adaptive_graph_memory.py:1032` | `"Adaptive graph memory: could not attribute tagged buffer of %s (ptr 0x%x) to a build-window segment; physical-isolation audit incomplete for it."` | **DANGEROUS** | Twelve lines above, the SAME audit raises `RuntimeError` when a buffer is attributed to another tag's window ("Pausing one state would unmap another state's memory"). An UNATTRIBUTABLE buffer is the case where the audit could not decide — and it proceeds. If that buffer does belong to another state's segment, a pause unmaps live graph memory and replay reads reclaimed pages: wrong logits, no fault. The in-code comment "Not fatal by itself (snapshot attribution can miss a segment)" is an asserted invariant, unpinned. | (iii) — the audit is the only evidence the mode is safe; an incomplete audit is not a pass. Either raise, or count unattributed buffers and refuse the offload mode above zero. |
| `speculative/adaptive_graph_memory.py:1218-1222` | `_maybe_verify_rank_sync`, `except Exception:` → `logger.warning("SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC check skipped", exc_info=True)` | DEGRADED-LOUD | The instrument for #50/G5 rank-divergent swaps silently does not run; the swap proceeds. Downgraded from DANGEROUS only because the whole check is opt-in (`SGLANG_ADAPTIVE_ALIAS_VERIFY_RANK_SYNC` is off by default), so it protects nothing by default anyway — a user who switched it ON is precisely the user who must be told it did not run. `RuntimeError` is correctly re-raised at `:1216`. | (i) — a check the user explicitly enabled must raise when it cannot execute. |
| `layers/quantization/npu_mxfp4.py:115-122` | `NPUMXFP4Config.get_quant_method`, `"MXFP4 W4A8 quantization is not yet supported for FusedMoE layers (prefix=%s). Falling back to unquantized MoE — MoE weights will run in full precision (BF16/FP16)."` → `return UnquantizedFusedMoEMethod(...)` | **DANGEROUS** | The message is a false SUCCESS CLAIM about state: an MXFP4 checkpoint has no bf16 MoE tensors to "run in full precision". The layer declares dense `w13_weight`/`w2_weight`, the checkpoint's packed tensors match nothing, and they are dropped by the model's own `not found in params_dict` warning — the exact compound failure of the occasion bug. Reach is NPU-only, so low on this rig, but the shape is textbook. | (i) — `raise NotImplementedError`, exactly as the sibling linear branch at `:108` already does for the same missing kernel. |
| `layers/quantization/modelslim/modelslim.py:212`, `:240` | `get_scheme` / `get_moe_scheme`, `logger.warning(f"Unsupported Linear modelslim scheme: ...")` → `return None` | **DANGEROUS** (NPU-only reach) | `None` routes the layer to the unquantized method over a quantized checkpoint; same packed-vs-dense drop as above. | (i) — refuse by scheme name. |
| `layers/quantization/fp8.py:582-585` | `Fp8LinearMethod._maybe_pad_weight` block check, `if skip_block_quant_check:` → `print_warning_once("Skipping block quantization checks for weight partition.")` | BENIGN | The two callers are `models/mimo_v2.py:492` and `layers/linear.py:2350`, both passing a literal `True` for a known-unshardable layout; the checks skipped are divisibility guards that would raise on a partition the caller has already established is not TP-split. One clause: the guard is skipped where its premise (`tp_size > 1 and input_size // input_size_per_partition == tp_size`) does not hold. | — |
| `layers/quantization/compressed_tensors/compressed_tensors.py:1030-1039` | NVFP4 lane fallback, `"NVFP4: layer '%s' has no form this rank's FP4 lane can serve (%s), so the checkpoint's packed weight is DEQUANTISED to dense %s at load"` | DEGRADED-LOUD | Model of what this class should look like: the change is named, the cost is named ("costs bf16 VRAM instead of 4 bits"), the numerics are argued exactly ("No precision is lost"), and VRAM is the observable. | — |
| `layers/quantization/compressed_tensors/compressed_tensors.py:966-971` | `"Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod"` → `return None` | BENIGN | Branch condition is `elif weight_quant is None:` — the layer is genuinely unquantized in the config; nothing packed to drop. | — |
| `model_loader/gguf_registry.py:347-357` | depth reconciliation, `"GGUF depth reconciliation: ... Using the file's %d (the rest of the geometry matches, ...)"` | BENIGN | Every non-depth geometry disagreement already raised at `:325`, and a `layer_types` pattern that cannot be extended raises at `:339`. Only the file-wins depth case warns, and the model is BUILT at the file's depth, so no layer is left unwritten. | — |
| `model_loader/loader.py:1660-1666`, `:2822-2828` | `ShardedStateLoader` / `RemoteModelLoader`, `"loading tensor of shape %s into parameter '%s' of shape %s"` | DEGRADED-LOUD | A short tensor loads into a narrowed view and the tail of the parameter keeps its construction value. Named per key, and the loop's own `if state_dict: raise ValueError(f"Missing keys ...")` covers the wholly-absent direction. The intended case (LoRA padding) is documented at the site. | — |
| `layers/moe/token_dispatcher/bar1ep.py:853-857` | `_selftest_if_needed`, `"bar1ep: byte proof skipped via SGLANG_BAR1EP_SELFTEST=0. With that, no number from this run carries any statement about whether the bytes actually arrive."` | BENIGN | Opt-out, off-by-default-safe, and the message states the epistemic consequence exactly. `:919`/`:930` both end in `return False` → `raise Bar1EPUnavailable` at `:876`. Template row. | — |
| `layers/moe/offload_capture_gate.py:412-413` | `resolved_backend`, `except Exception:` / `return None` | **DANGEROUS** — already filed | Feeds `validate_breakable_boot`'s `if backend is None: return` bypass at `:358`. Identical to audit #500's **#500-B8**; recorded here for cross-reference, not claimed as a new find. | (i), per #500-B8 |
| `layers/moe/cold_tier_shm.py:536-541` | `_register_host_memory`, `"could not pin peer segment %s (cudaHostRegister -> %s); fetches from it will be pageable and slower, but correct."` | BENIGN | Pinning is a bandwidth optimisation; the "but correct" claim is structural (the mapping is used either way). | — |
| `layers/moe/expert_offload.py:1232`, `:3930`, `:1652` | `except Exception:` / `pass` around `malloc_trim`, `gc`, and host-shard instrumentation | BENIGN | Host-memory hygiene and an instrument row; neither is on a correctness path, and each carries a `# noqa: BLE001` naming that. | — |
| `layers/moe/expert_stats.py:642-647` | `"SGLANG_EXPERT_STATS=1 but the MoE expert offload is inactive ...: nothing will be recorded."` | DEGRADED-LOUD | The instrument states it will produce nothing — the correct handling of a measurement that cannot run. | — |
| `speculative/dflash_solo_pool.py:262-268`, `:334-340` | zero-KV holes / LRU reclaim of draft slots | DEGRADED-LOUD | Draft KV holes degrade accept rate; the target verify still decides every emitted token, so output correctness is unaffected, and both messages name the accept-rate observable and the knob. | — |
| `speculative/eagle_info.py:224-235` | `filter_batch`, `error_msg = f"length of new_indices: ... != length of topk_p: ..., this should not happen"` → `raise` if strict else `logger.warning(error_msg)` then positional truncation | DEGRADED-LOUD | Would be DANGEROUS (positional truncation misassigns `draft_probs`, which breaks the rejection-sampling correctness argument, not just the accept rate) — except `SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK = EnvBool(True)` (`environ.py:1408`), so the default path RAISES. The warning arm is reachable only for a user who opted out. Recorded because the opt-out is the wrong default direction for a "this should not happen" invariant. | (i) if the flag is ever flipped |
| `speculative/dflash_worker_v2.py:254-258` | `"DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s."` | DEGRADED-LOUD | Runs the draft at a block size its checkpoint was not trained for; accept length is the observable and the message names both numbers. Same shape at `dspark_components/dspark_config.py:101-107`. | — |
| `speculative/dflash_worker_v2.py:935`, `:1630`, `:1668`, `:2113`, `:2435` | fused-KV / Triton fast paths, `"... failed, falling back to sequential path: %s"` then `self._use_fused_kv_materialize = False` | DEGRADED-LOUD | Each latches the flag off and recomputes via the eager path. The residual question — whether the failed fused call left a partial KV write that the fallback then overwrites completely — is a COMMENT-level claim at these sites, untested. Flagged as a testable-claim, not scored DANGEROUS. | (iii) if pursued: a falsifier that injects the exception mid-write and compares KV bytes against the never-fused reference |
| `speculative/draft_worker_common.py:45-51` | `"%s draft worker only supports attention_backend in %s for now, but got %r. Falling back to '%s'."` | BENIGN | Backend substitution, both names printed, no state left unwritten. | — |
| `speculative/dspark_components/dspark_planner.py:194-198` | `"DSpark SPS table is uninitialized (flat): the verify budget degenerates to verify-all (zero scheduling gain)."` | DEGRADED-LOUD | Verify-all is the conservative direction — more verification, not less; correctness unaffected, and the perf loss is named with its knob. | — |
| `speculative/spec_registry.py:112-118`, `layers/moe/utils.py:244-248` | deprecation notices | BENIGN | Forward-compat, behaviour named. | — |
| `layers/moe/utils.py:322`, `:332`, `:342`, `:350`, `:400` | `"X is not initialized, using <default>"` getters | BENIGN | Read-before-publish of a flags singleton; each names the default it installs, and the defaults are the documented ones. | — |
| `layers/quantization/gguf.py:273`, `:367`, `:414`, `:464`, `:623`, `:789` | wheel/capability probes, `except Exception:` → `False` / `0` / `(0, 0)` | BENIGN | Every one selects a SLOWER but correct kernel route (MXFP4 repack, #72 reroute, MMVQ instead of MMQ, no-`out=` dequant). No numeric path changes; `gguf.py:612-616` even documents deliberately not latching. Perf reach, not correctness. | — |
| `model_loader/utils.py:134`, `:144`, `:171`, `:186` | Transformers-impl compatibility gate skips | BENIGN (context-only read) | All four warn on an explicitly requested `--model-impl=transformers`, i.e. the user asked to bypass. Classified from the message text plus its condition; not opened in depth. | — |
| `model_loader/weight_utils.py:727-730` | `"Found mtp.safetensors but it's not referenced in {index_file}. This is a checkpoint packaging bug. Auto-adding it for loading."` | BENIGN | Adds a file that would otherwise be missed; the failure direction is toward loading MORE, and the guard in A1-01 covers the result. | — |

### Top findings

**#505-A1-01 — the draft-load completeness guard reaches 2 of 27 draft classes,
and no GGUF boot at all.** `raise_on_unloaded_draft_parameters`
(`weight_utils.py:2032`) is the tree's answer to exactly the occasion bug, and
its docstring states the ambition: *"Hoisting the check to the loader makes it a
property of loading A DRAFT, not of one model class."* Its actual reach is set
by three lines. First, `if loaded_params is None: return` (`:2058`): of the 27
draft classes in `models/`, only `qwen3_5_mtp.py` and `step3p5_mtp.py` return a
loaded set at all — `gemma4_mtp.py:382` returns its super's, so three in effect.
`deepseek_v4_nextn.py`, `qwen3_next_mtp.py`, `llama_eagle3.py`,
`kimi_k25_eagle3.py`, `qwen3_moe_mtp.py`, all `*_nextn.py` and the rest report
`None` and are skipped. `qwen3_next_mtp.py:149` is the sharpest case: the base
class DOES `return loaded_params` (`qwen3_next.py:1407`) and the MTP wrapper
drops it with a missing `return`. Second, the only call site is `loader.py:903`,
inside `DefaultModelLoader` — `GGUFModelLoader` discards the return value
outright (`loader.py:2409` `model.load_weights(_timed(iter(weights_iterator)))`),
so a GGUF MTP draft, the fork's own #113 territory, is unguarded. Third,
`QuantizedRLModelLoader`'s `load_weights_proxy` (`loader.py:1095-1102`) also
returns `None`, disabling the guard for anything it wraps. This is the
REACH-INCLUDES-PARAMETERS law applied to a guard rather than a threshold: the
mechanism exists, is correct, is tested
(`test/registered/unit/model_loader/test_draft_quantization_namespace.py`), and
does not act on the configuration where today's bug happened.
*Task:* `#505-A1-01: make the draft-load completeness check reach every draft class and every loader (GGUF, QuantizedRL), not only DefaultModelLoader x self-reporting models`

**#505-A1-02 — the spec-kernel-backend collective degrades to a per-rank answer
on any exception.** `speculative/eagle_utils.py:193-194`:
`except Exception:  # noqa: BLE001 -- single-process/unit-test contexts` /
`tp_group = None`. The group `all_reduce(MIN)` is then skipped and
`group_ok = local_ok` stands. The function's own log line at `:220` names the
consequence: *"one rank without the native ops puts every rank on Triton,
because verify decides accept counts and a mixture would desync the group
silently."* A rank taking this arm can end up on a different verify kernel from
its peers, with no warning and no observable — accept counts diverge, output
stays fluent. The comment naming the intended contexts is an asserted invariant
with nothing pinning it. This is the rank-local-before-collective family
inverted: the rank-local fallback is what breaks the group.
*Task:* `#505-A1-02: decide_spec_kernel_backend must not fall back to a per-rank verify kernel when get_tp_group() raises`

**#505-A1-03 — the QuantizedRL scale reload has three warn-and-proceed sites,
one of which writes scales at wrong offsets.** `loader.py:1223` (scale parameter
not found → `return`), `:1232` (shape mismatch → skip the `copy_`), and worst,
`:1257`: the divisibility warning does not guard anything — control falls
straight through to `offset = 0` and the per-shard write loop using the
truncated `rows_per_shard`, so the fp8 scales of a fused linear land on the
wrong shards. All three leave a requantised weight paired with a wrong or stale
scale. The observable that would normally catch this — changed output — is
preempted, because this path exists to change the weights between RL rollouts.
*Task:* `#505-A1-03: QuantizedRL scale reload must raise on a missing / mis-shaped / non-divisible scale instead of warning and writing`

**#505-A1-04 — an explicitly supplied KV-scale calibration file that fails to
load defaults every layer to 1.0.** `weight_utils.py:1929`, reached from three
`except` arms including a bare `except Exception`. With fp8 KV that is unscaled
e4m3, i.e. saturation past the representable range, with no observable but
answer quality. The per-tensor sibling is `maybe_remap_kv_scale_name`
(`:1766`, `:1796`), which returns `None` — read as `continue` by callers — so
individual `k_scale`/`v_scale` tensors are dropped one deduplicated warning at a
time and nothing states that zero scales were loaded in total.
*Task:* `#505-A1-04: a supplied --quantization-param-path that fails to load must raise; add a loaded-scale completeness check for fp8 KV`

**#505-A1-05 — DFLASH silently substitutes greedy verification for the
requested sampler.** `dflash_worker_v2.py:1859`, inside a method named
`_validate_phase1_sampling_support`, warns once on rank 0 at the first
non-greedy request and then verifies with argmax. The served distribution is not
the requested one and the output is plausible either way. The root is
`dflash_utils.py:44-47`: a bare `except Exception` around the `sgl_kernel`
import sets `_DFLASH_SAMPLING_VERIFY_AVAILABLE = False` with no log at all, so an
ABI/wheel mismatch (a known trap in this tree) is indistinguishable from a
device that genuinely has no kernel.
*Task:* `#505-A1-05: refuse non-greedy DFLASH requests when the sampling-verify kernels are absent, and log why the sgl_kernel import failed`

**#505-A1-06 — the "unmatched GGUF tensor is fatal" argument is written down
once and enforced in four files.** `deepseek_v4.py:3209-3231` states the general
principle — every GGUF tensor is mapped by an explicit table, so an unmatched
name is a mapping defect, and #391 walls 10-12 are what it costs when it only
warns — and then applies it only in `deepseek_v2.py`, `deepseek_v4.py`,
`llama4.py` and `deepseek_common/utils.py`. `qwen3_5.py:1593` / `:1815` /
`:1957` / `:2340`, the fork's flagship GGUF bring-up across four classes, still
does `logger.warning(...)` + `continue`. So does `gemma4_causal.py`, whose
parameter-side check exists but only logs (`:1437-1442`).
*Task:* `#505-A1-06: hoist _unmatched_gguf_tensor_message to the loader so an unmatched GGUF tensor is fatal for every model, not four`

**#505-A1-07 — an incomplete physical-isolation audit is reported as a
warning.** `adaptive_graph_memory.py:1032`. Twelve lines above, the same audit
raises `RuntimeError` when a tagged buffer is attributed to a FOREIGN build
window, with the reason spelled out: "Pausing one state would unmap another
state's memory." The unattributable case — where the audit could not decide — is
logged and the run continues under the mode the audit exists to license. Per the
SUCCESS-CLAIMS law, an instrument that could not discriminate has not returned a
pass.
*Task:* `#505-A1-07: an unattributable tagged buffer must fail the adaptive-graph-memory isolation audit, not warn`

**#505-A1-08 — a target model that loaded nothing is a warning; a draft that
loaded nothing is an error.** `gemma4_causal.py:1437-1442` computes exactly the
right set (`params_dict.keys() - loaded_params`) and routes real, unwritten
parameters to `logging.WARNING`. The argument the draft guard makes at
`weight_utils.py:2039-2046` — "the drafter then runs on `torch.empty` … the only
symptom is an accept rate that looks like a weak drafter" — transfers with the
symptom changed, not the mechanism: a target model with unwritten parameters
produces wrong logits and there is no accept rate to look weak. The escape hatch
already has a pattern (`SGLANG_ALLOW_UNLOADED_DRAFT_PARAMS`, `environ.py:243`).
*Task:* `#505-A1-08: unwritten real parameters must fail the target load too, with the same named escape env as the draft guard`

**#505-A1-09 — an unsupported quant scheme returns an unquantized method over a
quantized checkpoint.** `npu_mxfp4.py:115-122` is the clearest instance and its
message is a false state claim ("MoE weights will run in full precision
(BF16/FP16)" — there are no bf16 MoE weights in an MXFP4 checkpoint). The layer
then declares dense parameters, the packed tensors match nothing, and they are
dropped by the model's own `not found in params_dict` warning: the occasion bug
assembled from two sites, neither of which is wrong on its own. Same shape at
`modelslim.py:212` and `:240`. Reach is NPU-only, which is why it is ranked
here rather than higher. The correct handling is three lines above it, at
`npu_mxfp4.py:108`: `raise NotImplementedError` for the same missing kernel on
the linear branch.
*Task:* `#505-A1-09: an unsupported quant scheme must refuse, not hand a packed checkpoint to an unquantized method`

### Note on what this part does NOT claim

The 140 unopened sites are not certified benign. The largest coherent gap is
`layers/quantization/` outside the C1 string family (40 sites) — that is where
the packed-vs-dense family of `#443`/`#446` lives, and it deserves the same
sweep. `model_loader/ci_weight_validation.py` (18 sites) is a CI utility and was
deliberately deprioritised. No DANGEROUS row above rests on a context-only read,
a docstring, or a comment; where a comment is the only evidence for an
invariant, the row says so explicitly (`eagle_utils.py:193`,
`adaptive_graph_memory.py:1032`, the `dflash_worker_v2` fused-KV fallbacks).

---

# Part A2 — warn-and-continue: memory / managers / distributed / pool


Desk audit, nothing executed, no GPU. Base commit `d653405223`, branch
`docs/silent-wrongness-505`, worktree `/spinning/wt-505-silent`. Method copied
from `AUDIT_500_mechanism_reach.md` §§1-6: every row cites `file:line` and
quotes the operative line verbatim; a row that rests on a comment or docstring
rather than executed code says so.

Scope split: sub-auditor A1 holds `model_loader/`, `speculative/`,
`layers/moe/`. This part holds `mem_cache/`, `managers/`, `model_executor/`,
`distributed/` (incl. the barlink family), `layers/dcp/`, `memtier/`,
`disaggregation/`.

## The class of defect

A site DETECTS an anomaly, logs a WARNING (or nothing at all), and CONTINUES,
where the resulting state is silently WRONG rather than merely degraded. The
bar for **DANGEROUS** is a concrete answer to: *what exactly would be silently
wrong, and through which observable would a user NOT notice it?* Sites that
cannot answer that concretely are **BENIGN** — the list below is deliberately
short rather than padded.

- **BENIGN** — genuine compat shim, optional feature absent, cosmetic, or a
  path with no production reach.
- **DEGRADED-LOUD** — behaviour changes, but visibly, and the warning names it
  accurately.
- **DANGEROUS** — the wrongness stays silent.

## Coverage

| dir | grep total | reviewed at message level | opened at source |
|---|---|---|---|
| `mem_cache/` | 96 | 96 | 8 |
| `managers/` | 84 | 84 | 11 |
| `model_executor/` | 59 | 59 | 6 |
| `distributed/` | 86 (77 in `device_communicators/`) | 86 | 19 |
| `layers/dcp/` | 1 | 1 | 1 (+ `collective_guard.py` read whole) |
| `memtier/` | 1 | 1 | 1 |
| `disaggregation/` | 83 | 83 | 4 |
| **total** | **410** | **410** | **50** |

Extraction: `grep -rn "logger\.warning\|warning_once\|warnings\.warn\|logger\.warn(" --include=*.py <dir>`
(counts above), each re-run with `-A3/-A4` so all 410 were read with their
message text and the immediately following statement. 50 were then opened in
their surrounding logic. A second sweep extracted the *silent* shapes
(`except Exception:` followed by `pass`/`continue`/`return None`) across the
same directories: 51 hits, 8 opened.

**Not reached** (stated as a gap, not a clean bill of health):
- `mem_cache/storage/{flexkv,eic,hf3fs,nixl,aibrix_kvcache,simm,umbp,lmcache}`
  — ~40 warning sites, third-party storage backends, read at message level
  only. They are I/O-failure→cache-miss shapes, which is why they were
  deprioritised, but none was opened.
- `disaggregation/{mooncake,mori,nixl,ascend,fake}` — 79 of the 83
  disaggregation hits. Only `topology.py` was opened. The mooncake/mori KV
  transfer failure paths are the plausible place for a second finding of the
  A2-04 shape and were NOT audited.
- `managers/` multimodal, tokenizer, and detokenizer sites (~30) — read at
  message level, not opened.
- `model_executor/` NPU/Ascend and CPU paths.
- The `except: pass` sweep covered only bare `pass/continue/return None`;
  fallbacks that substitute a *default value* were followed only where a
  warning already pointed at them.

## Table

| file:line | site (symbol) | class | what would be silently wrong | fix pattern |
|---|---|---|---|---|
| `managers/kv_session_offload.py:2204-2207` | `KVSessionOffloadManager.__init__`, spec-in-tick scratch reservation | **DANGEROUS** | `self.allocator.size -= int(self.mtp_resident_slices)` inside `try: … except Exception: pass`. On the composite allocators the setter is a NO-OP (`@size.setter … pass`, `mem_cache/multi_ended_allocator.py:1764-1766` and `:1974-1976`), so the subtraction silently does nothing — no exception, nothing to catch. The slots ARE out of circulation (`self.allocator.alloc(...)` at `:2195` succeeded) but the pool still counts them, so the invariant the comment at `:2198-2203` names ("available + evictable + protected + session_held + uncached == total, total = allocator.size") is permanently off by `mtp_resident_slices`. `logger.info("… reserved %d draft-read scratch slots …")` at `:2208` then reports success for the half that failed. | (iii) independent state probe: read `allocator.size` back after the write and refuse by name when it did not move — `if self.allocator.size != _before - self.mtp_resident_slices: raise ValueError(...)`. Drop the bare `except`. The composite allocators must either grow a real carve-out API or be refused for this feature by name. |
| `model_executor/model_runner_kv_cache_mixin.py:1208-1209` | `note_post_capture_leftover` / `foreign_residue_warning` | **DANGEROUS** | `if foreign_msg is not None: logger.warning("%s", foreign_msg)` — and then the boot persists the correction anyway. The warning text itself states the consequence: *"the correction persisted for the next boot will under-book the KV pool"* (`:1078-1079`). Nothing in the written cache records the pollution (`grep -n foreign` over the file returns only `:1198-1209`), and the reader (`:896-910`) validates only shape and `mlp_vector`. So the wrongness materialises in the NEXT boot, in a different log, as a smaller `max_total_num_tokens` with provenance "cached" and no reason attached. | (i) named hard error / refusal at the site: do not persist when `foreign_b > 0` unless `SGLANG_MEASURED_KV_BUDGET_CTX_ALLOWANCE_MIB` was raised deliberately; alternatively write `"provenance": "polluted"` into the cache and have the reader refuse it by name the way `:908` already refuses `malformed`. |
| `managers/kv_session_offload.py:3426-3434` | `KVSessionOffloadManager.try_spill` | **DANGEROUS** | `if n_own > self.region_tokens:` warns and `return False` → the caller falls back to stock retraction (`managers/scheduler.py:4051`). `n_own` is `int(dev_idx.numel())` from `owned_device_indices(..., lo=self.lo, hi=self.hi, dcp_rank=self.dcp_rank)` (`:3415-3425`) — a RANK-LOCAL quantity under the weighted uneven-DCP owner rule, which is the whole point of `--rank-kv-ratio`. Every neighbouring decline in the same function is annotated rank-uniform (`":3344 Rank-uniform: the overhang count is replicated"`, `":3400 Replicated inputs -> rank-uniform verdict"`); this one is not, and it is the only one whose input is per-rank. A divergent spill/retract verdict means the ranks build different `ScheduleBatch`es — the file's own doctrine at `:501` is *"RANK-UNIFORMITY (this fork: divergence == NCCL hang, not a wrong number)"*. The identical condition in the sibling prefill-spill path RAISES: `raise RuntimeError("… needs {n_own} host rows > region {self.region_tokens}; the admission gate should have rejected it.")` (`:3874-3880`). | (i) raise like the `:3874` twin — the region is sized `max_ratio`-wide precisely so this cannot happen (`:3013 "every region is max_ratio-sized (holds any rank's per-rank shard of a full-context session)"`), so a hit is a broken invariant, not a runtime condition. If a soft decline is wanted, min-reduce it exactly as the sibling module already does: `reduce_extra()` → `transfer_verdict(min_done, min_ok, abandoned)` (`managers/kv_session_spill_destination.py:621-626`, `:309-330`). |
| `disaggregation/topology.py:421-427` | `_card_totals_mib` (CUDA→NVML bridge) | **DANGEROUS** | `except Exception as exc: logger.warning("PD topology: CUDA->NVML index bridge unavailable (%s); assuming identical enumeration orders.", exc)` then `mapping = {}` → `reindex_totals_cuda_order(nvml_totals, {})` is the identity. CLAUDE.md: *"Device identity strictly via the IdentityMap … torch order != NVML order on this rig"*. With the identity assumption, each card's VRAM total is attributed to the wrong rank on any rig where the orders differ — which is this rig. The only downstream guard is itself warn-only (`for warning in check_vram_feasibility(plan): logger.warning("PD topology: %s", warning)`, `:742-743`), so a plan is admitted whose feasibility was computed against the wrong card. Observable: none — the PD servers boot and OOM later, or silently get a smaller pool. The function's own docstring demands the opposite for the sibling case: *"Returns None when NVML is unusable; the caller must then say so instead of silently skipping"* (`:389-391`). | (i) return `None` here too (the docstring's contract), so the caller reports "no card totals" instead of fabricating them. AUDIT_331 is the precedent; `registry/nvml.py`'s IdentityMap is the sanctioned source. |
| `distributed/device_communicators/barlink.py:384-400` | `_build_transport` | **DANGEROUS** (unproven direction; see note) | `except Exception as e: … logger.warning("barlink: group %r does NOT get the requested transport %r …")` then `return None` → the gloo plane. The decision is per-rank and there is NO group agreement step: `_NO_FALLBACK = frozenset({"device", "host"})` (`:327`) deliberately excludes `bar1` and `matrix`, so those two are exactly the transports that may fall back on one rank alone. A one-rank fallback is the `#94/#194/#312/#431` family. Honest qualifier: bar1 bring-up itself contains collectives (`dist.all_gather_object(...)` at `barlink_bar1.py:1953`), so a failure raised BEFORE that point desyncs into a hang rather than a wrong number; a failure after it, or in `matrix`'s own probe path, leaves the group split with no diagnostic. Nothing in the module reconciles the outcome. | (i)+(iii) the in-tree pattern already exists twice: `parallel_state.py:975-992` all-gathers the local availability of custom all-reduce and disables it on EVERY rank when it diverges; `model_executor/model_runner.py:1365-1369` does the same for CUDA-graph plans. Apply that shape to `_build_transport`'s result before the communicator is used. |
| `distributed/device_communicators/barlink_path_dispatcher.py:468-478` | `refine_transport_choice` | **DANGEROUS-latent, reach zero today** | `if hint == HINT_GLOO: return None` demotes one message class to the gloo plane on a per-rank decision; the module docstring only *asserts* the saturation sensor "MUST be group-uniform" (`:246-248`) — a comment, not a predicate. Reach today is zero: `dispatcher_enabled()` reads `SGLANG_BARLINK_PATH_DISPATCHER` default `"0"` (`:428`) and a fresh dispatcher has an empty registry, so every decision is `status_quo` (`:431-443`, and catalog §7 says the same). Recorded because the wiring slice would activate it. | (i) when #279's measured slice lands, the decision must be reduced across the group (or derived only from replicated inputs) before it can change a transport; the `barlink_uniformity.first_divergence` recorder (`barlink_uniformity.py`) is the standing instrument for exactly this and is OFF by default (`:205`, #500-B20). |
| `model_executor/runner/decode_cuda_graph_runner.py:1243-1245`, `:1263-1265` | kvso spill-graph / C4 attention selftests | **DANGEROUS (instrument)** | `logger.warning("kvso spill-graph selftest raised (ignored): %r", _sess_e)` — a self-test whose exception is swallowed. CLAUDE.md rule (2): an instrument's verdict counts only after it passes a can-discriminate check. Here a *crashing* instrument is indistinguishable from a *passing* one in every downstream consumer: capture proceeds, and the only trace is one WARNING in a worker log — and `model_executor/forward_peak.py:14-17` records that worker `logger.warning` lines provably do not reach the server log on this rig. | (iii) the selftest must either raise (it is a gate) or record a machine-readable NOT-RUN verdict next to its PASS/FAIL, on the channel that has evidence behind it (the per-rank JSON dump named in `forward_peak.py:16-17`), so "no verdict" is never read as "passed". |
| `mem_cache/multi_ended_allocator.py:297-313` | `SubPoolAllocator.restore_state` | **BENIGN today (comment-asserted invariant, no production caller)** | `logger.warning("MultiEndedAllocator.restore_state: %d relocation(s) recorded inside a backup window … Eager compaction is not fully reversible; SGLang's spec path should not produce a free() inside a backup window.")` then `del self._inverse_history[n_inverse:]` — the records are DISCARDED and the relocations are not undone, so the rolled-back watermark would describe pages that moved. The invariant is asserted in a comment only (`:289-290 "Spec-decode allocates only inside a backup window (no free), so `_inverse_history` doesn't grow under correct usage."`). Reach: `grep -rn "backup_state=True"` over `srt/` returns nothing and the only `restore_state` callers outside `mem_cache/` are `hardware_backend/npu/dsv4/dsv4_allocator.py:727-735`, so no CUDA production path reaches it. | Per CLAUDE.md rule (3) the comment is a TESTABLE CLAIM: pin it with a unit test (a `free()` inside a backup window must be impossible), or convert the warning into a raise now that nothing can trip it. |
| `mem_cache/unified_radix_cache.py:2044-2055` | `write_backup_storage` (uneven-DCP owner mask) | **DEGRADED-LOUD** | `if device_value is None: logger.warning("[uneven-dcp hicache] skipping storage backup of node %d: device indices gone, page ownership unknown."); return`. The rank-local skip is argued at the site: *"Skipping is safe: batch_exists prefix semantics just truncate at the first missing page, identically on every rank (rank-shared file names)"* (`:2046-2049`). The whole node is skipped, so no partially-written rank-shared page is produced — the argument holds as written. | none; keep. (Rests on the comment for the `batch_exists` truncation semantics, which was not verified in the storage backend — flagged.) |
| `managers/kv_session_spill_destination.py:740-749` | `SpillDestinationController._worker_loop` | **DEGRADED-LOUD (exemplar)** | `logger.warning("kv-session-offload destinations: %s transfer of rid=%s failed on rank %d: %r", …); ok = False` — a per-rank I/O failure, but `ok` is published into `_io_ok` (`:756`) and MIN-reduced group-wide before any verdict (`reduce_extra`, `:621-626`; `transfer_verdict`, `:309-330`). A single rank's failure fails the transfer on every rank deterministically. | none — this is the pattern the two DANGEROUS rank-local rows above should copy. |
| `distributed/parallel_state.py:983-992` | `_reconcile_custom_allreduce` | **DEGRADED-LOUD (exemplar)** | *"Custom allreduce enablement diverges across group '%s' … disabling it on every rank so no collective is gated on a rank-local state."* Preceded by `torch.distributed.all_gather_object(gathered, bool(local_ok), group=self.cpu_group)` (`:976-978`). | none — cite as fix pattern. |
| `model_executor/model_runner.py:1365-1369` | CUDA-graph plan reconciliation | **DEGRADED-LOUD (exemplar)** | *"CUDA graph plan differs across the group for the %s phase (ranks resolved %s); disabling it on every rank so the collective sequence stays rank-uniform."* | none — cite as fix pattern. |
| `layers/dcp/collective_guard.py:111-122`, `:131-136` | `guard_dcp_step` | **DEGRADED-LOUD (exemplar)** | A count divergence raises `RuntimeError("weightless-KV anti-hang guard: DCP collective COUNT divergence …")` behind a bounded `monitored_barrier`; an order/type divergence raises after an `all_gather` of the step signature. Enabled only for the weightless lane and turned off around weightless decode graphs (`model_runner.py:4121 set_guard_enabled(not _wl_decode_graphs)`). | none — the in-tree instrument for the rank-local family; the A2-03/A2-05 sites are outside its reach. |
| `distributed/device_communicators/barlink.py:644-654` | `BarlinkCommunicator._select` fallback notice | **DEGRADED-LOUD** | Once per `(op, size class)`: *"… does NOT cover %r at %d bytes -- falling back to the host-staged layer … this run's numbers for this size are NOT %s numbers."* `handles()` is argued rank-uniform at `barlink_bar1.py:794` ("depends only on group-uniform state") and the a2a size check is explicitly widened to the group max (`barlink.py:1134-1137`). | none. (The rank-uniformity of `handles()` is a comment; `barlink_bar1.py:689-691` concedes *"rank-uniform, and nothing enforces that"* for the neighbouring case.) |
| `distributed/device_communicators/barlink_matrix_transport.py:316-325` | `window_for` clipping | **DEGRADED-LOUD** | Clips a group's BAR1 window and says payloads above it "fall back to the gloo layer without further notice". Per-rank clipping is reconciled by the group-wide MINIMUM at `barlink_bar1.py:1948-1978` (`common = min(proposals)`, `raise Bar1Unavailable` at 0). | none. |
| `distributed/device_communicators/barlink_matrix_transport.py:218-224` | `bar1_free_for` NVML identity | **DEGRADED-LOUD** | `except DeviceOrderUnresolvedError` → window sized from the sysfs GROSS aperture, provenance string `"sysfs-gross"` returned so the caller can see it. Contrast `disaggregation/topology.py:422`, which fabricates an identity mapping instead. | none — this is the correct shape of the A2-04 defect. |
| `distributed/parallel_state.py:735-743` | barlink ACHIEVED notice | **DEGRADED-LOUD** | Reports `requested` vs `ACHIEVED` and states the measurement is mixed. It is the *report* of the `barlink.py:393` fallback, not a second decision. | see A2-05 (the missing piece is agreement, not reporting). |
| `model_executor/model_runner_kv_cache_mixin.py:908-910` | measured-KV-budget cache read | **DEGRADED-LOUD** | `logger.warning("Measured KV-budget cache %s is malformed; ignoring.")`, `provenance = "malformed"`, `return 0` — a named provenance the rest of the path can read. | none — this is the shape A2-02's writer should adopt. |
| `model_executor/model_runner_kv_cache_mixin.py:4966-4972` | `--rank-kv-ratio speed` degradation | **DEGRADED-LOUD** | Falls back when no bandwidth scores exist; catalog §1 documents the degradation (`server_args.py:9615`). | none. |
| `model_executor/pool_configurator.py:1018-1023` | `--swa-pool-sizing` no-op notice | **DEGRADED-LOUD** | A user-set cap is ignored, and the line says so and names the selected configurator. | none. |
| `memtier/profile_store.py:225-229` | `PROFILE_TRUST_ENV` override | **DEGRADED-LOUD** | *"the profile's numbers are being read on hardware that did not produce them"* — the operator asked for it via an env var; the line names the consequence. | none. |
| `mem_cache/mamba_radix_cache.py:577-584`, `:730-737` | mamba checkpoint interval | **DEGRADED-LOUD** | Off-grid tracked state is dropped (`cache_len = 0`) / the request is not cached. Both refuse to *store* rather than storing a mismatched pair; `:596-600` argues why rounding down would be wrong. | none. |
| `mem_cache/hiradix_cache.py:1188-1192`, `:1301-1307` | `write_back` drop / `load_back` failure | **DEGRADED-LOUD** | Host-pressure drop and load-back failure both degrade to a cache MISS; correctness of the served tokens is unaffected. | none. |
| `managers/kv_session_offload.py:2219-2231` | spec-in-tick scratch unavailable | **DEGRADED-LOUD** | Spilled sessions fall back to the plain host tick; the reason and the flag to set are named. | none. |
| `managers/kv_session_offload.py:3311-3321`, `:3372-3382` | `try_spill` snapshot / bookkeeping declines | **DEGRADED-LOUD** | Both decline on REPLICATED inputs (annotated at `:3344`, `:3400`) and fall back to stock retraction. | none. |
| `managers/kv_pressure_runtime.py:500-504` | dcp_ratio rung stays planned-only | **DEGRADED-LOUD** | The rung is not armed and the operator is told which vectors to declare. A refusal, not a silent flip. | none. |
| `distributed/parallel_state.py:817-820`, `833` | custom / quick all-reduce setup failure | **BENIGN** | Optional accelerator absent; `:983` reconciles the group afterwards. | none. |
| `distributed/parallel_state.py:2636-2639`, `:2671-2677`, `distributed/utils.py:49-52` | global TCPStore | **BENIGN** | NIXL-only coordination; consumers check for `None`. | none. |
| `distributed/parallel_state.py:1656-1659`, `:3460-3462` | world-size-1 in-place all-gather, torch<2.5 host cache | **BENIGN** | Efficiency note; version shim. | none. |
| `distributed/device_communicators/{custom_all_reduce_utils,quick_all_reduce,pymscclpp,torch_symm_mem,triton_symm_mem_ag}.py` (17 sites) | import/probe failures | **BENIGN** | Optional dependency or capability absent; the feature disables itself uniformly (import-time, hence identical on every rank). | none. |
| `distributed/device_communicators/barlink_{bar1,ucx,host,shm,device}.py` teardown sites (`bar1:746`, `:4111`, `:4225`, `ucx:1983`, `host:1172`, `shm:72/79`, `device:822`) | close/release/unregister failures | **BENIGN** | Teardown after the real error; `bar1:465` says so verbatim (*"teardown must never mask the real error"*). | none. |
| `distributed/device_communicators/barlink_liveness.py:142`, `barlink_abort_gate.py:106` | env parse | **BENIGN** | Malformed env value → documented default, value echoed. | none. |
| `mem_cache/storage/**` (~40 sites) | third-party backends | **BENIGN (message level only)** | I/O failure → cache miss. Not opened — see coverage gap. | none. |
| `disaggregation/topology.py:398-399`, `:407-408` | NVML unavailable / query failed | **BENIGN** | Returns `None`; the caller must report it (docstring `:389-391`). | none. |
| `layers/dcp/comm.py:97-101` | deprecated-name shim | **BENIGN** | `DeprecationWarning` for a renamed symbol. | none. |

## Top findings

**#505-A2-01 — the kvso draft-scratch carve-out silently does nothing on hybrid
(mamba/SWA) allocators, and then logs success.**
`managers/kv_session_offload.py:2204-2207` writes `self.allocator.size -=
int(self.mtp_resident_slices)` inside `try: … except Exception: pass`. On
`UnifiedMambaAlloc` / `UnifiedSWAAlloc` — the allocators for exactly the hybrid
GDN models this fork serves — `size` is a computed property whose setter is
`pass` (`mem_cache/multi_ended_allocator.py:1764-1766`, `:1974-1976`). The
write is therefore a no-op with no exception, the reserved slots stay counted
in the pool's advertised capacity while being permanently out of circulation,
and `:2208` logs *"reserved %d draft-read scratch slots"* either way. The
comment at `:2198-2203` states the intent explicitly ("so the
SchedulerInvariantChecker … stays balanced"), so the failure is a broken stated
invariant, not an unstated assumption. Three of this project's standing laws
converge here: a success message treated as evidence, a returned/derived value
whose contract lives in a comment, and pool accounting that stays wrong
forever. Reach qualifier, stated honestly: the whole spec-in-tick arm sits
behind `KVSO_ALLOW_SPEC=1` (`server_args.py:6580`, recorded as #500-B10), so a
stock speculative boot does not reach it today.
*Task:* `#505-A2-01: kvso draft-scratch reservation must verify the pool carve-out took effect (composite allocators no-op the size setter)`

**#505-A2-02 — a foreign-residue-polluted KV budget correction is persisted and
is unmarked on read.**
`model_executor/model_runner_kv_cache_mixin.py:1208-1209` warns with a message
that itself predicts the harm — *"the correction persisted for the next boot
will under-book the KV pool"* (`:1078-1079`) — and then persists it. Nothing is
written into the cache to mark the reading as polluted, and the reader
(`:896-910`) checks only shape and `mlp_vector`. The damage lands in the NEXT
process, in a different log, as an unexplained smaller KV pool. The same file
already has the right shape one screen up: a malformed cache is refused by name
with `provenance = "malformed"`.
*Task:* `#505-A2-02: refuse to persist (or mark "polluted") a measured KV-budget correction taken over foreign device residue`

**#505-A2-03 — `try_spill` declines on a rank-LOCAL quantity where its own
sibling raises.**
`managers/kv_session_offload.py:3426` compares `n_own` — derived from this
rank's owner window `lo/hi` under weighted uneven DCP — against the replicated
`region_tokens`, warns, and returns `False`, which sends only this rank down
the stock-retraction path (`managers/scheduler.py:4051`). Every other decline
in the function is annotated as resting on replicated inputs; this one is not.
The prefill-spill twin raises a `RuntimeError` on the identical condition
(`:3874-3880`), and the module header states the doctrine that makes the
asymmetry a defect: *"RANK-UNIFORMITY (this fork: divergence == NCCL hang, not
a wrong number)"* (`:501`). Cheap fix, exact in-tree precedent: raise like the
twin, or min-reduce the verdict the way `kv_session_spill_destination.py`
already does for its I/O flags.
*Task:* `#505-A2-03: try_spill must not decline on a per-rank owned-row count — raise like the PS2 twin or min-reduce the verdict`

**#505-A2-04 — PD topology fabricates a CUDA→NVML identity mapping when the
bridge is unavailable.**
`disaggregation/topology.py:421-427` logs *"assuming identical enumeration
orders"* and sets `mapping = {}`, i.e. the identity — on a rig where CLAUDE.md
records that torch order != NVML order. Card VRAM totals are then attributed to
the wrong ranks, and the only downstream check is warn-only
(`check_vram_feasibility`, `:742-743`), so an infeasible plan boots. The
function's own docstring demands `None` + a caller-side report for the sibling
NVML failure, and `barlink_matrix_transport.py:218-224` shows the correct shape
(degrade to a NAMED provenance, `"sysfs-gross"`, never to a fabricated
identity).
*Task:* `#505-A2-04: PD topology must return None when the CUDA→NVML bridge is unavailable instead of assuming identical order`

**#505-A2-05 — barlink `bar1`/`matrix` fall back to gloo per rank with no group
agreement.**
`distributed/device_communicators/barlink.py:384-400` catches any bring-up
exception, warns, and returns `None` for the two transports deliberately
excluded from `_NO_FALLBACK` (`:327`). No step reconciles the outcome across
the group, so a probe that fails on one card only (BAR1 peer mapping, the
`dmabuf_holder` guard, the byte proof — `barlink_bar1.py:589`, `:597`,
`:4644-4656`) leaves the group split. Direction of failure is unproven and is
reported as such: a failure before `dist.all_gather_object` at
`barlink_bar1.py:1953` desyncs into a hang, one after it does not. The fix is
one collective, and the tree already contains it twice for precisely this class
(`parallel_state.py:975-992`; `model_runner.py:1365-1369`). This also touches
the CLAUDE.md barlink-default order: a group silently on gloo is a
NCCL/gloo run reported as a barlink run.
*Task:* `#505-A2-05: reconcile the barlink transport outcome across the group before use (all-gather like custom-allreduce does)`

**#505-A2-06 — two selftests in the decode graph runner swallow their own
exceptions.**
`model_executor/runner/decode_cuda_graph_runner.py:1243-1245` and `:1263-1265`
log *"selftest raised (ignored)"* and continue. A crashing instrument is
indistinguishable from a passing one downstream, and worker-process WARNING
lines are documented in this tree as not reaching the server log
(`model_executor/forward_peak.py:14-17`).
*Task:* `#505-A2-06: kvso spill-graph / C4 selftests must record a NOT-RUN verdict on an evidenced channel, not a swallowed warning`

**#505-A2-07 (registration, not a bug) — `MultiEndedAllocator.restore_state`
carries a comment-asserted invariant with a warn-and-continue arm and no
production caller.**
`mem_cache/multi_ended_allocator.py:288-313`. Per CLAUDE.md rule (3) the
comment at `:289-290` is a testable claim; the arm is currently unreachable
(`backup_state=True` has no caller in `srt/`), which is itself worth pinning
before some future spec path reaches it.
*Task:* `#505-A2-07: pin "no free() inside a backup window" with a test, or make restore_state raise now that nothing trips it`

## Fix-pattern references used above

- (i) named hard error at the site — precedent in this scope:
  `managers/kv_session_offload.py:3874-3880`,
  `layers/dcp/collective_guard.py:116-122`.
- (ii) completeness check in the #496-(b) shape —
  `_assert_required_params_loaded` is defined at
  `python/sglang/srt/models/deepseek_v4_dspark.py:868` and called at `:864`. It
  is the model-loader analogue: enumerate what MUST have been written and raise
  naming the missing entries. The A2-01 row is the same defect in the pool
  ledger and wants the same shape (assert the ledger moved by the amount that
  was carved out).
- (iii) independent state probe — `kv_session_spill_destination.py:621-626` +
  `:309-330` (min-reduced flags instead of a per-rank success belief);
  `distributed/parallel_state.py:975-992` (all-gather the local capability
  before letting it gate a collective).

## Catalog sections read

`CLAUDE.md` in full (MECHANISM REACH, REACH INCLUDES PARAMETERS, SUCCESS CLAIMS
ARE NOT EVIDENCE, the rank-uniformity and device-identity rules, the barlink
default order). `docs/dev/FEATURE_CATALOG.md` §1 (uneven parallelism, incl. the
two attention/KV distribution axes), §3 (memory tiers / offload / spill), §5
(multi-group runtime), §6 (weightless KV lane), §7 (collectives / transport),
§12 (robustness canon, incl. the #404 bookkeeping-owner register), §17
(combination matrix + eviction doctrine).
`docs/dev/AUDIT_500_mechanism_reach.md` §§1-6 (classification method, verbatim
predicate quoting, honest coverage statement, bug-candidate write-up shape).

---

# Part B — comment invariants as testable claims


Desk audit plus two executed hermetic falsifiers (`CUDA_VISIBLE_DEVICES=99`,
`PYTHONPATH=/spinning/wt-505-silent/python`, interpreter
`/spinning/htsglang-gpu/.venv/bin/python`). No GPU held, no server booted, no
`.py` file changed. Base commit `d653405223` on `docs/silent-wrongness-505`.

The axis is CLAUDE.md's third SUCCESS-CLAIMS law, verbatim:

> (3) a code comment asserting an invariant ("leaves no partial state",
> "never reached") is a TESTABLE CLAIM — pin it with a test or treat it as
> a bug report against the comment.

The occasion is #501: a comment asserting *"a decline leaves no partial
state"* while the code above it had already committed an irreversible
reclaim. Fix `1f2b24ef11` exists on a sibling branch and is **not** an
ancestor of this audit base (`git merge-base --is-ancestor 1f2b24ef11 HEAD`
→ false), so #501 is still live in the tree this sweep read. That makes it
the calibration row, not a new finding — see §"Calibration".

Method template: `docs/dev/AUDIT_500_mechanism_reach.md` §§1-6 — verbatim
quoting, `file:line` on every row, four honest verdict classes, coverage
gaps stated rather than papered over. Catalog sections read:
`FEATURE_CATALOG.md` §1, §3, §4, §6, §7, §12, §17, plus §0's registry rule.
`CLAUDE.md` read in full.

### Coverage + triage rule

**Grep total (this base, the briefing's extraction verbatim):**

```
cd python/sglang/srt && grep -rniE "(#|\"\"\"|').*\b(never|always|must |guaranteed|\
leaves no|cannot happen|can not happen|safe because|invariant|by construction|\
impossible|is not possible|no partial|atomic|exactly once|only ever)\b" \
  --include=*.py mem_cache managers model_executor speculative distributed \
  layers/dcp layers/moe memtier model_loader
```

**1345 hits**, not the ~1111 the briefing estimated (the briefing's per-dir
figures are all ~12 % lower; the base differs). Per directory, measured:

| dir | hits | briefing said |
|---|---|---|
| model_executor | 310 | 259 |
| managers | 286 | 251 |
| speculative | 192 | 174 |
| distributed | 175 | 141 |
| mem_cache | 171 | 135 |
| layers/moe | 135 | 97 |
| model_loader | 41 | 33 |
| memtier | 21 | 17 |
| layers/dcp | 14 | 4 |

**Triage rule, applied mechanically, then by hand.** A claim is KEPT only if
it asserts something about **STATE or ORDERING that other code relies on**.
Discarded without individual inspection: English-usage "must"/"always" in
docstrings ("the caller must pass a tensor"), restatements of the line below
them, prose in module headers with no referent, and every `must` that is an
argument-validation message rather than an invariant.

The 1345 were not read one by one. They were reduced by intersecting the
invariant vocabulary with the four priority vocabularies the briefing ranks,
each grep run over the same nine directories:

| priority | pattern axis | candidate lines |
|---|---|---|
| 1 | rollback / revert / restore / cleanup / on-failure / decline / abort / partial | 27 |
| 2 | all ranks / every rank / only rank 0 / lockstep / rank-uniform | 31 |
| 3 | alias / caller owns / safe to reuse / scratch / in-place / view of | 7 |
| 4 | lock held / reentrant / runs exactly once / idempotent / single-threaded | 33 |
| 4b | never None / always set / never empty / is never / always returns | 43 |
| 4c | "must run/be called/happen/precede … before\|after\|first\|last" | 13 |
| | **candidate union** | **154** |

Plus one AST pass (below) over `managers`, `mem_cache`, `model_executor`,
`layers/moe`, `speculative`, `memtier`, `distributed` that pairs an
invariant comment inside a function with an irreversible-looking call
followed by a decline (`return None/False`) or a `raise` — the mechanical
form of the #501 shape. Narrow vocabulary: 3 hits. Broad vocabulary: 43
hits, mostly heuristic noise.

- grep total: **1345**
- candidate union after the priority intersections: **154**
- sites **opened and read in full context**: **41**
- claims **KEPT** (load-bearing, verdict assigned): **22**
- **triaged out: 1323** (1345 − 22), of which 1191 never entered a candidate
  list and 132 were candidates that on reading restated the adjacent line or
  carried no state/ordering assertion.

**Not reached** (stated, not implied): `model_loader` (41 hits — only
`weight_utils.py:571` opened), `memtier` (21 hits — the registry's
provenance claims scanned but none opened), `layers/dcp` (14 hits, none
opened), `mem_cache/storage/**` (flexkv / lmcache / mooncake / umbp
sub-backends), and all of `managers/scheduler.py`'s 100+ hits beyond the
four opened. `speculative/` was sampled through the priority-2 list only.
A shallow all-clear over those is exactly what this task exists to prevent,
so they are named as unswept rather than as clean.

### Calibration — the instrument re-finds #501 without being told where

The AST pass, run with no file list and no knowledge of #501:

```
python/sglang/srt/managers/kv_session_offload.py try_spill invcomment@3397
   first muts: [(3358,'free'), (3437,'pop'), ...]  declines after: [3382, 3394, 3411, 3434]
```

Exactly one hit in the narrow vocabulary, and it is #501: the allocator
`free` at `:3358` precedes four `return False` declines, under a comment at
`:3397-3399` that claims the opposite. The sweep discriminates. Fix
`1f2b24ef11` lands this on another branch; nothing here re-opens it.

### Table CONTRADICTED (priority bug candidates)

| file:line | comment (verbatim) | contradicting code (file:line + verbatim) | failure mode |
|---|---|---|---|
| `python/sglang/srt/model_executor/offload_movement.py:916-918` | `# Error path: retrieval failed physically. The item stays`<br>`# PARKED (its bytes are still at the park target); the`<br>`# failure is reported, never swallowed.` | The `try` whose failure this handles does **not** end at the retrieval. `offload_movement.py:910-914`:<br>`elif mv.handle is not None:`<br>`    self._ops.wait(mv.handle)  # park copy must have landed`<br>`    back = self._ops.copy_in_tensors(mv.handle)`<br>`    self._ops.wait(back)`<br>`    self._ops.free_destination(mv.handle)`<br>and the handler at `:919` unconditionally writes `mv.state = STATE_PARKED`, while the booking release `self.ledger.release(mv.target, mv.peer_device, mv.booked_bytes)` sits **outside** the try at `:924`. `free_destination`'s own contract (`:333-335`) is *"Release the park destination (pinned rows / peer allocation) after a completed wave-in."* | If the exception originates at `:913` or `:914` — i.e. after `copy_in_tensors` has already returned — the bytes are **not** still at the park target: they are back on the device and the destination release is half-done. The item is nevertheless marked PARKED and its park booking is never released, so (a) the memtier ledger over-books the park target forever while the item is resident, and (b) a later `wave_in` (state PARKED) re-runs `wait(handle)` → `copy_in_tensors(handle)` on a handle whose destination `free_destination` was already invoked on. **EXECUTED**, hermetic, both injection points, see below. |

**Executed falsifier for the row above** (no file written; the registered
test module's own harness driven inline):

```
fail_at=copy_in:          MovementError state=parked booked=1000
                          ops=['copy_out','wait','wait','copy_in']
fail_at=free_destination: MovementError state=parked booked=1000
                          ops=['copy_out','wait','wait','copy_in','wait','free_destination']
```

The second line is the contradiction, in the code's own op trace: `copy_in`
**and its `wait` completed**, `free_destination` was entered, and the item is
still reported `parked` with 1000 bytes still booked at the park target.

The existing test injects only at the first op —
`test/registered/unit/model_executor/test_offload_movement.py:280-289`,
`test_failed_wave_in_stays_parked_and_keeps_booking`, with
`ops = FakeDeviceOps(fail_ops={"copy_in": RuntimeError("boom")})` — which is
the one injection point for which the comment is true. Nothing injects at
`wait(back)` or `free_destination`, and `FakeDeviceOps._op`
(`offload_movement.py:489-493`) already supports both names, so the falsifier
is one dict key away.

Honest scope: the contradiction is conditional on **which** statement inside
the try raises. The comment states its claim unconditionally, and the two
statements for which it is false are the last two in the block.

### Table UNPINNED (testable-claim backlog, ranked)

Ranked by what breaks if the claim is false.

| file:line | claim (restated as falsifiable) | what breaks if false | proposed falsifier test |
|---|---|---|---|
| `layers/moe/expert_offload.py:3529-3531` — `# Host snapshot of the current resident experts [0,R) so overwriting`<br>`# buf[0:R] in place can never corrupt a not-yet-moved source.` | For every hot-set permutation, every `_src(e)` read in `_apply_hotset_freeze` resolves to host memory (`resident_host` snapshot or `old_spill`), never to a `buf` row already overwritten by an earlier iteration of the `for i, e in enumerate(hot)` loop. | Silently wrong expert weights after a heat freeze — the #302a/#443 class: fluent output, no crash, no hang. The whole zero-extra-VRAM rearrange rests on this one line. | The method carries `def _apply_hotset_freeze(self, hot):  # pragma: no cover - requires CUDA` (`:3498`) and **NO TEST** references `_apply_hotset_freeze` / `_freeze_hotset` / `freeze_from_source` anywhere under `test/`. A CPU-tensor stand-in with a permutation that maps a hot expert onto a slot whose old occupant is itself hot (e.g. `hot=[1,0]` at `R=2`) falsifies the negation directly: drop the `.to("cpu")` snapshot and the second `copy_` reads a clobbered row. |
| `model_executor/runner/decode_cuda_graph_runner.py:436-444` — `# ORDERING IS LOAD-BEARING: this must come AFTER every mutation of`<br>`# self.capture_bs (the MoE-offload cap and the weightless-KV cap above).` … `# Registering last makes`<br>`# the published set exactly the captured set` | (a) No write to `self.capture_bs` follows `_register_gguf_decode_buckets(self.capture_bs, …)` at `:445`. (b) The stated rule — *published set == captured set* — holds for every publication of `capture_bs`, not only this one. | (a) holds today (writes at `:371, :387, :422` only; `:506, :648, :1275, :1396, :1975` are reads). **(b) does not.** `KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)` at `:392` publishes **before** the weightless-KV cap at `:422` shrinks the list, so under weightless-KV × ktransformers the KT wrapper holds a strict superset. A captured graph replaying a kernel other than the one it was captured with is the exact hazard the comment names. | Two tests. (1) An AST/source ratchet asserting no `self.capture_bs` assignment appears after the `_register_gguf_decode_buckets` call in `__init__` — the same shape as `test_barlink_port.py:264-271`. (2) A constructor-level test with `SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS` and `SGLANG_WL_GRAPH_MAX_BS` both binding, asserting the value handed to `set_capture_batch_sizes` equals the final `self.capture_bs`. `test_gguf_mmq_decode_threshold.py:201` pins union-merging of the registry but nothing pins either ordering. |
| `mem_cache/hiradix_cache.py:957-960` (identically `hi_mamba_radix_cache.py:440`, `:471`; `unified_radix_cache.py:2652`, `:2682`) — `# Every rank must enter the all_reduce below; ongoing_write_through can`<br>`# diverge across ranks (e.g. write_backup returning 0 on a subset under`<br>`# host memory pressure), so a conditional skip desyncs the NCCL op`<br>`# sequence and deadlocks under TP > 1.` | No path through `writing_check` / `loading_check` reaches a `return` before `self._all_reduce(...)` on a rank-divergent condition. | A TP-wide hang — the four-incident rank-local-condition-before-a-group-collective family (#94/#194/#312/#431). | The unconditional body is correct, but `writing_check(write_back=True)` **does** return at `:945` before the collective (`# blocking till all write back complete … return`). Nothing pins that `write_back` is itself rank-uniform at every call site. Falsifier: a two-rank harness where rank 1 enters with `write_back=True` and rank 0 with `False`, asserting the decision recorder (`barlink_uniformity.first_divergence`, already the standing instrument) reports a split. Existing tests only call `writing_check(write_back=True)` single-rank (`test_hiradix_cache_unit.py:112`, `test_unified_radix_cache_unittest.py:495`), which cannot fail on the negation. |
| `managers/vram_dial.py:497-498` — `# Compute all new budgets first, validate all, then commit all --`<br>`# a multi-rank dial must not half-apply.` | For `device="all"` with one rank below its floor, **no** rank's `budget_bytes` changes and `_op_seq` does not advance. | A partially applied group dial desynchronises the replicated budget vector; the consensus round at the next boundary then reduces over ranks that disagree — the failure the `_consensus_check` MIN-reduce exists to make loud, arrived at from inside instead of outside. | `test_vram_dial.py:365-377` (`test_below_floor_rejected_with_exact_numbers`) pins the claim only for `device="rank:0"`, and `_dial_setup()` builds a **one-rank** runtime (verified by execution: `n ranks: 1`). Falsifier: a ≥2-rank runtime where rank 0's request is valid and rank 1's is below `min_viable_budget_bytes(1)`, asserting `rt._ranks[0].budget_bytes` is unchanged. `test_group_grow_commits_on_every_rank:555` covers the commit path, not the rejection path. |
| `managers/tp_worker.py:587-596` — `# ORDERING (#143): the is_verify early return must come FIRST. On a`<br>`# verify step the head skips sampling entirely … so`<br>`# it never reaches the matching send below -- a weightless worker`<br>`# that took the recv branch here would block in gloo forever.` | `if is_verify: return batch_result` (`:597-599`) precedes the weightless `broadcast_pyobj` recv branch (`:601-613`) on every reachable path. | A permanent gloo block on the weightless workers — the reported #143 symptom, and structurally the same family as the hiradix row above. | Holds at this base (`:597` is above `:601`). No test asserts the ORDER; `test_weightless_chain_spec.py` and `test_draft_solo_placement.py` exercise the paths but not the precedence. Falsifier: a source/AST ratchet asserting the `is_verify` return statement's line number is lower than the `is_weightless_worker` branch's inside `forward_batch_generation` — cheap, and it is the only thing that survives a future edit that reorders the branches. |
| `managers/kv_reshard.py:449` — `# Bounds check BEFORE any write: the fitted ceiling must hold.` and `:481-484` — `# EXCHANGE (pool still untouched): a failure up to and including this`<br>`# point aborts the attempt with the pool byte-identical -- a later`<br>`# boundary may retry. Only the WRITE phase below is the`<br>`# no-return-on-error region.` | Every `raise KvReshardError` at `:452`, `:494`, `:504` fires with the KV pool byte-identical to its pre-call state. | A failed reshard that has already scribbled rows leaves the pool in a state no retry can recover — the #501 shape at pool scale. | The claim holds by reading (`max_new_row` bounds at `:450-457`; PACK is `read_rows` only; EXCHANGE writes nothing to the pool). `test/registered/scheduler/test_kv_reshard.py` exists but nothing asserts pool byte-identity across a raising `_execute`. Falsifier: checksum the pool before, force `_exchange` to return a short payload (the `:494` arm) or a bad checksum (`:504`), assert the checksum is unchanged. |
| `distributed/parallel_state.py:1077-1080` — `# Time this collective for the per-rank compute/wait split, then`<br>`# re-enter with the clock disarmed: the body runs exactly once,`<br>`# and a collective built out of other collectives is counted once`<br>`# rather than once per level.` | The `_COLLECTIVE_CLOCK.span()` re-entry executes `all_reduce`'s body exactly once and cannot recurse a second time. | Double-counted ms/round — and the ms/round figure is this project's declared measuring stick, so a silently doubled wait term corrupts every phase verdict built on it. | No test found for `_COLLECTIVE_CLOCK` re-entry. Falsifier: arm the clock, call `all_reduce` on a 1-GPU group, assert exactly one span was opened and the body's `world_size == 1` short-circuit was hit once (a counter on the fake group). |
| `distributed/device_communicators/barlink_ucx.py:1183-1186` — `` # `_get_out_buf` always returns an ``<br>`` # `empty_like` of an already-contiguous input, so this never fires; ``<br>`# if it ever does, it fails loudly instead of returning garbage.` | `out` reaching `out.view(-1)` at `:1187` is always contiguous. | `view` on a non-contiguous tensor raises — loud, per the comment's own escape clause — so the risk here is a *false* sense of coverage, not silent wrongness. Ranked low deliberately. | True at this base: `barlink.py:834` does `inp = input_.contiguous()` before `t.barlink_all_reduce(self, inp)` (`:838`), and `_get_out_buf` is `return torch.empty_like(ref)` (`barlink.py:775`). The *freshness* half is pinned (below); the *contiguity* half is not. Falsifier: call the ucx all-reduce path with a non-contiguous input through the public seam and assert it either succeeds or raises by name, never returns an untouched tensor. |
| `layers/moe/expert_offload.py:1462-1464` — `if R != len(plan.resident_ids):  # defensive: plan invariant` | The "defensive" check can prevent the damage it names. | It cannot: the loop at `:1459-1462` has already run `out[slot].copy_(source(expert_id))` **and** `release(expert_id)` for every resident id before the check executes. A guard placed after the irreversible work is decoration — the generic form of #501, in a file that stages weights. | Falsifier: construct an `ExpertStagingPlan` with `resident_count != len(resident_ids)` and a `release` callback that records calls; assert `release` was never invoked when the `RuntimeError` fires. Fails today. (Whether such a plan is constructible is exactly what the check claims not to know — that is the argument for hoisting it above the loop, not for deleting it.) |
| `model_executor/model_runner_kv_cache_mixin.py:2007-2012` — `# #364 resident-slot cap. Applied AFTER every profiling branch and`<br>`# after the uneven-TP min-sync, so it is the last word on the pool`<br>`# geometry and cannot be undone by a branch that recomputes a size.`<br>`# Rank-uniform without a collective: it is a server arg, identical on`<br>`# every rank by construction, and it only ever lowers` | (a) nothing after `:2050` recomputes `max_mamba_cache_size`; (b) `effective_state_slots` only ever lowers. | A cap that is undone downstream re-inflates the state pool after the KV budget was already sized against the capped figure — an OOM at capture, or KV tokens silently handed back. | (b) is pinned by construction and documented at `gdn_slot_ladder.py:86-91` (`cap_is_binding` is `resident_cap < profiled_slots`). (a) is not pinned by anything; the only later touch is the `<= 0` validation at `:2063`. Falsifier: a source ratchet asserting no assignment to `server_args.max_mamba_cache_size` follows the `#364` block inside `handle_max_mamba_cache`. |

### Table PINNED (evidence that the sweep discriminates)

Ten of the twenty-two kept claims are genuinely pinned. Per the SUCCESS
CLAIMS law an instrument counts only after it shows it can discriminate;
these rows are that demonstration, and they are why the UNPINNED table above
is a backlog rather than a verdict on the project's testing culture.

| file:line | claim | pinning test file:line |
|---|---|---|
| `speculative/eagle_utils.py:117-119` — `# So the choice is made ONCE, collectively: every rank contributes its own`<br>`# capability, the MINIMUM rules, and if one rank lacks the native kernels then`<br>`# ALL ranks take Triton. Never a mixture.` | one rank without native spec kernels moves the whole group to Triton | `test/registered/unit/spec/test_spec_kernel_backend.py:106` `test_one_rank_without_kernels_moves_EVERY_rank_to_triton`; `:121` `test_decision_is_uniform_across_ranks`; `:138` `test_refusal_leaves_no_half_decision_behind` |
| `speculative/eagle_utils.py:135-141` (docstring) — *"A second, DIFFERENT value is refused rather than silently winning"* | `set_spec_kernel_backend` refuses a conflicting second decision | `test_spec_kernel_backend.py:74` `test_conflicting_decision_is_refused` |
| `distributed/parallel_state.py:759-764` — `# RANK-UNIFORM by construction: the condition reads only`<br>`# envs.SGLANG_BARLINK and self.world_size` | the pynccl-suppression decision cannot split the group | `test/registered/unit/distributed/test_barlink_suppresses_pynccl.py:73` `test_decision_is_rank_uniform`; `:111` `test_matches_parallel_state_source` |
| `distributed/device_communicators/barlink.py:752-753` — `"""One FRESH output tensor per call — never a shape-keyed cache.` | `_get_out_buf` allocates per call; two same-shape results are never the same tensor | `test/registered/unit/distributed/test_barlink_port.py:264-271` — AST ratchet, *"_get_out_buf must allocate a fresh tensor per call"* |
| `distributed/device_communicators/barlink_bar1.py:3533-3537` — `# THE ONE CONDITION THAT IS STRICTER HERE THAN FOR ALL_GATHER: the`<br>`# round count must be the same on ALL ranks, even though only one`<br>`# sends.` | `bc_plan`'s round count is a function of `nbytes` and slot only | `test/registered/unit/distributed/test_barlink_bar1_broadcast.py:202` `test_round_count_is_rank_uniform`; `:352` `test_the_round_count_is_the_same_on_every_rank`; `:358` `test_the_kernel_variant_is_decided_group_uniformly` |
| `distributed/device_communicators/barlink_bar1.py:3541-3545` — *"the extension rejects `in is out`"* | broadcast never runs with input aliasing output | `test_barlink_bar1_broadcast.py:369` `test_input_and_output_are_never_the_same_buffer` |
| `model_executor/kv_pressure_ladder.py:1491` — `# Invariant 2: never jump over a relief rung into a geometry flip.` | ascent exhausts relief before geometry | `test/registered/unit/model_executor/test_kv_pressure_ladder.py:536` `test_full_climb_exhausts_relief_before_geometry`; `:552` `test_forced_target_skipping_relief_is_a_hard_error` |
| `model_executor/kv_pressure_ladder.py:1438` — `# Invariant 3: only at round boundaries, never inside a capture.` | no flip inside a capture | `test_kv_pressure_ladder.py:579` `test_capture_guard_blocks_the_flip`; `:590` `test_flip_only_at_round_boundaries` |
| `model_executor/kv_pressure_ladder.py:1505` / `:1553` / `:1564` — `# Invariant 4: no flip (and no pre-stage) onto uncaptured graphs.` / `# Invariant 6: external rungs need the long hysteresis.` / `# Invariant 5: protected sessions stay on the fast rung.` | each stated as a hard error / block | `:560` `test_flip_onto_uncaptured_graphs_is_a_hard_error`, `:570` `test_pre_stage_onto_uncaptured_graphs_is_the_same_hard_error`; `:616` `test_external_rung_needs_the_long_hysteresis`; `:598` `test_protected_sessions_stay_on_the_fast_rung`, `:607` `test_all_protected_blocks_the_ascent` |
| `model_executor/short_term_offload_register.py:190-192` — `#: Ranks :func:\`plan_spill\` will never plan. §8.5 is not a soft preference:`<br>`#: … FCFS, never by a memory-pressure planner reaching past four other rungs.` | active work is never a spill victim | `test/registered/unit/model_executor/test_short_term_offload_register.py:350` `test_active_work_is_never_planned`; `:245` `test_the_ladder_is_ordered_as_the_doctrine_writes_it`; `:550` `test_the_origin_card_is_never_its_own_park_target` |

### Top findings

**#505-B-01 — `wave_in`'s error path claims the bytes are still parked; for
the last two ops in its own try block they are not.**
`offload_movement.py:916-918` vs `:910-914` and `:924`. Executed both
injection points hermetically: with `fail_ops={"free_destination": …}` the
op trace is `['copy_out','wait','wait','copy_in','wait','free_destination']`
— the copy-in **and** its wait completed — and the item is still reported
`parked` with 1000 bytes still booked. Ledger over-booking is permanent
(the release at `:924` is outside the try), and a retry re-copies from a
destination that was already told to free. The registered test injects only
at `copy_in`, the one point where the comment is true.
*Task title:* `#505-B-01 offload_movement.wave_in: the error path must distinguish "never retrieved" from "retrieved, release failed"`

**#505-B-02 — the "published set == captured set" rule the decode graph
runner states for itself is broken one publication earlier.**
`decode_cuda_graph_runner.py:436-444` argues, correctly and at length, that
publishing `capture_bs` before a later shrink is only accidentally safe —
and then `KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)` at `:392`
does exactly that, ahead of the weightless-KV cap at `:422`. A rule stated
in a comment and honoured at one of two sites is the registry-disagreement
shape of #500, moved from the catalog into the source file.
*Task title:* `#505-B-02 publish capture_bs to every consumer after the last cap, not just to the GGUF bucket registry`

**#505-B-03 — the hot-set freeze's aliasing argument is carried entirely by
a comment, on a method marked `# pragma: no cover - requires CUDA`.**
`expert_offload.py:3529-3531`. The whole zero-extra-VRAM in-place rearrange
rests on `resident_host = buf[:R].to("cpu")` making every `_src(e)` a host
read. No test anywhere in `test/` names `_apply_hotset_freeze`,
`_freeze_hotset` or `freeze_from_source`. This is the highest-consequence
UNPINNED claim in the sweep: its negation is wrong expert weights with
fluent output, which is precisely the failure class #452's B2 arm could not
localise.
*Task title:* `#505-B-03 pin the hot-set freeze aliasing invariant with a CPU-tensor permutation test`

**#505-B-04 — a "defensive: plan invariant" check that runs after the work
it guards.** `expert_offload.py:1462-1464`: the residency loop has already
copied and `release`d every resident expert before `R != len(resident_ids)`
is tested. The generic form of #501, and cheap to fix by hoisting.
*Task title:* `#505-B-04 hoist the staging-plan consistency check above the loop it is meant to guard`

**#505-B-05 — the radix caches' "Every rank must enter the all_reduce" claim
has an exempt door nobody pins.** `hiradix_cache.py:957-960` (and three
siblings) argue the collective correctly, but `writing_check(write_back=True)`
returns at `:945` before it, and nothing establishes that `write_back` is
rank-uniform at every call site. Given this project's four-incident history
in exactly this family, an unpinned rank-uniformity precondition on a
collective's entry gate is worth a test even though the reading is clean.
*Task title:* `#505-B-05 pin write_back rank-uniformity on the writing_check collective gate`

**#505-B-06 — the multi-rank no-half-apply promise is tested only on a
one-rank runtime.** `vram_dial.py:497-498`; `_dial_setup()` builds `n=1`
(verified by execution), so `test_below_floor_rejected_with_exact_numbers`
cannot fail on the negation of the *multi-rank* claim.
*Task title:* `#505-B-06 vram dial: a group dial with one out-of-floor rank must leave every rank untouched`

**#505-B-07 — ordering claims that are true today and unratcheted.**
`tp_worker.py:587-596` (the `is_verify` return must precede the weightless
recv branch), `model_runner_kv_cache_mixin.py:2007-2012` (the #364 cap must
be the last word), `decode_cuda_graph_runner.py:436` (no `capture_bs` write
after publication). All three hold at this base and all three are one
reordering edit away from a silent hang / silent re-inflation, with nothing
to catch it. Source/AST ratchets in the shape of
`test_barlink_port.py:264-271` are the cheap instrument this tree already
owns.
*Task title:* `#505-B-07 source ratchets for the three load-bearing statement orderings`

**Finding about the sweep itself.** Ten of twenty-two kept claims are
pinned, and the pinning tests are unusually good ones — `test_barlink_port`'s
AST ratchet, `test_spec_kernel_backend`'s mocked MIN-collective,
`test_kv_pressure_ladder`'s six numbered invariants. The failure is not a
missing testing culture; it is that the culture stops at module boundaries.
Every UNPINNED row above sits in a module whose *neighbours* are pinned. The
#501 shape survives where an invariant spans two functions (guard here,
mutation there) or two consumers (publish here, cap there) — never inside a
single well-tested predicate.

---

# Part C — numeric defaults that bound nothing


Desk audit, nothing executed, no GPU, no boot. Worktree `/spinning/wt-505-silent`,
branch `docs/silent-wrongness-505`, base `d6534052231276171daf3a844476812ec702ccf3`.
Upstream reference for the fork-delta: `upstream/main` = `ec741e4161` (2026-08-02).

**The law under audit** (CLAUDE.md:23-29, verbatim): *"REACH INCLUDES PARAMETERS
(#493 lesson): a cap/threshold/budget that never BINDS at the served geometry has
reach zero — the #449 query-chunk cap existed and was correct, but shipped at a
desk-picked 2048 MiB above the real peak, so it protected nothing for weeks. Any
shipped numeric default that exists to bound something needs a binds-proof at the
target geometry (a falsifier where the default measurably changes behavior); 'the
mechanism exists' is not evidence that it acts."*

**Target geometry** (what "binds at the served geometry" means below): 1x RTX 5090
32 GiB sm120 + 2x RTX 3080 20 GiB sm86, single node, no NVLink, no GPUDirect P2P,
all PHB, GPU0 on x4. Standing recipe: uneven TP=3 (`--rank-tp-ratio`), uneven DCP
(`--rank-kv-ratio`), NEXTN speculation, barlink transport, 27B-35B models plus a
122B-A10B offloaded MoE and DeepSeek-V4-Flash GGUF.

**Evidence ladder used for the verdicts** (descending strength): (a) a test that
fails when the default is raised/removed — a real falsifier; (b) a recorded
MEASUREMENT in `docs/dev/*.md` or a commit message naming the value and the
observed peak/rate; (c) a NOTE at the constant stating how the value was derived
from a measurement.

---

## Coverage

| surface | fork | upstream/main | fork-added | numeric | bounding-worded (this audit's set) |
|---|---|---|---|---|---|
| `python/sglang/srt/environ.py` `class Envs` | 572 | 519 | **115** | 47 | **31** |
| `python/sglang/srt/server_args.py` `class ServerArgs` | 598 | 457 | **166** | 103 | **75** |
| **total** | | | 281 | 150 | **106** |

Enumeration is AST-based, not regex: `ast.parse` over both files, walking
`ClassDef("Envs")` `Assign` nodes and `ClassDef("ServerArgs")` `AnnAssign` nodes,
with the preceding comment block / the `A[type, Arg(help=...)]` string literals
attached to each entry, and set-differenced against the same walk over
`git show upstream/main:<file>`. The 166 fork-added `ServerArgs` fields reproduce
audit #500's count exactly, which is the cross-check that the walk is right. The
briefing said 573 `Env*` entries; the AST finds 572 assignments inside `class Envs`
— the difference is not chased further, it does not change any verdict.

"Bounding-worded" = the NAME or the comment/help text contains one of
`cap|budget|threshold|limit|max|min|timeout|reserve|margin|quota|headroom|ceiling|floor|chunk|watermark|interval|retries|rounds|bound|allowance|safety|above which|below which`.
`--rank-tp-ratio` / `--rank-mlp-ratio` / `--rank-vocab-ratio` / `--rank-moe-ratio`
match the filter only through "length must equal `tp_size`" and are **not** bounds;
they are excluded from the tables.

**Out of scope, counted not audited:** 519 upstream `Env*` entries and 457 upstream
`ServerArgs` fields. Per the standing rule ("fix bugs in OUR features, not all of
sglang") they were enumerated for the set difference and then dropped. One upstream
observation is recorded in §5 because it surfaced from the same script and is a
40x default drift.

**Opened individually** (consumer site read, gate predicate read at its source,
`docs/dev/` + `git log` searched for the value): **42** of 106. **Not opened: 64** —
listed in §6 with the reason. Nothing in §6 is claimed to be fine; it is unexamined.

### What AUDIT_434 already discharged (so this audit's delta is visible)

`docs/dev/AUDIT_434_planner_constants.md` was an exhaustive sweep of **module-level
numeric constants and in-function numeric literals** in exactly two places:
`python/sglang/srt/uneven_perf.py` (6125 lines) and `python/sglang/srt/planner/**`
(59 modules). 764 candidate literals, triaged into `PROBE-FED` (11) / `STRUCTURAL`
(17) / `RIG-FITTED` (19) / `POLICY` (15) / `UNKNOWN-PROVENANCE` (1), with 16
follow-up tasks FU-434-1..16. Its question was *generality* ("is this number fitted
on the reference rig and applied unconditionally elsewhere"), not *reach* ("does
this number ever bind here").

It did **not** touch `environ.py` or the `ServerArgs` dataclass — the two surfaces
this audit enumerates. The only overlap is four rows, and they are not re-reported
here: `SGLANG_PERF_DECODE_GEMV_RESIDUAL_EXP` / `_PEAK_COMPRESSION_EXP` /
`_NONWEIGHT_FRACTION` / `_PREFILL_INVARIANT_FRACTION` (`environ.py:611-614`) are the
declared *seams* for AUDIT_434's four calibration scalars; #434 established that the
seam exists, is documented, and that **nothing populates it automatically**
(FU-434-1/2/3). They are `EnvFloat(None)` — absent-markers, not bounds — so they
fall outside this axis anyway. `SGLANG_PLANNER_CORRIDOR_MIB` (`environ.py:621`) is
#434's `POLICY` row for the #330 corridor and is likewise not re-litigated.

Everything else below is new ground.

---

## 1. Table INERT (no consumer, or consumer behind a gate that is off by default)

**No fork-added bounding default was found with zero consumers.** The first pass
produced 14 apparent zero-consumer fields; every one resolved to a real consumer on
a second look — the #236 spill budgets are read through a string-built
`getattr(sa, "kv_session_offload_" + name, default)`
(`managers/kv_session_offload.py:1305-1321`), `pp_stage_ratio` is consumed inside
its own declaring file (`server_args.py:12847`), and so on. Recorded because
"accepted-then-inert with no consumer at all" was the strongest finding this axis
could have produced, and it is **not** present. What *is* present is the weaker but
much broader form: consumers behind a gate that is off in the standing recipe.

| posten | file:line | claimed protection (verbatim) | why inert (gate file:line) |
|---|---|---|---|
| `admission_throttle_high` = 0.9 | `server_args.py:1099` | "Pool-occupancy fraction (0..1] at or above which the dynamic admission limit (#287) is lowered." | `admission_limiter.py:209` `if not self.auto: return False` in `observe()`; armed only by `scheduler.py:2323` `auto=sa.max_running_requests_ceiling is not None`, default `None` (`server_args.py:1061`). The limiter object is always built, so it *looks* live in a snapshot; `observe()` returns before reading either mark. |
| `admission_release_low` = 0.7 | `server_args.py:1109` | "Pool-occupancy fraction at or below which the dynamic admission limit (#287) may be raised again." | same gate |
| `admission_release_hysteresis` = 8 | `server_args.py:1119` | "Consecutive samples at or below --admission-release-low required before the dynamic admission limit is raised" | same gate |
| `admission_floor` = 1 | `server_args.py:1089` | "Lowest value the dynamic admission limit (#287) may float down to." | same gate; also `min(sa.admission_floor, ceiling)` at `scheduler.py:2319` makes 1 unreachable as a bound except at ceiling 1 |
| `fast_lane_reserved_heavy_slots` = 1 | `server_args.py:1223` | "Anti-starvation floor: the minimum number of running lane='heavy' requests that fast-lane preemption may not go below" | `schedule_policy.py:1410` `if getattr(server_args, "enable_fast_lane", False):` — `max_heavy_preemptible` stays `None` otherwise; `enable_fast_lane` default `False` (`server_args.py:1210`) |
| `kv_pressure_ascend_threshold` 0.85, `_ascend_window` 4, `_descend_threshold` 0.55, `_descend_window` 64, `_pre_stage_threshold` 0.7, `_pre_stage_window` 3, `_abort_stage_window` 32, `_horizon_rounds` 32, `_external_hysteresis_rounds` 512, `_consensus_interval` 8 | `server_args.py:4874-4962` | e.g. "Flip mark of the KV pressure ladder: occupancy fraction (0..1] at or above which the ladder climbs." | `kv_pressure_ladder.py:1944` `spec = parse_kv_pressure_ladder(getattr(server_args, "kv_pressure_ladder", None))` → controller is `None`; `--kv-pressure-ladder` default `None` (`server_args.py:4833`). Docstring at `:1928-1931`: "or ``None`` when the flag is unset (= today's behavior, byte-identical: nothing is constructed, no hook is attached, no sample is taken)." |
| the ten #236 spill reglers: `kv_session_offload_budget_total_tokens` 0, `_session_tokens` 0, `_prefill_tokens` 0, `_decode_tokens` 0, `_rate_tokens_per_s` 0.0, `_episode_seconds` 0.0, `_max_sessions` 0, `_spill_progress_lock_tokens` 0, `_spill_hysteresis_steps` 0, `_spill_cooldown_seconds` 0.0 | `server_args.py:1977-2092` | e.g. "maximum host-resident spill volume in TOKENS across ALL spilled sessions" | DOUBLE gate: `enable_kv_session_offload` default `False` (`server_args.py:1763`), and inside the feature `SpillBudgetConfig.armed` (`managers/kv_session_offload.py:1324-1338`) is False while every regler is zero — "All-zero (the default) -> every hook is skipped, byte-identical." So even with the feature ON, the whole #236 budget is disarmed by default. |
| `kv_session_offload_budget_demote_grace_iters` = 256 | `server_args.py:2101` | "scheduler iterations a DEMOTED session may wait for its host tail before it falls back to a host finish" | as above, and additionally: 256 is the one #236 regler that is NOT zero, yet it is not a member of the `armed` disjunction (`kv_session_offload.py:1327-1338`) — so it cannot arm the machinery on its own and never binds unless some *other* regler is set. |
| `kv_session_offload_restore_margin_tokens` 4096, `_tick_interval` 1, `_tick_floor` 8, `_max_spills` 1, `_host_ram_gib` 0.0, `_mtp_resident_slices` 0, `_park_timeout_iters` 512, `_wave_back_min_free_tokens` 0 | `server_args.py:1796-2181` | e.g. "restore the spilled session only when the allocator has (session tokens + this margin)" | `enable_kv_session_offload` default `False` (`server_args.py:1763`); `_handle_kv_session_offload` (`server_args.py:6347`) refuses each standalone by name, so they are *validated* and then unreachable |
| `dual_group_lane_budget_mib` None, `_admission_ms` 2.0, `_pairing_sat_rows` 64, `_pairing_max_defer_ms` 500.0, `_lend_mib` 0, `_lend_threshold_s` 5.0, `_spec_adaptive_hysteresis` 4, `_share_window_s` 0.0, `_share_min` None, `_share_min_windows` 5, `_prefill_chunk` None | `server_args.py:4365-4704` | e.g. "Starvation cap for --dual-group-lane-pairing: a queue head skipped in favour of better-pairing jobs" | `dual_group_lane` default `False` (`server_args.py:4324`) |
| `weightless_kv_chunked_block_size` 0, `_host_spill_tokens` 0, `_spill_device_cap` 0 | `server_args.py:1706-1750` | e.g. "cap the ALLOCATABLE device-resident KV slots" | `weightless_kv_fastlane` default `False` (`server_args.py:1655`); each of the three is additionally its own on/off value at 0 |
| `SGLANG_WL_GRAPH_MAX_BS` = 1 | `environ.py:1243` | "Weightless-KV streaming block-decode graphs (#136a): max decode capture bucket." | `decode_cuda_graph_runner.py:403-408` requires `model_runner.is_weightless_head or .is_weightless_worker` AND `_wl_chunk_block_size` — both off without the lane |
| `SGLANG_MEASURED_KV_BUDGET_SAFETY_MIB` = "400" | `environ.py:376` | "Scalar MiB or a comma list with one value per TP rank (roles differ: the draft-solo host carries prompt-length-scaled serving transients)." | `SGLANG_MEASURED_KV_BUDGET` default `False` (`environ.py:373`); guard `model_runner_kv_cache_mixin.py:896` `if not envs.SGLANG_MEASURED_KV_BUDGET.get():`. See §4 finding #505-C-03 — when it IS on, the value contradicts a measurement recorded ten lines above it. |
| `SGLANG_MEASURED_KV_BUDGET_CTX_ALLOWANCE_MIB` = 1024 | `environ.py:383` | "how many MiB of this rank's device share may be used by things outside its own allocator reservation (CUDA context, NCCL buffers) before the leftover measurement is reported as contaminated by a FOREIGN consumer" | same gate; consumer `model_runner_kv_cache_mixin.py:1196` sits below `if not envs.SGLANG_MEASURED_KV_BUDGET.get(): return` at `:1098` |
| `SGLANG_MOE_HEAT_DECAY` 0.5, `_MIN_GAIN` 8.0, `_MAX_SWAPS` 4, `_MIN_OBS` 32 | `environ.py:1057-1072` | e.g. "Upper bound on swaps per layer per round; the burst is swaps x expert bytes" | `SGLANG_MOE_HEAT_MIGRATION` default `False` (`environ.py:1051`) |
| `SGLANG_MOE_COLD_TIER_MANIFEST_TIMEOUT_S` = 30.0 | `environ.py:1210` | "Bounded wait for a peer's cold-tier manifest at the FIRST fetch" | `SGLANG_MOE_COLD_TIER_SHM` default `False` (`environ.py:1206`) |
| `SGLANG_EXPERT_STATS_INTERVAL_SEC` = 0.0 | `environ.py:1169` | "Additionally dump every N seconds (0 = only on exit / SIGUSR2)." | `SGLANG_EXPERT_STATS` default `False` (`environ.py:1146`), and 0.0 is itself "off" |
| `SGLANG_VRAM_DIAL_CHUNK_MIB` 16, `vram_dial_consensus_interval` 8 | `environ.py:365`, `server_args.py:5097` | "physical commit chunk of the VMM-backed KV pool in MiB" | `enable_vram_dial` default `False` (`server_args.py:5067`) |
| `kv_reshard_consensus_interval` = 8 | `server_args.py:5056` | "Scheduler rounds between two consensus boundaries of the #297 KV reshard runtime" | `kv_reshard_vectors` default `None` (`server_args.py:5034`) |
| `gdn_state_set_ladder_hysteresis` = 2 | `server_args.py:4823` | "Lowering hysteresis of --gdn-state-set-ladder, in admission cycles" | `gdn_state_set_ladder` default `None` (`server_args.py:4780`) AND the #500-B11 register gate `SGLANG_OFFLOAD_REGISTER` default `False` (`environ.py:1079`, `model_executor/offload_gdn_states.py:344`) |
| `SGLANG_LOGITS_PROCESSER_CHUNK_SIZE` = 2048 | `environ.py:1513` | (no comment; the flag it belongs to is `SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK`) | `SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK` default `False` (`environ.py:1512`), read at `layers/logits_processor.py:418` |
| `SGLANG_MAMBA_CKPT_WINDOW` = 2 | `environ.py:394` | "how many of the deepest on-grid mamba checkpoints per radix path evict_mamba keeps live" | `mamba_checkpoint_interval` default `None` (`server_args.py:3812`); `mem_cache/mamba_radix_cache.py:447` "None = upstream behavior, byte-identical" |
| `SGLANG_GGUF_STREAM_TRIM_TARGET_GIB` = 0.0 | `environ.py:1800` | "Reclaim down to about here once the soft watermark is crossed." | the soft watermark itself is 0.0 = off (`environ.py:1798`) — see §4 finding #505-C-02 |
| `SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS` = 0 | `environ.py:1110` | "Max decode batch size eligible for the captured offload path. Buckets with bs*top_k > scratch (would need >1 wave) cannot be captured" | self-gated: `decode_cuda_graph_runner.py:380` `if _moe_offload_graph_bs > 0:` — the default disables the cap. **Low severity, deliberately**: the invariant it names is separately enforced by a hard raise at capture (`layers/moe/fused_moe_triton/layer.py:2276-2286`, "worst-case unique spill must fit the scratch region, or a captured step could silently drop spill experts (wrong output, not epsilon)"). This is the one place in the sweep where a disarmed cap is demonstrably backstopped. |
| `SGLANG_SPEC_STATE_HASH_MAX_MB` = 0 | `environ.py:429` | "0 = hash every tensor fully. >0 = tensors above this many MiB are fingerprinted from a strided sample" | `SGLANG_SPEC_STATE_HASH` default `False` (`environ.py:425`) |
| `SGLANG_ADAPTIVE_FORCE_SWAP_INTERVAL` = 0 | `environ.py:1425` | "TEST-ONLY: force an adaptive runtime-state swap every N verify completions" | self-gated at 0; test-only by its own text |
| `SGLANG_PP_BOUNDARY_STATS` = 0 | `environ.py:523` | "log the stage-boundary traffic every N crossings (0 = off)" | self-gated at 0 |
| `SGLANG_BARLINK_SLOT_MIB` 64, `_CHUNK_MIB` 8 | `environ.py:652-654` | "Per-rank shared-memory slot size (MiB) for payload staging" / "Chunk size (MiB) of the gloo data-plane pipeline" | belong to the shm/gloo data planes; `SGLANG_BARLINK_TRANSPORT` default `"device"` (`environ.py:645`). (`_SLOT_BYTES` is still consulted by the device transport's shm segment, `barlink_device.py:1254`; the *chunk* is not.) |
| `SGLANG_BARLINK_HOST_P2P_MIB` 4, `_HOST_BLOCKS` 32, `_HOST_SLOT_MIB` None | `environ.py:672-680` | "Grid width of the host transport's two data kernels … more blocks buy nothing below ~1 MiB and cost tail latency" | host transport only; gate as above |
| `SGLANG_BARLINK_UCX_CHUNK_MIB` 4, `_UCX_RING_KIB` 24, `_UCX_AG_RING_KIB` 32, `_UCX_GRAIN_ELEMS` 32768, `_UCX_TIMEOUT_S` 300, `_UCX_RING_MIB` None | `environ.py:694-721` | "all_reduce payload (KiB) at or above which the one-step flat exchange gives way to a ring" etc. | UCX/RDMA data plane only; gate as above, and the target rig is single-node with no RDMA. Their derivations ARE measured (see §3) — measured on a cross-rig world-4 UCX link that this geometry cannot reach. |
| `disaggregation_prefill_budget_mib` None, `disaggregation_prefill_lane_interval` 1 | `server_args.py:4294-4305` | "Prefill-side activation/scratch budget in MiB PER CARD, an explicit item of the boot-time VRAM check" | PD disaggregation off by default (`disaggregation_mode`/`disaggregation_topology` `None`, `server_args.py:4237`) |
| `training_idle_grace_seconds` 120.0, `_poll_seconds` 2.0, `_preempt_timeout_s` 120.0, `_save_steps` 50, `_event_stream_timeout_s` 120.0 | `server_args.py:5479-5523` | "How long a preempted trainer may take to checkpoint and exit before it is killed. Bounded on purpose" | training tenant off by default |
| `workbench_preempt_timeout_s` 60.0, `_segment_timeout_s` 1800.0, `_probe_max_age_s` 604800.0 | `server_args.py:5577-5617` | "Hard bound on one segment. A tenant whose work does not fit inside this must cut it into smaller iterations." | `enable_idle_workbench` default `False` (`server_args.py:5537`), `workbench_tenants` default `None` (`:5557`) |

**Total INERT rows: 24 groups covering 71 individual postens.** Of the 106
bounding-worded fork-added defaults, **71 (67 %) cannot act in the standing recipe
at all.** That is not by itself a defect — most are honest opt-in features that say
so at their site — but it is the denominator that matters for the next table: only
~35 of 106 shipped bounding defaults are even *reachable* on the served geometry.

---

## 2. Table UNPROVEN, ranked by damage potential

Reachable in the standing recipe, consumer read at its source, no evidence of class
(a), (b) or (c) that the value binds here.

| posten | default | consumer file:line | claims to protect | what happens if it never fires (or fires wrongly) | proposed binds-proof (concrete falsifier) |
|---|---|---|---|---|---|
| `DEFAULT_TIMEOUTS_S[LLM_STREAM]` | **90.0 s** | `liveness/classes.py:84`, resolved at `liveness/watchdog.py:121`, enforced at `watchdog.py:321-323` `if silent >= timeout: await self._declare_dead(...)`, which calls `tokenizer_manager.abort_request(rid)` (`liveness/stream.py:169-175`) | "Seconds of silence tolerated per class before the consumer is declared dead." The table is self-labelled at `classes.py:81-82`: **"Unmeasured; see :data:`DEFAULT_TIMEOUT_RATIONALE` for why each is what it is"**, and `classes.py:98-99`: "the numbers encode an argument about the consumer, not a measurement of the server". | This one fires **too early**, not too late. `last_progress_at` is set at watchdog construction (`watchdog.py:204-210`) and only advances when the transport ACCEPTS bytes (`note_progress`, `:217-224`). A stream that emits no bytes for 90 s — a request queued behind a long prefill, or a first-token latency on the 122B-A10B offloaded MoE at high context — is declared dead and **aborted while healthy**. Presents to the user as a randomly dropped stream with a `WARNING … releasing` line, not as an error. It is live by default on every OpenAI-shaped streaming endpoint: `serving_base.py:119-124` wraps every `request.stream` response in `guard_generate_stream(..., endpoint_class=self._liveness_endpoint_class())`, no feature flag. | Two parts. (i) **Does it bind?** Hermetic: construct a `ConsumerWatchdog` with `LLM_STREAM` policy and a fake clock, feed no `note_progress`, assert `_declare_dead` at 90 s and that `release()` calls `abort_request`. (ii) **Does it bind wrongly?** On the rig: measure TTFT (first byte accepted by the transport, not first scheduler token) for the 122B-A10B offloaded MoE at the longest supported context, and for a request queued behind one, and compare against 90 s. If max observed TTFT + queueing is within 2x of 90 s the default is unsafe on this geometry. NOTE: I did NOT verify whether the chat/completions generator emits an early role/keep-alive chunk that would restart the clock — establishing that is step 0 of the falsifier, and it decides whether the finding is severe or moot. |
| `SGLANG_GGUF_STREAM_TRIM_SOFT_GIB` | **0.0 = OFF** | `environ.py:1798`, consumed in the GGUF weight stream (`SGLANG_GGUF_STREAM_DROP_CACHE` path) | "Synchronous cgroup reclaim during the GGUF stream, in GiB of memory.current. 0 (default) = off, behaviour byte-identical to before." | The comment at `environ.py:1804-1808` records the measurement that motivated the mechanism: *"on a swapless box that gap is the whole budget (#391). An external sampler chasing it on a wall-clock interval can be outrun -- window 3 saw memory.current move 88 -> 102 GiB inside one 15 s window."* The watermark built to stop that ships **disabled**, so the host-RAM wall it protects against is unprotected on every GGUF boot. The standing recipe serves DeepSeek-V4-Flash GGUF and 27B GGUF on a swapless box. | The measurement already exists (14 GiB in 15 s, window 3). What is missing is a value: boot the GGUF stack with `SGLANG_GGUF_STREAM_TRIM_SOFT_GIB` at candidate watermarks derived from `memory.max` minus the observed 15 s slew, and show a boot that OOMs at 0.0 and survives at the candidate. A falsifier that fails with the default and passes with a value IS the binds-proof. |
| `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB` | **2048** | `layers/attention/dsv4/indexer.py:337` `budget_mib = envs.SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB.get()`, `:346` `rows = (int(budget_mib) * 1024 * 1024) // step_bytes` | "Bounds the per-query-token duplication of the KV gather described in ANALYSE_447 section 2.3 L1. 0 disables it (one pass over the whole query axis, the pre-#449 shape). See #449." (`environ.py:1631-1638`) | **This is the #493 lesson's own posten, and it is still at the desk value.** `docs/dev/NOTE_449_dsv4_indexer_query_chunk.md:213-226` says it outright: *"the default is a ceiling picked at desk, not a tuned value"*, and §5 of that note is headed **"GPU measurement arm — BOOT-PENDING, not run"**. Reachable at the target geometry: `server_args.py:11355` sets `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` to True for DeepSeek-V4 on sm120, so the chunked torch indexer is the path the 5090 runs. If 2048 MiB is above the real per-rank peak the loop runs exactly once and #449 protects nothing — the shape the law was written about. | NOTE_449 §5 already specifies it, step by step, and it has not been run: one boot of DeepSeek-V4-Flash-0731 in the `BENCH_394_v4flash_club3090.md` configuration, A-vs-A floor first, then interleaved A/B at 8K and 32K context with the budget at `0` vs `2048`, **peak allocated VRAM per rank as the primary result**. The binds-proof is: at the served context the default must measurably lower the peak. If it does not, the default is above the peak and must come down. |
| `SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK` | **8192** | `layers/attention/dsv4/indexer.py:256` `chunk_positions = envs.SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK.get()` | "Sequence-axis chunk (in KV positions) … Bounds its peak intermediate at O(batch x chunk x heads) instead of O(batch x context x heads); see #426 / upstream #33246." (`environ.py:1622-1629`) | Same family, same path, same exposure as the row above — and the sibling that #449's own note calls the axis it "composes with rather than replaces". At 8192 KV positions the bound is inactive for every request whose context is below 8K, i.e. for a large part of the served traffic; whether it binds at the served 32K+ contexts is unrecorded. | Fold into the same boot as the row above: sweep `SEQ_CHUNK` at 0 / 4096 / 8192 and record peak allocated VRAM per rank at 8K and 32K. The pair (query-MiB, seq-chunk) must be swept together, because the per-row byte cost `_indexer_logits_step_bytes(chunk_seq, …)` is a function of the seq chunk — changing one changes what the other bounds. |
| `SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S` | **600.0** | `mem_cache/unified_radix_cache.py:419` `self.collective_timeout_s = envs.SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S.get()`, enforced in `_wait_bounded` (`:455-467`) | "Deadline for every cross-rank control collective issued from this cache … without it a dead peer parks this rank in all_reduce until the two-hour gloo group timeout expires." | A hang guard of the rank-local-condition-before-a-collective family the fork has hit four times (`CLAUDE.md` / `rank-lokaler-test-vor-kollektiv`). 600 s is argued only relative to the thing it replaces (7200 s), never against how long a legitimate HiCache control collective takes here. Too high: a wedged boot burns 10 minutes per collective before the named error. Too low: a legitimate slow collective aborts a healthy server. Neither direction is measured. | Instrument `_wait_bounded` to record the observed completion time of every control collective for one full serving window (it already loops on `work.is_completed()`), and set the bound at a stated multiple of the observed max. The falsifier: a test that patches the clock and asserts `HiCacheCollectiveTimeoutError` at exactly the configured bound already half-exists (`test/registered/unit/mem_cache/test_hicache_collective_wedge.py:102` asserts the message names the env) — it must additionally assert the *value*, which it does not. Reachable only on the hierarchical-cache path; rank the campaign accordingly. |
| `SGLANG_PERF_PROBE_LINK_TIMEOUT_S` | **45.0** | `uneven_perf.py:1365` `return float(envs.SGLANG_PERF_PROBE_LINK_TIMEOUT_S.get())`, expiry branch `uneven_perf.py:1410-1425` | "Wall-clock cap (seconds) on the NETWORK phase of the stage-0 probe (the pairwise NCCL link matrix) … without a cap it inherits torch's 600 s default process-group timeout and charges it to every boot." (`environ.py:594-601`) | Inverted damage: firing too EARLY is the harm. On expiry the probe "keeps the per-card measurements, stores the reason next to the empty link table, and returns" — and per the message at `:1420-1421` **"the plan falls back to its uniform link assumption"**. On this rig the links are emphatically not uniform (no NVLink, no P2P, all PHB, GPU0 on x4). A 45 s budget that is occasionally short would silently hand the planner a uniform-link model on the one rig where the link asymmetry is the point, on some boots and not others. | Record the wall time of the link-matrix phase across N cold boots of the standing TP=3 recipe (the phase already reports its own reason string) and compare the distribution against 45 s. A binds-proof here is the opposite of the usual one: the default must be shown to be *comfortably above* the observed max, not below it. Cheap: the number is already printed at every `auto-performance` boot. |
| `SGLANG_PERF_PROBE_TIMEOUT_S` | **600.0** | `uneven_perf.py:1624` `return float(envs.SGLANG_PERF_PROBE_TIMEOUT_S.get())` | "Wall-clock cap (seconds) on the WHOLE stage-0 probe subprocess." (`environ.py:592-593`) | Same shape, coarser. No derivation anywhere; a round 10 minutes. If short, a slow first probe is killed and the plan runs on no profile at all; if long, a wedged probe costs 10 minutes of every boot. | Same instrument as the row above — total probe wall time across cold boots, on the same run. |
| `SGLANG_RETRACT_SOLO_OOM_MAX_RETRIES` | **8** | `managers/schedule_batch.py:2819` `max_retries = envs.SGLANG_RETRACT_SOLO_OOM_MAX_RETRIES.get()`, branch `:2820` `if last_req.solo_oom_count <= max_retries:` | "how many times in a row a request may be the sole survivor of retract_decode and still not fit before it is failed instead of re-queued again. Ordinary extreme pressure … resolves within a couple of scheduler iterations; a request still solo-OOMing past this many retries is structurally too large for the pool" (`environ.py:480-486`) | **Mechanism-proven, value-unproven — the cleanest example of the distinction this axis is about.** `test/registered/unit/managers/test_retract_decode_fcfs.py:219-265` proves the guard fires and that the failure is a clean 503, but it reads the default (`:232 max_retries = envs.….get()`) and loops `range(1, max_retries + 2)` — it passes for ANY value of 8. So nothing pins 8. If 8 is too low, a transiently contended request is failed with a 503 that the comment itself says should not happen ("transient, not a sign this request is unfittable"). If too high, an unfittable request occupies retract cycles for longer. | Measure the empirical distribution of `solo_oom_count` at which pressure actually resolves, under the load the comment names (kv-session-offload spill budget exhausted / extreme concurrency), and pin the default at a stated quantile. The unit test then gains a second case that fails when the default is halved. |
| `DEFAULT_ATTN_SCRATCH_BUDGET_MIB` / `--attn-scratch-budget-mib` | **640** (flag default `None` → falls back to 640) | `models/deepseek_common/attention_forward_methods/forward_mha.py:210-218` `budget_mib = get_server_args().attn_scratch_budget_mib; if budget_mib is None: budget_mib = DEFAULT_ATTN_SCRATCH_BUDGET_MIB` | "Per-rank MiB budget (#395) for DeepSeek's chunked-prefix / attention-scratch strategy switch" (`server_args.py:1331`) | Has a derivation note (`forward_mha.py:77-85`) but it is a **back-derivation, not a measurement**: 640 MiB is the value that reproduces upstream's old 8192-token threshold *bit-for-bit on DeepSeek-V3 at TP=1* (`num_local_heads=128`). The note says so and says the derived token threshold differs "by design" on every other geometry. Under uneven TP=3 on this rig no rank has 128 local heads, so the threshold every rank runs is an extrapolation of an upstream token count nobody measured a peak for. | Measure the actual per-rank scratch peak for the served DeepSeek geometry (`attn_scratch_bytes_per_token` is already a named function) and check whether 640 MiB is above or below it per rank. Falsifier: a boot at the derived threshold vs one at half of it, comparing peak allocated VRAM per rank. This posten is the sibling that #449's own comment cites as its model (`environ.py:1634-1636`), so proving one and not the other leaves the pattern unproven. |
| `SGLANG_BARLINK_PEER_TIMEOUT_S` | **120.0** | `barlink_liveness.py:107` `ENV_TIMEOUT_S = "SGLANG_BARLINK_PEER_TIMEOUT_S"`, policy applied in `barlink_shm.py:168` | "Seconds a host-side wait may make no progress before it gives up. Scaled by SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT…" (`environ.py:757-760`) | Live in the standing recipe (barlink is the standing transport, `SGLANG_BARLINK_PEER_LIVENESS` defaults True, `environ.py:756`). This is the guard against the wedge family; 120 s is a round number with no derivation. Too low and a legitimately slow cold-build barrier is declared a dead peer; too high and a real wedge costs 2 minutes per collective before anyone learns. | The scaling seam (`SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT`) proves the authors knew 120 s is too short for a cold build. Measure the longest legitimate host-side barrier wait across a cold-cache boot and a warm boot of the TP=3 recipe, and set the base from that; the falsifier is a test that patches the clock and asserts the named error fires at the configured bound, plus an arm proving a cold build does NOT trip it. |
| `client_liveness_grace_fraction` 0.25, `_poll_interval_s` 1.0, `_teardown_timeout_s` 30.0 | as shown | `entrypoints/http_server.py:291-296` → `LivenessConfig.parse` | "Fraction of a class's timeout after which a quiet consumer enters the grace window" | Live by default (same path as the 90 s row). Grace at 22.5 s for LLM streams puts claims on the reclaim ladder (`watchdog.py:328-336`) for any stream that is quiet for 22.5 s — well within a normal long prefill. No derivation. | Same boot as the LLM_STREAM row: record the observed distribution of inter-byte gaps for real traffic and check what fraction of healthy streams cross the grace mark. |
| the other 11 reachable `DEFAULT_TIMEOUTS_S` entries (`VIDEO_STREAM` 300, `PREVIEW_TAP` 15, `IMAGE_GENERATION` 900, `AUDIO_SPEECH` 300, `AUDIO_TRANSCRIPTION` 120, `REALTIME_SESSION` 60, `CONTROL` 60, `REGISTRY_LEASE` 120, `DASHBOARD_SSE` 60, `EMBEDDING` 60, `TRAINING_EVENTS` 120) | as shown | `liveness/classes.py:83-96` | same table, same self-label "Unmeasured" | Each aborts a live consumer of its class on expiry. `PREVIEW_TAP` at 15 s is the tightest and sits on the video-enhance chain, an asset class the ONE-RUNTIME law puts inside this server. | One table, one campaign: instrument `note_progress` to record inter-byte gap percentiles per class over a representative window, then set each default from its own distribution. Until then the table is 12 desk numbers that can each abort a healthy client. |

---

## 3. Table BOUND-PROVEN (the discriminating half)

**It is not empty — but only just, and not one row is evidence class (a).** No
shipped bounding default in this fork has a test that fails when the default is
raised or removed. Four rows have evidence class (c) — a note at the constant
naming the measurement it came from — and one has class (b). Every one of them
carries a caveat, given verbatim.

| posten | default | evidence file:line | class | caveat |
|---|---|---|---|---|
| `SGLANG_ADAPTIVE_SERVING_MARGIN_MIB` | 512 | `environ.py:1430-1432`: "Measured on the T102 rig: 148 MiB post-map free OOM'd at KV-full deep prefill, 1367 MiB survived; 512 is the enforced floor between them." Repeated at the enforcement site `speculative/adaptive_graph_memory.py:938-943` with the failing kernel named (`fla/wy_fast recompute_w_u_fwd`), enforced at `:947` `if free_bytes < max_bytes + margin_bytes: raise RuntimeError` | (c) | 512 is an **interpolation between an OOM point and a survival point**, not a measured threshold — the true boundary is somewhere in [148, 1367] and nobody narrowed it. The enforcement is a boot-time refusal, which is the right shape (fails fast rather than late), and the error text quotes the measured numbers. Strongest row in the sweep. |
| `SGLANG_BARLINK_UCX_RING_KIB` | 24 | `environ.py:695-701`: "Measured crossover on a cross-rig world-4 group is ~22 KiB (task #244), so a speculative verify all-reduce sits on the ring side and a bs=1 decode all-reduce on the flat side." | (c) | Measured on a geometry **this rig cannot reach** (cross-rig world-4 over RDMA). At the target geometry the UCX plane is not selected at all (`SGLANG_BARLINK_TRANSPORT="device"`), so the row is proven elsewhere and inert here. It is in this table because it shows the sweep can discriminate, not because it acts. |
| `SGLANG_BARLINK_UCX_AG_RING_KIB` | 32 | `environ.py:709-711`: "Measured crossover cross-rig at world 4 is ~32 KiB (task #263), so a bs=1 decode gather stays flat and a 4-token verify gather rings." | (c) | same caveat |
| `SGLANG_BARLINK_UCX_GRAIN_ELEMS` | 32768 | `environ.py:713-717`: "Co-located TP ranks enter their host passes together, so the OpenMP region's join lands on a descheduled thread and the 128 -> 256 KiB step cost milliseconds (task #263)." | (c) | same caveat; the number 32768 elements is one step below the named 256 KiB knee at 8 bytes/element, i.e. derived from the measurement rather than measured directly |
| `SGLANG_BARLINK_PIPE_CHUNK_MIB` | `None` → **calibrated at boot** | `barlink_device.py:1070-1140` `_resolve_pipe_chunk`: sweeps candidates `[1, 2, 4, 8]` MiB over the REAL `all_reduce` path with barriers, gathers per-rank times, picks the summed minimum, logs it | (b), and the right pattern | The one posten in the sweep that does not ship a value at all: the default is a measurement. Two residues: the pre-calibration seed is a **duplicated literal** `os.environ.get("SGLANG_BARLINK_PIPE_CHUNK_MIB", "4")` at `barlink_device.py:989` where `environ.py:658` declares `EnvStr(None)`; and the candidate grid `[1,2,4,8]` is itself desk-picked — an optimum at 16 MiB is unfindable. |

---

## 4. Top findings (ranked)

**#505-C-01 — the client-liveness timeout table is live by default, self-labelled
"Unmeasured", and aborts healthy streams.**
`liveness/classes.py:81-96` ships twelve silence budgets, `LLM_STREAM = 90.0` among
them, under the comment *"Unmeasured; see :data:`DEFAULT_TIMEOUT_RATIONALE` for why
each is what it is"* and *"the numbers encode an argument about the consumer, not a
measurement of the server"*. Unlike almost everything else in this sweep it is
**not behind a feature gate**: `serving_base.py:119-124` wraps every OpenAI-shaped
streaming response in `guard_generate_stream`, and on expiry
`watchdog.py:336-350` → `stream.py:169-175` calls `abort_request` on a live request.
The clock starts at watchdog construction and advances only on bytes accepted by
the transport, so a long first-token latency counts in full. On the standing recipe
(122B-A10B offloaded MoE, long contexts, single-stream serving) 90 s is not
obviously above the worst legitimate TTFT — and nobody has checked. Highest damage
of anything found: silent, user-visible, on the default path.
*Task:* `#505-C-01: measure per-class inter-byte gaps and derive the liveness timeout table, starting with LLM_STREAM=90s vs real TTFT`

> **CORRECTED by #514** — see the correction under finding 1 in section 5. TTFT is
> OUTSIDE the budget (the response is built only after `generator.__anext__()`
> returns, and only that response is wrapped), so "aborts healthy streams" does not
> hold. The default was deliberately left unchanged; the load-bearing ordering is now
> pinned instead.

**#505-C-02 — the GGUF host-RAM watermark ships at 0 (off) although the measurement
that motivated it is recorded at the flag.**
`environ.py:1798` `SGLANG_GGUF_STREAM_TRIM_SOFT_GIB = EnvFloat(0.0)`, four lines
under *"window 3 saw memory.current move 88 -> 102 GiB inside one 15 s window"*.
The rig is swapless and the standing recipe streams GGUF weights. A guard whose
motivating measurement is written next to it and whose default is "off" is the #449
shape with the sign flipped: not a bound too high to bind, a bound never armed.
*Task:* `#505-C-02: derive and arm SGLANG_GGUF_STREAM_TRIM_SOFT_GIB from the measured stream slew, or state at the flag why off is correct`

**#505-C-03 — the measured-KV-budget safety margin (400 MiB) contradicts a
measurement recorded in its own consumer.**
`environ.py:376` ships `SGLANG_MEASURED_KV_BUDGET_SAFETY_MIB = EnvStr("400")` as a
scalar for all ranks. Its consumer's docstring
(`model_runner_kv_cache_mixin.py:809-815`) records: *"the draft-solo host carries
the dual-prefill / draft-append serving transients, which scale with prompt length
— measured 2026-07-22: 10k prefill needs ~1 GiB, 50k ~2-3.5 GiB on the host, while
shadow ranks served everything with ~1.6 GiB"*. Every one of those measured numbers
is above 400 MiB, on every rank. The surrounding comment says *"the only ASSUMED
number left is the safety margin itself"* (`:786-788`). The feature is opt-in
(`SGLANG_MEASURED_KV_BUDGET` default False, `environ.py:373`), which is why this is
finding 3 and not finding 1 — but the moment it is switched on, the shipped default
is known-too-small by the file's own evidence, and under NEXTN speculation the
draft-solo host is exactly the rank it is too small for.
*Task:* `#505-C-03: derive the measured-KV safety margin per rank ROLE from the recorded 2026-07-22 numbers instead of one scalar 400`

**#505-C-04 — #449's own default is still the desk number the law was written
about, and its measurement arm is still not run.**
`environ.py:1638` `SGLANG_DSV4_INDEXER_QUERY_CHUNK_MIB = EnvInt(2048)`.
`NOTE_449_dsv4_indexer_query_chunk.md:226` — *"the default is a ceiling picked at
desk, not a tuned value"* — and §5 of the same note is titled **"GPU measurement arm
— BOOT-PENDING, not run"**. There is no `NOTE_493` or later commit in this tree that
retunes it (`git log --grep` over 449/493 finds only the four #449 commits), and
`FEATURE_CATALOG.md` §15 says the same in its own words: *"it does not remove it,
and the speed effect is unmeasured (no GPU window taken)"*
(`docs/dev/FEATURE_CATALOG.md:1400-1402`). The posten that supplied the lesson in
`CLAUDE.md:23-29` has not itself been discharged.
Its sibling `SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK = 8192` (`environ.py:1629`) is in
the same state and must be swept jointly, because the query-MiB budget is converted
to rows through a per-row byte cost that is a function of the seq chunk
(`indexer.py:341-346`).
*Task:* `#505-C-04: run NOTE_449 §5 — peak-VRAM-per-rank A/B for the DSV4 indexer query and seq chunks at 8K/32K`

**#505-C-05 — two thirds of the fork's bounding defaults cannot act in the standing
recipe, and no shipped bounding default anywhere has a value-pinning falsifier.**
71 of 106 (67 %) are behind a gate that is off in the served configuration (§1).
Of the ~35 reachable ones, the BOUND-PROVEN table has five rows, four of them
evidence class (c) and three of those measured on a geometry this rig cannot reach.
**Zero rows are evidence class (a).** The pattern is visible in
`test_retract_decode_fcfs.py:232`, which reads the default it is testing
(`max_retries = envs.SGLANG_RETRACT_SOLO_OOM_MAX_RETRIES.get()`) and therefore
passes for every possible value: the fork tests that its guards FIRE, never that
its numbers are RIGHT. That is a systematic instrument gap, not a per-posten
oversight, and it is the reason #449 could ship inert for weeks.
*Task:* `#505-C-05: add a value-pinning convention — every shipped bounding default gets a test that fails when it is doubled or removed`

**#505-C-06 — barlink's knobs are read through raw `os.environ` at import time, so
their declarations in `environ.py` are decorative.**
`barlink.py:52` `_CHUNK_BYTES = int(os.environ.get("SGLANG_BARLINK_CHUNK_MIB", "8"))`,
`:70` `_SLOT_BYTES = … "64"`, `barlink_host.py:127` `_BLOCKS = … "32"`,
`barlink_ucx.py:91` `_RING_BYTES = … "24"`, `:121` `_TIMEOUT_S = … "300"`,
`barlink_device.py:989` `… "4"`. Three consequences: (i) each default exists twice
and can drift silently — `barlink_device.py:989`'s `"4"` already disagrees with
`environ.py:658`'s declared `None`; (ii) they are read **once at module import**, so
`envs.X.set()` after import — which `server_args.py` does for other envs at
`:11350-11366` — has no effect, the #500-B20 shape; (iii)
`envs.SGLANG_BARLINK_*.override(...)` in a test changes nothing, so any test written
against the declared env is silently inert. This blocks the falsifiers proposed for
`SGLANG_BARLINK_PEER_TIMEOUT_S` above.
*Task:* `#505-C-06: route the barlink env reads through envs.* at call time and delete the duplicated literal defaults`

**#505-C-07 — the KV-pressure ladder's defaults exist twice with no single source
of truth.**
`kv_pressure_ladder.py:244-255` defines `DEFAULT_ASCEND_THRESHOLD = 0.85`,
`DEFAULT_ASCEND_WINDOW = 4`, `DEFAULT_DESCEND_THRESHOLD = 0.55`,
`DEFAULT_DESCEND_WINDOW = 64`, `DEFAULT_PRE_STAGE_THRESHOLD = 0.70`,
`DEFAULT_PRE_STAGE_WINDOW = 3`, `DEFAULT_ABORT_STAGE_WINDOW = 32`,
`DEFAULT_HORIZON_ROUNDS = 32`, `DEFAULT_EXTERNAL_HYSTERESIS_ROUNDS = 512`; the
identical nine values are declared again as `ServerArgs` defaults at
`server_args.py:4874-4962` and joined by `getattr(server_args, "<name>", DEFAULT_*)`
at `kv_pressure_ladder.py:1880-1922`. They agree today. Nothing keeps them agreeing,
and the `getattr` fallback is exactly the construct that hides the disagreement when
they stop. The same duplication-with-fallback appears at
`managers/admission_limiter.py:52-55` vs `server_args.py:1089-1119`. Low severity
while the ladder is off by default (§1), which is why it is last.
*Task:* `#505-C-07: make the ServerArgs field the single source of the kv-pressure and admission defaults, or assert equality at import`

---

## 5. One upstream observation (out of scope, recorded not audited)

`SGLANG_VLM_CACHE_SIZE_MB` is declared `EnvInt(100)` (`environ.py:1465`, identical in
`upstream/main`) and read as
`int(os.environ.get("SGLANG_VLM_CACHE_SIZE_MB", "4096"))`
(`disaggregation/encode_server.py:330`) — a 40x drift between the declared default
and the effective one. Upstream code, upstream declaration; reported for the
upstream-PR pile, not audited further here.

---

## 6. Not opened (64 of 106) — honest gap

Enumerated by the AST walk and filtered into the bounding set, but not individually
traced to a consumer and a gate. **No verdict is implied for any of them.**

- The remaining `dual_group_lane_*`, `workbench_*`, `training_*` and
  `disaggregation_*` numeric fields beyond the ones tabled in §1: the feature gate
  was confirmed off by default, so they were placed in §1 by their group gate
  without reading each consumer individually.
- `rank_perf_loose_ctx_percent` (0.0), `rank_auto_reserve_mib` ("auto"),
  `rank_gpu_memory_mib` (None), `gdn_resident_state_slots` (None),
  `mamba_checkpoint_interval` (None), `max_running_requests_ceiling` (None): all
  reachable in the standing recipe, all with `None`/`auto`/`0` defaults that mean
  "derive it" rather than "bound it at N". They need a different question than this
  axis asks (does the DERIVATION bind), and they are the natural first cut of a
  follow-up.
- The `SGLANG_KV_CANARY_*`, `SGLANG_TEST_*` and other test-only numeric envs: named
  test-only at their site.
- The 32 fork-added `ServerArgs` fields whose help text is bounding-worded but whose
  default is a string/enum rather than a number (`rank_kv_ratio` "coupled",
  `rank_auto_reserve_mib` "auto", `swa_pool_sizing` "ratio",
  `speculative_cross_algorithm_ctx_gate` "auto", `lane_offload_profile` "latency",
  `regime_controller` "off", …): out of this axis by construction, but several are
  policy selectors that resolve to numeric bounds downstream and would belong to a
  follow-up sweep.

**PROGRESS MARKER: reached a complete enumeration of both surfaces (106 bounding
fork-added numeric defaults), 42 opened individually with consumer + gate read at
source; remaining 64 listed above with the reason, unexamined.**

---

# Part D — catalog §14/§16 against their code predicates


Method: `AUDIT_500_mechanism_reach.md` §2, applied to the two sections audit #500
explicitly left unswept ("§14 (dashboard) and §16 (instruments) were not swept for
predicates — the context went to §§1-7 … That is a stated coverage gap, not a clean
bill of health", AUDIT_500 §2). Desk audit, nothing executed, no GPU.

Classes: **[WIDER]** code more general than the catalog line — **[NARROWER]** catalog
over-promises, split DOC-CANDIDATE / BUG-CANDIDATE — **[EXACT]** — **[NOT-FOUND]** no
predicate located.

### Coverage

§14 carries 7 conditional claims, §16 carries 6. All 13 were resolved to a predicate.
`expert_stats (router distribution + hit rate)` is the one §16 item taken at face
value from module presence rather than traced to a consumer — recorded as such.

Not done, stated plainly: **Direction 2 for these sections** — the inverse sweep
("wired but not written down"). §14's six lines describe `planner/webui.py` (14,816
lines) plus `rigmon/` (9,971) plus `comm_suite.py`/`energy.py`/`jtok_counter.py`/
`github_share.py`/`self_update.py`/`wizard*.py`/`rig_artifact.py` (~8,500) — roughly
33,000 lines of fork code summarised in 43 words. The size of that gap is recorded
here; its contents are not enumerated. §16's six lines cover `rigmon/` (19 modules),
`utils/collective_clock.py`, `model_executor/forward_peak.py` and the `scripts/`
harness, likewise not enumerated.

### §14 — Dashboard

| # | catalog claim (§14) | predicate (file:line) | class |
|---|---|---|---|
| D-1 | "Guided config wizard with honest refusals" | `planner/wizard.py:703-714`, `:1469`, `:1521` | [WIDER] |
| D-2 | "comm benchmark suite with anonymization gate" | `planner/rig_artifact.py:558`, `:784-795`; sole caller `planner/webui.py:4853` | [EXACT] for that route |
| D-3 | (same line, other posting route) | `planner/github_share.py:176`, `:214`; `planner/webui.py:4690-4721` | [NARROWER / BUG-CANDIDATE] |
| D-4 | "energy metering (tok/s + J/token)" | `planner/energy.py:23-24`, `:383-412` | [EXACT], caveat missing from catalog |
| D-5 | "benchmark tiles with measured/estimate/absent provenance" | `planner/cost_model.py:142-146` | [EXACT] + vocabulary defined 3x |
| D-6 | "one-click knee-point probe" | none — see below | [NOT-FOUND] |
| D-7 | "self-update with auto-rollback" | `planner/self_update.py:659-688`, `:726-799`; gate `planner/webui.py:3632` + `:3567` | [NARROWER / DOC-CANDIDATE] |
| D-8 | (same line, the rollback instrument) | `planner/self_update.py:691-712` | [BUG-CANDIDATE] |
| D-9 | "GitHub result posting (opt-in PAT)" | `planner/github_share.py:89`, `:97-105` | [EXACT], redaction narrower than it reads |

**D-1 [WIDER].** The refusal machinery is a family/matrix engine, not a form with
warnings. `wizard.py:703-705` — *"Why this cell of the matrix cannot exist. Empty
means it can. / Each reason names its source, because a refusal without a citation
is"* … — and `:1469` *"engine will NOT refuse on its own is refused here"*, `:1521`
*"The wizard never emits a flag it cannot explain."* Six catalog words for
`wizard.py` (1,822 lines) plus `wizard_islands/_lanes/_links/_offload/_tipping`.

**D-3 [NARROWER / BUG-CANDIDATE] — the two public-posting routes have opposite
anonymity policies, and the weaker one carries the start command.**
Route A (#271 rig share) passes `rig_artifact.build_digest`, whose docstring is
explicit that the steps are inseparable (`rig_artifact.py:789-791`: *"curate, then
scrub, then the anonymity gate. The only entry point a UI may use -- the three steps
are not separable in practice, and making them separable is how one of them gets
skipped."*) and whose gate refuses absolute filesystem paths, IPs, UUIDs, hostname
and username (`assert_anonymized`, `:558-588`, `_ABS_PATH_RE` at `:571`).
Route B (#152 result share) does not. `github_share.py` contains no `scrub_tree` and
no `assert_anonymized` call at all. `build_report` states `:186` *"the EXACT start
command. argv is emitted verbatim"* and does exactly that at `:214`:

```python
md.append(" ".join(str(a) for a in argv))
```

`webui.share_submit_payload` (`webui.py:4690-4721`) posts that markdown to a public
GitHub issue. On the reference rig a real start command contains
`/spinning/llm_stuff/club-3090/models-cache/...` — a string Route A refuses by name.
Env values are redacted only when the env NAME ends in one of five suffixes
(`_SECRET_ENV_SUFFIXES = ("TOKEN", "SECRET", "PASSWORD", "KEY", "PAT")`,
`github_share.py:89`), so a credential in a differently-named variable is posted
verbatim too. The module also declares its own network path untested
(`github_share.py:51`: *"needs a real PAT + network, deferred to live validation"*).
*Task #505-D3:* `github_share: route the #152 result share through scrub_tree + assert_anonymized, or state in the catalog that only the rig-artifact route is gated`

**D-4 [EXACT], with a caveat the catalog drops.** J/token is genuinely measured —
NVML board power integrated over each phase's wall clock (`energy.py:23-24`,
sampler at `:383-412`, `nvmlDeviceGetPowerUsage`). The code is honest about what
that excludes and the catalog is not: `energy.py:278-279` — *"GPU (NVML) power only
— excludes CPU/RAM/PSU-conversion losses (NOT wall-socket power)"*, repeated in the
emitted provenance string at `:350`. A J/token figure read as wall-socket energy is
wrong by the CPU/PSU share. Corrected in §14.

**D-5 [EXACT] — and this is the sweep's discriminator.** `cost_model.Provenance`
(`cost_model.py:142-146`) — *"Where a number came from. There is no 'probably' tier
on purpose."* It is used as designed: `wizard_tipping.py:600-607` refuses to call an
uncomputable decode knee safe — *"the guard needs measured per-card memory-bandwidth
scores and there are none on disk, so the knee is not computable -- not 'safe'"*.
That is the SUCCESS-CLAIMS law already implemented, and it is why the empty and
narrower rows elsewhere in this audit can be believed.
Minor defect: the same vocabulary is redeclared as bare strings in
`planner/split_probe.py:85` and `planner/rig_coupling.py:103-105`, independent of the
enum. Three definitions of one vocabulary is the registry-disagreement shape of
AUDIT_500 §3, in miniature.
*Task #505-D5:* `planner: one Provenance vocabulary — split_probe.py and rig_coupling.py redeclare cost_model.Provenance as bare strings`

**D-6 [NOT-FOUND] — there is no knee-point probe.** The string `knee` does not occur
in `planner/webui.py` at all. Two things exist and neither is the claim:
1. A **modelled** guard. `advantage.py:92-93` — *"#: knee guard — only computable
   when measured membw scores exist. / `decode_knee_ok: Optional[bool] = None`"*,
   surfaced by `wizard_tipping.py:582-612`, which classes it ESTIMATE and says so:
   *"Modelled is not measured, and the honest reading of a modelled guard is that it
   says which side it believes we are on, not by how much"* (`:587-589`).
2. `energy.power_limit_sweep` (`energy.py:1217`) — which WOULD measure an efficiency
   knee, and is the mechanism behind the `power_target_sweep` scenario's hypothesis
   *"there is a knee well below the stock limit"* (`scenarios.py:470-472`). It has
   **no production caller anywhere in the repository**; the only references are its
   own `__all__` (`energy.py:90`) and two test call sites
   (`test/registered/unit/planner/test_energy_dashboard.py:251`, `:277`). `webui.py`
   exposes `/api/power_profile` (`:5374`) and no sweep endpoint.
This is AUDIT_421's "built but not wired" shape, and the catalog line is the reason
nobody noticed: it reads as though the probe ships.
*Task #505-D6:* `wire power_limit_sweep to a dashboard endpoint, or drop "one-click knee-point probe" from §14`

**D-7 [NARROWER / DOC-CANDIDATE].** Auto-rollback is real (`apply_health_result`,
`self_update.py:659-688`; supervisor loop `:726-799`) but reachable only under
`--serve-supervised`. `webui.py:3632`:

```python
if not _supervised():
    return {"ok": False, "error": "version switching needs the supervisor: start the "
            "dashboard with --serve-supervised (plain --serve keeps "
            "running the launch checkout and cannot restart itself)"}
```

with `_supervised()` = `os.environ.get("SGLANG_DASHBOARD_SUPERVISED") == "1"`
(`webui.py:3567`). Plain `--serve` gets install-only, and the UI says so
(`webui.py:9730`). The refusal is honest; the catalog line is short of it.
Corrected in §14.

**D-8 [BUG-CANDIDATE] — the auto-rollback instrument cannot discriminate.**
`wait_health` (`self_update.py:691-712`) returns True on the first
`resp.status == 200` for `GET /`:

```python
with urllib.request.urlopen(url, timeout=2) as resp:
    if resp.status == 200:
        return True
```

A new dashboard version that serves its index page while computing wrong numbers is
"healthy", is marked good (`store.mark_good`, `:672`) and becomes the rollback target
for the NEXT version. CLAUDE.md's SUCCESS-CLAIMS rule (2): an instrument's verdict
counts only after the instrument passes a can-discriminate check on known-different
inputs. This gate has none — it cannot fail on the failure class an auto-rollback
exists for.
*Task #505-D8:* `self-update health check: probe a computed endpoint with a known-answer assertion, not HTTP 200 on /`

### §16 — Measurement / window infrastructure

| # | catalog claim (§16) | predicate (file:line) | class |
|---|---|---|---|
| D-10 | "gpu-arb (UUID-based holder + heartbeat …)" | `registry/ledger.py:17`, `:607`; `registry/arbiter.py:1025`; `test/conftest.py:47-67` | [NARROWER] convention, not enforcement |
| D-11 | "forward_peak.py (VRAM corridor judged AT PEAK, not idle)" | `model_executor/forward_peak.py:150-155`; wired at `model_executor/model_runner.py:4060-4081` | [NARROWER / BUG-CANDIDATE] |
| D-12 | "cachetrim with --ready-url self-retirement" | `scripts/dsv4/cachetrim.sh:80-82`, `:295` | [WIDER] |
| D-13 | "measured-KV-budget stale-boot trap" | `environ.py:373`; `uneven_perf.py:2617`; `rigmon/kvbudget.py:16-22`; `planner/runner.py:203`, `:231-238` | [NARROWER / DOC] |
| D-14 | "CollectiveClock (compute vs wait per rank)" | `managers/scheduler_components/metrics_reporter.py:341-360`, `:134-136` | [NARROWER / DOC] — prefill only |

**D-10 [NARROWER] — gpu-arb is a convention with no runtime enforcement.** The code
says so itself, three times: `registry/ledger.py:17` *"the ``/spinning/gpu-arb``
cross-session convention"*, `:607` *"Touch the holder line and push the lease out.
The gpu-arb convention."*, `registry/arbiter.py:1025` *"gpu-arb convention: touch
every held lease so nothing is reaped."* No path refuses GPU work when no holder
exists. The one place arbitration is actually ENFORCED runs in the opposite
direction: `test/conftest.py:47-67` fails a pytest run that WROTE the shared arb
paths (`session.exitstatus = 1`, after #438 chased four planted 0-byte lock files).
Worth recording because CLAUDE.md states the gpu-arb rule as non-negotiable while
nothing in the tree can catch a violation of it.

**D-11 [NARROWER / BUG-CANDIDATE] — the peak probe is off by default and unknown to
the env registry.** `maybe_create` (`forward_peak.py:150-155`), docstring verbatim:
*"The tracker, or None when the probe is off (the default)."*

```python
path = os.getenv("SGLANG_FORWARD_PEAK_PATH")
if not path:
    return None
```

It is properly wired into the runner (`model_runner.py:4060-4081`), so the mechanism
works — but the corridor is judged at peak only in runs that opted in, which is the
opposite of how §16's line reads, and the VRAM-corridor rule (free >= 400 MiB at
peak) is stated as a standing rule. Second defect: `SGLANG_FORWARD_PEAK_PATH` is read
by raw `os.getenv` and has **no entry in `environ.py`** (`grep FORWARD_PEAK environ.py`
returns nothing), so the instrument the corridor rule depends on is invisible to the
env catalog and to AUDIT_500's Direction-2 enumeration, which read `environ.py` by AST.
*Task #505-D11:* `register SGLANG_FORWARD_PEAK_PATH in environ.py and decide whether the corridor rule requires the probe on by default`

**D-12 [WIDER].** The script is better than the line: with no ready signal it emits a
refusal carrying its own measured counter-number
(`cachetrim.sh:295`: *"NO ready signal given -- it will run until the server exits,
which costs throughput during serving (#391 w4 vs w5: floor 39.91% vs 2.55%)"*).

**D-13 [NARROWER / DOC].** The trap is real and well-documented
(`rigmon/kvbudget.py:16-22`, *"A shift of roughly 4x has been observed from boot
order"*) and the benchmark harness neutralises it by default
(`planner/runner.py:203` `reset_kv_budget: bool = True`, with a refusal at
`:231-238`). But the feature that creates the trap is itself off by default:
`environ.py:373` `SGLANG_MEASURED_KV_BUDGET = EnvBool(False)`, consumed at
`uneven_perf.py:2617` `if not envs.SGLANG_MEASURED_KV_BUDGET.get():`. On a stock boot
there is no measured KV budget and therefore no stale budget to inherit. §16's line
reads as a standing hazard; the predicate makes it opt-in.

**D-14 [NARROWER / DOC] — "compute vs wait per rank" covers prefill only.** The clock
is installed unconditionally for the reference recipe (no env gate) but with three
named narrowings, `metrics_reporter.py:341-352`:

```python
if (getattr(self.scheduler, "device", "") != "cuda"
        or self.scheduler.server_args.pp_size != 1):
    return
self.rank_prefill_log.clock = collective_clock()
```

and a fourth in `RankPrefillLog`'s own docstring (`:134-136`): *"only
ForwardMode.is_plain_prefill forwards are wrapped, and those are exactly the ones
reported through ``report_prefill_stats``"*, plus target-runner-only
(`:342-344`: *"Deliberately NOT the draft runners"*). Every narrowing is argued at
its site. The consequence for the ms/round rule is that the per-rank compute-vs-wait
split exists for prefill and **not for decode**, which is the phase the
slowest-rank/bandwidth reasoning turns on. Recorded, not proposed as a fix — pairing
a decode-side clock to speculative verify rounds is its own task.
*Task #505-D14:* `per-rank compute-vs-wait for the decode/verify round, not only plain prefill`

### Top findings, ranked

1. **#505-D3** — the #152 GitHub result share posts the start command's argv verbatim
   to a public issue with no anonymity gate, while the sibling #271 share route
   refuses absolute paths by name. Cross-route policy inconsistency on the one surface
   that is public by construction.
2. **#505-D6** — "one-click knee-point probe" has no implementation; the sweep that
   would measure a knee (`energy.power_limit_sweep`) has test callers only.
3. **#505-D8** — the auto-rollback health gate is `HTTP 200 on /`; it cannot fail on
   a version that serves but computes wrong.
4. **#505-D11** — `SGLANG_FORWARD_PEAK_PATH` is off by default and absent from
   `environ.py`, so the VRAM-corridor instrument is both opt-in and uncatalogued.
5. **#505-D14** — the per-rank compute-vs-wait clock covers plain prefill only;
   the decode/verify round, where the slowest-rank argument lives, has no equivalent.
6. **#505-D5** — three independent declarations of the measured/estimate/absent
   vocabulary.
7. **#505-D10** — gpu-arb is a convention; no code enforces it (recorded, not
   proposed as a fix — enforcement may be deliberate).

---

## Cross-cutting — the reach of the C1 remedy itself

**There are TWO completeness remedies in this tree, at different altitudes, and
neither covers the other's surface.** Read this section together with axis A1's
finding #505-A1-01, which audits the loader-level one in depth; this section audits
the model-level one and states the combined picture.

- **Loader-level:** `model_loader/weight_utils.raise_on_unloaded_draft_parameters`
  (`:2032`), called once, from `DefaultModelLoader` (`model_loader/loader.py:903`).
  Its own docstring states the ambition — *"Hoisting the check to the loader makes
  it a property of loading A DRAFT, not of one model class"* — and immediately below
  it, two silent-return arms bound it: `if loaded_params is None: return` (`:2058`)
  and a `if not loaded: return` for the empty-set case (`:2063-2067`). Reach and the
  GGUF / QuantizedRL gaps are quantified in **#505-A1-01**.
- **Model-level:** `models/deepseek_v4_dspark.py:868` `_assert_required_params_loaded`,
  counted below.

The two do not compose: the loader guard fires only for DRAFT loads through
`DefaultModelLoader` on self-reporting models, and the model-level check exists in
two files. Everything else in `models/` is on the warning path.

The model-level fix is well argued (`models/deepseek_v4_dspark.py:868-886`):

> Every parameter the draft DECLARES must have been written. / The loop above drops
> a checkpoint tensor it cannot match to a parameter with only a warning, and that
> direction stays a warning on purpose: a checkpoint may legitimately carry tensors
> this build has no module for … The opposite direction must not be silent. A remap
> that produces a name matching no parameter leaves that parameter at its
> uninitialised construction value; the draft still "loads", and the only symptom is
> a speculative accept rate pinned at zero.

Pinned by `test/registered/unit/models/test_dspark_draft_load_completeness.py:28`.

**Its reach, counted at this tip** (`python/sglang/srt/models/`, 186 files defining
`load_weights`):

| | count |
|---|---|
| models defining `load_weights` | 186 |
| of those, containing a `logger.warning` in the loader | 55 |
| containing any completeness comparison at all | 11 |
| where that comparison **raises** | **2** — `deepseek_v4_dspark.py:896`, `dflash.py:582` |
| where it only warns | `deepseek_v4.py:3123`, `mllama4.py:718` |
| where it logs at a computed level | `gemma4_unified.py:434` (`logger.log(level, ...)`, DEBUG bucket for non-persistent buffers) |
| where it is commented out | `gemma3_causal.py:903-907` (upstream, `9d02bb3e2a`, 2025 — not a fork defect) |

So the model-level remedy binds in two files out of 186, both fork files, both
speculative-draft loaders — i.e. exactly where the incident happened and nowhere
else — and the loader-level remedy that was written to generalise it reaches, per
#505-A1-01, three draft classes and no GGUF boot. That is a REACH INCLUDES PARAMETERS
result in the structural rather than the numeric sense: the mechanism exists, is
correct, is tested, and covers a few percent of the surface on which the same silence
is possible. The remedy was hoisted once already, for exactly this reason, and the
hoist did not travel.

*Task #505-X1:* `promote the load-completeness check to a shared helper and apply it to the TARGET bring-ups (Qwen3.5/3.6, Gemma4, DSV4/GGUF), not only the draft path` — this is the target-side twin of #505-A1-01 and #505-A1-08; the three should be scoped together rather than fixed one loader at a time.

Note on `gemma3_causal.py`: independently of the commented-out check, `loaded_params.add(name)`
sits at `:902`, one indent level OUT of the block that assigns `name` in the inner
loop, so the set records only the last name of each outer iteration. Upstream code,
recorded for completeness, not proposed as fork work.
