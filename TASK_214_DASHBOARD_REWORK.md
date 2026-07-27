# Task #214 — one-click rig pairing over the network, and the dashboard rework

Branch `feat/dashboard-rework`, based on `integration/r3-probe-next2`.

This file is the durable record of the analysis and the decisions. It is
written for a reader who has no chat context: everything needed to continue
the work is here.

## 0. Baseline as measured on the base commit

Environment: no GPU (`CUDA_VISIBLE_DEVICES=99`), tools from
`/spinning/htsglang-gpu/.venv/bin`, `PYTHONPATH` pointed at this worktree's
`python/` directory (the venv has a different checkout installed).

| gate | command | baseline |
| --- | --- | --- |
| tests | `pytest test/registered/unit/planner test/registered/unit/rigmon` | 1067 passed, 1 failed, 38 skipped |
| ruff | `ruff check python/sglang/srt/{planner,rigmon} tools/rig_dashboard` | 20 errors |
| codespell | `codespell --config .codespellrc <same paths>` | 8 hits |
| mypy | `mypy --ignore-missing-imports --follow-imports=skip <own modules>` | 67 errors |

The one failing test is pre-existing and unrelated to this task:
`test_webui.py::TestNewHttpRoutes::test_reference_png_static_route` fails
because `python/sglang/srt/planner/assets/quality_chess_reference.png` does
not exist in the tree, so `GET /assets/quality_chess_reference.png` returns
404. Nothing in this task touches that asset.

None of these four numbers may grow.

## 1. What already exists (inventory)

### 1.1 Three separate HTTP servers, easy to confuse

| server | module | default port | role |
| --- | --- | --- | --- |
| planner web UI | `srt/planner/webui.py` | 8780 | the dashboard this task reworks |
| rigmon aggregator | `srt/rigmon/aggregator.py` | 8770 | read-only multi-node telemetry store |
| rig_dashboard | `tools/rig_dashboard/server.py` | 8770 | the older standalone prototype |

The last two collide on port 8770 by default. They are not meant to run at
the same time; rigmon's package docstring names `tools/rig_dashboard` its
predecessor. `planner/hardware.py` loads `rig_dashboard/server.py` by file
path to reuse `sample_nvml`, and `rig_dashboard/server.py` imports
`planner/crossover.py` — that is the whole coupling.

### 1.2 `srt/planner/webui.py` (6529 lines)

* Lines 1–2582: the API functions, one per endpoint, all pure-Python and
  GPU-free.
* Lines 2583–2830: `_Handler`, a `BaseHTTPRequestHandler` with prefix
  matching in an if-chain. Route order is load-bearing
  (`/api/detect_endpoint` must be tested before `/api/detect`).
* Lines 2849–6529: `INDEX_HTML`, one raw string holding the whole
  front end — CSS 2856–3080, markup 3082–3665, script 3667–6526.

Seven top-level views: landing, bench, landscape, energy, quality, explore,
runner.

### 1.3 The compute layer the dashboard must call rather than re-implement

* `placement.compute_placement(model_cfg, flags)` returns per-card
  `segments: [{id, label, mib, detail, replicated, replication_factor,
  replication_reason}]` that sum to `total_mib`. This is the granular VRAM
  breakdown the expert view needs. Its docstring already states that the web
  UI renders these and never re-derives them.
* `capacity.CapacityReport` / `RankCapacity` give the coarse per-rank totals
  the simple view needs (`budget_mib`, `budget_used_pct`,
  `max_context_tokens`).
* `feasibility.plan(...)` returns `PlanResult` with `fits` plus
  `infeasible_reasons`, and raises `PlanRejected` (which carries
  `.reasons: List[str]`) for structurally invalid input. Both shapes must be
  surfaced; they are different things.
* `flags.catalog()` is the real knob catalog — `FlagSpec` carries `type`,
  `allowed`, `requires`, `mutually_exclusive_with`, `tuple_len_flag`.
  `flags.cross_field_errors(...)` produces the hard rejects.
* `roofline.estimate_roofline` / `roofline.roofline_energy` produce the
  speed and energy estimates, both tagged `provenance="planner-estimate"`.
* `device_map.device_map()` bridges NVML and CUDA index spaces. Card dicts
  must be keyed through it, not by raw index.

### 1.4 rigmon

* `capabilities.probe_all(env=None) -> CapabilityReport` is the capability
  table. `Capability` has `key, label, state, reason, evidence, probe`;
  `state` is one of `active / available / unavailable / unknown`. Rows
  include `nccl_colocation` (runtime `ncclGetVersion`, needs 2.30+) and
  `mps` (existence of the MPS control directory). There is **no** `remedy`
  field on `Capability` — remedy prose is folded into `reason`.
  `ProbeEnv` (`capabilities.py:188`) is the injection seam that makes all of
  this testable without hardware.
* `compat.check_compatibility(local, remote) -> CompatReport` is the join
  gate. `CompatCheck` **does** have `remedy`. Verdicts `BLOCK / WARN / OK`.
* `transport.choose_all_pairs(probe, available_facilities, colocated_pairs)`
  turns a measured pair matrix into ranked `TransportOption`s with verdicts
  `recommended / usable / unavailable / unknown`, and emits an `unknown`
  entry for every pair in `probe.missing_pairs()` so an unmeasured matrix
  never looks complete.
* `probe.ProbeResult` is the #213 short-probe result. It is **not persisted
  by rigmon**; it is reconstructed from `~/.cache/sglang/hw_profile-*.json`
  via `probe.from_hardware_profile(...)`.
* `facilities.facilities(env, ...)` says what this host can measure or
  control, and why not. `transport.TransportSpec.requires` refers to these
  facility keys.

### 1.5 What is built but reaches no UI today

* `managers/kv_session_offload.budget_stats()` — its own docstring says "for
  the dashboard"; its only caller is a log line.
* `layers/moe/expert_offload.expert_offload_released_device_bytes()` and the
  `ExpertOffloadRelease` tally — the reclaim posts.
* `runner.window_metrics(...)` already produces `ms_per_verify_round`,
  `ms_per_decode_round`, `ms_per_1k_prefill_tokens`, `ms_per_draft_pass`,
  `accept_length`, `verify_ct`. The canonical metric definitions live in
  `scenarios.py:268–312`. None of these reach the front end.
* `bench_suite` measures `ttft_ms` and `prefill_tps` into `result.detail`,
  which the front end drops on the floor (`benchEvent` reads only
  `metric`/`status`/`reason`/`detail.http_code`).
* `crossover.CrossoverFinding` is rendered by the *old* rig_dashboard, and
  used indirectly by the planner to shape the max-perf preset, but has no
  panel of its own in the planner UI.
* `counters_every` (renamed from `profile_every` in task #220) is only
  visible as `snapshot()["cadence"]["counters_every"]`. One caller still
  uses the old spelling: `planner/cli.py:495`.

## 2. The defects the rework has to fix

Diagnosed against the current `INDEX_HTML`, with evidence.

1. **Refresh destroys collapse state.** `landing_config` is rewritten by
   `renderLanding` every 2 s (`webui.py:4839`), and `renderStartConfig`
   builds two `<details>` blocks inside it ("full launch command + env",
   "raw server_info"). Opening either collapses again within 2 s,
   unconditionally. The `<pre>` scroll position goes with it.
2. **Search box slams sections shut.** `filterFlags()` sets
   `det.open=false` for every `cfg-section` and `det.open=!!q && vis>0` for
   every generated flag group whenever the query is empty
   (`webui.py:5192–5200`). Clearing the search closes everything the user
   had opened by hand.
3. **Boot log re-opens itself.** `renderServerStatus` forces
   `boot_log.open=true` on every 2 s poll while booting
   (`webui.py:6122`), so a manual collapse does not stick.
4. **Refresh overwrites typed input.** `autofillQuality()` writes
   `q_endpoint` and `q_model` unguarded on *every* switch to the quality
   tab (`webui.py:6267`). `applyProfile()` resets every rendered flag
   control. `gguf_choice`, `dl_quant` and `profile_select` are rebuilt via
   `innerHTML` without preserving the current selection — unlike
   `applyFieldStates()`, which does it correctly and is the model to follow.
5. **No timeouts, no cancellation, no de-duplication.** Zero
   `AbortController` occurrences; no `AbortSignal.timeout`; no in-flight
   guard. `landingPoll` can overlap itself when a snapshot takes longer than
   2 s, and `onFlagChange` fires `resolveFlags` + `refreshRunnerPlacement` +
   `schedulePlan` together with no ordering, so a slow `/api/placement`
   response can overwrite a newer one.
6. **Undebounced slider traffic.** `mrrFromSlider` / `mrrFromNum` call
   `doPlan()` directly on every tick (`webui.py:5277`, `5282`), one POST per
   pixel.
7. **Dead subsystem.** `liveScrape` / `toggleLive` / `remeasureNow`
   (`webui.py:5928–5965`) address `live_target`, `live_res`, `live_btn`,
   `live_out` — none of which exist in the DOM. Their route
   `POST /api/live` is unreachable. Remove.
8. **Bench results drop their best numbers.** See §1.5.
9. **No simple/expert split** exists at all; the only progressive
   disclosure is one `advanced_toggle` and five `<details>`.

## 3. Design

### 3.1 Update foundation (etappe 2)

Add a small runtime at the top of the `INDEX_HTML` script, then retrofit the
render sites. No framework, no CDN — the page stays self-contained.

The tree diff itself is **morphdom 2.7.8** (MIT, no dependencies, 12 KB),
vendored under `srt/planner/assets/` and inlined into the page at import
time. It supplies the algorithm; the policy is ours and lives in the
`onBeforeElUpdated` hook next to `setHTML`. See §3.1.1 for the reasoning.

* `setHTML(el, html)` — the call site replacement for `el.innerHTML = html`.
  Runs the new markup through morphdom against the live tree instead of
  replacing it, with a per-element cache of the last applied string so
  identical markup does no DOM work at all. A panel the user is typing
  inside is deferred until focus leaves it.
* `_beforeElUpdated(fromEl, toEl)` — the policy. A field being edited is
  skipped entirely, subtree included. The live `open` state of a `<details>`
  is written back onto the incoming markup so the reader's collapse wins over
  the renderer's. Scroll offsets are captured and restored.
* `_nodeKey(n)` — `id`, or `data-key` for nodes that have no business owning
  a global id.
* `flagRowHtml` now renders the current value into the markup
  (`value=`, `checked`, `selected`). Under a patcher the markup is the
  description of state, so a row that claims to be empty would wipe what the
  user entered. This also fixes the older bug where a flag-surface re-render
  dropped every entered value: `_flagSettings` held the truth but nothing
  wrote it back into the DOM.
* `api(path, opts)` — the only fetch entry point. Adds
  `AbortSignal.timeout`, a per-key in-flight registry that aborts the
  previous request under the same key (last-write-wins becomes correct
  rather than accidental), and a generation counter so a late response is
  discarded instead of applied.
* `stale(el)` — marks a panel as refreshing without replacing its content,
  so a slow call shows the previous numbers dimmed instead of a spinner. The
  dim engages only after 250 ms, so the normal fast answer produces no
  visible change at all.
* `debounce(fn, ms)` — one implementation, applied to every slider and text
  input that triggers a backend call.

The three offending forced-state writes (§2.2, §2.3) become
"only on first render" or "only when the user has not touched it".

#### 3.1.1 Third-party code: what is taken in, and what is not

The rule for this branch: prefer a small, established, permissively licensed
library over a local reimplementation — and never replace something the repo
already does well.

**Taken in.**

* **morphdom 2.7.8** (MIT, zero dependencies, 12 KB minified, single file),
  vendored at `python/sglang/srt/planner/assets/morphdom-umd.min.js` with its
  licence beside it. *For:* the DOM tree diff. *Why not built here:* the diff
  is fiddly in exactly the places that matter — reordering keyed nodes, and
  the form elements whose attribute and property values diverge
  (`<option selected>`, `<input value>`, `<select selectedIndex>`,
  `<textarea>`). A first local attempt was written and then dropped in favour
  of this; it was a worse version of the same code. The page inlines it at
  import time rather than linking it, because the dashboard must work on a
  machine with no internet and CDN links are barred.
* **esprima** (BSD, test-only, registered under the `test` extra in
  `python/pyproject.toml`). *For:* parsing the embedded `<script>` in a unit
  test. *Why:* the whole front end is one string inside a Python module, so a
  syntax slip is invisible to every Python test and surfaces only as a blank
  page in a browser. This closes that gap in CI.

**Deliberately not replaced.** The planner already does these well, and a
package would be a downgrade: `PerfCostModel` / `roofline` / `capacity` /
`placement` (all sizing and cost arithmetic stays server-side),
`planner/scrub.py` (redaction), `planner/github_share.py` (token handling,
preview-and-confirm, update-in-place via a body marker),
`planner/scenarios.py` plus `tools/rig_dashboard/studies/` (the study
machinery), `rigmon/capabilities.py` and `rigmon/compat.py` (the capability
table and the join gate).

**Still open.** The GitHub Discussions API needs GraphQL. That is a single
`POST` of a JSON body to one endpoint, which `urllib` already does in
`github_share.py`; a GraphQL client library would be larger than the code it
replaces. Etappe 6 follows the existing `urllib` pattern.

### 3.2 Simple and expert views (etappe 3)

A `view mode` selector persisted in `localStorage`, defaulting to simple.

* **Simple:** per card, one bar showing total VRAM use against the card's
  total, and one slider "maximum VRAM to use" per card. Nothing else beyond
  the model selector, the load/eject actions and the verdict.
* **Expert:** the existing `placement.segments` breakdown, plus every
  dimension the fork allows as a live control — `rank-tp-ratio`,
  `rank-mlp-ratio`, `rank-vocab-ratio`, `rank-kv-ratio`, reserves,
  `max-running-requests`, context length, and the rest of the flag catalog.
* **Live propagation:** every change posts the whole current flag set to a
  single recompute endpoint and renders the returned dependent values. The
  arithmetic stays server-side in `placement` / `capacity` / `roofline` /
  `feasibility`; the front end never re-derives a number.

Templates (`flags.profiles(...)`: `uneven-max-tokens`, `uneven-max-perf`,
plus the `rank_perf_tune` objective `both / dec / enc / maxkv`) are starting
points. Applying one seeds the controls; the controls keep their full range
afterwards. Slider bounds come from the hard rejects
(`flags.cross_field_errors`, `PlanRejected`, card totals), never from the
template value.

### 3.3 Rig pairing, task #214 (etappe 4)

New module `srt/rigmon/pairing.py`. rigmon today has no client that reads a
*remote* rigmon; it only receives pushes. This module adds one.

1. **Couple.** `reach(url, timeout)` performs `GET <remote>/api/nodes` and
   `GET <remote>/api/snapshot` and reports reachability, round-trip time and
   the remote node identity. Target host and access path are entered in the
   UI or pre-filled from the local environment; no value is baked into the
   repository.
2. **Compatibility gate.** `gate(local, remote)` runs the existing
   `compat.check_compatibility` and holds the two `capabilities.probe_all`
   tables against each other. Every unmet precondition is listed with its
   reason and its remedy, per the existing table rule. Nothing is greyed out
   silently.
3. **Transport.** `propose_transport(...)` calls
   `transport.choose_all_pairs`. When #213 measurements exist for the pair
   it recommends from them. When they do not, the pair is reported as "not
   yet measured" together with an offer to run the probe. The button
   triggers the existing study machinery; it never boots anything
   cross-rig by itself.
4. **Result.** The flow ends in a validated start configuration — an argv
   plus environment block to copy, or a handover to the existing start path.
   No cross-rig boot happens in this task.

### 3.4 Benchmark and chess windows (etappe 5)

Running and finished runs separated. Results as tables of
configuration / measure / value. `ms_per_verify_round` and
`ms_per_1k_prefill_tokens` are the lead metrics, alongside `ttft_ms` which
`bench_suite` already records and the front end currently discards. The
chess quality suite gets the same layout.

### 3.5 Discussion export (etappe 6)

New module `srt/planner/discussion_export.py`.

* A bundle composer with selectable packages — bench table only, bench plus
  system, bench plus system plus energy, and the energy-efficiency metrics
  in their sensible groupings.
* Preview before sending; Markdown output.
* System details always pass through `planner/scrub.py`
  (`scrub_text`, `scrub_launch_flags`, `scrub_path`). No IP, path or host
  name may reach the export. Card models and driver versions may.
* GitHub Discussions need the GraphQL API — there is no GraphQL client
  anywhere in this tree, so one is added here.
  `planner/github_share.py` is the pattern to follow for token handling,
  preview plus confirmation, and update-in-place via a body marker.
* **Gated.** With no discussion URL or ID configured, the button renders the
  preview and reports "no target configured". Nothing is created or posted
  automatically.
* The token is read from a file whose path comes from the environment, is
  never logged, and never appears in a URL — only in the `Authorization`
  header, and every error path runs through `github_share.redact`.

## 4. Conventions for this branch

* No emoji anywhere, including the UI.
* Colour is used only to distinguish states, never for decoration.
* No real environment values in repository files. Hosts, ports, paths and
  tokens come from environment variables with placeholder defaults in the
  `${VAR:-<placeholder>}` form.
* Each etappe ends with the four gates of §0 at or below baseline, then a
  commit and a push.

## 5. Open points that only a real boot can settle

Recorded here rather than claimed as done:

* Cross-rig reachability against a second physical rig — the pairing flow is
  exercised against an injected transport in tests only.
* The GraphQL create/update path is exercised against a mocked API, the same
  standard `github_share.py` holds itself to.
