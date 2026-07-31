# Task #342 -- Frontend Information Architecture v2: survey

Scope of this document: a compact survey of navigation and control-placement
patterns from five existing model-serving / fine-tuning UIs, and the IA
decision they inform for this dashboard's nav skeleton (Task #342). This
task builds the skeleton and the Models hub's registry binding only; visual
design finalization (colors, spacing, iconography) is explicitly deferred to
a later, dedicated task and is out of scope here.

## Surveyed products

### LLaMA-Factory WebUI (LlamaBoard)

A Gradio-based UI divided into four top-level areas: Training, Evaluation &
Prediction, Chat, Export. Each area is a single page holding every control
for that concern (dataset, method, hyperparameters, etc.) rather than being
split across further sub-pages. Pattern taken: **one top-level tab per
task-verb** (train / evaluate / chat / export), not per artifact type.
(Sources: [WebUI Usage -- DeepWiki](https://deepwiki.com/llm-factory/LLaMA-Factory-Doc/2.3-webui-usage),
[WebUI -- LLaMA Factory docs](https://llamafactory.readthedocs.io/en/latest/getting_started/webui.html))

### H2O LLM Studio

Experiment-centric: creating a run opens a dialog that shows the *commonly
used* settings first, with additional fields (e.g. LoRA adapter options)
appearing conditionally once a relevant choice is made (Training Mode =
lora/qlora); settings that are rarely touched (tokenizer, architecture,
environment, inference parameters) live in a separate "Advanced
Configuration" YAML editor rather than in the main dialog. Once an experiment
runs, it exposes its own sub-tabs -- Charts (live train/val loss + metrics),
Summary, Train data insights (first batch, for verifying data representation
before wasting a training run on a misconfiguration), and Chat (available
only after training completes). Pattern taken: **progressive disclosure**
(default fields visible, expert fields behind an explicit expand/advanced
step) and **per-run live charts as their own view**, separate from the run's
configuration. (Sources: [Create an experiment](https://docs.h2o.ai/h2o-llmstudio/guide/experiments/create-an-experiment),
[View and manage experiments](https://docs.h2o.ai/h2o-llmstudio/guide/experiments/view-an-experiment),
[Experiment settings](https://docs.h2o.ai/h2o-llmstudio/guide/experiments/experiment-settings))

### OpenAI fine-tuning dashboard

A two-pane layout: a left panel lists jobs, the right panel shows the
selected job's detail (hyperparameters, training metrics, checkpoints). Job
creation is reachable from the same dashboard or via the API -- the UI is a
thin, optional layer over the same contract the API exposes, not a separate
surface with its own logic. A separate Playground UI exists specifically for
comparing model outputs/quality/performance side by side. Pattern taken:
**list-then-detail** for a collection of long-running, named jobs, and
**the UI composes the same API a script would call**, never a private
shortcut. (Sources: [List fine-tuning jobs -- API reference](https://developers.openai.com/api/reference/resources/fine_tuning/subresources/jobs/methods/list),
[Model optimization guide](https://platform.openai.com/docs/guides/fine-tuning))

### LM Studio

A left sidebar switches between a small number of task-oriented top-level
areas -- Chat, Local Server ("Developer" -- REST API/CLI/advanced config),
model search/download -- each occupying the full remaining window rather
than being nested inside another view. The currently loaded model and its
memory usage are shown persistently in the sidebar/status area regardless of
which top-level area is active. A "Developer Mode" setting is an explicit,
sidebar-toggled escape hatch that reveals advanced controls across the app
(model loader dialog, etc.) rather than showing them by default. Pattern
taken: **persistent sidebar with the load/resource state always visible**,
and **an app-wide simple/advanced toggle** rather than per-page ones.
(Sources: [LM Studio Tutorial -- DataCamp](https://www.datacamp.com/tutorial/lm-studio),
[Introducing LM Studio 0.4.0](https://lmstudio.ai/blog/0.4.0),
[LM Studio Developer Docs](https://lmstudio.ai/docs/developer))

### ComfyUI

A left sidebar with distinct, icon-labeled sections -- Assets, Nodes, Models,
Workflows, Templates -- each a flat list/browser, separate from the large
central node-graph canvas that is the actual work surface. A bottom toolbar
holds cross-cutting utilities (console/logs, shortcuts, settings) that apply
regardless of which sidebar section or workflow is open. A right-hand control
area holds the primary action (run/queue) and the queue's live state.
Pattern taken: **primary action and its live status pinned to one fixed
corner** (top/right), **logs and cross-cutting diagnostics pinned to a fixed
edge** (bottom), independent of whichever content view is open in the middle.
(Sources: [ComfyUI Interface Overview](https://docs.comfy.org/interface/overview),
[ComfyUI User Interface -- Wiki](https://comfyui-wiki.com/en/interface/basic))

## Cross-cutting patterns and their application here

| Pattern (surveyed) | Applies to this dashboard? |
|---|---|
| One top-level tab per task-verb, not per artifact (LLaMA-Factory) | Yes -- the new group bar (Models / Playground / Training / Video & Media / Rig / Benchmarks / Settings) is organized this way already; see mapping below. |
| Progressive disclosure: common fields default, rare fields behind an explicit expert step (H2O) | Already the Guide/Playground tab's own idiom (three steps, then the full flag surface as the expert step); the Models hub's add-engine dialog follows the same shape (engine_id/model/tp_size/placement up front, no separate expert step needed yet -- the registry's EngineSpec surface is still small). |
| List-then-detail for a collection of named, long-running items; UI composes the same API a script would call (OpenAI) | Directly applicable to the Models hub: engine cards are the list, actions are the same `/registry/*` calls documented in `sglang.srt.registry.http_api`, proxied server-side (see `registry_snapshot_payload` and neighbors in `webui.py`) so the page and `curl` never disagree. |
| Persistent sidebar with load/resource state always visible; app-wide simple/advanced toggle (LM Studio) | The dashboard already has a persistent "loadbar" above the tab strip (which model is loaded); the existing simple/expert `view_mode` toggle predates this task and is left as-is. Not otherwise adopted in v1 of this skeleton -- a left sidebar (vs. the current top tab strip) is a layout change reserved for the design-finalization pass named in the task's scope note. |
| Primary action pinned to a fixed corner; logs/diagnostics pinned to a fixed edge (ComfyUI) | Not adopted in this skeleton pass -- the existing top-bar-only layout is kept unchanged in this task ("no design polish": existing look is preserved, only structure changes). Worth revisiting at design finalization. |
| Wizard + presets + expert-step collapse | Matches the existing Guide tab exactly (three steps -> command, expert step = full planner); reused as-is under the new Playground group, not rebuilt. |

## Resulting nav-group skeleton (this task)

A two-level navigation was added over the existing single-row tab strip: a
new top-level group bar, and the existing per-tab row filtered to show only
the active group's own tabs. No existing tab was renamed, deleted, or had
its DOM id changed; only a `data-group` attribute and membership in the new
`NAV_GROUPS` JS table were added. See `python/sglang/srt/planner/webui.py`,
the comment above `<div class="tabs" id="nav_groups">`, for the exact
mapping and rationale, reproduced here:

| Group | Tabs (unchanged ids) | Status |
|---|---|---|
| Models | `models` | New (Task #342): registry (M1) hub. |
| Playground | `wizard` | Pre-existing "Guide" tab, regrouped, not rebuilt. |
| Training | `training` | New, stub only -- real content is Task #341. |
| Video & Media | `video` | New (Task #342): read-only M2 job-list binding. |
| Rig | `landing`, `data`, `pair` | Pre-existing Monitor/Data/Pair-rig tabs, regrouped. |
| Benchmarks | `bench`, `quality`, `history` | Pre-existing Benchmark/Quality/History tabs, regrouped. |
| Settings | `about` | Pre-existing About/Update tab, regrouped. |

Deep links: a minimal `#<group>/<tab>` hash router was added
(`showGroup`/`routeFromHash` in `webui.py`) so a reload or a shared link
reopens the same tab instead of always defaulting to `landing`. No hash
routing existed before this task, so this is new capability rather than a
preserved one; every tab that already existed remains reachable by its
original id regardless of the hash router.

## Explicitly out of scope for this task

- Visual redesign (colors, spacing, iconography, a left-sidebar layout as
  seen in LM Studio/ComfyUI): the task's hard scope note defers this to a
  later, dedicated design-finalization task.
- Training tab content (Task #341).
- Any write path against the M2 video service (this task's Video & Media
  tab is read-only by design).
