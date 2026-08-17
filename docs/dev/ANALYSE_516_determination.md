# ANALYSE #516 — MoE-Offload x Graphs, re-evaluated per half

Read-only determination, base `4a16043d1a` (`train/0817-control`). No GPU, no
boot, nothing executed. #516 was filed as three halves before most of the
surrounding work landed; this asks, per half, what is delivered, what is
superseded, and what is genuinely open.

**Method note, because it changes one answer.** Commit ancestry is the WRONG
test here. Every task cited below has a first-matching commit that is *not* an
ancestor of this branch (`#254 42bebf8bcf`, `#439 79956c7a9f`, `#462
7c4b07b5ae`, `#494 038f3f431e`, `#390 6d5ad379ba`, …) — yet the code is
present and readable in this tree, because it arrived by MERGE rather than by
that commit. Delivery below is therefore asserted from CODE PRESENCE at
file:line, which is checkable, and never from "the commit is an ancestor",
which here is false and misleading.

---

## Half 1 — Breakable Route: **DELIVERED**

| piece | status | evidence (READ) |
| --- | --- | --- |
| #462 breakable MoE-offload graph route | present | `model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py`; flags in `server_args.py`, config in `model_executor/cuda_graph_config.py` |
| #494 CUDA-event break-cost instrument | present **and wired** | `utils/break_cost_clock.py`; imported and used at `breakable_cuda_graph.py:42`, `:297`, `:301`, and `layers/moe/expert_offload.py:120` |
| #468 `va_stable_required` correction | present | descriptor field in `model_executor/short_term_offload_register.py` (documented with the captured-graph reason); applied at `model_executor/offload_gdn_states.py:360` |

**The brief's open question — "did #494 reach the main lines?" — is answered
YES.** It is not merely present: `break_cost_phase` is imported by the expert
offload itself, so the instrument sits on the path it measures rather than in a
side tool.

Nothing open on this half.

---

## Half 2 — "Schlaues Layout": **BUILT, BUT OPT-IN**

| piece | status | evidence (READ) |
| --- | --- | --- |
| #254/#256 expert-major waves, fp8 presplit | present | `layers/moe/expert_offload.py`, `layers/moe/lazy_expert_staging.py`, `layers/moe/fused_moe_triton/layer.py` |
| #439 link-proportional compute assignment | present, own module | `layers/moe/expert_compute_placement.py` (the VRAM-neutrality fixed point is documented at `:566-575`) |
| #458 link-solve as default | **NOT default** | `compute_policy_label()` (`expert_compute_placement.py:560-562`) returns `os.environ.get(COMPUTE_POLICY_ENV, "") or "base-plan"`. `expert_stats._moe_compute_policy` (`:127-140`) states the same in prose: `base-plan` is "the truthful answer for every launch that did not ask for a solve" |

**So "schlaues Layout" is built and is not the default.** What remains of the
half is therefore NOT a build. It is one decision — whether the solve should be
the default — and that decision is **window-gated**, not desk-fundable: it turns
on a measured comparison of the solved vector against the base plan on this
rig, and #439's own module records that the first battery walked into a fixed
point (`:566`) precisely because the runtime did not honour the invariant the
solve assumed. That is the kind of thing a desk cannot settle.

INFERRED (marked as such): I did not find a gate that refuses making it
default; the opt-in shape reads as caution pending measurement rather than a
prohibition.

---

## Half 3 — Miss-Slot-Budget: **NOT BUILT** (and not refused either)

Searched `python/sglang/srt/` for `miss_slot`, `slot_budget`, `expert_miss`,
`miss_budget`: **zero hits, none of the four**.

What exists nearby, and why neither is a budget:

* **#390 `layers/moe/expert_stats.py` (689 lines) is an INSTRUMENT.** Its
  surface is `hit_rate()` (`:291`), `unique_hit_rate()` (`:297`), `snapshot()`
  (`:301`) and a collector with a periodic dump (`:353-386`). It records and
  reports; nothing in it caps, admits or refuses. The brief's own phrasing —
  "an instrument is not a budget" — is confirmed at code.
* **#302a `layers/moe/expert_heat_migration.py` is a PLACEMENT policy, not a
  budget.** It re-ranks the resident set on a decayed heat window and performs
  an **EQUAL-COUNT swap** against the pinned host pool (module docstring,
  `:1-13`). Equal-count is the point: it changes WHICH experts are resident,
  never HOW MANY misses may be spent. It is also off by default
  (`SGLANG_MOE_HEAT_MIGRATION`).

**On the absence claim, precisely.** The gate rule asks for the refusing gate's
file:line before declaring something absent. **There is no refusing gate — and
that is itself the finding.** A miss-slot budget is not refused anywhere; it
was simply never built. Those are different states and the difference matters
for whoever picks this up: there is no prior decision to overturn, no
architectural objection on record, and therefore no evidence bar to clear
before building — only the ordinary one of showing it beats the equal-count
re-rank that already exists.

**Desk-fundable**: the budget's *shape* (what a slot budget would cap, and
against which of `expert_stats`' existing counters) can be designed at a desk
against the recorded `expert_stats_*.json` data the #302a desk harness already
replays (`scripts/dev/302a_heat_desk/`). **Window-gated**: any claim that a
budget beats the current placement, since #302a's own evidence is a simulation
against recorded boots and its oracle ceiling (0.98) sits far above the
realised static rates (0.76-0.85) — the headroom is real but unproven live.

---

## Rollup

#516 asked for three things and the answer differs per half, which is why it
looked stuck. The **breakable route is done**, including the piece the brief
was unsure about: #494's break-cost clock is not only present but wired into
the expert offload itself. The **layout half is done as a mechanism and open as
a decision** — the link-proportional solve exists and is opt-in, so what remains
is a defaulting question that only a measurement can settle, not code. The
**miss-slot budget is the one genuinely unbuilt half**, and it is unbuilt rather
than rejected: nothing refuses it, the nearest neighbours are an instrument
(#390) and an equal-count re-rank (#302a), and neither spends a budget. #516
can close as determined, carrying one desk-fundable design item (the budget's
shape) and two window-gated questions (default the solve; prove a budget beats
equal-count re-ranking).
