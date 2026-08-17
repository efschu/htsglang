# #485 completion verdict — the per-family × per-phase matrix

**2026-08-17, Slot-3. Verdict only: nothing was built, changed, or wired for
this document.** Every claim below was spot-verified against the tree rather
than taken from a survey summary.

---

## 1. The directive, verbatim

Source: `/spinning/htsglang/CLAUDE.md:87-96` (user law, 2026-08-03). Note it is
**not tracked in this worktree** — `git ls-files` finds no `CLAUDE.md` here and
there is no deletion commit; it lives in the main clone.

> **PER-FAMILY x PER-PHASE OPTIMA (user law, 2026-08-03):** every component
> family (KV heads, GDN/mamba states, KV cache, linear layers per quant lane,
> nonlinear kernels, vocab, experts, ...) has its OWN optimal distribution PER
> phase/regime — and this holds for EVERY workload on this rig: any LLM,
> diffusion, SR/video chain stages, TTS/ASR, training tenants. The cards differ
> on every resource axis, so "one layout serves both phases" is a red flag,
> never a default assumption [...] Single-family/single-axis arms are
> DIAGNOSTIC only — never phrase their result as a phase-level verdict.

Echoed as **THE MATRIX DOCTRINE** in `docs/dev/FEATURE_CATALOG.md:315-399`
("rows are FAMILIES, columns are PHASES (#485) ... A layout is not one vector.
It is a table") and in `docs/dev/NOTE_485_joint_phase_vectors.md:57-61`, which
states plainly that "Slice 1 delivers the **prefill column** ... The decode
column is untouched here."

**The ticket-number caveat resolves cleanly.** #485 has two facets and they are
the same ticket, not a collision: Facet A is the matrix doctrine and its
prefill-column solver; Facet B is the seam/memory certification that gates
whether Facet A's cut can be armed on metal.

---

## 2. DELIVERED

### The prefill column, solved per family

| commit | what it delivered |
|---|---|
| `c08f61348e` | #485 slice 1: prefill cut per FAMILY, not per MLP vector. Adds `_ATTN_VECTOR_SHARDS`, `_attn_candidates`, `_with_attn`, `_shard_fractions`, `mamba_pool_bytes_for` and a joint `(mlp_vector, attn_vector)` candidate space in `uneven_perf.py`. |
| `b675630b7b` | merge of the above into the main line |
| `b3c980232d` (dup `06e03d2ab8`) | `planner/pp_cut.py`: the PP layer-split gets its own attention-target vector, decoupled from the layer-count vector |
| `e645aa70c0` (dup `45706a17ce`) | gates `--pp-layer-ratio` on a hybrid-family census check |

### The prefill column's own second axis

| commit | what it delivered |
|---|---|
| `d82a778eb9` (#492) | the attention/KV family has TWO distribution axes, not one (replication + token-sharding). This is the doctrine's "a row can have more than one AXIS". |
| `447249a3f9` (#503) | correctness fix on #492: gates the attention token axis on the predicate that actually installs it |

### Facet B — the feasibility gate on that cut

`407b382646`, `f912c2e046`, `b2b61e9ca1`, `8149d963ec` and the `scripts/cert_485/`,
`scripts/seam_485/`, `docs/dev/485/` tooling. Its own conclusion is negative and
is part of the delivered state: **the certification threshold is not reachable
at the certification pool** under the current corridor law, after decomposing a
1838 MiB VRAM seam.

**Delivered, stated exactly:** one column of the matrix (prefill), for one
family boundary (attention vs GDN/linear), on one workload (the LLM), with a
second axis on the attention row — plus a feasibility gate that currently says
the cut does not fit.

---

## 3. SUPERSEDED

**`d937d5f76b` — "[#705] Desk verdict: REFUSE the family split; uneven TP
captures the win instead."** #705 was explicitly framed as the decode-column
member of this family (`docs/dev/NOTE_705_family_split_tp_decode.md:3`: "a
member of the #485 phase-matrix family applied to the DECODE phase, as #702 was
applied to prefill").

Its measured reasoning, which is sound and which I verified in the commit body:

| decode baseline | ms/round |
|---|---|
| solo on 5090 | 3.192 |
| EQUAL 1/3 shard (what runs today, `ratio=None`) | 2.506 |
| PROPORTIONAL (uneven TP, already shipped) | 1.726 |

Against the *honest* proportional baseline the family split beyond uneven TP is
worth **+0.090 ms/round (~0.3% of a ~30 ms bs=1 round)** in exchange for
concentrating a family on one rank, moving 4,688 MiB of residency, and taking
on the #115 zero-shard machinery. Capacity was never the obstacle — the world
total is unchanged at 1,432,230 tokens either way. The win was.

**What this supersedes is narrower than "the decode column".** It refuses ONE
candidate cell — *concentration* of a family on one rank — and establishes that
bandwidth-proportional sharding captures nearly all of the available gain. It
does not solve the decode column per family, and it does not show that other
families' decode optima coincide.

**A tension worth naming rather than smoothing over:** the directive says
"Single-family/single-axis arms are DIAGNOSTIC only — never phrase their result
as a phase-level verdict." `d937d5f76b` is a desk verdict on one family's
concentration, and it is worded as a phase-level refusal ("REFUSE the family
split"). Read strictly against its own governing law, it closes a candidate,
not the column. This verdict adopts the narrow reading.

---

## 4. GENUINELY OPEN — successor scope

Stated as crisply as the evidence allows, in descending order of how much of
the directive they leave unaddressed.

**O1. No `(family × phase)` structure exists anywhere in the tree.** Verified:
`planner/pp_cut.py` solves the prefill column only (`solve_pp_cut`,
`solve_pp_cut_for_prefill_speed`, `pp_phase_pool`; `tp_phase_pool` appears only
as a comparison term, never as a placement solve). `planner/family_split.py`
solves the decode column only — `solve_family_placement` returns
`FamilySplitSolution.by_family: Dict[str, FamilyPlacement]`, with **no phase
argument and no phase dimension in the return type**. No function returns a
structure indexed jointly by family and phase. The "matrix" is prose plus two
separately-invoked single-phase solvers. (Other planner "matrix" names —
`cost_model.PairMatrix`, `explorer.plan_matrix`, `wizard.build_matrix` — are
card-to-card bandwidth and model×rig capacity, unrelated.)

**O2. The decode-column solver is dead code.** `solve_family_placement` has
exactly two references in the tree: its own definition
(`planner/family_split.py:141`) and its test
(`test/registered/unit/planner/test_family_split_705.py:34,74`). Zero call
sites in any boot or runtime path. Whatever #705 concluded, nothing computes it
at run time.

**O3. Most families named in the law were never cut at all.** Delivered work
covers the attention vs GDN/linear boundary. The directive also names **vocab,
experts, nonlinear kernels, and linear layers per quant lane**. None of these
has a per-phase optimum solved, in either column.

**O4. The law is explicitly not LLM-scoped, and everything else is untouched.**
"this holds for EVERY workload on this rig: any LLM, diffusion, SR/video chain
stages, TTS/ASR, training tenants." No diffusion, SR, TTS/ASR or training-tenant
family has any per-phase treatment. This is by far the largest open surface and
it is invisible if #485 is read as an LLM-layout ticket.

**O5. The prefill column's feasibility gate is currently RED.** Facet B's
finding (`407b382646`) stands: the threshold is not reachable at the
certification pool. The delivered column is therefore desk-valid and not
arm-able as it stands.

**Suggested successor split**, so the residue is not carried as one vague
ticket: (a) O1+O2 as one engineering slice — give the two existing solvers a
joint return type and wire the decode one, which is the smallest change that
makes the word "matrix" true of the code; (b) O3 as a per-family enumeration
ticket scoped to the LLM; (c) O4 as its own programme, since it shares nothing
with the LLM layout machinery but the law; (d) O5 tracked with Facet B.

---

## 5. Verdict

**#485 is PARTIALLY DELIVERED and should not be closed.**

- The **prefill column** is delivered for the attention/GDN boundary, with a
  second axis on the attention row (#492, corrected by #503) — desk-valid, and
  gated RED by its own feasibility facet.
- The **decode column** has one candidate refused (#705) on sound measured
  reasoning, and its solver is unwired dead code. The column is not solved.
- The **matrix itself is not delivered**: no data structure in the tree is
  indexed by family and phase, so the doctrine's central claim — "a layout is
  not one vector, it is a table" — is not yet true of the code.
- **#704 / #704a / #704b are not #485 work** and should not be counted toward
  it. They are a PP-cut layout *ladder* and its arming machinery, which consume
  the prefill-column solve as an input. Counting them would inflate the
  matrix's delivery with work on a different axis.

The honest one-line status: **one of two columns, one of many families, one of
several workloads — and the table is still prose.**
