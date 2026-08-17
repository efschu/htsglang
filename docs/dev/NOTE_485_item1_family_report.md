# #485 item 1 — BF16-resident family reporting: partially fixed, and sharpened

Date: 2026-08-17. Desk only, no boots.

`NOTE_485_joint_phase_vectors.md` §7 item 1 reads: "`checkpoint_compute_format_families`
does not report the BF16-resident GDN family as diverging, so #324 hands it the
checkpoint-wide lane. The bracket contains the right answer but does not know it
is the right answer."

That note is **not on this branch** (it lives on the `audit/*` lineage); the
same amendment is owed there and is a merge-train item.

## What was wrong, and is now fixed

`_per_family_formats` learned about per-family schemes from two places only:
ModelOpt's `quantized_layers`, and compressed-tensors' `config_groups` **when
`len(groups) > 1`**. A checkpoint applying one scheme in one group and then
excluding whole families through `ignore` therefore reported no split at all.

Two changes, both in `uneven_perf.py:_per_family_formats`:

* `config_groups` is now read at ANY group count. A single-group config can
  still describe a split, because the split may live in `ignore` rather than in
  a second group.
* A GEMM family named in `ignore` contributes **bf16** evidence. A family the
  quantizer was told to skip is bf16-resident, and that is a fact about the
  checkpoint exactly as much as a declared scheme is. Non-GEMM ignores
  (`re:.*norm.*`, `re:.*conv1d.*`) map to no family and correctly manufacture
  nothing — pinned by a reverse test.

A family carrying BOTH bf16 and quantized evidence is now left OUT rather than
collapsed. `_dominant` picks by CONFIG-ENTRY count, which bears no relation to
how many LAYERS carry each scheme, so collapsing would invent the answer.

## What this does NOT fix, verified against the real checkpoint

Running the fixed function on `Qwen3.8-27B-INT8-yarn1.5` still returns `{}`.
Item 1 is **two problems, not one**, and only the first is a reporting bug:

1. **Class-selector attribution.** The checkpoint declares its single group as
   `targets: ["Linear"]` — a module-CLASS selector, which
   `gemm_family_of_module` (`uneven_perf.py:1982`) maps to no family. So the
   quantized side is invisible, every remaining signal comes from `ignore` and
   is bf16, and the caller's uniformity check collapses it exactly as before.
   Attributing a class selector means "every GEMM family EXCEPT the ignored
   ones" — a complement, not an enumeration.
2. **The family vocabulary cannot express it.** `GEMM_FAMILY_ATTN_GDN`
   (`:1963-1977`) deliberately spans `self_attn` AND `linear_attn`, and this
   checkpoint quantizes the first while ignoring the second. That family has no
   single correct key at any granularity this function reports at. The real
   split is 48 linear-attention layers against 16 full-attention ones, which is
   per-LAYER information — #371's census — not per-config-entry.

**So closing item 1 needs the per-layer census plus class-selector
attribution.** It is desk work, but a bigger slice than §7 implied, and it
should not be re-listed as a one-line reporting tweak.

## Interactions checked

* **#371** (per-layer family census, `uneven_perf.py:2842`, `:2879`, `:3693`)
  is untouched: this change reads the quantization config, not `layer_types`.
  It is also the mechanism item 1 now depends on.
* **#324** consumers (`uneven_perf.py:5340`, `planner/key_solver.py:1314`) read
  the returned values as `_FORMAT_LANES` keys. A contract test asserts every
  reported value is one. Writing it caught a real hazard: `_ct_group_format`
  returns `int8_a16` for weight-only int8 (`:2044`), which has **no lane table**
  and would drop #324 into its loud bf16 fallback. The serving checkpoint is
  W8A8 and yields `int8`, a real key — but a W8A16 checkpoint would now surface
  that gap more often than before. Named here rather than left to be met.

## Tests

7 tests, red first (3 red, 4 pre-existing-behaviour pins green from the start).
Mutation-proven: reverting the group-count guard reds the two reporting tests;
dropping the mixed-family guard reds the mixed test. 120 passed / 10 skipped
across the planner consumer suites, unchanged.

---

# Item 1 CLOSED (2026-08-17, second slice)

The two structural problems named above are both fixed, and the acceptance is
the one this note itself set: the fixed function on the real checkpoint no
longer returns `{}`.

```
Qwen3.8-27B-INT8-yarn1.5
  checkpoint-wide : int8
  per-family      : {'moe': 'int8', 'mlp': 'int8', 'vocab': 'bf16', 'attn_gdn': 'bf16'}
```

## 1. Class-selector attribution, as a COMPLEMENT

`_is_class_selector` recognises a bare module-class target (`Linear`) and
distinguishes it from a path (`model.layers.0.mlp`) or a regex
(`re:.*mlp.*`) — both of which name specific modules and must not be widened.
A class selector then contributes its key to **every GEMM family the `ignore`
list does not name**, derived from `_ALL_GEMM_FAMILIES` rather than spelled out
a second time.

The complement direction is the load-bearing choice: claiming a family the
quantizer was explicitly told to skip would overwrite a bf16-resident fact with
a scheme that was never applied. Two reverse pins hold the line — a regex target
and a dotted path each stay narrow.

## 2. The ATTN_GDN span, resolved by LAYERS

`_weigh_attn_gdn_by_layers` resolves the attention family's mixed evidence using
#371's per-layer counts: `linear_attn` evidence weighs the GDN layers,
everything else in the family weighs the full-attention ones. Reversing the
split reverses the answer, which is what makes it a measurement rather than a
vote — `_dominant`'s config-entry count could never do that.

**#371 was reused, not rebuilt.** Its census (`layer_family_census`,
`uneven_perf.py:2905`) covers the `block_configs` axis; the GDN-hybrid axis was
already derived from `layer_types` (`:3712-3717`, `full_layers` / `gdn_layers`).
No extension was needed — the per-layer truth existed and was simply not
reaching this function. The caller now derives the same split and passes it.

## 3. What the report still cannot say, disclosed rather than hidden

On this checkpoint `attn_gdn` resolves through the COMPLEMENT rule (its only
evidence is the ignored `linear_attn`), not the layer-weighted one. So
`attn_gdn=bf16` is a **majority statement**: 48 GDN layers are bf16-resident,
and the 16 full-attention layers the class selector would otherwise have
quantized are **not described by that key**.

An earlier draft of this slice printed "resolved per layer" there, which
overclaimed. The description now states the span and names the layers the key
does not cover, so a reader cannot assume uniformity:

```
[attn_gdn spans 48 linear_attention / 16 full_attention layers;
 the key describes its majority, and does NOT describe the 16 full_attention layer(s)]
```

Reporting a single key per family is the contract #324 consumes; carrying a
genuine per-layer vector would be a different return type and a larger change
than item 1. **Named as the residue rather than pretended away.**

## 4. `int8_a16` — declared, not lane-registered

`FORMATS_WITHOUT_LANES` records weight-only INT8 as **recognised but
unmeasurable**, and `rank_gemm_scores` now says so in its own branch. It is
deliberately NOT added to `_FORMAT_LANES`: that table's own comment says
registering a lane the serving path cannot take "would make the plan lie", and
this tree has no weight-only int8 arm. The fix is the #606 distinction — "known
format, no measurable lane" versus "unrecognised format" — not a fabricated
lane.

## Tests

20 tests. Mutation-proven: disabling class-selector recognition and disabling
the layer weighting together red 7, including all four real-checkpoint
acceptance tests. 133 passed / 10 skipped across the planner consumer suites
(was 120 before this slice; the delta is these tests). `Mapping` was missing
from the typing import — caught by ruff, invisible at runtime under postponed
annotations; `uneven_perf.py` is back to its 6 pre-existing findings.

## Named follow-up: the per-layer VECTOR contract

The majority key shipped here is honest but lossy, and the lossy part is
exactly what a downstream consumer is starting to need. `attn_gdn=bf16` says
nothing about the 16 full-attention layers, and the planner's family-placement
work is precisely a question about GDN-versus-full-attention layer splits.

**Follow-up:** return a per-layer-resolved vector, not a single key per family,
for the families that span more than one layer kind.

* **Consumer that needs it:** the per-(rank, family) scores (#324) and the
  family-placement pricing of the #705 class — anything that prices a stage by
  what its layers actually cost. A majority key charges 16 full-attention
  layers at the GDN family's bf16 rate, which is the wrong lane for them.
* **Feed:** #371's census already carries the counts (`layer_family_census`,
  and `full_layers` / `gdn_layers` from `layer_types`); no new measurement is
  required, only a wider return type.
* **Why not here:** it changes the contract `checkpoint_compute_format_families`
  publishes and every #324 consumer reads. That is a deliberate interface
  change with its own red-first slice, not a widening smuggled inside a
  reporting fix.

Until it lands, the majority key stands WITH its disclosure — the description
names the layers the key does not describe, so a consumer that cares can see it
is being handed a majority rather than a uniform fact.
