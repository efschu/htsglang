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
