# NOTE 706 remainder — determination on the harvest-composite lineage

Base `c546eed923` + today's stack. Four questions, answered with file:line, and
one gap found and fixed red-first.

## 1. Is the geometry-neutral format the ACTIVE write path? — ACTIVE, gated

Not dark, but flag-gated:

* `server_args.py:5550` `phase_flip_canonical_kv_page`, default **off**;
* `server_args.py:8075-8090` refuses it without
  `--hicache-storage-backend file` ("no other backend implements" the partial
  writes a page assembled across stages needs);
* `cache_controller.py:660-672` builds the window **only** when the flag is on
  ("Built only when the format is switched on").

**The harvest boot ran it ON** (`boot_735_comp4.log`:
`phase_flip_canonical_kv_page=True`), so on this lineage it is the live write
path, not built-but-dark.

## 2. Cross-BOOT retention — the key drops geometry, the HASH did not

The suffix side is correct. `hicache_storage.py` builds
`kv_config_suffix = f"_{model_name}"` and then adds the tp suffix only when
`not is_mla_model and not dcp_owner_mode and canonical_kv_page is None`, and
the pp suffix only while "a stage's page holds just that stage's layers". Under
the canonical page both are dropped, with the rule stated in place: "the key
carries exactly the axes the stored bytes depend on".

**The identity hash was not given the same treatment, and that is the gap.**
`cache_controller.py` called `compute_model_identity_hash(server_args)`, whose
`include_parallel_vectors` defaults to True, so `rank_tp_ratio` and
`rank_kv_ratio` entered the key. Those are geometry — how the KV is SPLIT
across ranks — not a property of a canonical page, which holds every attention
layer at full width.

Live, not hypothetical: the harvest boot ran `rank_tp_ratio=None` (falsy,
skipped) **and `rank_kv_ratio='coupled'` (truthy, appended)**. A geometry term
was in the key today, so two boots of the same model and kv-dtype with
different kv-ratios write byte-identical pages and miss each other.

**Fixed** (`canonical_identity_hash_for`): the parallel tail is dropped when
and only when the canonical format is on. The flag to do it already existed
(#631a guard 1); it simply was not used here. Revision, dtype, quantization and
kv_cache_dtype are NOT dropped — they describe the bytes, and confusing two
byte formats is the silent wrong hit the hash exists to prevent.

## 3. Cross-PHASE within a boot — uniform, and for a structural reason

The key has no phase term at all once the pp suffix is dropped, and the
identity hash is built from `ServerArgs`, which do not change across a flip.
So a prefix written in the PP phase and read in the TP phase produces the
**same key by construction** rather than by coincidence.

The #718/#719 rebind work disarms the DEVICE tier off-phase; the host/disk path
keyed above is unaffected by that disarm, which is what makes the host tier the
phase-uniform one. Note this is a reading of the key construction, not a
measurement — item 3's boot proof is still F4-r5's.

## 4. Expected hit pattern, and an instrument caveat that matters more

The soak driver (`soak_driver.sh`) runs **4 concurrent growing sessions**:
`hist = hist + response`, ~220 new tokens per turn, and each request sends
`hist[-48000:]` plus a fixed suffix.

**That last detail bounds the hit rate independently of #706.** While a
session's history is under 48000 chars the prompt is a strict prefix-extension
of the previous turn, so nearly the whole prompt should hit. Once it exceeds
48000 chars (~12k tokens, roughly 50 turns at 220 tokens/turn) the driver sends
a SLIDING WINDOW: the prefix shifts every turn and prefix reuse collapses to
near zero — by construction, in the harness, with a correct cache underneath.

This is the most likely reading of the #740/#703 open rest (absolute 4/121):
a low absolute rate over a long soak is what this traffic model produces
regardless of #706. **An absolute hit rate is therefore the wrong acceptance
number for the harvest boot.**

What the cache report SHOULD show if #706 works:

* **Per-session, per-turn, while under the window:** `cached_tokens` on turn N
  ≈ the prompt length of turn N-1 (hit ratio approaching
  `1 - 220/prompt_tokens`). Zero here is a real failure.
* **The #706 acceptance proper — ACROSS A FLIP:** a turn served in the TP
  decode phase whose session had a previous turn in the PP prefill phase must
  still report `cached_tokens > 0`. A drop to 0 exactly at a flip boundary,
  with a non-zero value on the turns either side, is the phase-uniformity
  failure signature and the only one that indicts #706.
* **After the window starts sliding:** near-zero is EXPECTED and must not be
  read as a #706 failure.

Concrete acceptance for F4-r5: take the first flip in the harvest log, find the
next request from a session that already ran a turn before that flip, and
require `cached-token > 0` on it. That is one number, it is decidable from the
existing cache-report line, and it is not confounded by the 48k window.
