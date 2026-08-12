# HANDOFF — bug bundle #643 / #644 / #647

Branch `fix/loader-pd-bundle-643`, worktree `/spinning/wt-643-bundle`, base
`aca5037531` (on `feat/route-a-631`, newer than the then-current
`origin/feat/route-a-631` = `3be93fa943`). Desk shift, no GPU boot, no
arbitration taken. Three disjoint defects, far from the flip line. Nothing
merged — the operator sequences.

Errors first. Each bug is reported as **fixed**, **refused**, or
**recipe-gated**, and the recipes are the honest part: what could not be
established at a desk is named rather than implied.

**On the line numbers in this document:** every `file:line` citation refers to
the tree **as it was at base `aca5037531`**, i.e. before the three code
commits on this branch (a fourth carries this document). That is deliberate — the citations exist to let a reader verify
the DEFECT, and the defect only exists in the pre-fix tree. Lines in the files
this branch touched have shifted; use `git show aca5037531:<path>` to read
them at the quoted coordinates.

---

## 0. THE ONE-LINE STATE PER BUG

| # | verdict | what actually ships |
|---|---|---|
| #643 | **REFUSED** (not repaired) | A non-divisible PD TP pair now raises a named error at five points, earliest at the decode arm's handshake. The general split is NOT implemented — recipe in §1e. |
| #647 | **FIXED** | A dense BF16/F16 router gate keeps `.weight` and arrives as real values, via a table-driven rule in the generic iterator. Pinned by a synthetic-GGUF test. |
| #644 | **FIXED** | The MoE expert set is released from host anon memory once the parameter is materialized, and the `torch.stack` second full copy is gone. The mmap half needed nothing — it was already handled. |

The single most important number: the #643 sweep found **30 sites** carrying
the unguarded arithmetic, against the "6+" the task named. The family
precedent ("the count grows") held, and it undercounted by 5x.

---

## 1. #643 — PD-STAGING SILENT CORRUPTION

### 1a. The defect, demonstrated before it was guarded

The PD KV transfer path splits heads with a bare floor division,
`total_kv_heads // attn_tp_size` on each side
(`common/staging_buffer.py:698-699`), then maps one source rank's slice onto
one destination rank's. That is a partition only when one TP size is an
integer multiple of the other.

Prefill TP=3 against decode TP=2, 6 KV heads — the smallest non-divisible
pair — computed, from the shipped code, in pure integers:

```
src rank 0 -> dst heads [0,2)
src rank 1 -> dst heads [2,4)
src rank 2 -> dst heads [1,3)
dst rank capacity = 6 // 2 = 3 heads, valid indices [0,3)
```

Four independent defects fire at once, none of which said anything:

1. **Overlap** — heads 1 and 2 are each written by two source ranks. One
   rank's KV silently overwrites the other's.
2. **Out of range** — head 3 is written into a slice that holds 3 heads.
3. **Undersized staging / early completion** — `num_writers = src // dst`
   (`staging_buffer.py:736`, `staging_handler.py:92-97`,
   `mooncake/conn.py:1366`) is 1 for a 3->2 pair, so the region is sized for
   one writer while three write into it, and the receiver declares the chunk
   complete after the first arrival.
4. **Ranks never contacted** — `common/conn.py:534-539` strides by
   `info.attn_tp_size // self.attn_tp_size`; for 3->2 the stride is 1 and
   **prefill rank 2 is never contacted at all**, its heads never transferred.

Defect 4 is upstream of every head computation: the *pairing* is wrong before
a single head is sliced. The only thing standing at the pairing point before
this change was a `logger.warning_once` about **performance**
(`common/conn.py:517-521`).

### 1b. Why refusal and not a correct split

A correct partition already exists in this tree —
`draft_kv_canonical.local_head_window` (`draft_kv_canonical.py:189-224`),
largest-remainder, every head owned exactly once. Its own docstring
(`:198-207`) names `compute_head_slice_params` as the thing it declines to
reuse:

> `staging_buffer.compute_head_slice_params` takes `num_kv_heads // tp_size`
> and would make that (1, 1, 1), dropping head 3 — silently, since nothing
> downstream counts heads. That is the arithmetic wall general reslicing runs
> into, and the reason this function does not reuse it.

The wall is structural, not a bad expression: `compute_head_slice_params`
returns **one** head count used as both the gather and the scatter extent
(`staging_buffer.py:717` returns `num_heads_to_send` twice), so it cannot
represent "this source rank owns 2 heads, this destination rank wants 1" —
which is exactly the non-divisible pair. Repairing it means changing the
transfer loop structure in three transport backends, and that cannot be shown
correct without a two-instance PD boot. So: refuse loudly now, recipe for the
split in §1e.

### 1c. Where the refusal sits, and why not at parse time

`python/sglang/srt/disaggregation/common/tp_pair.py` (new) —
`HeadSplitNotRepresentable` + `validate_tp_pair_divisible`. Deliberately
dependency-free: no torch, no triton, no sglang import, because the live
handshake path must not pull triton to ask an integer question.
`staging_buffer.py` re-exports both names so the historical import site keeps
working.

Called from five points:

| where | file:line | role |
|---|---|---|
| decode handshake | `common/conn.py`, after the #631a identity guard, before `_resolve_rank_mapping` | the earliest point the pair exists |
| chokepoint | `staging_buffer.compute_head_slice_params` | defence in depth |
| chokepoint | `staging_buffer.compute_staging_layout` | the writer-count truncation |
| prefill mirror | `mooncake/conn.py` `send_kvcache_slice` | after its existing #641 raise |
| prefill mirror | `nixl/conn.py` `_init_hetero_tp_prep_handle`, `mori/conn.py` `_build_tp_slice_config` | ditto |

What an operator actually sees, verbatim:

```
[PD handshake with prefill arm at 10.0.0.4:8998] PD KV transfer refuses a
non-divisible TP pair: prefill attn_tp_size=3, decode attn_tp_size=2
(total_kv_heads=6). Neither size divides the other (3 % 2 = 1), so the head
split this path performs is not a partition: source ranks overlap on
destination heads, writes land past a destination rank's head capacity, the
writer count truncates so the receiver completes a chunk early, and some
prefill ranks are never contacted at all. Before #643 this corrupted KV
silently and answered with fluent wrong output. Choose prefill and decode
attn_tp_size values where one is an integer multiple of the other (e.g.
decode 2 with prefill 4, or equal sizes).
```

**A parse-time gate was considered and is impossible.** Unlike #636 and #642,
which are single-arm properties decidable from `ServerArgs`, this is a *pair*
property: a PD arm's `ServerArgs` never names the peer's TP size. Verified —
`disaggregation_prefill_tp|decode_tp|prefill_tp_size|decode_tp_size` has no
hit in `server_args.py` or `arg_groups/`. A gate in
`pd_disaggregation_hook.py` would have nothing to read.

One deliberate robustness property: `validate_tp_pair_divisible` uses
`total_kv_heads` **only in the message text**, never in the decision. The
decision is purely the two TP sizes. That matters at the handshake, where
`kv_args.total_kv_head_num` may not be populated yet — the guard is correct
either way, and at worst the message reports `total_kv_heads=0`. Do not
"improve" this by gating the check on a non-zero total; that would silently
disable it exactly where it is needed earliest.

The prefill mirror is placed in the slice functions, **not** in the
registration listener (`mooncake/conn.py:1762`, `nixl/conn.py:1296`,
`mori/conn.py:681`), because that listener is a background thread where a
raise dies quietly — the "fires from inside the sender, later still" failure
mode #636 records.

### 1d. What is deliberately NOT guarded

The other 25 of the 30 sites are downstream of the two pairing points and are
unreachable with a bad pair once the handshake refuses. Sprinkling 30 guards
would add noise without adding safety. The full site table, with per-site
judgement, is in the investigation record; the clusters are:
`staging_buffer.py` (12), `staging_handler.py` (3), the mooncake/nixl/mori
inline copies (9), `common/conn.py` rank mapping (6).

Two sites are worth naming because they are *partially* guarded and could
mislead a future reader:

* `mooncake/conn.py:1372-1385` — the **only** genuinely uneven-aware site in
  the tree: it uses a registered prefix-sum offset and range-checks it. Its
  fallback branch (`:1384`) reverts to the broken `%` form. This is the
  precedent for what "doing it properly" looks like.
* `mori/conn.py:803-807` — catches destination-token *overflow*, but a
  non-divisible pair typically *under*-fills, so it stays silent for #643's
  shape.

### 1e. RECIPE — validating a general split on metal (not done this shift)

Requires two instances and a card each; no arbitration was taken, so this is
written, not run.

1. Implement the split as interval intersection, not a pairwise formula: for
   `(src_rank, dst_rank)`, intersect
   `local_head_window(total, src_tp, src_rank)` with
   `local_head_window(total, dst_tp, dst_rank)` — reuse
   `draft_kv_canonical.local_head_window`, do not write a second one. The
   return type must carry `(src_start, dst_start, n)` with **independent**
   source and destination extents; the current 4-tuple cannot express it.
2. Change the three transports' writer loops from "one writer per
   `src // dst`" to "iterate the non-empty intersections", and size the
   staging region from their sum.
3. Hermetic gate first: extend
   `test_pd_staging_tp_divisibility_643.py::HazardTest` with a coverage
   assertion — for every `(total, src_tp, dst_tp)` in a sweep, every
   destination head is written exactly once, no write exceeds capacity, and
   every source rank is contacted. That test must pass before any boot.
4. Boot: prefill TP=3 / decode TP=2 on this rig's three cards, mooncake
   backend, and compare decode output token-for-token against a TP=2/TP=2
   pair on the same prompt and seed. Byte-identical or the split is wrong.
5. Only then relax `validate_tp_pair_divisible`, and relax it **per
   transport** — the guard should stay for backends whose loop was not
   converted.

---

## 2. #647 — BF16 DENSE ROUTER GATE LANDS AS `.qweight`

### 2a. The defect

`gguf_quant_weights_iterator` decides quantized-vs-dense with one string
comparison against one type name — `weight_utils.py:1448` for the type
marker, `:1517` for the payload:

```python
if weight_type.name != "F32":
    name = gguf_quantized_name(name, "qweight")
```

`GGMLQuantizationType` distinguishes F32, F16 and BF16 as three separate
unquantized types, and the layer's own notion of unquantized is
`UNQUANTIZED_TYPES = {F32, F16, BF16}` (`layers/quantization/gguf.py:199`).
The iterator's test is narrower than that set **by exactly F16 and BF16**.
That one-line gap is the whole bug.

Chain to the wrong answer:

1. `blk.N.ffn_gate_inp.weight` maps to `model.layers.N.mlp.gate.weight`.
2. The iterator renames it to `...mlp.gate.qweight` and emits a
   `...mlp.gate.qweight_type` beside it.
3. The router gate is built `ReplicatedLinear(..., quant_config=None)`
   (`models/qwen3_moe.py:313-317`) — it owns a `.weight` and nothing else.
4. Both renamed tensors are dropped without a word by
   `if name not in params_dict: continue` (`qwen3_moe.py:1204-1205`,
   `:1246-1247`).
5. The gate keeps its uninitialised values. The MoE routes on garbage —
   fluent wrong output, the #212 shape, not a crash.

**Measured, not asserted.** Against a synthetic GGUF holding a BF16 router
gate, the pre-fix iterator emits:

```
PRE-FIX : ['...input_layernorm.weight', '...mlp.gate.qweight', '...mlp.gate.qweight_type']
WITH FIX: ['...input_layernorm.weight', '...mlp.gate.weight']
```

Both orphans on the left; neither has a parameter to land in.

### 2b. Why the existing carve-out does not catch it

`gguf_adapter_base.unquantized_module_prefixes` (`:239-291`) skips any module
whose name lacks `"proj"` (`:269-271`), and a router gate is `mlp.gate`. It is
also F32-only by design: dense F16/BF16 shards are meant to stay
quantized-resident (task #64), and for a module that HAS a quant method that
is correct — the GGUF layer dequantizes them (`gguf.py:1250-1256`). The bug is
specifically modules with **no** quant method.

### 2c. The fix

Table-driven, in the generic iterator, so every family **and** the
no-adapter path are covered by one rule:

* `weight_utils.GGUF_DENSE_PARAM_SUFFIXES` — HF-name suffixes that are always
  plain dense parameters. Currently `(".gate.weight",)`.
* `weight_utils.is_dense_gguf_target(hf_name, weight_type)` — fires only when
  **both** the name matches the table **and** the stored GGML type is
  genuinely unquantized (F32/F16/BF16). The type condition is what keeps the
  rule from misfiring: a module that really is quantized stays on the
  quantized path even if its name matches.
* `weight_utils.gguf_dense_payload` — reinterprets the raw uint8 payload
  gguf-py returns for F16/BF16 (last dimension doubled). Generalised from
  `gguf_deepseek4._bytes_to_bf16`, which did this for one family only.

Applied at both dispatch sites, so the orphan type marker is suppressed too.

Per "extend, don't duplicate": #479 was checked and does **not** touch this
dispatch — its two commits are documentation, a test, and an offload refusal
message; `git blame` puts `weight_utils.py:1448-1449` and `:1513-1522` on
upstream `85e1a6f3aa5` and the #391 commits. The thing that WAS duplicating is
`gguf_deepseek4.py:366-375`, and that is now the generalised rule's special
case; its branches are retained as a family fallback for a gate stored under
some other non-F32 type, with a comment saying so.

### 2d. Open risk, stated plainly

The table has **one entry**. It was derived from the router-gate case and from
the two bespoke repairs already in tree (`gguf_deepseek4.py:366-375`,
`gguf_qwen35.py:503-525` — the latter a dense F16 `out_proj`, which is a
`"proj"` and therefore covered by the F32 carve-out route, not by this table).
No corpus sweep over real GGUF checkpoints was run this shift, so **other
dense non-`proj` modules may exist that this table does not name**. The
failure mode if so is unchanged from today's (silent drop), not worse.

**RECIPE — corpus pin (not run):** iterate
`/spinning/llm_stuff/club-3090/models-cache/*.gguf`, build each file's name
map, and report every tensor whose GGML type is in
`{F32, F16, BF16}` and whose HF target is not a `"proj"` module and does not
end in a table suffix. Each hit is either a new table entry or a justified
exclusion. This needs the checkpoints on disk but no GPU.

---

## 3. #644 — GGUF LOADER HOST-RAM DOUBLE-RESIDENCY

Reported in §3a-§3d below.

### 3a. What the trace established

The claim "the loader keeps the host tensor after the H2D copy" is **true, but
not where it reads**. Two distinct host populations, with opposite remedies:

* **Mapped (page cache)** — `gguf.GGUFReader` uses `np.memmap`, so
  `ReaderTensor.data` is a view into a mapping, never an in-RAM copy. This
  half is **already handled**: `ConsumedPageDropper` (`gguf_shards.py:676`,
  factory `:953-971`, `MADV_PAGEOUT` with a `MADV_DONTNEED`+`fadvise`
  fallback at `:896-935`) is wired into the GGUF iterator at
  `weight_utils.py:1466`, `:1470-1472`, `:1524-1525`, `:1527`. Nothing to do,
  and `mincore`-based tests will report it working.
* **Materialized (anon)** — created at `weight_utils.py:1519-1521`
  (`torch.tensor(...)` always copies). `madvise` cannot touch it; it needs an
  explicit reference release.

Note for the record: the page-drop machinery is **#391**
(`a8a2f7bc22`), not #408 as the task text said; #408 is the separate
safetensors whole-file drop at `weight_utils.py:970`.

The important consequence: **a `mincore` residency test cannot see this bug.**
It measures file page residency only, and would report the dropper healthy
while host anon still holds the model. Any falsifier must measure anon/object
liveness instead.

### 3b. Where the anon copies survive

* **Plain dense/column/row/vocab params — not retained.** After `copy_` only
  the generator's local holds a reference, rebound next iteration.
* **MoE experts — retained permanently on the default path.** This is the
  strongest form of the claim. `fused_moe_triton/layer.py:1318-1324` stores
  each expert's CPU tensor in *two* places (`expert_data_map` and
  `data_container`); the default branch at `:2758-2763` does
  `torch.stack(...)` — a second full host copy of the layer's expert set —
  then materializes, copies, and `continue`s **without clearing either
  container**. The offload branches do clear (`:2772`, `:2780-2781`), so a
  plain GGUF MoE boot at resident fraction 1.0 takes the leaking branch.
* **Merged/QKV linears — retained for the whole load pass.**
  `linear.py:979-980`, `:1014-1015`, `:1780-1781` append `.narrow()` *views*,
  which pin the full untrimmed host storage per rank; they are cleared only in
  `gguf.py:1295` / `:1668`, and those clears are gated on
  `len(data_container) > 1`, so a single-shard container is never cleared.

### 3c. Verdict — FIXED

One hunk at `fused_moe_triton/layer.py:2758`. `data_container` is cleared
**before** the fill (with both holders aliasing, releasing one frees nothing,
so this is what lets `drop()` return each expert's bytes as it lands), the
parameter is materialized from the `row_shape`/`dtype` the source already
reports, filled expert-by-expert with `del` + `drop` per expert, and
`expert_data_map` cleared after. The `torch.stack` second full copy is gone.

One semantic guard was added on purpose: `torch.stack` **rejected** a ragged
or mixed-dtype expert set, while `copy_` would broadcast or cast it silently.
The rejection is kept as an explicit `ValueError` naming the parameter and the
offending expert index, so the refactor cannot convert a hard failure into
silent corruption.

Red-first, **re-verified independently** by reverting the hunk and re-running
(an agent's report is not evidence):

```
data_container still holds 8 live expert tensors        (object level)
two whole-set torch.stack calls, one per parameter      (peak level)
8728576 bytes beyond the parameter survived materialization
  of a 16 MiB expert set                                 (RSS level)
```

The object-level assertion is the one that pins the fix. **The RSS assertion
is corroborating only** and says so in the test: an earlier draft with a 2 MiB
payload *passed on broken code*, because that is below the allocator noise
floor. Anyone tightening it should raise the payload, not the threshold.

Post-fix: 6 passed; `unit/model_loader/` + `unit/quantization/` **467 passed,
29 skipped, 94 subtests** (= base 455 + 6 from #647 + 6 from #644, no
failures, skips and subtests unchanged); `tests/moe_offload/` 20 passed;
`unit/layers/moe/` 214 passed.

Cost: per-expert H2D replaces one bulk copy — for a 128-expert layer, 128
pageable copies of order 10 µs against one full host stack allocation plus
memcpy per layer. Expected net win. **Not measured, and not claimed.**

### 3d. Deliberately not changed

* `linear.py:979-980`, `:1014-1015`, `:1780-1781` append `.narrow()` views
  that pin full host storages, but they are cleared during
  `process_weights_after_loading` (`gguf.py:1295`, `:1668`) — a transient
  load-time peak, not permanent residency. Bounding it needs a `.clone()`
  that trades peak for a copy and is a net loss at TP=1, where `narrow`
  returns the full range anyway.
* The `len(data_container) > 1` clear gates at `gguf.py:1260`, `:1651` are
  **correct, not leaks**: `create_weights` sets the container on every linear
  while only the merged/QKV sites append, so the length is 0 or >= 2. A
  length of 1 would leave `qweight` unmaterialized and fail at forward time
  first. Reasoned from the append sites, **not** from an exhaustive
  enumeration of call sites — recorded as unproven.
* `tests/moe_offload/test_gguf_moe_offload.py:285` is now **misnamed**:
  `test_materialize_default_path_still_builds_the_full_stack`. It still
  passes — it asserts shape and byte identity, which this change preserves —
  but the name now asserts something false, since the default path no longer
  builds a stack. Left alone to keep the diff minimal; rename when next
  touched.

---

## 4. HOW TO RUN WHAT IS HERE

```
cd /spinning/wt-643-bundle
PYTHONPATH=/spinning/wt-643-bundle/python CUDA_VISIBLE_DEVICES=99 \
  /spinning/htsglang-gpu/.venv/bin/python -m pytest \
  test/registered/unit/disaggregation/ \
  test/registered/unit/model_loader/ \
  test/registered/unit/quantization/ -q

./scripts/run_631_flip_family.sh -q
```

`PYTHONPATH` is not optional — without it the run tests
`/spinning/htsglang-gpu/python` and the result is meaningless. Check that
traceback paths say `/spinning/wt-643-bundle`.

### Measured, base vs branch

Baselines were taken on the untouched base tree before any edit, so "green"
here means "no worse than the tree I started from", not "the suite passes".

| suite | base `aca5037531` | this branch |
|---|---|---|
| `run_631_flip_family.sh` | 1095 passed | **1095 passed** |
| `unit/disaggregation/` | 225 passed, 190 subtests | **227 passed, 195 subtests** |
| `unit/model_loader/` + `unit/quantization/` | 455 passed, 29 skipped, 94 subtests | **467 passed, 29 skipped, 94 subtests** |
| `tests/moe_offload/test_gguf_moe_offload.py` | — | 20 passed |

The two extra tests in `unit/disaggregation/` are #643's; the +12 in the
loader/quant column is 6 from #647 and 6 from #644. The flip family is
unchanged, which is the point — this branch is disjoint from the flip line and
must not move it.

### A pre-existing red that is NOT this branch

`test/registered/unit/distributed/` carries failures at base. Measured on both
trees, whole directories, same interpreter and same `CUDA_VISIBLE_DEVICES=99`:

| `unit/server_args/` + `unit/distributed/` | result |
|---|---|
| pristine base `aca5037531` (scratch worktree) | 22 failed, 3303 passed, 12 skipped, 1048 subtests |
| this branch | 22 failed, 3303 passed, 12 skipped, 1048 subtests |

Identical in every column. Narrowed to the two owning files, again on both
trees: `test_uneven_dcp_pool_geometry.py` and `test_uneven_tp_memory.py` give
8 failed / 40 passed on base and 8 failed / 40 passed here.

So these are red **before** any change in this bundle. Do not attribute them
to it — and do not let them sit, either: they are in a directory people run,
which means they are actively training readers to ignore a red result. They
belong to whoever owns the uneven-DCP pool geometry and should be filed.

Per-bug numbers are also in the commit messages, each against the same
baselines.

---

## 5. WHAT THE NEXT SHIFT SHOULD NOT REDO

* Do not look for a parse-time gate for #643. It cannot exist; §1c has the
  grep that settles it.
* Do not "fix" `compute_head_slice_params` in place. Its return type is the
  wall; §1b.
* Do not write a `mincore` test for #644. It cannot see the bug; §3a.
* Do not add a second largest-remainder split. Use
  `draft_kv_canonical.local_head_window`.
* Do not widen the #647 dispatch to "all non-quantized types keep `.weight`".
  Dense F16/BF16 landing on `.qweight` is **correct and intended** for modules
  that have a quant method (task #64, `gguf.py:1250-1256`); only
  quant-method-less modules are the bug.
* Do not gate #643's check on a non-zero `total_kv_heads`. §1c.
* Do not chase the `unit/distributed/` reds as a regression from this branch.
  They are red at base; §4.

---

## 5b. RECIPE — the one boot this bundle needs

Not run: no card was taken this shift. Claim a window via `/spinning/gpu-arb/`
(holder file + heartbeat; stop the heartbeat BEFORE releasing) and do not
disturb serving on 30030 or the router on 30099.

Target: any GGUF MoE checkpoint, since #647 and #644 both sit on that path —
the Qwen3.6-35B-A3B UD-Q4_K_XL file the #651 strand uses is the obvious one.

1. **#647, the load-time proof.** Boot with the branch and grep the load log
   for the router gate. The gate must appear as a loaded `...mlp.gate.weight`.
   Cheaper and stronger than reading logs: after load, assert in-process that
   every `mlp.gate.weight` is finite and not all-zero — an unloaded gate keeps
   its initialization, so `.abs().sum() == 0` or a `GGUFUninitializedParameter`
   class on the parameter is the tell.
2. **#647, the behaviour proof.** Same prompt and seed, branch vs base. If the
   gate was previously unloaded the routing changes, so outputs SHOULD differ,
   and the branch's output should be the coherent one. This is the rare case
   where a diff is the pass condition — record both outputs.
3. **#644, the memory proof.** Sample host RSS of the loading process across
   the whole load (100 ms, `/proc/<pid>/status` `VmRSS`), branch vs base, same
   checkpoint. Compare the PEAK and the post-load PLATEAU. The plateau is the
   one that matters: base should sit roughly one expert-set higher than
   branch. Report both curves, not a single number.
4. **#644, the correctness proof.** Byte-identical output, same prompt and
   seed, branch vs base. This one must NOT differ — unlike #647, #644 changes
   only where bytes live, never their values. A diff here means the
   expert-by-expert fill is wrong and the ragged-set `ValueError` did not
   catch it.

Order matters: run 4 before 2, so a #644 byte difference is not misread as
#647's intended routing change.

---

## 6. THE THREE THINGS MOST LIKELY TO BE WRONG HERE

Stated so the next shift attacks these first rather than re-deriving them.

1. **#647's table has one entry.** `GGUF_DENSE_PARAM_SUFFIXES = (".gate.weight",)`.
   It was derived from the router-gate case and the two bespoke repairs in
   tree, not from a corpus sweep. If another dense non-`proj` module exists,
   it is still silently dropped today. Recipe in §2d; it needs checkpoints on
   disk but no GPU, so it is the cheapest real risk to close.
2. **#643 refuses a configuration that is physically transferable.** Prefill
   TP=3 / decode TP=2 is a legitimate thing to want on a three-card rig, and
   this branch makes it a hard error rather than making it work. That is the
   right trade against silent corruption, but it is a capability regression
   for anyone who was unknowingly running one — they will see a new refusal
   where they previously saw (wrong) output. Worth a release note.
3. **Nothing here has been on a GPU.** Three commits, zero boots. The #643
   guards are integer arithmetic and safe; #647 changes what a real GGUF load
   emits; #644 changes how a real MoE parameter is filled. #647 and #644 both
   deserve one boot of a GGUF MoE checkpoint before anyone trusts them in
   production — that is the single highest-value follow-up, and it needs a
   card and an arbitration window this shift did not take.
