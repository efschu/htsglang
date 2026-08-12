# HANDOFF: #656 KV-UNIVERSE R1 — four tickets, errors first

Branch `feat/kv-universe-656`, cut from `origin/integration/r2` and moved
forward onto **Merge R6 `481411ac6b`**. Comparison baseline for the code arm
is the frozen `/spinning/wt-merge-r6-base` tree (R6 kept it at the pre-merge
tree `598f570ba4`); the metal arm compares against the live ship capture
`/spinning/evidence-631/s485/ship_argv.txt`.

Evidence: `/spinning/evidence-631/kvuniverse-r1/` (RESULTS.md, boot logs,
generations). GPU window 22:11Z–22:30Z, released with the heartbeat stopped
first. Serving downtime ~17 min; ship restored and verified with three real
generations.

---

## 0. ERRORS FIRST — three premises this shift falsified

**E1. "Remove the `max_total_num_tokens` cap" is falsified ON METAL.**
Boot A set `--max-total-tokens 100000000`. The boot DIED:

```
cuMemCreate: committing 8388608 bytes would leave 116 MiB free, below the
1024 MiB corridor law floor ... would leave -4 MiB free
cuMemCreate: 8388608 bytes refused by the driver
RuntimeError: cuMemCreate failed: <CUresult.CUDA_ERROR_OUT_OF_MEMORY: 2>
```

The cap is **not** conservatism and not a soft flag that can simply be
deleted. Uncapped, the TP-phase pool sizes itself to its *own* budget and
eats the VRAM the PP pool needs. The order "the cap is to be REMOVED, not
justified" cannot be executed as a flag change; it requires the TP pool to be
**sized to the PP id space** in code. That code is NOT written (see §5).

**E2. "Move layers onto the 5090 to raise KV" is falsified ON METAL.**
Boot B set `--pp-stage-ratio 18,7,7`. PP0 died:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 810.00 MiB.
GPU 0 has a total capacity of 31.34 GiB of which 421.75 MiB is free.
```

Two facts behind it, both worth carrying forward:

* **torch gpu0 IS the 5090** (FASTEST_FIRST ordering); NVML index 1 is the
  5090. The rank→card mapping is NOT NVML order. Any reasoning that assumes
  `--rank-gpu-id 0,1,2` follows `nvidia-smi` order is wrong.
* **rank0 was already at its physical ceiling**: budget 31800 MiB of a 32607
  MiB card leaves 807 MiB — *below* the 1024 MiB corridor floor. The 5090 has
  no room to receive layers.

**E3. HANDOFF_662 §5's arithmetic for `[24,23,17]` used the wrong cost model.**
662 costed a layer move as *additive* on the receiving rank ("frees ~1.77 GiB
on rank0 but adds ~1.31 GiB on rank1"). The weight arena is a **max over
layouts**, not a sum — `phase_flip_boot.py`:

```python
arena_total = max(layout_pp.total_bytes, layout_tp.total_bytes)
```

So a rank whose TP weight share already exceeds its new PP layer share pays
**nothing** for received layers. Measured on the ship boot: PP weights
13482.18 / 8144.00 / 9114.95 MiB against TP weights 13692.29 / 7659.52 /
7659.52 — ranks 1 and 2 sit at their PP term, rank 0 at its TP term. 662's
conclusion (don't do it) still held for rank0, but for the reason in E2, not
the reason 662 gave.

---

## 1. T1 — PP-phase ceiling: WHAT BINDS IT, and the fix

**The binding constraint, named with numbers.** The pool is a **min-reduction
across ranks**. Boot A lifted the cap so every rank reports its true local
capacity under the ship vector `14,10,8`:

| rank | true local capacity | stranded |
|---|---|---|
| PP0 | 1252026 | 748076 |
| PP1 | **503950** | 0 — **THIS RANK BINDS** |
| PP2 | 727592 | 223642 |

The rig was serving at **503950 tokens while 971718 token-slots were
physically present and unusable** on the other two ranks.

**Why nobody could see it:** the ship cap of 620000 made PP0 and PP2 each
report exactly `620000`. The cap did not lower the pool (the min was below
it) — it **masked the imbalance that would have told you where the capacity
went.** That is the real cost of the cap, and it is why T1 and T2 are one
ticket.

**Why the token vector alone cannot fix it.** Per-rank KV cost is
`T x 32 KiB x max(layer_share, token_share)` (HANDOFF_662 §5, HANDOFF_663).
rank1's cost is **floored by its layer share**, so lowering only
`SGLANG_UNEVEN_TOKEN_VECTOR` buys nothing on the binding rank. Either the
layers move (E2 says they cannot move onto the 5090) or the budget changes.

**The fix, metal-proven (boot C).** The stranded resource is not the layer
ratio at all: `--rank-gpu-memory-mib 31800,14000,15600` gave the two RTX 3080s
**14000 and 15600 MiB out of 20480 each — 11.3 GiB of VRAM the engine was
never handed.** Boot C keeps the ship vector `14,10,8` untouched and only
rebudgets to `31500,16000,16000` (rank0 down 300 MiB to buy back corridor
headroom on the 5090):

| quantity | ship | boot C |
|---|---|---|
| pool / id space | 503950 | **620000 (+23.0 %)** |
| ranks stranding capacity | 2 of 3 (971718 tok) | **0 of 3** |
| binding rank | PP1 | none — the CAP binds |
| tracebacks | — | 0 |
| server | — | fired up 22:23:52 |
| real generations | 3 (ship baseline) | **3**, accept 2.29–2.67 |
| NVML free, idle | — | 1879 / 3460 / 3261 MiB |

Every rank now reports `THIS RANK BINDS; 0 stranded`. 620000 is the cap, not
a rank — **the PP phase is no longer the ceiling.**

**Is 620000 the optimum? No, and it is not claimed to be.** 855 / 2436 / 2237
MiB remain above the 1024 corridor floor, so more is backable. The next
binding thing is the TP-phase pool (§2), not the PP layout.

---

## 2. T2 — the 500000/620000 cap

**Where it is set:** nowhere in `python/sglang`. It is the boot flag
`--max-total-tokens` (`server_args.py:1151`, default `None`), passed through
verbatim to `MemoryPoolConfig.max_total_num_tokens` and written as `self.size`
at `mem_cache/memory_pool.py:2481`. No `min()`, no clamp, no assertion.

**Is it structural? No.** `ReqToTokenPool.req_to_token` is
`dtype=torch.int32` (`memory_pool.py:373-375`) — ~2.1e9 addressable, four
orders of magnitude above any pool discussed here. **There is no integer
width or allocator row limit to widen.** The order to widen one has no target.

**So why can it not just be deleted?** Because of E1: removing it OOMs the
boot. The documented reason is confirmed by the boot A failure —
`PROD_BRINGUP_BENCH.md:642-663`: the TP layout can only address ids the PP
allocator issues, so an uncapped TP pool allocates VRAM it can never address
and starves the PP pool.

**Verdict: NOT DELIVERED as ordered, and the order as written is not
executable.** The honest fix is not "remove the number" but "stop there being
a number": size the TP-phase pool to the PP id space, so the cap is a
*derived* quantity rather than an operator guess, and it rises automatically
whenever T1 raises the id space. That change is specified in §5 and is not
written. What IS delivered is that the cap no longer *masks* anything: after
boot C every rank reports zero stranded, so the cap's damage is now bounded
and visible.

---

## 3. T3 — YaRN toward 1M, and the bug in the way

**The zero-upfront-cost falsifier ran first, as ordered, and it found a
blocker that is not a memory cost at all.**

Structures scaling with the context ceiling, 393216 → 1048576:

| site | scales with | Δ at mrr=4 |
|---|---|---|
| `memory_pool.py:373` `req_to_token` | ceiling × (max_running_requests+1), int32 | **12.5 MiB** |
| `utils/common.py:5049` RoPE `cos_sin_cache` eager reserve | ceiling × rotary_dim, fp32, per instance | **~320 MiB @ rotary_dim 128** |
| `flashinfer_backend.py:1784` decode-graph kv_indices | ceiling × max_num_tokens | ~10 MiB |
| `flashinfer_backend.py:7741` MTP draft kv_indices | ceiling × topk × steps | ~5 MiB |
| `flashinfer_backend.py:1978` full-CG prefill kv_indices | ceiling × num_slots | up to ~4.2 GB — **conditional, not in the ship graph mode** |

The named prime suspect (`req_to_token`) is **not** the cost. The dominant
one is `reserve_rope_cache_for_long_sequences`, called unconditionally from
`model_runner.py:2496` before graph capture — it pre-expands the cos/sin cache
to the *full ceiling* at boot. It defeats laziness deliberately: growing the
cache after capture would reallocate a tensor whose address the graphs have
baked in (the same silent-corruption class as `phase_flip_boot.py` §4c).

**The blocker found on the way — a silent correctness bug, now FIXED.**
`YaRNScalingRotaryEmbedding` builds its cache with
`_compute_inv_freq(self.scaling_factor)` and `* self.mscale`, but does not
override `_ensure_cos_sin_cache_length`. The inherited base growth path used
`self._compute_inv_freq(self.base)` — **passing the RoPE theta where a
scaling factor is expected — and applied no mscale at all.** Every appended
row carried different frequencies and the wrong amplitude from the rows before
it. No exception; just wrong attention past the boot cache length.

Three classes share the defect: `YaRNScalingRotaryEmbedding` (yarn.py),
`DeepseekScalingRotaryEmbedding` (rope_variant.py), `YaRNScalingMRotaryEmbedding`
(mrope.py).

It fires **today**, latently: the boot cache is
`max_position_embeddings × scaling_factor` = 393216 rows, while
`reserve_rope_cache_for_long_sequences` asks for
`ceiling + steps×draft×2 + 256` aligned to 128 = 393600 — so ~384 rows are
already appended through the broken path at every boot. They are unreachable
only because `context_len` caps positions at 393215. **Raising the ceiling is
exactly the change that makes them reachable**, which is why this had to be
fixed before any 1M work, not after.

Fix: an overridable hook pair on the base class
(`_cos_sin_cache_inv_freq`, `_cos_sin_cache_row_scale`), overridden in all
three scaled variants so the growth path extends with the parameters the
cache was built with.

Test `test/srt/test_yarn_rope_cache_growth.py`, three cases (formula match,
seam continuity, mscale applied). **Can-fail proof, no `git stash`:**

| arm | tree | result |
|---|---|---|
| fixed | `/spinning/wt-kv-universe/python` | **3 passed** |
| unfixed | `/spinning/wt-merge-r6-base/python` (hook absent, verified) | **3 failed** |

Import path was printed in both arms — this is not the worktree-PYTHONPATH
false-red.

**NOT DELIVERED for T3:** no 1M boot, no corridor sample at 1M, no
admitted-at-1M session, no decode past 393216. The measured Δ above is from
code arithmetic, not from a boot. The lazy/VA-stable conversion of the RoPE
reserve (the actual "zero cost until used" mechanism, §5) is not written.

---

## 4. T4 — PP-after-spill composition

**NOT STARTED.** No falsifier constructed, no verdict. The window went to T1's
three boots and the ship restore. Nothing is claimed about it. The specific
hazard remains as briefed: the spilled session's `req_to_token` sentinel rows
and host-tier regions crossing a KV reshard, with silent corruption as the
failure class to hunt.

---

## 5. THE WORK THIS SHIFT SPECIFIES BUT DID NOT WRITE

1. **Size the TP pool to the PP id space.** Removes the `--max-total-tokens`
   guess entirely and makes T2's order executable. Boot A is the proof it is
   needed; without it, no cap change is safe.
2. **VA-stable RoPE cache reserve.** Reserve virtual address space for the
   full ceiling and commit pages on use, so the address graphs bake stays
   fixed while the cost stays proportional to what is used. The machinery
   already exists in-tree: `VmmWeightsArenaCarrier` / the RUNG 3 reservation
   in `phase_flip_boot.py`. This is what makes "1M grantable at zero upfront
   cost" true rather than asserted.
3. **Re-solve the vector after rebudgeting.** Boot C left 855/2436/2237 MiB
   above the corridor floor. With the 3080s finally holding their own cards
   the optimal stage vector moves (desk estimate `15,9,8`), but that must be
   re-measured against boot-C-class budgets, not against the ship numbers.
4. **Corridor under sustained load.** Boot C was sampled at idle and after
   three generations, NOT as a continuous 100 ms minimum under load. The
   corridor law is a continuous-minimum law; boot C is not yet a corridor
   acceptance.
