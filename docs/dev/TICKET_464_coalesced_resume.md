# TICKET #464 -- coalesced VMM resume: determination, and the window gate

Date: 2026-08-17. Hermetic (`CUDA_VISIBLE_DEVICES=""`), no boots, no window
taken.

**Status: the coalescer is BUILT (prior slice `17e7c8e36a`). This slice makes
it REACHABLE and makes the measurement turnkey. The timing itself is
GPU-PENDING.**

## What was already there, and must not be rebuilt

`17e7c8e36a` "[#464] Coalesce the VMM resume: one handle per contiguous run,
DEFAULT OFF" landed on 2026-08-16 with `coalesce_commit_plan`, a
`CoalesceReport`, refusal-on-hole with `VmmCoalesceRefused`, and 10 hermetic
tests including the 1001->3 reduction under a mocked driver. All 10 still pass.

It is **not in the shipping lineage**: it exists only on
`feat/704-prefill-ladder`, `fix/701-ledger-wiring` and
`reconcile/cluster-b-seam-model`. This slice is branched directly from it so
the work composes rather than duplicating. That is a merge-train item.

## Three findings this determination adds

### 1. The flag was unreachable, so the measurement it exists for could not be taken

`KvVmmArena.__init__` took `coalesce_resume: bool = False`, and **nothing
passed it**: both carrier arenas (`phase_flip_spill.py:597`, `:927`) and the KV
seam owner (`memory_pool.py:2488`) construct without the argument. So the flag
was False in every real boot. The slice's own rationale -- "the flag exists so
the measurement can be taken, not because the win is assumed" -- could not be
acted on, because there was no way to turn it on outside a unit test.

That is the counter-vs-actuator shape (#679/#681/#684/#715): a switch whose
actuator does not exist. A dark feature has a lever; this had none.

**Fixed here.** `resolve_coalesce_resume` reads `SGLANG_VMM_COALESCE_RESUME` at
the single place the flag is stored, so no construction site needs editing and
none can drift out of sync. An explicit argument still wins over the
environment -- ambient state must not overrule a caller that decided. Default
off, byte-identical; 9 tests, including two that pin the WIRING (that
`__init__` resolves through the helper and that its default is `None`, since a
hard `False` would swallow the environment for every caller that omits it).

### 2. The 40-85 ms band belongs to a DIFFERENT mechanism

The ticket's acceptance is "measured against the 40-85 ms band". That band is
real and sourced -- `adaptive_graph_memory.py:207-214`, "organic avg 40-51 ms,
max 85 ms (vs 14 ms Stage-1 -- the price of remapping+zeroing ~1 GB per swap)",
carried into `DESIGN_584:141` as "graph restore ... 40-85 ms ... #464
improvement pending".

But that path is **not this path**. `adaptive_graph_memory.py` swaps through
`torch_memory_saver` pause/resume (`self._adapter.pause/resume(tag)`, `:903`,
`:1108-1113`) -- a third-party package in site-packages -- and references none
of `KvVmmArena`, `commit_range`, `cuMemMap`, `cuMemCreate` or `vmm_utils`.
Meanwhile `KvVmmArena` backs the draft-weight carrier (rung 2), the
weights-arena tail (rung 3) and the KV pool. **No graph state is routed through
it**; grepped `KvVmmArena` across `python/sglang/srt/` and every hit outside
`kv_vmm_backing.py`/`phase_flip_spill.py` is a comment or the ledger.

So "beat 40-85 ms" is not a like-for-like criterion for the coalescer as built.
Judging this change by that number would be the #709 mistake -- an acceptance
rule that cannot see the thing it claims to judge.

**Consequence for the gate:** the acceptance below compares ON against OFF on
the SAME path in the same process, and cites the band only as the analogous
prior measurement it is. The runner refuses to print it as a threshold, and a
test pins that refusal.

This does not devalue the slice: coalescing `commit_range` is a real
improvement to the phase-flip spill and KV-seam restore paths. It relocates the
claim.

### 3. "~500 x 2 MiB calls" is chunk-dependent, and no default produces it

Extents are `ceil(nbytes / chunk)`, and the chunks actually in use are:

| path | chunk | extents / GiB | driver calls | after coalescing | saved |
|---|---|---|---|---|---|
| KV seam (`SGLANG_FLIP_SEAM_CHUNK_MIB`, default **8 MiB**, `memory_pool.py:2477`) | 8 MiB | 128 | 257 | 3 | **254** |
| carriers (`CARRIER_COMMIT_CHUNK`, `phase_flip_spill.py:219`) | **64 MiB** | 16 | 33 | 3 | **30** |
| the ticket's illustration | 2 MiB | 512 | 1025 | 3 | 1022 |

The 2 MiB figure is the allocation-granularity fallback, not a chunk any
default sets. The KV seam is where the win is, by 8x over the carriers. The
runner computes these from the real chunk rather than asserting the ticket's
number, and a test pins the two defaults so the illustration is not silently
adopted as this path's shape.

## The gate

`bench/464/run_464_resume.py`. The call-count half is **arithmetic** and is
already discharged at the desk (19 hermetic checks); only the wall time needs a
card. That split is deliberate -- the window buys timing, nothing else.

```bash
# hermetic, any time
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/python \
  $V/bin/python bench/464/run_464_resume.py --self-test

# in a claimed window (/spinning/gpu-arb/; the script claims nothing itself)
PYTHONPATH=$PWD/python $V/bin/python bench/464/run_464_resume.py \
  --run --device <ordinal> --mib 1024 --chunk-mib 8 --out <artifact dir>
```

Exit 0 = measured, 1 = a check failed, 2 = could not run.

### Acceptance

Green when, at the KV-seam chunk (8 MiB) on a ~1 GiB payload:

1. driver calls fall **257 -> 3** (analytic, already pinned; the run must
   confirm the arena took the coalesced plan, via the `#464 coalesced ...`
   log line);
2. the ON arm's median wall time is **below** the OFF arm's, both measured in
   the same process on the same device, median of >= 5 repeats (median, not
   mean: one driver hiccup must not set the number);
3. the ON arm maps **exactly the same bytes** as OFF -- a resume that maps a
   different range is a bug, not an optimisation (already pinned hermetically);
4. a run with an interior hole REFUSES to coalesce and says where, rather than
   merging across it.

Explicitly NOT an acceptance criterion: any comparison against 40-85 ms. See
finding 2.

### What would falsify the win

- ON and OFF within noise at 8 MiB -> the ~254 saved calls are not the cost
  driver, and the remaining time is the memset plus the driver's own page
  work, which coalescing does not touch. That is a real and reportable result:
  the feature would then be correct and worthless, and should stay default-off.
- ON slower than OFF -> one large `cuMemCreate` is contending where many small
  ones did not; the handle size may be crossing a driver granularity boundary.
- A hole appearing in the common case -> the coalescer refuses and the
  measurement is not takeable on that arm at all; report the hole rate first.

```
call count 257 -> 3 confirmed : PENDING
off arm median ms             : PENDING
on  arm median ms             : PENDING
verdict                       : PENDING
```

## Not done here

- **No boots.** `/spinning/gpu-arb/holder` reads `706-retry` with a live
  heartbeat; the cards belong to another strand.
- **No `torch_memory_saver` work.** The 40-85 ms path is third-party code
  (own-bugs-before-foreign-bugs); if the graph-state resume is the real target,
  that is a separate determination and it starts by deciding whether we touch
  an external package at all.
- **`expandable_segments` was not reached for.** It is forbidden -- it costs
  #93/#102/#89.
