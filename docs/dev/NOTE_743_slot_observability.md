# NOTE 743 — the slot instrument, and the two things it deliberately does not do

Companion to `NOTE_743_mamba_slot_hitrate.md`. That note is
determination-only and proposes three remedies. This one records what was
BUILT for remedies 1-2, and settles remedy 3 as **refuted**.

Branch: `fix/743-slot-evict-instrument`.

## 1. What was built (remedies 1 and 2)

`python/sglang/srt/mem_cache/mamba_slot_observer.py`, wired into both
lineages:

| event | site |
|---|---|
| `MAMBA-SLOT EVICT` | `mamba_radix_cache.evict_mamba`, `hi_mamba_radix_cache.evict_mamba` |
| `MAMBA-SLOT TRUNCATED` | `_match_post_processor` in both files, inside the existing `len(value) > best_value_len` branch |

One observer, both lineages, per #747's rule — two spellings of one emitter
is how the lineages drift and how a reader compares numbers that do not mean
the same thing. `lineage=` is the only difference in the wording, and on the
host tier it is load-bearing: the same truncation there may resolve to a
load-back rather than a full re-prefill.

Cadence is a token bucket: sustained `SGLANG_MAMBA_SLOT_LOG_RATE` (default
2/s) with a decoupled capacity of 8, and a `SUPPRESSED` rollup that carries
the totals of everything it hid. Slot pressure does not arrive at a steady
rate — it arrives as several evictions inside one scheduler step — so a
bucket sized to its own drain rate would print the first and suppress exactly
the burst that matters. Rate 0 restores the pre-#743 silence exactly.

## 2. Filed, not built: the mamba lineages never feed the eviction metric

`BasePrefixCache.update_eviction_metrics` (`base_prefix_cache.py:340-345`) is
called by every sibling cache:

- `radix_cache.py:595`
- `hiradix_cache.py:1356`
- `swa_radix_cache.py:686`
- `radix_cache_cpp.py:160`
- `unified_radix_cache.py:843`

and by **neither** `mamba_radix_cache.py` nor `hi_mamba_radix_cache.py`, even
though both call `init_metrics_collector()` (`mamba_radix_cache.py:544`). So
the two mamba caches are the only ones whose eviction is invisible to
Prometheus, and a soak reading `eviction_num_tokens` sees zero from them
regardless of how much they evict.

**`evict_full` is a clean gap** — it evicts TOKENS and returns a token count,
which is exactly what the metric wants. Three lines on each lineage.

**`evict_mamba` must NOT be wired to it, and that is the reason this is filed
rather than fixed in passing.** `increment_eviction_num_tokens`
(`observability/metrics_collector.py:2114-2115`) counts tokens.
`evict_mamba` returns SLOTS. Feeding slots into a token counter would not add
observability, it would corrupt an existing metric with a second unit — the
`eviction_num_tokens` series would silently stop meaning tokens. A mamba slot
eviction needs its own counter, which is a metrics-schema decision, not a
drive-by.

The log instrument above covers the slot half in the meantime, and carries
cumulative totals in every line precisely so the absence of a metric is not
also an absence of numbers.

## 3. REFUTED: remedy 3, the "unconditional pinned-checkpoint term"

`NOTE_743_mamba_slot_hitrate.md` §4.3 proposes:

> Drop the unconditional pinned-checkpoint term when
> `mamba_checkpoint_interval is None` (`mamba_pool_floor.py`), red-first.
> Recovers `max_running_requests` slots of headroom.

**This premise is wrong and the change must not be made.** It would remove
budget for a slot that is still held, under-flooring the pool and
resurrecting the #581 late assert — the exact failure
`mamba_pool_floor.py:124-131` already warns about for the sibling constant
under #755.

The pin is gated on the RADIX CACHE being on, not on the checkpoint interval.
Decisive chain:

1. `mamba_ckpt_utils.py:34-38` — `is_on_interval` returns `True`
   unconditionally when `interval is None` ("a valid checkpoint position
   (always, when off)"). The grid check therefore never rejects with the flag
   unset.
2. `mamba_radix_cache.py:811` (also `:641`, `:669`) — that check is the only
   gate on attaching a mamba value during a cache insert, so with the flag
   unset the ordinary path proceeds and inserts a real `mamba_value`
   (`:743`, `:949`), which `_insert_helper` attaches unconditionally
   (`:1836-1910`, asserting non-None at `:1850`).
3. `mamba_radix_cache.py:996` — `inc_lock_ref(new_last_node)` runs on that
   same path, with no interval gate.
4. `mamba_radix_cache.py:1313-1318` — the pin increments `mamba_lock_ref` and
   moves the slot from evictable to protected whenever
   `node.mamba_value is not None`, which is true on the ordinary path.
5. `schedule_policy.py:1168-1171`, called from `:1574`, `:1582`, `:1627` —
   the admission-side pin of `req.last_node` runs on every prefill add and
   has no `mamba_checkpoint_interval` gate at all.

So with the interval unset a running request still holds the pinned slot, and
the floor is right to charge for it.

What `mamba_checkpoint_interval` actually gates is WHERE resume points may
fall on the token axis (grid alignment, for run-to-run determinism), never
WHETHER states are cached or pinned — consistent with its own help text at
`server_args.py:4126-4139`.

The existing tests already encode the correct axis, and all three would have
had to be broken to land the proposed change:

- `test_mamba_pool_floor.py:195-205` asserts the per-request sum INCLUDING
  `MAMBA_FLOOR_PINNED_CHECKPOINT_SLOTS`, with the interval unset;
- `:230-237` pins the production `1+2+1+1 = 5` and `mamba_hard_floor(sa, 4) == 20`,
  with the interval unset;
- `:225-228` is the only test that drops the pinned term, and it does so via
  `disable_radix_cache=True` — naming the real gating axis.

**Status: closed as refuted. No code change.** If pool headroom is wanted at
concurrency 4, the levers that survive this analysis are the #755 reorder
(`mamba_pool_floor.py:124-131`, already implemented and env-gated) or raising
`--max-mamba-cache-size`, not deleting a term for a slot that is genuinely
held.
