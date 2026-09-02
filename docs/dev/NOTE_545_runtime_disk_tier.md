# #545 — runtime attach/resize/detach of the HiCache disk tier

Desk only, 2026-08-17. No boot, no server, no GPU. **Nothing built**, and the
reason is that it is already built. §4 is the one gap that is real.

## 0 — The control path exists, end to end

The brief asked to build it. It is there:

| operation | route | file:line |
|---|---|---|
| attach | `PUT /hicache/storage-backend` | `entrypoints/http_server.py:1415` |
| detach | `DELETE /hicache/storage-backend` | `:1449` |
| status | `GET /hicache/storage-backend` | `:1476` |
| resize | `POST /hicache/storage-backend/resize` | `:1495` |
| clear | `POST /hicache/storage-backend/clear` | `:1395` |

Each carries `@auth_level(AuthLevel.ADMIN_OPTIONAL)` **plus** an explicit
`if not server_args.admin_api_key: return _admin_api_key_missing_response()`
inside the handler — the #510 regime, applied with the belt-and-braces the
most sensitive routes are supposed to get. They reach
`attach_storage_backend` / `detach_storage_backend` / `resize_storage_backend`
on the tree cache through scheduler handlers
(`managers/scheduler.py:7625/7681/7728`).

The semantics the brief specified are the semantics implemented:

- **attach/detach refuse a non-idle scheduler by name** — *"Reject attach:
  scheduler is not idle. #queue-req=… #running-req=…"* (`scheduler.py:7635`).
- **resize-down evicts before acknowledging.** `LRUFileEvictor.set_limits`
  (`storage/file/lru_file_evictor.py:257-337`): *"Shrinking evicts LRU victims
  inline … and returns once usage is back under the new cap. In-flight
  (reserved but uncommitted) writes are never evicted"* — so it does not
  truncate live pages, and the write interlock is `_pending_writes` rather
  than a race with the backup queue.
- **resize deliberately does NOT require idleness**, because it only touches
  the evictor's own lock and counters. That is a narrower and better interlock
  than attach/detach's whole-scheduler gate, and it is intentional.

## 1 — Retraction: I claimed resize was untested. It is not.

I wrote a hermetic pin file on the finding that "resize has no coverage at
all", having grepped for `resize` inside the E2E attach/detach test and found
nothing.

**Wrong.** `test/registered/unit/mem_cache/test_hicache_runtime_resize_545.py`
exists with **21 tests**, covering every property I pinned and several I did
not: grow-evicts-nothing, shrink-until-under-cap, LRU victim order,
enable-at-runtime-adopts-existing-files, in-flight-write-not-evicted,
lifting-the-cap-disables-eviction, non-owner-MLA-rank-inert, the request
validation layer, and both `HiRadixCache` and `UnifiedRadixCache`.

My file was 100% duplicate and is **deleted**. Shipping it would have created
a second authority for the same properties — the thing I refused in #536 when
I reverted a client-side default that duplicated `launch.py`.

The error is the same one I made in #726 and #677: concluding absence from the
file I happened to open, instead of grepping for the thing itself. Third
instance; the rule is that "no coverage" requires a search for the coverage,
not a search inside one file.

My harness also had a defect worth recording: it passed `max_size_bytes` /
`min_free_bytes` as `extra_config` keys, but the real names are `max_size` /
`min_free_space` (`lru_file_evictor.py:157-162`). Every evictor it constructed
came up UNCONFIGURED, and most pins still passed — because they configure via
`set_limits()` anyway. A harness passing for the wrong reason.

## 2 — Two more premise items that do not exist

**`MixedLayoutError`** — the guard the brief said attach must reuse. **Not in
this tree.** Zero hits across `python/`. The commit the brief cites
(`19f4c68864`, #558) describes it in its message, but no such class is in the
worktree; the nearest relative is `DraftKvLayoutMismatch`
(`disaggregation/draft_kv_canonical.py:65`), which guards PD draft-KV, not
HiCache attach. The guard attach actually uses is a same-backend check in
`hiradix_cache.py:544-568`, refusing a *different* backend by name and telling
the caller to detach first.

**`ReadBufferPool` / #720** — **not in this tree either.** No class of that
name, no `#720` reference. Whatever accounting the brief meant, it is not
present under that name and I did not design around a guess.

## 3 — Prior art the brief did not mention

`docs/dev/NOTE_544_hicache_runtime_preserve_thinking.md` is a prior
desk-complete investigation of this exact ticket, with the layer diagram and
the design that was then implemented, and
`docs/advanced_features/hicache_storage_runtime_attach_detach.md` is the
user-facing documentation of the shipped feature. Both predate this task.

## 4 — The one gap that IS real, and it lands on this rig's models

`UnifiedRadixCache.attach_storage_backend` and `detach_storage_backend`
(`mem_cache/unified_radix_cache.py:2769`, `:2783`) are **hard stubs that always
fail**:

```
"UnifiedRadixCache does not support runtime HiCache storage attach yet.
 Configure hicache_storage_backend at startup instead."
```

`resize_storage_backend` on that class works; attach and detach do not.

**Why that matters here rather than being a footnote:** `UnifiedRadixCache` is
what `mem_cache/registry.py:191` constructs, on the path that appends the
MAMBA component for `is_hybrid_ssm` — the hybrid-GDN family this rig serves. So
on our own models the shipped runtime story is **resize yes, attach/detach
no**, and the ticket's headline capability is exactly the half that is stubbed.

I did not verify which cache class our specific boot instantiates (that needs
the boot config, not the source) — so this is "the path hybrid-SSM models are
routed to", not "confirmed for our running server".

Also NOT ESTABLISHED, from the map: no test raises or mocks a real
`OSError(ENOSPC)`; disk-full is exercised only through the `min_free_space`
watermark refusal.

## 5 — Recommendation

Do not build a control path. Two candidates, in order:

1. **Implement `UnifiedRadixCache.attach/detach`** so the hybrid-GDN family
   gets the capability the rest already has. That is the ticket's actual
   remaining work, and it is a real implementation task rather than a
   plumbing one — the stubs are stubs because the unified cache's controller
   lifecycle differs from `HiRadixCache`'s, which is what would have to be
   made restartable.
2. **A real ENOSPC injection test** for the write path, since the current
   coverage stops at the watermark.

## 6 — Acceptance for the live window (filed, not run)

For the Flip+HiCache boot list, on a server started **without** a storage
backend:

1. attach mid-load via `PUT /hicache/storage-backend` with `admin_api_key`
   set; expect success only when the scheduler is idle, and the named
   non-idle refusal otherwise.
2. hit rate rises afterwards — measured from the same `#cached-token` source
   the #703 acceptance uses, never `cache_hit_rate`.
3. no stall attributable to the attach: per-request latency across the attach
   boundary stays inside its band.
4. `POST /resize` down while serving: usage falls under the new cap and the
   call returns only after it has, with no request failure.
5. `DELETE` detach: subsequent lookups become misses, never corruption —
   content-addressed, the same argument #703 uses for drops.

~~On a hybrid-GDN model, expect (1) and (5) to fail with the §4 refusal.~~
**SUPERSEDED — the gap is now implemented (§7).** On a hybrid-GDN model,
attach and detach are expected to SUCCEED, with the state (MAMBA) component
covered rather than refused, because the controller's host pools span every
component it owns. There is no partial-capability outcome to accept.

---

## 7 — The §4 gap, implemented

`UnifiedRadixCache.attach_storage_backend` / `detach_storage_backend` are no
longer stubs. They mirror the shipped `HiRadixCache` pair — same validation,
same named refusals, same "same backend is success, different backend is a
refusal rather than a silent swap" rule — with one addition this cache needed
at the time: `_symmetrize_prefetch_capacity()` after the config is applied.

**Superseded by #1068 slice 2.** That method and its `HiRadixCache` twin are
deleted. The prefetch budget is now the `prefetch_capacity_limit` property of
the host pool the controller is bound to, rank-uniform under `--hicache-size`
(MIN-synced rows), and ratio sizing under uneven DCP with `tp_world_size > 1`
is refused by name at init (G8) instead of repaired with an all_reduce. The
paragraph below records why the all_reduce was safe while it existed.

**Why that addition was safe, which was the open question.** That method entered
an **all_reduce** across DCP/TP ranks, and its own guard says a rank-local
early return "would leave the other ranks in the all_reduce with no partner".
A single-rank attach would hang. It cannot happen: attach fans out through
`FanOutCommunicator` (`managers/tokenizer_control_mixin.py:125`, `:360`) to
every rank and merges the results, so the group runs it together — and the
scheduler already refuses a non-idle scheduler by name before reaching it.

**Detach order is the contract**, taken from HiRadixCache: drain the control
queues *before* tearing the controller down, or acks and releases can no
longer be matched to their nodes and host pages and locks leak; drain again
afterwards to sweep what the shutdown produced. The drain is **local**
(`None` limits = everything on this rank) because the steady-state path
derives its counts from an all_reduce, and a detach may not depend on a
collective its peers may already have left.

**No silent partial capability** (#268): the state component is covered, not
refused. `_get_hybrid_storage_attach_kwargs` passes
`cache_controller.mem_pool_host.entries`, which spans every component the
controller owns — the same set the boot-time attach passes — so a hybrid
attach is whole rather than KV-only wearing a success message.

Pins: `test/registered/unit/mem_cache/test_unified_attach_detach_545.py`, 16
tests, sibling to the resize file rather than merged into it (that file owns
the resize authority; one authority per behaviour). Red-first: restoring the
stubs fails 15 of 16.

~~Still open: the ENOSPC injection test from §5(2), and
`tokenizer_control_mixin.py:370`'s `# TODO: partial rollback if failed`.~~
**Both closed — see §8 and §9.**

## 8 — Partial rollback (closes the pre-existing TODO)

A mixed fan-out result left the group **half-attached**: some ranks running
storage threads with a backend bound, others not — exactly the state the
detach contract exists to prevent, reported as a clean failure.

Now the terminal state is all-attached or all-detached, never mixed. On a
mixed result the coordinator detaches the group and, if that rollback fails
anywhere, **names the stranded ranks** instead of reporting a clean failure.

Three things this needed:

- **A rank on the reply.** `FanOutCommunicator.handle_recv` appends results in
  **arrival order**, so list position does not identify a rank — "ranks 0 and
  2 are stranded" was literally unsayable. `AttachHiCacheStorageReqOutput` and
  `DetachHiCacheStorageReqOutput` now carry `rank` (flat world rank,
  `pp_rank * tp_size + tp_rank`), stamped in a **wrapper** so every return
  path gets it; stamping at each `return` would let a later-added path ship
  unstamped.
- **A group detach, not a targeted one.** Detach is idempotent by
  construction — it asks the controller to clean up even when
  `enable_storage` is already False, precisely to sweep partial-attach
  leftovers — and the communicator offers no rank-addressed send.
- **Collective-free**, per §7's own rule: the detach path drains with local
  (`None`) limits and enters no all_reduce, so a rank whose peers have already
  left a collective cannot hang on it.

The verdict is never flipped to success; rollback only changes what the
message can tell you. Pins:
`test/registered/unit/managers/test_attach_partial_rollback_545.py` (9).
Mutation: dropping the rollback fails 4.

## 9 — ENOSPC injection (closes §5(2))

`test/registered/unit/mem_cache/test_hicache_enospc_545.py`, 11 pins against a
real `HiCacheFile` over a real tmpdir with the failing syscall patched. The
*technique* of the canonical store's three-site injection is reused; none of
its code is, since that lives on an unmerged train branch.

Pinned: a write hitting ENOSPC returns False **and the key is genuinely not
readable** (False is only honest if the page really is absent); no torn page is
ever visible, because the page appears only at `os.replace` — a failure before
it leaves the final path absent and a failure at it leaves prior content
intact; the watermark **refuses at the door** when a cap is configured rather
than failing mid-write with the IO already spent; and the tier still works
afterwards.

**One pin had to be rebuilt to be honest.** The reservation-leak check first
asserted that a later write still fits after five failed ones. That could not
fail: with a cap in force the evictor simply **evicts** to admit, so a leaked
reservation causes extra eviction rather than a refusal — the pin passed with
`abort` removed. It now asserts on `_total_bytes` directly, which is what
distinguishes released from leaked. Mutation (removing `abort` and the tmp
cleanup) now fails 2; before the rebuild it failed 1.
