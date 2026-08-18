# NOTE 738 root cause: the flag was inert BEFORE its own fix landed — and the fix holds on metal

**Verdict, both hypotheses discriminated with fresh numbers (2026-08-18,
this pool, `tools/probe_pageout_zfs.py`, 4 GiB urandom file, mincore):**

* **(a) is TRUE for the form #408 shipped and FALSE for the form the tree
  ships now.** `MADV_PAGEOUT` through a fresh, after-the-fact mapping
  returns rc=0 and frees NOTHING (100.0% resident before and after — the
  accepted-and-did-nothing silent success). The same advice through the
  LIVE `safe_open` mapping's own addresses — what
  `_drop_file_cache_after_load` does since `aa2abc525b` — drops
  **100.0% → 0.0%**.
* **(b) is TRUE historically and RESOLVED on this lineage.** The 99G
  specimen (2026-08-17, flag set, release only at exit) predates
  `aa2abc525b` ("[#738] Don't cache the weights you cannot evict",
  2026-08-17 20:58) — the commit whose own docstrings CITE that day's dead
  measurement. `aa2abc525b` is an ancestor of the composite `c546eed923`:
  the fix is on the deployed lineage.

Supporting probe numbers, same run:

| phase | result |
| --- | --- |
| A touched mmap baseline | 100.0% resident |
| B fresh-mapping `MADV_PAGEOUT` (live mapping standing) | rc=0, **100.0%** resident |
| C ladder `drop_page_cache_range(address=live)` | **0.0%** resident |
| D `read_file_direct` 4096 MiB (O_DIRECT) | 0.0% → **0.0%** (zero growth) |
| E buffered `read()` of 1 GiB | **0.0% resident** — see below |

**Phase E is a finding of its own:** on this OpenZFS pool a buffered
`read()` never touches the page cache at all — the data lands in the ARC,
which mincore cannot see and which ZFS reclaims under its own pressure
rules. Two consequences: the classic fadvise-no-op claim is untestable via
mincore (nothing was in the page cache to drop), and the 99G load-time
peak was the MMAP path's page-cache population, exactly the population the
live-extents drop now removes.

## Loader branch coverage on `c546eed923` (the sweep)

| branch | coverage |
| --- | --- |
| `safetensors_weights_iterator` (the served default) | covered: live extents per tensor, drop per file (`weight_utils.py:1112-1126`) |
| `multi_thread_` / `buffered_multi_thread_safetensors` | covered, same mechanism (`:1222-1236`) |
| `--weight-loader-direct-io` | covered at the cause: `read_file_direct`, zero cache growth (phase D) |
| FASTSAFETENSORS | honest refusal with the flag (#742, absorbed per the pass-3 ledger) — GDS reads bypass the page cache and the drop would be a no-op-as-feature |
| GGUF | its own streaming ladder (`gguf_shards.ConsumedPageDropper`), the origin of the live-mapping principle |
| `pt_weights_iterator` / `multi_thread_pt` (.bin/.pt) | NOT wired — and phase E shows why wiring it buys nothing here: `torch.load`'s buffered reads land in ARC, not page cache; the flag's mechanism has nothing to act on. Documented rather than wired: a red-first test for an ARC-resident path would assert something mincore cannot observe. |
| `np_cache_weights_iterator`, `runai_` | legacy/remote paths, out of serving scope; unwired, same ARC reasoning applies to the former |

## What this task adds

* `tools/probe_pageout_zfs.py` — the metal-truth probe as a reusable
  2-minute command (no GPU, no serving; writes/removes a 4 GiB scratch
  file, `PROBE738_PATH` overridable). The next "is the drop flag inert"
  question is a command, not a desk day.
* This note. The wiring and mechanism tests already exist and are green on
  this branch (`test_direct_io_read_738.py`,
  `test_safetensors_cache_drop.py`: 12 passed).

## Boot-gated residue

The 99G peak has not been re-measured on a REAL composite boot with the
flag set since `aa2abc525b` landed. Expected on the covered path: page
cache returns to baseline per shard as the load progresses, peak bounded
by ~one shard plus ARC (which is reclaimable and not an OOM-killer
input). One line for the next window's checklist: `free -g` peak during
weights load with `--weight-loader-drop-cache-after-load` on the composite
— target: no 99G plateau.
