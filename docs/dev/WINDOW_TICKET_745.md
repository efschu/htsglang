# WINDOW TICKET 745 — GDN state survival beyond the device slots

**One boot, one question: does tiering GDN state off the device lift the
prefix-cache hit rate?** Everything below is desk-derived; nothing is
boot-verified. Owner runs this in an F4-r4 window.

## 0. Why this boot, in one line

`NOTE_740` measured the adapter and template CLEAN (turn N is a 99.7 % token
prefix of turn N+1) and `NOTE_745` found the writer and the
resume-from-nearest path already built. The remaining cause is EVICTION: 12
device slots shared between the running requests and every retained checkpoint,
under LRU. This boot gives those checkpoints somewhere to live.

## 1. Preconditions

* GPU window claimed via `/spinning/gpu-arb/` (holder file + heartbeat; stop the
  heartbeat BEFORE releasing). Whoever stops serving owns bringing it back.
* Baseline captured FIRST (section 3) on the current config, same prompt shape.
* HOST-LEDGER post booked: **3591 MiB (3.51 GiB)** total, 1272 / 1197 / 1122 MiB
  for stages 0 / 1 / 2 (derivation in `NOTE_745` section 6c). If the arm changes
  `hicache_ratio`, re-derive: `GDN_layers_of_stage x 3.1172 MiB x host_slots`.

## 2. The arm

Take the current 30030 launch line and add exactly:

```
--enable-hierarchical-cache
--hicache-storage-backend file
--hicache-storage-file-path <a path on the disk tier>
```

Change NOTHING else on the first arm — not the slot count, not the cap. The
question is tiering alone.

Three things to know before reading the log:

1. **The cache class changes.** With `--enable-hierarchical-cache` on a
   hybrid-SSM model the registry returns **UnifiedRadixCache**, not
   `MambaRadixCache` and NOT `HiMambaRadixCache` (which has no construction site
   at all — `registry.py:107-112`, and see `NOTE_745` section 6a). Expect the
   unified tree's behaviour, including its own match/evict paths.
2. **`--mamba-checkpoint-interval` must stay OFF.** It is refused against both
   hierarchical cache and the unified tree (`server_args.py:14066-14075`). This
   costs nothing here: `is_on_interval(pos, None) -> True`
   (`mamba_ckpt_utils.py:34-38`), so with the interval unset EVERY position is
   an eligible checkpoint position.
3. **This is a real behaviour change**, not a tuning knob. If the arm is worse,
   that is a result, not a failure.

## 3. Acceptance — one log, three lines

**(a) Hit-rate lift (the question).** Re-run the `NOTE_740` shape: one agent
session, a tool loop of >= 100 requests, same model alias. Record
`cached_tokens` / prompt tokens per request from `enable_cache_report`
(already `True` live).

* baseline to beat: **4 hits in 121 requests**;
* PASS = a materially higher hit fraction on the same shape. State the number;
  do not round a small lift into a verdict.
* The prefix is known-good (99.7 %), so a still-low hit rate means the states
  are not surviving even to host, and the answer is "tiering does not fix it" —
  which closes the question just as usefully.

**(b) HOST-LEDGER line.** The measured mamba host pool at steady state, against
the 3591 MiB booked above. A variance beyond ~10 % means the ratio assumption
in section 6c is wrong and the ledger needs re-deriving before this config is
carried anywhere.

**(c) Resume evidence.** At least one request whose match came from a
host-backed node — the `load_back` path at
`unified_cache_components/mamba_component.py:139-144`. Without this line, (a)
could be explained by device-side retention alone and the boot proves nothing
about tiering.

## 4. Not in this ticket

* **Disk-vs-host split.** The file backend already carries `PoolName.MAMBA`
  (`NOTE_745` section 6b), so no build is needed to reach disk; whether states
  land on disk or stop at host is an observation from this boot, not a
  precondition of it.
* **Step (5), the slot floor.** Dropping 12 -> 6 returns ~898 MiB
  (318/299/281 per stage) which **raises the binding stage's token ceiling**.
  That is a SECOND arm and must not be folded in: it changes the KV pool size,
  which would confound the hit-rate reading of arm 1.
* **The `:14066` refusal.** Lifting it buys deterministic resume positions. Only
  worth doing if arm 1 lifts the hit rate.
* **#743** (slot-count probe) stays the cheap comparison arm and is now directly
  interpretable: it varies exactly the quantity named as binding.

## 5. Rollback

Remove the three flags. No state is written that the previous config reads, and
the file path can be deleted. `--enable-hierarchical-cache` off restores
`MambaRadixCache` via `registry.py:133`.
