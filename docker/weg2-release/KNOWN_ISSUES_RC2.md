# Known issues: htsglang weg2 RC2 (27B line)

Scope: RC2 tree `desk/27b-release-rc2-0924` @ `4aded781ab` plus the review fixes on
`desk/27b-up-rc2-review-fix-0924` (`e24f906554`, `8c05eabb54`, `7684225857`). Source: the
RC2 review of the idle policy (`95049973e0`) and flipfast (`502e97ef3c`), 2026-09-24.
**RC2-final** (`b5c7d01614` on the same branch, INT8 freeze accepted 2026-09-25) carries all of the
above unchanged; the entries under "RC2-final" at the end come from its INT8 acceptance boot
(`weg2rc2f`) and from the FP8 form (Agent K).
Fixed on that branch, so not listed here: SHORT drain re-taking a request D's budget refused
(`e24f906554`), law 4 summed over a drain (W153, `8c05eabb54`), and `--d-hold-s 0` meaning
off (`7684225857`).

**Drained SHORT requests can wait out a P phase (L2).**
- Effect: a SHORT request that `--d-short-drain-tokens` moved into D's admission line can be
  carried across a D->P flip. It is not lost (D serves it after the P phase), but its latency
  grows by one P phase plus two flips.
- Trigger: `--d-short-drain-tokens > 0`, a LONG request queued, and D's last decode ending
  just before the admitter's next poll (50 ms vs. the 200 ms controller tick). The same race
  exists for P-prefilled requests.
- Workaround: none needed; `--d-short-drain-tokens 0` turns the drain off.

**Front waits for the next request after fairness plus an abort (L3).**
- Effect: requests waiting for a D seat stay unserved until any new request arrives.
- Trigger: the fairness bound fired (`--fairness-w-s`), then `/abort_request` removed every
  queued request while others still waited for a D seat. Client disconnects do not trigger it.
- Workaround: send any request to the front; avoid `/abort_request` during a pre-emption.

**The D hold can end early after very short requests (L5).**
- Effect: under `--d-hold-s T` the idle flip to P can come up to T earlier than T seconds
  after D's last work.
- Trigger: a request that starts and ends on D between two controller ticks (< 0.2 s).
- Workaround: none needed (one flip earlier, nothing lost); choose a larger T if it matters.

**Drained requests look like P-prefilled ones in the logs (L6).**
- Effect: requests the drain hands to D log `WEG2 D-ADMIT ... source=batch`, like requests P
  prefilled; the `WEG2 D-SHORT-DRAIN` line lists only the first 8 request ids.
- Trigger: `--d-short-drain-tokens > 0`; log analysis that reads `source=batch` as "prefilled
  on P", or drains of more than 8 requests.
- Workaround: count drained requests with the counters `d_short_drain_requests` and
  `d_short_drain_tokens` in `GET /weg2/state`, not from D-ADMIT lines.

**A cancelled image request can still trigger the urgent vision flip (L7).**
- Effect: with `SGLANG_WEG2_VISION_FLIP_URGENT=1`, an image request whose client went away
  stays in the front queue and still flips D->P; P prefills a request nobody waits for (one
  wasted flip and prefill, no hang).
- Trigger: a queued image request cancelled by its client before its P phase (only
  `/abort_request` removes queue entries).
- Workaround: abort it through the front's `POST /abort_request` with its rid, or leave the
  switch off (the request then waits for the fairness bound or `--d-hold-s`).

**The exchange census underprices the dormant residue, INT8 and FP8 alike (K, weg2xsn442).**
- Effect: the census's `dormant_proc_used_mib` comes from weg2xsn246 (5090: 1622 MiB), but the
  current form measures 2442-2476 MiB (INT8 weg2rc1) and 2616-2652 MiB (FP8 weg2xsn442) per
  process on the 5090, so W71's flip-peak check is priced about 0.85 GiB per process too low.
- Why the boots still run: a census measured on the current FP8 form (xchg_census_weg2xsn442_
  b434831067.json) makes the check refuse (5090 p2d peak 32202 of 32607 MiB, below the 1229
  MiB floor), yet weg2xsn442 flipped six times on the same bytes -- the peak model is
  conservative by at least that much, and the old census happens to cancel it.
- Workaround: keep the INT8 census (FP8: plus `--weg2-xchg-census-foreign`); do not swap in a
  freshly measured census until the dormant/peak terms are re-derived.

## RC2-final (`b5c7d01614`)

**The FP8 checkpoint prefills at about 55-65 % of the INT8 speed (K, FP8 form).**
- Effect, in K's words: "The FP8 checkpoint (Qwen/Qwen3.8-27B-FP8) runs as weight-only FP8 (W8A16, Marlin) on
  every card, the RTX 5090 included, so that both flip groups hold the same byte layout. Prefill reaches about
  55-65 % of the INT8 checkpoint's speed (each pipeline stage computes in BF16 instead of INT8 tensor cores), while
  decode speed is about the same. If prefill speed matters, use the INT8 checkpoint."
- Trigger: profile `27b-fp8` (`--fp8-uniform-marlin`). K's figure comes from weg2xsn442 (tree `b434831067`).
- Measured on the RC2-final acceptance boots (P-prefill ladder, prompt tokens over the median P wall of the front's
  leg 1, no flip; same method for both): FP8 `weg2rc2f8` 3054 / 3899 / 3806 tok/s against INT8 `weg2rc2f`
  6600 / 8809 / 7711 tok/s at 2048 / 8192 / 32768 prompt tokens -- **44-49 %**, below the 55-65 % above. The
  per-rank GPU time agrees (INT8/FP8 0.38-0.48). Decode stays at INT8 level (FP8 acceptance, operator 25.09.).
- Workaround: the `27b` profile (INT8) where prefill time matters.

**The first D->P flip after start takes 8-16 s (L, non-blocking).**
- Effect: the first flip from the decode group to the prefill group after a boot takes 8-16 s; later flips of the
  same boot are several times faster. In `weg2rc2f` (front log, `WEG2-FLIP begin` -> `done`) the first D->P flip
  (epoch 0 -> 1, 01:12:21 -> 01:12:37) took 16.2 s (`sleep=15367 ms`), the P->D flip after it 6.0 s, every later
  flip of that boot 1.5-2.9 s. (These are the front's flip spans, not the flip time from prefill end to the first
  decoded token.)
- Trigger: the first D->P flip of every boot. With `--idle-layout pp` that is usually the idle flip to P after the
  first requests. The cause is not named yet; it happens once per boot.
- Workaround: none needed. A client that sends its first long request right after a boot waits up to that long
  once; a warm-up request after `serving` moves the cost out of the user's path.

**A client that closes right after a complete answer leaves a 503 in the access log (L, cosmetic).**
- Effect: the front logs `WEG2 leg2 rid=<rid> failed: ClientConnectionResetError: Cannot write to closing transport`
  and an access line `"POST /v1/chat/completions HTTP/1.1" 503 0`, although the client already received the
  complete answer. Nothing is lost; D frees the seat as usual (`WEG2 D-REFILL ... freed_by=leg2_finished`).
- Trigger: a client that closes its connection as soon as it has read the full response (seen with Python
  `urllib` in the acceptance probes: `weg2rc2f`, rid `weg2-14-47`, 01:16:55-01:17:01).
- Workaround: none needed. Do not count these access-log 503s as failed requests; count failures on the client
  side, or pair each 503 with its `leg2 ... failed` line and the client's own result.

## RC4 (`9738626129`)

**A wake disk reload would take the process-wide load format, not its runner's (K, latent, fix not in RC4) -- behoben in RC5 (`5f13f1aad9`).**
- Effect: `--speculative-draft-load-format auto` (needed so the DFlash2-lued-W8 draft does not inherit `gguf` from a GGUF target, S) makes the scheduler set `server_args.load_format` to the DRAFT's format for the whole process. Both disk-reload wake paths read that value (target `_weg2_wake_reload_weights`, draft `_weg2_xchg_draft_reload_from_disk` #1394), so a target reload would open the `.gguf` with `auto` and die named (`Weg2WakeRefused`).
- Not reachable in the four RC4 forms: under `exchange` + `authoritative` the target wake is the exchange collect, and on every quantized checkpoint (INT8, FP8, NVFP4, GGUF, the lued draft) both disk paths are refused by W4 (`assert_backup_off_wake_refill_is_defined`) before the format is read -- a GGUF boot cannot show this fallback.
- Fix candidate for the next state: `desk/27b-up-wake-reload-loadcfg-0925` (`b857a22cb5`) -- each reload passes its own runner's `load_config.load_format`.
- **Behoben in RC5 (`5f13f1aad9`, `desk/27b-release-rc5-0925`).** `b857a22cb5` is on the first-parent chain (fast-forward on `9738626129`); fetch-checked 2026-09-25 with `git ls-remote` tip = `5f13f1aad9008c08cbd94c788cd602c458299dc0` and `merge-base --is-ancestor`. `weight_updater.py` is unchanged from `b857a22cb5` to `5f13f1aad9`. S's `1f8c24d338` in the same RC also removes the root cause: there is no process-wide `server_args.override(load_format=...)` any more. Each runner builds `LoadConfig(load_format=ModelRunner._this_runners_load_format())`, so both reloads read their own runner's format.

**Group P does not zero the Marlin lock workspaces of its DFlash draft on wake (L, latent, fix not in RC4/RC5).**
- Effect: after every wake of group P, the 21 Marlin lock workspaces (`scheme.workspace`) of the W8 DFlash2 draft
  on P's last stage (the #1233 draft-KV producer, tag `weights_draft`) keep whatever the recycled pages hold. The
  wake zeroing (`WEG2-RESUME local-scratch zeroed=`) reaches the target and group D's draft, but it asks only
  `draft_worker` for the draft (`_weg2_wake_models` -> `_weg2_model_for_group("D")`), and P's last stage has no
  `draft_worker`: it holds its drafter in `draft_kv_producer.draft_runner`. `_weg2_drafter_of` (#1378 xsn78)
  already knows both places.
- Measured on the RC4 boots `weg2rc4f8`, `weg2rc4`, `weg2rc4n4`: the `WEG2-XCHG-COVER` local_scratch sum per rank
  against the `zeroed=` count per wake. Group D is complete (FP8 277 of 277, INT8 21 of 21, NVFP4 279 of 278).
  PP0 and PP1 are complete. PP2 is short by exactly 21 (`weights_draft`) in all three formats: FP8 44 of 65,
  NVFP4 45 of 66, INT8 0 of 21 (no resume line at all).
- Why nothing happens today: with `--dflash-produce-on-p off` (user decision 2026-09-24, the RC4 standard form)
  the producer's draft is built and stays cold-resident, and no kernel runs on it after a wake. Each RC4 boot shows
  the `WEG2 DFLASH-PRODUCE-ON-P off` lines and 0 `WEG2 DRAFT-KV-PRODUCE rid=` lines. A non-zero lock only matters
  once the producer computes: then the kernel can spin or read a partial tile.
- Trigger: `--dflash-produce-on-p on` (group P env `SGLANG_WEG2_DFLASH_PRODUCE=1`) with a draft that has Marlin
  workspaces. The lued W8 draft has 21.
- Workaround: keep `--dflash-produce-on-p off`. Before switching it on, `_weg2_wake_models` must take the drafter
  through `_weg2_drafter_of(self)` (a one-line fix plus a test). The SOLL/IST check R4 counts resume lines against
  wakes, not the zeroed count against the COVER sum, which is why it does not show this; a per-rank check
  "zeroed per wake == COVER local_scratch sum" would.
