# GPU ticket #486 — does the per-decode KV reserve touch radix at all?

Branch `fix/eagle-reserve-need-486`. The desk work is done and committed; this
ticket is the measurement that either confirms or refutes the desk conclusion.

**Read the desk conclusion first, because it decides whether this window is
worth its slot.** #486 derived the reserve as `W + L` (write footprint + commit
lag) and found that on the NEXTN production recipe `W == L == 4`, so the
pre-#486 blanket `2 x W = 8` was already exactly the derived need. The fix
saves **zero slots on the production recipe**. Standing occupancy there is
`bs x 8` slots = **2.0 MiB at bs=16** on the heaviest rank, against a per-rank
pool of ~84k-175k tokens — i.e. **~0.15 % of the smallest rank's pool**.

Upstream issue #32459 reports EAGLE collapsing radix reuse 97% -> 40-53% and
upstream PR #32574 blames this reserve. **On our shape that mechanism is too
small by three orders of magnitude to produce that.** The purpose of this
window is therefore NOT to demonstrate a win. It is to answer one question:

> Is the per-decode reserve measurably visible in radix hit rate / pool
> occupancy at all on our recipe — yes or no?

A clean **no** is the expected and useful result. It closes the #32459
adoption question for this fork and tells anyone re-reading upstream's issue
that the diagnosis does not transfer here. Do not manufacture a win.

### The desk arithmetic this window is checking

Qwen3.6-27B-INT8-W8A8, `--kv-cache-dtype fp8_e4m3`, tp=3. 64 layers with
`full_attention_interval 4` -> 16 KV-holding layers; `num_key_value_heads 4`,
`head_dim 256`; the heaviest of three ranks holds 2 of the 4 KV heads, so
`16 x 2 x 2 x 256 = 16384 B/slot` in the target pool plus `1 x 2 x 2 x 256 =
1024 B/slot` in the NEXTN draft pool (`mtp_num_hidden_layers 1`) =
**17.0 KiB per reserved slot on the heaviest rank**.

Reserve slots per running request, and the MiB #486 reclaims:

| shape | old `2xW` | new `W+L` | delta | bs 1 | bs 8 | bs 16 |
|---|---|---|---|---|---|---|
| NEXTN s3 / topk1 / d4, overlap — **production** | 8 | 8 | **0** | 0.000 | 0.000 | 0.000 |
| NEXTN s3 / topk1 / d4, `--disable-overlap-schedule` | 8 | 4 | 4 | 0.066 | 0.531 | 1.062 |
| chain s6 / topk1 / d4, overlap | 12 | 10 | 2 | 0.033 | 0.266 | 0.531 |
| tree topk4 / s3 / d8, page 1 | 24 | 20 | 4 | 0.066 | 0.531 | 1.062 |
| tree topk4 / s3 / d8, page 64 | 1024 | 520 | 504 | 8.367 | 66.938 | 133.875 |

Whole standing reserve posten on the production shape: 0.133 / 1.062 /
**2.125 MiB** at bs 1 / 8 / 16 — **0.153 %** of the smallest rank's pool
(83 775 tokens, the `auto-performance` capacity from `NOTE_433`) at bs=16.

DSV4F offload recipe: same structure with the DFLASH block as `W` and `L`
(`block_size` both terms under overlap), so its delta is 0 as well; the
per-slot bytes there are MLA-shaped (`kv_lora_rank + qk_rope_head_dim`,
replicated not TP-split) and should be measured on the boot rather than
predicted.

---

## 0. Preconditions

- Hold `/spinning/gpu-arb/` for the whole window; stop the heartbeat BEFORE
  releasing.
- Corridor sampler started **before** the server, sampling **during load**.
  Floor: **>=400 MiB free per card**, judged at peak, not idle.
- `--rank-gpu-id` takes **CUDA ordinals**, not NVML indices. Resolve via the
  IdentityMap (`registry/nvml.py`) at run time; never assume.
- `--enable-metrics` and `--enable-cache-report` on **every** boot line
  (cache-report is what makes `cached_tokens` readable per request — it is the
  radix hit-rate instrument, and #32459's numbers come from it).
- Own PIDs via pidfiles. Never `pkill -f`. `py-spy dump` before killing
  anything wedged. Bounded waits only.
- **A-vs-A floor first** (Benchmark-Harness-Pflicht 5). Two identical arms of
  the SAME build before any A/B. Nothing below that floor gets reported.
- Full perf: CUDA graphs + spec on. Not eager.
- Time-bounded arms (10-20 s of steady decode), prefill the working point
  rather than growing into it.

## 1. Boot line — standard INT8 recipe

```bash
setsid "$VENV/bin/python" -u -m sglang.launch_server \
  --model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8 \
  --tp-size 3 --rank-gpu-id <NVML-resolved> --rank-tp-ratio auto-performance \
  --rank-auto-reserve-mib 5500,3800,3800 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --context-length 32768 --max-running-requests 16 \
  --enable-metrics --enable-cache-report \
  --host 127.0.0.1 --port 30486
```

`$VENV` = `/spinning/htsglang-gpu/.venv`.

## 2. Arms

| arm | build | reserve on the production shape | purpose |
|---|---|---|---|
| A0 | this branch | 8 slots/req | A-vs-A floor, run twice |
| A1 | this branch | 8 slots/req | the A-vs-A partner |
| B  | this branch + `--disable-overlap-schedule` | 4 slots/req (was 8) | the only production-reachable arm where #486 actually changes the number |
| C  | this branch, `--speculative-eagle-topk 4 --speculative-num-draft-tokens 8` | 20 slots/req (was 24) | topk>1, if it boots on this rig |

A0/A1 must be indistinguishable. If they are not, the window stops and the
harness gets fixed first.

Arm B confounds overlap itself with the reserve change, so B is **not** a
clean attribution of the reserve — read it only as "does halving the reserve
move anything at all". To separate the two, run B twice: once on this branch
and once on `origin/integration/r3-probe-next2` (which reserves 8 even without
overlap). That pair IS clean: same overlap setting, reserve 4 vs 8.

## 3. Counters — work-matched, per rank

Per arm, over the same fixed prompt set (multi-turn, each request extending
the previous turn's history — that is the traffic shape #32459 used):

- `cached_tokens / prompt_tokens` per request, **binned by prompt depth**.
  Deep bins (>=8k here, the rig cannot do #32459's 20k) are the ones that
  isolate reuse from workload mix.
- Radix evictions per decode step and evicted-token count
  (`tree_cache.evict` call count + `num_tokens`).
- Pool occupancy at steady state: `allocator.available_size()` and the
  committed/reserve/radix split of `C_target` — the three postens named in
  `DESIGN_330_vram_dial.md` §3b. **Report the reserve posten explicitly**;
  the point of #486 is that it is never again an uncounted transient.
- **ms/verify round and ms/prefill-1k**, per rank, compute vs wait. Not tok/s.
- `meta_info.spec_accept_length` + `spec_verify_ct` / decode seconds. NOT
  `spec_ema_accept_len`.

## 4. Deliverable: one honest sentence

> On the standard INT8 NEXTN recipe at bs 1-16, the per-decode KV reserve
> occupies X MiB (Y % of pool), radix hit rate is Z% with the derived reserve
> vs Z'% with the old blanket 2x, and the difference is / is not above the
> A-vs-A floor of F.

If Z == Z' within the floor — the expected outcome — record it and close the
#32459 adoption question. That is the result.

## 5. What this window does NOT do

- It does not chase upstream #32459's actual root cause. If radix reuse under
  spec IS bad here, that is a NEW ticket: the candidates are page-aligned
  committed-length truncation at insert time and the spec insert path's
  `enable_kv_committed_len = topk is None or topk == 1` gating in the
  flexkv/lmcache radix wrappers — not the reserve.
- It does not touch the DFLASH/DSpark solo lanes. Their reserve now shares the
  same derivation (`dflash_info_v2.py` `prepare_for_decode`), and their
  `W == L` too under overlap, so they have the same zero-delta prediction.
