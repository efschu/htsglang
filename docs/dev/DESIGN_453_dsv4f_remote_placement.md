# DESIGN #453 — extending DSV4-Flash MoE placement onto remote resources

Design only. No build, no GPU allocation, no change to rig 2. This document
prices three axes separately against MEASURED numbers from this rig pair,
designs the user-headroom reservation that any rig-2 use is conditional on,
states the fixposten of every window it would take to decide anything, and
ends with a recommendation per axis — including the two axes it recommends
closing.

It supersedes the desk assessment `NOTE_453_remote_expert_lane.md` on two
points and confirms it on a third; each divergence is named in §7.

Tree: `docs/design-453` off `origin/integration/r3-probe-next2` @ `f51b90f564`.

---

## 0. Summary of verdicts

| axis | verdict | the number that decides it |
|---|---|---|
| (a) rig-2 RAM as a cold expert tier | **REJECT** | the fastest measured remote rate (2.83 GB/s) is 2.28x SLOWER than the slowest local H2D link (6.45 GB/s), and the cold set already fits local host RAM |
| (b) rig-2 CPU as an expert compute lane | **PARK**, with a numeric revisit predicate | capacity-capped yield is 2.2 % end-to-end against a 4.09 % A-vs-A floor; the lane needs >=35 % of the cold set in remote RAM, i.e. >=30 GiB against 15 GiB installed |
| (c) 2080 Ti as an expert-compute arm | **REJECT** | format-gated before weight load (`loader.py:229-236`), the card is 4.5x slower than the local decode card on compute (#212), and it hangs off a Gen3 x4 chipset link shared with the NIC |

The one thing worth doing cheaply is neither of the three: **register rig-2's
RAM and CPU as declared #407 tiers with headroom-reduced capacity**, and point
them at latency-tolerant consumers (hibernate staging, the #306 compression
workbench, #224 session park) where 2.83 GB/s beats the 1.8 GB/s local NVMe
tier on rate. That is S effort and touches no MoE code. See §6.4.

---

## 1. The measured base this document prices against

Everything below is derived from artifacts on this box. Nothing is imported
from general knowledge, and every derived quantity names its inputs.

### 1.1 The vehicle

DeepSeek-V4-Flash-0731-GGUF `UD-IQ3_XXS`, 43 layers, 256 routed experts,
top-6, `hidden_size` 4096, `moe_intermediate_size` 2048, one shared expert
(`config.json` beside the GGUF shards). Checkpoint on disk: 98 GB. Cold
(host-RAM-resident) expert tier on the reference recipe: ~88 GiB
(`ANALYSE_393_ik_llama.md:783`, the `/dev/shm` remount preflight).

### 1.2 Decode and expert traffic, measured

Window `/spinning/gpu-battery-results/2026-08-03_439_green/`, `equal` arm
(TP=3 uneven, bs=1, eager offload, `--disable-cuda-graph`). Decode
**135.07 ms/token** (that arm's `decode_equal.json`, quoted as the TP=3
reference in `2026-08-03_445_pp3/RESULTS.md:23`).

From the three `expert_stats_equal.tp*ep0.json` dumps of that same arm. Note
the dump's `tokens` field counts **layer-token events**: 163 486 / 43 layers =
**3802 real tokens**, and `activations` 980 916 = 3802 x 43 x 6 confirms it
exactly (top-6).

| rank | card / slot | `h2d_bytes` | per token | measured link | transfer time/token |
|---|---|---:|---:|---:|---:|
| tp0 | RTX 5090, x8 | 2 017 174 290 432 B | **530.6 MB** | 14.42 GB/s | 36.8 ms |
| tp1 | RTX 3080, **x4** | 1 285 686 362 112 B | **338.2 MB** | 6.45 GB/s | **52.4 ms** |
| tp2 | RTX 3080, x8 | 1 335 825 530 880 B | **351.3 MB** | 13.41 GB/s | 26.2 ms |
| aggregate | | | **1220.1 MB/token** | 32.4 GB/s | |

Links from `ANALYSE_389_nvme_expert_tier.md:328-330` /
`DESIGN_407_memory_tier_registry.md:134`. The x4 rank is the clock, at
**52.4 ms/token** of expert transfer, i.e. **1.22 ms per MoE layer**.

This 1.22 ms is the single most load-bearing number in this document: it is
the budget any remote round trip has to fit inside, and the value any remote
share has to buy back.

### 1.3 The exposure coefficient (why the naive share is wrong)

52.4 / 135.07 = 38.8 % of the token is expert transfer on the clock rank —
but transfer overlaps compute, so only part of it is exposed end-to-end. The
exposure is **measured**, not assumed, from the two #439 windows that moved
exactly this term:

| window | transfer-term speedup | end-to-end | implied saving of the term | exposed share f |
|---|---|---|---|---|
| `2026-08-03_439_green` | 1.4307x | **-6.42 %** | 1 - 1/1.4307 = 30.1 % | 6.42/30.1 = **21.3 %** |
| `2026-08-03_439_confirm` | 1.496x | **-7.67 %** | 33.2 % | **23.1 %** |

(both quoted in `FEATURE_CATALOG.md:578-582` and
`ANALYSE_456_dsv4f_matrix_sweep.md:40-52`.)

So **f = 0.21-0.23**: a lever that removes share `s` of the clock rank's
expert transfer moves the token by `s x 22 %`, not by `s x 38.8 %`. Equivalently,
the EXPOSED transfer is 0.22 x 135.07 = **29.7 ms** of the nominal 52.4 ms, an
overlap factor of 0.57. Every end-to-end figure below uses f = 0.22, i.e.

    end_to_end_delta(s) = 22 % x s      (of decode)
    exposed_ms(s)       = s x 29.7 ms

One asymmetry to keep in view: f was measured on a lever that REMOVED transfer.
Applying the same 0.57 overlap to a lever that ADDS transfer (axis (a), §3.2)
is conservative — added bytes are less hideable than removed bytes are
recoverable — so the regression figures in §3.2 are lower bounds.

### 1.4 The measurement floor

Same-boot A-vs-A on this recipe: **4.09 %** (the #439 green window's own
floor), and 3.6-5 % on 2026-08-02
(`ANALYSE_393_ik_llama.md:769-771`). Anything predicted below ~4 % end-to-end
is not callable on this rig even if it is real. This floor kills axis (b) at
its reachable operating point, and it does so on arithmetic, not on taste.

### 1.5 The remote link and the remote box

| quantity | value | source |
|---|---|---|
| 40G RoCE, NCCL over sockets | **2.07 GB/s** | `INTEGRATION_R3_VALIDATION.md:5053` (#201); tier table `DESIGN_407_memory_tier_registry.md:137` |
| 40G, staged RDMA at 1 MiB | **2.83 GB/s** | `DESIGN_407_memory_tier_registry.md:137` |
| 100G line, slot-limited | 3.43 GB/s, "PCIe-3.0-x4-bound" | same row |
| point latency, 8 B | 1.47 us (40G) | same row |
| UCX all_reduce, 8 KiB | 44.92 us | same row |
| sglang PP boundary at bs=1 | 10 KiB in **142 us** + **249 us** pickled metadata, one way | `INTEGRATION_R3_VALIDATION.md:5049-5052` |
| NCCL verbs on this RoCE line | broken (`IBV_WC_REM_INV_REQ_ERR`); sockets and UCX work | `INTEGRATION_R3_VALIDATION.md:5058-5062` |
| the NIC relay serialises | 3 parallel pairs: 2.34x latency for 1.28x aggregate | `DESIGN_407_memory_tier_registry.md:144-146` (#278 V5) |

**Rig-2 host facts** (recorded read-only 2026-07-25 and corrected 2026-07-28;
**NOT re-verified this session, because rig 2 is under a user lock — see
§3.1**): Ryzen 5700G APU, 8 cores / 16 threads, Zen 3, AVX2, no AMX;
**15 GiB RAM**; RTX 2080 Ti (sm75) behind the B550 chipset at Gen3 x4; NIC is
a **ConnectX-5 Ex on the CPU link at Gen3 x8** — not a CX-4 at x4; no discrete
Vega currently installed; it is a **used desktop with an active display
manager and a logged-in user**.

Two provenance notes, both honest and both cheap to settle:

* The briefing's premise "PCIe-x4 wall of the CX-4" is **not what the recorded
  inventory says**. The x4 wall belongs to the 2080 Ti's chipset link, not to
  the NIC. This does not change any measured number — the 2.07/2.83/3.43 GB/s
  figures were measured end to end and stand as the ceiling regardless — but
  it does change the *reason*, and a wrong reason produces wrong follow-ups
  (e.g. "move the NIC to another slot").
* `DESIGN_407:137` calls the 100G line "PCIe-3.0-x4-bound" while the inventory
  puts the ConnectX-5 Ex at Gen3 x8. One of the two is stale or they describe
  different cards. This is a Cut-0 item (§8.1), not a blocker: both candidate
  ceilings (3.43 and 2.83 GB/s) are far below the slowest local link, which is
  what axis (a) turns on.

---

## 2. Mechanism reach — what exists, what does not, quoted at the source

Required before any exclusion claim below is allowed to stand.

**(1) No existing mechanism reaches another host with expert bytes.** The
#394 cold tier is host-local shared memory. `cold_tier_fetch.py` header,
lines 36-40:

> `ColdTierResolver` maps the segment once per process and hands back a
> zero-copy `torch` view over the peer's DRAM
> (`cold_tier_shm.peer_row_tensor`). The existing fetch path then issues the
> same `copy_` it always did — **over THIS rank's own PCIe link, which is the
> only link a rank can pull over.**

Its `remote_ids` (`cold_tier_fetch.py:231`, `:495-496`) means "owned by
another RANK on this host", never another machine. So a remote tier or a
remote lane is **new transport work**, not a configuration of something that
already exists. Any effort estimate that assumes otherwise is wrong.

**(2) The GGUF-MoE offload exclusion that ANALYSE_393 called "on the critical
path" is GONE.** `ANALYSE_393_ik_llama.md:450-458` states
`_OFFLOAD_UNSUPPORTED_QUANT_METHOD_NAMES` hard-aborts for `GGUFMoEMethod` and
that lifting it is a non-trivial prerequisite for any CPU lane. Code today:

```python
# expert_offload.py:2071-2073
_OFFLOAD_CONDITIONAL_QUANT_METHOD_NAMES = {
    "GGUFMoEMethod": "_moe_offload_gguf_staged",
}
```

admitted per layer at `expert_offload.py:2111-2114`
(`if getattr(layer, marker, False): continue`). The unconditional denylist
(`:2058-2066`) now holds only `GGUFMoEAscendMethod`, `MoeWNA16Method` and the
three NVFP4 MoE methods. **A CPU or remote expert lane on the GGUF vehicle no
longer has this prerequisite.** ANALYSE_393 §7.7's first paragraph should be
marked superseded when someone next touches that file.

**(3) The sm75 GGUF gate, and it fires before weight load.** The chain,
verbatim:

```python
# gguf.py:40-62  (module import)
_has_sgl_gguf_kernels = False
if _is_cuda:
    try:
        from sgl_kernel import moe_sum
        from sgl_kernel.quantization import (ggml_dequantize, ggml_moe_a8, ...)
        _has_sgl_gguf_kernels = True
    except ImportError:
        ...

# gguf.py:150-152
if _is_cuda:
    return _has_sgl_gguf_kernels        # GGUFConfig.supports_current_device

# base_config.py:185-193 — GGUFConfig does NOT override this
def needs_device_kernel(self) -> bool:
    return True

# loader.py:345-346
if not _is_npu and quant_config.needs_device_kernel():
    _enforce_capability_floor(quant_config, model_config)

# loader.py:229-236
if supported is False:
    raise ValueError(
        f"The quantization method {model_config.quantization} is not "
        "supported on this device: the kernel it requires does not exist here. ..."
    )
```

That is the excluding predicate for axis (c), and it is a load-time
`ValueError`, reached for every GGUF checkpoint on a card whose sgl-kernel
GGUF cubins failed to import — which is exactly Turing.

**Register correction owed.** `planner/rejected.py:263-281`
(`key="gguf_on_sm75"`) still says the failure is *"the promised loud failure
at GGUFConfig is not wired (open bug #269)"* and that the rank *"loads its
weights and then dies mid-forward"*. Code (above) contradicts it: the refusal
is wired and fires before weight load. Code wins; the entry's `verdict`,
`cost` and `evidence` fields should be updated in whatever change next touches
that file. This document does not edit code.

**(4) The headroom primitives already exist and are generic.**
`memtier/tiers.py:345-368` — `TierCapacity` carries `total`, `floor`,
`reserved` ("Bytes declared by other holders") and `corridor` (#330's
absolutely-free rule). `training/tenant.py:65-77` defines
`DemandSample`/`DemandSource` as a plain callable contract, and
`IdleMonitor.sample()` (`:207-224`) combines any number of them into one
verdict with a grace period. `workbench/arb.py` (header, lines 1-38)
implements published-availability + heartbeat + "before every access the
hardware is checked" as a cross-session protocol whose directory path "is a
flag, not a constant". **None of these has a remote-host consumer today** —
but none of them needs a new concept for one, which is what §4 builds on.

---

## 3. Axis (a) — rig-2 RAM as a cold tier for experts

### 3.1 The precondition that gates even the cheap probe

Rig 2 is **locked by user order (2026-07-29)**: no tests, no SSH from us until
released. Independently, sm75/Turing work was deferred "all the way back"
(user, 2026-07-28), covering #168, #235 and the sm75 half of #201 slice 3.
Neither order is overridden by this design. Every window in §8 is therefore
conditional on the lock being lifted, and the lock is why §1.5's rig-2 facts
carry a "not re-verified" label rather than a fresh read.

### 3.2 Rate: the remote tier is slower than the slowest local link

| path | rate | vs the clock rank's own link (6.45 GB/s) |
|---|---:|---:|
| local pinned host RAM -> tp1 (x4) | 6.45 GB/s | 1.00x |
| local pinned host RAM -> tp0/tp2 (x8) | 13.4-14.4 GB/s | 2.1-2.2x faster |
| **remote rig-2 RAM, staged RDMA** | **2.83 GB/s** | **2.28x slower** |
| remote rig-2 RAM, NCCL sockets | 2.07 GB/s | 3.12x slower |
| 100G line, slot-limited | 3.43 GB/s | 1.88x slower |

Shifting share `s` of the clock rank's 338.2 MB/token to the best measured
remote path multiplies that portion's time by 2.28:

    delta_transfer(s)   = s x 52.4 ms x (2.28 - 1) = s x 67.1 ms   (nominal)
    delta_end_to_end(s) = 0.57 x delta_transfer(s)                 (exposed)

At s = 0.10 that is +6.7 ms nominal, **+3.8 ms/token exposed, +2.8 % decode**
— a regression, not a gain, and a lower bound per §1.3's asymmetry note. There
is no `s > 0` at which the substitution helps on rate.

### 3.3 Capacity: the motive is empty for this vehicle and dominated for the next

`NOTE_453` §1 conceded the rate argument and kept remote RAM alive as a
**capacity** tier "for the coldest experts". That surviving claim does not
survive the numbers:

* **This vehicle does not need it.** The cold set is ~88 GiB inside a
  98.5 GiB cgroup limit (`DESIGN_407_memory_tier_registry.md:133`,
  `ANALYSE_393_ik_llama.md:783-786`). Nothing overflows, so there is no
  capacity to buy.
* **For a vehicle that does overflow, the local NVMe tier dominates it.**
  T3 = 729 GB free at 1.8 GB/s measured cold
  (`DESIGN_407_memory_tier_registry.md:135`); T4 = at most ~9 GiB usable after
  a desktop reservation, at 2.83 GB/s. That is **80x the capacity at 0.64x the
  rate**. A capacity tier that holds 1.2 % of what the alternative holds is
  not a capacity tier; and both are unbuilt, so neither gets a head start.
* **The share it could hold is a rounding error.** 15 GiB gross / 88 GiB cold
  set = 17 %; after a plausible 6 GiB desktop reserve, ~10 %.

### 3.4 The one surviving question belongs to #423, not here

`DESIGN_423_striped_offload.md` §2 states the only condition under which a
remote path adds anything: disjointness **at the currently contended
resource**. Applied here, with its own two bullets:

* **On the clock rank the contended resource is its own PCIe hop.** DESIGN_423
  §2 bullet 1: *"A fetch from local host RAM and a fetch from a remote NIC both
  end by crossing that GPU's own PCIe link into VRAM. Striping 'local RAM'
  against 'remote NIC' adds nothing at that hop."* tp1 pulls 338.2 MB/token
  over 6.45 GB/s while local DRAM can deliver 32-45 GB/s; its binding
  constraint is the x4 link, which the remote path shares. **Not disjoint at
  the contended resource — refused by #423's own predicate.**
* **Host-DDR contention is plausible but unmeasured.** Aggregate demand is
  1220.1 MB per 52.4 ms = **23.3 GB/s**, i.e. 52-73 % of the 32-45 GB/s DDR
  band (`ANALYSE_393_ik_llama.md:299-300`, an ESTIMATE, flagged as such). So
  the DDR channel is loaded but not obviously saturated, and whether it ever
  binds is exactly the question `DESIGN_423` §4's probe exists to answer.

**Recommendation:** do not open a remote expert tier. Add one arm to the #423
probe when it runs — "remote-served share under artificial local-DDR load" —
because it costs one extra configuration in a session that already has the
rig-2 setup standing, and it is the only form of the question with a live
hypothesis behind it. Everything else about axis (a) is closed by §3.2/§3.3.

---

## 4. Axis (b) — rig-2 CPU as an expert compute lane

This is the axis the briefing asked to be worked through properly, because the
payload inverts: the wire carries **activations**, not weights. That inversion
is real, it is bigger than previously stated, and it still does not save the
axis. Both halves below.

### 4.1 The inversion, computed on this checkpoint

At bs=1 a MoE layer's remote call carries one hidden vector out and one
partial back:

    4096 x 2 B (bf16) in + 4096 x 2 B out   = 16 KiB per MoE layer per token
    x 43 layers                              = 688 KiB per token

Against 1220.1 MB/token of weight traffic (§1.2) that is a ratio of
**1 : 1770** — larger than ANALYSE_393 §7.2's 675:1, because that figure used
a per-hit activation model while a per-layer batched crossing is the shape a
real lane would take. At 2.83 GB/s the whole per-token payload costs
**0.24 ms**. Compare `NOTE_453` §1's verdict on remote RAM (2.28x too slow):
for a compute lane, **the bandwidth objection genuinely does not apply**. The
inversion holds.

### 4.2 What binds instead — four terms, priced

**(i) Round-trip latency, 43 times per token, and whether it hides.**
Routing for layer L is known at layer L's entry (the wave planner already
resolves per forward which experts are cold and which token rows route to
them — `ANALYSE_393_ik_llama.md:441-444`), so the crossing can be issued at
layer entry and awaited at the combine, concurrently with the local fetch of
the same layer. The budget is §1.2's **1.22 ms per layer**. Candidates:

| transport | round trip | fits 1.22 ms? |
|---|---:|---|
| sglang PP boundary as measured at bs=1 (142 + 249 us one way) | ~0.78 ms | yes, at p50 |
| UCX class (8 KiB all_reduce 44.92 us) | ~0.09-0.10 ms | comfortably |

So — and this corrects the intuition that 43 serial network round trips must
sink the idea — **the round trip hides at bs=1 on an idle link**. Latency is
not the binding term at p50. It becomes binding at p99 on a machine somebody
is using, which is why §8.2's falsifier is stated on p99 and not on p50.

**(ii) Rig-2's own DRAM read.** ANALYSE_393 §7.4's law applies unchanged to
the remote box: *"For any RAM-resident expert tier, DRAM is a hard floor that
no compute-device choice can beat."* Rig 2 serves share `s` of the aggregate
1220.1 MB/token from its own DDR4. At an assumed 25-30 GB/s (dual-channel
DDR4 on an APU, **unmeasured — this is a Cut-1 item**) and s = 0.10:
122 MB/token = 2.84 MB/layer = **95-114 us/layer**. Inside the budget.

**(iii) Rig-2's GEMM.** 122 MB/token of ~3.4 bpw weights is ~287 M params,
~0.57 GFLOP/token at 2 FLOP/param. On 8 Zen-3 AVX2 cores at the 0.5-1.2
TFLOP/s quantized-GEMM band ANALYSE_393 §7.2 established for the same
microarchitecture: 0.5-1.1 ms/token = **12-27 us/layer**, and it shares the
loop with (ii) rather than adding to it (`max`, not sum). Not binding.

**(iv) Capacity — the term that actually binds.** The lane can only compute
experts whose weights are in rig-2's RAM. `s <= 15 GiB / 88 GiB = 0.17`
gross, ~**0.10** after a desktop reservation. The inversion moved the *bytes*
off the wire; it did **not** move the *capacity* requirement, and that is the
honest bottom of this axis.

### 4.3 The yield, and why it is not measurable

Saving = the local transfer the lane removes from the clock rank, exposed
through f = 0.22 (§1.3): `end_to_end(s) = 22 % x s`.

| s | remote RAM needed | transfer saved (nominal) | exposed | end-to-end | vs the 4.09 % floor |
|---:|---:|---:|---:|---:|---|
| 0.10 | 9 GiB (reachable today, with reserve) | 5.2 ms | 3.0 ms | **2.2 %** | **below the floor — not callable** |
| 0.17 | 15 GiB (all of it, no reserve — refused by §6) | 8.9 ms | 5.1 ms | 3.7 % | at the floor |
| 0.35 | 31 GiB | 18.3 ms | 10.4 ms | 7.7 % | ~2x floor — callable |
| 0.50 | 44 GiB | 26.2 ms | 14.9 ms | 11.0 % | clearly callable |

The theoretical ceiling of a perfect lane on this pair, ignoring capacity, is
where the remote side becomes the clock:

    (1 - s) x 52.4 ms  =  43 x RTT  +  s x (1220.1 MB / R_rig2)
    with R_rig2 = 28 GB/s and RTT = 0.10 ms:
    52.4 - 52.4 s = 4.3 + 43.6 s   ->   s* = 0.55, token transfer 23.4 ms

i.e. **2.24x on the transfer term, about -12 % end-to-end** (22 % x 0.55),
needing ~48 GiB of remote RAM. That is the whole prize, and it is 3.2x the RAM
the machine has.

**So axis (b) is not absurd and it is not free money.** At the reachable
operating point it produces an effect this rig cannot distinguish from noise,
in exchange for an L-effort execution lane.

### 4.4 The costs that are paid before the first token

* **Byte identity.** A CPU GEMM has a different reduction order than the CUDA
  kernel (`ANALYSE_393_ik_llama.md:469-476`), so a remote-computed expert is
  not bit-identical; and *which* experts land remote depends on residency
  state unless the split is frozen. Under the quality-last rule this sits
  behind every byte-identical win. A remote lane inherits this in full, plus a
  second machine's libraries and ISA in the identity surface.
* **Graph route.** The #462 breakable route's per-layer host rendezvous
  (`FEATURE_CATALOG.md:497-502`: 43 irreducible rendezvous per step) becomes a
  per-layer NETWORK rendezvous. 43 crossings/token at p99 on a desktop is a
  tail-latency surface the local path does not have: one browser start turns a
  hidden 0.19 ms call into 43 stalls per token. The local fallback path must
  therefore always exist, which it does — the lane is an optimisation over a
  fetch that still works.
* **NIC serialisation.** Three TP ranks calling one remote host share one NIC,
  and #278 V5 measured 2.34x latency for 1.28x aggregate on exactly that
  shape (`DESIGN_407_memory_tier_registry.md:144-146`). The per-layer budget
  is a p99 budget for **three** concurrent callers, not one.
* **One runtime.** The rig-2 side must be an htsglang worker — a lane of the
  same runtime holding expert weights in host RAM and returning partials —
  not a foreign process. That is the law, and it is also what makes the
  headroom ledger of §6 possible at all: a foreign process is invisible to it.

### 4.5 Verdict and revisit predicate

**PARK.** Revisit when any ONE of these becomes true, and not before:

1. **Rig-2 RAM >= 48 GiB** (measured, after the reservation of §6), giving
   s >= 0.5 on this vehicle and a predicted -11.0 % end-to-end; or
2. **a vehicle whose cold set is <= 3x the usable remote RAM** — the yield is
   driven by the *ratio* s, not by absolute size, so a model with a ~27 GiB
   cold set and 9 GiB of usable remote RAM lands at s = 0.33 and the same
   ~8 % relative win; or
3. **the transport lands anyway** for NORDSTERN (RDMA TP=5 across the two
   rigs), at which point the lane's marginal cost is the dispatch and the
   determinism gate, not the transport.

Absent all three, the axis stays parked and no window is spent on it beyond
Cut 0/Cut 1 (§8), which are cheap and also serve #423.

---

## 5. Axis (c) — the 2080 Ti arm

**Closed, and the format gate is only the first of four independent reasons.**

1. **Format gate, quoted at the source (§2 item 3).** `loader.py:229-236`
   raises before weight load for any GGUF checkpoint on a card without
   sgl-kernel GGUF cubins. DSV4-Flash's driver on this rig is GGUF
   (UD-IQ3_XXS active, Q3_K_XL staged). What *would* run on sm75 is an fp16
   safetensors checkpoint (no native bf16 on Turing) or the #168 fp8-W8A16
   path — and #168 is a standing user refusal, not an open question
   (`NOTE_453_remote_expert_lane.md:65-68`). There is no checkpoint class in
   this project's DSV4F line that the card can execute.
2. **The card is the slow part, measured.** #212's satellite attribution:
   **93.5 % of the satellite's 2.892 s TTFT was 2080 Ti compute**
   (2385 vs 10850 tok/s), 1.8 % transport
   (`INTEGRATION_R3_VALIDATION.md:5012-5016`). An expert-compute arm is a
   compute arm; the one number we have about this card as a compute arm says
   it is 4.5x slower than the local card at prefill-class work. The satellite
   paid off in *undisturbedness*, not throughput — and an expert lane inside a
   token's critical path has no undisturbedness to sell.
3. **Its link is the worst one in the pair.** The 2080 Ti hangs off the B550
   chipset at Gen3 x4, sharing the chipset uplink with the NIC (inventory
   2026-07-28). Expert bytes destined for that card would contend with the
   wire delivering them.
4. **Standing user orders.** sm75 deferred to the back (2026-07-28); rig 2
   locked (2026-07-29).

Reason 1 alone closes it for this vehicle; reasons 2 and 3 close it for any
vehicle on this hardware. Recommended `planner/rejected.py` entry (text ready
to lift, to be added by whatever change next touches that file):

```
key="remote_2080ti_expert_arm"
what="routing DSV4-Flash expert compute onto rig-2's RTX 2080 Ti"
verdict="BLOCKED: the GGUF checkpoint is refused before weight load on sm75
    (loader.py:229-236 via GGUFConfig.supports_current_device, gguf.py:150-152),
    and the only sm75-runnable alternative (fp8-W8A16, #168) is a standing user
    refusal. Independently NOT WORTH IT: #212 measured 93.5 % of satellite TTFT
    as 2080 Ti compute (2385 vs 10850 tok/s), and the card sits on a Gen3 x4
    chipset link shared with the NIC uplink."
level=BLOCKED
evidence="#212 (INTEGRATION_R3_VALIDATION.md:5012-5016); gate chain gguf.py:40-62,
    :150-152 + base_config.py:185-193 + loader.py:345-346, :229-236;
    DESIGN_453_dsv4f_remote_placement.md §5"
tags=("moe", "sm75", "cross-rig")
```

---

## 6. User headroom on rig 2 — a first-class design element

Rig 2 is somebody's desktop. This section is a precondition on every window in
§7, including the probes — a measurement is not exempt from politeness because
it is a measurement.

### 6.1 Principle: the desktop user is a holder, not an exception

The fork already models "someone else has a claim on this resource" three
times over, and none of it needs a new concept:

* `TierCapacity` (`memtier/tiers.py:345-368`) carries `reserved` — *"Bytes
  declared by other holders, read from the cross-process ledger"* — and
  `corridor`, #330's absolutely-free rule. **The desktop user is another
  holder.** Declared capacity of a remote tier is
  `total - reserved_user - corridor`, and the registry already refuses on the
  capacity axis when bytes are requested (`registry.py:455`, `:458`).
* `DemandSource` / `IdleMonitor` (`training/tenant.py:65-77`, `:193-224`) is
  a generic callable contract with a grace period, combining any number of
  demand signals into one verdict. A rig-2 desktop demand source (seat/session
  activity, load average, `MemAvailable` trend) is one more implementation of
  an interface that already exists.
* `workbench/arb.py` (header) is the cross-session protocol: availability is
  published, not negotiated; holders heartbeat; and *"before every access,
  regardless of what any file says, the hardware is checked — the files are
  intentions and can go stale, the hardware cannot."*

### 6.2 The reservation, concretely

| knob | meaning | default |
|---|---|---|
| `--remote-lane-host` | the rig-2 endpoint; absent = the whole lane does not exist | unset |
| `--remote-lane-reserve-ram-mib` | RAM the lane may never touch | **mandatory** when the host is set |
| `--remote-lane-reserve-cpus` | cores left to the desktop | **mandatory** when the host is set |
| `--remote-lane-cpu-quota` | cgroup `cpu.max` for the lane process | derived from the two above |
| `--remote-lane-yield-on-demand` | drain the lane when the desktop demand source fires | on |

The mutual-requirement shape (setting one without the other is a hard error,
not a defaulted fallback) is deliberately the same as `--rank-gpu-id` /
`--rank-gpu-memory-mib`, so operators meet one rule, not two.

**Enforcement is on both sides, and the remote side is authoritative.** The
local planner never plans beyond declared capacity; the remote worker
additionally caps itself with cgroup v2 (`memory.max`, `cpu.max`) so that a
planner bug cannot page out the user's session. A cap that only the requester
honours is not a cap.

**Refusal, never overbooking.** A request that does not fit refuses by name:
physical host, reserved bytes, requested bytes, the resulting share `s`, and
the predicted yield at that `s`. This is the same refusal shape the
memory-validation rules already use elsewhere, and it means a too-small
reservation shows up as a boot-time message rather than as a desktop that
starts swapping at 21:00.

### 6.3 Yield semantics — per request, not per token

An expert lane cannot be paused mid-token: the combine needs its partial.
So the yield contract is:

* a demand signal **disables new admissions** immediately;
* the lane **drains at the end of the current request**;
* the local fetch path is always present, so dropping the lane costs latency,
  never correctness — which is what makes an abrupt yield safe.

This differs from the workbench's between-segment preemption
(`workbench/scheduler.py` header) on purpose: a serving lane's segment is one
request, not one training step.

### 6.4 The cheap, generic thing worth doing regardless

Independently of all three axes: **register rig-2's RAM (and, if ever
relevant, its disk) as declared #407 tiers with headroom-reduced capacity and
measured rates**, per the #407 doctrine that every reachable memory is a
registered tier with measured numbers. Two consequences, both free:

* Consumers that are latency-tolerant can *choose* it — hibernate image
  staging (#89), the #306 compression workbench, #224 session park — and there
  the comparison is against **local NVMe at 1.8 GB/s**, which remote RAM at
  2.83 GB/s beats on rate while losing on capacity. That is a real trade, and
  "every usage form is offerable" is the standing rule.
* Consumers that must not use it (anything on a token's critical path) get a
  refusal grounded in a measured number instead of an argument.

Effort S. Touches no MoE code. This is the recommendation this ticket actually
produces.

---

## 7. Where this supersedes NOTE_453

| NOTE_453 | this document | why |
|---|---|---|
| §1: remote RAM is not bandwidth-competitive but composes "as a capacity tier for the coldest experts" | the capacity motive is **also** empty (§3.3) | the cold set fits local RAM (88 of 98.5 GiB), and for a model that overflows, T3 NVMe holds 80x more at 0.64x the rate |
| §2.1: one afternoon decides the compute lane by comparing one expert's round trip against streaming that expert's weights | that comparison is the **wrong falsifier** (§4.3) | the round trip vs one weight stream was never in doubt — 1:1770 (§4.1). What decides the lane is `s`, the share of the cold set remote RAM can hold, and that is answered by `free -g` plus arithmetic, not by a timing probe |
| §3: the 2080 Ti arm is format-gated and stays gated | confirmed, and closed for three further reasons (§5) | the gate is now quoted at its source; #212's 93.5 % compute attribution closes it independently of format |
| §4: any CPU-lane use requires a configurable headroom reservation | confirmed, and made concrete (§6) | mapped onto `TierCapacity.reserved` / `IdleMonitor` / `workbench/arb.py` rather than invented |

---

## 8. Fixposten before any window, and the phase cuts

Machbarkeit-vor-Messung: no window is proposed without stating what it costs
and what it decides. **All of them are additionally gated on the rig-2 lock
being lifted (§3.1).**

### 8.1 Cut 0 — desk, read-only inventory (cost: minutes, no GPU, no build)

**Reads** rig-2's `/proc/meminfo`, `lscpu`, `dmidecode -t 17` (DDR channels
and speed), `lspci -vv` negotiated widths for the NIC and the 2080 Ti, and
resolves the DESIGN_407-vs-inventory conflict about which line is x4 (§1.5).

**Decides:** `s_max = usable_remote_RAM / cold_set`. This single number
settles axis (b) without any measurement, because §4.3's table is monotone in
`s`.

**Falsifier:** RAM < 24 GiB (i.e. `s_max < 0.25` at a 6 GiB reserve) → axis
(b) stays parked, Cuts 1-4 are not run, and the register gets the parking
entry. On the recorded 15 GiB this cut is expected to close the axis.

### 8.2 Cut 1 — transport and throughput proof, no model (cost: ~1 h, no GPU allocation on rig 1, no model load)

The cheap proof the briefing asks for as Cut 1, and it is worth running even
if Cut 0 parks axis (b), because **#423 needs the same setup** and the numbers
feed the #407 tier rows either way.

* **P1 — round-trip distribution.** 16 KiB payload, rig-1 host <-> rig-2 host,
  over the transport a lane would actually use (UCX host plane; NCCL sockets
  as the control arm — verbs is broken on this line, #201). Report p50/p99/max
  **twice**: idle desktop and a deliberately busy desktop. Also run the
  three-caller variant, because #278 V5 says one NIC serialises.
* **P2 — rig-2 host rates under the reservation.** Sustained DRAM read
  (`dd`-class, direct, reproduced 3x per the #389 discipline) and one
  quantized-GEMM microbench, both inside the cgroup caps of §6.2 rather than
  on a free-running box.

**Decides:** two of the four terms in §4.2 (transport, remote DRAM), and it
converts two ESTIMATE rows in the #407 table into MEASURED ones.

**Falsifiers, stated so the probe can fail:**

* p99 round trip (busy desktop, three callers) **> 1.0 ms** — against the
  1.22 ms per-layer budget — → the lane cannot hide its crossing under
  realistic conditions → axis (b) closed by measurement, not parked.
* rig-2 sustained DRAM read **< 12 GB/s** → at s = 0.35 the remote serve
  (2.84 MB/layer at that share) no longer fits the budget → axis (b) closed
  regardless of a future RAM upgrade.
* Both green → the axis is *parked on capacity only*, and revisit predicate
  §4.5 is the whole gate.

### 8.3 Cut 2 — trace replay against the real crossing (cost: ~2 h, no GPU, no model rebuild)

Only if Cut 0 and Cut 1 are both green. Drive a recorded #390 routing trace
(the `expert_stats` dumps already carry per-layer activation histograms)
through a stub lane that performs the **real** crossing and a **real** remote
GEMM over the delegated share, and measure the added per-layer latency
distribution against the 1.22 ms budget.

**Decides:** whether the overlap assumption of §4.2(i) survives contact with a
real dispatch (it is the one modelled step in the chain).

**Falsifier:** added p99 per-layer latency > 0.6 x budget → the lane is a
latency risk at bs=1 and the axis closes.

### 8.4 Cut 3 — economics gate (desk, free)

Recompute §4.3 with the measured `s`, RTT and `R_rig2`. **Require a predicted
end-to-end >= 2x the current A-vs-A floor (>= 8.2 %) before any build is
authorised.** Below that, the build cannot be validated even if it works,
which is the trap `#493` (a bound that never binds) in another form.

### 8.5 Cut 4 — build (L), only past Cut 3

A native remote expert lane: rig-2 htsglang worker holding its expert shard in
host RAM, a per-layer crossing issued at layer entry and awaited at the
combine, headroom enforcement per §6, determinism gate per §4.4, and the local
fetch retained as the always-available fallback. Not designed further here on
purpose: three cuts have to pass first, and Cut 0 is expected to stop at the
first.

---

## 9. Recommendation, effort against yield

| axis | effort | yield at the reachable operating point | recommendation |
|---|---|---|---|
| (a) remote RAM cold tier | M (new transport) + L (tier integration) | **negative** (+2.8 % decode at s = 0.1) | **REJECT.** Fold the DDR-contention residual into #423's probe as one extra arm |
| (b) remote CPU lane | L (execution lane, determinism gate, remote runtime) | +2.2 %, below a 4.09 % floor | **PARK** behind §4.5's predicate. Spend only Cut 0 (minutes) and, if #423 runs anyway, Cut 1 |
| (c) 2080 Ti arm | — | none reachable | **REJECT**, register entry drafted in §5 |
| (d) rig-2 as declared #407 tiers for latency-tolerant consumers | **S** | choice, plus a grounded refusal for everything else | **DO THIS ONE** |

Proposed `planner/rejected.py` parking entry for axis (b), alongside §5's
entry for axis (c):

```
key="remote_cpu_expert_lane_rig2"
what="computing cold DSV4-Flash experts on rig-2's CPU over the 40G link"
verdict="PARKED, not blocked: the activation-payload inversion is real
    (16 KiB/layer vs 1220 MB/token of weights, 1:1770) and the round trip hides
    inside the 1.22 ms per-layer local budget. The yield is capped by how much
    of the 88 GiB cold set rig-2's RAM can hold: 15 GiB installed -> s ~ 0.10 ->
    2.2 % end-to-end against a 4.09 % A-vs-A floor. Revisit at >= 48 GiB remote
    RAM, or on a vehicle whose cold set is <= 3x the usable remote RAM, or if the
    NORDSTERN RDMA transport lands anyway."
level=NOT_WORTH_IT
evidence="DESIGN_453_dsv4f_remote_placement.md §4; expert_stats of
    /spinning/gpu-battery-results/2026-08-03_439_green/; #201 40G figures"
tags=("moe", "cross-rig", "cpu-lane")
```

---

## 10. Open risks

* **Rig-2 facts are recorded, not re-verified.** The lock (§3.1) forbids
  reading them this session. Every rig-2 number here carries that label, and
  Cut 0 exists to replace it. If the box has been upgraded since 2026-07-25,
  §4.5 predicate 1 may already be satisfied and axis (b) reopens — that is
  precisely why Cut 0 is minutes and not an assumption.
* **`R_rig2` (remote DRAM) is an assumption in a document that otherwise runs
  on measurements.** It is used in exactly two places (§4.2(ii), §4.3's
  ceiling), both flagged, and it never carries a verdict on its own: axis (b)
  is parked on capacity, which is measured arithmetic.
* **The exposure coefficient f = 0.22 is derived from two windows of the same
  lever**, not measured directly. It is the best available and it is
  consistent across the two (21.3 % / 23.1 %), but a lever with a different
  overlap profile would not inherit it.
* **The 100G-vs-40G provenance conflict** (§1.5) is unresolved. It cannot flip
  any verdict here (both candidates are below the slowest local link), but it
  should not stay in the #407 table unresolved.
* **`planner/rejected.py:263-281` is stale** against the code (§2 item 3), and
  a stale register entry is exactly the input that produces a wrong
  feasibility answer later. Correction owed in the next change touching that
  file.
