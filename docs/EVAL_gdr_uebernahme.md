# Adoption evaluation — dmabuf GPU-RDMA (`dmabuf_rdma`)

Input: the formal handover `/spinning/gdr-uebergabe/` (README, `GDR_BEFUNDE.md`,
`BUILD.md`, `gpurdma_0{1..4}.c`, `read_dmabuf_flag.sh`), 2026-07-28.
Base commit: `a2c8f76c42` (`integration/r3-probe-next2`).
Desk work only — no GPU was touched for this document. Everything below is
either a quote from an existing measurement, arithmetic on such a quote, or a
named absence.

Naming follows the handover: the capability is called **dmabuf GPU-RDMA**, the
feature would be `dmabuf_rdma` / `--gpu-rdma`. No vendor marketing name is used
as a feature name.

**Provenance tags used throughout**

| tag | meaning |
|---|---|
| `[M]` | measured on real hardware, source named |
| `[D]` | derived by arithmetic from `[M]` values in this document; the arithmetic is shown |
| `[E]` | estimate / extrapolation, flagged as such by its own source |
| `[A]` | absent — the number does not exist anywhere and is needed |

---

## 0. Verdict in one paragraph

The handover proves a real capability on consumer cards and that part is not in
doubt. The *adoption* case, however, does not survive contact with our own
measurements. The headline "39-45 % intra-rig" compares two RDMA arms against
each other, not against the path our stack actually runs intra-rig (NCCL SHM),
and #199 already shows NCCL sitting at its own back-to-back floor there
[M, §2.2]. Cross-rig, our entire collective traffic sits on the *losing* side of
the handover's own crossover — at 64 KiB the direct path is 2.9x slower [M,
§1.5] and our verify collective is 80 KiB [M, §3.1]. The "86 % staging" figure
that motivated reopening cross-rig TP is a derived residual, and our own
instrumented phase split puts the actually-staging part at 12-17 % at
verify-scale payloads [M, §3.2], not 86 %. Therefore: **do not build a
`dmabuf_rdma` transport now.** Build the two cheap things that make the question
answerable and rig-portable — a capability gate row (#214) and a comm-suite arm
that measures the crossover per rig — and run one 30-minute falsification that
could flip the large-message verdict, because the one card on this rig with
Resizable BAR was never tested as the RDMA target [§1.4].

---

## 1. Reading the handover correctly

### 1.1 What the benchmark actually is

`gpurdma_04_bench.c` is a two-process RC-queue-pair ping-pong. Per iteration the
**client** posts one RDMA-write of the payload plus one 8-byte flag write; the
**server** polls the flag and answers with an 8-byte flag write. Reported latency
is `(t1 - t0) / 2`, i.e. half the full round trip, and the payload crosses the
link exactly **once** per full round trip (`gpurdma_04_bench.c:369-424`).

Two arms:

- `gdr` — the payload MR is a VMM allocation exported through the dmabuf chain
  (`gpu_dmabuf()`, `gpurdma_04_bench.c:167-247`), registered with
  `ibv_reg_dmabuf_mr(pd, 0, MAXSZ, 0, dfd, ...)` at `:287`. Remote addressing is
  `iova 0` (`:291`), i.e. offset-based, not a virtual address.
- `stage` — the payload MR is host memory from `posix_memalign` (`:295-297`) and
  the GPU buffer is a plain `cuMemAlloc` (`:309`), with `cuMemcpyDtoH` on the
  client before the send (`:371-373`) and `cuMemcpyHtoD` on the server after the
  flag (`:408-410`).

The synchronisation flag is in host memory in **both** arms (`:316-321`), which
the handover correctly calls out as the fairness condition.

Consequence that matters for adoption: **`stage` is the staging shape of *this
benchmark*, not the staging shape of our engine.** It is one blocking
`cuMemcpyDtoH` plus one blocking `cuMemcpyHtoD` per message, on a buffer that
CUDA sees as pageable. Neither NCCL's SHM transport nor our barlink transports
work that way.

### 1.2 Correction A — "intra-rig" means NIC loopback

In §6.1 of the handover both processes run on rig 1, so the traffic goes
GPU A -> ConnectX-4 -> GPU B. There is no wire, but there are still two PCIe
crossings and a full NIC round trip. This is genuinely interesting on a rig with
no GPU-GPU P2P, because it makes the NIC a P2P relay.

The PCIe topology confirms the relay is not confined to one switch leg
(`lspci -tv`, read-only, this document) `[M]`:

```
00:01.2 -[02-09]-- 03:01.0 -[04] ConnectX-4 (2 ports)
                   03:02.0 -[05] RTX 3080
00:03.1 -[0a] RTX 5090
00:03.2 -[0b] RTX 3080
```

Only **one** of the three GPUs shares a PCIe switch with the NIC. The 5090 and
the second 3080 reach the NIC through the root complex — and §6.1 measures the
5090 working as an RDMA source anyway (4.84 us at 8 B). That is a **counter-datum
to the handover's own topology hypothesis** for the 2080 Ti's
`local protection error` (§7 of the handover): on this rig, a GPU that does *not*
share the NIC's switch works fine as a source. The 2080 Ti failure therefore
needs a different or additional explanation, and "same switch as the NIC" should
not be written into any gate row as a precondition without more evidence.

### 1.3 Correction B — the "24 %" is the 8-byte row

The handover's §9 says *"kleine Kollektive sind die Zielgruppe … dort betraegt
der Gewinn ueber die Rechnergrenze rund 24 Prozent"*. That number is exactly the
8-byte row of §6.2: `4.99 -> 3.77 us` = -24.4 % `[D]`. The very next column
already reverses sign: at 4 KiB direct is 6.08 vs 5.31 staged, i.e. **14.5 %
worse** `[D]`.

So "small collectives" in the handover's sense means *messages of a few hundred
bytes*, not "collectives whose payload is small compared to a prefill chunk".
Section 3 below shows none of our collectives are in that class.

### 1.4 Correction C — the BAR-write explanation does not survive both tables

The handover attributes the large-message cross-rig loss to BAR write bandwidth
(§6.2: direct 0.83 GB/s vs staged 2.8 GB/s). Applying the handover's own
bandwidth convention (one payload per full round trip) to §6.1 `[D]`:

| case | half-RT @1 MiB `[M]` | full RT `[D]` | effective B/W `[D]` |
|---|---|---|---|
| intra-rig 3080↔3080, direct | 166.74 us | 333.5 us | **3.14 GB/s** |
| intra-rig 5090↔3080, direct | 252.75 us | 505.5 us | 2.07 GB/s |
| cross-rig into a 3080, direct | 634.13 us | 1268.3 us | **0.83 GB/s** |
| cross-rig into host, staged | 185.57 us | 371.1 us | 2.83 GB/s |

The same NIC writing into the same class of 256-MiB-BAR 3080 achieves 3.14 GB/s
locally and 0.83 GB/s remotely — a factor of 3.8. Even fully serialising the
wire (1 MiB at the measured 28.64 Gb/s = 293 us) with a 334 us BAR write only
accounts for 627 us of the observed 1268 us `[D]`. **"The BAR window is the
wall" is not established by these two tables.** Something in the remote-write
path — RoCE arrival granularity, posted-write ordering into uncached MMIO, the
sender rig, the 40G leg — carries the remaining ~640 us, and none of it has been
isolated `[A]`.

Two consequences:

1. Per the rig-is-a-lower-bound rule, the large-message negative must be
   published as a **per-rig measurable**, not as a property of GPU RDMA.
2. **The one card on this rig with Resizable BAR was never used as the target.**
   `lspci -v` `[M]`: BAR1 is 256 MiB on both 3080s (`05:00.0`, `0b:00.0`) and
   **32 GiB on the 5090** (`0a:00.0`). The entire cross-rig table §6.2 was taken
   against a 3080. If the BAR story is right at all, repeating §6.2 with the 5090
   as the rig-1 target is the single cheapest experiment that could move the
   crossover by orders of magnitude. It needs no code: the existing binary plus
   `<ord>` of the 5090.

### 1.5 The numbers, with provenance

Direct from `/spinning/gdr-uebergabe/GDR_BEFUNDE.md`, 2000 iterations after 200
warm-ups, median, A-vs-A noise floor under 0.1 % `[M]`:

| pair / direction | 8 B | 4 KiB | 64 KiB | 1 MiB |
|---|---|---|---|---|
| intra 3080↔3080 direct vs stage `[M:176-179]` | -42.8 % | -40.9 % | -40.3 % | -39.0 % |
| intra 5090↔3080 direct vs stage `[M:180-181]` | -45.2 % | -43.7 % | **-23.5 %** | **-15.5 %** |
| cross-rig direct vs stage `[M:190-195]` | **-24.4 %** | +14.5 % | **+188.6 %** | **+241.7 %** |

Negative = direct wins. Percentages `[D]` from the quoted microsecond values.

Note the "39-45 % across all sizes" summary holds only for the 3080↔3080 pair.
The **mixed** pair — which is the pair our TP=3 actually uses, since every rank
pair on this rig includes the 5090 or crosses it — falls to 23.5 % at 64 KiB and
15.5 % at 1 MiB `[D]`.

Concurrency, handover §6.3, two pairs running at once `[M:214-218]`, ratio `[D]`:

| | 8 B | 4 KiB | 64 KiB | 1 MiB |
|---|---|---|---|---|
| 5090↔3080 direct, under load / alone | **2.40x** | **2.60x** | 1.56x | 1.26x |

This is the datum that decides Target 1 (see §2.3): the small-message regime,
which is the only regime where the direct path is convincingly ahead, degrades
by 2.4-2.6x as soon as a second flow shares the NIC.

### 1.6 What the handover does not contain `[A]`

1. **MR registration cost.** The chain is `cuMemCreate` -> export -> two ioctls ->
   `ibv_reg_dmabuf_mr`. None of it is timed. For a static pool this is amortised
   to zero; for anything dynamic it is the whole design.
2. **Throughput under real load** — the handover says so itself (§8.4).
3. **Only four support points** (8 B, 4 KiB, 64 KiB, 1 MiB). The cross-rig
   crossover is bracketed to a factor of 512. Our smallest interesting collective
   sizes (20 KiB) fall in an unmeasured gap.
4. **No GPU-to-GPU across the rig boundary** (handover §8.2) — which is exactly
   the NORDSTERN shape.
5. **Allocation granularity is 2 MiB** (`gpurdma_02_register.c:90-92`, comment at
   `:31`) `[M]`. Any exported buffer is a multiple of that. Fine for a pool,
   fatal for per-message allocation.

---

## 2. Target 1 — intra-rig collectives

### 2.1 Vehicle (a): NCCL's own dmabuf path — dead here, for three independent reasons

1. **NCCL's dmabuf support calls `cuMemGetHandleForAddressRange`**, which the
   handover reports as unavailable on GeForce (`GDR_BEFUNDE.md:64-67, 88-91`)
   `[M]`. This is precisely the "convenience function" the handover routes
   around; NCCL has no such workaround.
2. **The legacy alternative is not installed.** `modinfo nvidia_peermem` ->
   *"Module not found"*, and no `nvidia_peermem` in `lsmod` (this document, host
   inspection, read-only) `[M]`. Without either peermem or dmabuf, NCCL cannot
   register GPU memory with the NIC at all.
3. **Even if it could, NCCL would not use it intra-node.** NCCL picks P2P or SHM
   between two GPUs on one host; the NET transport is an inter-node path. Driving
   intra-node traffic through the NIC requires forcing
   `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`, which is not a configuration anyone
   would ship blind.

`NCCL_DMABUF_ENABLE` does not appear anywhere in the repo, and neither does any
`NCCL_NET_GDR*` variable `[M]`.

**Answer to the vehicle question: NCCL's own dmabuf path is dead on this
hardware, and the intra-rig opportunity is structurally outside what NCCL would
choose to do anyway.** If it is to be had, barlink is the only vehicle.

There is exactly one cheap way to falsify point 3 before writing any code, and it
belongs in the GPU window (§9, item **W2**):

```
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET_GDR_LEVEL=SYS NCCL_DEBUG=INFO
```

and read the transport line. If NCCL announces `[send] via NET/IB/0/GDRDMA`, the
capability is reachable with zero lines of our code. Expected outcome given
points 1-2: it announces a host-buffered NET path or fails to find a net device.

### 2.2 Vehicle (b): barlink — the seam is ready, and it is not where the briefing assumed

The transport seam is real and documented as an extension point
(`python/sglang/srt/distributed/device_communicators/barlink.py:67-80`,
*"Adding a transport … is one registry entry plus its module; no dispatch site
changes"*). Concretely a `dmabuf_rdma` transport is:

| step | location |
|---|---|
| registry entry | `barlink.py:112-120` `TRANSPORT_REGISTRY` |
| factory | beside `_make_ucx_transport`, `barlink.py:103-108` |
| module | new file next to `barlink_ucx.py`, exposing `handles(op, nbytes)` + the four collectives |
| size dispatch | `BarlinkCommunicator._select(op, nbytes)`, `barlink.py:173-183` — the *only* dispatch shape; per-transport predicate `handles()` |
| graph capture | add to `CAPTURABLE_BARLINK_TRANSPORTS`, `parallel_state.py:232`, only if there is no host sync; otherwise `_enforce_cpu_transport_needs_eager` (`parallel_state.py:235-258`) correctly forces eager |
| GPU-side reduce | template exists: `barlink_add_kernel` in `barlink_device.py` `_CUDA_SRC`, and `flat.add_(peer_dev)` at `barlink_shm.py:227` |
| what disappears | `_slot`/`_staging` (`barlink_ucx.py:681-726`), `_staged_copy`/`_staged_add` (`:823-859`), `_h2d_async`/`_slot_guard` (`:764-810`) |

**Correction to the briefing's premise:** #240's `--collective-net-small` /
`--collective-net-bulk` are **not** a size switch. They are NIC pinning
(`UCX_NET_DEVICES`), resolved in `ServerArgs._handle_collective_net_env()` and
consumed as `ucp_config_modify(config, b"NET_DEVICES", ...)`. The runbook
(#240, §4.3.1) states outright that small and bulk TP collectives are **not
separable** today — one `UcpWorker`, one UCX context, one endpoint per peer
carries both — and specifies what a genuine per-class split needs: a second
worker, a second address exchange at rendezvous, a size-keyed selector, and a
guarantee that all ranks pick the same worker for the same collective, ~200 LoC
plus a cross-rig validation pass. The size hook for `dmabuf_rdma` is therefore
`_select` + `handles`, and `barlink_shm.py:171-173` is the one existing
size-conditioned `handles` to copy from.

**Hazard to design around, stated in-tree** (`barlink_ucx.py:144-152`): a
size-keyed `handles` is only safe when the payload size is rank-uniform, or ranks
diverge and the group deadlocks instead of returning a wrong answer. `all_reduce`
and `broadcast` are uniform; `all_gather`/`reduce_scatter` under uneven TP are
padded to equal shapes first and are uniform only *after* the pad.

Also to be overturned, not merely edited: the design rationale at
`barlink_ucx.py:14-19` — *"There is deliberately no GPUDirect: this hardware has no
P2P between the NIC and the GPUs … staging through the host is not a
simplification, it is the only path that exists."* The handover falsifies the
premise of that sentence. It is also the natural anchor for a future
`dmabuf_rdma` docstring.

One in-tree precedent worth knowing: `flashinfer_comm_fusion.py:232` already sets
`prop.allocFlags.gpuDirectRDMACapable = 1` on a VMM allocation — the only place
the repo already asks the driver for RDMA-exportable memory.

### 2.3 Expected value, honestly — the intra-rig case does not hold

Three measurements, none of them from the handover, decide this.

**(i) NCCL is already at its floor intra-rig.** Task #199, TP=3 uneven DCP,
torch profiler over 40 decode steps, all three ranks
(`INTEGRATION_R3_VALIDATION.md:4300-4319`) `[M]`: pure comm is 252.2 ms = **15.8 %**
of the window, and *"NCCL latency floor on this rig … 10 KiB/40 KiB all-reduce =
55-58 us isolated, **31-37 us back-to-back**. Measured in-server pure comm is
27.7 us per AR bf16 — NCCL is already at its floor; there is no slack in the
transport itself."* Whatever a new transport does, it competes against 27.7 us
for a 3-rank all-reduce, not against a strawman.

**(ii) A NIC-relayed all-reduce needs many more NIC round trips than one.** A
3-rank ring all-reduce is `2(W-1) = 4` serially dependent point-to-point steps.
Using the handover's own direct point-to-point costs (4 KiB: 5.33 us on the mixed
pair `[M]`), four steps land at ~21 us **before** the reduce kernels, the flag
traffic per step, and the rendezvous `[D]`. That is at best a wash against
NCCL's 27.7 us, and the estimate is generous because it ignores everything the
transport has to do besides move bytes.

**(iii) Concurrency, which a collective *is*.** Handover §6.3: two concurrent
pairs inflate small-message latency by **2.40-2.60x** `[D from M]`. A TP=3
collective is at minimum two concurrent flows through one ConnectX-4 port, and
under dual-group runtime (#274) more. The regime where the direct path wins is
exactly the regime that degrades worst under contention.

**(iv) The 68 % is not transport slack.** #252
(`INTEGRATION_R3_VALIDATION.md:4724-4735`) `[M]` measures *wait inside
collectives* at ~68 % of the prefill window on every rank of the TP=3 FP8 boot
(rank0 1837.6 gpu-ms = 196.6 compute + 1641.1 wait; ranks 1/2 586.5/558.5 compute
+ 1251.0/1279.3 wait), and its own reading attributes ~390 ms of the imbalance to
shard skew, i.e. genuine idling, not bytes in flight. Also worth recording: the
briefing's "68-75 %" has no 75 % measurement behind it — the only occurrence of
the range in the repo is a forward-looking framing sentence
(`INTEGRATION_R3_VALIDATION.md:5075`) `[A]`.

**Target 1 verdict: NOT the big lever.** The 39-45 % headline is measured against
a staging shape our stack does not run intra-rig; against the shape it does run
(NCCL SHM, already at its floor, with the reduce done by kernels rather than by
`cuMemcpy` pairs) the available evidence points to a wash or a loss, and the
contention data points to a loss. This should not be built on the strength of the
handover.

---

## 3. Target 2 — cross-rig small class: where our messages actually sit

### 3.1 Our message-size distribution against the crossover

Byte counts are defined from model geometry in
`scripts/r3val/link_collective_cost.py:38-41` (`DECODE_BYTES = 5120*4`,
`VERIFY_BYTES = 5120*4*4`) and the comm-suite ladder
`SIZES_KIB = (20, 80, 256)` (`python/sglang/srt/planner/comm_suite.py:98`) `[M]`.

Cross-rig crossover from the handover: **between 8 B and 4 KiB**; at 4 KiB the
direct path is already 14.5 % worse `[D]`.

| our message | bytes | count per round | side of the crossover | measured cost |
|---|---|---|---|---|
| spec broadcast `predict` (bs=1, k=3) | 32 B `[D]` | 1 / verify | **winning side** (-24 % at 8 B) | not separately timed `[A]` |
| spec broadcast `accept_index` | 32 B `[D]` | 1 / verify | winning side | `[A]` |
| spec broadcast `num_correct_drafts` | 8 B `[D]` | 1 / verify | winning side | `[A]` |
| barrier / rendezvous control | O(10-100 B) `[E]` | few / boot | winning side | 1 GbE barrier 146.63 us `[M]` |
| PP metadata (`send_tensor_dict`) | **not recorded** `[A]` | 1 / crossing | unknown | 249 us/round `[M]`, but rides **gloo on the 1 GbE control plane**, not the RDMA line |
| decode `all_reduce` | **20480 B** `[M]` | 2 x L per step (128/step at #199) | **losing side** | 88.9 us world 4 `[M]` |
| verify `all_reduce` (4-token MTP) | **81920 B** `[M]` | 2 x L per verify | **losing side**, 64 KiB is 2.89x worse `[D]` | 202.1 us world 4 `[M]` |
| verify `all_gather` | 81920 B `[M]` | 1 / verify | losing side | 301.2 us `[M]` |
| logits all_gather | `tokens x vocab/W x 4` | 1 / forward | losing side | included above |
| prefill chunk `all_reduce` | 2048 x 5120 x 2 = **20 MiB** `[D]` | 2 x L per chunk | far losing side (3.4x at 1 MiB) | — |

**Answer to "where does our distribution sit relative to the crossover":
essentially all of it sits on the losing side.** By bytes, the fraction below
4 KiB is under 0.01 %; by message count, the sub-KiB spec broadcasts are ~3 of
~130 collectives per verify `[D]`. The absolute saving on those three, at the
handover's own 8-byte delta of 1.22 us, is **~3.7 us against a measured 166.16 ms
cross-rig verify round** `[D from M]` — 0.002 %.

The one item that looks like a small-message win and is not: **PP metadata,
249 us/round, 64 % of the crossing at bs=1** (`docs/rig-runbook.md:860-877`)
`[M]`. It is a pickled `send_tensor_dict` payload on **gloo over 1 GbE**. The
recorded cheapest fix is caching it (static shapes per batch size); moving it to
the 40G link would be the second. Neither has anything to do with dmabuf, and
its byte size is not recorded anywhere `[A]`.

### 3.2 The "86 % staging" claim, corrected

`FEATURES_VS_UPSTREAM.md:548-556` `[M]` derives it: the 4-token verify
`all_reduce` is ~202 us at world 4, the raw link does 80 KiB in 28.1 us
(`ucx_perftest tag_lat`), so *"the wire is ~14 % of that verify collective and
the other ~86 % is per-rank host work **and lock-step turnaround**"*. The
register-form restatement (`INTEGRATION_R3_VALIDATION.md:4950-4953`) compresses
that to *"host staging carries ~86 % of the cost"* — the compression is where
"staging" enters, and it drops the lock-step half.

Our own instrumented phase split says how much is really staging `[M]`:

| payload | total | stage | finish | stage+finish share `[D]` |
|---|---|---|---|---|
| 8 KiB (`FEATURES_VS_UPSTREAM.md:581`) | ~27 us | 4.5 us | 4.2 us | **32 %** (post 8.1 + wait 8.3 are the ctypes+UCX+RTT floor) |
| 128 KiB (`:451`) | 270.6 us | 20.6 us | 12.8 us | **12.3 %** |
| 256 KiB (`:452`) | 524.1 us | 52.2 us | 35.0 us | **16.6 %** |

There is no stage/finish row at the 80 KiB verify size `[A]`, but it is bracketed
by the two above: **~12-17 %, not 86 %.** The remaining ~85 % is UCX progress,
`2(W-1)` serially dependent ring hops and single-core lock-step turnaround —
none of which a dmabuf transport removes.

So the cross-rig arithmetic for a `dmabuf_rdma` transport at 80 KiB is: delete
~25-35 us of staging, and pay the direct-path wire penalty, which at the nearest
measured point (64 KiB) is **+29.3 us per point-to-point hop** (44.84 vs 15.50 us
half-RT) `[D]` — multiplied by `2(W-1)` hops. **Net clearly negative.**

**Target 2 verdict: no cross-rig class of ours is on the winning side, and the
share the direct path could address is 12-17 %, not 86 %.** The recoverable
cross-rig cost lives in the transport implementation (progress thread, hop
count, lock-step), not in where the bytes are staged. The reopening of
"cross-rig TP druecken" on GDR grounds is, on this evidence, **not justified at
the verify size** — it should be narrowed to the ReBAR question in §1.4.

---

## 4. Target 3 — VMM integration

The precondition is a `cuMemCreate` buffer with
`requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`. Three routes
were named in the brief; the fork's actual state changes the answer.

### 4.1 Route (a) — global `expandable_segments`: **reject**

Two independent blockers, one of them already codified in our own code.

1. **Hard conflict with #93/#102 and #89.**
   `speculative/adaptive_graph_memory.py:338-342` already refuses it:

   ```python
   if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
       return _fail_or_resident(
           "PYTORCH_CUDA_ALLOC_CONF expandable_segments is incompatible with "
           "torch_memory_saver")
   ```

   `auto` silently degrades to `resident` — the entire VRAM saving of #93 is lost
   — and an explicit `--speculative-adaptive-graph-memory offload` refuses to
   boot. Ratcheted by `test_auto_degrades_on_expandable_segments`. The same
   library backs `--enable-memory-saver` (#89), which is not even guarded and
   would fail at runtime. Note this resolution happens in the *launcher* process,
   at scheduler spawn: it is not flippable per request.

   To answer the brief's question directly: **#93 is real VMM, not pointer
   swapping** — `torch_memory_saver` `pause(tag)` unmaps physical pages and
   `resume(tag)` re-creates them via `cu_mem_create` at the *same* virtual
   addresses, which is what lets captured graphs survive (swap cost 40-51 ms avg,
   85 ms max `[M]`). It is the same API family, and that is exactly why the two
   cannot share the default allocator. **Conflict, not synergy.**

2. **PyTorch's expandable segments are not exportable by default anyway.** The
   installed `libc10_cuda.so` (torch 2.11.0+cu128) references `cuMemCreate`,
   `cuMemExportToShareableHandle` and `cuMemImportFromShareableHandle`, but also
   carries the string `TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC` and a function-local
   static `enable_ipc_handles` inside `ExpandableSegment::map`, plus the message
   *"Tensors allocated with expandable_segments:True cannot be shared between
   processes"* (binary inspection, this document) `[M]`. The POSIX-FD handle type
   is requested only when that opt-in is on. And even then there is no public API
   mapping a tensor's `data_ptr()` to its segment handle — you would need a C++
   extension reaching into the allocator.

### 4.2 Route (b) — a dedicated VMM pool for collective staging buffers: **cheap, and the right first step if anything is built**

Small, bounded, no allocator-wide blast radius, no interaction with graph
capture, and the 2 MiB granularity is irrelevant for a pool. The buffers it would
replace are named and localised: `barlink_ucx.py:681-722` (`_slot`, pinned
`torch.empty(..., pin_memory=True)`), keyed per `(key, numel, dtype)` and cached
for the transport's lifetime — i.e. **already a static pool**, which means MR
registration cost is paid once at rendezvous and amortised to zero.

Cost estimate: the VMM allocation + export helper is ~150 lines of ctypes/C on
top of what `vmm_utils.py` already does (§4.3), plus the transport module itself.

### 4.3 Route (c) — VMM-allocated KV pools: **mostly already built, opt-in, off by default**

This is the finding that changes the shape of the recommendation.

| piece | location | state |
|---|---|---|
| VMM arena for KV (`cuMemAddressReserve` / `cuMemCreate` / `cuMemMap` / `cuMemSetAccess`, 256 GiB VA reserve, `CUDAPluggableAllocator` + `torch.cuda.MemPool(no_split=True)`) | `python/sglang/srt/mem_cache/kv_vmm_backing.py:106-248` | **built**, gated by `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` (`environ.py:310`, default **False**) |
| export to a shareable POSIX fd | `distributed/device_communicators/vmm_utils.py:127-170` — `cuMemExportToShareableHandle(..., FABRIC, 0)` with **POSIX_FD fallback** | **built** |
| "is this pointer VMM?" runtime branch | `vmm_utils.is_vmm_pointer()` :41, consumed at `custom_all_reduce_v2.py:118-126` | **built**, working precedent |
| what is missing for dmabuf | `KvVmmArena._prop` (`kv_vmm_backing.py:116-119`) does not set `requestedHandleTypes` | ~2 lines |

So the "biggest open item" of the handover is, in this fork, roughly a two-line
property change plus reuse of an existing export path. What remains genuinely
expensive is everything *after* the export: the transport, the rendezvous, the
size dispatch, the validation.

Constraints that must be honoured if route (c) is ever used:

- Registration must happen **after** `finalize_backing()`
  (`memory_pool.py:1811`). `kv_vmm_backing.ensure_prefix()` deliberately backs
  only a prefix before capture; registering earlier hands the NIC unbacked VA.
- Every RDMA consumer registers KV as flat `(ptr, len)` extents via
  `get_contiguous_buf_infos()` (`memory_pool.py:1851-1860`, and
  `disaggregation/mooncake/conn.py:244-269`). A VMM arena satisfies this;
  `expandable_segments` in the default pool would not, per §4.1.
- `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` excludes MLA, `dcp_size != 1`,
  `enable_memory_saver`, mooncake custom pool and DP attention
  (`server_args.py:4584-4593`) — i.e. it is currently incompatible with our own
  uneven-DCP default.

### 4.4 Recommendation

**(b) now if anything, (c) later and only for a KV-transfer use case, (a)
never.**

Reasoning: (a) costs #93/#102 and #89 outright and buys nothing that (b)/(c)
don't give — the fork's own VMM allocators already produce `cuMemCreate`-backed
pointers, and `vmm_utils` already exports them. (b) is bounded and reversible.
(c) is the only route that opens the "lendable segment + RDMA export from one
mechanism" synergy the brief asks about, but it is gated behind a flag that is
mutually exclusive with uneven DCP today, so it cannot be the entry point.

---

## 5. Target 4 — the negative list, and why

These paths stay on host staging. Recorded here so nobody converts them wholesale
on the strength of the "39-45 %" headline.

| path | code | typical size | why it stays |
|---|---|---|---|
| Weight loading | `model_loader/*` | GB, once at boot | not latency-sensitive; orders of magnitude past the crossover |
| HiCache L2 writeback / prefetch | `hiradix_cache.py:789-865`, `pool_host/*` | tens of GB pool | **the destination is host RAM**; there is nothing to make direct |
| KV session offload / spill (#224) | `managers/kv_session_offload.py` | ~12 KiB/token; 32k ctx ≈ 400 MB | destination is host RAM by definition |
| Expert offload (#77) | `layers/moe/expert_offload.py` | MB per wave | source is host RAM; `device_view_of_pinned()` (:371) is already the zero-copy answer in that direction |
| PD KV transfer (mooncake / nixl) | `disaggregation/mooncake/conn.py:479-538` | 64 MB staging ring | bulk chunks are ≥ 1 MiB, where the direct path is 3.4x worse cross-rig; already RDMA, already prefers a `cuMemCreate`-backed pool |
| Prefill chunk collectives | 2048 x 5120 x 2 | **20 MiB** | 20x past the largest measured point, on the losing side of every cross-rig row |

Rule to carry forward: **the destination decides.** If the other end is host RAM
(spill, HiCache, offload), dmabuf is definitionally irrelevant. If the other end
is GPU memory, size decides, and every bulk case we have is on the wrong side of
the measured crossover — subject to the ReBAR question in §1.4.

---

## 6. Target 5 — NORDSTERN and the #214 gate table

### 6.1 What changes for TP=5 cross-rig

Nothing improves on the current evidence. The cross-rig verify collective is
80 KiB `[M]`, which is 2.9x worse on the direct path at the nearest measured
point `[D]`, and TP=5 raises the hop count (`2(W-1)` = 8), amplifying the wire
penalty rather than the staging saving. The handover also records that
**GPU-to-GPU across the rig boundary was never demonstrated** (§8.2) — the far
rig has no card with a working RDMA source path (2080 Ti: reproducible
`local protection error`). NORDSTERN would need both ends direct.

What *does* change: the ladder's L0/L1/L2 framing gains a fourth question —
"is the receiving GPU's BAR resizable?" — because §1.4 shows the only ReBAR card
on this rig was never tested as the target. That question is cheap and belongs in
the rig profile, not in a transport.

### 6.2 The #214 capability row

`python/sglang/srt/planner/rig_coupling.py` — `GateRow` at `:385-405` with
`key, label, verdict, reason, remedy, local, remote, provenance, evidence,
register_key`; verdicts `ok`/`warn`/`block` (`:97-113`); provenance
`measured`/`estimate`/`absent`; sides `dashboard`/`pve-host`/`far-rig`. Rows are
assembled in `gate()` at `:771-801`. **Adding a capability is one `_row_*`
builder plus one append** — no schema change.

Proposed row `dmabuf_rdma`, with the preconditions each side must satisfy and
what this rig reports today (all read-only, this document) `[M]`:

| check | rig 1 (local) | how to read it |
|---|---|---|
| open kernel modules | **OK** — `NVIDIA UNIX Open Kernel Module … 595.58.03` | `/proc/driver/nvidia/version` contains "Open Kernel Module" |
| kernel ≥ 5.12 with mlx5 dmabuf | **OK** — 6.17.2-2-pve, `mlx5_ib` loaded | `uname -r`, `lsmod` |
| rdma-core with `ibv_reg_dmabuf_mr` | **OK** — libibverbs 50.0 | `dpkg -l libibverbs1` (available since rdma-core 34) |
| RDMA device present | **OK** — `rocep4s0f0`, `rocep4s0f1` | `/sys/class/infiniband` (already read by `_known_net_devices()` for #240) |
| per-card `dma_buf_supported` | **OK** on all three | `read_dmabuf_flag.sh` — but this needs `gdb` on `/proc/kcore` as root, so it is a **host-side probe, not an in-server check** |
| GPU works as RDMA **source** | **OK** rig 1 (all three cards) / **BLOCK** rig 2 (2080 Ti, `local protection error`) | functional probe only; not inferable from any flag |
| target BAR resizable | 5090 32 GiB **OK**; 3080s 256 MiB **WARN** | `lspci -v` BAR1 size |

Two rules the row must respect, both already in force in that file: a verdict
names its basis (`provenance` + `evidence`, or the row is a question and says
so), and `measured` may only come from real cross-rig wire evidence — loopback
UCX is explicitly rejected as wire evidence. On that rule, **every dmabuf row
today is `estimate` or `absent` except the four package/kernel checks**, because
the functional source probe has not been run inside the server.

The `MESSAGE_CLASSES` table in the same file (`:121-125`, keys `tp_small`,
`tp_bulk`, `kv_bulk`, `control`) is where a `dmabuf_rdma` carrier would later be
offered per class — and §3.1 says today it would be offered for `control` only.

---

## 7. Target 6 — how to couple it in

The NVIDIA open-kernel-module headers (`nv-ioctl.h`, `nv_escape.h`,
`ctrl0000unix.h`, …) are **deliberately not vendored**; they are meant to be
cloned at the exact installed driver version, because the ioctl struct layouts
must match the running driver (`BUILD.md:8-25`).

**Recommendation: optional module, headers fetched at build/run time, never
vendored.** Concretely, and idiomatic for this fork:

1. Do **not** copy the NVIDIA headers into the tree. Version drift against the
   installed driver is a correctness hazard (wrong ioctl struct layout is a
   silent memory bug), not just a licence question.
2. Follow the existing barlink pattern rather than inventing a build step: the
   `device` transport already JIT-compiles its native code at runtime with
   `torch.utils.cpp_extension.load_inline` (`barlink_device.py:822-875`) and needs
   no build system. A `dmabuf_rdma` helper can do the same, taking the header
   path from an env var (e.g. `SGLANG_DMABUF_RDMA_OGKM_INCLUDE`).
3. Absent headers, absent `-open` driver, or absent `ibv_reg_dmabuf_mr` -> the
   transport simply does not register, and the #214 row reports `absent` with the
   missing precondition named. No hard dependency is added to the default build.
4. Keep the reference C files as documentation of the chain (they are the only
   readable spec for steps 3-4), MIT header intact, under `3rdparty/` or beside
   the module — not compiled by default.

---

## 8. Open numbers and named falsifiers

| # | question | status | falsifier |
|---|---|---|---|
| O1 | Does the crossover move when the target has Resizable BAR? | `[A]` — §6.2 of the handover only ever targeted a 256-MiB-BAR 3080 | rerun `gpurdma_04_bench` cross-rig with the 5090 as rig-1 target. No code. **Highest information per minute of any item here.** |
| O2 | Why is direct 3.14 GB/s locally and 0.83 GB/s remotely into the same BAR class? | `[A]` — §1.4 shows the stated cause does not cover both tables | `ib_write_bw` with a dmabuf MR, loopback vs remote, same target card |
| O3 | Would NCCL take a GDR net path intra-node at all? | `[A]` | `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_DEBUG=INFO`, read the transport line |
| O4 | MR registration cost for the full chain | `[A]` | time `gpurdma_02_register` from `cuMemCreate` to `ibv_reg_dmabuf_mr` |
| O5 | Is a torch `expandable_segments` pointer exportable in practice? | `[A]`; binary inspection says only under `TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC` | 5-line script: allocate under `expandable_segments:True`, attempt the chain. Moot if route (a) stays rejected |
| O6 | stage/finish split at the 80 KiB verify size | `[A]` — bracketed 12.3-16.6 % by the 128/256 KiB rows | `link_collective_cost.py` already emits the phase split; add 80 KiB |
| O7 | PP metadata byte size | `[A]` | instrument `send_tensor_dict` |
| O8 | Does the 2080 Ti source failure really follow from PCIe topology? | `[A]`, and §1.2 supplies a counter-datum from rig 1 | move the card, or test another rig-2 card |

---

## 9. Priority recommendation

Ordered by information per unit of risk, not by ambition.

**P1 — `dmabuf_rdma` capability row in the #214 gate table.** Desk work plus a
host-side probe. One `_row_*` builder in `rig_coupling.py` (§6.2), reporting
`open-modules / mlx5+kernel / rdma-core / ib-device / per-card dma_buf / source
probe / target BAR` with honest provenance. Value: the capability becomes visible
and portable *before* anyone builds on it, and the ReBAR asymmetry of this rig
gets recorded rather than rediscovered. No GPU window.

**P2 — a comm-suite arm that measures the crossover per rig.** Template is
`_arm_collective_barlink_ucx` (`comm_suite.py:780-801`); it shells out to
`link_collective_cost.py` with `--op` and `--sizes`. Add a `gdr_crossover` arm
(kind `network`) that runs the handover's binary across the size ladder and
reports the crossover point as a rig property. Value: turns "GDR is bad at 1 MiB"
into "this rig's crossover is X" — which is what the rig-is-a-lower-bound rule
demands, and what a rig with ReBAR everywhere and a Gen5 NIC would answer
differently. Small GPU window.

**P3 — run O1 and O3 (§8).** ~30-60 minutes together, no code. O1 can move the
whole large-message verdict; O3 closes the NCCL vehicle question with evidence
instead of inference. **These two decide whether P4 is ever worth opening.**

**P4 — build a `dmabuf_rdma` barlink transport. NOT NOW.** Only if O1 shows the
crossover moving past 64 KiB on a ReBAR target, *and* O3 confirms NCCL will not
do it for us. Then: VMM staging pool per route (b) + registry entry + module +
size-keyed `handles` respecting the rank-uniformity hazard. Estimated well north
of the ~200 LoC the #240 runbook quotes for the far simpler per-class worker
split, plus a cross-rig validation pass.

**P5 — the things #266's residual actually points at, which are not dmabuf.**
The 85 % that is *not* staging (§3.2) is UCX progress, `2(W-1)` serial hops and
lock-step turnaround; the dedicated progress thread is already named and
untested. PP metadata caching (249 us/round, 64 % of the crossing at bs=1) is
already recorded as "the cheapest remaining win at the boundary". Both are
cheaper than a new transport and both survive regardless of what O1 says.

### GPU-window items (named, not executed)

| id | window | what runs | needs |
|---|---|---|---|
| **W1** | ~30 min, no server boot | `gpurdma_04_bench` cross-rig with the **5090** as rig-1 target, full size ladder (O1). Extend the ladder to 16/32/256 KiB to bracket 20 KiB and 80 KiB | both rigs, NIC idle, the handover binary rebuilt per `BUILD.md` |
| **W2** | ~20 min | TP=3 boot with `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET_GDR_LEVEL=SYS NCCL_DEBUG=INFO`, read the transport line only (O3) | rig 1, exclusive GPU |
| **W3** | ~15 min | time the registration chain (O4) and, if desired, the `expandable_segments` export probe (O5) | rig 1, one card free |
| **W4** | ~20 min | `link_collective_cost.py` with an 80 KiB phase split to close O6 | cross-rig, both rigs |

W1 and W2 are independent and can share one window. W3 is desk-adjacent and can
ride along. None of them require a model load.

---

## 10. What this document deliberately does not conclude

- It does not conclude that dmabuf GPU-RDMA is useless. It concludes that on
  **this** rig, for **our** message sizes, the adoption case is not there — and
  that two of the three load-bearing arguments for it (the 39-45 % intra-rig
  headline and the 86 % staging residual) do not mean what the summary sentences
  say they mean.
- It does not gate the feature on this rig's hardware. The crossover is a
  measurable, P2 makes it one, and a rig with Resizable BAR on every card and a
  Gen5 x16 NIC may well land on the other side.
- It does not close "cross-rig TP druecken". It narrows the reopening: the
  justification should be O1 (ReBAR target) and the transport-side items in P5,
  not the 86 % figure.
