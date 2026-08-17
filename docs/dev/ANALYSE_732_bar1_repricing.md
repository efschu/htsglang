# ANALYSE #732 — the family-split refusals, re-priced against barlink BAR1

Date: 2026-08-17. Desk only, no boots. Read-only re-pricing plus one verdict
per case; no planner rule is changed here.

The user's correction: **we HAVE P2P via barlink/BAR1.** Both refusals were
checked against that. One of them is mine and its stated reason was wrong.

## Summary

| case | premise used | premise correct? | verdict |
|---|---|---|---|
| (a) #705 TP-decode family split | NCCL-measured collective cost | **yes, but NCCL not BAR1** | refusal **STANDS and STRENGTHENS** — margin goes negative |
| (b) PP family placement (mine) | "no P2P, all PHB, each crossing host-staged" | **NO — conflation** | conclusion stands, **reason replaced** |

---

## 1. What each analysis actually priced

### (a) #705 — `d937d5f76b`, rules landed `90e4b3a2d4`

Priced against `ar_10kb_us = 31.0–33.7 µs` for the 10 KB bs=1 payload, from the
hardware profile's `__group__` link-matrix rows
(`INTEGRATION_R3_VALIDATION.md:12995`, `:13024`).

**That probe is NCCL.** `uneven_perf.py:1329-1330` opens the probe world with
`dist.init_process_group("nccl", ...)` and times `dist.all_reduce`
(`:1353`, `:1355`). So #705's number is a *measured on-rig* cost, correctly
labelled, but it is an **NCCL** cost — it is not barlink and not BAR1. #705 did
not conflate host-staging with transport absence; it simply priced the transport
the probe measures.

### (b) PP family placement — `79046dca7e` (mine)

`VERDICT_family_placement_cut.md` states:

> "**31 card crossings per token, against 2 for the incumbent — 15.5x more**.
> This rig has no P2P and every pair is PHB, so each crossing is a host-staged
> hop."

**That reason is wrong, and the user is right to correct it.** "No CUDA P2P /
all PHB in the topology matrix" describes what the *driver* will do for
`cudaMemcpyPeer`. It does not describe what this fork can do: barlink's BAR1
transport writes into a peer's BAR1 aperture, and it is registered as tier
**T1 peer VRAM** with measured numbers (`DESIGN_407_memory_tier_registry.md:131`).
Equating the two is exactly the conflation #732 names.

---

## 2. The measured BAR1 corpus, and its shape

From `DESIGN_407_memory_tier_registry.md:131` (**MEASURED**, sourced to
`EVAL_gdr_uebernahme.md:141`, `FEATURES_VS_UPSTREAM.md:1339,1341`, commit
`137e3a6c25`):

| payload | BAR1 vs NCCL | note |
|---|---|---|
| 20 KiB | **1.13x** | smallest datapoint: three-rank all_reduce at **45.59 µs** |
| 80 KiB | **1.34x** | |
| 1 MiB | **1.15x** | |
| 4 MiB | **1.04x** | |
| 16 MiB | **1.30x** | |

Aperture: 3080 **256 MiB** nominal / **96 MiB** contiguous; 5090 **32 GiB**
(ReBAR).

**Every one of these is a COLLECTIVE.** The row says so itself — "a collective,
not a point latency". That is not an accident of the harness; it is what the
transport is:

**barlink BAR1 has no point-to-point path.** Its dispatch seams in
`parallel_state.py` are `all_reduce` (`:1100`), `reduce_scatter_along_dim`
(`:1299`), `reduce_scatter_tensor` (`:1374`), `_all_to_all_single` (`:1438`),
`all_to_all_single` (`:1450`), `all_to_all_single_v` (`:1480`) and
`reduce_scatter` (`:1498`). Grepped `barlink_bar1.py` for `def send` / `def
recv` / `p2p`: **nothing**.

---

## 3. Case (a): re-priced — the refusal STRENGTHENS

#705's own arithmetic:

```
solo on 5090                      3.192 ms
EQUAL 1/3 shard (today)           2.506 ms
PROPORTIONAL (uneven TP, shipped) 1.726 ms
removing 48 collectives saves     1.488 – 1.618 ms
net vs proportional               +0.022 – +0.152 ms      (saving − 1.466)
```

The proposal's entire value is **removing collectives**. A faster transport
makes each removed collective *worth less*, so BAR1 moves this the wrong way for
the proposal. Applying the measured BAR1-vs-NCCL ratio to the NCCL baseline —
1.13x at 20 KiB being the nearest measured point to the 10 KB payload, 1.34x the
most favourable in the table:

```
saving under BAR1   1.488 / 1.34 = 1.110 ms   …   1.618 / 1.13 = 1.432 ms
net vs proportional 1.110 − 1.466 = −0.356 ms …   1.432 − 1.466 = −0.034 ms
```

**Under BAR1 the family split is net NEGATIVE: −0.034 to −0.356 ms.** It does
not merely fail to pay for itself, it costs. #705 refused it at +0.090 ms
(~0.3 % of a 30 ms round); on the transport the user is pointing at, the sign
flips against it.

**VERDICT (a): the refusal STANDS, with a larger margin.** No re-decision, no
window ticket — a faster interconnect cannot make collective-removal more
valuable.

*Caveat, stated:* this scales one measured number by another measured ratio. A
direct BAR1 `ar_10kb` row would replace the scaling with a measurement; it does
not exist in the corpus (see §5).

---

## 4. Case (b): the reason was wrong, the conclusion survives for a different reason

The corrected chain:

1. BAR1 **is** a direct peer path — the premise "no P2P" was wrong.
2. BAR1 is **collective-only** — it has no send/recv (§2).
3. A PP stage handoff is **point-to-point**: `send_tensor_dict`
   (`parallel_state.py:2178`) calls `send_func(tensor, self.ranks[dst],
   group=comm_group)`, a torch.distributed p2p send on the PP group, and that
   group is built `use_custom_allreduce=False` (`:3121-3127`).

So the 31 crossings would be **31 point-to-point sends per token**, and the
transport the user is correctly pointing at **cannot carry them today**. The
foreclosure holds — but because the direct transport that exists is
collective-only, not because "this rig has no P2P".

**VERDICT (b): the conclusion STANDS; the stated reason is REPLACED.**
`VERDICT_family_placement_cut.md` is **not on this branch** (it lives on
`train/0817-desk-family-placement`, commit `79046dca7e`), so the correction is
owed there and is a **merge-train item**: replace "this rig has no P2P and every
pair is PHB, so each crossing is a host-staged hop" with "BAR1 is a direct peer
transport but is collective-only; a PP handoff is point-to-point, so no direct
path carries it today". Recorded here meanwhile so the wrong sentence is never
the only account. That correction
matters even though the answer is unchanged: the old sentence would have made
any future reader dismiss BAR1 for a *collective* workload too, which is exactly
the case where it is the fastest thing on the rig.

**What would flip (b):** a BAR1 point-to-point send/recv seam. Then 29 extra
crossings/token at a p2p cost bounded above by the 45.59 µs collective figure
would be ≤1.32 ms/token against a ~30 ms round — **≤4.4 %**, a trade-off rather
than a foreclosure. That is a *build*, not a measurement, and it is filed as
such below rather than assumed.

---

## 5. What the corpus cannot answer

**There is no measured BAR1 point-to-point number**, because there is no BAR1
point-to-point path to measure. The 45.59 µs figure is a three-rank all_reduce
and the row flags it. Any per-crossing p2p cost is therefore **INFERRED** and is
used above only as an upper bound, never as a price.

---

## 6. Coupling with #709 (item 4)

#709's proportional uneven-TP is shipped-dark: **+0.780 ms/round at zero
capacity cost**, already in-tree, simply not enabled (`rank_tp_ratio=None`).

The re-pricing does **not** change the recommended bundle, and now says so with
a firmer number:

* **Enable uneven TP.** Its gain is a *shard-balance* effect — proportional
  weights against unequal cards — and a faster interconnect does not erode it.
  BAR1 leaves +0.780 ms/round standing.
* **Do not take the family split.** It was +0.090 ms under NCCL and is
  **negative** under BAR1.

The two were always one layout question, and the answer is now less ambiguous
than when #705 wrote it: the whole of the available win is in the shard vector,
none of it in the family split.

**One blocker on collecting it, from the #588 sweep:** the live server runs
`--tp-size 1 --pp-size 3`. With no TP sharding there is nothing to make uneven,
so #709 cannot be enabled on the current boot at all — the bundle needs a TP>1
layout before either half is collectable.

---

## 7. Filed

1. **BAR1 point-to-point seam** — a `send`/`recv` on the BAR1 transport. It is
   the only thing that would reopen case (b), and it is a build with its own
   red-first slice, not a measurement. No ticket claims it today.
2. **A direct BAR1 `ar_10kb` row** — would replace §3's scaling with a
   measurement. Cheap: the same probe the profile already runs, under
   `SGLANG_BARLINK_TRANSPORT=bar1`. **Window item.**
3. **#709 activation needs a TP>1 boot** — currently uncollectable.

No planner rule was changed by this document. Case (a) needs none; case (b)
needs a text correction, which is made in its own verdict file.

---

# Amendment (same day): the measurement corpus, exhaustively searched

A full sweep of the corpus landed after the first pass. It confirms the two
verdicts and sharpens four things — one of which weakens my own case (b) a
second time.

## A. "No CUDA P2P" was FACTUALLY RIGHT. The inference from it was wrong.

`/spinning/gpu-battery-results/2026-07-30_bar1/s01_p2p_reprobe/results/capability_matrix.json`
reports `"can_access_peer": false` for **all six directed pairs**. Standard CUDA
peer access really is off on this rig, and `d2d_bench.json`'s near-identical
`direct` vs `staged` timings are the visible consequence.

So the first clause of my sentence was true. The error was the second clause:
**BAR1 does not go through CUDA P2P at all.** It maps the peer's aperture via
dmabuf and writes into it, which is exactly why `can_access_peer: false` does
not bound it. I read a true fact about one mechanism as a bound on another.

## B. The magnitude argument was wrong too — and this is the bigger error

`benchmark/bench_host_transport.py:12` records a MEASURED point-to-point
ping-pong: **7.30 µs against NCCL's 37.41 µs at 20 KiB**. That is the
`barlink_host` transport — host-staged, the very class my verdict assumed was
prohibitively expensive.

A PP crossing is one-way 10 KiB; the ping-pong is a 20 KiB round trip, so
7.30 µs is a generous upper bound for one crossing. The 29 EXTRA crossings then
cost:

```
29 x 7.30 us = 0.212 ms/token   ~=  0.7 % of a ~30 ms bs=1 round
```

**So "31 crossings vs 2, therefore foreclosed by the interconnect" does not
survive its own arithmetic, even host-staged.** The 15.5x crossing-count ratio
is real and is not the point: the per-crossing cost is small enough that the
count does not foreclose anything.

**Net effect on case (b): the conclusion still stands, but now on ONE ground
only — barlink exposes no `send`/`recv`, so no direct transport carries a p2p
crossing today.** It does not stand on cost. Both of my original supports —
"no P2P" and "15.5x is prohibitive" — are withdrawn.

The in-tree statement of the API absence is more explicit than my seam grep:
`scripts/probe/barlink_vs_nccl.py` says outright that barlink "implementiert
ausschliesslich Kollektive (all_reduce, broadcast, all_gather, reduce_scatter)
und KEIN send/recv".

## C. `DESIGN_407:131` miscites its own source

That row cites `EVAL_gdr_uebernahme.md:141`. That document is about **dmabuf
GPU-RDMA over RoCE** — a different transport — and contains none of the
numbers: grepping it for `45.59`, `1.13`, `1.34`, `interleaved` returns **zero
matches**. The real source is `FEATURES_VS_UPSTREAM.md:1341` plus commit
`137e3a6c25`, which the same row also cites. The measurement is sound; the
citation points at the wrong file. **Filed as a catalog/doc correction.**

The same DESIGN_407 also claims `scripts/p2p_readiness/` "has never been run,
no `results/` exists". It has: the results directory above is that run's output.
**Second stale claim in the same document.**

## D. A 2-rank weak spot the 3-rank ratios hide

`FEATURES_VS_UPSTREAM.md:1349`: on the fast x8 **pair** the transport *loses*
between 1 and 8 MiB, down to **0.81x**, with 2-rank ratios
1.11/1.13/0.97/0.86/0.99 recorded as "unerklaert".

This does not disturb §3: #705's TP-decode collectives are 3-rank on a TP=3
group, so the 3-rank ratios are the right ones. But BAR1 is **not** a uniform
win, and a future re-pricing on a 2-rank group must not reuse §3's numbers.

## E. The p2p tooling exists and has never been run

`scripts/probe/p2pproof.cu` is a raw CUDA point-to-point BAR1 probe that prints
`P2PDATA` / `CANACCESS` lines. Searching the entire `/spinning` tree (~30
worktrees) for both literals finds them **only inside the source file** — never
as captured output. So the instrument for the one number this analysis lacks
exists and has never been executed.

That is a cheap window item, and it is now the second one filed here.

## Filed, amended

1. **BAR1 point-to-point seam** — the build that would reopen case (b) on cost
   grounds. Unchanged.
2. **Run `scripts/probe/p2pproof.cu`** — would give the first real BAR1 p2p
   number. Window item, cheap; the probe is written.
3. **A direct BAR1 `ar_10kb` row** — would replace §3's scaling. Window item.
4. **`DESIGN_407:131` citation fix** and its stale `p2p_readiness` claim —
   doc corrections, merge-train items.
