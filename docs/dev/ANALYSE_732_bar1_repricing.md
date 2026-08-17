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
