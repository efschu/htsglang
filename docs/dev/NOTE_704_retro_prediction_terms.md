# #704 item 2 — the retro-prediction term set, derived from config

Review-gate item 2 (`REVIEW_GATE_704.md` §3): the corrected pool model must
retro-predict all four measured boots within ~2k tokens, with **every term from
config or instruments and nothing fitted**, before any new cut boots.

Status: **the pool-side terms are closed and validated to <1.1 % on all three
ranks. One term remains open — the graph reserve — and it, not the arming
floor, is the real blocker.** That supersedes my earlier claim that the +-7 %
unbooted-floor uncertainty was the blocking uncertainty.

---

## 1 — Instruments used (no fitting)

`boot_bundle.log`, the live [28,20,16] boot, gives a complete per-rank chain:

| rank | avail at load begin | after weights | after pool | pool consumed |
|---|---:|---:|---:|---:|
| PP0 | 30.46 GiB | 14.82 | 7.91 | **6.91** |
| PP1 | 19.11 | 9.32 | 4.37 | **4.95** |
| PP2 | 19.10 | 8.70 | 4.73 | **3.97** |

PP2's weight draw is identical in `boot_armC2.log` ([32,16,16]), which is the
gate's byte-identity finding reproduced independently here.

## 2 — The KV term is byte-exact from config

`kv_cache_dtype='fp8_e4m3'` (1 byte), `num_key_value_heads=4`,
`head_dim=256`, so K alone is `4 x 256 x 1 = 1024 B` per token per ATTENTION
layer and K+V is **2048 B**. The boot log prints K per rank at 436,766 tokens:

| rank | attn | attn x 1024 B x 436,766 | logged |
|---|---:|---:|---:|
| PP0 | 7 | 2.916 GiB | 2.92 |
| PP1 | 5 | 2.083 GiB | 2.08 |
| PP2 | 4 | 1.666 GiB | 1.67 |

Byte-exact on all three. The constant is **read from config, never fitted**;
Slot-3's doc had 4096 B (bf16), which this falsifies directly.

## 3 — The mamba residency term, derived

From the allocation sites (`mem_cache/memory_pool.py:583-608`, `:655-665`,
`:693-705`), with `max_mamba_cache_size=12` (so 13 slots),
`max_running_requests=4` (so 5 spec slots), `speculative_num_draft_tokens=4`,
`temporal_state_shape=(48,128,128)`, `conv_dim=10240`, `win=K-1=3`, ssm/conv
dtype bf16:

| component | shape | MiB per GDN layer |
|---|---|---:|
| `temporal_state` | 13 x 48 x 128 x 128 x 2 B | **19.50** |
| `conv_state` | 13 x 10240 x 3 x 2 B | 0.762 |
| `intermediate_ssm_state_cache` | 5 x 4 x 48 x 128 x 128 x 2 B | **30.00** |
| `intermediate_conv_window_cache` | 5 x 10240 x (4+3-1) x 2 B | 0.586 |
| **total** | | **50.85** |

The 19.50 figure the gate quoted is the `temporal_state` component alone; the
full per-GDN-layer residency is 50.85 MiB, dominated by the SPECULATIVE
intermediate cache at 30.00. A model charging only 19.5 under-charges the GDN
layers by 2.6x.

## 4 — Validation: forward prediction, nothing fitted

`pool_consumed_r = attn_r x 2048 B x tokens + gdn_r x 50.85 MiB`

| rank | attn | gdn | KV | mamba | predicted | measured | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| PP0 | 7 | 21 | 5.831 | 1.043 | 6.874 | 6.91 | **-0.036 GiB** |
| PP1 | 5 | 15 | 4.165 | 0.745 | 4.910 | 4.95 | **-0.040 GiB** |
| PP2 | 4 | 12 | 3.332 | 0.596 | 3.928 | 3.97 | **-0.042 GiB** |

Under 1.1 % on every rank, and the residual is **nearly constant (~40 MiB)
rather than scaling with layers or attention** — i.e. a fixed per-rank
allocator/metadata term, not a missing per-layer term. That constancy is the
evidence that the per-layer terms above are complete.

## 5 — What is still open, and a correction to my own blocker claim

Retro-prediction runs the equation BACKWARDS: given the memory a rank has, solve
for tokens. That needs everything the sizer sets aside before the pool:

```
tokens_r = (avail_after_weights_r - floor_r - graph_reserve_r - mamba_r - fixed_r)
           / (attn_r x 2048 B)
```

Every term is now config-derived except **`graph_reserve_r`**. From the chain:
`avail_after_pool` is 7.91 / 4.37 / 4.73 GiB, and subtracting the measured #676
floors (1728 / 1825 / 2467 MiB) leaves **6,372 / 2,650 / 2,377 MiB** for graphs
plus slack. That is a large term, it varies per rank, and it is not derivable
from config alone — it depends on the captured batch-size ladder and shapes.

**Correction on the record.** I previously named the +-7 % unbooted-floor
uncertainty as the blocker for bootable predictions. That was wrong in
emphasis: the floor is measured per layout and its uncertainty is second-order
next to a graph reserve of 2.4-6.4 GiB per rank. The right move is not to model
the graph reserve independently but to have the pool solve CONSUME the sizer's
own reserve terms, exactly as it must consume the #676 floor — the sizer
already computes both.

## 6 — Structural finding to encode (gate §3.5)

No cut that keeps rank2 = layers 48-63 can beat the incumbent pool. Rank2 is
byte-identical across [28,20,16] and [32,16,16] (weights 10.40 GiB in both
logs), and it binds at 436,766. Under the min-rule that is a hard ceiling for
any such cut. Pool gains therefore require shrinking rank2's ATTENTION count
(e.g. a 12-layer rank2 = 3 attn, cap x4/3) or Part B decoupling — not
rebalancing rank0/rank1.

The incumbent's binder is **PP2**, per the boot log (PP1 cap 463,406, PP2
436,766). My rev5 calibration assigned it to rank1 and solved a free constant
from that assumption; the derivation above needs no such fit and supersedes it.

---

## 7 — The ~14 % mamba discrepancy, RESOLVED: my model was right, the POST is short

The operator flagged the metal mamba posts (0.895 / 0.639 / 0.511 GiB) as
~14 % below my derived 50.85 MiB/GDN-layer, with the suspected cause "an
over-charged spec-slot sub-term" in my model. That suspicion is **wrong**, and
the allocator's own instrument settles it without any new code: MambaPool
already logs every sub-term at construction (`memory_pool.py:756-765`,
`"Mamba Cache is allocated. ... conv_state size / ssm_state size /
intermediate_ssm_state_cache size / intermediate_conv_window_cache size"`).

Read off the live boots:

| rank | GDN layers | conv | ssm | inter_ssm | inter_cw | allocated | per layer | budget post | post/alloc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PP0 | 21 | 0.02 | 0.40 | 0.62 | 0.01 | 1.05 GiB | **51.20 MiB** | 0.895 | **0.852** |
| PP1 | 15 | 0.01 | 0.29 | 0.44 | 0.01 | 0.75 GiB | **51.20 MiB** | 0.639 | **0.852** |
| PP2 | 12 | 0.01 | 0.23 | 0.35 | 0.01 | 0.60 GiB | **51.20 MiB** | 0.511 | **0.852** |

Two facts fall out:

**(a) The derivation in §3 is confirmed.** 50.85 MiB/GDN-layer against a
measured 51.20 — inside the log's own 2-decimal GiB rounding (each of four
terms rounds to 0.01 GiB, so ±1 MiB/layer). The individual terms match too:
`ssm_state` is exactly `layers x 13 slots x 1.5 MiB` on every rank (21 -> 0.400,
15 -> 0.286, 12 -> 0.229 GiB), and `intermediate_ssm` is exactly
`layers x 5 spec slots x 4 draft x 1.5 MiB` (30.0 MiB/layer measured on all
three). So the spec-slot count of 5 is right, not over-charged.

**(b) The BUDGET POST under-charges the allocation by a constant 14.8 %.**
`post/allocated` is **0.852 on all three ranks** — exactly constant, so this is
a systematic formula divergence, not rounding and not noise. The sizer hands
out ~158 / 111 / 89 MiB more mamba memory than it charges to its own budget.

**Consequence for the retro-prediction gate.** The pool solve must consume the
ALLOCATED figure (the `Mamba Cache is allocated` line), not the budget post.
A solve fed the post is optimistic by ~150 MiB/rank, and that error is
systematic rather than averaging out. This also means the earlier plan to feed
`PhasePoolModel` from the budget posts alone would have baked the under-charge
into the model.

What is NOT established: which term in `handle_max_mamba_cache` produces the
0.852. It is exactly constant, so it is one factor, not an accumulation of
approximations, but naming it needs a read of that function rather than more
arithmetic from outside — the same discipline that produced this resolution.
Recorded as the open sub-item; it does not block the gate, because the gate
now consumes the allocation.
