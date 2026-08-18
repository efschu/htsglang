# NOTE 735 — Independent arithmetic check of the #732 family-plan material

Date: 2026-08-17. Desk-only re-derivation. No documents edited; nothing committed.

Sources checked (all three present, recovery paths noted):

| document | where it was found |
| --- | --- |
| `DESIGN_family_fullplan.md` | **absent from both working trees** (`/spinning/htsglang`, `/spinning/wt-602-slot2`); recovered from commit `dd75cfc1cf` (`git show dd75cfc1cf:docs/dev/DESIGN_family_fullplan.md`) |
| `NOTE_732_breakable_crossing.md` | branch `feat/barlink-p2p-seam` (`git show feat/barlink-p2p-seam:docs/dev/NOTE_732_breakable_crossing.md`) |
| `NOTE_732_transport_selection.md` | branch `feat/barlink-p2p-seam` (same path) |

One further source, because two of the six checks do not sit in the three named
documents: the GDN-state claim (817 152 el / 149.6 MiB / 21–24 slots) lives in
`/spinning/wt-602-slot2/docs/dev/DESIGN_pp_layer_set.md` §4 (also present in
`/spinning/wt-cat`), the successor design for the same plan. It is checked here.
Checkpoint facts (hidden_size, FA positions, KV geometry, GDN dims) were
re-verified directly against
`/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8-yarn4.0/config.json`:
`hidden_size 5120`, `num_hidden_layers 64`, `vocab_size 248320`,
`full_attention_interval 4`, `num_key_value_heads 4`, `head_dim 256`,
`tie_word_embeddings false`, FA at exactly `[3,7,…,63]` (16 FA, 48 GDN),
GDN `linear_num_key_heads 16`, `linear_num_value_heads 48`,
`linear_key_head_dim 128`, `linear_value_head_dim 128`. All cited facts
CONFIRMED at the source.

## Verdict table

| # | claim | their number | my number | verdict |
|---|---|---|---|---|
| 1 | crossing count for 16 FA `[3,7,…,63]` in 48 GDN, terminal-layer effect | 31 (16 out, 15 back) | 31 inter-layer; **32** incl. the layer-63→head send | **CONFIRMED** as the inter-layer count; caveat below (32nd crossing to the head) |
| 2 | per-crossing payload, hidden 5120 bf16, chunk 512 | 5.0 MiB | 5 242 880 B = exactly 5.0 MiB | **CONFIRMED** |
| 3 | transport share of a prefill pass | ~5.18 % (per-link) / 5.7 % (uniform 5.6 GB/s, 29 extra) | 24.68 ms / 476 ms = 5.185 %; 27.15 ms / 476 ms = 5.70 % | **CONFIRMED** (two different framings, both correct in theirs) |
| 4 | decode break-even at 1/2/5/10 % of a 30 ms round, 31 breaks/token | 2.8 / 12.5 / 41.5 / 89.9 µs | 2.84 / 12.52 / 41.55 / 89.94 µs on their basis; 2.37 / 12.04 / 41.08 / 89.46 µs on a self-consistent 31-crossing basis | **CONFIRMED** on the stated basis; mixed 29/31 basis flagged |
| 5 | GDN state 817 152 el/l/slot f32 → 149.6 MiB/48 layers/slot; slots 21–24 at 4738 MiB − 1–1.5 GiB pool | 149.6 MiB; 21–24 | 156 893 184 B = 149.625 MiB; floor 24 / floor 21 | **CONFIRMED** |
| 6 | KV 852 MiB per FA layer; 4.41 GiB KV per percentage point of pass time | 852 MiB; 4.41 GiB/pp | 852.10 MiB; 4.408 GiB/pp (and 948.2 MiB/ms) | **CONFIRMED** |

No DISCREPANCY at the level of the claimed values: all six reproduce. Two
basis-level caveats carry real (small) weight and are itemized in §7.

## 1. Crossing count (item 1)

FA positions from `config.json`: `3,7,11,…,63` — `(63−3)/4 + 1 = 16` FA,
`64 − 16 = 48` GDN, and **63 is an FA layer** (terminal). Confirmed against the
file, not just the document.

Execution order is 16 blocks of `G G G F`. No two FA layers are adjacent, so
every FA layer is entered from the GDN card:

```
5090 -> FA : once per FA layer              = 16
FA -> 5090 : after each FA layer EXCEPT the terminal one = 15
total      = 16 + 15 = 31 = 2*16 - 1
```

The terminal-layer effect (15 vs 16 per side) is exactly the stated
asymmetry: layer 63 is never followed by a GDN block, so it costs one crossing
instead of two. The document's phrasing "the '29 extra' elsewhere is the same
schedule counted against a 2-crossing PP baseline" also reproduces: `31 − 2 = 29`.

**Caveat (caveat, not a miscount).** `DESIGN_family_fullplan.md` §2.1 places
`lm_head` on the 5090 (int8, 1212 MiB). The output of the terminal FA layer 63
therefore must still move from the FA card to the 5090 for final norm +
lm_head — a further crossing (16 out + 16 back = **32** card movements per
forward, 30 extra over the PP baseline). The family's own successor document
concedes this: `DESIGN_pp_layer_set.md` §5 — "16 crossings leave the GDN card,
15 return, and the **missing 32nd** is the terminal layer, whose output goes to
the head". So "31" is correct *as the count of ownership changes between
consecutive layers* (which is also what the crossing-schedule code is built to
emit); every *cost* figure derived from 31 (5.18 %, 0.227 ms/token, 155 MiB/pass,
the break-even table) is missing exactly one crossing of the same class
(~3 % of the transport term, +10 240 B/token in decode). It does not change any
verdict in the family.

## 2. Per-crossing payload (item 2)

```
per token  = 5120 x 2 B = 10 240 B = 10.0 KiB        (bs=1, doc's "exact" figure)
chunk 512  = 10 240 B x 512 = 5 242 880 B = 5.0 MiB  (exactly; doc: 5.0 MiB)
chunk 2048 = 10 240 B x 2048 = 20 971 520 B = 20.0 MiB
31 x 5.0  = 155 MiB/pass;  31 x 20.0 = 620 MiB/pass
```

CONFIRMED to the byte. Unit discipline: the documents use decimal GB/s
(10⁹ B/s) for link bandwidth and binary MiB (2²⁰ B) for payload — mixed but
internally consistent; the conversions above use that convention.

## 3. Transport share of a prefill pass (item 3)

Per-link crossing time at 5 242 880 B:

```
x8 pair (5090<->id2, 9.06 GB/s): 5 242 880 / 9.06e9 = 578.7 us
x4 pair (5090<->id0, 5.10 GB/s): 5 242 880 / 5.10e9 = 1028.0 us   (1.78x slower)
```

Both reproduce. The transport note's 8/8 row:

```
16 x 578.7 us + 15 x 1028.0 us = 9259.2 + 15420.0 = 24 679.2 us = 24.68 ms
24.68 / 476 ms = 5.185 %  ->  "5.18 %"
```

CONFIRMED. The full split table reproduces: 10/6 → 22.88 ms (4.81 %),
12/4 → 21.08 ms (4.43 %), 16/0 → 17.94 ms (3.77 %); the layer-63 lever
(15·1028.0 + 16·578.7 = 24.68 vs 16·1028.0 + 15·578.7 = 25.13 ms, diff
0.449 ms → "0.45 ms/pass"); "2 x (1028.0 − 578.7) = 0.899 ms" per FA layer
moved off x4; "3.59 ms concession 8/8 vs 12/4" (24.679 − 21.085 = 3.594 ms);
"15 x 1028 us = 15.4 ms, 62 %" (15.42 ms, 62.48 %); delta vs the 2-crossing
baseline "~1.6 ms → 23.1 ms ≈ 4.85 %" (23.08 ms → 4.849 %).

Framing clarification, because the two notes use different counts:
**5.18 % prices ALL 31 crossings per link** (transport note §3); **5.7 % prices
the 29 EXTRA crossings at a uniform 5.6 GB/s** (breakable note §3):
`5 242 880 / 5.6e9 = 936.2 us`; `29 x 936.2 us = 27.15 ms`; `27.15/476 = 5.70 %`.
Chunk 2048 row likewise: `29 x 3744.9 us = 108.6 ms`; `108.6/1906 = 5.70 %` —
the chunk-invariance claim holds because payload and pass length scale
together. The reference 476 ms is itself `1906 ms / 4 = 476.5 ms`, linearly
scaled (marked INFERRED in the source). Unit consistency: clean
(B / (B·s⁻¹) = s; ms / ms = %). CONFIRMED.

## 4. Decode break-even thresholds (item 4)

Model: a 30 ms round, budget b ∈ {1,2,5,10} % = {0.30, 0.60, 1.50, 3.00} ms,
consumed by transport + 31 breaks/token: `31 x break_cost = b - transport`.

The document does not state its transport offset in the table; re-solving all
four rows for it gives a single consistent value:

```
b = 0.30 ms: (0.300 - T)/31 = 2.8 us  ->  T = 0.2119 ms
b = 0.60 ms: (0.600 - T)/31 = 12.5 us ->  T = 0.2118 ms
b = 1.50 ms: (1.500 - T)/31 = 41.5 us ->  T = 0.2120 ms
b = 3.00 ms: (3.000 - T)/31 = 89.9 us ->  T = 0.2123 ms
```

So the table is exactly `(budget − 0.212 ms)/31`, where 0.212 ms is the
cited "29 crossings ≈ 0.212 ms" figure from #732's amendment. All four values
reproduce at the stated precision (2.839, 12.516, 41.548, 89.935 µs).
CONFIRMED on that basis.

**Basis caveat.** The offset is the *29-crossing* transport total while the
break count is *31* — a 29/31 mix within one table. On the self-consistent
basis the same note states elsewhere ("31 crossings x ~7.3 us", i.e.
`31 x 7.3103 us = 0.2263 ms`; note 0.212/29 = 7.3103 µs is the origin of the
"7.3"), the thresholds become:

| budget | their (T = 0.212 ms) | self-consistent (T = 31 x 7.31 µs = 0.2263 ms) |
|---|---|---|
| 1 % | 2.8 µs | 2.37 µs |
| 2 % | 12.5 µs | 12.04 µs |
| 5 % | 41.5 µs | 41.08 µs |
| 10 % | 89.9 µs | 89.46 µs |

A shift of 0.4–0.5 µs (~1 % relative) that moves no decision: the headline
"~41.5 µs at a 5 % budget" is robust either way, and if the §1 caveat's 32nd
crossing is included (`32 x 7.31 µs = 0.2339 ms`, 32 breaks) the 5 % value is
39.6 µs — same band. The same 29/31 mix appears in breakable-note §5:
"equal at break_cost = 6.8 µs" is `0.212 ms / 31 = 6.84 µs`, while the
transport bound it quotes there is 0.227 ms (`0.227/31 = 7.32 µs`); and
"worth ~6x it at 41.5 µs" is `31 x 41.5 us / 0.212 ms = 6.07x` (5.67x on the
0.227 basis). Units throughout are consistent (ms, µs, per-token); this is a
count-basis issue, not a unit error.

## 5. GDN state arithmetic (item 5)

Source: `DESIGN_pp_layer_set.md` §4 (not in the three named documents — see
recovery note above). Element count re-derived from `config.json` and the tree's
shape code (`configs/mamba_utils.py`: temporal = `(num_heads, head_dim,
state_size)`, conv = `(intermediate + 2·n_groups·state_size, conv_kernel−1)`):

```
temporal = 48 x 128 x 128                  = 786 432 el
conv     = (6144 + 2 x 16 x 128) x (4 - 1) = 10 240 x 3 = 30 720 el
per layer per slot                          = 817 152 el   (48 = value heads,
                                                           16 = key heads, 128 = both head dims, conv kernel 4)
```

Multiplication the task asks to verify:

```
817 152 x 4 B x 48 = 156 893 184 B
156 893 184 / 1 048 576 = 149.625 MiB  ->  "149.6 MiB"   (exactly 149 5/8)
per layer per slot: 817 152 x 4 = 3 268 608 B = 3.117 MiB
```

Slot ceilings at 4738 MiB free (4738 = 32768 − 25 278 weights − 1728 floor −
1024 corridor; the 25 278 = 48 x 476.1 GDN + 2 x 1212.5 int8 head chain
reproduces) minus a 1–1.5 GiB graph pool:

```
pool 0    MiB: 4738 / 149.625  = 31.67 -> floor 31      (doc: 31)
pool 1024 MiB (1 GiB): 3714/149.625 = 24.82 -> floor 24  (doc: 24)
pool 1536 MiB (1.5 GiB): 3202/149.625 = 21.40 -> floor 21  (doc: 21)
```

"**21–24 concurrent mamba slots**" CONFIRMED. The bf16-vocab companion
("(2313 − 1024)/149.625 = 8.6 → ~8 slots") and the "2x concentration" statement
(48 vs 24 GDN layers on the 5090) also reproduce.

**Dependent caveat (not a check failure).** 4738 MiB uses the nominal 32768 MiB
5090 total; the live `nvidia-smi` row in the transport note reads 32607 MiB.
With the NVML total the free figure is 4577 MiB and the ceiling becomes
floor(4577−1024)/149.625 = 23 to floor(4577−1536)/149.625 = 20, i.e. 20–23.
The 21–24 claim is arithmetic-correct for its stated input; the input's last
three digits are nominal, not NVML.

## 6. KV arithmetic (item 6)

From `config.json`: `num_key_value_heads 4`, `head_dim 256`:

```
K+V per token per FA layer = 2 x 4 x 256 = 2048 el; fp8 = 2048 B = 2 KiB
at --max-total-tokens 436 275:
KV per FA layer = 436 275 x 2048 B = 893 491 200 B = 852.10 MiB  ->  "852 MiB"
4 layers  = 3408.4 MiB   (doc: 3408)      16 layers = 13 633.6 MiB (doc: 13 634)
```

Exchange rate for the link-aware rebalancing (8/8 -> 12/4 saves
3.5947 ms/pass by the §3 per-link figures and moves 4 layers = 3408.4 MiB of
KV):

```
3408.4 MiB / 3.5947 ms   = 948.2 MiB KV per ms saved        (doc: ~948)
3.5947 ms / 476 ms       = 0.75518 percentage points
3408.4 MiB / 0.75518 pp  = 4513.3 MiB/pp = 4.408 GiB/pp     (doc: ~4.41)
```

CONFIRMED; the "~" hedges on both are earned (0.2 % off at most).

## 7. Caveats that survive the check (carried, not fixed)

1. **The 32nd crossing** (§1): lm_head sits on the 5090, so the terminal layer's
   output crosses after all; 31 is the inter-layer count, 32 the full-forward
   movement count. ~3 % of the transport term; acknowledged by
   `DESIGN_pp_layer_set.md` §5, not by the two #732 notes or the family design.
2. **The 29/31 basis mix** (§4): break-even offsets and the 6.8 µs / 0.227 ms
   pairings use the 29-crossing transport total next to 31-break counts; shifts
   the decode thresholds by ≤0.5 µs, no verdict change.
3. **Nominal vs NVML 5090 total** (§5): 32768 vs 32607 MiB moves the slot
   ceiling 21–24 → 20–23.
4. Minor, non-load-bearing: 31 x 7.3 µs = 0.2263 ms is printed as "~0.227";
   the family design's "roughly 5x more room for KV on the small cards" does
   not reproduce from its own numbers (14 790 vs 7 565/6 989 MiB on the
   contiguous-arm 3080s ≈ 2.0x; vs the 5090's 4 738 MiB ≈ 3.1x); the 476 ms
   pass reference remains INFERRED (1906 ms/4), as the sources themselves say.

## Method

Every line above was re-computed independently (Python, exact integer
arithmetic; bandwidths as decimal GB/s, sizes as binary MiB, matching the
sources' convention). Checkpoints facts were read from the config file, not
from the documents. Where a document's number is a rounded print of an exact
value, the exact value is shown. No document was modified; this note is
uncommitted by design.
