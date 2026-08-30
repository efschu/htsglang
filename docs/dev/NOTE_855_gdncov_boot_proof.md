# W855 — #855 GDN-covered W8A8: boot proof on metal

Window: 2026-08-30 04:25Z – 04:45Z, cards 0,1,2 (holder `855-int8-gdncov`).
Tree: `/spinning/wt-855-int8` @ `0840f82601` (`feat/855-int8-gdncov`, branched
off the flip strand's shipping state `feat/969-deletion-cut`).
Log: `/spinning/evidence-665-f1/boot_855_gdncov_0840f82601_0830_042735.log`
Dead-man switch ARMED for the whole window (`boot_deadman.sh`, port 30030) —
never fired.

## What was actually new

Steps (1) and (2) of the #855 brief were ALREADY DONE and were not rebuilt:

- **(1) Microbench — done 2026-08-24**, commits `6186b06a45` + `c65e419b8b`,
  GPU window W25. Verdict stands: W8A16/Marlin **vetoed on speed**
  (prefill 2.89x sm120 / 3.27x sm86 slower; decode 1.81x sm120, a wash on
  sm86), and the GDN-covered W8A8 lane is a **1.39x (sm120) / 1.46x (sm86)
  prefill GAIN**. No veto of the INT8 lane on GDN shapes — the opposite.
- **(2) Requant artifact — done 2026-08-24**, commit `3dc2af094d`
  (`Qwen3.8-27B-INT8-gdncov`, 28.70 GiB, 16/16 structural + 12/12 dispatch +
  numerical checks).

**The gap this window closed**, and the reason a straight arm-B boot of
`3dc2af094d`'s artifact would have been a REGRESSION: that artifact was built
from the PLAIN incumbent `Qwen3.8-27B-INT8`, but the live serving checkpoint is
`Qwen3.8-27B-INT8-vocabint8-embed` (the #727 int8-embed axis, -1.18 GiB).
Booting plain gdncov would have bought the GDN win by giving the embed win back.

So the artifact booted here is the **UNION of both INT8 axes**:

    Qwen3.8-27B-INT8-gdncov-vocabembed   27.541 GiB
      = tools/requant_gdn_int8_855.py (3dc2af094d, UNCHANGED)
        applied on top of Qwen3.8-27B-INT8-vocabint8-embed (32.695 GiB)

No code change was needed: `rewrite_ignore()` (`tools/requant_gdn_int8_855.py`
:185-198) only requires `re:.*linear_attn.*` in the ignore list and preserves
every other entry, and the two axes are disjoint. 144/144 target tensors
quantized, 16 shards rewritten, 2 hardlinked.

## Measured

**Weight image: 32.695 -> 27.541 GiB = -5.154 GiB per rank image (exact).**

KV pool, arm A = the 03:07Z boot on the SAME tree with byte-identical flags and
the incumbent checkpoint (`boot_969nogrid_a51e5e8f28_0830_030705.log`):

| | arm A (incumbent) | arm B (gdncov+vocabembed) | delta |
|---|---|---|---|
| PP phase, per-rank tokens | 438,469 | 643,198 | **+204,729 (+46.7 %)** |
| TP phase, global `max_total_num_tokens` | 967,616 | 1,372,096 | **+404,480 (+41.8 %)** |
| PP0 `available_bytes` | 6.691 GiB | 9.814 GiB | +3.124 GiB |
| PP1 `available_bytes` | 4.031 GiB | 5.333 GiB | +1.302 GiB |
| PP2 `available_bytes` | 4.078 GiB | 5.307 GiB | +1.228 GiB |
| TP0/1/2 `available_bytes` | 18.895/10.976/11.437 | 24.602/14.040/14.073 | +5.707/+3.065/+2.637 |

This is FAR above NOTE_855 §6's prediction of +168,897 tokens. The prediction
was framed against the plain incumbent in a single layout; measured here is the
union artifact across the live PP/flip geometry. **The +37.0 % figure in
NOTE_855 §4 is superseded by these two measured numbers, not corroborated.**

CAVEAT, per WINDOW-PROTOCOL gate 2: the seam reserve is read from the previous
boot's cache record, so the KV-token delta is not a clean A/B of the checkpoints
alone. The **weight-image delta (-5.154 GiB) is exact and carries no caveat**;
the token delta is the operationally interesting number and inherits it.

## Acceptance

- **BOOTS**: server healthy 3.7 min after launch (04:27:35Z -> 04:31:26Z).
- **COHERENT — the #763 gate, read the right way.** Probed through `/generate`
  (no reasoning parser in front) AND through `/v1/chat/completions` reading
  `reasoning_content`, because #763's garbage signature hides as
  `CONTENT: ''` + soup in `reasoning_content`. Determined answers:
  Paris / 17x24=408 / doubling sequence / 143-67=76 / Jupiter, and a coherent
  chain-of-thought on the sheep riddle (`content='9 sheep are left.'`,
  `finish_reason=stop`). No empty-content-with-soup anywhere.
- **COHERENT ACROSS FLIPS**: 105 completed `PHASE-FLIP DONE` cycles under
  agent-shaped load (5 sessions, 300 s, 8.4k-token shared prefix); the three
  post-flip probes above were taken after those cycles.
- **SPEC ACCEPTANCE HEALTHY**: `spec_accept_length` 2.19-2.26 of a maximum 3
  (`--speculative-num-draft-tokens 3`), accept rate 62.4 %, histogram
  [46, 41, 90]. Nowhere near the #774 broken signature of 1.02. The GDN change
  did not disturb drafting.
- **DECODE**: 18.8-24.8 tok/s at bs=1.
- **NO NEW FAULT CLASS**: 2 tracebacks, both the same named `StrayHostIndexError`
  HiCache rebind refusal (`memory_pool_host.py:262`) — a deliberate guard that
  refuses by name rather than returning a wrong page. **Pre-existing and FEWER
  than arm A** (arm A: 6 refusals / 5 tracebacks; arm B: 4 / 2). Not attributable
  to the checkpoint; nothing in this change touches the HiCache binding axis.
- **VRAM corridor under load** (819-1229 MiB NVML-free/card): 5090 dipped to
  744-826 MiB at the load peak, the two 3080s sat 1275-2175 MiB — i.e. ABOVE the
  band. Read as: the 3080s still carry slack at the unchanged
  `--rank-gpu-memory-mib 31800,18800,19800`, so the freed VRAM is not yet fully
  converted. That is a follow-up tuning lever, not a defect of the artifact.

## Accountability (user law int8-ueberall-mit-rechenschaft, 2026-08-30)

**ADOPTED.** Quality cost: 144 GDN dense projections (in_proj_qkv, in_proj_z,
out_proj x 48 layers) move BF16 -> int8 per-channel symmetric weights with
dynamic per-token int8 activations. Measured weight error (from the identical
scheme on the plain build, NOTE_855_gdncov_artifact.md): rel-Frobenius median
1.018 %, worst 1.447 % (layers.57.out_proj, SNR 36.8 dB). Crest factors on GDN
(median 4.0-4.3) match the MLP tensors the incumbent already quantizes (median
3.8-4.5) — same population, no pathological range. Bought with it: -5.154 GiB of
weight image, +41.8 % TP / +46.7 % PP KV pool, and a measured 1.39-1.46x prefill
linear-layer gain. **Ratio judgement: overwhelmingly worth it**, and the
behavioural axis that the desk work could NOT bound — 144 formerly-BF16
activation paths now running `per_token_quant_int8` — came back clean on every
coherence and spec-acceptance probe above.

**NOT TAKEN, deliberately.** `lm_head` int8 (the `Qwen3.8-27B-INT8-vocabint8-both`
axis, a further -1.18 GiB). Reason is attribution, not quality doubt: stacking a
second unproven axis into the same boot makes any regression unattributable.
It is the **next lever** if this state holds, not a rejection.

## Follow-ups this window creates

1. `--rank-gpu-memory-mib` retune: the 3080s are above the corridor band, so
   part of the -5.154 GiB is currently unspent.
2. `lm_head` int8 on top (a further -1.18 GiB), as its own arm.
3. KLD / club-3090 quality suite against arm A — NOTE_855 §6 measurement (2).
   Coherence and spec acceptance are proven here; the KLD number is not, and
   this note does not claim it.
