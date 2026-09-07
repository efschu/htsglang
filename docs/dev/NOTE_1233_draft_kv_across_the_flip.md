# NOTE #1233: draft KV across the Weg-2 flip (2026-09-07)

User order (verbatim): "der draft (mtp head aus dem originalmodell) muss selbstverständlich
auch im pp3 layout laufen und der draft kv muss layoutunabhängig in den hicache und in der
tp3 phase muss neben dem normal kv, den mamba/gdn states AUCH der draft kv lesen."

Spec of record: `/spinning/gpu-arb/weg2/WEG2_DRAFTKV_SPEC_0907.md` (judge ruling, C1-C22,
T1-T16, L1-L12, S1-S7). Branch `weg2/draftkv-0907`.

## What changed

* **R1 -- the producer.** `--speculative-draft-kv-only` (server_args) lets the Weg-2
  prefill group (PP=3, tp 1) carry the checkpoint's own `mtp.*` head on its LAST attention
  stage as a draft-KV *producer*: after every target chunk the scheduler runs
  `EagleDraftWorker._draft_extend_for_prefill` (which returns right after the draft
  forward under `draft_kv_only`), so the draft KV of every prompt token is written into a
  draft pool at target-token parity. No proposal, no verify, no draft graphs; the scheduler's
  own `spec_algorithm` stays NONE and every spec-keyed PP branch takes today's path. The
  head is built and run under `draft_pp_scope()` (a single-rank pp group built on every
  rank, `distributed/parallel_state.py`), its `embed_tokens` is loaded resident from the
  checkpoint (placement A; a `PPMissingLayer` embedding is refused by name,
  `Qwen3_5MtpEmbeddingAbsent`), its `lm_head` shared from the target. P carries D's four
  speculative flags byte-for-byte because they hash into the drafter identity; the producer
  flag is deliberately not hashed (W5).
* **R2 -- the canonical draft page.** The draft page in the store is the WHOLE 2048 B token
  page of the draft layer (`CanonicalPageSpec(1, 2048)`), written whole by P and cut on read
  by head extents on D (`canonical_page_store.build_draft_window`, head shares 2/1/1 over
  the three D ranks by `disaggregation/draft_kv_canonical.local_head_window`). While the
  window is installed the draft key drops `_{tp_rank}_{tp_size}` and `_{pp_size}_{pp_rank}`
  exactly as the KV key does (`HiCacheFile._build_key_suffixes`, third suffix); without it
  every draft key is byte-identical to before. `read_extents` (all-or-nothing) is the
  validity bit; no per-page flag.
* **R3 -- the D-side read.** The presence probe asks the draft pool as a presence-only
  `ALL_PAGES` transfer (`PoolTransfer.caps_claim=False`) in the same round trip as the KV
  pages and the mamba anchor; `resolve_draft_claim` decides `full` / `trim` (re-prefill at
  most one HiCache chunk, #939) / `cold` by name over `[d, k)` (the #993 zero fill is the
  deterministic fill behind the name). A miss inside the claimed draft prefix terminates the
  fetch like a target miss. The claim is rank-uniform through ONE MIN all_reduce over
  `[claim, -claim]`; a disagreement is `WEG2 DRAFT-DISAGREE STOP`.

## Instruments

`WEG2 DRAFT-KV-PRODUCER armed` (P, once), `WEG2 DRAFT-KV-PRODUCE` (P, per chunk with
`peak_mib`), `#706 canonical DRAFT page active` (both, at registration), `#706 draft page
write` (P, per backup batch, issued/refused), `WEG2 DRAFT-PRESENCE` (D, per prefetch:
kv_pages / draft_pages / claim / mode), `PHASE-FLIP-DRAFT ADMISSION draft-cold` with the
third reason (D), `WEG2 PUBLISH-SWEEP ... draft_issued= draft_refused=`, launcher `W10
DRAFTER-IDENTITY`, front `W9 store census (shard-walked)` and the leg-2 line's
`draft_pages= draft_miss= accept_len= accept_src=`; `/get_server_info` carries
`draft_l3_hits/misses`, `draft_l3_write_issued/refused`, `draft_cold_requests`,
`draft_trim_requests`, `drafter_identity`.

## Evidence tier

DESK-PROVEN only (hermetic tests T1-T16 green, red at 7e3a9150b4). The producer path
(C6/C7), the resident embedding load and the P VRAM ledger (`--max-total-tokens 428000`)
are boot items: `weg2dk0` (instrument baseline) then `weg2dk1` per spec section 7.
