# Provenance: Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf (derived, modified artifact)

This is a MODIFIED model artifact, not the published checkpoint. It exists as
a containment for the gfx1103 Q6_K kernel defect family (HANDOFF §12.2) and
must never be presented as the original.

| | |
|---|---|
| Source file | `/root/lh/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (laptop efeu-TP14) |
| Source SHA-256 | `0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b` |
| Source size | 22,663,387,424 B (21,614 MiB) |
| Derived file | `/root/651-p2/models/Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf` (laptop efeu-TP14) |
| Derived SHA-256 | `b2ddc153fe8f7b81351800f3adbead1c3a563642b5c7b23be0c680f15928e51d` |
| Derived size | 22,981,589,280 B (21,917 MiB, +303 MiB) |
| Tool | `docs/dev/651/requant_no_q6k.py` (this repo, commit 10f43e3da5) |
| Library | gguf-py from `/root/lh/venv` (Python 3.12.13), numpy oracle path |
| Command | `python requant_no_q6k.py <src> <dst>` |
| Date | 2026-08-08, on efeu-TP14 |

## Exact modification

The four Q6_K tensors — the only Q6_K in the file — requantized to Q8_0 via
numpy dequantize (gguf.quants, the format's reference) -> quantize. All other
tensors and all metadata byte-identical pass-through.

| Tensor | Was | Now | Packed size | Roundtrip max abs err vs Q6_K-dequantized values |
|---|---|---|---|---|
| `output.weight` | Q6_K | Q8_0 | 398 -> 515 MiB | 1.064e-03 |
| `blk.34.ffn_down_exps.weight` | Q6_K | Q8_0 | 210 -> 272 MiB | 1.186e-03 |
| `blk.38.ffn_down_exps.weight` | Q6_K | Q8_0 | 210 -> 272 MiB | 9.260e-04 |
| `blk.39.ffn_down_exps.weight` | Q6_K | Q8_0 | 210 -> 272 MiB | 2.518e-03 |

Precision note: Q8_0 (8.5 bpw) is strictly finer than Q6_K (6.5625 bpw); the
error above is the Q8_0 re-quantization error measured against the exact
dequantized Q6_K values — i.e. the derived file deviates from the original's
*represented* weights by at most ~2.5e-3, which is two orders below the
1e-1-scale nondeterministic noise the defective Q6_K kernels were injecting.
This is a precision-RAISING patch of 4 tensors under checkpoint policy §6.0
(Q4_K_M remains the quality class).

## Effect (measured, HANDOFF §12)

Serving from the derived file restored **full greedy determinism** (8/8
identical texts and first-token logprob sets; short and long dispatch paths)
where the original file was non-deterministic 8/8-distinct. Content coherence
remained broken at first — that residual defect is separate (not in the GGUF
quant kernels; all types/kernels used by this file audit clean) and is tracked
in the phase-2 log.

## Reproduction

Anyone with the source file reproduces the derived file bit-for-bit:

```bash
python docs/dev/651/requant_no_q6k.py \
    Qwen3.6-35B-A3B-UD-Q4_K_M.gguf Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf
sha256sum Qwen3.6-35B-A3B-UD-Q4KM-noQ6K.gguf
# expect b2ddc153fe8f7b81351800f3adbead1c3a563642b5c7b23be0c680f15928e51d
```

(gguf-py's Q8_0 quantizer is deterministic, so the hash is stable across
runs and hosts of the same gguf-py version.)
