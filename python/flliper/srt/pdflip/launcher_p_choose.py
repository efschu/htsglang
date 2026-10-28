"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from typing import Optional
from typing import Tuple

from flliper.srt.pdflip.launcher_d_mid import CONTEXT_LENGTH_TOKENS
from flliper.srt.pdflip.launcher_hicache_calib import _rope_context_terms

def p_rope_context_delta_mib(ns, n_stages: int) -> Tuple[Tuple[float, ...], Optional[str]]:
    """YaRN x2 27B (29.09.): the per-stage VRAM the EAGER RoPE cos/sin cache
    grows by when group P boots a longer context than the one its per-stage
    posts (``--pp-cut-stage-fixed-mib``, P_PP_STAGE_FIXED_MIB) were read at.

    ``reserve_rope_cache_for_long_sequences`` sizes every cache to the context
    (fp32, ``rows x rotary_dim``) in ``load_model``, i.e. BEFORE the KV pool is
    sized, so the growth comes straight out of the pool. MEASURED at 262144
    (dkr27browauthoritybar1fs09290956, P): 'MRotaryEmbedding 262528x64 float32
    64.1 MiB' on PP0/PP1/PP2 and the DFlash2 draft's 'RotaryEmbedding
    262528x128 float32 128.2 MiB' on PP2 (the draft sits cold on the last
    stage). At 524288 the pool model would otherwise price PP0 +64 MiB and PP2
    +192 MiB too optimistic -- 4.1k / 24.6k tokens under the cap+chunk floor
    it promised (per_request_max > realised pool). Delta = (ctx - 262144) x
    cols x 4 B; target on every stage, the DFLASH draft on the last one.
    ``((0,)*n, None)`` at or below 262144 -- every x1 form is byte-identical."""
    zero = tuple(0.0 for _ in range(int(n_stages)))
    if n_stages <= 0:
        return zero, None
    ctx, src, extra_rows, tcols, dcols, dname = _rope_context_terms(ns, "p")
    if extra_rows <= 0:
        return zero, None
    per_col = extra_rows * 4 / (1024.0 * 1024.0)
    delta = [tcols * per_col for _ in range(int(n_stages))]
    delta[-1] += dcols * per_col
    line = (f"PP-CUT ROPE-KONTEXT (YaRN x2 27B): P context {ctx} (from {src}) > {CONTEXT_LENGTH_TOKENS} "
            f"(the posts' context) -> eager cos/sin cache +{extra_rows} rows x 4 B: target {tcols} cols "
            f"= +{tcols * per_col:.1f} MiB on every stage"
            + (f", draft {dname} {dcols} cols = +{dcols * per_col:.1f} MiB on the last stage" if dcols else "")
            + f"; stage_fixed += {[round(x, 1) for x in delta]} MiB")
    return tuple(delta), line
