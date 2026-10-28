"""Module-level helpers moved out of the source module (refactor, behaviour-neutral): bodies are byte-identical,
the source module re-exports every name (tests and callers keep ``<module>.<name>``). Logger name unchanged."""

from flliper.srt.pdflip.launcher_p_choose import p_rope_context_delta_mib

def p_rope_draft_delta_mib(ns) -> float:
    """YaRN x2 27B (29.09., heu5g3 W11b death): the DFlash draft's share of :func:`p_rope_context_delta_mib`
    -- the SAME function, read at two stages (target on both, the draft only on the last), so the W11 draft
    budget and the W11b build accounting price exactly the MiB the P pool floor already priced. 0.0 at or
    below 262144 (every x1 form byte-identical) and without a DFLASH draft."""
    delta, _line = p_rope_context_delta_mib(ns, 2)
    return max(0.0, float(delta[-1]) - float(delta[0]))
