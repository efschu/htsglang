"""#128: der Planer muss denselben KV-Deckel kennen wie der Server.

fnFL2w60: der Server allokierte auf allen drei P-Raengen 269952 Tokens
(`_hybrid_kv_token_cap`, #79), der PP-Cut bepreiste 937950. Der Planer
reservierte Platz fuer das 3,5-Fache und refuste jede hoehere
Experten-Residenz (W40) -- bei 8,5-10,5 GB freiem VRAM je Karte.
"""
import types

import pytest

from sglang.srt.server_args import ServerArgs


def _sa(**kw):
    sa = ServerArgs.__new__(ServerArgs)
    sa.max_running_requests = kw.get("running", 1)
    sa.max_mamba_cache_size = kw.get("mamba", None)
    sa.context_length = kw.get("ctx", 262144)
    sa.model_config = kw.get("model_config", types.SimpleNamespace(mambaish_config=object()))
    sa.disable_radix_cache = kw.get("disable_radix", True)
    sa.speculative_num_draft_tokens = kw.get("draft", None)
    sa.speculative_algorithm = kw.get("spec", None)
    return sa


def test_dichtes_modell_bekommt_keinen_deckel():
    """Reine Attention-Pools duerfen running x ctx legitim ueberschreiten --
    sie tragen den Praefix-/Radix-Cache."""
    sa = _sa(model_config=types.SimpleNamespace(mambaish_config=None))
    assert sa._pp_cut_hybrid_pool_cap() is None
    assert _sa(model_config=None)._pp_cut_hybrid_pool_cap() is None


def test_hybrid_deckelt_auf_running_mal_kontext():
    cap = _sa(running=1, ctx=262144)._pp_cut_hybrid_pool_cap()
    assert cap is not None
    # der w60-Wert: 262144 + extra, einmal
    assert 262144 <= cap < 262144 * 2, cap


def test_der_deckel_faellt_nie_unter_den_des_servers():
    """DIE FALLE: der Server nimmt max(running, mamba_cache // ratio). Ohne
    den max()-Term plant der Planer zu wenig Pool, der Server allokiert
    mehr -- das endet im OOM statt in einer Refusal."""
    from sglang.srt.mem_cache.mamba_pool_floor import mamba_slots_per_running_req

    sa = _sa(running=1, mamba=64)
    ratio = max(int(mamba_slots_per_running_req(sa)), 1)
    erwartete_conc = max(1, 64 // ratio)
    ohne = _sa(running=1, mamba=None)._pp_cut_hybrid_pool_cap()
    mit = sa._pp_cut_hybrid_pool_cap()
    # STRIKT: der mamba-Term muss den Deckel um genau diesen Faktor heben.
    # Ein `min(concurrency, 1)` an der Stelle laesst `mit == ohne` und waere
    # ein Planer-Deckel UNTER dem des Servers -> OOM statt Refusal.
    assert mit == ohne * erwartete_conc, (
        f"mamba-Term wirkt nicht: mit={mit} ohne={ohne} conc={erwartete_conc}"
    )
    assert erwartete_conc > 1, "die Probe misst nichts -- ratio zu gross gewaehlt"


def test_ohne_nebenlaeufigkeit_oder_kontext_kein_deckel():
    assert _sa(running=0, mamba=None)._pp_cut_hybrid_pool_cap() is None
    assert _sa(running=1, ctx=0)._pp_cut_hybrid_pool_cap() is None


def test_der_aufrufer_nimmt_den_kleineren_wert():
    import inspect

    src = inspect.getsource(ServerArgs._handle_pp_solve_cut)
    code = "\n".join(z for z in src.split("\n") if not z.lstrip().startswith("#"))
    assert "_pp_cut_hybrid_pool_cap()" in code
    assert "_hyb < pool_tokens" in code, (
        "der Deckel darf den Pool nur SENKEN, nie anheben"
    )
