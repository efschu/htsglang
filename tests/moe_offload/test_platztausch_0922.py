"""Platztausch (Nutzer-Entscheid 22.09. 22:20Z): Experten flippen NICHT als
27B-Gewichtsbild, sondern als geschachtelte Residenz -- der gemeinsame Praefix
wandert ueber den Austausch (Zeilenschnitt), alles andere liegt dauerhaft in
festen Store-Plaetzen und wird beim Wake lokal nachgeladen.

Hermetisch (CPU). Der Rundlauf unten spielt P->D->P mit echten Bytes durch:
jeder Experte e ist ein Tensor voller e, und nach jedem Flip muss jede
Puffer-Zeile genau den Experten tragen, den ihre Residenzliste nennt.
"""

import pytest
import torch

from sglang.srt.layers.moe import expert_map as em
from sglang.srt.layers.moe.expert_offload import (
    _refill_runs,
    plan_load_time_staging,
    resident_slot_count,
)

TOTAL = 512
RATIOS = [183, 137, 168]          # -> Spannen 192/144/176, Grenzen 0/192/336
SPLIT = [29, 11, 8]               # P-Layer je Stufe (fn7s)
LAYER_STAGE = [s for s, n in enumerate(SPLIT) for _ in range(n)]


def _karte(fr_pp, fr_tp):
    return em.build_nested(TOTAL, RATIOS, fr_pp, fr_tp, LAYER_STAGE, pad_tp=1)


# --- Karte -------------------------------------------------------------------


def test_p_gross_haelt_d_ganz_plus_extra():
    k = _karte([0.62, 1.0, 1.0], [0.55, 0.66, 0.55])
    d_all = sorted(g for ids in k["phases"]["D"]["resident"] for g in ids)
    for s, c in enumerate(k["phases"]["P"]["common"]):
        assert c == d_all, f"Stufe {s}"
        res = k["phases"]["P"]["resident"][s]
        assert res[: len(c)] == c                      # Praefix = common
        assert len(res) == resident_slot_count(TOTAL, [0.62, 1.0, 1.0][s])
    # Store = Komplement der Schnittmenge, in beiden Phasen dieselbe Abbildung
    assert k["slots"] == TOTAL - len(d_all)
    assert k["phases"]["P"]["slot_of"] == k["phases"]["D"]["slot_of"]
    assert em.refuse_if_inconsistent(k) is None
    assert em.nested_join_verdict(k) == []


def test_p_klein_nimmt_proportionalen_teil_und_d_traegt_extra():
    k = _karte([0.20, 1.0, 1.0], [0.55, 0.66, 0.55])
    c0 = k["phases"]["P"]["common"][0]
    assert len(c0) == resident_slot_count(TOTAL, 0.20)
    # jede D-Spanne ist vertreten (keine Breite 0 im Zeilenschnitt)
    for r, pre in enumerate(k["phases"]["D"]["prefix_by_stage"]):
        assert len(pre[0]) > 0, f"Rang {r} haette Breite 0"
    # D's Extra fuer Stufe-0-Layer liegt im Store
    for ext in k["phases"]["D"]["extra_by_stage"]:
        for g in ext[0]:
            assert em.slot_of(k, "D", g) is not None
    assert em.refuse_if_inconsistent(k) is None
    assert em.nested_join_verdict(k) == []


@pytest.mark.parametrize("span,f", [(144, 0.382), (176, 0.314), (192, 0.28),
                                    (192, 0.006)])
def test_pad_zaehlt_wie_der_rang(span, f):
    """w139 starb an EINER Id: die Karte zaehlte ohne Pad, der Rang mit."""
    ratios = [span, 512 - span]
    k = em.build_nested(TOTAL, ratios, [1.0], [f, 0.5], [0] * 4, pad_tp=1)
    real = len(k["phases"]["D"]["resident"][0])
    assert real + 1 == resident_slot_count(span + 1, f)


def test_rank_layout_adressiert_p_ueber_die_stufe():
    k = _karte([0.62, 1.0, 0.9], [0.55, 0.66, 0.55])
    for layer, s in ((0, 0), (29, 1), (47, 2)):
        pre, extra, res = em.rank_layout(k, "P", layer, 0)   # moe_tp_rank 0!
        assert res == k["phases"]["P"]["resident"][s]


# --- Puffer-Reihenfolge und Laeufe ----------------------------------------------


def test_plan_respektiert_die_reihenfolge_der_karte():
    order = [5, 6, 7, 0, 1, 2]
    plan = plan_load_time_staging(20, fraction=0.3, pinned_experts=order,
                                  resident_order=order)
    assert list(plan.resident_ids) == order
    assert not plan.is_static_layout
    with pytest.raises(ValueError):
        plan_load_time_staging(20, fraction=0.3, pinned_experts=order,
                               resident_order=[5, 6, 7, 0, 1, 9])


def test_refill_laeufe_werden_zusammengefasst():
    runs = _refill_runs([(10, -1), (11, 40), (12, 41), (13, 42), (14, 50)])
    assert runs == [(10, -1, 1), (11, 40, 3), (14, 50, 1)]


# --- Rundlauf P -> D -> P mit echten Bytes -------------------------------------


def _expert_bytes(e, cols=4):
    return torch.full((cols,), float(e))


def _store_fuer(k):
    """Der Store, wie beide Gruppen ihn beim Laden fuellen: jeder Experte mit
    Platz liegt DAUERHAFT in seinem Platz."""
    slot_of = k["phases"]["D"]["slot_of"]
    store = torch.zeros((k["slots"], 4))
    for g, p in slot_of.items():
        store[int(p)] = _expert_bytes(int(g))
    return store


def _p_puffer(k, layer):
    pre, extra, res = em.rank_layout(k, "P", layer, 0)
    return torch.stack([_expert_bytes(g) for g in res]), pre, res


def _d_puffer_leer(k, layer, r):
    pre, extra, res = em.rank_layout(k, "D", layer, r)
    return torch.zeros((len(pre) + 1 + len(extra), 4)), pre, extra


def _nachladen(buf, k, phase, extra, n_pre, pad, store):
    zeile = n_pre
    if pad:
        buf[zeile].zero_()
        zeile += 1
    for g in extra:
        buf[zeile] = store[int(k["phases"][phase]["slot_of"][str(g)])]
        zeile += 1


@pytest.mark.parametrize("fr_pp", [[0.62, 1.0, 1.0], [0.20, 1.0, 0.9]])
def test_rundlauf_p_d_p_jede_zeile_traegt_ihren_experten(fr_pp):
    k = _karte(fr_pp, [0.55, 0.66, 0.55])
    store = _store_fuer(k)
    for layer in (0, 28, 29, 40, 47):
        p_buf, p_pre, p_res = _p_puffer(k, layer)
        # P -> D: Zeilenschnitt des Praefix ueber die drei D-Raenge
        off = 0
        d_bufs = []
        for r in range(3):
            d_buf, d_pre, d_extra = _d_puffer_leer(k, layer, r)
            w = len(d_pre)
            d_buf[:w] = p_buf[off : off + w]
            off += w
            _nachladen(d_buf, k, "D", d_extra, w, True, store)
            erwartet = d_pre + [None] + d_extra
            for i, g in enumerate(erwartet):
                soll = torch.zeros(4) if g is None else _expert_bytes(g)
                assert torch.equal(d_buf[i], soll), (layer, r, i, g)
            d_bufs.append((d_buf, d_pre))
        assert off == len(p_pre)
        # D -> P: P waechst frisch auf (Muell), der Praefix kommt zurueck, das
        # Extra aus dem Store
        p_neu = torch.full_like(p_buf, -7.0)
        off = 0
        for d_buf, d_pre in d_bufs:
            p_neu[off : off + len(d_pre)] = d_buf[: len(d_pre)]
            off += len(d_pre)
        _nachladen(p_neu, k, "P", p_res[len(p_pre):], len(p_pre), False, store)
        assert torch.equal(p_neu, p_buf), layer


# --- Wake-Nachladen und Pool-Tabellen ------------------------------------------


def test_reinit_pool_tabellen_stellt_den_ausgangszustand_her():
    from sglang.srt.layers.moe.expert_pool_device import (
        allocate_pool_tables,
        reinit_pool_tables,
    )

    E, rows, R, staging = 10, 8, 3, 2
    hot = {0: 0, 4: 1, 7: 2}
    host = [-1 if e in hot else e for e in range(E)]
    t = allocate_pool_tables("cpu", E, rows, R, staging, hot, host)
    ref = {f: getattr(t, f).clone() for f in ("hot_phys", "host_row", "row_key",
                                              "row_use", "staging_rows")}
    for f in ref:
        getattr(t, f).fill_(-3)
    t.error.fill_(5)
    reinit_pool_tables(t, hot, host)
    for f, v in ref.items():
        assert torch.equal(getattr(t, f), v), f
    assert int(t.error[0]) == 0


def test_rearm_fuellt_pad_und_extra_ohne_installierten_cache():
    from sglang.srt.layers.moe.expert_offload import rearm_expert_offload_after_wake

    class Layer(torch.nn.Module):
        pass

    layer = Layer()
    buf = torch.full((6, 4), -1.0)          # 2 Praefix, Pad, 2 Extra, 1 Scratch
    store = torch.stack([_expert_bytes(100 + i) for i in range(5)])
    layer._moe_offload_presplit = {"w13_weight_packed": (buf, store)}
    layer._moe_offload_refill_runs = ((2, -1, 1), (3, 1, 2))
    model = torch.nn.Sequential(layer)
    lay, zeilen = rearm_expert_offload_after_wake(model)
    assert (lay, zeilen) == (1, 2)
    assert torch.equal(buf[2], torch.zeros(4))
    assert torch.equal(buf[3], _expert_bytes(101))
    assert torch.equal(buf[4], _expert_bytes(102))
    assert torch.equal(buf[0], torch.full((4,), -1.0))   # Praefix unberuehrt
