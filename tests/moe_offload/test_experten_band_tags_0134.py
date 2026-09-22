"""#134: ein Experten-Band ist ein Layer-Band, feiner geschnitten.

Der 27B-Flip verschiebt ganze Layer ueber Chunk-Tags (`weights_<n>`) und
pausiert/resumiert sie je Tag. Nutzer-Order 22.09.: *"wir verschieben nicht
nur ganze layer sondern auch ganze experten (aber das sind ja auch nur
layer)"*. Diese Datei haelt fest, dass der feinere Tag in JEDER Naht
mitgefuehrt wird, die den groeberen schon kennt -- und dass er ohne die Env
NICHTS aendert.

Die vier Naehte, jede mit ihrer eigenen Todesart, falls sie fehlt:
  * `is_weights_chunk_tag`      -> sonst W74 Weg2XchgSourceMissing je Tensor
  * `chunk_tag_cards`           -> sonst `missing`-Zweig, Identitaets-Ordnung
  * `weights_family_tags`       -> sonst kennt keine Welle den Tag
  * `derive_waves`              -> sonst eine Welle JE BAND (gemessener Preis
                                   +25,8..+36,8 % laut DESIGN §1.2)
"""
import os

import pytest

from sglang.srt.managers import weg2_memory_saver as m
from sglang.srt.weg2.weight_exchange import derive_waves


@pytest.fixture
def chunks(monkeypatch):
    monkeypatch.setenv(m.WEIGHT_CHUNK_ENV_LAYERS, "3")
    monkeypatch.setenv(m.WEIGHT_CHUNK_ENV_COUNT, "4")
    monkeypatch.delenv(m.EXPERT_BAND_ENV_SIZE, raising=False)
    monkeypatch.delenv(m.EXPERT_BAND_ENV_COUNT, raising=False)


@pytest.fixture
def baender(chunks, monkeypatch):
    monkeypatch.setenv(m.EXPERT_BAND_ENV_SIZE, "64")
    monkeypatch.setenv(m.EXPERT_BAND_ENV_COUNT, "8")


# --- ohne die Env aendert sich NICHTS -------------------------------------

def test_ohne_env_ist_alles_wie_vorher(chunks):
    assert m.weights_family_tags() == [
        "weights_0", "weights_1", "weights_2", "weights_3", "weights"
    ]
    assert m.weight_chunk_tag(7) == "weights_2"
    assert m.weight_chunk_tag(7, 200) == "weights_2", (
        "ohne Bandteilung muss eine expert_id den Layer-Tag liefern -- sonst "
        "traegt jeder Aufrufer, der sie durchreicht, einen Tag, den die "
        "Familie nicht kennt"
    )
    assert m.expert_band_geometry() == (0, 0)


def test_halbe_geometrie_ist_AUS_nicht_geraten(chunks, monkeypatch):
    # Nur die Breite, ohne die Anzahl: ein Schreiber ohne Leser. Die Tags
    # entstuenden, aber weights_family_tags zaehlte sie nie auf -> W74.
    monkeypatch.setenv(m.EXPERT_BAND_ENV_SIZE, "64")
    assert m.expert_band_geometry() == (0, 0)
    assert m.weight_chunk_tag(7, 200) == "weights_2"
    monkeypatch.delenv(m.EXPERT_BAND_ENV_SIZE)
    monkeypatch.setenv(m.EXPERT_BAND_ENV_COUNT, "8")
    assert m.expert_band_geometry() == (0, 0)
    assert m.weight_chunk_tag(7, 200) == "weights_2"


# --- der Tag selbst --------------------------------------------------------

def test_band_tag_wird_gebildet(baender):
    assert m.weight_chunk_tag(7, 200) == "weights_2_e3"   # 200 // 64 == 3
    assert m.weight_chunk_tag(7) == "weights_2"           # der Rest des Layers
    assert m.weight_chunk_tag(0, 0) == "weights_0_e0"


def test_id_jenseits_des_letzten_bandes_wird_geklemmt(baender):
    # 8 Baender a 64 = 512; eine Id darueber darf keinen Tag erfinden, den
    # die Familie nicht aufzaehlt -- das waere W74 beim ersten Flip.
    assert m.weight_chunk_tag(7, 9999) == "weights_2_e7"
    assert m.weight_chunk_tag(7, 9999) in m.weights_family_tags()


# --- Naht 1: die Praedikate ------------------------------------------------

def test_band_ist_chunk_und_familie(baender):
    assert m.is_weights_chunk_tag("weights_2_e3")
    assert m.is_weights_family_tag("weights_2_e3")
    assert m.chunk_of_band_tag("weights_2_e3") == "weights_2"
    assert m.expert_band_of_tag("weights_2_e3") == 3
    assert m.chunk_of_band_tag("weights_2") == "weights_2"
    assert m.expert_band_of_tag("weights_2") is None


def test_der_praefix_test_bleibt_verboten(baender):
    # Die Klasse, fuer die das Integer-Praedikat ueberhaupt geschrieben wurde:
    # weights_draft teilt den Praefix und gehoert NICHT in die Chunk-Familie.
    assert not m.is_weights_chunk_tag("weights_draft")
    assert not m.is_weights_chunk_tag("weights_2_ex")
    assert not m.is_weights_chunk_tag("weights_2_e")
    assert not m.is_weights_chunk_tag("weights_2_e3_x")
    assert m.chunk_of_band_tag("weights_draft") is None


# --- Naht 2: die Kartenzuordnung ------------------------------------------

def test_band_erbt_die_karte_seines_chunks(baender):
    karten = m.chunk_tag_cards([6, 3, 3], 3, 4, [0, 1, 2])
    for tag, cs in karten.items():
        chunk = m.chunk_of_band_tag(tag)
        assert cs == karten[chunk], f"{tag} liegt auf {cs}, sein Chunk auf {karten[chunk]}"
    assert "weights_0_e7" in karten, (
        "ohne Karten-Eintrag nimmt interleave_pause_order den missing-Zweig "
        "und JEDER Flip faellt auf die Identitaets-Ordnung zurueck"
    )


# --- Naht 3: die Familie ---------------------------------------------------

def test_familie_traegt_jedes_band_und_die_basis_bleibt_letzte(baender):
    tags = m.weights_family_tags()
    assert tags[-1] == m.GPU_MEMORY_TYPE_WEIGHTS, (
        "derive_waves liest tags[-1] als Basis-Tag, der die letzte Welle "
        "schliesst -- ein Band dort wuerde IHN zur Basis machen"
    )
    for k in range(4):
        for b in range(8):
            assert f"weights_{k}_e{b}" in tags
        assert f"weights_{k}" in tags
        # Baender VOR ihrem Chunk: die Ordnung ist die Freigabe-Ordnung.
        assert tags.index(f"weights_{k}_e7") < tags.index(f"weights_{k}")
    assert len(tags) == 4 * 9 + 1


def test_jeder_gebildete_tag_ist_in_der_familie(baender):
    tags = set(m.weights_family_tags())
    for layer in range(12):
        assert m.weight_chunk_tag(layer) in tags
        for eid in (0, 63, 64, 200, 511):
            assert m.weight_chunk_tag(layer, eid) in tags, (layer, eid)


# --- Naht 4: die Wellen (der Regress-Test) --------------------------------

def _wellen(bands: bool):
    karten = m.chunk_tag_cards([6, 3, 3], 3, 4, [0, 1, 2])
    return derive_waves(m.weights_family_tags(), karten, [0, 1, 2])


def test_baender_aendern_die_wellenzahl_nicht(chunks, monkeypatch):
    ohne = _wellen(False)
    monkeypatch.setenv(m.EXPERT_BAND_ENV_SIZE, "64")
    monkeypatch.setenv(m.EXPERT_BAND_ENV_COUNT, "8")
    mit = _wellen(True)
    assert len(mit) == len(ohne) == 2, (
        f"{len(ohne)} Wellen ohne Baender, {len(mit)} mit. Ein Band bringt "
        f"KEINE neue Karte; eine Welle je Band kostet laut DESIGN §1.2 "
        f"+25,8..+36,8 % Transport."
    )


def test_ein_band_faehrt_in_der_welle_seines_chunks(baender):
    wellen = _wellen(True)
    welle_von = {t: i for i, w in enumerate(wellen) for t in w}
    for tag, i in welle_von.items():
        chunk = m.chunk_of_band_tag(tag)
        if chunk is None or chunk == tag:
            continue
        assert welle_von[chunk] == i, f"{tag} in Welle {i}, {chunk} in {welle_von[chunk]}"


def test_die_wellen_sind_eine_permutation_der_familie(baender):
    wellen = _wellen(True)
    flach = [t for w in wellen for t in w]
    assert sorted(flach) == sorted(m.weights_family_tags()), (
        "build_plan verlangt eine PERMUTATION: ein Tag in zwei Wellen ist "
        "W68, ein Tag in keiner ist W74"
    )
    assert wellen[-1][-1] == m.GPU_MEMORY_TYPE_WEIGHTS


# --- die Bandbreite kommt aus der GEOMETRIE, nicht aus dem Arm ------------

def test_bandbreite_trifft_jede_D_grenze():
    from sglang.srt.layers.moe.expert_map import band_geometry, bounds, scaled_spans

    spans = scaled_spans([183, 137, 168], 512)
    assert spans == [192, 144, 176]
    size, count = band_geometry([183, 137, 168], 512)
    assert (size, count) == (16, 32)
    # JEDE Besitzgrenze von D faellt auf eine Bandgrenze -- sonst haette ein
    # Band zwei Halter, und genau daran ist die Karte heute frueh gescheitert
    # ("298 Ids sind resident UND im Store").
    for grenze in bounds(spans) + [512]:
        assert grenze % size == 0, (grenze, size)


def test_unbrauchbare_ratios_schalten_die_teilung_AUS():
    from sglang.srt.layers.moe.expert_map import band_geometry

    # ggT 1 -> 512 Baender je Chunk. Die Ausweichrichtung ist LANGSAM
    # (Transport je Layer-Chunk wie vorher), nie falsch.
    assert band_geometry([1, 1, 1], 512) == (0, 0)
    assert band_geometry([5, 3, 1], 512) == (0, 0)
    assert band_geometry([], 512) == (0, 0)
    assert band_geometry([183, 137, 168], 0) == (0, 0)
    # und eine grobe Teilung bleibt erlaubt
    assert band_geometry([2, 1, 1], 512) == (128, 4)
