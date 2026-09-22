"""#108 Erstboot-Adoption: die beiden Riegel, ohne die sie gefaehrlich waere.

Nutzer-Order 22.09. 07:25Z: D soll beim ersten Laden P's Layer-Bytes
nehmen statt den Checkpoint nochmal von Platte zu ziehen -- schneller,
und vor allem faellt jeder Fehler des Flip-Pfads dann in der ersten
Minute auf statt nach der fuenften.
"""

import pytest

from sglang.srt.weg2 import adopt


@pytest.fixture(autouse=True)
def _sauber(monkeypatch):
    """Jeder Test startet ohne Adoption und ohne Platzhalter."""
    monkeypatch.delenv(adopt.ADOPT_ENV, raising=False)
    adopt._PLACEHOLDER["pending"] = False
    adopt._PLACEHOLDER["reason"] = ""
    yield
    adopt._PLACEHOLDER["pending"] = False
    adopt._PLACEHOLDER["reason"] = ""


def test_ohne_flag_keine_adoption():
    assert adopt.adopt_armed() is False
    assert adopt.store_writes_denied() is False, (
        "ohne Adoption darf der normale Plattenweg schreiben wie immer")


def test_der_launcher_liest_sein_argv_der_rang_die_env(monkeypatch):
    """Zwei Leser, eine Antwort -- die Trennung, die S6 Fix E erzwungen hat."""
    assert adopt.adopt_armed(explicit="on") is True
    assert adopt.adopt_armed(explicit="off") is False
    monkeypatch.setenv(adopt.ADOPT_ENV, "on")
    assert adopt.adopt_armed() is True, "ein Rang hat kein argv"


# --- Riegel 1: kein dummy-Spill in den GETEILTEN Store -------------------

def test_unter_adoption_schreibt_D_NICHT_in_den_geteilten_store(monkeypatch):
    """DER GEFAEHRLICHSTE FALL, und er ist still.

    Der Host-Store ist EINE Datei je Layer/Attribut fuer beide Gruppen
    (#107). Spillt D seinen dummy-Presplit hinein, ueberschreibt es P's
    echte Experten-Bytes mit Zufall -- Groesse und Struktur bleiben
    korrekt, nur der Inhalt ist zerstoert.
    """
    monkeypatch.setenv(adopt.ADOPT_ENV, "on")
    adopt.arm_placeholder()
    assert adopt.store_writes_denied() is True


def test_nach_dem_erstflip_darf_D_wieder_schreiben(monkeypatch):
    monkeypatch.setenv(adopt.ADOPT_ENV, "on")
    adopt.arm_placeholder()
    adopt.mark_adopted(filled=34, expected=34)
    assert adopt.store_writes_denied() is False, (
        "mit echten Bytes ist D ein normaler Schreiber")


# --- Riegel 2: keine Antwort auf Platzhaltern ---------------------------

def test_platzhalter_verweigern_die_generierung():
    adopt.arm_placeholder("dummy-load")
    with pytest.raises(adopt.Weg2AdoptWeightsArePlaceholder) as exc:
        adopt.refuse_if_placeholder()
    assert adopt.REFUSAL_MARKER in str(exc.value)


def test_ein_HALB_gefuellter_inject_loest_den_riegel_NICHT():
    """Ein halb gefuelltes Modell ist gefaehrlicher als ein leeres --
    es rechnet. Die Halter-Karte sagt, wieviele Tensoren erwartet sind."""
    adopt.arm_placeholder()
    adopt.mark_adopted(filled=33, expected=34)
    assert adopt.weights_are_placeholder() is True
    assert "33 von 34" in adopt.placeholder_reason()
    with pytest.raises(adopt.Weg2AdoptWeightsArePlaceholder):
        adopt.refuse_if_placeholder()


def test_null_erwartete_tensoren_sind_kein_erfolg():
    """Sonst wuerde ein Inject, der GAR NICHTS fand, den Riegel loesen --
    dieselbe Klasse wie NULL-NUR-BEI-ERREICHTEM-EMITTER."""
    adopt.arm_placeholder()
    adopt.mark_adopted(filled=0, expected=0)
    assert adopt.weights_are_placeholder() is True


def test_vollstaendiger_inject_loest_den_riegel():
    adopt.arm_placeholder()
    adopt.mark_adopted(filled=34, expected=34)
    assert adopt.weights_are_placeholder() is False
    adopt.refuse_if_placeholder()  # wirft nicht mehr


def test_ohne_adoption_ist_nichts_platzhalter():
    """Der normale Plattenboot darf von alldem nichts merken."""
    assert adopt.weights_are_placeholder() is False
    adopt.refuse_if_placeholder()


# --- Der Riegel AM SCHREIBPFAD, nicht nur als Funktion ------------------

def test_write_rows_schreibt_unter_adoption_wirklich_nichts(monkeypatch):
    """Die Naht: der Riegel muss IM Schreibpfad sitzen, nicht daneben.

    Genau diese Unterscheidung hat am 22.09. drei Boots gekostet (#106:
    Leser ohne Schreiber) und einen weiteren (#107/2: Vorspann statt
    Zweig). Der Test faehrt deshalb `write_rows` selbst und prueft die
    BYTES, nicht die Absicht.
    """
    import torch

    from sglang.srt.layers.moe import expert_store as es

    store = torch.zeros((8, 4), dtype=torch.int8)
    src = torch.full((3, 4), 7, dtype=torch.int8)

    monkeypatch.setenv(adopt.ADOPT_ENV, "on")
    adopt.arm_placeholder()
    geschrieben = es.write_rows(store, src, local_ids=[0, 1, 2], lo=0, pad=False)
    assert store.sum().item() == 0, (
        "unter Adoption darf KEIN Byte in den geteilten Store -- sonst "
        "ueberschreibt D's dummy-Presplit P's echte Experten")
    assert geschrieben == {}

    adopt.mark_adopted(filled=3, expected=3)
    es.write_rows(store, src, local_ids=[0, 1, 2], lo=0, pad=False)
    assert store.sum().item() > 0, (
        "nach dem Erstflip ist D ein normaler Schreiber")


def test_ohne_adoption_schreibt_write_rows_wie_immer():
    """Der normale Plattenboot darf von #108 nichts merken."""
    import torch

    from sglang.srt.layers.moe import expert_store as es

    store = torch.zeros((8, 4), dtype=torch.int8)
    src = torch.full((3, 4), 5, dtype=torch.int8)
    es.write_rows(store, src, local_ids=[0, 1, 2], lo=0, pad=False)
    assert store.sum().item() > 0


def test_argv_d_bekommt_dummy_nur_unter_adoption(monkeypatch):
    """Der Plattenboot darf von #108 nichts merken -- kein Flag, keine
    Aenderung an D's Kommandozeile."""
    from sglang.srt.weg2.launcher import _adopt_load_format_flag

    monkeypatch.delenv(adopt.ADOPT_ENV, raising=False)
    assert _adopt_load_format_flag() == []
    monkeypatch.setenv(adopt.ADOPT_ENV, "on")
    assert _adopt_load_format_flag() == ["--load-format", "dummy"]


# --- Der TRIGGER: das Flip-Paar vor dem ersten Request ------------------

def test_die_front_faehrt_ein_flip_PAAR_nicht_nur_einen():
    """P->D holt die Bytes, D->P stellt die Rollen wieder her.

    Ohne den zweiten Halbschritt startete das Serving mit vertauschten
    Rollen: D waere wach und P schliefe, obwohl P prefillt.
    """
    import inspect

    from sglang.srt.weg2 import front

    src = inspect.getsource(front.Front._adopt_first_flip)
    assert src.index('self.flip("P", "D")') < src.index('self.flip("D", "P")'), (
        "erst P->D (Bytes holen), dann D->P (Rollen zurueck)")


def test_der_trigger_laeuft_VOR_dem_serving_loop():
    """Sonst kaeme der erste Request auf Platzhalter-Gewichten an."""
    import inspect

    from sglang.srt.weg2 import front

    src = inspect.getsource(front.Front.controller)
    assert "_adopt_first_flip" in src
    assert src.index("_adopt_first_flip") < src.index("while True"), (
        "der Erstflip gehoert vor die Serving-Schleife")


def test_der_trigger_ist_ohne_flag_ein_no_op():
    """Der normale Boot darf kein zusaetzliches Flip-Paar fahren."""
    import inspect

    from sglang.srt.weg2 import front

    src = inspect.getsource(front.Front._adopt_first_flip)
    i_guard = src.index("adopt_armed()")
    i_flip = src.index('self.flip("P", "D")')
    assert i_guard < i_flip, "erst pruefen, dann flippen"
    assert "return" in src[i_guard:i_flip], "ohne Flag sofort zurueck"


def test_kein_try_except_um_das_flip_paar():
    """Schlaegt die Adoption fehl, MUSS der Boot stehenbleiben.

    Ein verschlucktes Scheitern ergaebe einen Boot, der laeuft und auf
    jede Anfrage den Forward-Riegel wirft -- er saehe gesund aus und
    beantwortete nichts.
    """
    import inspect

    from sglang.srt.weg2 import front

    src = inspect.getsource(front.Front._adopt_first_flip)
    koerper = src[src.index('self.flip("P", "D")'):]
    assert "except" not in koerper


# --- Die Kette Inject -> Deckung -> Riegel ------------------------------

def test_die_deckung_wird_dort_festgehalten_wo_sie_bekannt_ist():
    """#106 und #107/2 sind daran gescheitert, dass eine Zahl zweimal
    abgeleitet wurde. Die Deckung wird deshalb im Inject GEMERKT und im
    Router nur GELESEN."""
    import inspect

    from sglang.srt.managers.scheduler_components import weight_updater as wu

    # `_weg2_xchg_inject_from_peer` ist die Methode, die `plan.descs` und
    # die getragenen `_cdescs` BEIDE in der Hand hat -- nicht
    # `_weg2_xchg_inject_weights`, das nur der Einstieg ist. Der erste Lauf
    # dieses Tests hat genau diese Verwechslung aufgedeckt.
    inject = inspect.getsource(wu.SchedulerWeightUpdaterManager._weg2_xchg_inject_from_peer)
    assert "_weg2_last_inject_cover" in inject, (
        "die Deckung muss dort festgehalten werden, wo plan.descs und die "
        "getragenen Descriptors beide bekannt sind")
    cover = inspect.getsource(wu.SchedulerWeightUpdaterManager._weg2_adopt_cover)
    assert "_weg2_last_inject_cover" in cover and "plan" not in cover, (
        "der Leser darf sie NICHT neu ableiten")


def test_unermittelbare_deckung_laesst_den_riegel_stehen():
    """`(0, 0)` ist kein Erfolg -- sonst oeffnete ein Inject, der gar
    nichts fand, das Modell fuer Zufallszahlen."""
    adopt.arm_placeholder()
    adopt.mark_adopted(0, 0)
    assert adopt.weights_are_placeholder() is True
