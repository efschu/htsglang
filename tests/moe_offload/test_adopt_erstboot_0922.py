"""#108 Erstboot-Adoption: die beiden Riegel, ohne die sie gefaehrlich waere.

Nutzer-Order 22.09. 07:25Z: D soll beim ersten Laden P's Layer-Bytes
nehmen statt den Checkpoint nochmal von Platte zu ziehen -- schneller,
und vor allem faellt jeder Fehler des Flip-Pfads dann in der ersten
Minute auf statt nach der fuenften.
"""

import pytest

from flliper.srt.pdflip import adopt


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Jeder Test startet ohne Adoption und ohne Platzhalter."""
    monkeypatch.delenv(adopt.ADOPT_ENV, raising=False)
    adopt._PLACEHOLDER["pending"] = False
    adopt._PLACEHOLDER["reason"] = ""
    yield
    adopt._PLACEHOLDER["pending"] = False
    adopt._PLACEHOLDER["reason"] = ""


def test_without_flag_no_adoption():
    assert adopt.adopt_armed() is False
    assert adopt.store_writes_denied() is False, (
        "ohne Adoption darf der normale Plattenweg schreiben wie immer")


def test_launcher_reads_argv_rank_reads_env(monkeypatch):
    """Zwei Leser, eine Antwort -- die Trennung, die S6 Fix E erzwungen hat."""
    assert adopt.adopt_armed(explicit="on") is True
    assert adopt.adopt_armed(explicit="off") is False
    monkeypatch.setenv(adopt.ADOPT_ENV, "on")
    assert adopt.adopt_armed() is True, "ein Rang hat kein argv"


# --- Riegel 1: kein dummy-Spill in den GETEILTEN Store -------------------

def test_under_adoption_d_does_not_write_shared_store(monkeypatch):
    """DER GEFAEHRLICHSTE FALL, und er ist still.

    Der Host-Store ist EINE Datei je Layer/Attribut fuer beide Gruppen
    (#107). Spillt D seinen dummy-Presplit hinein, ueberschreibt es P's
    echte Experten-Bytes mit Zufall -- Groesse und Struktur bleiben
    korrekt, nur der Inhalt ist zerstoert.
    """
    monkeypatch.setenv(adopt.ADOPT_ENV, "on")
    adopt.arm_placeholder()
    assert adopt.store_writes_denied() is True


def test_after_first_flip_d_may_write_again(monkeypatch):
    monkeypatch.setenv(adopt.ADOPT_ENV, "on")
    adopt.arm_placeholder()
    adopt.mark_adopted(filled=34, expected=34)
    assert adopt.store_writes_denied() is False, (
        "mit echten Bytes ist D ein normaler Schreiber")


# --- Riegel 2: keine Antwort auf Platzhaltern ---------------------------

def test_placeholders_refuse_generation():
    adopt.arm_placeholder("dummy-load")
    with pytest.raises(adopt.PdFlipAdoptWeightsArePlaceholder) as exc:
        adopt.refuse_if_placeholder()
    assert adopt.REFUSAL_MARKER in str(exc.value)


def test_half_filled_inject_does_not_release_guard():
    """Ein halb gefuelltes Modell ist gefaehrlicher als ein leeres --
    es rechnet. Die Halter-Karte sagt, wieviele Tensoren erwartet sind."""
    adopt.arm_placeholder()
    adopt.mark_adopted(filled=33, expected=34)
    assert adopt.weights_are_placeholder() is True
    assert "33 von 34" in adopt.placeholder_reason()
    with pytest.raises(adopt.PdFlipAdoptWeightsArePlaceholder):
        adopt.refuse_if_placeholder()


def test_zero_expected_tensors_is_not_success():
    """Sonst wuerde ein Inject, der GAR NICHTS fand, den Riegel loesen --
    dieselbe Klasse wie NULL-NUR-BEI-ERREICHTEM-EMITTER."""
    adopt.arm_placeholder()
    adopt.mark_adopted(filled=0, expected=0)
    assert adopt.weights_are_placeholder() is True


def test_complete_inject_releases_the_guard():
    adopt.arm_placeholder()
    adopt.mark_adopted(filled=34, expected=34)
    assert adopt.weights_are_placeholder() is False
    adopt.refuse_if_placeholder()  # wirft nicht mehr


def test_without_adoption_nothing_is_placeholder():
    """Der normale Plattenboot darf von alldem nichts merken."""
    assert adopt.weights_are_placeholder() is False
    adopt.refuse_if_placeholder()


# --- Der Riegel AM SCHREIBPFAD, nicht nur als Funktion ------------------

def test_write_rows_under_adoption_writes_nothing(monkeypatch):
    """Die Naht: der Riegel muss IM Schreibpfad sitzen, nicht daneben.

    Genau diese Unterscheidung hat am 22.09. drei Boots gekostet (#106:
    Leser ohne Schreiber) und einen weiteren (#107/2: Vorspann statt
    Zweig). Der Test faehrt deshalb `write_rows` selbst und prueft die
    BYTES, nicht die Absicht.
    """
    import torch

    from flliper.srt.layers.moe import expert_store as es

    store = torch.zeros((8, 4), dtype=torch.int8)
    src = torch.full((3, 4), 7, dtype=torch.int8)

    monkeypatch.setenv(adopt.ADOPT_ENV, "on")
    adopt.arm_placeholder()
    written = es.write_rows(store, src, local_ids=[0, 1, 2], lo=0, pad=False)
    assert store.sum().item() == 0, (
        "unter Adoption darf KEIN Byte in den geteilten Store -- sonst "
        "ueberschreibt D's dummy-Presplit P's echte Experten")
    assert written == {}

    adopt.mark_adopted(filled=3, expected=3)
    es.write_rows(store, src, local_ids=[0, 1, 2], lo=0, pad=False)
    assert store.sum().item() > 0, (
        "nach dem Erstflip ist D ein normaler Schreiber")


def test_without_adoption_write_rows_writes_as_before():
    """Der normale Plattenboot darf von #108 nichts merken."""
    import torch

    from flliper.srt.layers.moe import expert_store as es

    store = torch.zeros((8, 4), dtype=torch.int8)
    src = torch.full((3, 4), 5, dtype=torch.int8)
    es.write_rows(store, src, local_ids=[0, 1, 2], lo=0, pad=False)
    assert store.sum().item() > 0


def test_argv_d_gets_dummy_only_under_adoption():
    """ERSETZT die w52-Fassung, die GRUEN WAR und NICHTS BEWIES.

    Sie setzte die Env und rief ``_adopt_load_format_flag()`` ohne
    Argument -- genau der Aufruf, der im Launcher-Prozess immer ``[]``
    liefert. Der Test gruen, das argv leer, D las von Platte. Jetzt geht
    der Wert als PARAMETER hinein, und der Test prueft den Rueckgabewert.
    """
    from flliper.srt.pdflip import launcher as lx

    assert lx._adopt_load_format_flag(True) == ["--load-format", "dummy"]
    assert lx._adopt_load_format_flag(False) == []


# --- Der TRIGGER: das Flip-Paar vor dem ersten Request ------------------

def test_front_runs_a_flip_pair_not_one():
    """P->D holt die Bytes, D->P stellt die Rollen wieder her.

    Ohne den zweiten Halbschritt startete das Serving mit vertauschten
    Rollen: D waere wach und P schliefe, obwohl P prefillt.
    """
    import inspect

    from flliper.srt.pdflip import front

    src = inspect.getsource(front.Front._adopt_first_flip)
    assert src.index('self.flip("P", "D")') < src.index('self.flip("D", "P")'), (
        "erst P->D (Bytes holen), dann D->P (Rollen zurueck)")


def test_trigger_runs_before_serving_loop():
    """Sonst kaeme der erste Request auf Platzhalter-Gewichten an."""
    import inspect

    from flliper.srt.pdflip import front

    src = inspect.getsource(front.Front.controller)
    assert "_adopt_first_flip" in src
    assert src.index("_adopt_first_flip") < src.index("while True"), (
        "der Erstflip gehoert vor die Serving-Schleife")


def test_trigger_without_flag_is_a_no_op():
    """Der normale Boot darf kein zusaetzliches Flip-Paar fahren."""
    import inspect

    from flliper.srt.pdflip import front

    src = inspect.getsource(front.Front._adopt_first_flip)
    i_guard = src.index("adopt_armed()")
    i_flip = src.index('self.flip("P", "D")')
    assert i_guard < i_flip, "erst pruefen, dann flippen"
    assert "return" in src[i_guard:i_flip], "ohne Flag sofort zurueck"


def test_no_try_except_around_flip_pair():
    """Schlaegt die Adoption fehl, MUSS der Boot stehenbleiben.

    Ein verschlucktes Scheitern ergaebe einen Boot, der laeuft und auf
    jede Anfrage den Forward-Riegel wirft -- er saehe gesund aus und
    beantwortete nichts.
    """
    import inspect

    from flliper.srt.pdflip import front

    src = inspect.getsource(front.Front._adopt_first_flip)
    body = src[src.index('self.flip("P", "D")'):]
    assert "except" not in body


# --- Die Kette Inject -> Deckung -> Riegel ------------------------------

def test_coverage_is_recorded_where_it_is_known():
    """#106 und #107/2 sind daran gescheitert, dass eine Zahl zweimal
    abgeleitet wurde. Die Deckung wird deshalb im Inject GEMERKT und im
    Router nur GELESEN."""
    import inspect

    from flliper.srt.managers.scheduler_components import weight_updater as wu

    # `_pdflip_xchg_inject_from_peer` ist die Methode, die `plan.descs` und
    # die getragenen `_cdescs` BEIDE in der Hand hat -- nicht
    # `_pdflip_xchg_inject_weights`, das nur der Einstieg ist. Der erste Lauf
    # dieses Tests hat genau diese Verwechslung aufgedeckt.
    inject = inspect.getsource(wu.SchedulerWeightUpdaterManager._pdflip_xchg_inject_from_peer)
    assert "_pdflip_last_inject_cover" in inject, (
        "die Deckung muss dort festgehalten werden, wo plan.descs und die "
        "getragenen Descriptors beide bekannt sind")
    cover = inspect.getsource(wu.SchedulerWeightUpdaterManager._pdflip_adopt_cover)
    assert "_pdflip_last_inject_cover" in cover and "plan" not in cover, (
        "der Leser darf sie NICHT neu ableiten")


def test_undeterminable_coverage_keeps_the_guard():
    """`(0, 0)` ist kein Erfolg -- sonst oeffnete ein Inject, der gar
    nichts fand, das Modell fuer Zufallszahlen."""
    adopt.arm_placeholder()
    adopt.mark_adopted(0, 0)
    assert adopt.weights_are_placeholder() is True


# --- DER TEST, DER IN w52 GEFEHLT HAT -------------------------------------
#
# 18 Tests waren gruen, und im argv des Boots stand `--load-format dummy`
# NULL mal: sie banden den HELFER und die LOG-ZEILE, nie das argv. Der
# Launcher schrieb "D startet mit --load-format dummy", waehrend D mit
# `load_format='auto'` 288 Presplit-Layer von Platte las. Diese drei Tests
# binden das ERGEBNIS: was argv_d zurueckgibt, und dass die Produktions-
# Aufrufstelle die Entscheidung mitgibt statt sie raten zu lassen.


def _argv_d_minimal(**kw):
    from flliper.srt.pdflip import launcher as lx

    return lx.argv_d(
        "py", "/m", [1, 1, 1], 1, 1, lx.RING_FORM_SENTINEL_STORE_CFG, [], **kw
    )


def test_argv_d_has_dummy_when_adoption_armed():
    argv = _argv_d_minimal(d_adopt=True)
    assert "--load-format" in argv
    assert argv[argv.index("--load-format") + 1] == "dummy"


def test_argv_d_has_no_dummy_without_adoption():
    assert "dummy" not in _argv_d_minimal(d_adopt=False)


def test_argv_d_reads_no_env(monkeypatch):
    # w52s Wurzel in einem Satz: die Env des LAUNCHER-Prozesses ist nicht
    # die der Kinder. Ein Helfer, der sie liest, gibt im Launcher immer [].
    monkeypatch.setenv("FLLIPER_PDFLIP_D_ADOPT", "on")
    assert "dummy" not in _argv_d_minimal(d_adopt=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_D_ADOPT", raising=False)
    assert "dummy" in _argv_d_minimal(d_adopt=True)


def test_production_call_site_passes_the_decision():
    import inspect

    from flliper.srt.pdflip import launcher as lx

    src = inspect.getsource(lx)
    bauer = [z for z in src.split("\n") if "argv_d(py, ns.model" in z]
    assert bauer, "keine Produktions-Aufrufstelle von argv_d gefunden"
    for z in bauer:
        assert "d_adopt=" in z, (
            "eine argv_d-Aufrufstelle gibt d_adopt NICHT mit -- dann faellt "
            "sie auf den Default False und der Flag verschwindet still: "
            + z.strip()[:120]
        )
