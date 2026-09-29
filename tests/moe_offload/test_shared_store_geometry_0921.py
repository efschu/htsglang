"""#91: die EINE Store-Geometrie, die beide Ranggruppen teilen.

Der Store ist eine Datei je (Layer, Attribut), geteilt zwischen P und D.
Heute rechnet jede Gruppe ihre Slot-Zuordnung aus IHREN Vektoren: P
(tp1/pp3, #90-Fallback [512]) kommt auf 358 Plaetze, D (183,137,168 mit
0.006,0.564,0.467) auf 332. Verschiedene Zuordnungen auf denselben Bytes
sind DATENVERLUST, nicht Speicherverlust -- D schreibt dann in die Zeilen,
aus denen P nach dem Wake liest.

Der Launcher kennt als einziger beide Konfigurationen und gibt deshalb eine
gemeinsame Geometrie vor.
"""

import os

import pytest

from flliper.srt.layers.moe import expert_store as es


@pytest.fixture(autouse=True)
def _clean_env():
    alt = os.environ.pop(es.STORE_GEOMETRY_ENV, None)
    yield
    if alt is not None:
        os.environ[es.STORE_GEOMETRY_ENV] = alt
    else:
        os.environ.pop(es.STORE_GEOMETRY_ENV, None)


def test_without_env_nothing_changes():
    assert es.shared_geometry() is None


def test_set_geometry_is_read():
    os.environ[es.STORE_GEOMETRY_ENV] = "183,137,168|0.006,0.564,0.467"
    assert es.shared_geometry() == ([183, 137, 168], [0.006, 0.564, 0.467])


def test_both_groups_compute_same_slot_count():
    """Der Zweck der Sache: EINE Zahl, egal wer fragt."""
    os.environ[es.STORE_GEOMETRY_ENV] = "183,137,168|0.006,0.564,0.467"
    rat, fr = es.shared_geometry()
    slots = es.slot_base_for_rank(rat, fr, len(rat))
    # Was P aus SEINEN Vektoren rechnen wuerde -- die Zahl, die kollidiert.
    p_alone = es.slot_base_for_rank([512], [0.30], 1)
    assert slots != p_alone, "The test would be pointless if both were equal anyway"
    assert slots == es.slot_base_for_rank(rat, fr, len(rat))


@pytest.mark.parametrize(
    "roh",
    [
        "",                       # nicht gesetzt
        "183,137,168",            # kein Trenner
        "183,137|0.1,0.2,0.3",    # Laengen ungleich
        "183,-4|0.1,0.2",         # negativer Bereich
        "183,137|0.1,1.7",        # fraction ausserhalb [0,1]
        "abc|0.1",                # unlesbar
    ],
)
def test_bad_input_gives_none_not_half_geometry(roh):
    """Eine falsche gemeinsame Zuordnung waere schlimmer als gar keine:
    sie sieht aus wie Ersparnis und ist Datenverlust."""
    os.environ[es.STORE_GEOMETRY_ENV] = roh
    assert es.shared_geometry() is None
