"""#99: der Disarm-Pfad des Abort-Gates nennt eine Variable, die es nicht gibt.

fnFL2w30, Gruppe D, TP0 -- der erste Boot, der ueberhaupt so weit kam:

    File ".../barlink_abort_gate.py", line 468, in poll_status_words
        _disarm = getattr(t, "_abort_poll_disarm", None)
    NameError: name 't' is not defined
    Fatal Python error: Segmentation fault

Die Schleife heisst `for transport in registered()`; an dieser einen
Stelle steht `t`. Der Block ist der #1330-Fix gegen genau diese Klasse --
sein Kommentar nennt sie "fault-amplifier ... three sites, one fault, and
two of them innocent" und beschreibt einen Boot, der "in the driver a
moment later with a segfault" starb. Der Fix hat dieselbe Form
reproduziert, die er verhindern sollte: der NameError im Fehlerpfad
verhinderte das Disarm, der naechste Poll vergiftete den Kontext, und TP0
starb im Segfault statt an der Ausnahme.

Der Pfad ist erst erreichbar, wenn ein Poll WIRKLICH scheitert -- deshalb
hat ihn kein Test und kein Boot je betreten.
"""

import pytest

from flliper.srt.distributed.device_communicators import barlink_abort_gate as gate


class _BrokenTransport:
    """Sein Poll wirft -- genau der Fall, fuer den der Disarm gebaut ist."""

    def __init__(self):
        self.disarmed_mit = None

    def poll_status_word(self):
        raise RuntimeError("die Statuszeile ist nicht gemappt")

    def _abort_poll_disarm(self, grund):
        self.disarmed_mit = grund


def test_failed_poll_disarms_its_transport(monkeypatch):
    """DER FALL, DER TP0 IN w30 TOETETE."""
    t = _BrokenTransport()
    # `poll_status_words` steigt bei leerem `_transports` VOR der Schleife aus
    # und liest die Liste dann ueber `registered()`; beide muessen gesetzt sein,
    # sonst prueft der Test nichts (erster Lauf: rot aus dem falschen Grund).
    monkeypatch.setattr(gate, "_transports", [t])
    monkeypatch.setattr(gate, "registered", lambda: [t])
    monkeypatch.setattr(gate, "abort_check_enabled", lambda: True)
    monkeypatch.setattr(gate, "polling_paused", lambda: False)
    gate.poll_status_words()          # darf NICHT NameError werfen
    assert t.disarmed_mit is not None, "der Transport wurde nicht stillgelegt"
    assert "poll" in t.disarmed_mit.lower()


def test_without_disarm_method_no_crash(monkeypatch):
    """Ein Transport ohne die Methode ist zulaessig -- getattr(..., None)."""
    class _Without:
        def poll_status_word(self):
            raise RuntimeError("kaputt")
    o = _Without()
    monkeypatch.setattr(gate, "_transports", [o])
    monkeypatch.setattr(gate, "registered", lambda: [o])
    monkeypatch.setattr(gate, "abort_check_enabled", lambda: True)
    monkeypatch.setattr(gate, "polling_paused", lambda: False)
    gate.poll_status_words()


def test_healthy_transport_is_not_disarmed(monkeypatch):
    class _Heil:
        def __init__(self): self.disarmed_mit = None
        def poll_status_word(self): return 0
        def _abort_poll_disarm(self, grund): self.disarmed_mit = grund
    h = _Heil()
    monkeypatch.setattr(gate, "_transports", [h])
    monkeypatch.setattr(gate, "registered", lambda: [h])
    monkeypatch.setattr(gate, "abort_check_enabled", lambda: True)
    monkeypatch.setattr(gate, "polling_paused", lambda: False)
    gate.poll_status_words()
    assert h.disarmed_mit is None


class _CountingTransport:
    """Zaehlt, ob er ueberhaupt gepollt wurde."""

    def __init__(self):
        self.polls = 0
        self.disarmed_mit = None

    def poll_status_word(self):
        self.polls += 1
        return False

    def _abort_poll_disarm(self, grund):
        self.disarmed_mit = grund


def test_failed_poll_ends_the_whole_round(monkeypatch):
    """#100 (fnFL2w32/w33): der ZWEITE Poll riss den Prozess mit.

    Gemessen: der Poll scheiterte zweimal in derselben Runde -- einmal je
    registriertem Transport --, danach starb TP0 still (kein Traceback, kein
    Signal). Ursache war ein cu_mem_create-OOM, der das Kontrollwort-Mapping
    unbrauchbar zurueckliess; seine Signaturen stehen in keiner Gift-Liste,
    also lief die Schleife weiter auf einen beschaedigten Kontext.
    """
    kaputt = _BrokenTransport()
    after_state = _CountingTransport()
    monkeypatch.setattr(gate, "_transports", [kaputt, after_state])
    monkeypatch.setattr(gate, "registered", lambda: [kaputt, after_state])
    monkeypatch.setattr(gate, "abort_check_enabled", lambda: True)
    monkeypatch.setattr(gate, "polling_paused", lambda: False)
    gate.poll_status_words()
    assert kaputt.disarmed_mit is not None, "der Werfer wurde nicht stillgelegt"
    assert after_state.polls == 0, (
        "nach einem gescheiterten Poll wurde ein weiteres Geraet gelesen -- "
        "genau der zweite Zugriff, der TP0 in w32/w33 toetete"
    )


def test_metal_signature_is_not_poison_but_aborts(monkeypatch):
    """Die Gegenprobe zur Marker-Liste: der Text taeuscht, das Verhalten nicht.

    `unknown parameter type` ist KEIN Gift-Marker (und soll keiner werden --
    die Liste haette die naechste Signatur wieder nicht). Der Abbruch darf
    deshalb nicht am Text haengen.
    """
    class _Metall(_BrokenTransport):
        def poll_status_word(self):
            raise RuntimeError("unknown parameter type")

    kaputt = _Metall()
    after_state = _CountingTransport()
    assert not gate.is_poison_error(RuntimeError("unknown parameter type")), (
        "wenn diese Signatur Gift WAERE, pruefte dieser Test den #100-Pfad nicht"
    )
    monkeypatch.setattr(gate, "_transports", [kaputt, after_state])
    monkeypatch.setattr(gate, "registered", lambda: [kaputt, after_state])
    monkeypatch.setattr(gate, "abort_check_enabled", lambda: True)
    monkeypatch.setattr(gate, "polling_paused", lambda: False)
    gate.poll_status_words()
    assert after_state.polls == 0


def test_without_error_all_transports_still_polled(monkeypatch):
    """Damit das `break` nicht den Normalfall abschneidet (sonst prueft #100 nichts)."""
    a = _CountingTransport()
    b = _CountingTransport()
    monkeypatch.setattr(gate, "_transports", [a, b])
    monkeypatch.setattr(gate, "registered", lambda: [a, b])
    monkeypatch.setattr(gate, "abort_check_enabled", lambda: True)
    monkeypatch.setattr(gate, "polling_paused", lambda: False)
    gate.poll_status_words()
    assert (a.polls, b.polls) == (1, 1), "das Disarm-break greift im Normalfall"
