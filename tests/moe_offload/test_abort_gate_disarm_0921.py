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

from sglang.srt.distributed.device_communicators import barlink_abort_gate as gate


class _KaputterTransport:
    """Sein Poll wirft -- genau der Fall, fuer den der Disarm gebaut ist."""

    def __init__(self):
        self.disarmed_mit = None

    def poll_status_word(self):
        raise RuntimeError("die Statuszeile ist nicht gemappt")

    def _abort_poll_disarm(self, grund):
        self.disarmed_mit = grund


def test_ein_gescheiterter_poll_disarmt_seinen_transport(monkeypatch):
    """DER FALL, DER TP0 IN w30 TOETETE."""
    t = _KaputterTransport()
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


def test_ohne_disarm_methode_kein_absturz(monkeypatch):
    """Ein Transport ohne die Methode ist zulaessig -- getattr(..., None)."""
    class _Ohne:
        def poll_status_word(self):
            raise RuntimeError("kaputt")
    o = _Ohne()
    monkeypatch.setattr(gate, "_transports", [o])
    monkeypatch.setattr(gate, "registered", lambda: [o])
    monkeypatch.setattr(gate, "abort_check_enabled", lambda: True)
    monkeypatch.setattr(gate, "polling_paused", lambda: False)
    gate.poll_status_words()


def test_gesunder_transport_wird_nicht_disarmt(monkeypatch):
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
