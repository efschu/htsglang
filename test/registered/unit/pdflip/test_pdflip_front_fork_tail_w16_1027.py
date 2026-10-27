"""W16 im Dual (27B dualm, Boot dkr27bnvfp4dualbar1drei10271342_34ebca1816, 27.10.):
der fork-geschnittene Hand-back-Rest ist D's eigener Extend, kein W16 ueber X.

BEFUND. Dual-Layout: X=1 (--dual-d-prefill-tokens 0). Mit FLLIPER_PDFLIP_FORK_ANCHOR_TOKEN=248045
schneidet P's Intake jeden Leg-1 VOR dem Generation-Prompt (``P-TRIM-END-ANCHOR ... fork
rid=pdflip-0-1 tokens=4022->4015``); D rechnet den 7-Token-Rest selbst, sein X-Gate laesst ihn
durch (``PDFLIP X-GATE FORK-TAIL rid=pdflip-0-1 uncached=7 tail=7 X=1 verdict=admit``, Q-360).
Der Front aber preiste leg 2 weiter gegen X=1 (``_leg2_verdict``): uncached=7 > 1 -> reroute
(zweiter P-Prefill, 1,45 s), danach W16, D's fertige Antwort (HTTP 200) wurde verworfen und der
Client bekam 503 ``W16 PdFlipDoublePrefillExceeded rid=pdflip-0-1 uncached=7``.

DIE TESTS laufen den echten Front-Verdikt (``Front._leg2_verdict``) mit der echten Fork-Schnitt-
Funktion. Rot auf 34ebca1816 (reroute/W16), gruen danach. Gefahrenrichtungen gepinnt: ein Rest
ueber dem Fork-Rest wird weiter gegen X gepreist, Schalter aus = altes Verhalten, ohne Token-Ids
= kein Spielraum.
"""
import numpy as np
import pytest

from flliper.srt.pdflip.front import Front, Pending

IM_START = 248045
GEN7 = [IM_START, 74455, 198, 248068, 198, 271, 248069]  # 7-Token-Generation-Prompt


def _front(x=1):
    return Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, carrier_max_tokens=648806,
                 tp_prefill_max_tokens=x)


class _Tok:
    def __init__(self, ids):
        self._ids = ids

    def ids_for(self, text):
        return self._ids


def _pending():
    return Pending("r", "/v1/chat/completions", {}, "x", 0.0, None)


def _prompt(n_body):
    return np.asarray(list(range(1000, 1000 + n_body)) + GEN7, dtype=np.int32)


def test_fork_tail_served_not_w16(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_FORK_ANCHOR_TOKEN", str(IM_START))
    f = _front(1)
    ids = _prompt(4015)
    f.ftok = _Tok(ids)
    v = f._leg2_verdict(len(ids), 4015, True, _pending(), False, False, "r", text="t")
    assert v == "serve"
    assert f.counters["W16_PdFlipDoublePrefillExceeded"] == 0


def test_fork_tail_stream_not_counted_as_w16(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_FORK_ANCHOR_TOKEN", str(IM_START))
    f = _front(1)
    ids = _prompt(46)
    f.ftok = _Tok(ids)
    v = f._leg2_verdict(len(ids), 46, True, _pending(), False, True, "r", text="t")
    assert v == "serve"
    assert f.counters["W16_PdFlipDoublePrefillExceeded"] == 0


def test_match_short_of_the_fork_is_still_priced_against_x(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_FORK_ANCHOR_TOKEN", str(IM_START))
    f = _front(1)
    ids = _prompt(4015)
    f.ftok = _Tok(ids)
    # D hat nur bis 3000 gelesen: Rest 1022 > Fork-Rest 7 -> X wird nicht angehoben
    assert f._leg2_verdict(len(ids), 3000, True, _pending(), False, False, "r", text="t") == "reroute"
    p = _pending()
    p.reroutes = 1
    assert f._leg2_verdict(len(ids), 3000, True, p, False, False, "r", text="t") == "W16"


def test_switch_off_keeps_w16(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_FORK_ANCHOR_TOKEN", raising=False)
    f = _front(1)
    ids = _prompt(4015)
    f.ftok = _Tok(ids)
    assert f._leg2_verdict(len(ids), 4015, True, _pending(), False, False, "r", text="t") == "reroute"


def test_without_front_ids_no_allowance(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_FORK_ANCHOR_TOKEN", str(IM_START))
    f = _front(1)
    f.ftok = None
    assert f._leg2_verdict(4022, 4015, True, _pending(), False, False, "r", text="t") == "reroute"
    assert f._leg2_verdict(4022, 4015, True, _pending(), False, False, "r") == "reroute"


def test_prompt_without_fork_token_gets_no_allowance(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_FORK_ANCHOR_TOKEN", str(IM_START))
    f = _front(1)
    ids = np.asarray(list(range(1000, 1030)), dtype=np.int32)
    f.ftok = _Tok(ids)
    assert f._leg2_verdict(30, 23, True, _pending(), False, False, "r", text="t") == "reroute"
