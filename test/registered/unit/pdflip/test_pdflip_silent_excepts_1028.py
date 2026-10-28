"""ENTWURF (nicht ausgefuehrt, Nutzer 28.10. 09:40Z: Desk ohne CPU/GPU): Testfaelle zu silent_excepts_infeasible_rewake_1028.patch.
Ablage nach Freigabe: test/registered/unit/pdflip/test_pdflip_silent_excepts_1028.py, Patch zuerst anwenden (git apply), dann pytest -q dieser Datei;
dann je Mutation (Warnung entfernen / Einmal-Sperre entfernen) auf rot pruefen wie bei test_pdflip_d_transient_lend_1001.py."""
from __future__ import annotations

import logging
import os
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import card_kv_ledger as ckl  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402
from flliper.srt.pdflip import dual_p_kv_stage as pk  # noqa: E402


def test_infeasible_cards_warns_once_when_the_ledger_peek_fails(caplog):
    """A failing peek (mmap/OSError) answered [] = 'feasible' for every round without a line; one WARNING per boot names it,
    the answer stays [] (never an exception into the round)."""
    pk._INFEASIBLE_WARNED.clear()
    stages = [{"step": 1, "top": 2, "bytes": [0, 1, 2], "ledger": "x"}]
    with caplog.at_level(logging.WARNING, logger=pk.logger.name), mock.patch.object(ckl, "peek", side_effect=OSError("mmap")):
        assert pk.infeasible_cards(stages, {}, 1) == []
        assert pk.infeasible_cards(stages, {}, 1) == []
    lines = [r.getMessage() for r in caplog.records if "infeasible_cards: the ledger peek failed" in r.getMessage()]
    assert len(lines) == 1, lines
    pk._INFEASIBLE_WARNED.clear()


def test_rewake_switch_unreadable_warns_once_and_is_off(caplog):
    """An unreadable D-SEAT-REWAKE switch stopped the live seat re-plan like a switched-off one, silently."""
    from flliper.srt.environ import envs

    dsv._REWAKE_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger=dsv.logger.name), \
            mock.patch.object(type(envs.FLLIPER_PDFLIP_D_SEAT_REWAKE), "get", side_effect=RuntimeError("registry")):
        assert dsv._rewake_on() is False
        assert dsv._rewake_on() is False
    lines = [r.getMessage() for r in caplog.records if "D-SEAT-REWAKE switch unreadable" in r.getMessage()]
    assert len(lines) == 1, lines
    dsv._REWAKE_WARNED.clear()
