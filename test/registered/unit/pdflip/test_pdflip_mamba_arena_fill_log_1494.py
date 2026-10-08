"""1494: Instrument 'Mamba-Arena-Fuellstand loggen'.

Kein Log nennt, wie viele Arena-Slots belegt sind (Q-1303 musste ARENA-CLAIM
REFUSED / ARENA-DROP als Proxy nehmen). Der Schalter
FLLIPER_PDFLIP_MAMBA_ARENA_FILL_LOG (Default AUS) laesst `_claim` (auch fuer den
Mamba-Pool) genau eine Zeile schreiben:

  PDFLIP MAMBA-ARENA-FILL event=claim pool=... used=<n>/<slots> claimed=..
  complete=.. free=.. staging_used=<m>/<staging_rows>

Hermetisch (CPU): `ArenaMHAHostPool._claim` auf einem Minimal-Pool mit
Fake-Arena, deren `stats()` die Zaehler der echten Arena liefert.
"""

import logging
from types import SimpleNamespace

import torch

from flliper.srt.environ import envs
from flliper.srt.mem_cache.pool_host import arena_pool
from flliper.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-pdflip-unit")

MARKER = "PDFLIP MAMBA-ARENA-FILL"


class _FakeArena:
    path = "fake"

    def claim_slots(self, stems, totals, role=None):
        return [(i, 0, 1) for i in range(len(stems))]  # status 0 = fresh claim

    def stats(self):
        return {"slots": 32, "complete": 5, "claimed": 3, "slot_bytes": 1024, "file_bytes": 0}


def _pool() -> SimpleNamespace:
    return SimpleNamespace(
        arena=_FakeArena(),
        _page_bytes=1024,
        _pending_mask=None,
        _pending={},
        _pend_mark=lambda slots, flag: None,
        staging_rows=4,
        free_slots=torch.arange(3, dtype=torch.int64),  # 1 of 4 staging rows used
    )


def _records(caplog) -> list:
    return [r.getMessage() for r in caplog.records if MARKER in r.getMessage()]


def test_arena_fill_log_off_by_default(caplog):
    """Schalter aus (Default) = keine Zeile, der Claim liefert wie vorher."""
    caplog.set_level(logging.INFO, logger="flliper.srt.mem_cache.pool_host.arena_pool")
    assert not envs.FLLIPER_PDFLIP_MAMBA_ARENA_FILL_LOG.get()
    arena_pool._FILL_LOG_LAST.clear()

    assert ArenaMHAHostPool._claim(_pool(), ["a", "b"]) == [0, 1]

    assert _records(caplog) == []


def test_arena_fill_log_on_logs_exactly_one_line(caplog):
    """Schalter an + ein Claim = genau eine Zeile mit allen Feldern."""
    caplog.set_level(logging.INFO, logger="flliper.srt.mem_cache.pool_host.arena_pool")
    arena_pool._FILL_LOG_LAST.clear()
    pool = _pool()

    with envs.FLLIPER_PDFLIP_MAMBA_ARENA_FILL_LOG.override(True):
        assert ArenaMHAHostPool._claim(pool, ["a", "b"]) == [0, 1]
        assert ArenaMHAHostPool._claim(pool, ["c"]) == [0]  # inside the 1 s gap: no 2nd line

    records = _records(caplog)
    assert records == [
        "PDFLIP MAMBA-ARENA-FILL event=claim pool=SimpleNamespace used=8/32 claimed=3 "
        "complete=5 free=24 staging_used=1/4"
    ], records
