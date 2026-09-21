"""#73 (21.09.): der ipc-Arm pinnt keinen Host-Puffer, also gibt es keine
Bounce-Geometrie zu verlangen.

fnFL2v97 erreichte beide Gruppen READY, die Front und den ERSTEN FLIP -- und
starb dann am Wake: `W4 Weg2WakeRefused ... SGLANG_WEG2_XCHG_BOUNCE_TERMS was
not published`, auf allen drei P-Raengen, gefolgt von W29 (rank disagree) und
einem `cudaErrorInvalidValue` aus `MemPool::~MemPool()` unter
`_PyModule_Clear` -- letzteres der Shutdown-Folgeschaden, nicht die Wurzel.

Die Wurzel waren ZWEI PRAEDIKATE FUER EINE FRAGE:

  Launcher:  xchg_bounce_arm_pins_host(weight_source, oncard_mode)  -- BEIDE
  Wake:      "ist exchange die Quelle?"                             -- EINES

Bei `--weg2-xchg-oncard ipc` exportiert der Arm eine VRAM-Bounce, die mit
ihrem eigenen Leg stirbt: kein bounce-FILE, kein gepinntes Host-Byte. Der
Launcher publiziert dort ZU RECHT nichts -- und der Wake verweigerte trotzdem.
"""

import inspect

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.weg2 import weight_exchange_transport as wt


def _src():
    return inspect.getsource(wu)


def test_the_arm_decides_not_the_source_alone():
    src = _src()
    assert "_wt.ENV_ONCARD_MODE" in src
    assert "_oncard == _wt.ONCARD_MODE_IPC" in src


def test_only_an_explicit_ipc_passes():
    """Eine FEHLENDE Variable heisst 'unbekannter Arm' und bleibt eine
    Verweigerung -- sonst wuerde jeder Boot ohne publizierte Arm-Env still
    ohne Terme injizieren."""
    src = _src()
    i_cmp = src.index("_oncard == _wt.ONCARD_MODE_IPC")
    # der Vergleich ist auf GLEICHHEIT mit dem ipc-Wert, nicht auf
    # "!= host" -- das ist der Unterschied zwischen "weiss es" und "raet"
    assert "_oncard != " not in src[i_cmp - 200:i_cmp + 200]
    # und der else-Zweig verweigert weiterhin
    assert "raise Weg2WakeRefused(" in src[i_cmp:i_cmp + 2000]


def test_the_refusal_names_the_arm_it_saw():
    """Eine Verweigerung, die den Arm nicht nennt, laesst den Leser genau die
    Frage offen, die sie beantwortet hat."""
    src = _src()
    assert "Arm here: oncard=" in src
    assert "{_oncard!r}" in src


def test_the_priced_line_survives_terms_none():
    """`priced` las `terms.total_bytes` unbedingt -- auf dem ipc-Arm waere das
    ein AttributeError an der Stelle, an der gerade erst die Verweigerung
    aufgehoben wurde."""
    src = _src()
    assert 'priced = ("(no host bounce priced: oncard=ipc)" if terms is None' in src


def test_the_leg_is_built_for_terms_none():
    """Der Grund, warum das ueberhaupt zulaessig ist: die Leg TRAEGT
    `terms=None` in ihrer Signatur und prueft es an jeder lesenden Stelle.
    Faellt das kuenftig, faellt dieser Test -- und der Fix oben mit ihm."""
    leg = inspect.getsource(wu.SchedulerWeightUpdaterManager._weg2_xchg_bounce_leg)
    assert "terms=None" in leg.split("\n")[0] or "terms=None" in leg[:600]
    assert "terms is not None" in leg


def test_the_two_oncard_modes_are_the_ones_we_branch_on():
    """Kein drittes Wort, das der Zweig nicht kennt."""
    assert wt.ONCARD_MODE_IPC == "ipc"
    assert wt.ONCARD_MODE_HOST == "host"
    assert wt.ENV_ONCARD_MODE == "SGLANG_WEG2_XCHG_ONCARD"
