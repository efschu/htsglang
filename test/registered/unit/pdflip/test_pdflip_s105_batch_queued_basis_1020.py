# SPDX-License-Identifier: Apache-2.0
"""S105 (NF 20.10.): the ``PDFLIP-ROUTE ... BATCH queued`` line names the number the arrival was classified on.

Six NF boots on 20.10. (rc22 ..._094747/_110748, rc23 _105621, rc24 _113244, rc25 _121958) were diagnosed as "after
the boot the front classifies the waiting requests on est_prompt (the whole prompt) -> gross -> flip to P". The front
logs disprove it: every one of the 1272 CLASSIFY records of those boots (count 12:46Z, rc25 still running) carries ``u from d_probe`` (D's tree + the
L2/L3 store at the decision), zero MEASURE/n_est records, and each boot-start flip trigger's u equals the tokens P then
computed (rc25 pdflip-0-1 87361/87361, rc22 pdflip-0-2 11772/11772, rc23 pdflip-0-2 8434/8434, rc22b pdflip-0-3
4452/4452). The P-PREFILL-UNTER-X alarms were kurz requests that arrived in P's phase and rode P's batch (P-BATCH-ALL,
user 02.10. / SOLL R1a, front ``SHORT-IN-P-PHASE``) and set members whose uncached tail an older member of the same P
batch had just prefilled. The misreading came from this one line: it printed ``est_prompt`` next to ``remainder`` and
nothing that said which of the two the class stands on.

Pinned here: the line carries ``u_src`` and ``class`` from the same derivation as the CLASSIFY record, ``remainder`` is
that u, and ``p_phase_rider=1`` marks the P-BATCH-ALL case. The old fields stay (parsers: logindex ``queue=(\\d+)``,
prefill_x_wache ``est_prompt=``). Hermetic: the real ``Front.handle_generate`` against stubbed group RPCs.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_early_flip_1002 import X, _front, _payload, _run  # noqa: E402

from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=8, suite="stage-a-pdflip-unit")

BQ = re.compile(r"PDFLIP-ROUTE rid=(\S+) BATCH queued \(awake=(\S+) admit_d=\S+ est_prompt=(\d+) remainder=(\d+) "
                r"queue=(\d+) u_src=(\S+) class=(\S+) p_phase_rider=([01])\)")
CLS = re.compile(r"kind=arrival action=CLASSIFY .* members=\S+?:(\d+)/\d+@(\S+?)/\d+ .*classes=(\S+)")


def _lines(caplog, pat):
    out = []
    for r in caplog.records:
        m = pat.search(r.getMessage())
        if m:
            out.append(m.groups())
    return out


def test_a_kurz_arrival_in_ps_phase_names_its_class_source_and_the_rider(caplog):
    """rc25 pdflip-0-5 shape: a kurz request arriving while P is awake joins P's batch; the line says kurz, the
    source of u, and p_phase_rider=1 -- est_prompt is printed, but beside the class basis, not as it."""
    caplog.set_level(logging.INFO)
    f = _front({"short": (300, 0.0)})
    f.awake = "P"
    asyncio.run(_run(f, _payload("short", 900), settle=0.3))
    bq = _lines(caplog, BQ)
    assert len(bq) == 1, [r.getMessage()[:200] for r in caplog.records if "BATCH queued" in r.getMessage()]
    rid, awake, est_prompt, remainder, _q, u_src, klass, rider = bq[0]
    assert awake == "P" and klass == "kurz" and rider == "1"
    assert u_src not in ("?", "")
    cls = _lines(caplog, CLS)
    assert cls, "no CLASSIFY record"
    # the line's remainder, source and class are the CLASSIFY record's -- one derivation, two places
    assert cls[0][0] == remainder and cls[0][1] == u_src and cls[0][2] == klass
    assert any("SHORT-IN-P-PHASE" in r.getMessage() for r in caplog.records)


def test_a_gross_arrival_in_ds_phase_is_no_rider():
    """A request over X while D is awake: class gross, p_phase_rider=0 (it is the flip's reason, not a rider)."""
    f = _front({"big": (3 * X, 0.0)})
    logs = []

    class _H(logging.Handler):
        def emit(self, record):
            logs.append(record.getMessage())

    h = _H(level=logging.INFO)
    lg = logging.getLogger(F.logger.name)
    old = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(h)
    try:
        asyncio.run(_run(f, _payload("big", 3 * 3 * X), settle=0.3))
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    bq = [BQ.search(m).groups() for m in logs if BQ.search(m)]
    assert len(bq) == 1, [m[:200] for m in logs if "BATCH queued" in m]
    _rid, awake, _ep, remainder, _q, _src, klass, rider = bq[0]
    assert klass == "gross" and rider == "0" and int(remainder) >= X


def test_the_basis_helper_never_raises_and_mirrors_classify_arrival():
    f = _front({})
    assert F.Front._batch_queued_basis(f, None, 100, X, False) == ("n_est", "kurz")

    class _Xx:
        u_src = "d_probe"

    assert F.Front._batch_queued_basis(f, _Xx(), X, X, False) == ("d_probe", "gross")
    assert F.Front._batch_queued_basis(f, _Xx(), X // 2, X, False) == ("d_probe", "halb")
    assert F.Front._batch_queued_basis(f, _Xx(), 10, X, True) == ("d_probe", "p_only")

    class _Bad:
        @property
        def u_src(self):
            raise RuntimeError("boom")

    assert F.Front._batch_queued_basis(f, _Bad(), 10, X, False) == ("?", "?")
