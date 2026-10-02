# SPDX-License-Identifier: Apache-2.0
"""L15-FIX-CAP0-KEEP (N3l 02.10. 02:29:56Z): a cap-0 rank pins NO VRAM at
the D sleep.

N3l's first armed hold logged on TP0 (cap 0 = the 5090, "not held here",
refilled from L2 at the wake) "L15-KEEP-ALIGN rank=0 ... kept_bytes=555745280":
the retain armed keep windows on TP0 exactly like on the capped ranks, so
the 5090 carried held KV through the P phase that never budgeted it -- at
the caps c1=7616 MiB the TP0 share of a full hold is of the same order, an
OOM of P on the 5090. Pinned here with the retain test's own scenario:
cap 0 -> every set_keep gets an EMPTY range set, the manifest is still
published; a capped rank keeps its windows.
"""

from __future__ import annotations

import importlib.util
import os

from sglang.srt.weg2 import l15_retain

_HERE = os.path.dirname(__file__)


def _mod():
    spec = importlib.util.spec_from_file_location(
        "test_weg2_l15_retain_0930",
        os.path.join(_HERE, "test_weg2_l15_retain_0930.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(tmp_path, caps):
    m = _mod()
    sc = m.make_scenario(tmp_path, [])
    kw = dict(sc["kwargs"])
    kw["caps_rows_by_rank"] = caps
    # only the two candidates the scenario has slots for (with rank 1 at
    # cap 0 nothing would block r_big, which has no slots in the fixture)
    kw["candidates"] = [c for c in kw["candidates"]
                        if c.rid in ("r_seat", "r_parked")]
    res = l15_retain.retain_at_sleep(**kw)
    keeps = [c for c in sc["set_keep_calls"] if c[0] == "set_keep"]
    return res, keeps, sc["manifest_path"]


def test_cap0_rank_arms_empty_keep_windows_but_publishes(tmp_path):
    # RANK is 1 in the scenario; cap 0 on rank 1, room on rank 0
    res, keeps, mpath = _run(tmp_path, (10, 0))
    assert res is not None
    assert keeps, "set_keep must still be called (it clears the base)"
    assert all(c[2] == () for c in keeps), keeps
    assert os.path.exists(mpath)


def test_capped_rank_keeps_its_windows(tmp_path):
    res, keeps, _ = _run(tmp_path, (10, 10))
    assert res is not None
    assert any(c[2] != () for c in keeps)
