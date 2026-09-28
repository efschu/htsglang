"""#239 rc12z29d: W7/W10 (launcher half) under the KV token cut.

rc12z29d -st-cut (bd238e58db) reached D READY and was refused by the launcher,
docker_...09282029/launcher.log 20:34:28Z:
  W7/W10 launcher half, group D log: '#706 canonical KV page active' x1,
  'canonical GDN blob active' x1
  REFUSED: W7/W10 (launcher half): D logged kv x1 blob x1, need 3 each
The D log had all three ranks' canonical lines -- TP0 the #706 KV page and GDN
blob, TP1/TP2 (Form A workers owning token rows under --uneven-token-vector
0,48,16) the F14 KV-worker window (cache_controller F14 branch). The counter
knew only the pre-cut worker line ("no page window"), so it counted 1.

The riegel's point stays: a rank that holds KV rows without a canonical page is
refused -- under the cut that is a worker that owns rows but logged no F14 window.
"""
from __future__ import annotations

import inspect
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=4, suite="stage-a-weg2-unit")

# D log boot_weg2_dkrnfh91dprsavisnoadoptstcutbar1dauer09282029_bd238e58db_0928_202950.D.log,
# lines 10146/10148/10160/10161, verbatim
METAL = [
    "[2026-09-28 20:34:14 TP1] #239 F14 KV-WORKER-WINDOW: Form A worker owns token rows (4, 0, 3) "
    "of every 64-token page; KV page window only (no mamba/QSA/draft).",
    "[2026-09-28 20:34:14 TP2] #239 F14 KV-WORKER-WINDOW: Form A worker owns token rows (4, 3, 4) "
    "of every 64-token page; KV page window only (no mamba/QSA/draft).",
    "[2026-09-28 20:34:14 TP0] #706 canonical KV page active: slots [0, 12) of 12, 65536 B per slot; "
    "K/V-major extents [(0, 786432)] of 786432 B; KV keys carry content only (no tp/pp suffix).",
    "[2026-09-28 20:34:14 TP0] #706 canonical GDN blob active: layers [0, 36) of 36, 58834944 of "
    "58834944 blob bytes on this rank, 1 extent(s).",
]
NO_WINDOW = ("[2026-09-21 TP{r}] #706 canonical KV page: this rank is a Form A expert worker "
             "(no attention layer) -- no page window, null storage tier")
EXTRA_D_CUT = "--rank-tp-ratio 1,0,0 --rank-moe-ratio 215,113,160 --uneven-token-vector 0,48,16"


def _log(tmp_path, lines):
    p = tmp_path / "d.log"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_the_cut_boot_counts_all_three_ranks(tmp_path):
    """RED before: kv x1 blob x1 -- the rc12z29d refusal."""
    from sglang.srt.weg2 import launcher as L

    n_kv, n_blob, n_worker = L.canonical_marker_counts(_log(tmp_path, METAL))
    assert (n_kv, n_blob, n_worker) == (3, 3, 2)


def test_the_pre_cut_form_a_boot_counts_as_before(tmp_path):
    from sglang.srt.weg2 import launcher as L

    lines = METAL[2:] + [NO_WINDOW.format(r=1), NO_WINDOW.format(r=2)]
    assert L.canonical_marker_counts(_log(tmp_path, lines)) == (3, 3, 2)


def test_which_workers_must_build_a_kv_window():
    from sglang.srt.weg2 import launcher as L

    assert L.d_kv_worker_ranks(EXTRA_D_CUT) == [1, 2]
    assert L.d_kv_worker_ranks("--rank-tp-ratio 1,0,0 --uneven-token-vector 0,48,0") == [1]
    assert L.d_kv_worker_ranks("--rank-tp-ratio 1,0,0") == []           # no cut
    assert L.d_kv_worker_ranks("--rank-tp-ratio 2,1,1 --uneven-token-vector 3,1,1") == []  # not Form A
    assert L.d_kv_worker_ranks("") == []


def test_a_row_owning_worker_without_a_window_is_still_refused(tmp_path):
    """The riegel stays: under the cut the no-window line of a worker that owns
    rows passes the plain count (3/3) but must not pass the gate."""
    from sglang.srt.weg2 import launcher as L

    lines = METAL[0:1] + METAL[2:] + [NO_WINDOW.format(r=2)]   # TP2 owns rows, built none
    path = _log(tmp_path, lines)
    assert L.canonical_marker_counts(path)[:2] == (3, 3)
    workers = L.d_kv_worker_ranks(EXTRA_D_CUT)
    assert L.count_marker(path, L.FORM_A_KV_WORKER_CANONICAL_MARKER) < len(workers)


def test_the_d_gate_checks_the_row_owning_workers():
    """RED before: the D half knew no F14 line and no cut."""
    from sglang.srt.weg2 import launcher as L

    # rc12z30f: the D half is the RankState gate (IPC Phase 1); the row-owning
    # workers from the launcher's argv are handed to it
    src = inspect.getsource(L.main)
    i = src.index("canonical_state_gate(spec_d")
    assert "kv_owner_ranks=d_kv_worker_ranks(" in src[i:i + 200]


def test_the_f14_marker_is_the_line_the_rank_prints():
    """Two sides of one seam: the launcher's marker is a prefix of the
    cache_controller's F14 log format."""
    import sglang.srt.managers.cache_controller as cc
    from sglang.srt.weg2 import launcher as L

    src = inspect.getsource(cc)
    assert '"' + L.FORM_A_KV_WORKER_CANONICAL_MARKER[:40] in src
    assert L.FORM_A_KV_WORKER_CANONICAL_MARKER in METAL[0]
    assert L.FORM_A_WORKER_CANONICAL_MARKER in NO_WINDOW
