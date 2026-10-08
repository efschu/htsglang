"""F0-F (rename 08.10.): the owned-miss rank records written BEFORE the rename lie under ``records/<old subsystem>``
(5838 files on the rig), the tree writes ``records/pdflip``. The launcher reads both; writers keep the one root.

* ``record_dirs_for`` names the line's directory first and the pre-rename sibling second, only for the default
  ``.../records/pdflip`` root (an ``--env-d`` root of another name has no sibling);
* ``read_owned_miss_rank_records`` takes one path (as before) or several and pools them;
* ``launcher.d_owned_miss_ms`` / ``d_owned_miss_ms_rank`` find a window that exists only under the old root;
* the youngest-window rule picks among both roots (a new record beats an older legacy one).

The old subsystem name is taken from ``name_compat.STEM_TOKENS`` (split there: the rename tool must not rewrite it here).
"""

from __future__ import annotations

import json
import os
import types

import pytest

from flliper.srt import name_compat as nc
from flliper.srt.layers.moe import pool_miss_cost as pmc
from flliper.srt.planner import expert_residency as er
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

OLD, NEW = nc.STEM_TOKENS
MODEL = "/models/Qwen3.8-Flash-Next-INT4"


def _rec(rank, fetch_ms, rows, t, rounds=100):
    return {"kind": er.OWNED_MISS_RANK_KIND, "rank": rank, "time_unix": t, "fetch_ms": fetch_ms, "miss_rows": rows,
            "rounds": rounds, "pairing": er.OWNED_MISS_PAIRING, "model": MODEL}


def _put(root, name, rec):
    d = pmc.record_dir_for(str(root), MODEL)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as fh:
        json.dump(rec, fh)


@pytest.fixture
def roots(tmp_path):
    base = tmp_path / "evidence" / "records"
    return base / NEW, base / OLD


def test_legacy_root_is_the_sibling_of_the_default_root_only(roots):
    new, old = roots
    assert pmc.legacy_root_for(str(new)) == str(old)
    assert pmc.legacy_root_for(str(new) + "/") == str(old)
    assert pmc.legacy_root_for("/somewhere/else") is None
    assert pmc.legacy_root_for(str(old)) is None            # the old root itself has no older sibling
    assert pmc.record_dirs_for(str(new), MODEL) == (pmc.record_dir_for(str(new), MODEL), pmc.record_dir_for(str(old), MODEL))
    assert pmc.record_dirs_for("/somewhere/else", MODEL) == (pmc.record_dir_for("/somewhere/else", MODEL),)


def test_reader_takes_one_path_or_several(roots):
    new, old = roots
    _put(new, "owned_miss_D_tp0_a.json", _rec(0, 24.0, 4800, 1000.0))
    _put(old, "owned_miss_D_tp0_b.json", _rec(1, 96.0, 4800, 1000.0))
    d_new, d_old = pmc.record_dir_for(str(new), MODEL), pmc.record_dir_for(str(old), MODEL)
    assert [r["rank"] for r in er.read_owned_miss_rank_records(d_new)] == [0]          # a plain path: as before
    assert sorted(r["rank"] for r in er.read_owned_miss_rank_records((d_new, d_old))) == [0, 1]
    assert er.read_owned_miss_rank_records((d_new, "/missing/dir", d_old))               # a missing directory is skipped
    assert er.read_owned_miss_rank_records(None) == [] and er.read_owned_miss_rank_records(()) == []


def test_launcher_reads_a_window_that_lies_only_under_the_old_root(roots):
    from flliper.srt.pdflip import launcher

    new, old = roots
    # 16+ paired forwards on every rank, all written by the pre-rename tree
    _put(old, "owned_miss_D_tp0_a.json", _rec(0, 24.0, 4800, 1000.0))
    _put(old, "owned_miss_D_tp1_a.json", _rec(1, 96.0, 4800, 1000.0))
    env_d = {"FLLIPER_PDFLIP_OWNED_MISS_RECORD": str(new)}
    ns = types.SimpleNamespace(profile=None, model=MODEL)
    ms, src = launcher.d_owned_miss_ms(ns, env_d=env_d, host=0)
    assert ms == pytest.approx((24.0 / 4800, 96.0 / 4800)) and src.startswith("RECORD")
    per_rank = launcher.d_owned_miss_ms_rank(ns, env_d=env_d, n=2)
    assert per_rank[0] == pytest.approx((24.0 / 4800, 96.0 / 4800)) and per_rank[1].startswith("RECORD")


def test_a_root_without_a_sibling_reads_only_itself(tmp_path):
    from flliper.srt.pdflip import launcher

    other = tmp_path / "other-root"
    _put(other, "owned_miss_D_tp0_a.json", _rec(0, 24.0, 4800, 1000.0))
    _put(other, "owned_miss_D_tp1_a.json", _rec(1, 96.0, 4800, 1000.0))
    ns = types.SimpleNamespace(profile=None, model=MODEL)
    ms, _ = launcher.d_owned_miss_ms(ns, env_d={"FLLIPER_PDFLIP_OWNED_MISS_RECORD": str(other)}, host=0)
    assert ms == pytest.approx((24.0 / 4800, 96.0 / 4800))


def test_the_youngest_window_wins_across_both_roots(roots):
    from flliper.srt.pdflip import launcher

    new, old = roots
    # an old window (cost 0.1 / 0.4 per row) and, much younger than the rank window, a new one (0.005 / 0.02)
    _put(old, "owned_miss_D_tp0_a.json", _rec(0, 480.0, 4800, 1000.0))
    _put(old, "owned_miss_D_tp1_a.json", _rec(1, 1920.0, 4800, 1000.0))
    young = 1000.0 + 10 * er.OWNED_MISS_RANK_WINDOW_S
    _put(new, "owned_miss_D_tp0_b.json", _rec(0, 24.0, 4800, young))
    _put(new, "owned_miss_D_tp1_b.json", _rec(1, 96.0, 4800, young))
    ns = types.SimpleNamespace(profile=None, model=MODEL)
    ms, _ = launcher.d_owned_miss_ms(ns, env_d={"FLLIPER_PDFLIP_OWNED_MISS_RECORD": str(new)}, host=0)
    assert ms == pytest.approx((24.0 / 4800, 96.0 / 4800))


def test_the_writer_root_is_unchanged():
    from flliper.srt.pdflip import launcher

    assert os.path.basename(launcher.owned_miss_record_root()) == NEW
    assert os.path.basename(os.path.dirname(launcher.owned_miss_record_root())) == "records"
