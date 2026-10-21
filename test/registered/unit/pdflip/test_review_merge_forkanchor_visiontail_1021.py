"""Merge probe review (21.10.): D-FORK-ANCHOR (b197da0c60) x IMAGE-END TRACK (visiontail e850daaa74 / d542693035).

After the merge, ``turn_anchor.note_step`` calls ``_note_image_step`` on group D too (armed wherever the turn token
is, and D-FORK-ANCHOR arms D). ``_note_image_step`` draws with the P rule (``_alloc_slot``: a slot or NO plan), not
the D position plan (``_draw_slot`` / ``_put_plan``): plan existence follows each rank's own free list. A Form A
worker's bookkeeping pool has free slots exactly where TP0's fork slots are held -> the worker plans (and inserts
key[:t], split-only) where TP0 plans nothing -> the trees differ (the "only TP0 splits" class). D does prefill image
steps since VISION-ROWS (an image P encoded once is computed by D from P's tower rows)."""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


TA0 = _load("_turn_anchor_0929_for_d_fork", _HERE / "test_pdflip_turn_anchor_0929.py")
DF = _load("_d_fork_anchor_1021_for_review", _HERE / "test_nf_rc30_d_fork_anchor_1021.py")
group_d = DF.group_d
PAD = 1_000_000
IMG = (10, 20)


def _img_prompt():
    ids = list(TA0.PROMPT)
    ids[IMG[0]:IMG[1]] = [PAD + 7] * (IMG[1] - IMG[0])
    return ids


def _d_image_prefill(monkeypatch, worker: bool):
    from flliper.srt.pdflip import turn_anchor as ta

    DF._as_worker(monkeypatch, worker)
    fx = TA0._fixture(False)
    req = TA0._req(fx, _img_prompt())
    if not worker:
        # TP0: every free slot held -- by its in-flight / in-tree fork slots in production (the workers hold none)
        while fx.pool.mamba_allocator.available_size() > 0:
            fx.pool.mamba_allocator.alloc(1)
    TA0._write_kv(fx, req, TA0.N)
    batch = SimpleNamespace(reqs=[req], req_to_token_pool=fx.pool)
    desc = ta.note_step(batch=batch, desc=None, req=req, row=0, prefix=0, end=TA0.N,
                        track_mask=True, main_track=TA0.N - 1, chunk=1, page=1, tok=TA0.IM)
    if desc is not None and not worker:
        desc.done.add("gdn")
    TA0._state(fx, req.mamba_pool_idx, TA0.S_END)
    TA0._finish(fx, req)
    return fx, req


# MERGE FINDING V1 fixed (_note_image_step plans through _draw_slot/_put_plan on group D): strict-xfail removed.
def test_merged_image_track_on_group_d_keeps_the_trees_equal(group_d):
    host, _ = _d_image_prefill(group_d, worker=False)
    wk, _ = _d_image_prefill(group_d, worker=True)
    assert DF._structure(host) == DF._structure(wk), (sorted(DF._structure(host)), sorted(DF._structure(wk)))
