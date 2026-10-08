# SPDX-License-Identifier: Apache-2.0
"""L15-POOL: the park sidecar follows the REUSE restamp of the manifest (1840 Fix A).

A D sleep is two flushes. Flush 1 (front /flush_cache) stamps the manifest with the
STALE flip index of the previous release (12); ``park_at_release`` runs between the
flushes and copies that epoch into the park sidecar. Flush 2 (release RPC) takes the
reuse branch and restamps the manifest to this release's flip (14); the wake then
compares sidecar 12 != hold 14 and drops the park (boot 0134: 2 of 2 parks lost).

Fix A: ``l15_park.restamp_sidecar`` stamps the same new epoch into the sidecar at the
reuse restamp (epoch field only -- pieces/sums/guests stay byte for byte). A
non-reuse round does not restamp: the mismatch stays on purpose (wake -> L2).

Hermetic: no CUDA, no model, no network.

Run (own worktree, capped):
  cd /spinning/wt-27b-l15-restamp-1005 && /spinning/gpu-arb/pytest_gedeckelt.sh \
    test/registered/unit/pdflip/test_pdflip_l15_park_restamp_1005.py
"""

from __future__ import annotations

import dataclasses
import inspect
import json

import torch

from flliper.srt.pdflip import l15_manifest, l15_sleep_once
from flliper.srt.pdflip import l15_park as P

import test_pdflip_l15_park_1002 as T  # _World, _bufs, _entries, _run_ranks


def _manifest(epoch):
    from flliper.srt.pdflip.l15_manifest import HoldSpan, Manifest

    sp = HoldSpan(rid="q0", depth=3, slots=(1, 2, 3), anchor_slot=1, l2_slots=(), l2_gens=())
    return Manifest(epoch=epoch, pid=1, spans=(sp,), rows_by_rank=(12, 5, 6), anchor_slots=1)


def _setup(monkeypatch, tmp_path, epoch):
    """Real manifest FILES (so the real restamp runs); l15_manifest.read is made to
    read the file again (pid 1 is alive) instead of the fixed mock of ``_entries``."""
    bufs = {0: T._bufs(0, fill=3), 1: T._bufs(1), 2: T._bufs(2)}
    w, env, scheds = T._entries(monkeypatch, tmp_path, [None] * 3, bufs, [0, 15, 30])
    paths = [str(tmp_path / ("m%d" % r)) for r in range(3)]
    for p in paths:
        l15_manifest.write(p, _manifest(epoch))
    monkeypatch.setattr(l15_manifest, "read", lambda path: _read_file(path))
    env["FLLIPER_PDFLIP_L15_POOL"] = "1"
    return bufs, w, env, scheds, paths


def _read_file(path):
    try:
        with open(path, "rb") as fh:
            return l15_manifest.from_bytes(fh.read())
    except FileNotFoundError:
        return None


def _d_sleep(bufs, env, scheds, logs):
    """Flush 1 stamped 12 -> park_at_release reads it."""
    sent = T._run_ranks(lambda r: P.park_at_release(scheds[r], env, logs.append))
    assert sent[0] > 0
    return sent


def _wake(bufs, env, scheds, paths, logs):
    for b in bufs[0]:
        b.zero_()                                   # KV pause
    hold_epoch = int(_read_file(paths[0]).epoch)    # as the wake reads it
    return hold_epoch, T._run_ranks(lambda r: P.park_back_at_wake(
        scheds[r], env, logs.append, epoch=hold_epoch, group_ok=True))


def test_d_sleep_12_restamp_14_d_wake_parks_back(monkeypatch, tmp_path):
    bufs, w, env, scheds, paths = _setup(monkeypatch, tmp_path, 12)
    orig0 = [b.clone() for b in bufs[0]]
    logs = []
    _d_sleep(bufs, env, scheds, logs)
    assert json.load(open(P.sidecar_path(0, env)))["epoch"] == 12
    # Flush 2 = reuse branch: manifest -> 14, sidecar with it (Fix A)
    for r in range(3):
        l15_sleep_once.restamp(paths[r], 14)
        assert P.restamp_sidecar(r, env, 14) is True
    hold_epoch, back = _wake(bufs, env, scheds, paths, logs)
    assert hold_epoch == 14
    assert back == [True, True, True]
    assert all(torch.equal(bufs[0][L][:12], orig0[L][:12]) for L in range(2))
    assert not any("park epoch" in x for x in logs)
    assert not list(tmp_path.glob("pdflip_l15_park.*"))       # still take-once


def test_without_restamp_the_park_is_refused__characterization(monkeypatch, tmp_path):
    """Green before and after Fix A: a non-reuse round does not restamp the sidecar,
    the guard against a park of an older plan stays."""
    bufs, w, env, scheds, paths = _setup(monkeypatch, tmp_path, 12)
    logs = []
    _d_sleep(bufs, env, scheds, logs)
    for r in range(3):
        l15_sleep_once.restamp(paths[r], 14)        # manifest only
    hold_epoch, back = _wake(bufs, env, scheds, paths, logs)
    assert hold_epoch == 14 and back == [False] * 3
    assert any("park epoch 12 != hold epoch 14" in x for x in logs)


def test_restamp_sidecar_changes_only_the_epoch(monkeypatch, tmp_path):
    env = {"FLLIPER_PDFLIP_L15_PARK_DIR": str(tmp_path)}
    pcs = [P.ParkPiece(0, 1, 0, 5, 7), P.ParkPiece(0, 2, 7, 1, 3)]
    P.write_sidecar(1, env, 12, pcs, sums={"1": [1, 2]}, anchor_guests=[[0, 1, 2]],
                    anchor_sums={"a": 5})
    before = json.load(open(P.sidecar_path(1, env)))
    assert P.restamp_sidecar(1, env, 14) is True
    after = json.load(open(P.sidecar_path(1, env)))
    assert after["epoch"] == 14
    before.pop("epoch"); after.pop("epoch")
    assert after == before                          # fingerprint/pieces/sums/guests untouched
    assert not list(tmp_path.glob("*.tmp"))
    assert P.take_sidecar(1, env)[0] == 14


def test_restamp_sidecar_without_a_file_is_a_noop(tmp_path):
    env = {"FLLIPER_PDFLIP_L15_PARK_DIR": str(tmp_path)}
    assert P.restamp_sidecar(0, env, 14) is False
    assert not list(tmp_path.iterdir())


def test_restamp_sidecar_only_if_epoch_guard(tmp_path):
    """A sidecar that does not belong to the manifest this restamp replaces is left alone."""
    env = {"FLLIPER_PDFLIP_L15_PARK_DIR": str(tmp_path)}
    P.write_sidecar(0, env, 8, [P.ParkPiece(0, 1, 0, 5, 7)])
    assert P.restamp_sidecar(0, env, 14, only_if_epoch=12) is False
    assert json.load(open(P.sidecar_path(0, env)))["epoch"] == 8
    assert P.restamp_sidecar(0, env, 14, only_if_epoch=8) is True
    assert json.load(open(P.sidecar_path(0, env)))["epoch"] == 14


def test_release_reuse_branch_restamps_the_park_sidecar_wiring():
    from flliper.srt.managers import scheduler as sch

    src = inspect.getsource(sch)
    i = src.index("L15-RETAIN reuse epoch=%s")
    seg = src[i - 2200:i]
    assert "restamp_sidecar" in seg
    assert seg.index("l15_sleep_once.restamp(") < seg.index("restamp_sidecar")
    assert "park_on(os.environ)" in seg                 # default off: no file, no call
    assert "int(_l15_flip)" in seg[seg.index("restamp_sidecar"):]
    # only the reuse branch: the fresh round has no restamp_sidecar call
    assert src.count("restamp_sidecar(") == 1
