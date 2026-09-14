# SPDX-License-Identifier: Apache-2.0
"""#1344 round 2: the weg2 shm-residue ratchet must see EVERY artefact class
a test can leave, not just the bare region directory. See `conftest.py`'s
own module docstring for the full derivation (measured by execution, not
read from the code alone): `_own_residue()` matched `weg2-xchg-<nonce>` but
missed the semaphores' Linux backing files (`sem.weg2-xchg-<nonce>-*`) and
the bounce slot table (`weg2-xchg-bnc-<nonce>`, #1363's own finding, reused
here).

RED-FIRST RECORD (2026-09-14, against train tip `8de5fdc4b3`): with
`conftest.py`'s fix reverted (the OLD `_own_residue()`, region-directory-only
check restored by hand), `test_deliberate_semaphore_residue_is_caught` below
failed to find its own deliberately-created semaphores -- verified by git
diff / git apply, not merely asserted. See this file's accompanying commit
for the reverted-and-reapplied proof.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_weg2_xchg_transport_1273 import _fresh_boot  # noqa: E402

from sglang.srt.weg2 import weight_exchange_bounce as wb  # noqa: E402
from sglang.srt.weg2 import weight_exchange_region as xr  # noqa: E402


def _load_weg2_conftest():
    """A plain `import conftest` binds to `test/conftest.py` (pytest's own
    rootdir conftest is already in `sys.modules` under that bare name by
    collection time) -- load THIS directory's `conftest.py` by explicit
    path instead, the same technique `test_weg2_1363_smoke_stub_drift.py`
    uses for the non-package smoke script."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "conftest.py")
    spec = importlib.util.spec_from_file_location("weg2_own_conftest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_own_residue = _load_weg2_conftest()._own_residue


def test_deliberate_semaphore_residue_is_caught():
    """The exact reproduction: `xr.create_semaphores(nonce)` alone (no full
    region, no `LayerBounce`/`BounceSlots`) must be visible to `_own_residue`
    -- this is what a test that opens semaphores and dies before its own
    `unlink_semaphores` call leaves behind, and what escaped detection
    before this round.

    Cleans up via the SAME `xr.unlink_semaphores` the product itself uses
    (never a raw glob-unlink) -- this test is itself a caller and returns
    what it created, exactly the rule it is testing.
    """
    nonce = _fresh_boot()
    xr.create_semaphores(nonce)
    try:
        found = _own_residue()
        found_names = {os.path.basename(p) for p in found}
        # At least one semaphore backing file must be visible -- the exact
        # gap this round closes. Not asserting the full count (~60): that
        # number is `create_semaphores`' own business, not this test's.
        assert any(n.startswith(f"sem.{xr.REGION_PREFIX}{nonce}")
                  for n in found_names), (
            f"created real semaphores for nonce={nonce} but _own_residue() "
            f"found none of them: {sorted(found_names)[:5]}...")
    finally:
        xr.unlink_semaphores(nonce)
    # And after the proper holder-scoped cleanup, nothing of this nonce
    # remains -- proving the fix's OWN prefix match doesn't fire on thin air.
    after = {os.path.basename(p) for p in _own_residue()
            if nonce in os.path.basename(p)}
    assert not after, f"unlink_semaphores left residue behind: {after}"


def test_deliberate_bounce_slots_residue_is_caught(tmp_path):
    """The #1363-class gap, reused rather than rediscovered:
    `weight_exchange_bounce.BounceSlots` writes directly under `shm_root`
    with its OWN prefix (`weg2-xchg-bnc-<nonce>`), not nested under the
    region directory `_own_residue` already covered -- so a test that
    builds one against the REAL shm root (not a tmp_path double) needs the
    same explicit prefix this round adds.
    """
    nonce = _fresh_boot()
    slots = wb.BounceSlots(nonce, shm_root=xr.SHM_ROOT, create=True)
    try:
        found_names = {os.path.basename(p) for p in _own_residue()}
        assert f"{wb.BOUNCE_SLOT_PREFIX}{nonce}" in found_names, (
            f"created a real BounceSlots table for nonce={nonce} but "
            f"_own_residue() did not find {wb.BOUNCE_SLOT_PREFIX}{nonce}: "
            f"{sorted(found_names)[:5]}...")
    finally:
        slots.unlink()
    after = {os.path.basename(p) for p in _own_residue()
            if nonce in os.path.basename(p)}
    assert not after, f"BounceSlots.unlink() left residue behind: {after}"


def test_M_removing_the_sem_prefix_check_goes_blind(monkeypatch):
    """MUTANT (operator-required danger direction): reproduce the ORIGINAL,
    region-directory-only `_own_residue` by hand (no `sem.` / bounce-slot
    prefixes) against the SAME real semaphores the first test above proves
    are visible today, and assert the mutant is BLIND to them -- the
    calibration proof that this ratchet's own logic, not merely the
    presence of a fix, is what closes the gap.
    """
    nonce = _fresh_boot()
    xr.create_semaphores(nonce)
    try:
        root = xr.SHM_ROOT
        mine = f"{xr.REGION_PREFIX}{nonce}"          # the OLD, incomplete check
        try:
            names = os.listdir(root)
        except OSError:
            names = []
        old_found = [n for n in names if n.startswith(mine)]
        assert old_found == [], (
            "the mutant (region-directory-only prefix) unexpectedly found "
            f"something -- it should be structurally blind to bare "
            f"semaphores: {old_found}")
        # ...while the REAL, current _own_residue() (imported above, not
        # reimplemented) DOES see them -- the fix and the mutant disagree on
        # the identical filesystem state, which is the point.
        new_found = {os.path.basename(p) for p in _own_residue()}
        assert any(n.startswith(f"sem.{mine}") for n in new_found)
    finally:
        xr.unlink_semaphores(nonce)


if __name__ == "__main__":
    import unittest

    unittest.main()
