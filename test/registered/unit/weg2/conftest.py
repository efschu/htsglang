"""#1344: the weg2 suite must not leave `/dev/shm` residue behind.

THIRD SIGHTING, and each one cost a boot seat pre-flight time: boots weg2xsn17
and weg2xsn18 both opened with 12-20 dead `/dev/shm/weg2-xchg-s4b<pid>x<n>`
directories to sweep, and the producer turned out to be THIS SUITE, not a
crashed serving process.  `_fresh_boot()` (test_weg2_xchg_transport_1273.py)
mints a nonce `s4b<pid>x<n>`, tests open a region under the real
`weight_exchange_region.SHM_ROOT` with it, and nothing removed it afterwards.

Precedent #438(b): residue with a producer gets cleaned at the producer, and a
RATCHET asserts the absence so the third sighting is the last.

THE SWEEP IS BOUND TO THIS PROCESS'S OWN PID, and that bound is the whole
safety argument.  `#1217`/`#1310` are the incidents where a broad `/dev/shm`
sweep ran against a LIVE holder; a serving boot's region is named
`weg2-xchg-<epoch>` and a test's is `weg2-xchg-s4b<pid>x<n>` with THIS
interpreter's pid, so this fixture cannot reach a boot's region even if one is
running concurrently -- which on this box it may well be.  No pid scan, no
holder check, no lsof needed: the name itself is the proof of ownership.
"""

import os
import shutil

import pytest

from sglang.srt.weg2 import weight_exchange_region as xr


def _own_residue():
    """Region dirs/files this interpreter's own `_fresh_boot` nonces created.

    Reads `xr.SHM_ROOT` live rather than caching it: a test that monkeypatches
    the constant to `tmp_path` gets its residue cleaned by pytest instead, and
    this must not then sweep the real root under a stale assumption.
    """
    root = getattr(xr, "SHM_ROOT", "/dev/shm")
    mine = f"{xr.REGION_PREFIX}s4b{os.getpid()}x"
    try:
        names = os.listdir(root)
    except OSError:
        return []
    return [os.path.join(root, n) for n in names if n.startswith(mine)]


@pytest.fixture(autouse=True)
def _weg2_shm_residue_guard():
    """Remove this process's own region residue after every weg2 test.

    AUTOUSE and in `finally`, because the leak happened on the paths that
    RAISED: a test asserting a refusal leaves the region behind precisely when
    the product refused, which is most of this suite.
    """
    try:
        yield
    finally:
        for p in _own_residue():
            try:
                if os.path.isdir(p):
                    # truncate before unlink so the tmpfs pages are returned
                    # even if something still holds a descriptor open
                    for f in os.listdir(p):
                        fp = os.path.join(p, f)
                        try:
                            with open(fp, "r+b") as fh:
                                fh.truncate(0)
                        except OSError:
                            pass
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.unlink(p)
            except OSError:
                pass


@pytest.fixture(scope="session", autouse=True)
def _weg2_shm_ratchet():
    """THE RATCHET: zero own-pid weg2 residue when the session ends.

    Fails the session rather than warning.  A warning is what the previous two
    sightings effectively were -- the residue was visible in every boot
    pre-flight and got swept by hand each time instead of fixed here.
    """
    yield
    left = _own_residue()
    assert not left, (
        "#1344 RATCHET: the weg2 suite left its own /dev/shm residue behind: "
        + ", ".join(os.path.basename(p) for p in left)
        + " -- clean it in the fixture that creates the region, not in a boot "
          "seat's pre-flight"
    )
