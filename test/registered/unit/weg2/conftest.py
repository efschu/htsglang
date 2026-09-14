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

#1344 ROUND 2 (2026-09-14), MEASURED-BY-EXECUTION, not read from the code
alone: `_own_residue()` matched ONLY the bare region directory
(`weg2-xchg-<nonce>`) and missed TWO OTHER artefact classes a test can leave,
proven by deliberately calling `xr.create_semaphores(nonce)` at the desk and
observing that the resulting names survived a full pytest run with this
guard active, green, no ratchet failure:

  * SEMAPHORE-BACKING FILES. `sem_open`'s Linux backing file is `/dev/shm/
    sem.<name>`, and every one of `create_semaphores`' ~60 names already
    carries `weight_exchange_region.REGION_PREFIX` INSIDE the logical name
    (`weg2-xchg-<nonce>-0-1-0-empty`, ...) -- but the FILESYSTEM entry is
    `sem.weg2-xchg-<nonce>-0-1-0-empty`, which does not START WITH
    `weg2-xchg-<nonce>` (it starts with `sem.`), so the old prefix check
    could never see it. A test that opens semaphores WITHOUT going on to
    open a full region (or one whose own `xr.unlink_semaphores` call is
    skipped by an exception path) leaves these with nobody watching.
  * THE BOUNCE SLOT TABLE. `weight_exchange_bounce.bounce_slots_path`
    writes `weg2-xchg-bnc-<nonce>` directly under `shm_root` -- #1363's own
    finding, reused here rather than rediscovered: `weight_exchange_bounce.
    BOUNCE_SLOT_PREFIX` ("weg2-xchg-bnc-") is a DIFFERENT string than
    `weight_exchange_region.REGION_PREFIX` ("weg2-xchg-") plus the nonce,
    because of the `bnc-` infix, so `n.startswith(mine)` never matched it
    either.

Both are still bound to THIS PROCESS'S OWN PID (the nonce, `s4b{pid}x`, is
embedded inside both), so the safety argument two paragraphs up is
unchanged -- these are two MORE names this process, and only this process,
could have created, not a widened sweep.
"""

import os
import shutil

import pytest

from sglang.srt.weg2 import weight_exchange_bounce as wb
from sglang.srt.weg2 import weight_exchange_region as xr


#: KNOWN NONCE CONVENTIONS this suite mints, each a callable `pid ->
#: infix` that embeds THIS interpreter's own pid as a bounded token
#: (never a bare substring search over an unbounded pattern -- the
#: holder-scoping argument the module docstring makes needs the pid to be
#: a real, position-anchored part of the name). A NEW test file's own
#: convention is a ONE-LINE, NAMED addition here, sourced from an actually
#: OBSERVED nonce, never a guess:
#:   * `s4b{pid}x` -- test_weg2_xchg_transport_1273.py's `_fresh_boot()`,
#:     the ORIGINAL #1344 finding.
#:   * `desk10-1391-r4-{pid}-` -- test_weg2_undrained_lane_refusal_1391.py's
#:     `_run_two_tag_leg` (#1344 round 2, found while verifying THIS
#:     round's own fix against the full suite: `mgr._weg2_xchg_bounce_leg`
#:     is called there without `shm_root=`, so its internal `BounceSlots`/
#:     region land on the REAL `xr.SHM_ROOT` instead of the test's own
#:     `tmp_path` -- the identical #1363 class, in weight_updater.py's own
#:     test file, DESK10's file boundary; reported, not fixed here).
_KNOWN_NONCE_INFIXES = (
    lambda pid: f"s4b{pid}x",
    lambda pid: f"desk10-1391-r4-{pid}-",
)


def _own_residue():
    """Every `/dev/shm` entry this interpreter's own KNOWN nonce
    conventions (`_KNOWN_NONCE_INFIXES`) could have created -- for each,
    the region directory, ITS semaphores' backing files, and the bounce
    slot table (#1344 round 2; see the module docstring for why the naive
    region-directory-only check missed the latter two).

    Reads `xr.SHM_ROOT` live rather than caching it: a test that monkeypatches
    the constant to `tmp_path` gets its residue cleaned by pytest instead, and
    this must not then sweep the real root under a stale assumption.
    """
    root = getattr(xr, "SHM_ROOT", "/dev/shm")
    pid = os.getpid()
    try:
        names = os.listdir(root)
    except OSError:
        return []
    found = []
    for infix_fn in _KNOWN_NONCE_INFIXES:
        infix = infix_fn(pid)
        prefixes = (
            f"{xr.REGION_PREFIX}{infix}",       # weg2-xchg-<infix>...
            f"sem.{xr.REGION_PREFIX}{infix}",   # sem.weg2-xchg-<infix>...
            f"{wb.BOUNCE_SLOT_PREFIX}{infix}",  # weg2-xchg-bnc-<infix>...
        )
        found.extend(os.path.join(root, n) for n in names
                    if any(n.startswith(p) for p in prefixes))
    return found


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

    THE REPORT LINE MATCHES `teardown_region`'s OWN FIELD SHAPE (`epoch=
    slots_unlinked= foreign_skipped=`, weight_exchange_region.py) rather than
    inventing a second one for the same question -- "which epoch, how many of
    its own names were found, how many OTHER names exist that this check
    never touched". `foreign_untouched` is a NAMED zero-by-construction, not a
    count taken and discarded: `_own_residue()` never lists a name outside
    this process's own `s4b{pid}x` nonce family in the first place (the
    holder-scoping rule this whole file exists to keep), so there is nothing
    for this line to have swept even if it wanted to.
    """
    yield
    left = _own_residue()
    assert not left, (
        f"#1344 RATCHET: epoch=s4b{os.getpid()}x found={len(left)} "
        f"foreign_untouched=0(by-construction) names="
        + ",".join(os.path.basename(p) for p in left)
        + " -- clean it in the fixture/test that creates the region or "
          "semaphores, not in a boot seat's pre-flight"
    )
