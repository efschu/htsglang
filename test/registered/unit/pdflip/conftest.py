"""#1344: the pdflip suite must not leave `/dev/shm` residue behind.

THIRD SIGHTING, and each one cost a boot seat pre-flight time: boots weg2xsn17
and weg2xsn18 both opened with 12-20 dead `/dev/shm/pdflip-xchg-s4b<pid>x<n>`
directories to sweep, and the producer turned out to be THIS SUITE, not a
crashed serving process.  `_fresh_boot()` (test_pdflip_xchg_transport_1273.py)
mints a nonce `s4b<pid>x<n>`, tests open a region under the real
`weight_exchange_region.SHM_ROOT` with it, and nothing removed it afterwards.

Precedent #438(b): residue with a producer gets cleaned at the producer, and a
RATCHET asserts the absence so the third sighting is the last.

THE SWEEP IS BOUND TO THIS PROCESS'S OWN PID, and that bound is the whole
safety argument.  `#1217`/`#1310` are the incidents where a broad `/dev/shm`
sweep ran against a LIVE holder; a serving boot's region is named
`pdflip-xchg-<epoch>` and a test's is `pdflip-xchg-s4b<pid>x<n>` with THIS
interpreter's pid, so this fixture cannot reach a boot's region even if one is
running concurrently -- which on this box it may well be.  No pid scan, no
holder check, no lsof needed: the name itself is the proof of ownership.
"""

import os
import shutil

import pytest

from flliper.srt.pdflip import weight_exchange_region as xr


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
def _pdflip_shm_residue_guard():
    """Remove this process's own region residue after every pdflip test.

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
def _pdflip_shm_ratchet():
    """THE RATCHET: zero own-pid pdflip residue when the session ends.

    Fails the session rather than warning.  A warning is what the previous two
    sightings effectively were -- the residue was visible in every boot
    pre-flight and got swept by hand each time instead of fixed here.
    """
    yield
    left = _own_residue()
    assert not left, (
        "#1344 RATCHET: the pdflip suite left its own /dev/shm residue behind: "
        + ", ".join(os.path.basename(p) for p in left)
        + " -- clean it in the fixture that creates the region, not in a boot "
          "seat's pre-flight"
    )


#: H25 (Nutzer-Order 24.09. 08:25Z): ``FLLIPER_PDFLIP_DRAFT_ON_P`` defaults to
#: False since the order -- group P carries no draft and ``weights_draft``
#: leaves the weights family.  The tests of this suite written BEFORE H25 pin
#: the form they were written against (draft on P, the tag in the family),
#: which stays a supported arm under ``FLLIPER_PDFLIP_DRAFT_ON_P=1``.  A module
#: that tests the H25 form itself sets ``H25_OWN_DRAFT_FORM = True`` and pins
#: its own value per case.  Saved and restored around every case, because
#: ``launcher.main`` PUBLISHES the resolved value into ``os.environ``.
_DRAFT_ON_P_ENV = "FLLIPER_PDFLIP_DRAFT_ON_P"


@pytest.fixture(autouse=True)
def _pdflip_draft_on_p_form(request):
    backup = os.environ.get(_DRAFT_ON_P_ENV)
    if request.module.__dict__.get("H25_OWN_DRAFT_FORM", False):
        os.environ.pop(_DRAFT_ON_P_ENV, None)
    else:
        os.environ[_DRAFT_ON_P_ENV] = "1"
    try:
        yield
    finally:
        if backup is None:
            os.environ.pop(_DRAFT_ON_P_ENV, None)
        else:
            os.environ[_DRAFT_ON_P_ENV] = backup


#: The boot form ``launcher.main`` PUBLISHES into ``os.environ``
#: (pdflip/form.py FORM_ENV) and the speculative form it installs into
#: ``launcher._SPEC_FORM`` outlive the case that ran ``main`` -- every later
#: module then reads the registry defaults of THAT boot's profile. Measured
#: 29.09.: once the qwen27b row turned --d-replayssm-spec and X-EXACT on, the
#: dry-run cases (#1378, #1386, #1392) leaked d_replayssm_spec=on and the
#: qwen27b form into NF wake-credit (h14, pd_h34), pd_free0 h92d and X-EXACT
#: cases. Saved and restored around every case, like the H25 value above.
_FORM_ENV = "FLLIPER_PDFLIP_FORM"


@pytest.fixture(autouse=True)
def _pdflip_published_form():
    from flliper.srt.pdflip import launcher  # imported by nearly every case anyway

    backup = os.environ.get(_FORM_ENV)
    spec = dict(launcher._SPEC_FORM)
    try:
        yield
    finally:
        if backup is None:
            os.environ.pop(_FORM_ENV, None)
        else:
            os.environ[_FORM_ENV] = backup
        launcher._SPEC_FORM.clear()
        launcher._SPEC_FORM.update(spec)


@pytest.fixture(autouse=True)
def _pdflip_decode_collect_old_path(monkeypatch):
    """SET SCHEDULER (SOLL v2, 17.10.) ships ON with T_halb = 4.5 s. The suite's arrival / park / flip tests pin the
    immediate paths they were written for (no wait while D decodes), so T_halb is 0 -- the set decides at once -- unless
    a test sets it. The old DECODE-COLLECT window is gone on the flip path (27B port: pinned 0 as before for any 27B
    reader left, e.g. the dual layout)."""
    if "FLLIPER_PDFLIP_HALF_FLIP_S" not in os.environ:
        # A7: ``FLLIPER_PDFLIP_TEST_T_HALF=4.5 pytest <arrival/flip files>`` runs the same files with the production T_halb
        monkeypatch.setenv("FLLIPER_PDFLIP_HALF_FLIP_S", os.environ.get("FLLIPER_PDFLIP_TEST_T_HALF", "0"))
    if "FLLIPER_PDFLIP_DECODE_COLLECT_WINDOW_S" not in os.environ:
        monkeypatch.setenv("FLLIPER_PDFLIP_DECODE_COLLECT_WINDOW_S", "0")
    yield


@pytest.fixture
def decider_off():
    """EZ-S2a: pins ``FLLIPER_PDFLIP_DECIDER`` OFF for a test that describes the S1 set code (the queue-sum second set, FLIP-ECONOMICS as
    a verdict, P-by-state under X). The old code stays reachable until S4 (plan A13); the new behaviour is pinned by
    ``test_decide_set_1019``."""
    from flliper.srt.environ import envs
    with envs.FLLIPER_PDFLIP_DECIDER.override(False):
        yield
# ---------------------------------------------------------------------------------------------------------------------
# S109b seam (NF 20.10.): the spills read the write-behind's shared record and only QUEUE a missing L3 copy
# (mem_cache/l3_secured; user 20.10.: no synchronous write into the cache in the scheduler thread). The tests of the
# spill LOGIC were written against the copy existing; they keep that premise by running "the write-behind" at once:
# ``secure_rows_to_l3`` / ``_secure_child_anchors`` are the writer's forms (``*_now``) here. A module that tests the
# asynchronous behaviour itself sets ``L3_ASYNC_REAL = True`` and keeps the scheduler forms.
import pytest as _pytest_s109b  # noqa: E402


@_pytest_s109b.fixture(autouse=True)
def _s109b_write_behind_instant(request, monkeypatch):
    if getattr(request.module, "L3_ASYNC_REAL", False):
        yield
        return
    try:
        from flliper.srt.mem_cache.pool_host import arena_pool as _ap
        from flliper.srt.pdflip import forma_l3_subtree as _fls
    except Exception:  # noqa: BLE001 -- a module without these imports is not affected
        yield
        return
    if hasattr(_ap.ArenaMHAHostPool, "secure_rows_to_l3_now"):
        monkeypatch.setattr(_ap.ArenaMHAHostPool, "secure_rows_to_l3", _ap.ArenaMHAHostPool.secure_rows_to_l3_now)
    if hasattr(_fls, "_secure_child_anchors_now"):
        monkeypatch.setattr(_fls, "_secure_child_anchors", _fls._secure_child_anchors_now)
    try:
        from flliper.srt.mem_cache import l3_secured as _l3s

        monkeypatch.setattr(_l3s, "check", _l3s.check_now)
    except Exception:  # noqa: BLE001
        pass
    yield
