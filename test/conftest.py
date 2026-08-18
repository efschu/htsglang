"""Tree-wide pytest guards.

Currently:
* neutralize process-wide torch state leaked during collection;
* fail the run if it wrote to the cross-session GPU arbitration paths (#438).
"""

import os
import sys

import pytest

#: The cross-session arbitration paths a test run must never write. They are
#: shared with every other session on the box: a card lock is acquired with
#: ``mkdir`` and a holder is a claim line, so a test that plants either one
#: blocks somebody else's window with a claim nobody can explain or release.
#: #438 chased exactly that -- four 0-byte regular files with identical
#: nanosecond mtimes, the signature of one ``touch a b c d``.
_ARB_PATHS = (
    "/tmp/gpu-card-0.lock",
    "/tmp/gpu-card-1.lock",
    "/tmp/gpu-card-2.lock",
    "/tmp/gpu-owner.lock",
    "/tmp/gpu-quiet.lock",
    "/spinning/gpu-arb/holder",
)


def _arb_state():
    """``{path: (exists, is_dir)}``.

    Deliberately not mtime, and deliberately NOT the inode either. The
    inode was here on the reasoning that "creation, deletion, and a change
    of type or inode cannot happen [from a foreign heartbeat]". That claim
    is false and was falsified in practice (#654): a window held by ANOTHER
    session refreshes its holder by writing a temp file and renaming it
    over the target, which is the correct atomic way to do it and which
    mints a NEW inode every time. A long test run overlapping a foreign
    heartbeat therefore failed at sessionfinish with zero test failures --
    the guard accusing the run of a write it never made.

    Existence and type are the only invariants a foreign writer cannot
    disturb, so they are all this guard asserts. A content hash would have
    the same false positive as the inode, for the same reason.

    The trade, named: a test that REPLACED the holder via rename would no
    longer be caught here. The guard's primary purpose -- catching tests
    that CREATE arbitration files (the 0-byte-holder class, #438b) or
    delete them -- is unaffected.
    """
    state = {}
    for path in _ARB_PATHS:
        try:
            os.stat(path, follow_symlinks=False)
        except OSError:
            state[path] = (False, False)
        else:
            state[path] = (True, os.path.isdir(path))
    return state


def pytest_sessionstart(session):
    session.config._arb_state_at_start = _arb_state()


def pytest_sessionfinish(session, exitstatus):
    # #749 first: this block must run whatever the arb guard below decides,
    # and the arb guard returns early on the common path.
    if _LEAKS_SEEN:
        lines = [f"  {nodeid}: {what}" for nodeid, what in _LEAKS_SEEN]
        print(
            "\n#749 PROCESS-GLOBAL LEAK GUARD: the test(s) below finished with "
            "a module attribute still replaced. A mock.patch was not restored "
            "-- most often because it was entered from inside a worker THREAD, "
            "where mock's restore stack races between ranks. Patch process "
            "globals ONCE, in the main thread, around the threaded section.\n"
            "The baseline was restored after each, so the failures you see "
            "elsewhere in this run are NOT consequences of these:\n"
            + "\n".join(lines),
            file=sys.stderr,
        )
        session.exitstatus = 1

    before = getattr(session.config, "_arb_state_at_start", None)
    if before is None:
        return
    after = _arb_state()
    leaked = {p: (before[p], after[p]) for p in before if before[p] != after[p]}
    if not leaked:
        return
    lines = [f"  {p}: {was} -> {now}" for p, (was, now) in sorted(leaked.items())]
    print(
        "\nARB PATH GUARD: this run wrote to the cross-session GPU "
        "arbitration paths. Redirect them (HTSGLANG_CARD_LOCK_ROOT, "
        "ArbDirectory(root=...), BATTERY_LOCK_ROOT) instead of using the "
        "production defaults:\n" + "\n".join(lines),
        file=sys.stderr,
    )
    session.exitstatus = 1


def pytest_collection_finish(session):
    """Reset the torch default device before any test runs.

    test/registered/unit/batch_invariant_ops/test_batch_invariant_ops.py
    calls ``torch.set_default_device(<accelerator>)`` at module scope, so it
    executes during pytest collection and stays set for the whole process.
    On a machine without a usable accelerator (e.g. CUDA build with
    ``CUDA_VISIBLE_DEVICES`` pointing at no device) every later tensor
    construction through the default device then fails in whichever test
    modules happen to be collected afterwards -- visible as exhausted
    ``retry()`` loops far from the root cause, with the victim set changing
    whenever collection order changes (task #249: 12 phantom regressions in
    a merge validation run, all green standalone).

    ``torch.set_default_device(None)`` restores the pristine state, so the
    reset is safe when nothing leaked.  ``torch.get_default_device()`` is
    deliberately not consulted first: it constructs a tensor on the default
    device and would itself crash on the polluted no-GPU path.

    Runtime leaks (``setUpClass`` setting the default device without a
    teardown reset) are outside this hook's reach; they only bite GPU runs
    and are tracked separately.
    """
    torch = sys.modules.get("torch")
    if torch is not None and hasattr(torch, "set_default_device"):
        torch.set_default_device(None)


# ---------------------------------------------------------------------------
# #749: process-global leak guard
# ---------------------------------------------------------------------------
#
# `unittest.mock.patch` mutates a module attribute and restores it on exit. It
# is NOT thread-safe, and `test_collective_family_siblings_610.py` used to
# enter three such patches from inside every rank thread of a 3-thread
# harness. The restore stack raced; one lost restore left
# `sglang.srt.distributed.utils.uneven_dcp_active` permanently `lambda: True`
# for the rest of the process.
#
# The damage was entirely elsewhere and looked nothing like its cause: every
# later test then took `scheduler.py:4615`'s pinned-admission branch and asked
# its tree-cache fake for `evictable_size()` at :4622, which the minimal fakes
# in the distributed/ collective-floor suites do not implement. ~50 failures
# across six different error signatures, none of them near the leak -- and
# INTERMITTENT, because whether a restore is lost depends on thread
# interleaving. The same commit produced 0 and 50 failures on two runs of one
# command, which is what made it look like a code regression for a while.
#
# THIS GUARD ASSERTS RATHER THAN CLEANS, and the distinction is the point: a
# silent reset would keep the suite green while leaving the next leaker
# undiscovered. It names the test that leaked, at the moment it leaked.
#
# It also REPAIRS after reporting. Failing only the leaker and letting the run
# continue on a clean baseline is deliberate -- the alternative is the very
# cascade this exists to prevent, where one bad teardown produces fifty
# downstream failures that bury it.
_LEAK_SENTINELS = (
    ("sglang.srt.distributed.utils", "uneven_dcp_active"),
    ("torch.distributed", "all_reduce"),
    ("torch.distributed", "get_world_size"),
)

#: Snapshotted at SETUP, compared at TEARDOWN. An earlier version recorded the
#: baseline lazily at teardown, which is wrong in the one case that matters:
#: the FIRST test to touch a module is also the first that can leak it, and its
#: leaked value was then recorded as the baseline. The guard's own can-fail
#: test caught that -- which is the argument for writing the can-fail test.
_LEAK_BASELINE = {}


def _leak_sentinel_state():
    state = {}
    for modname, attr in _LEAK_SENTINELS:
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        try:
            state[(modname, attr)] = getattr(mod, attr)
        except AttributeError:
            continue
    return state


#: Leaks seen this session, as ``(nodeid, description)``. Collected rather than
#: raised on the spot: raising inside ``pytest_runtest_teardown`` aborts the
#: teardown chain and pytest then errors the NEXT item with "previous item was
#: not torn down properly" -- turning one leak into two failures, which is the
#: cascade this guard exists to end. The session fails at the end instead, with
#: every leaker named.
_LEAKS_SEEN = []


def pytest_runtest_setup(item):
    """Snapshot the sentinels BEFORE the test runs (#749)."""
    _LEAK_BASELINE.clear()
    _LEAK_BASELINE.update(_leak_sentinel_state())


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """Record and REPAIR a process-global left patched (#749).

    ``trylast`` is load-bearing: pytest runs fixture finalizers from its OWN
    ``pytest_runtest_teardown``, so a hook that ran first would see
    ``monkeypatch``'s patches still in place and accuse honest tests. The guard
    must observe the state AFTER every finalizer has had its turn.

    Deliberately does not raise; see ``_LEAKS_SEEN``.
    """
    for key, value in _leak_sentinel_state().items():
        if key not in _LEAK_BASELINE:
            # Imported DURING this test, so there is no pre-test value to
            # compare against. Not a leak.
            continue
        if _LEAK_BASELINE[key] is not value:
            modname, attr = key
            _LEAKS_SEEN.append(
                (item.nodeid, f"{modname}.{attr} -> {value!r}")
            )
            # Repair, so this leak does not become the next fifty failures.
            setattr(sys.modules[modname], attr, _LEAK_BASELINE[key])
