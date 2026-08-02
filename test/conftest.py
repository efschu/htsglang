"""Tree-wide pytest guards.

Currently:
* neutralize process-wide torch state leaked during collection;
* fail the run if it wrote to the cross-session GPU arbitration paths (#438).
"""

import os
import sys

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
    """``{path: (exists, is_dir, inode)}``.

    Deliberately not mtime: a window held by ANOTHER session runs its own
    heartbeat, which legitimately refreshes an existing holder while an
    unrelated test run is in progress. Creation, deletion, and a change of
    type or inode cannot happen that way -- each of those is a write, and a
    write to these paths from inside pytest is the bug this guard is for.
    """
    state = {}
    for path in _ARB_PATHS:
        try:
            st = os.stat(path, follow_symlinks=False)
        except OSError:
            state[path] = (False, False, None)
        else:
            state[path] = (True, os.path.isdir(path), st.st_ino)
    return state


def pytest_sessionstart(session):
    session.config._arb_state_at_start = _arb_state()


def pytest_sessionfinish(session, exitstatus):
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
