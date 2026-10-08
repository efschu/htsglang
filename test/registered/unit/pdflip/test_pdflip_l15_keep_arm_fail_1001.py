"""AP L15-12c-F2: a failed keep-span arm at sleep must not leave a
manifest that claims a hold.

Finding F2 (L15-RETAIN-E2E-REVIEW): after retain_at_sleep has moved the
held rows and written this rank's manifest, the scheduler hook arms the
keep byte spans per allocation base and RAISED on rc != 0 -- the pause
then discards the held pages while the manifest on disk still says
"held", so the wake restores garbage KV as a prefix hit. The arm step
(extracted to flliper.srt.pdflip.l15_keep_arm.arm_keep_spans) must instead:
discard this rank's manifest, clear the keep sets already armed on the
OTHER bases of this rank (best effort), log ONE failure line and return
False without raising, so the sleep proceeds as a plain flush on this
rank (at the wake it votes None while peers vote a fingerprint -> mixed
-> group fallback, the existing verdict rule).

Hermetic: fake adapter, real manifest file under tmp_path, no GPU.
"""

import json
import os
from types import SimpleNamespace

from flliper.srt.pdflip import l15_keep_arm, l15_manifest, l15_restore
from flliper.srt.pdflip.l15_keep_arm import arm_keep_spans


def _base(i):
    return SimpleNamespace(id=i)


class _Ad:
    """Adapter duck: records (base.id, tuple(spans)) calls. fail_on is the
    base id whose ARM call fails; fail_kind "rc" returns 1, "raise" raises.
    clear_raises: a clear call (empty spans) on that base raises instead of
    returning 0."""

    def __init__(self, fail_on=None, fail_kind="rc", clear_raises=False):
        self.calls = []
        self.fail_on = fail_on
        self.fail_kind = fail_kind
        self.clear_raises = clear_raises

    def set_keep_byte_spans(self, base, spans):
        self.calls.append((base.id, tuple(spans)))
        if not spans:  # a clear call
            if self.clear_raises:
                raise RuntimeError("clear boom")
            return 0
        if self.fail_on is not None and base.id == self.fail_on:
            if self.fail_kind == "raise":
                raise RuntimeError("adapter boom")
            return 1
        return 0


def _manifest(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text("{}")
    return str(p)


def _two_bases():
    return {100: (_base(1), [(4, 8)]), 200: (_base(2), [(16, 20)])}


def test_failed_arm_rc_discards_manifest_and_clears_earlier_bases(tmp_path):
    # Second base returns rc=1: no raise; manifest gone; base 1 (armed
    # before the failure) gets a () clear; the FAILED base is not cleared
    # ("other bases" only); exactly one failure line with the rc.
    mpath = _manifest(tmp_path)
    ad = _Ad(fail_on=2)
    logs = []
    ok = arm_keep_spans(ad, _two_bases(), mpath, rank=3, log=logs.append, granule=1)
    assert ok is False
    assert not os.path.exists(mpath)
    assert (1, ((4, 8),)) in ad.calls  # armed before the failure
    assert (2, ((16, 20),)) in ad.calls  # armed, rc=1
    assert (1, ()) in ad.calls  # earlier base cleared
    assert (2, ()) not in ad.calls
    assert len(logs) == 1
    assert "L15-RETAIN keep-arm FAILED rank=3 rc=1" in logs[0]
    assert "manifest discarded, clears_failed=0, this rank holds nothing" in logs[0]


def test_failed_arm_exception_degrades_the_same_way(tmp_path):
    # An adapter exception mid-arm: same contract as rc != 0.
    mpath = _manifest(tmp_path)
    ad = _Ad(fail_on=2, fail_kind="raise")
    logs = []
    ok = arm_keep_spans(ad, _two_bases(), mpath, rank=0, log=logs.append, granule=1)
    assert ok is False
    assert not os.path.exists(mpath)
    assert (1, ()) in ad.calls
    assert len(logs) == 1
    assert "L15-RETAIN keep-arm FAILED rank=0" in logs[0]


def test_failing_clear_is_best_effort(tmp_path):
    # The () clear itself raising must not propagate: the original failure
    # still wins (False, manifest discarded, one log line).
    mpath = _manifest(tmp_path)
    ad = _Ad(fail_on=2, clear_raises=True)
    logs = []
    ok = arm_keep_spans(ad, _two_bases(), mpath, rank=1, log=logs.append, granule=1)
    assert ok is False
    assert not os.path.exists(mpath)
    assert (1, ()) in ad.calls
    assert len(logs) == 1


class _OsShim:
    """os replacement for l15_keep_arm: unlink always raises unlink_exc (the
    F-B corner -- an OSError other than FileNotFoundError, e.g. EACCES on
    the directory); when write_fails, the os.open of the in-place rewrite
    raises EROFS as well. Everything else delegates to the real os, so the
    fsync/fdopen of a succeeding rewrite are real."""

    def __init__(self, unlink_exc, write_fails=False):
        self._unlink_exc = unlink_exc
        self._write_fails = write_fails

    def unlink(self, path):
        raise self._unlink_exc

    def open(self, path, flags, mode=0o644):
        if self._write_fails:
            raise OSError(30, "Read-only file system")
        return os.open(path, flags, mode)

    def __getattr__(self, name):
        return getattr(os, name)


def test_unlink_failure_invalidates_for_the_wake(tmp_path, monkeypatch):
    # F-B: os.unlink raising PermissionError must NOT leave the manifest
    # claiming a hold. The failure path rewrites the file IN PLACE with a
    # dead-pid tombstone, and the WAKE loader (l15_restore.load_for_wake ->
    # read_and_clear -> read) refuses it: returns None (the no-hold vote,
    # exactly as a real discard) and removes the file.
    mpath = _manifest(tmp_path)
    monkeypatch.setattr(
        l15_keep_arm, "os", _OsShim(PermissionError(13, "Permission denied"))
    )
    ad = _Ad(fail_on=2)
    logs = []
    ok = arm_keep_spans(ad, _two_bases(), mpath, rank=1, log=logs.append, granule=1)
    assert ok is False
    # the file survived the failing unlink but is no longer a hold record:
    assert os.path.exists(mpath)
    rec = json.loads(open(mpath).read())
    assert rec["pid"] == 0  # the dead-pid tombstone
    # the wake refuses it:
    assert l15_restore.load_for_wake(mpath) is None
    assert not os.path.exists(mpath)  # reaped by the read
    assert len(logs) == 1
    assert "manifest invalidated" in logs[0]


def test_double_failure_is_named_but_never_raises(tmp_path, monkeypatch):
    # F-B corner: unlink AND the in-place rewrite both fail. Raising here
    # would leave flush_cache with no handler (the scheduler dispatcher
    # does not catch -> SIGQUIT on that rank while peers finish the sleep)
    # and split the rank group, so the contract stays no-raise: the FAILED
    # line NAMES that the stale record could not be neutralised.
    mpath = _manifest(tmp_path)
    monkeypatch.setattr(
        l15_keep_arm, "os",
        _OsShim(PermissionError(13, "Permission denied"), write_fails=True),
    )
    ad = _Ad(fail_on=2)
    logs = []
    ok = arm_keep_spans(ad, _two_bases(), mpath, rank=2, log=logs.append, granule=1)
    assert ok is False  # no raise: the plain flush still runs on this rank
    assert os.path.exists(mpath)
    assert len(logs) == 1
    assert "NOT invalidated" in logs[0]


def test_clear_failure_is_counted_in_the_failed_line(tmp_path):
    # F-C: a raising () clear is still best effort (no raise, no second
    # log line) but must be COUNTED -- the FAILED line may not silently
    # claim "holds nothing" while a base may still keep pages pinned.
    mpath = _manifest(tmp_path)
    ad = _Ad(fail_on=2, clear_raises=True)
    logs = []
    ok = arm_keep_spans(ad, _two_bases(), mpath, rank=4, log=logs.append, granule=1)
    assert ok is False
    assert len(logs) == 1
    assert "clears_failed=1" in logs[0]


def test_arm_success_keeps_manifest_and_logs_nothing(tmp_path):
    # All rc=0: manifest untouched, no clear calls, no log, True.
    mpath = _manifest(tmp_path)
    ad = _Ad()
    logs = []
    ok = arm_keep_spans(ad, _two_bases(), mpath, rank=2, log=logs.append, granule=1)
    assert ok is True
    assert os.path.exists(mpath)
    # success logs exactly the L15-KEEP-ALIGN accounting line (granule 1:
    # nothing widened, extra 0)
    assert len(logs) == 1 and logs[0].startswith("L15-KEEP-ALIGN rank=2 ")
    assert "extra_mib=0.0" in logs[0]
    assert (1, ((4, 8),)) in ad.calls
    assert (2, ((16, 20),)) in ad.calls
    assert (1, ()) not in ad.calls
