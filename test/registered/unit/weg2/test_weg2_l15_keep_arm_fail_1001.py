"""AP L15-12c-F2: a failed keep-span arm at sleep must not leave a
manifest that claims a hold.

Finding F2 (L15-RETAIN-E2E-REVIEW): after retain_at_sleep has moved the
held rows and written this rank's manifest, the scheduler hook arms the
keep byte spans per allocation base and RAISED on rc != 0 -- the pause
then discards the held pages while the manifest on disk still says
"held", so the wake restores garbage KV as a prefix hit. The arm step
(extracted to sglang.srt.weg2.l15_keep_arm.arm_keep_spans) must instead:
discard this rank's manifest, clear the keep sets already armed on the
OTHER bases of this rank (best effort), log ONE failure line and return
False without raising, so the sleep proceeds as a plain flush on this
rank (at the wake it votes None while peers vote a fingerprint -> mixed
-> group fallback, the existing verdict rule).

Hermetic: fake adapter, real manifest file under tmp_path, no GPU.
"""

import os
from types import SimpleNamespace

from sglang.srt.weg2.l15_keep_arm import arm_keep_spans


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
    ok = arm_keep_spans(ad, _two_bases(), mpath, rank=3, log=logs.append)
    assert ok is False
    assert not os.path.exists(mpath)
    assert (1, ((4, 8),)) in ad.calls  # armed before the failure
    assert (2, ((16, 20),)) in ad.calls  # armed, rc=1
    assert (1, ()) in ad.calls  # earlier base cleared
    assert (2, ()) not in ad.calls
    assert len(logs) == 1
    assert "L15-RETAIN keep-arm FAILED rank=3 rc=1" in logs[0]
    assert "manifest discarded, this rank holds nothing" in logs[0]


def test_failed_arm_exception_degrades_the_same_way(tmp_path):
    # An adapter exception mid-arm: same contract as rc != 0.
    mpath = _manifest(tmp_path)
    ad = _Ad(fail_on=2, fail_kind="raise")
    logs = []
    ok = arm_keep_spans(ad, _two_bases(), mpath, rank=0, log=logs.append)
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
    ok = arm_keep_spans(ad, _two_bases(), mpath, rank=1, log=logs.append)
    assert ok is False
    assert not os.path.exists(mpath)
    assert (1, ()) in ad.calls
    assert len(logs) == 1


def test_arm_success_keeps_manifest_and_logs_nothing(tmp_path):
    # All rc=0: manifest untouched, no clear calls, no log, True.
    mpath = _manifest(tmp_path)
    ad = _Ad()
    logs = []
    ok = arm_keep_spans(ad, _two_bases(), mpath, rank=2, log=logs.append)
    assert ok is True
    assert os.path.exists(mpath)
    assert logs == []
    assert (1, ((4, 8),)) in ad.calls
    assert (2, ((16, 20),)) in ad.calls
    assert (1, ()) not in ad.calls
