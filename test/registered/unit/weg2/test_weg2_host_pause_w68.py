"""W68 (NF 14.10. 12:17Z): a host freeze is credited back to lane budgets."""

from sglang.srt.weg2 import host_pause as hp


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_normal_slices_are_not_credited():
    c = _Clock()
    dl = hp.PauseAwareDeadline(120.0, 0.5, what="t", clock=c)
    for _ in range(100):
        c.t += 0.6  # slice plus scheduling noise
        dl.left()
    assert dl.credited == 0.0
    c.t += 0.6
    assert dl.left() < 120.0 - 60.0


def test_freeze_is_given_back_and_real_hang_still_expires():
    c = _Clock()
    dl = hp.PauseAwareDeadline(120.0, 0.5, what="t", clock=c)
    c.t += 0.5
    dl.left()
    c.t += 117.0  # the 12:15:38 -> 12:17:35 freeze inside one slice
    left = dl.left()
    assert abs(dl.credited - 116.5) < 1e-6
    assert left >= 119.0 - 1e-6  # only the two slices were charged
    # a peer that stays silent while this process runs still expires
    for _ in range(250):
        c.t += 0.5
        if dl.left() <= 0:
            break
    assert dl.expired()


def test_credit_is_capped(monkeypatch):
    monkeypatch.setenv(hp.ENV_PAUSE_CREDIT_MAX_S, "100")
    c = _Clock()
    dl = hp.PauseAwareDeadline(120.0, 0.5, what="t", clock=c)
    c.t += 1000.0
    dl.left()
    assert dl.credited == 100.0
    assert dl.expired()
