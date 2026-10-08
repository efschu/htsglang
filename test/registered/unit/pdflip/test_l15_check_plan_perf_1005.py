"""stratified_check_plan was O(n * guests) (``e not in in_guest``): 36 s inside the
wake RPC at ~170k plan rows with 75k guest rows (j4, 05.10.). Same result, linear time."""
import time

from flliper.srt.pdflip import l15_pool


def _old(plan, ranges, k, guest_share=0.5):
    entries = sorted(plan, key=lambda e: int(e[1]))
    in_guest = [e for e in entries if any(lo <= int(e[1]) < hi for lo, hi in ranges)]
    if not in_guest:
        return l15_pool._even(entries, k)
    home = [e for e in entries if e not in in_guest]
    kg = min(len(in_guest), max(1, int(k * guest_share)))
    kh = min(len(home), max(0, k - kg))
    kg = min(len(in_guest), k - kh)
    return sorted(l15_pool._even(in_guest, kg) + l15_pool._even(home, kh),
                  key=lambda e: int(e[1]))


def _plan(n):
    return [("rid%d" % (i // 1000), i, 1000 + i, 1) for i in range(n)]


def test_same_result_as_the_quadratic_version():
    for n, ranges, k in ((500, [(0, 120)], 16), (500, [(100, 200), (300, 350)], 16),
                         (50, [], 16), (40, [(0, 40)], 16), (7, [(2, 4)], 16)):
        plan = _plan(n)
        assert l15_pool.stratified_check_plan(plan, ranges, k) == _old(plan, ranges, k)


def test_linear_time_at_wake_scale():
    plan = _plan(172_051)
    t0 = time.monotonic()
    out = l15_pool.stratified_check_plan(plan, [(0, 74_888)], 32)
    dt = time.monotonic() - t0
    assert len(out) == 32
    assert dt < 3.0, "stratified_check_plan took %.1f s (was 36 s)" % dt
