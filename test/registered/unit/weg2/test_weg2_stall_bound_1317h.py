"""#1317h -- the flip-stall bound could not be satisfied by a healthy flip.

THE TWO RECORDED FIRES, boot weg2sn6g
(`boot_weg2_weg2sn6g_004b8df192_0910_093559.front.log`), both with the boot
healthy and serving:

    WEG2-FLIP STALL epoch=4 elapsed=17.5 s bound=14.1 s stage=drain awake=D queue=0
    WEG2-FLIP STALL epoch=6 elapsed=16.1 s bound=13.5 s stage=drain awake=D queue=1

Both fired IN THE DRAIN STAGE. The old bound was `FLIP_STALL_SLACK * last
flip_ms` = 4 x ~3.4 s, and `flip_ms` is measured AFTER the drain has already
succeeded -- so the drain's own cost was never in the bound at all. A drain
may legitimately wait up to `drain_deadline_s` (120 s) for a decode that
#1011 forbids cutting, so a ~14 s bound convicts every flip that queues
behind real work.

TWO MECHANISMS, and the file pins both because either alone leaves a hole:
* the BOUND is now the drain distribution (p99 of this boot's own drains) plus
  the measured flip -- both halves measured on this boot;
* and at the bound, LIVE PROGRESS decides. A flip whose group is still
  emitting is WAITING, not stalled. This is the half that saves the two fires
  above even once the drain distribution is fast and the bound is small again.

MUTANTS -- the danger direction here is a detector talked out of firing:
  M1  a genuinely stalled flip must STILL latch
      -> test_a_flip_with_no_progress_still_latches
  M2  an unreadable progress sample must not suppress the latch
      -> test_an_unreadable_progress_sample_does_not_suppress
  M3  a sample from BEFORE this flip must not credit it
      -> test_a_sample_older_than_the_flip_is_not_credited
  M4  forward passes without tokens must not count as progress (inherited
      from #1317c, re-asserted here because THIS caller is new)
      -> test_forward_passes_alone_do_not_suppress_the_latch
"""

import collections

import pytest

from sglang.srt.weg2 import front as F

# The two fires, verbatim.
FIRE_1 = dict(epoch=4, elapsed=17.5, old_bound=14.1, stage="drain", queue=0)
FIRE_2 = dict(epoch=6, elapsed=16.1, old_bound=13.5, stage="drain", queue=1)
SN6G_FLIP_MS = 3400.0        # ~3.4 s; 4x it is the ~13.6 s the old bound used


def _front(drains=(), flip_ms=SN6G_FLIP_MS, deadline=120.0):
    f = F.Front.__new__(F.Front)
    f.drain_deadline_s = deadline
    f.flip_log = [{"flip_ms": flip_ms, "sleep": "D", "wake": "P"}] if flip_ms else []
    f._drain_log = collections.deque(drains, maxlen=F.DRAIN_LOG_MAX)
    f.counters = collections.Counter()
    f.epoch = 4
    f.awake = "D"
    f.queue = []
    f.state = "flipping"
    f._flip_t0 = 1000.0
    f._flip_stage = "drain"
    f._flip_stall_reported_epoch = None
    f._flip_progress_seen = None
    f._flip_progress_start = None
    f.flip_stage_age_s = lambda: 0.0
    return f


# --------------------------------------------------------------------------
# the bound
# --------------------------------------------------------------------------

def test_the_old_bound_is_reproduced_so_the_delta_is_visible():
    """4 x 3.4 s = 13.6 s, which is where both recorded bounds sit."""
    assert F.FLIP_STALL_SLACK * SN6G_FLIP_MS / 1000.0 == pytest.approx(13.6, abs=0.1)
    for fire in (FIRE_1, FIRE_2):
        assert 13.0 < fire["old_bound"] < 15.0
        assert fire["elapsed"] > fire["old_bound"], "both fires overran the old bound"


def test_a_young_drain_distribution_falls_back_to_the_declared_deadline():
    """Under 4 samples is not a distribution; the front's own published
    `drain_deadline_s` stands in -- the same fallback the previous revision
    used for epoch 0, and the ceiling every drain is already refused against."""
    for n in (0, 1, 2, 3):
        f = _front(drains=[2.5] * n)
        bound, src = f._drain_p99_s()
        assert bound == 120.0, n
        assert f"only {n} drain sample" in src


def test_neither_recorded_fire_survives_the_new_bound_on_a_young_boot():
    """THE RED-FIRST ASSERTION. Both fires happened within the first handful of
    flips, so the drain distribution was young and the bound is
    120 + 3.4 = 123.4 s. 17.5 s and 16.1 s are nowhere near it."""
    f = _front(drains=[2.4, 3.1])          # 2 samples -> fallback
    bound, _ = f._flip_stall_bound_s()
    assert bound == pytest.approx(123.4, abs=0.1)
    for fire in (FIRE_1, FIRE_2):
        assert fire["elapsed"] < bound, fire


def test_the_bound_is_the_drain_p99_plus_the_measured_flip():
    f = _front(drains=[2.0, 2.2, 2.4, 9.0])
    bound, src = f._flip_stall_bound_s()
    assert bound == pytest.approx(9.0 + 3.4, abs=0.05)
    assert "drain p99" in src and "+ flip" in src
    assert "#1011" in src, "the law that licenses a waiting drain is not named"


def test_p99_equals_max_below_a_hundred_samples_and_says_so():
    """Stated because the docstring used to claim trimming it does not yet do:
    ceil(0.99*n)-1 is the last index for n < 100, so this is a max until the
    distribution is large -- the conservative direction."""
    f = _front(drains=[2.4] * 24 + [118.0])
    bound, src = f._drain_p99_s()
    assert bound == 118.0
    assert "max 118.0 s" in src
    import inspect
    doc = inspect.getdoc(F.Front._drain_p99_s)
    assert "EQUALS the max" in doc


# --------------------------------------------------------------------------
# live progress at the bound
# --------------------------------------------------------------------------

def _emit(f, tokens_before, tokens_after, seen_at=1010.0):
    f._flip_progress_start = {"gen_tokens_total": tokens_before,
                              "prefill_tokens_total": 0, "forward_ct": 10}
    f._flip_progress_seen = (seen_at, {"gen_tokens_total": tokens_after,
                                       "prefill_tokens_total": 0,
                                       "forward_ct": 99})


def test_a_flip_whose_group_is_still_emitting_is_not_latched():
    """Even once the drain distribution is fast and the bound is small again,
    a flip behind a live decode must not be convicted -- #1011."""
    f = _front(drains=[2.0, 2.1, 2.2, 2.3])       # bound = 2.3 + 3.4 = 5.7 s
    _emit(f, 1000, 9000)
    assert f.flip_stall_check(now=f._flip_t0 + 17.5) is None
    assert f.counters["flip_stall_waiting"] == 1
    assert f.counters["flip_stall"] == 0


def test_a_flip_with_no_progress_still_latches():
    """M1. The teeth."""
    f = _front(drains=[2.0, 2.1, 2.2, 2.3])
    _emit(f, 1000, 1000)
    line = f.flip_stall_check(now=f._flip_t0 + 17.5)
    assert line is not None and "WEG2-FLIP STALL" in line
    assert f.counters["flip_stall"] == 1


def test_an_unreadable_progress_sample_does_not_suppress(caplog=None):
    """M2. None means "cannot say", and a detector may never be silenced by
    an unreadable instrument."""
    f = _front(drains=[2.0, 2.1, 2.2, 2.3])
    f._flip_progress_start = None
    f._flip_progress_seen = (1010.0, None)
    assert f.flip_stall_check(now=f._flip_t0 + 17.5) is not None


def test_a_sample_older_than_the_flip_is_not_credited():
    """M3. A reading from the previous flip describes the previous flip."""
    f = _front(drains=[2.0, 2.1, 2.2, 2.3])
    _emit(f, 1000, 9000, seen_at=f._flip_t0 - 5.0)     # BEFORE this flip began
    assert f.flip_stall_check(now=f._flip_t0 + 17.5) is not None


def test_forward_passes_alone_do_not_suppress_the_latch():
    """M4. A livelock spinning forward passes and emitting nothing is exactly
    what this detector exists to catch. Inherited from #1317c's predicate and
    re-asserted here because THIS caller is new."""
    f = _front(drains=[2.0, 2.1, 2.2, 2.3])
    f._flip_progress_start = {"gen_tokens_total": 500, "prefill_tokens_total": 0,
                              "forward_ct": 10}
    f._flip_progress_seen = (1010.0, {"gen_tokens_total": 500,
                                      "prefill_tokens_total": 0,
                                      "forward_ct": 9000})
    assert f.flip_stall_check(now=f._flip_t0 + 17.5) is not None


def test_the_latch_still_fires_only_once_per_flip():
    f = _front(drains=[2.0, 2.1, 2.2, 2.3])
    _emit(f, 1000, 1000)
    assert f.flip_stall_check(now=f._flip_t0 + 17.5) is not None
    assert f.flip_stall_check(now=f._flip_t0 + 30.0) is None


# --------------------------------------------------------------------------
# the deadman follows the front's line (no second timer)
# --------------------------------------------------------------------------

def test_the_front_emits_exactly_the_marker_the_deadman_keys_on():
    """Item 3 of the spec needed CITATION, not code -- and the citation lives
    in the RECORD, not in this file, because it is about a file outside the
    repo.

    VERIFIED BY HAND (2026-09-10) in `/spinning/gpu-arb/devtools/boot_deadman.sh`:
    its header line 17 names tier 3 as *"the Weg-2 front's OWN 'WEG2-FLIP
    STALL' line, read from the log"*, its trigger sets `FLIP_STALL_SEEN=1` off
    that string (:311), and `FLIP_STALL_SLACK` appears NOWHERE in the script --
    so the deadman carries no flip timer of its own and moving the bound in the
    front moves it everywhere.

    WHAT THIS TEST CAN HONESTLY ASSERT is the repo's own half: that the front
    still emits that exact marker. An earlier revision read the script by
    absolute path and went red on the remote desk with FileNotFoundError --
    a hermetic test may not depend on a path outside the worktree, and a
    `skipif` there would have proved nothing while looking green."""
    import inspect

    src = inspect.getsource(F.Front.flip_stall_check)
    assert "WEG2-FLIP STALL" in src, (
        "the deadman's tier-3 trigger keys on this exact string; renaming it "
        "disarms the watcher silently"
    )


# --------------------------------------------------------------------------
# the manual-flip admit_d leak
# --------------------------------------------------------------------------

def test_manual_flip_restores_admit_d_to_its_previous_value():
    """`admit_d` was set False and NEVER restored, so ONE manual flip left D
    admission off for the rest of the boot. Restored to its PREVIOUS value,
    not to True: the caller may legitimately have had it off."""
    import inspect

    src = inspect.getsource(F.Front.handle_manual_flip)
    assert "_admit_before = self.admit_d" in src
    assert "finally:" in src
    assert "self.admit_d = _admit_before" in src
    i_try = src.index("try:")
    i_fin = src.index("finally:")
    assert i_try < i_fin
