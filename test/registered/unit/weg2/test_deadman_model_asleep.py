"""Dual-model: a sleeping model's waiting requests are not a progress stall.

In the dual form requests for the sleeping model wait in its own front until
the arbiter switches (bounded by T_max + one switch). The PROGRESS-STALL
latch (state_file.progress_step) would read "outstanding > 0, nothing served
for 60 s" as HAENGT. The front reports ``model_state``; anything but
"awake" resets the latch. A boot that never writes the field (every
single-model boot) behaves exactly as before.
"""
from sglang.srt.weg2 import state_file as SF


def _st(model_state=None, served=5, out=2):
    fr = {"outstanding": out, "queue": 0, "awake": "D", "state": "serving",
          "outstanding_by_group": {"D": out, "P": 0}, "served": {"D": served, "P": 0},
          "served_tokens": {"D": {"prompt": 100, "completion": 50}, "P": {"prompt": 100, "completion": 0}}}
    if model_state is not None:
        fr["model_state"] = model_state
    return {"boot_id": "b1", "lifecycle": {"state": "serving"}, "front": fr}


def run(states, stall_s=60.0):
    memo, evs = None, []
    for t, st in states:
        memo, ev = SF.progress_step(memo, st, float(t), stall_s)
        evs.append(ev)
    return evs


def test_asleep_model_with_waiters_never_latches():
    evs = run([(t, _st("asleep")) for t in range(0, 300, 10)])
    assert "HAENGT" not in evs


def test_switching_states_do_not_latch_either():
    for ms in ("sleeping", "waking"):
        assert "HAENGT" not in run([(t, _st(ms)) for t in range(0, 300, 10)])


def test_awake_model_still_latches():
    assert "HAENGT" in run([(t, _st("awake")) for t in range(0, 300, 10)])


def test_field_absent_is_the_single_model_behaviour():
    assert "HAENGT" in run([(t, _st(None)) for t in range(0, 300, 10)])


def test_stall_clock_restarts_at_wake():
    # asleep until t=190 (last sample), then awake with nothing moving: HAENGT only 60 s
    # after the last asleep observation, never earlier
    hist = [(t, _st("asleep")) for t in range(0, 200, 10)] + [(t, _st("awake")) for t in range(200, 300, 10)]
    evs = run(hist)
    first = evs.index("HAENGT")
    assert hist[first][0] >= 190 + 60
