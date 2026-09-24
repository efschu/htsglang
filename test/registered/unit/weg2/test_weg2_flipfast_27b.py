# SPDX-License-Identifier: Apache-2.0
"""27B flipfast (desk/27b-up-flipfast-0924): three front-side costs of every
P-routed request, each behind its own switch, default off.

MEASURED (boots weg2xsn429/430/432/433/435, 45 ladder requests, front log;
tool: the per-rid decomposition in the flipfast report). "Client wall minus
P wall" of ~3.95-4.24 s (median) is TWO flips plus front time:

    route -> D->P flip begin       74-133 ms  (median)   -- the controller tick
    D->P flip                    1.56-1.60 s
    D->P done -> P leg 1 start   199-203 ms            -- the controller tick, EVERY request
    P end -> D first token       2.01-2.20 s            -- the user's "Flipzeit"

Inside the D->P flip, between the kv wake answering and `done`, the residue
reading (`ps` + `nvidia-smi`, two blocking subprocesses on the event loop)
costs 43-53 ms -- after P is already awake, before the controller may
dispatch P's leg 1.

* F2 ``SGLANG_WEG2_CTL_KICK_ARRIVAL=1``: a request joining the P queue wakes
  the controller at once instead of at its next 0.2 s tick.
* F3 ``SGLANG_WEG2_CTL_KICK_AFTER_FLIP=1``: a completed flip wakes the
  controller at once -- P's drain starts right after the D->P flip.
* F1 ``SGLANG_WEG2_DC_OFF_PATH=1``: once nothing gates on the residue reading
  (W19 settled at the first D sleep, the group's dormant image sampled), it
  runs in a worker thread after the flip closed.

DANGER DIRECTIONS guarded here:
* default (all three off) must keep today's timing -- the 0.2 s tick and the
  synchronous reading are still there;
* W19 must still be taken SYNCHRONOUSLY at the first D sleep, switch or not;
* the deferred reading must still land (log line + the flip record);
* every enqueue site in the request path must carry the arrival kick, and the
  literal ``self.queue.append`` stays (test_weg2_no_route_1290 pins it);
* no kick may race d_admitter: the after-flip kick follows only a D->P flip,
  and an arrival kick is held while prefilled requests wait in _ready_for_d
  (the D->P decision does not read that deque).

Hermetic: no GPU, no boot, no HTTP (the group RPCs are stubbed).
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import threading
import time
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front as front_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

MIB = 1024 * 1024
ENV_ARRIVAL = "SGLANG_WEG2_CTL_KICK_ARRIVAL"
ENV_AFTER_FLIP = "SGLANG_WEG2_CTL_KICK_AFTER_FLIP"
ENV_DC = "SGLANG_WEG2_DC_OFF_PATH"
ALL_ENVS = (ENV_ARRIVAL, ENV_AFTER_FLIP, ENV_DC)
CARD = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"


class _Env:
    """Set exactly the named switches for one Front construction."""

    def __init__(self, **on):
        self.on = on
        self.saved = {}

    def __enter__(self):
        for k in ALL_ENVS:
            self.saved[k] = os.environ.pop(k, None)
        for k, v in self.on.items():
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k in ALL_ENVS:
            os.environ.pop(k, None)
            if self.saved.get(k) is not None:
                os.environ[k] = self.saved[k]
        return False


def _front(prefill_sid=0, decode_sid=0, dc_reserve=None, **env):
    with _Env(**env):
        f = front_mod.Front(
            prefill="http://p", decode="http://d", awake="D", tag="flipfast",
            store_dir="/tmp", prefill_sid=prefill_sid, decode_sid=decode_sid,
            dc_reserve=dict(dc_reserve or {}), w_s=45.0, weight_chunks=2,
            flip_min_work_tokens=1,
        )
    f.stops = []

    async def rpc(g, path, body, timeout):
        if path == "/flush_cache":
            return 200, "{}"
        await asyncio.sleep(0.001)
        tags = tuple((body or {}).get("tags", ()))
        return 200, json.dumps({"per_tag": {t: [MIB, 1.0] for t in tags},
                                "critical_path": "rank=0 card=GPU-x ms=1"})

    f.rpc = rpc
    f.do_stop = lambda name, detail: f.stops.append((name, detail))
    return f


def _pending(f, rid="ff-1", tokens=8192):
    fut = asyncio.get_running_loop().create_future()
    return front_mod.Pending(rid, "/generate", {}, "x", time.time(), fut,
                             est_prompt=tokens, est_uncached=tokens)


async def _cancel(task):
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# F3 -- the D->P flip is done; how long until P's leg 1 goes out?
# ---------------------------------------------------------------------------


async def _gap_flip_done_to_leg1(f) -> float:
    seen = {}

    async def leg1(p):
        # t_awake is the D->P flip's own `done` stamp: P->D has not run yet.
        seen.setdefault("gap", time.time() - f.t_awake)
        seen.setdefault("awake", f.awake)
        p.leg1_prompt_tokens = 64

    f.leg1 = leg1
    f.queue.append(_pending(f))
    task = asyncio.create_task(f.controller())
    deadline = time.time() + 5.0
    while "gap" not in seen and time.time() < deadline:
        await asyncio.sleep(0.005)
    await _cancel(task)
    assert "gap" in seen, "the controller never dispatched leg 1"
    assert seen["awake"] == "P", seen
    return seen["gap"]


class KickAfterFlip(CustomTestCase):
    def test_default_keeps_the_tick_between_the_flip_and_the_drain(self):
        # Today's cost, pinned so the default path cannot silently change:
        # the controller sleeps its 0.2 s tick after flip("D","P") returns.
        gap = asyncio.run(self._run())
        self.assertGreater(gap, 0.14, f"default gap {gap * 1000:.0f} ms -- the 0.2 s tick is gone")

    def test_switch_on_dispatches_leg1_right_after_the_flip(self):
        gap = asyncio.run(self._run(**{ENV_AFTER_FLIP: "1"}))
        self.assertLess(
            gap, 0.08,
            f"{ENV_AFTER_FLIP}=1: leg 1 went out {gap * 1000:.0f} ms after the D->P flip "
            f"closed -- the controller still waited for its tick")

    @staticmethod
    async def _run(**env):
        f = _front(**env)
        return await _gap_flip_done_to_leg1(f)


# ---------------------------------------------------------------------------
# F2 -- a LONG request arrives while D is awake and at rest; how long until
# the D->P flip begins?
# ---------------------------------------------------------------------------


async def _gap_arrival_to_flip(f) -> float:
    seen = {}
    at_rest = asyncio.Event()
    orig_idle = f._idle_disposition

    def idle(awake, at_rest_flag):
        out = orig_idle(awake, at_rest_flag)
        if at_rest_flag:
            at_rest.set()
        return out

    orig_flip = f.flip

    async def flip(src, dst):
        seen.setdefault("flip", (src, dst, time.time()))
        return await orig_flip(src, dst)

    async def leg1(p):
        p.leg1_prompt_tokens = 64

    f._idle_disposition = idle
    f.flip = flip
    f.leg1 = leg1
    task = asyncio.create_task(f.controller())
    await asyncio.wait_for(at_rest.wait(), 2.0)  # the controller just rested: a tick starts now
    kick = getattr(f, "_kick_controller", None)
    t0 = time.time()
    f.queue.append(_pending(f))
    if kick is not None:
        kick("arrival")
    deadline = time.time() + 3.0
    while "flip" not in seen and time.time() < deadline:
        await asyncio.sleep(0.005)
    await _cancel(task)
    assert "flip" in seen, "the controller never flipped for the queued request"
    assert kick is not None, "Front has no _kick_controller -- the arrival kick does not exist"
    src, dst, t = seen["flip"]
    assert (src, dst) == ("D", "P"), seen
    return t - t0


class KickArrival(CustomTestCase):
    def test_switch_on_flips_at_once_for_a_queued_long_request(self):
        gap = asyncio.run(self._run(**{ENV_ARRIVAL: "1"}))
        self.assertLess(
            gap, 0.08,
            f"{ENV_ARRIVAL}=1: the D->P flip began {gap * 1000:.0f} ms after the "
            f"arrival -- the controller still waited for its tick")

    def test_default_waits_for_the_tick(self):
        gap = asyncio.run(self._run())
        self.assertGreater(gap, 0.14, f"default gap {gap * 1000:.0f} ms -- the tick is gone")

    @staticmethod
    async def _run(**env):
        f = _front(**env)
        return await _gap_arrival_to_flip(f)


async def _gap_last_leg2_to_flip(f) -> float:
    """A LONG request waits while D decodes; D's last leg 2 ends -- how long
    until the D->P flip begins? (leg2's `finally` pops D.outstanding and kicks;
    the pop + kick are replayed here, the wiring is pinned in ArrivalKickWiring.)"""
    seen = {}
    orig_flip = f.flip

    async def flip(src, dst):
        seen.setdefault("flip", (src, dst, time.time()))
        return await orig_flip(src, dst)

    async def leg1(p):
        p.leg1_prompt_tokens = 64

    D = f.groups["D"]
    D.outstanding["decoding"] = time.time()
    f.queue.append(_pending(f))
    f.flip = flip
    f.leg1 = leg1
    task = asyncio.create_task(f.controller())
    # the controller has seen the queued request and held it behind D's decode
    await asyncio.sleep(0.45)
    assert "flip" not in seen, "flipped while D still held a request"
    t0 = time.time()
    D.outstanding.pop("decoding", None)
    if f.queue and not D.outstanding:
        f._kick_controller("arrival")
    deadline = time.time() + 3.0
    while "flip" not in seen and time.time() < deadline:
        await asyncio.sleep(0.005)
    await _cancel(task)
    assert "flip" in seen, "the controller never flipped after D's work ended"
    return seen["flip"][2] - t0


class KickOnLastLeg2(CustomTestCase):
    def test_switch_on_flips_when_ds_last_leg2_ends(self):
        async def body():
            return await _gap_last_leg2_to_flip(_front(**{ENV_ARRIVAL: "1"}))

        gap = asyncio.run(body())
        self.assertLess(gap, 0.08, f"flip began {gap * 1000:.0f} ms after D's last leg 2 -- still ticking")

    def test_leg2_finally_kicks_only_when_d_is_empty_and_p_has_work(self):
        src = inspect.getsource(front_mod.Front.leg2)
        fin = src[src.rfind("finally:"):]
        self.assertIn("g.outstanding.pop(rid, None)", fin)
        self.assertIn("if self.queue and not g.outstanding:", fin)
        self.assertIn('self._kick_controller("arrival")', fin)
        self.assertLess(fin.find("g.outstanding.pop(rid, None)"),
                        fin.find("if self.queue and not g.outstanding:"),
                        "the kick must read D's outstanding set AFTER this rid left it")


class ArrivalKickWiring(CustomTestCase):
    """Every enqueue in the request path carries the kick right after it."""

    SITES = ("handle_generate", "leg2", "_requeue_after_x_refusal")

    def test_each_queue_append_in_the_request_path_is_kicked(self):
        for name in self.SITES:
            src = inspect.getsource(getattr(front_mod.Front, name))
            lines = src.splitlines()
            idx = [i for i, l in enumerate(lines) if re.search(r"self\.queue\.append\(", l)]
            self.assertTrue(idx, f"{name}: no self.queue.append at all")
            for i in idx:
                nxt = "\n".join(lines[i + 1:i + 3])
                self.assertIn(
                    'self._kick_controller("arrival")', nxt,
                    f"{name}: `{lines[i].strip()}` is not followed by the arrival kick")

    def test_the_literal_append_stays_for_the_1290_pin(self):
        src = inspect.getsource(front_mod.Front.handle_generate)
        self.assertLess(src.find('route == "none"'), src.find("self.queue.append"))

    def test_the_intake_stall_requeue_is_not_kicked(self):
        # appendleft during a P drain: the drain itself ends and flips, a kick
        # would only re-run the P arm it is already in.
        src = inspect.getsource(front_mod.Front._requeue_intake_stalled)
        self.assertIn("self.queue.appendleft(p)", src)
        self.assertNotIn("_kick_controller", src)


class KickHelper(CustomTestCase):
    def test_kicks_are_inert_when_their_switch_is_off(self):
        async def body():
            f = _front()
            f._kick_controller("arrival")
            f._kick_controller("after_flip")
            t0 = time.time()
            await f._ctl_wait()
            return time.time() - t0

        self.assertGreater(asyncio.run(body()), 0.15)

    def test_an_unknown_reason_is_refused_by_name(self):
        async def body():
            f = _front(**{ENV_ARRIVAL: "1", ENV_AFTER_FLIP: "1"})
            with self.assertRaises(ValueError):
                f._kick_controller("typo")

        asyncio.run(body())

    def test_a_kick_ends_one_wait_and_is_consumed(self):
        async def body():
            f = _front(**{ENV_ARRIVAL: "1"})
            f._kick_controller("arrival")
            t0 = time.time()
            await f._ctl_wait()
            first = time.time() - t0
            t1 = time.time()
            await f._ctl_wait()
            second = time.time() - t1
            return first, second, f.counters.get("ctl_kicked", 0)

        first, second, kicked = asyncio.run(body())
        self.assertLess(first, 0.03)
        self.assertGreater(second, 0.15, "a consumed kick must not end the next wait too")
        self.assertEqual(kicked, 1)

    def test_the_after_flip_kick_follows_only_a_d_to_p_flip(self):
        # After P->D nothing waits on the controller (d_admitter hands leg 2
        # over), and an immediate D-branch pass could race the admitter.
        async def body():
            f = _front(**{ENV_AFTER_FLIP: "1"})
            await f.flip("D", "P")
            after_dp = f.counters.get("ctl_kick_after_flip", 0)
            await f._ctl_wait()  # consume it
            await f.flip("P", "D")
            after_pd = f.counters.get("ctl_kick_after_flip", 0)
            t0 = time.time()
            await f._ctl_wait()
            return after_dp, after_pd, time.time() - t0, f.stops

        after_dp, after_pd, wait, stops = asyncio.run(body())
        self.assertEqual(stops, [])
        self.assertEqual((after_dp, after_pd), (1, 1))
        self.assertGreater(wait, 0.15, "a P->D flip must not end the next tick early")

    def test_an_arrival_kick_is_held_while_prefilled_requests_wait_for_d(self):
        # The D->P decision reads D.outstanding and the hand-off window, not
        # _ready_for_d: an early pass could flip D away before d_admitter's
        # next poll admits them. Held -> the tick keeps today's ordering.
        async def body():
            f = _front(**{ENV_ARRIVAL: "1"})
            f._ready_for_d.append(_pending(f, rid="prefilled"))
            f._kick_controller("arrival")
            t0 = time.time()
            await f._ctl_wait()
            return time.time() - t0, f.counters.get("ctl_kick_held_ready_for_d", 0)

        wait, held = asyncio.run(body())
        self.assertEqual(held, 1)
        self.assertGreater(wait, 0.15)

    def test_a_front_built_without_init_is_on_the_default_path(self):
        # Several suites assemble a partial Front via __new__ and drive
        # handle_generate / flip on it (test_weg2_no_route_1290 among them):
        # no switches there means no kick and no deferral, never an AttributeError.
        f = front_mod.Front.__new__(front_mod.Front)
        f._kick_controller("arrival")
        f._kick_controller("after_flip")
        self.assertFalse(f._dc_reading_deferrable("D"))
        self.assertFalse(f._dc_reading_deferrable("P"))

    def test_the_switch_state_is_announced(self):
        f = _front(**{ENV_DC: "1"})
        line = f.flipfast_line()
        self.assertIn("kick_arrival=off", line)
        self.assertIn("kick_after_flip=off", line)
        self.assertIn("dc_off_path=on", line)


# ---------------------------------------------------------------------------
# F1 -- the post-wake residue reading
# ---------------------------------------------------------------------------


class _Fakes:
    """Slow stand-ins for `ps` and `nvidia-smi`, recording their thread."""

    DELAY = 0.08

    def __init__(self, reading):
        self.reading = dict(reading)
        self.calls = []

    def session_pids(self, sid):
        self.calls.append(("pids", threading.get_ident(), time.time()))
        time.sleep(self.DELAY)
        return {101, 102}

    def nvml_process_mib(self, pids):
        self.calls.append(("nvml", threading.get_ident(), time.time()))
        time.sleep(self.DELAY)
        return dict(self.reading)


class DcOffPath(CustomTestCase):
    def setUp(self):
        self._saved = (front_mod._session_pids, front_mod._nvml_process_mib)

    def tearDown(self):
        front_mod._session_pids, front_mod._nvml_process_mib = self._saved

    def _install(self, reading):
        fk = _Fakes(reading)
        front_mod._session_pids = fk.session_pids
        front_mod._nvml_process_mib = fk.nvml_process_mib
        return fk

    async def _four_flips(self, f):
        walls = []
        for src, dst in (("D", "P"), ("P", "D"), ("D", "P"), ("P", "D")):
            t0 = time.time()
            await f.flip(src, dst)
            walls.append(time.time() - t0)
        await asyncio.sleep(4 * _Fakes.DELAY + 0.3)  # let any off-path reading land
        return walls

    def test_default_measures_on_the_flip_every_time(self):
        fk = self._install({CARD: 1700})

        async def body():
            f = _front(prefill_sid=11, decode_sid=22, dc_reserve={CARD: 2000})
            loop_tid = threading.get_ident()
            walls = await self._four_flips(f)
            return f, walls, loop_tid

        f, walls, loop_tid = asyncio.run(body())
        self.assertEqual(f.stops, [])
        for w in walls[2:]:
            self.assertGreater(w, 2 * _Fakes.DELAY - 0.01, walls)
        self.assertTrue(all(tid == loop_tid for _, tid, _ in fk.calls), "default must stay synchronous")
        self.assertEqual(f.flip_log[-1]["dc_mib"], {CARD: 1700})

    def test_switch_on_moves_the_settled_reading_off_the_flip(self):
        fk = self._install({CARD: 1700})

        async def body():
            f = _front(prefill_sid=11, decode_sid=22, dc_reserve={CARD: 2000}, **{ENV_DC: "1"})
            loop_tid = threading.get_ident()
            with self.assertLogs(front_mod.logger, level=logging.INFO) as cm:
                walls = await self._four_flips(f)
            return f, walls, loop_tid, cm.output

        f, walls, loop_tid, out = asyncio.run(body())
        self.assertEqual(f.stops, [])
        # flips 1 and 2 are each group's FIRST sleep: W19 and the dormant image
        # still take the reading synchronously.
        self.assertGreater(walls[0], 2 * _Fakes.DELAY - 0.01, walls)
        self.assertGreater(walls[1], 2 * _Fakes.DELAY - 0.01, walls)
        # flips 3 and 4 are settled: the flip no longer pays the reading.
        for w in walls[2:]:
            self.assertLess(w, _Fakes.DELAY, f"the settled flip still paid the reading: {walls}")
        late = [c for c in fk.calls if c[1] != loop_tid]
        self.assertGreaterEqual(len(late), 4, f"the off-path reading did not run in a worker: {fk.calls}")
        # ...and it still LANDS: the record and the same WEG2-DC line.
        for rec in f.flip_log[2:]:
            self.assertEqual(rec["dc_mib"], {CARD: 1700}, rec)
            self.assertTrue(rec.get("dc_off_path"), rec)
        dc_lines = [l for l in out if "WEG2-DC group=" in l and f"uuid={CARD} measured=1700 MiB" in l]
        self.assertGreaterEqual(len(dc_lines), 4, out)
        self.assertTrue(any("WEG2-DC-OFFPATH" in l for l in out), out)

    def test_w19_still_stops_synchronously_at_the_first_d_sleep(self):
        self._install({CARD: 2600})

        async def body():
            f = _front(prefill_sid=11, decode_sid=22, dc_reserve={CARD: 2000}, **{ENV_DC: "1"})
            await f.flip("D", "P")
            return f

        f = asyncio.run(body())
        self.assertEqual([n for n, _ in f.stops], ["W19 DormantResidueRefused"], f.stops)
        self.assertEqual(f.awake, "D", "W19 must stop the flip before it closes")

    def test_an_empty_first_reading_keeps_the_d_side_synchronous(self):
        # `dc_measured_d` stays unset when the first reading is empty; the
        # W19 gate is then NOT settled and every D sleep keeps measuring
        # synchronously -- the conservative side.
        fk = self._install({})

        async def body():
            f = _front(prefill_sid=11, decode_sid=22, dc_reserve={CARD: 2000}, **{ENV_DC: "1"})
            loop_tid = threading.get_ident()
            walls = await self._four_flips(f)
            return f, walls, loop_tid

        f, walls, loop_tid = asyncio.run(body())
        self.assertGreater(walls[2], 2 * _Fakes.DELAY - 0.01, walls)  # D sleep: synchronous
        self.assertLess(walls[3], _Fakes.DELAY, walls)                # P sleep: off-path


if __name__ == "__main__":
    unittest.main()
