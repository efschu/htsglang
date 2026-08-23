# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#824 W4: an aborted flip must not wedge the PP ring.

THE MEASURED WEDGE (boot_827_review_0823_0910c)
-----------------------------------------------
A flip abandoned, and the ring went silent for 31 s until health failed.
py-spy caught all three ranks (evidence
/spinning/evidence-665-f1/pyspy_wedge_18g_0918_*.txt):

  * PP0 blocked in a SEND: ``_pp_commit_admission_send_work``
    (scheduler_pp_mixin.py:4583) -> ``_pp_commit_comm_work`` (:4071), i.e.
    the admission-decision channel.
  * PP1 and PP2 both blocked in ``_advance`` (pp_chain_receiver.py:218) on
    ``self._size_work.wait()`` -- an UNBOUNDED irecv wait on the REQUEST
    RELAY channel, a different channel from the one PP0 was sending on.

Two channels, and the ring was split across them: PP0 waiting on A, PP1/PP2
parked for ever on B. Nobody was wrong locally; the ring was deadlocked.

WHY BOUNDING THIS IS NOT THE OBVIOUS ONE-LINER
----------------------------------------------
Two transport facts on this build (both re-measured 2026-08-23, and both
pinned as tests in this file's sibling ``test_pp_chain_receiver.py``):

  1. ``Work.is_completed()`` NEVER reports True, even seconds after the
     payload has demonstrably landed. So a poll-loop cannot bound a wait.
     (The module docstring calls this "corpse F".)
  2. ``Work.wait(timeout=...)`` DOES fire on time -- and DESTROYS the gloo
     pair while doing it: "Application timeout caused pair closure", on
     BOTH sides. Every later collective on that group then fails. So a
     timed wait cannot be a retry; it is terminal, and using it as the
     bound would convert a recoverable stall into a torn ring.

That leaves exactly one shape, and it is what the fix implements: the
blocking ``wait()`` is parked on a dedicated waiter thread that this
receiver owns, and the caller joins it with a deadline. The wait itself is
never interrupted, so the pair is never poisoned and the posted irecv stays
framed; the CALLER simply regains control and may run a named recovery
branch. If the peer sends late, the SAME parked wait still lands the
message intact -- which is what the last two assertions here check.

WHAT THIS TEST DOES
-------------------
Three real gloo processes in the measured geometry:

  pass 1   a chain message flows 0 -> 1 -> 2 (proves the harness is
           wire-correct before anything is stressed);
  abort    rank 0 stops feeding the chain and instead does a blocking
           admission round-trip on channel A with rank 1, exactly as PP0
           did; ranks 1 and 2 are inside ``recv()`` on channel B.
  recovery ranks 1 and 2 must LEAVE that wait by a named branch. Rank 1
           then services channel A, which releases rank 0. Rank 0 finally
           sends the chain message it owed, and both downstream ranks must
           receive it CORRECTLY on the very wait they had bounded.

RED BEFORE THE FIX: ``recv()`` is unbounded, no rank ever leaves it, all
three processes are killed at the harness deadline and the test fails.
GREEN AFTER: every rank exits 0, both recovery branches are named, and the
late payload arrives unmangled.
"""

import multiprocessing as mp
import os
import socket
import time

import pytest

# The stall bound is configured by environment, not by a new call
# signature, so that this exact test body is runnable against the
# pre-fix tree (where it hangs) and the post-fix tree (where it passes).
STALL_ENV = "SGLANG_PP_CHAIN_RECV_STALL_S"
STALL_S = "3.0"

ADMISSION_TAG = 77
WARMUP_PAYLOAD = {"rid": "warmup-0", "prefix_lens": 0}
LATE_PAYLOAD = {"rid": "350bf3b4-late", "prefix_lens": 0}

HARNESS_TIMEOUT_S = 75.0


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _init(rank, port):
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=3,
    )


def _stalled_exc_type():
    """The named recovery signal, imported defensively.

    Before the fix the class does not exist, and the placeholder returned
    here is never raised -- which is precisely why the pre-fix run hangs
    instead of erroring. That is the RED this test is built to show.
    """
    try:
        from sglang.srt.managers.pp_chain_receiver import PpChainRecvStalled

        return PpChainRecvStalled
    except ImportError:

        class _AbsentPreFix(RuntimeError):
            pass

        return _AbsentPreFix


def _recv_with_recovery(rx, stalled_type, on_stall=None, budget_s=30.0):
    """Call recv(), running ``on_stall`` when the named branch fires.

    ``on_stall`` is what a real scheduler would do with the control it has
    just been handed back: service whatever the peer is actually blocked
    on. It runs at most once.

    Returns (payload, recovered: bool).
    """
    deadline = time.perf_counter() + budget_s
    recovered = False
    while True:
        try:
            return rx.recv(), recovered
        except stalled_type:
            if not recovered and on_stall is not None:
                on_stall()
            recovered = True
            if time.perf_counter() > deadline:
                raise


def _pp0(port, q):
    """The rank that abandons the flip and then blocks in an admission send."""
    try:
        import torch
        import torch.distributed as dist

        from sglang.srt.utils import point_to_point_pyobj

        _init(0, port)
        out = {}

        # pass 1: a healthy forward down the chain.
        point_to_point_pyobj([WARMUP_PAYLOAD], 0, None, 0, 1)

        # The flip abandons here. PP0 does NOT feed the chain; it commits
        # an admission decision on channel A and waits for the peer, which
        # is the send py-spy caught it in.
        started = time.perf_counter()
        dist.send(torch.tensor([1], dtype=torch.long), dst=1, tag=ADMISSION_TAG)
        ack = torch.zeros(1, dtype=torch.long)
        dist.recv(ack, src=1, tag=ADMISSION_TAG)
        out["admission_roundtrip_s"] = round(time.perf_counter() - started, 2)
        out["admission_ack"] = int(ack.item())

        # Released. Now pay the chain message that was owed all along --
        # it must still land intact on the downstream's bounded wait.
        point_to_point_pyobj([LATE_PAYLOAD], 0, None, 0, 1)

        dist.barrier()
        dist.destroy_process_group()
        q.put(("pp0", out))
    except Exception as exc:  # noqa: BLE001
        q.put(("pp0", {"error": repr(exc)[:400]}))


def _pp1(port, q):
    """Midstream: parked on the chain while PP0 waits on the admission channel."""
    try:
        import torch
        import torch.distributed as dist

        from sglang.srt.managers.pp_chain_receiver import PpChainReceiver
        from sglang.srt.utils import point_to_point_pyobj

        _init(1, port)
        out = {}
        stalled_type = _stalled_exc_type()
        rx = PpChainReceiver(group=None, src=0, dst=1)

        warmup = rx.recv()
        out["warmup_ok"] = warmup == [WARMUP_PAYLOAD]
        point_to_point_pyobj(list(warmup), 1, None, 1, 2)

        # THE WEDGE. Unbounded before the fix: PP0 will never feed this
        # channel until PP1 services channel A, and PP1 cannot service
        # channel A while it is in here.
        started = time.perf_counter()
        try:
            payload, recovered = _recv_with_recovery(
                rx, stalled_type, on_stall=_pp1_service_admission
            )
        finally:
            out["blocked_s"] = round(time.perf_counter() - started, 2)
        out["recovered"] = recovered
        out["late_payload_ok"] = payload == [LATE_PAYLOAD]
        point_to_point_pyobj(list(payload), 1, None, 1, 2)

        dist.barrier()
        dist.destroy_process_group()
        q.put(("pp1", out))
    except Exception as exc:  # noqa: BLE001
        q.put(("pp1", {"error": repr(exc)[:400]}))


def _pp1_service_admission():
    """Channel A service, run from PP1's named recovery branch."""
    import torch
    import torch.distributed as dist

    decision = torch.zeros(1, dtype=torch.long)
    dist.recv(decision, src=0, tag=ADMISSION_TAG)
    dist.send(torch.tensor([int(decision.item())], dtype=torch.long), dst=0, tag=ADMISSION_TAG)


def _pp2(port, q):
    """Tail of the ring: parked on the chain behind PP1."""
    try:
        import torch.distributed as dist

        from sglang.srt.managers.pp_chain_receiver import PpChainReceiver

        _init(2, port)
        out = {}
        stalled_type = _stalled_exc_type()
        rx = PpChainReceiver(group=None, src=1, dst=2)

        warmup = rx.recv()
        out["warmup_ok"] = warmup == [WARMUP_PAYLOAD]

        started = time.perf_counter()
        payload, recovered = _recv_with_recovery(rx, stalled_type)
        out["blocked_s"] = round(time.perf_counter() - started, 2)
        out["recovered"] = recovered
        out["late_payload_ok"] = payload == [LATE_PAYLOAD]

        dist.barrier()
        dist.destroy_process_group()
        q.put(("pp2", out))
    except Exception as exc:  # noqa: BLE001
        q.put(("pp2", {"error": repr(exc)[:400]}))


def _run_ring():
    env_backup = os.environ.get(STALL_ENV)
    os.environ[STALL_ENV] = STALL_S
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = {
        "pp0": ctx.Process(target=_pp0, args=(port, q)),
        "pp1": ctx.Process(target=_pp1, args=(port, q)),
        "pp2": ctx.Process(target=_pp2, args=(port, q)),
    }
    for p in procs.values():
        p.start()

    results = {}
    deadline = time.perf_counter() + HARNESS_TIMEOUT_S
    while time.perf_counter() < deadline and len(results) < 3:
        try:
            name, value = q.get(timeout=1.0)
            results[name] = value
        except Exception:  # noqa: BLE001 - empty queue
            if not any(p.is_alive() for p in procs.values()):
                break

    # Give every rank a grace period to finish its teardown. A rank posts
    # its result BEFORE it has exited, so "still alive right now" is not
    # wedged -- it is usually just on its way out.
    grace_until = time.perf_counter() + 20.0
    for p in procs.values():
        p.join(timeout=max(0.1, grace_until - time.perf_counter()))

    # Decide who is wedged BEFORE killing anybody. Killing one rank tears
    # that rank's gloo pairs down, which makes every peer raise a moment
    # later -- so a kill-then-look order would report the cascade instead
    # of the wedge, and would blame one rank for all three.
    wedged = [name for name, p in procs.items() if p.is_alive()]
    for p in procs.values():
        if p.is_alive():
            p.kill()
    # Drain anything the cascade produced, so a rank that CRASHED is not
    # silently indistinguishable from a rank that hung.
    drain_until = time.perf_counter() + 5.0
    while time.perf_counter() < drain_until and len(results) < 3:
        try:
            name, value = q.get(timeout=0.5)
            results.setdefault(name, value)
        except Exception:  # noqa: BLE001 - empty queue
            pass
    exitcodes = {}
    for name, p in procs.items():
        p.join(timeout=5.0)
        exitcodes[name] = p.exitcode
    results["_exitcodes"] = exitcodes

    if env_backup is None:
        os.environ.pop(STALL_ENV, None)
    else:
        os.environ[STALL_ENV] = env_backup
    return results, wedged


@pytest.mark.timeout(180)
def test_aborted_flip_does_not_wedge_the_pp_ring():
    results, wedged = _run_ring()

    assert not wedged, (
        f"PP ring wedged: {wedged} never left their blocking wait within "
        f"{HARNESS_TIMEOUT_S}s and had to be killed. This is the measured "
        f"boot_827 deadlock -- PP0 parked in the admission send while the "
        f"downstream ranks sat in an unbounded chain recv. Collected so "
        f"far: {results}"
    )
    for name in ("pp0", "pp1", "pp2"):
        assert name in results, f"{name} produced no result; got {results}"
        assert "error" not in results[name], f"{name} failed: {results[name]}"

    # Both downstream ranks must have LEFT the wait by the named branch,
    # not merely have got lucky on timing.
    assert results["pp1"]["recovered"] is True, (
        "PP1 never entered the named recovery branch; it cannot have been "
        f"what released the admission send. {results['pp1']}"
    )
    assert results["pp2"]["recovered"] is True, (
        f"PP2 never entered the named recovery branch. {results['pp2']}"
    )

    # The admission send was actually released by that recovery.
    assert results["pp0"]["admission_ack"] == 1, results["pp0"]

    # ...and the bounded wait did not poison the stream: the late chain
    # message still arrives, on both ranks, unmangled. This is the property
    # a Work.wait(timeout) based bound provably cannot deliver on this
    # build (it closes the pair).
    assert results["pp1"]["warmup_ok"] and results["pp1"]["late_payload_ok"], results["pp1"]
    assert results["pp2"]["warmup_ok"] and results["pp2"]["late_payload_ok"], results["pp2"]
