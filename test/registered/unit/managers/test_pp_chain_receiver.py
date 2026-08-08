"""#631: the PP request-chain receiver, and the boot-18 falsifier.

BOOT 18, the specimen this file exists for. OBSERVED: rank 0 inside the
phase flip's consensus reduction, rank 1 blocked in ``work.wait()``
(scheduler_pp_mixin :1109) from the ORDINARY top-of-pass commit (:705) of
the previous pass's chain forward. Rank 2's stack was never recorded --
see the corpse table in phase_flip_presence for what is evidence and what
is inference.

The mechanism the fix assumed: that commit PRECEDES the flip's entry gate,
so rank 1's forward was not completing because its downstream had stopped
consuming; and rank 0 was inside the reduction, so it had seen a full
quorum -- meaning rank 1's flag was up while rank 1 still owed that send.

READ THE LAST TEST IN THIS FILE BEFORE TRUSTING THAT MECHANISM. It is
measured, and it contradicts half of it. On this gloo build the upstream's
commit returns in 0.00 s with the forward UNCONSUMED, and a downstream
that polls is_completed() on a posted irecv never completes it at all. So:

  * a single unconsumed forward does not block the upstream, which means
    "the downstream stopped consuming" does not by itself explain the
    boot-18 stall;
  * a non-blocking drain built on is_completed() polling absorbs nothing
    on this transport, so the decided clause (ii) is not constructible in
    that shape -- and neither is the pre-existing send-side pump, which
    reaps on the same predicate.

What IS pinned here and stands on its own:
  * the receiver as a state machine -- point_to_point_pyobj is a TWO-STEP
    protocol (size, then payload) that cannot be consumed by halves
    without misframing every later message. That is a property of the
    wire format, independent of how progress is driven;
  * the transport measurement itself, so the next design starts from it.

CPU-only.
"""

import multiprocessing as mp
import socket
import time

import pytest
import torch

# ---------------------------------------------------------------------------
# Part 1: the state machine, against a fake transport.
# ---------------------------------------------------------------------------


class _FakeWork:
    """A dist Work whose completion the test controls."""

    def __init__(self, tensor, source, ready_after=0):
        self._tensor = tensor
        self._source = source
        self._ready_after = ready_after
        self._polls = 0
        self.waited = False

    def is_completed(self):
        self._polls += 1
        if self._polls > self._ready_after:
            self._deliver()
            return True
        return False

    def wait(self):
        self.waited = True
        self._deliver()

    def _deliver(self):
        if self._source is not None:
            self._tensor.copy_(self._source)
            self._source = None


class _FakeDist:
    """Stands in for torch.distributed on one chain stream.

    Holds a queue of already-serialized messages and hands them out in the
    same two-step shape the real transport uses. A frame is taken off the
    queue when its SIZE recv is posted, which mirrors the real stream:
    from that point the message is committed to this receiver.

    ``size_delay``/``data_delay`` are the number of ``is_completed()``
    polls each step reports incomplete for, so a test can hold a message
    half-received on purpose.
    """

    def __init__(self, messages, size_delay=0, data_delay=0):
        import pickle

        self._queue = [None if m == [] else pickle.dumps(m) for m in messages]
        self._current = None
        self._size_delay = size_delay
        self._data_delay = data_delay
        self.posted = []
        self.works = []

    def _track(self, work):
        self.works.append(work)
        return work

    def irecv(self, tensor, src=None, group=None):
        if tensor.dtype == torch.long:
            if not self._queue:
                # A quiet upstream: an irecv that simply never completes.
                self.posted.append("size(idle)")
                return self._track(_FakeWork(tensor, None, ready_after=10**9))
            self._current = self._queue.pop(0)
            size = 0 if self._current is None else len(self._current)
            self.posted.append(f"size({size})")
            return self._track(
                _FakeWork(
                    tensor,
                    torch.tensor([size], dtype=torch.long),
                    self._size_delay,
                )
            )
        frame = self._current
        self._current = None
        self.posted.append(f"data({len(frame)})")
        return self._track(
            _FakeWork(
                tensor,
                torch.frombuffer(bytearray(frame), dtype=torch.uint8),
                self._data_delay,
            )
        )


def _receiver(monkeypatch, messages, size_delay=0, data_delay=0):
    from sglang.srt.managers import pp_chain_receiver as mod

    fake = _FakeDist(messages, size_delay=size_delay, data_delay=data_delay)
    monkeypatch.setattr(mod, "dist", fake)
    return mod.PpChainReceiver(group=None, src=0, dst=1), fake


def test_poll_absorbs_whole_messages_and_never_blocks(monkeypatch):
    rx, fake = _receiver(monkeypatch, [["a"], ["b", "c"]])
    assert rx.poll() == 2
    assert list(rx.inbox) == [["a"], ["b", "c"]]
    assert rx.mid_message is False
    # A poll that blocks is the whole bug this class exists to avoid.
    assert not any(w.waited for w in fake.works), "poll() waited on a work"


def test_can_fail_a_half_received_message_is_resumed_not_restarted(monkeypatch):
    """THE FRAMING RULE. Once the size has been received the payload is
    already on the wire and MUST be received. A poll that gave up and
    re-posted a fresh size recv would read the payload AS a size and
    misframe every later message on the stream."""
    rx, fake = _receiver(monkeypatch, [["a"]], data_delay=1)

    # First poll: the size lands, the payload does not.
    assert rx.poll() == 0
    assert rx.mid_message is True, (
        "the receiver forgot it was mid-message; the payload already on "
        "the wire would be read as the next message's SIZE"
    )
    # The second poll RESUMES the same message rather than restarting it.
    assert rx.poll() == 1
    assert list(rx.inbox) == [["a"]]

    # THE FRAMING INVARIANT: exactly one size recv and one data recv were
    # posted for this one message. A restart would show a second size recv
    # for it -- which on the real stream would consume the payload as a
    # length and misframe everything after it. (The trailing "size(idle)"
    # is the receiver correctly reaching for the NEXT message once this one
    # was complete, which is why the count excludes it.)
    real_sizes = [p for p in fake.posted if p.startswith("size(") and "idle" not in p]
    datas = [p for p in fake.posted if p.startswith("data(")]
    assert len(real_sizes) == 1, f"message was restarted: {fake.posted}"
    assert len(datas) == 1, f"payload was re-posted: {fake.posted}"
    assert not any(w.waited for w in fake.works)


def test_empty_forward_is_a_message_not_a_silence(monkeypatch):
    """An empty forward still carries the pass. The scheduler counts on
    one receive per upstream send, so swallowing it would desynchronise
    the chain by one message."""
    rx, _ = _receiver(monkeypatch, [[], ["a"]])
    assert rx.poll() == 2
    assert list(rx.inbox) == [[], ["a"]]


def test_recv_hands_over_the_inbox_before_taking_anything_new(monkeypatch):
    """INBOX FIRST, IN ORDER. Messages absorbed while a flip was armed are
    the OLDEST on the stream; handing out a newer one first would reorder
    the request chain."""
    rx, _ = _receiver(monkeypatch, [["armed-era"], ["after"]])
    rx.poll()
    assert rx.recv() == ["armed-era"]
    assert rx.recv() == ["after"]


def test_recv_blocks_through_a_message_the_poll_left_half_received(monkeypatch):
    rx, _ = _receiver(monkeypatch, [["a"]], data_delay=1)
    rx.poll()
    assert rx.mid_message is True
    assert rx.recv() == ["a"]
    assert rx.mid_message is False


def test_poll_is_bounded_so_a_fast_upstream_cannot_hold_an_armed_rank(
    monkeypatch,
):
    """An armed rank has a gate to get to. A drain that ran until the
    stream went quiet would let a busy upstream keep it here."""
    rx, _ = _receiver(monkeypatch, [["m"]] * 50)
    assert rx.poll(max_messages=4) == 4
    assert len(rx.inbox) == 4


# ---------------------------------------------------------------------------
# Part 2: the falsifier, on a real gloo pair.
# ---------------------------------------------------------------------------
#
# This is the part that makes the rest mean something. It reproduces the
# boot-18 geometry with two processes and measures the upstream's commit:
# the very call rank 1 was found blocked in.


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
        world_size=2,
    )


def _upstream(port, payload_kb, q):
    """The rank that owes a forward, and then tries to flush it.

    This models scheduler_pp_mixin :705 exactly: an async forward issued
    on a previous pass, committed at the top of this one.
    """
    try:
        import torch.distributed as dist

        from sglang.srt.utils import point_to_point_pyobj

        _init(0, port)
        payload = ["q" * (payload_kb * 1024)]
        works = point_to_point_pyobj(payload, 0, None, 0, 1, async_send=True)
        started = time.perf_counter()
        for w in works:
            w.work.wait()
        q.put(("flushed_after_s", time.perf_counter() - started))
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        q.put(("error", repr(exc)))


def _downstream(port, drain, hold_s, q):
    """The ARMED rank. It performs no blocking receive either way -- the
    only difference is whether it keeps CONSUMING while it waits."""
    try:
        import torch.distributed as dist

        from sglang.srt.managers.pp_chain_receiver import PpChainReceiver

        _init(1, port)
        rx = PpChainReceiver(group=None, src=0, dst=1)
        deadline = time.perf_counter() + hold_s
        while time.perf_counter() < deadline:
            if drain:
                rx.poll()
            time.sleep(0.01)
        q.put(("absorbed", len(rx.inbox)))
        if not drain:
            # Drain now, so the peer can finish and the group can be torn
            # down without leaving a message stranded on the wire.
            while not rx.inbox:
                rx.poll()
                time.sleep(0.01)
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        q.put(("error", repr(exc)))


def _run_pair(drain, hold_s=3.0, payload_kb=512, timeout=40.0):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    up = ctx.Process(target=_upstream, args=(port, payload_kb, q))
    down = ctx.Process(target=_downstream, args=(port, drain, hold_s, q))
    up.start()
    down.start()
    results = {}
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline and len(results) < 2:
        try:
            key, value = q.get(timeout=1.0)
        except Exception:  # noqa: BLE001 - empty queue
            if not up.is_alive() and not down.is_alive():
                break
            continue
        results[key] = value
    for p in (up, down):
        p.join(timeout=5.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)
    return results


def test_measured_gloo_does_not_progress_a_posted_irecv_by_polling():
    """THE TRANSPORT FACT, measured 2026-08-08, and the reason the
    poll-based drain above CANNOT deliver on this build.

    Geometry is exactly boot 18's: the upstream commits its forward
    (work.wait(), scheduler_pp_mixin :705 -> :1109) while the downstream
    has a posted irecv and only ever calls is_completed(), never wait() --
    an armed rank parked at the flip's entry gate.

    MEASURED, both at 8 B and at 512 KiB:
      * the downstream's is_completed() NEVER returns True (4 s of
        polling at 10 ms) -- so a drain built on it absorbs nothing, ever;
      * the upstream's commit returns in 0.00 s ANYWAY, without the
        downstream having consumed anything.

    Both halves matter and both are bad news for the decided fix. The
    first says clause (ii) is not constructible by polling. The second
    says a single unconsumed forward does not block the upstream at all,
    so the boot-18 stall is NOT explained by "the downstream stopped
    consuming one message".

    This is the same premise the module docstring of phase_flip_presence
    REJECTED for a posted-and-polled all_reduce -- "an unverified
    transport assumption of exactly the kind that has already killed
    designs here". It is false for point-to-point too. Pinned as a test
    so the next design starts from the measurement instead of re-deriving
    it, and so a torch/gloo upgrade that CHANGES it is noticed here rather
    than in a wedge.
    """
    result = _run_poll_vs_commit(payload_bytes=512 * 1024, poll_s=4.0)
    assert "error" not in result, result
    assert result["upstream_commit_s"] < 0.5, (
        "the upstream's commit blocked on an unconsumed forward. If this "
        "fires, gloo backpressure HAS changed and the boot-18 mechanism "
        "is reproducible after all -- re-open the clause (ii) design"
    )
    assert result["poll_completed"] is False, (
        "a posted irecv completed by POLLING is_completed(). If this "
        "fires, this build progresses works without wait() and the "
        "non-blocking drain becomes constructible as originally decided"
    )


def _poll_side(port, nbytes, poll_s, q):
    try:
        import torch.distributed as dist

        _init(1, port)
        buf = torch.zeros(nbytes, dtype=torch.uint8)
        work = dist.irecv(buf, src=0)
        deadline = time.perf_counter() + poll_s
        completed = False
        while time.perf_counter() < deadline:
            if work.is_completed():
                completed = True
                break
            time.sleep(0.01)
        q.put(("poll_completed", completed))
        # Always finish the message, so the pair tears down cleanly.
        if not completed:
            work.wait()
        time.sleep(0.5)
    except Exception as exc:  # noqa: BLE001
        q.put(("error", repr(exc)))


def _commit_side(port, nbytes, q):
    try:
        import torch.distributed as dist

        _init(0, port)
        buf = torch.ones(nbytes, dtype=torch.uint8)
        work = dist.isend(buf, 1)
        started = time.perf_counter()
        work.wait()
        q.put(("upstream_commit_s", time.perf_counter() - started))
        time.sleep(2.0)
    except Exception as exc:  # noqa: BLE001
        q.put(("error", repr(exc)))


def _run_poll_vs_commit(payload_bytes, poll_s, timeout=60.0):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    up = ctx.Process(target=_commit_side, args=(port, payload_bytes, q))
    down = ctx.Process(target=_poll_side, args=(port, payload_bytes, poll_s, q))
    up.start()
    down.start()
    out = {}
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline and len(out) < 2:
        try:
            key, value = q.get(timeout=1.0)
        except Exception:  # noqa: BLE001
            if not up.is_alive() and not down.is_alive():
                break
            continue
        out[key] = value
    for p in (up, down):
        p.join(timeout=5.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)
    return out
