"""The handoff's acceptance list, executed against the STUB backend set.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        python -m pytest test/registered/copilot/test_acceptance_stub.py -v

One class per acceptance item, named after it. Everything runs over the real
ASGI stack (FastAPI routing, real WebSocket lifecycle) against
``backend="stub"`` -- the same code path a browser drives, with the stub's
timings turned down so the suite stays fast.

What this file CANNOT prove, stated here so nobody reads it as more than it is:
item 1 is about two capture chains in a real browser, and no server-side test
can produce a ``getDisplayMedia`` stream. What is proven here is the half that
lives on this side of the socket -- that two independently tagged audio sources
are attributed to two independent transcription streams and never mixed. The
browser half is a separate, executed run recorded in the report.
"""

import json
import time

import pytest
from fastapi.testclient import TestClient

from sglang.srt.copilot.backends import build_backend_set
from sglang.srt.copilot.briefing import parse_briefing
from sglang.srt.copilot.config import CopilotConfig
from sglang.srt.copilot.protocol import ServerFrame, Track, encode_audio_frame
from sglang.srt.copilot.server import CopilotService, build_app
from sglang.srt.copilot.stubs import stub_briefing_text

RECEIVE_TIMEOUT_S = 10.0


def fast_config(**overrides) -> CopilotConfig:
    """The stub set with its clocks turned down.

    Every value here is a TIMING, never a behaviour: the script, the partials,
    the eviction capacity and the cold penalty are all the shipped ones. A test
    that also changed the behaviour would be testing a fourth backend.
    """
    base = dict(
        backend="stub",
        stub_word_ms=6.0,
        stub_final_ms=10.0,
        stub_gap_ms=20.0,
        stub_hint_latency_ms=10.0,
        stub_hint_jitter_ms=2.0,
        stub_cold_penalty_ms=120.0,
        stub_prepare_ms=1.0,
        min_hint_interval_s=0.05,
        expander_first_delay_s=0.15,
        expander_interval_s=0.15,
    )
    base.update(overrides)
    return CopilotConfig(**base)


def make_client(config=None):
    config = config or fast_config()
    service = CopilotService(
        config=config,
        backends=build_backend_set(config),
        default_briefing=parse_briefing(stub_briefing_text(), source="stub"),
    )
    return service, build_app(service)


@pytest.fixture
def client():
    service, app = make_client()
    with TestClient(app) as test_client:
        test_client.service = service
        yield test_client


def _receive_json(ws, timeout=RECEIVE_TIMEOUT_S):
    """Bounded read.

    ``WebSocketTestSession.receive_json`` blocks forever when the expected
    frame never arrives, which turns a broken assertion into a wedged test run
    -- and a wedged run is indistinguishable from a slow one. The bound is
    enforced through the session's own anyio portal. If starlette ever removes
    the stream this reaches for, this helper FAILS LOUDLY rather than falling
    back to an unbounded read.
    """
    import anyio

    stream = getattr(ws, "_send_rx", None)
    if stream is None:  # pragma: no cover - version guard
        raise AssertionError(
            "starlette's WebSocketTestSession no longer exposes _send_rx; "
            "update the bounded-receive helper instead of blocking forever"
        )

    async def _recv():
        with anyio.fail_after(timeout):
            return await stream.receive()

    message = ws.portal.call(_recv)
    if message["type"] == "websocket.close":
        raise AssertionError(f"socket closed while waiting: {message}")
    return json.loads(message["text"])


def drain_for(ws, seconds=0.6):
    """Everything that arrives within a WALL-CLOCK window. May be empty.

    A separate helper because the absence of a frame is not the failure of a
    condition: ``collect`` would sit on its own generous deadline and report a
    timeout, which reads as a broken test rather than as the proven silence it
    actually is.

    The bound is on the window, not on each read. Bounding only the read makes
    the helper loop forever against a chatty server, which is how the can-fail
    proof for the audio gate first showed up as a hung run instead of a red one
    -- the same lesson as the bounded-receive helper, learned twice.
    """
    deadline = time.monotonic() + seconds
    frames = []
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            return frames
        try:
            frames.append(_receive_json(ws, timeout=left))
        except TimeoutError:
            return frames


def collect(ws, until, limit=4000, feed=None):
    """Read frames until ``until(frames)`` is true. Returns everything read.

    ``feed`` is called every 20 frames so a test can keep audio flowing: the
    stub speaks only while a track is streaming, exactly as a real transcriber
    only transcribes what it is given.
    """
    frames = []
    for i in range(limit):
        frames.append(_receive_json(ws))
        if until(frames):
            return frames
        if feed is not None and i % 20 == 19:
            feed()
    raise AssertionError(
        f"condition never met after {limit} frames; kinds seen: "
        f"{[f.get('kind') for f in frames[-40:]]}"
    )


def kinds(frames):
    return [f.get("kind") for f in frames]


def of_kind(frames, kind):
    return [f for f in frames if f.get("kind") == kind]


def audio(track, n=1, seq=0):
    return [encode_audio_frame(track, bytes(640), seq + i) for i in range(n)]


class Feeder:
    """Keeps both tracks streaming, which is what makes the stub speak."""

    def __init__(self, ws, tracks):
        self.ws = ws
        self.tracks = tracks
        self.seq = 0

    def __call__(self, n=3):
        for track in self.tracks:
            for _ in range(n):
                self.ws.send_bytes(encode_audio_frame(track, bytes(640), self.seq))
                self.seq += 1


def start(ws, tracks=(Track.SELF, Track.OTHER)):
    """Handshake, open the given tracks, return (ready frame, feeder)."""
    ws.send_json({"kind": "hello"})
    ready = collect(ws, lambda f: kinds(f)[-1] == "session.ready")[-1]
    for track in tracks:
        ws.send_json({"kind": "track.open", "track": track.value})
    feeder = Feeder(ws, tracks)
    feeder()
    return ready, feeder


class TestItem1TwoTracksNeverMixed:
    """Acceptance 1, server half: both sources attributed, never merged."""

    def test_both_tracks_produce_lines_under_their_own_attribution(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            _, feed = start(ws)
            frames = collect(
                ws,
                lambda f: (
                    {x["track"] for x in of_kind(f, "transcript.line")}
                    >= {"self", "other"}
                ),
                feed=feed,
            )
        lines = of_kind(frames, "transcript.line")
        assert {ln["track"] for ln in lines} == {"self", "other"}
        # The script alternates, and the two sides say different things: a
        # single mixed stream could not produce this.
        by_track = {}
        for ln in lines:
            by_track.setdefault(ln["track"], []).append(ln["text"])
        assert by_track["self"][0] != by_track["other"][0]

    def test_a_frame_for_a_closed_track_is_refused_not_reattributed(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            start(ws, tracks=(Track.SELF,))
            ws.send_bytes(audio(Track.OTHER)[0])
            frames = collect(
                ws,
                lambda f: any(x.get("stage") == "audio" for x in of_kind(f, "error")),
            )
        err = [x for x in of_kind(frames, "error") if x["stage"] == "audio"][-1]
        assert err["track"] == "other"
        assert of_kind(frames, "transcript.line") == [] or all(
            ln["track"] == "self" for ln in of_kind(frames, "transcript.line")
        )

    def test_the_stub_speaks_only_while_audio_flows(self):
        """No audio, no words -- otherwise a dead capture chain looks fine.

        ``stub_silence_hold_s`` is turned down for THIS test specifically: with
        the shipped five-second hold, a stub that ignored the audio gate would
        still emit nothing inside a short observation window, and the test would
        pass without the gate existing. Measured: with the gate removed and the
        hold at 5 s the assertion held anyway; with the hold at 0.15 s it fails,
        which is what makes it a test of the gate rather than of the clock.
        """
        config = fast_config(stub_silence_hold_s=0.15)
        service, app = make_client(config)
        with TestClient(app) as tc:
            with tc.websocket_connect("/api/copilot/stream") as ws:
                ws.send_json({"kind": "hello"})
                collect(ws, lambda f: kinds(f)[-1] == "session.ready")
                ws.send_json({"kind": "track.open", "track": "self"})
                collect(ws, lambda f: kinds(f)[-1] == "track.state")
                # Never send a single audio frame. The prepare loop does emit
                # topic state in this window, so the assertion is about
                # TRANSCRIPT, not about an idle socket.
                quiet = drain_for(ws, 1.0)
                assert of_kind(quiet, "transcript.delta") == []
                assert of_kind(quiet, "transcript.line") == []
            assert list(service.sessions.sessions.values())[0].transcript == []


class TestItem2SuggestionsAndMeasuredLatency:
    """Acceptance 2: a continuously updating read pane, latency measured."""

    def test_partials_arrive_before_the_final_line(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            _, feed = start(ws, tracks=(Track.OTHER,))
            frames = collect(ws, lambda f: of_kind(f, "transcript.line"), feed=feed)
        seq = kinds(frames)
        assert seq.index("transcript.delta") < seq.index("transcript.line")
        deltas = of_kind(frames, "transcript.delta")
        # Growing text, which is what makes the pane update continuously.
        assert len(deltas) >= 3
        assert len(deltas[-1]["text"]) > len(deltas[0]["text"])
        assert all(d["stub"] is True for d in deltas)

    def test_every_hint_names_its_source_and_its_latency(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            _, feed = start(ws, tracks=(Track.OTHER,))
            frames = collect(ws, lambda f: of_kind(f, "hint"), feed=feed)
        hint = of_kind(frames, "hint")[0]
        assert hint["source_kind"] in ("line", "partial")
        assert hint["source_line_id"] is not None or hint["source_item_id"] is not None
        assert isinstance(hint["pipeline_ms"], (int, float))
        assert hint["pipeline_ms"] >= 0
        assert hint["bullets"]
        assert hint["stub"] is True

    def test_a_pending_frame_precedes_every_hint(self, client):
        """The pane must be able to say THINKING instead of standing still."""
        with client.websocket_connect("/api/copilot/stream") as ws:
            _, feed = start(ws, tracks=(Track.OTHER,))
            frames = collect(ws, lambda f: of_kind(f, "hint"), feed=feed)
        seq = kinds(frames)
        assert seq.index("hint.pending") < seq.index("hint")
        pending = of_kind(frames, "hint.pending")[0]
        hint = of_kind(frames, "hint")[0]
        assert pending["topic_id"] == hint["topic_id"]

    def test_the_pipeline_figure_is_bounded_by_the_stub_latency(self, client):
        """A latency number nobody can check is decoration.

        The stub's own decode latency is configured, so the app's reported
        line-to-hint figure must be at least that and not absurdly more. This
        catches a pipeline_ms wired to the wrong clock -- the failure mode that
        makes a latency display worse than none.
        """
        config = fast_config(stub_hint_latency_ms=60.0, stub_hint_jitter_ms=0.0)
        service, app = make_client(config)
        with TestClient(app) as tc:
            with tc.websocket_connect("/api/copilot/stream") as ws:
                _, feed = start(ws, tracks=(Track.OTHER,))
                frames = collect(ws, lambda f: of_kind(f, "hint"), feed=feed)
            hint = of_kind(frames, "hint")[0]
            assert hint["pipeline_ms"] >= 55.0
            assert hint["pipeline_ms"] < 5000.0
            session = list(service.sessions.sessions.values())[0]
            report = session.latency_report()
            assert report["samples"] >= 1
            assert report["p50_ms"] >= 55.0
            assert report["stub"] is True


class TestItem3TopicSwitchWithoutReload:
    """Acceptance 3: selecting another prepared session swaps the context."""

    def test_switching_swaps_the_suggestion_context_on_the_live_socket(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            _, feed = start(ws, tracks=(Track.OTHER,))
            frames = collect(ws, lambda f: of_kind(f, "hint"), feed=feed)
            first_topic = of_kind(frames, "hint")[0]["topic_id"]
            ws.send_json({"kind": "topic.focus", "topic_id": "migration-timeline"})
            after = collect(
                ws,
                lambda f: any(
                    h["topic_id"] == "migration-timeline" for h in of_kind(f, "hint")
                ),
                feed=feed,
            )
        assert first_topic != "migration-timeline"
        focus = [
            f for f in of_kind(after, "topic.state") if f.get("reason") == "focus"
        ][0]
        assert focus["topic_id"] == "migration-timeline"
        assert isinstance(focus["prepared"], bool)
        assert focus["switch_ms"] < 50.0
        assert any(
            h["topic_id"] == "migration-timeline" for h in of_kind(after, "hint")
        )

    def test_a_prepared_switch_is_cheaper_than_an_unprepared_one(self, client):
        """The one thing the prepared-context mechanism claims to buy.

        The stub briefing has four sections against a capacity of three, so one
        topic is measurably NOT prepared. Switching to it must cost more than
        switching to a held one -- and if it ever does not, this assertion is
        how the claim gets withdrawn instead of repeated.
        """
        service = client.service
        prep = service.backends.prep
        with client.websocket_connect("/api/copilot/stream") as ws:
            _, feed = start(ws, tracks=(Track.OTHER,))
            collect(ws, lambda f: of_kind(f, "hint"), feed=feed)
            held = set(prep.report()["held"])
            all_topics = {
                "contract-renewal",
                "support-sla",
                "migration-timeline",
                "price-adjustment",
            }
            cold = sorted(all_topics - held)
            assert cold, f"capacity {prep.capacity} held everything: {held}"
            warm = sorted(held)[0]

            ws.send_json({"kind": "topic.focus", "topic_id": warm})
            warm_frames = collect(
                ws,
                lambda f: any(h["topic_id"] == warm for h in of_kind(f, "hint")),
                feed=feed,
            )
            warm_ms = [
                h for h in of_kind(warm_frames, "hint") if h["topic_id"] == warm
            ][0]["latency_ms"]

            ws.send_json({"kind": "topic.focus", "topic_id": cold[0]})
            cold_frames = collect(
                ws,
                lambda f: any(h["topic_id"] == cold[0] for h in of_kind(f, "hint")),
                feed=feed,
            )
            cold_hints = [
                h for h in of_kind(cold_frames, "hint") if h["topic_id"] == cold[0]
            ]
            cold_ms = cold_hints[0]["latency_ms"]

        assert cold_ms > warm_ms
        assert cold_ms - warm_ms >= 0.5 * client.service.config.stub_cold_penalty_ms

    def test_an_unknown_topic_is_refused_not_silently_ignored(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            start(ws, tracks=(Track.SELF,))
            ws.send_json({"kind": "topic.focus", "topic_id": "no-such-topic"})
            frames = collect(
                ws,
                lambda f: any(x.get("stage") == "topic" for x in of_kind(f, "error")),
            )
        err = [x for x in of_kind(frames, "error") if x["stage"] == "topic"][0]
        assert "no-such-topic" in err["message"]


class TestItem4BriefingEditorAndBackgroundExtension:
    """Acceptance 4: pre-brief, then extension events during the session."""

    def test_a_briefing_set_before_capture_becomes_the_topic_set(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            ws.send_json({"kind": "hello"})
            collect(ws, lambda f: kinds(f)[-1] == "session.ready")
            ws.send_json(
                {
                    "kind": "briefing.set",
                    "text": "# Pre-brief\n\n## Notice period\nSixty days.\n",
                }
            )
            frames = collect(
                ws,
                lambda f: [
                    x for x in of_kind(f, "topic.state") if x.get("reason") == "primed"
                ],
            )
        update = of_kind(frames, "briefing.update")[0]
        assert update["reason"] == "briefing.set"
        assert [s["anchor"] for s in update["briefing"]["sections"]] == [
            "notice-period"
        ]
        primed = [x for x in of_kind(frames, "topic.state") if x["reason"] == "primed"][
            0
        ]
        assert primed["primed_tokens"] > 0
        # A prime alone never claims warmth.
        assert primed["warmth"] == "unknown"

    def test_the_editor_can_read_the_current_document_back(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            start(ws, tracks=(Track.SELF,))
            ws.send_json({"kind": "briefing.get"})
            frames = collect(
                ws,
                lambda f: [
                    x
                    for x in of_kind(f, "briefing.update")
                    if x.get("reason") == "briefing.get"
                ],
            )
        got = [
            x
            for x in of_kind(frames, "briefing.update")
            if x["reason"] == "briefing.get"
        ][0]
        assert "## Support SLA" in got["markdown"]
        assert got["briefing"]["sections"]

    def test_background_extension_events_arrive_during_the_session(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            _, feed = start(ws, tracks=(Track.OTHER,))
            frames = collect(
                ws,
                lambda f: [
                    x
                    for x in of_kind(f, "briefing.update")
                    if x.get("reason") == "expanded"
                ],
                feed=feed,
            )
        expanded = [
            x for x in of_kind(frames, "briefing.update") if x["reason"] == "expanded"
        ][0]
        assert expanded["generated"] is True
        assert expanded["added_title"]
        assert expanded["added_body"]
        # The generated section is marked as such in the document the editor
        # shows, so a reader can always tell it from what they wrote.
        gen = [s for s in expanded["briefing"]["sections"] if s["generated"]]
        assert gen and gen[0]["provenance"].startswith("copilot expansion")

    def test_an_extension_never_edits_user_text(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            _, feed = start(ws, tracks=(Track.OTHER,))
            collect(
                ws,
                lambda f: [
                    x
                    for x in of_kind(f, "briefing.update")
                    if x.get("reason") == "expanded"
                ],
                feed=feed,
            )
        session = list(client.service.sessions.sessions.values())[0]
        original = parse_briefing(stub_briefing_text())
        for section in original.sections:
            assert session.briefing.section(section.anchor).body == section.body


class TestItem5ReconnectKeepsHistory:
    """Acceptance 5: killing the connection mid-session loses nothing."""

    def test_a_killed_connection_resumes_with_transcript_and_hints_intact(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            ready, feed = start(ws, tracks=(Track.OTHER,))
            session_id = ready["session_id"]
            # BOTH, because a hint can be triggered by a partial: waiting only
            # for a hint can leave the transcript empty and assert nothing.
            frames = collect(
                ws,
                lambda f: of_kind(f, "hint") and of_kind(f, "transcript.line"),
                feed=feed,
            )
            before_lines = [ln["text"] for ln in of_kind(frames, "transcript.line")]
            before_hints = [h["hint_id"] for h in of_kind(frames, "hint")]
            cursor = max(f["seq"] for f in frames if "seq" in f) + 1

        # The transport died; the conversation did not.
        killed = client.post("/api/copilot/stub/disconnect").json()
        assert killed["closed"] >= 0

        with client.websocket_connect("/api/copilot/stream") as ws:
            ws.send_json({"kind": "hello", "session_id": session_id, "resume_from": 0})
            frames = collect(ws, lambda f: kinds(f)[-1] == "session.ready", limit=2)
            ready = frames[-1]
            assert ready["session_id"] == session_id
            replayed = collect(
                ws, lambda f: len(f) >= ready["replayed"], limit=ready["replayed"] + 5
            )
        replay_lines = [ln["text"] for ln in of_kind(replayed, "transcript.line")]
        replay_hints = [h["hint_id"] for h in of_kind(replayed, "hint")]
        assert before_lines and before_lines == replay_lines[: len(before_lines)]
        assert before_hints and before_hints == replay_hints[: len(before_hints)]
        assert all(f.get("replay") is True for f in replayed if "seq" in f)
        assert ready["next_seq"] >= cursor

    def test_a_resume_from_a_cursor_replays_only_what_is_new(self, client):
        with client.websocket_connect("/api/copilot/stream") as ws:
            ready, feed = start(ws, tracks=(Track.OTHER,))
            session_id = ready["session_id"]
            frames = collect(ws, lambda f: of_kind(f, "transcript.line"), feed=feed)
            cursor = max(f["seq"] for f in frames if "seq" in f) + 1

        with client.websocket_connect("/api/copilot/stream") as ws:
            ws.send_json(
                {
                    "kind": "hello",
                    "session_id": session_id,
                    "resume_from": cursor,
                }
            )
            ready = collect(ws, lambda f: kinds(f)[-1] == "session.ready", limit=2)[-1]
        # Everything the client already had is NOT resent, and the cursor the
        # server hands back is the first sequence it has not delivered.
        assert ready["replay_from"] == cursor
        assert ready["next_seq"] >= cursor

    def test_the_journal_floor_is_reported_instead_of_silently_truncating(self):
        config = fast_config(journal_max_events=8)
        service, app = make_client(config)
        with TestClient(app) as tc:
            with tc.websocket_connect("/api/copilot/stream") as ws:
                ready, feed = start(ws, tracks=(Track.OTHER,))
                session_id = ready["session_id"]
                # Wait for the journal to have PROVABLY overflowed: more
                # journalled events emitted than it retains. Waiting for "a
                # hint" instead left the floor at zero whenever the hint came
                # early, and the test then failed for a reason that had nothing
                # to do with what it asserts.
                collect(
                    ws,
                    lambda f: (
                        len([x for x in f if "seq" in x])
                        > 2 * config.journal_max_events
                    ),
                    feed=feed,
                )

            with tc.websocket_connect("/api/copilot/stream") as ws:
                ws.send_json(
                    {"kind": "hello", "session_id": session_id, "resume_from": 0}
                )
                frames = collect(ws, lambda f: kinds(f)[-1] == "resume.gap", limit=40)
        gap = of_kind(frames, "resume.gap")[0]
        assert gap["requested"] == 0
        assert gap["floor"] > 0


class TestItem6HonestDegradation:
    """Acceptance 6: a broken backend is visible, never a silent freeze."""

    def test_a_failing_hint_backend_produces_a_named_degraded_error(self, client):
        assert client.post(
            "/api/copilot/stub/hint-fault", json={"fail": True}
        ).json() == {"fail": True}
        with client.websocket_connect("/api/copilot/stream") as ws:
            _, feed = start(ws, tracks=(Track.OTHER,))
            frames = collect(
                ws,
                lambda f: [x for x in of_kind(f, "error") if x.get("stage") == "hint"],
                feed=feed,
            )
        err = [x for x in of_kind(frames, "error") if x["stage"] == "hint"][0]
        assert err["degraded"] is True
        assert "StubFault" in err["message"]
        # The pending frame was already on screen, so the pane knows which card
        # to mark failed rather than leaving it spinning forever.
        assert of_kind(frames, "hint.pending")
        assert of_kind(frames, "hint") == []

    def test_the_transcript_keeps_running_while_hints_are_down(self, client):
        """Degradation must be partial: the user still reads the conversation."""
        client.post("/api/copilot/stub/hint-fault", json={"fail": True})
        with client.websocket_connect("/api/copilot/stream") as ws:
            _, feed = start(ws, tracks=(Track.OTHER,))
            frames = collect(
                ws, lambda f: len(of_kind(f, "transcript.line")) >= 2, feed=feed
            )
        assert [x for x in of_kind(frames, "error") if x["stage"] == "hint"]
        assert len(of_kind(frames, "transcript.line")) >= 2

    def test_recovery_is_automatic_once_the_fault_is_cleared(self, client):
        client.post("/api/copilot/stub/hint-fault", json={"fail": True})
        with client.websocket_connect("/api/copilot/stream") as ws:
            _, feed = start(ws, tracks=(Track.OTHER,))
            collect(
                ws,
                lambda f: [x for x in of_kind(f, "error") if x.get("stage") == "hint"],
                feed=feed,
            )
            client.post("/api/copilot/stub/hint-fault", json={"fail": False})
            frames = collect(ws, lambda f: of_kind(f, "hint"), feed=feed)
        assert of_kind(frames, "hint")[0]["bullets"]

    def test_an_overflowing_socket_is_told_and_closed_not_thinned(self):
        """A client slower than the conversation must never be served gaps."""
        config = fast_config(subscriber_queue_max=1)
        service, app = make_client(config)
        with TestClient(app) as tc:
            with tc.websocket_connect("/api/copilot/stream") as ws:
                ws.send_json({"kind": "hello"})
                collect(ws, lambda f: kinds(f)[-1] == "session.ready")
                session = list(service.sessions.sessions.values())[0]

                async def flood():
                    # On the APP's loop, and without an await between emits, so
                    # the writer provably cannot drain in between. Emitting from
                    # the test thread would poke an asyncio.Queue from the wrong
                    # thread and prove nothing.
                    for _ in range(50):
                        session.emit(ServerFrame.SESSION_STATE, {"flood": True})

                ws.portal.call(flood)
                frames = collect(
                    ws,
                    lambda f: [
                        x for x in of_kind(f, "error") if x.get("stage") == "transport"
                    ],
                    limit=60,
                )
        err = [x for x in of_kind(frames, "error") if x["stage"] == "transport"][0]
        assert "overflow" in err["message"]

    def test_health_names_the_backend_so_a_stub_run_is_never_mistaken_for_real(
        self, client
    ):
        body = client.get("/api/copilot/health").json()
        assert body["backend"] == "stub"
        assert body["stub"] is True
        assert body["prep"]["capacity"] == 3


class TestStubRoutesAreNotAFeature:
    def test_fault_routes_exist_only_for_a_non_real_backend(self, client):
        """They are the acceptance harness, not part of the product."""
        paths = {r.path for r in client.app.routes}
        assert "/api/copilot/stub/disconnect" in paths
        assert "/api/copilot/stub/hint-fault" in paths
        # And they are registered from the backend set, not from a global flag,
        # so a real backend cannot carry them in.
        from sglang.srt.copilot.deskfakes import desk_fake_backend_set

        config = CopilotConfig()
        real_ish = desk_fake_backend_set(config)
        real_ish.stub = False
        app = build_app(CopilotService(config=config, backends=real_ish))
        assert "/api/copilot/stub/disconnect" not in {r.path for r in app.routes}


class TestStubQuotesTheConversationNotItsOwnInstruction:
    """Regression from the first real browser run.

    Every card ended with ``heard: "to say out loud. No greetings, no
    preamble."`` -- the stub was quoting the hint INSTRUCTION back at the user,
    because it read the prompt to the end instead of stopping at the tail.
    Nothing in the hermetic suite noticed, because nothing asserted on the
    quoted text.
    """

    def test_the_echoed_line_comes_from_the_transcript(self, client):
        from sglang.srt.copilot.hints import HINT_INSTRUCTION

        with client.websocket_connect("/api/copilot/stream") as ws:
            _, feed = start(ws, tracks=(Track.OTHER,))
            frames = collect(
                ws,
                lambda f: [
                    h
                    for h in of_kind(f, "hint")
                    if any(b.startswith("heard:") for b in h["bullets"])
                ],
                feed=feed,
            )
        heard = [
            b
            for h in of_kind(frames, "hint")
            for b in h["bullets"]
            if b.startswith("heard:")
        ]
        assert heard
        instruction_words = set(HINT_INSTRUCTION.lower().split())
        for quote in heard:
            words = set(quote[len("heard:") :].strip(" “”").lower().split())
            # A quote drawn from the instruction would overlap it almost
            # entirely; a quote from the scripted conversation barely does.
            overlap = len(words & instruction_words) / max(1, len(words))
            assert overlap < 0.6, quote
