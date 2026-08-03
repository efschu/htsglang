# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The translator's HTTP/WebSocket surface.

===============================================  ============================
``GET  /api/translator/languages``               runtime-derived language set
``GET  /api/translator/health``                  backends, budgets, sessions
``GET  /api/translator/sessions``                open conversations
``POST /api/translator/sessions``                open one explicitly
``POST /api/translator/enroll/{session_id}``     register a known voice
``WS   /api/translator/stream``                  the audio conversation
``GET  /``                                       the PWA client
===============================================  ============================

**The WebSocket protocol.** One connection carries both directions. Text
frames are JSON control messages; binary frames are codec payloads. The client
opens with a ``hello`` naming its session id (or none, for a new conversation),
its participant languages, its codec offers and its resume cursor. The server
answers ``ready`` with the negotiated codec, the language matrix and the
journal position, then replays anything the client missed.

The uplink is deliberately dumb: the client streams audio frames and, in
push-to-talk mode, sends ``release``. All turn logic lives here, because the
phone is the component we cannot debug from a train in Spain.

**Reconnect.** The client reconnects to the same URL with the same
``session_id`` and its last processed ``seq``. The session is found by id,
its segmenter is reset (the audio in flight is gone), and the journal replays
from the cursor. If the cursor predates the journal floor, a ``resume.gap``
event precedes the replay so the client knows samples are missing rather than
silently continuing. Nothing about a reconnect touches the speaker registry --
the reference buffers, which are the expensive thing to rebuild, survive.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from sglang.srt.translator.audio import (
    PIPELINE_SAMPLE_RATE,
    AudioCodec,
    CodecError,
    Pcm16Codec,
    available_codecs,
    encode_event_audio,
    negotiate_codec,
    resample,
)
from sglang.srt.translator.backends import AudioChunk
from sglang.srt.translator.config import TranslatorConfig
from sglang.srt.translator.languages import (
    ConversationLanguages,
    LanguageError,
    LanguageMatrix,
    display_name,
)
from sglang.srt.translator.session import (
    EventKind,
    Journal,
    SessionManager,
    TranslatorSession,
)
from sglang.srt.translator.voices import (
    OutputMode,
    VoiceClass,
    VoiceMode,
    VoicePool,
    VoicePoolError,
)

logger = logging.getLogger(__name__)

__all__ = ["TranslatorService", "build_app", "CLIENT_DIR"]

CLIENT_DIR = Path(__file__).resolve().parent / "client"


@dataclasses.dataclass
class Stack:
    """The four live backends. Built by the caller, injected here.

    The service never constructs a backend. That is what lets the whole HTTP
    surface be tested with fakes under ``CUDA_VISIBLE_DEVICES=99``, and it is
    also what lets the ASR or TTS backend be swapped without this file
    knowing which model is behind it.
    """

    asr: Any
    embedder: Any
    mt: Any
    tts: Any

    def matrix(self) -> LanguageMatrix:
        """Derive the deployment language set from the live backends.

        Called per request rather than cached: a backend that hot-swaps its
        checkpoint changes the answer, and an answer that is stale in exactly
        that case is worse than no answer.
        """
        mt_languages = self.mt.supported_languages()
        return LanguageMatrix.from_backends(
            asr_languages=self.asr.supported_languages(),
            tts_languages=self.tts.supported_languages(),
            mt_languages=None if mt_languages is None else list(mt_languages),
        )


class TranslatorService:
    """Owns the sessions, the stack and the config. One per process."""

    def __init__(
        self,
        config: TranslatorConfig,
        stack: Stack,
        voice_pool: Optional[VoicePool] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.stack = stack
        self.voice_mode = VoiceMode.parse(config.tts.voice_mode)
        # The pool is loaded once, at startup, and SHARED across sessions --
        # but each session keeps its own sticky assignment, so two concurrent
        # conversations do not fight over voices. See `_make_session`.
        self.voice_pool_template = voice_pool
        if voice_pool is None and config.tts.preset_voice_dir is not None:
            self.voice_pool_template = VoicePool.from_directory(
                config.tts.preset_voice_dir,
                sample_rate=config.tts.output_sample_rate,
            )
        self.started_at = time.time()
        self.sessions = SessionManager(
            factory=self._make_session,
            max_sessions=config.max_sessions,
            idle_timeout_s=config.session_idle_timeout_s,
        )

    def _make_session(
        self, session_id: str, conversation: ConversationLanguages
    ) -> TranslatorSession:
        return TranslatorSession(
            session_id=session_id,
            asr=self.stack.asr,
            embedder=self.stack.embedder,
            mt=self.stack.mt,
            tts=self.stack.tts,
            matrix=self.stack.matrix(),
            conversation=conversation,
            journal=Journal(
                max_events=self.config.journal_events,
                max_audio_bytes=self.config.journal_audio_mib << 20,
            ),
            min_reference_seconds=self.config.tts.min_reference_seconds,
            voice_mode=self.voice_mode,
            voice_pool=self._session_pool(),
        )

    def _session_pool(self) -> Optional[VoicePool]:
        """A per-session pool sharing the loaded voices but not the mapping.

        The audio is immutable and shared; the sticky speaker assignment is
        not, because two simultaneous conversations each need the full pool.
        Sharing the assignment would make speaker-1 of conversation B inherit
        conversation A's voice and then run out of voices twice as fast.
        """
        if self.voice_pool_template is None:
            return None
        return VoicePool(self.voice_pool_template.presets())

    # -- read surfaces ------------------------------------------------------

    def languages(self) -> Dict[str, object]:
        matrix = self.stack.matrix()
        payload = matrix.to_json()
        payload["backends"] = {
            "asr": getattr(self.stack.asr, "name", "unknown"),
            "mt": getattr(self.stack.mt, "name", "unknown"),
            "tts": getattr(self.stack.tts, "name", "unknown"),
            "embedder": getattr(self.stack.embedder, "name", "unknown"),
        }
        payload["default_participants"] = list(self.config.default_participants)
        # Every language any stage knows about, each labelled usable or not
        # WITH THE REASON. The client greys out the unusable ones and shows
        # why; silently filtering them would leave the user wondering whether
        # a language is missing or merely unsupported by one checkpoint.
        catalogue = sorted(matrix.asr | matrix.tts | (matrix.mt or frozenset()))
        selectable = []
        for code in catalogue:
            missing = []
            if code not in matrix.asr:
                missing.append("ASR cannot hear it")
            if code not in matrix.tts:
                missing.append("TTS cannot speak it")
            if not matrix.unconstrained_mt and code not in matrix.mt:
                missing.append("MT does not cover it")
            entry = display_name(code)
            entry["as_source"] = code in matrix.sources
            entry["as_target"] = code in matrix.targets
            entry["usable"] = code in matrix.bidirectional
            entry["reason"] = "; ".join(missing)
            selectable.append(entry)
        payload["selectable"] = selectable
        # The default pair must itself be routable; advertising a default the
        # deployment cannot run is the exact failure the runtime derivation
        # exists to prevent, so it is checked here rather than trusted.
        try:
            ConversationLanguages.of(
                self.config.default_participants
            ).validate_against(matrix)
            payload["default_participants_supported"] = True
            payload["default_participants_error"] = None
        except LanguageError as exc:
            payload["default_participants_supported"] = False
            payload["default_participants_error"] = str(exc)
        return payload

    def health(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "uptime_s": round(time.time() - self.started_at, 1),
            "sessions": len(self.sessions),
            "max_sessions": self.config.max_sessions,
            # Age and idleness per session, so an accumulation is visible
            # before it is felt as latency.
            "session_detail": self.sessions.to_json(),
            "codecs": list(available_codecs()),
            "voice_mode": self.voice_mode.value,
            "preset_voices": (
                len(self.voice_pool_template)
                if self.voice_pool_template is not None
                else 0
            ),
            "pipeline_sample_rate": PIPELINE_SAMPLE_RATE,
            "budgets_mib": {
                "asr": self.config.asr.budget_mib,
                "tts": self.config.tts.budget_mib,
                "diarization": self.config.diarization.budget_mib,
                "total": self.config.total_budget_mib(),
            },
            "backends": {
                "asr": getattr(self.stack.asr, "name", "unknown"),
                "mt": getattr(self.stack.mt, "name", "unknown"),
                "tts": getattr(self.stack.tts, "name", "unknown"),
                "embedder": getattr(self.stack.embedder, "name", "unknown"),
            },
        }

    def open_session(
        self,
        participants: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> TranslatorSession:
        conversation = ConversationLanguages.of(
            participants or list(self.config.default_participants)
        )
        return self.sessions.open(conversation, session_id=session_id)


class _Connection:
    """One WebSocket, bound to one session for its lifetime.

    Separated from the service because the session outlives the connection:
    that is the whole reconnect story, and mixing the two lifetimes is how a
    reconnect ends up with a fresh speaker registry.
    """

    def __init__(
        self, service: TranslatorService, websocket: WebSocket
    ) -> None:
        self.service = service
        self.ws = websocket
        self.session: Optional[TranslatorSession] = None
        self.codec: AudioCodec = Pcm16Codec()
        self.sent_seq = 0
        self._pump: Optional[asyncio.Task] = None
        self._closing = False

    async def run(self) -> None:
        await self.ws.accept()
        try:
            await self._handshake()
            self.session.attach()
            self._pump = asyncio.create_task(self._journal_pump())
            await self._receive_loop()
        except WebSocketDisconnect:
            logger.info("client disconnected from session %s", self._sid())
        except CodecError as exc:
            await self._send({"kind": "error", "stage": "codec", "message": str(exc)})
        except LanguageError as exc:
            await self._send({"kind": "error", "stage": "language", "message": str(exc)})
        finally:
            self._closing = True
            if self.session is not None:
                # The session stays for the resume grace; what goes now is the
                # claim that somebody is listening.
                self.session.detach()
            if self._pump is not None:
                self._pump.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._pump
            # The session is NOT closed here. A dropped mobile link must be
            # resumable, so the session lives on until the idle collector
            # takes it.
            with contextlib.suppress(Exception):
                await self.ws.close()

    def _sid(self) -> str:
        return self.session.session_id if self.session else "<none>"

    async def _handshake(self) -> None:
        raw = await self.ws.receive_text()
        hello = json.loads(raw)
        if hello.get("kind") != "hello":
            raise CodecError(f"expected a hello frame, got {hello.get('kind')!r}")

        offers = hello.get("codecs") or ["pcm16"]
        self.codec = negotiate_codec(offers)
        participants = hello.get("participants")
        session_id = hello.get("session_id")
        resume_from = int(hello.get("resume_from", 0))

        self.session = self.service.open_session(participants, session_id)
        matrix = self.service.stack.matrix()

        await self._send(
            {
                "kind": "ready",
                "session_id": self.session.session_id,
                "codec": dataclasses.asdict(self.codec.info()),
                "pipeline_sample_rate": PIPELINE_SAMPLE_RATE,
                "languages": matrix.to_json(),
                "state": self.session.state(),
                "resumed": bool(session_id),
            }
        )

        # The written record travels with the handshake, not with the journal
        # replay (§17.2). A phone whose tab was evicted comes back with
        # ``transcript_from=0`` and gets the whole conversation; one that kept
        # its lines asks for the tail. The journal cannot serve this: it
        # evicts, and a conversation older than its budget would come back
        # truncated with no way for the client to know.
        transcript_from = int(hello.get("transcript_from", 0))
        await self._send(
            {
                "kind": "transcript",
                **self.session.transcript.to_json(since=transcript_from),
            }
        )

        # Seed the delivery cursor BEFORE replaying. A fresh connection starts
        # at zero, and if the resume window happens to be empty nothing would
        # advance it -- the pump would then re-send the whole journal from the
        # beginning, which on a reconnect means replaying the entire
        # conversation the client already has.
        self.sent_seq = max(self.sent_seq, resume_from)

        events, gap = self.session.journal.since(resume_from)
        if gap:
            await self._send(
                {
                    "kind": EventKind.RESUME_GAP.value,
                    "requested_from": resume_from,
                    "available_from": self.session.journal.floor,
                    "message": (
                        "the journal no longer holds every event since the "
                        "requested cursor; some turns are missing"
                    ),
                }
            )
        for event in events:
            await self._emit(event)

    async def _receive_loop(self) -> None:
        assert self.session is not None
        while not self._closing:
            message = await self.ws.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            if (payload := message.get("bytes")) is not None:
                await self._on_audio(payload)
            elif (text := message.get("text")) is not None:
                await self._on_control(json.loads(text))

    async def _on_audio(self, payload: bytes) -> None:
        assert self.session is not None
        chunk = self.codec.decode(payload)
        if chunk.sample_rate != PIPELINE_SAMPLE_RATE:
            chunk = resample(chunk, PIPELINE_SAMPLE_RATE)
        for segment in self.session.push_audio(chunk):
            self.session.enqueue(segment)
        await self._kick()

    async def _on_control(self, message: Dict[str, object]) -> None:
        assert self.session is not None
        kind = message.get("kind")
        if kind == "release":
            segment = self.session.release()
            if segment is not None:
                self.session.enqueue(segment)
            await self._kick()
        elif kind == "state":
            await self._send({"kind": "state", "state": self.session.state()})
        elif kind == "ack":
            # The client confirming a cursor is advisory: the journal is
            # bounded by size, not by acknowledgement, so an ack cannot pin
            # memory. It exists so the server can log how far behind a client
            # actually is.
            logger.debug("session %s client acked seq %s",
                         self._sid(), message.get("seq"))
        elif kind == "suggestion.confirm":
            try:
                changed = self.session.confirm_suggestion(
                    str(message.get("suggestion_id", "")),
                    str(message.get("speaker_id") or "") or None,
                )
            except (KeyError, ValueError) as exc:
                await self._send(
                    {"kind": "error", "stage": "speaker", "message": str(exc)}
                )
                return
            await self._send(
                {"kind": "suggestion.applied",
                 "suggestion_id": message.get("suggestion_id"),
                 "lines_updated": len(changed)}
            )
        elif kind == "name.undo":
            took_back = self.session.undo_name(
                str(message.get("suggestion_id", ""))
            )
            await self._send({"kind": "name.undone", "known": took_back,
                              "state": self.session.state()})
        elif kind == "suggestion.discard":
            dropped = self.session.discard_suggestion(
                str(message.get("suggestion_id", ""))
            )
            await self._send(
                {"kind": "suggestion.discarded",
                 "suggestion_id": message.get("suggestion_id"),
                 "known": dropped}
            )
        elif kind == "line.resolve":
            # A tap on an uncertain badge (§17.4). ``speaker_id`` absent or
            # null means the "speaker-N (new)" candidate.
            changed = self.session.resolve_line(
                int(message.get("line_id", 0)),
                str(message.get("speaker_id") or "") or None,
            )
            if changed is None:
                await self._send(
                    {"kind": "error", "stage": "transcript",
                     "message": f"no line {message.get('line_id')!r}"}
                )
                return
            await self._send({"kind": "line.resolved", "line": changed.to_json()})
        elif kind == "line.undo":
            reverted = self.session.undo_resolution(int(message.get("line_id", 0)))
            if reverted is None:
                await self._send(
                    {"kind": "error", "stage": "transcript",
                     "message": f"nothing to undo on line {message.get('line_id')!r}"}
                )
                return
            await self._send({"kind": "line.resolved", "line": reverted.to_json()})
        elif kind == "speaker.arm":
            # The speaker button (§17.5). Arming is per-utterance, so the
            # client sends this immediately before the user speaks, and the
            # server answers so a lit button reflects server state rather
            # than an optimistic local guess.
            try:
                armed = self.session.arm_speaker(message.get("speaker_id") or None)
            except KeyError:
                await self._send(
                    {"kind": "error", "stage": "speaker",
                     "message": f"no speaker {message.get('speaker_id')!r}"}
                )
                return
            await self._send({"kind": "speaker.armed", "speaker_id": armed})
        elif kind == "speaker.add":
            try:
                voice_class = (
                    VoiceClass(str(message["voice_class"]).strip().lower())
                    if message.get("voice_class") else None
                )
            except ValueError:
                await self._send(
                    {"kind": "error", "stage": "voice",
                     "message": f"unknown voice class "
                                f"{message.get('voice_class')!r}"}
                )
                return
            speaker_id = self.session.add_speaker(
                str(message.get("label") or "").strip() or None,
                voice_class=voice_class,
            )
            # Arm it at once: the "+" button exists because somebody new is
            # about to speak, and requiring a second tap to say so would make
            # the first utterance -- the one that seeds the centroid -- the
            # one that goes through automatic matching.
            self.session.arm_speaker(speaker_id)
            await self._send(
                {"kind": "speaker.added", "speaker_id": speaker_id,
                 "armed": True, "state": self.session.state()}
            )
        elif kind == "output.mode":
            try:
                mode = self.session.set_output_mode(
                    OutputMode.parse(message.get("mode"))
                )
            except VoicePoolError as exc:
                await self._send(
                    {"kind": "error", "stage": "voice", "message": str(exc)}
                )
                return
            await self._send({"kind": "output.mode", "mode": mode.value})
        elif kind == "transcript":
            await self._send(
                {
                    "kind": "transcript",
                    **self.session.transcript.to_json(
                        since=int(message.get("since", 0))
                    ),
                }
            )
        elif kind == "transcript.clear":
            # Destructive and explicit: the client confirms before sending it,
            # and nothing else in the protocol empties the record.
            removed = self.session.clear_transcript()
            await self._send({"kind": "transcript.cleared", "removed": removed})
        elif kind == "speaker.delete":
            try:
                gone = self.session.delete_speaker(
                    str(message.get("speaker_id", ""))
                )
            except KeyError:
                await self._send(
                    {"kind": "error", "stage": "speaker",
                     "message": f"no speaker {message.get('speaker_id')!r}"}
                )
                return
            await self._send(
                {"kind": "speaker.deleted", **gone,
                 "state": self.session.state()}
            )
        elif kind == "speaker.name":
            try:
                changed = self.session.name_speaker(
                    str(message.get("speaker_id", "")),
                    str(message.get("label", "")).strip(),
                )
            except KeyError:
                await self._send(
                    {"kind": "error", "stage": "speaker",
                     "message": f"no speaker {message.get('speaker_id')!r}"}
                )
                return
            await self._send(
                {"kind": "speaker.named",
                 "speaker_id": message.get("speaker_id"),
                 "label": message.get("label"),
                 "lines_updated": len(changed)}
            )
        elif kind == "diag":
            # Client-side capture telemetry, logged where it can be read
            # against the session it belongs to. This is the only channel by
            # which a defect between the microphone and the socket becomes
            # visible to anyone but the person holding the phone.
            # Both halves, always, on one line: a device that hears nothing is
            # either not being sent audio or not playing it, and the two have
            # nothing in common except that they look identical from here.
            logger.info(
                "session %s capture diagnostics: %s playback: %s",
                self._sid(),
                json.dumps(message.get("capture") or {}, sort_keys=True),
                json.dumps(message.get("playback") or {}, sort_keys=True),
            )
            await self._send({"kind": "diag.ack"})
        elif kind == "ping":
            await self._send({"kind": "pong", "at": time.time()})
        elif kind == "close":
            self._closing = True
            self.service.sessions.close(self.session.session_id)
        else:
            await self._send(
                {"kind": "error", "stage": "protocol",
                 "message": f"unknown control frame {kind!r}"}
            )

    async def _kick(self) -> None:
        """Drain queued turns without blocking the receive loop.

        The receive loop must keep draining the socket while a turn runs, or
        the client's audio backs up in the kernel buffer and the next turn
        starts with stale audio. So the drain runs as its own task and the
        lock inside the session serializes turns.
        """
        assert self.session is not None
        if self.session.pending() == 0:
            return
        asyncio.create_task(self._drain_guarded())

    async def _drain_guarded(self) -> None:
        assert self.session is not None
        try:
            await self.session.drain()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("turn pipeline failed in session %s", self._sid())
            await self._send(
                {"kind": "error", "stage": "pipeline", "message": str(exc)}
            )

    async def _journal_pump(self) -> None:
        """Forward new journal events to the client as they appear.

        A poll rather than a callback: the journal is the source of truth for
        both live delivery and replay, and having exactly one delivery path
        means a reconnect cannot produce a different event stream than the
        original connection did.
        """
        assert self.session is not None
        while not self._closing:
            events, _gap = self.session.journal.since(self.sent_seq)
            for event in events:
                await self._emit(event)
                self.sent_seq = event.seq + 1
            await asyncio.sleep(0.02)

    async def _emit(self, event) -> None:
        payload = event.to_json()
        if event.audio is not None:
            # Audio rides the binary channel; the JSON event announces it so
            # the client can attribute the frames that follow to a turn.
            payload["audio_follows"] = True
            # The rate on the WIRE, which is the codec's, not the rate the
            # audio had in the session. Pcm16Codec.encode() resamples to its
            # own rate, so announcing event.audio.sample_rate described the
            # samples BEFORE the codec touched them: the TTS produces 24 kHz,
            # the codec ships 16 kHz, and the client faithfully played 16 kHz
            # samples at 24 kHz -- 1.5x too fast and a fifth too high. The
            # front-door harness could not see it because it decodes with the
            # codec and ignores this field; only a real client trusts it.
            payload["sample_rate"] = self.codec.sample_rate
            await self._send(payload)
            for frame in self.codec.encode(event.audio):
                await self.ws.send_bytes(frame)
        elif event.kind is EventKind.TURN_AUDIO:
            payload["audio_evicted"] = True
            await self._send(payload)
        else:
            await self._send(payload)
        self.sent_seq = max(self.sent_seq, event.seq + 1)

    async def _send(self, payload: Dict[str, object]) -> None:
        await self.ws.send_text(json.dumps(payload, ensure_ascii=False))


def build_app(service: TranslatorService) -> FastAPI:
    """Wire the service into a FastAPI app."""

    async def _collect_forever() -> None:
        """Collect on a TIMER, not only when a new session is opened.

        Collection used to run inside SessionManager.open(), so a server
        nobody connects to never collected and abandoned sessions accumulated
        until the next arrival -- who then paid for all of them. Six of them
        turned 4 s of first-audio latency into 52 s on the live service.
        """
        while True:
            await asyncio.sleep(15.0)
            try:
                dropped = service.sessions.collect()
            except Exception:  # noqa: BLE001 - a sweeper must not die
                logger.exception("session collection failed")
                continue
            if dropped:
                logger.info(
                    "collected %d abandoned session(s): %s",
                    len(dropped), ", ".join(dropped),
                )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        collector = asyncio.create_task(_collect_forever())
        try:
            yield
        finally:
            collector.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await collector

    app = FastAPI(
        title="htsglang live translator", version="1", lifespan=lifespan
    )
    app.state.service = service

    @app.get("/api/translator/languages")
    async def languages() -> JSONResponse:
        return JSONResponse(service.languages())

    @app.get("/api/translator/health")
    async def health() -> JSONResponse:
        return JSONResponse(service.health())

    @app.get("/api/translator/sessions")
    async def list_sessions() -> JSONResponse:
        out = []
        for sid in service.sessions.ids():
            session = service.sessions.get(sid)
            if session is not None:
                out.append(session.state())
        return JSONResponse({"sessions": out})

    @app.post("/api/translator/sessions")
    async def create_session(body: Dict[str, Any]) -> JSONResponse:
        try:
            session = service.open_session(
                body.get("participants"), body.get("session_id")
            )
        except LanguageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return JSONResponse(session.state())

    @app.post("/api/translator/enroll/{session_id}")
    async def enroll(session_id: str, body: Dict[str, Any]) -> JSONResponse:
        session = service.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        try:
            from sglang.srt.translator.audio import decode_event_audio

            audio = decode_event_audio(body["audio"])
        except (KeyError, CodecError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if audio.sample_rate != PIPELINE_SAMPLE_RATE:
            audio = resample(audio, PIPELINE_SAMPLE_RATE)
        speaker_id = await session.enroll_speaker(
            label=str(body.get("label", "user")),
            audio=audio,
            text=str(body.get("text", "")),
            language=body.get("language"),
        )
        return JSONResponse(
            {"speaker_id": speaker_id, "state": session.state()}
        )

    @app.post("/api/translator/sessions/{session_id}/routing")
    async def set_routing(session_id: str, body: Dict[str, Any]) -> JSONResponse:
        """Replace a session's manual routing table. Empty list = auto mode.

        Entries are UNORDERED pairs ``{"a": ..., "b": ...}``: one row per
        relationship, both directions, and a repeated pair is deduplicated
        rather than refused.
        """
        session = service.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        raw = body.get("pairs", body.get("rules", []))
        try:
            pairs = [(str(entry["a"]), str(entry["b"])) for entry in raw]
        except (KeyError, TypeError) as exc:
            raise HTTPException(
                status_code=400,
                detail="each pair needs an 'a' and a 'b' language code",
            ) from exc
        try:
            session.set_routing_pairs(pairs)
        except LanguageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(session.state())

    @app.get("/api/translator/voices")
    async def voices() -> JSONResponse:
        pool = service.voice_pool_template
        return JSONResponse(
            {
                "default_mode": service.voice_mode.value,
                "modes": [m.value for m in VoiceMode],
                "pool": pool.to_json() if pool is not None else None,
                "preset_available": pool is not None,
            }
        )

    @app.post("/api/translator/sessions/{session_id}/voice")
    async def set_voice(session_id: str, body: Dict[str, Any]) -> JSONResponse:
        session = service.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        try:
            if "mode" in body:
                session.set_voice_mode(VoiceMode.parse(body["mode"]))
            if "output_mode" in body:
                # Reading mode rides the voice endpoint because it is the
                # same user-facing control: "how does this session come out".
                session.set_output_mode(OutputMode.parse(body["output_mode"]))
            if "speaker_id" in body and "voice_class" in body:
                session.override_voice_class(
                    str(body["speaker_id"]), VoiceClass(str(body["voice_class"]))
                )
        except VoicePoolError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(session.state())

    @app.get("/api/translator/sessions/{session_id}/transcript")
    async def get_transcript(session_id: str, since: int = 0) -> JSONResponse:
        """The written record (§17.2), server-held so a reconnect restores it.

        ``since`` is a line cursor, not a sequence number: a client that kept
        its lines asks only for what it is missing, and one that lost them
        (fresh tab, evicted page) asks from zero and gets the conversation
        back whole.
        """
        session = service.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        return JSONResponse(session.transcript.to_json(since=since))

    @app.delete("/api/translator/sessions/{session_id}/transcript")
    async def clear_transcript(session_id: str) -> JSONResponse:
        session = service.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        return JSONResponse({"removed": session.clear_transcript()})

    @app.post("/api/translator/sessions/{session_id}/speakers/{speaker_id}/name")
    async def name_speaker(
        session_id: str, speaker_id: str, body: Dict[str, Any]
    ) -> JSONResponse:
        """Name a speaker by hand — retroactively over the whole record."""
        session = service.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        label = str(body.get("label", "")).strip()
        try:
            changed = session.name_speaker(speaker_id, label)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"no speaker {speaker_id!r}"
            ) from exc
        return JSONResponse(
            {"speaker_id": speaker_id, "label": label, "lines_updated": len(changed)}
        )

    @app.post("/api/translator/sessions/{session_id}/lines/{line_id}/speaker")
    async def resolve_line(
        session_id: str, line_id: int, body: Dict[str, Any]
    ) -> JSONResponse:
        """Settle an uncertain attribution (§17.4). No body key => new speaker."""
        session = service.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        changed = session.resolve_line(
            line_id, str(body.get("speaker_id") or "") or None
        )
        if changed is None:
            raise HTTPException(status_code=404, detail=f"no line {line_id}")
        return JSONResponse(changed.to_json())

    @app.delete("/api/translator/sessions/{session_id}/lines/{line_id}/speaker")
    async def undo_line(session_id: str, line_id: int) -> JSONResponse:
        session = service.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        reverted = session.undo_resolution(line_id)
        if reverted is None:
            raise HTTPException(
                status_code=404, detail=f"nothing to undo on line {line_id}"
            )
        return JSONResponse(reverted.to_json())

    @app.delete("/api/translator/sessions/{session_id}/speakers/{speaker_id}")
    async def delete_speaker(session_id: str, speaker_id: str) -> JSONResponse:
        """Forget a speaker: centroid, reference buffer, label, preset slot."""
        session = service.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        try:
            return JSONResponse(session.delete_speaker(speaker_id))
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"no speaker {speaker_id!r}"
            ) from exc

    @app.post("/api/translator/sessions/{session_id}/speakers")
    async def add_speaker(session_id: str, body: Dict[str, Any]) -> JSONResponse:
        """The "+" button: declare a new speaker and arm them (§17.5)."""
        session = service.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        speaker_id = session.add_speaker(
            str(body.get("label") or "").strip() or None
        )
        session.arm_speaker(speaker_id)
        return JSONResponse({"speaker_id": speaker_id, "armed": True})

    @app.post("/api/translator/sessions/{session_id}/arm")
    async def arm_speaker(session_id: str, body: Dict[str, Any]) -> JSONResponse:
        session = service.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"no session {session_id!r}")
        try:
            armed = session.arm_speaker(body.get("speaker_id") or None)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"no speaker {body.get('speaker_id')!r}"
            ) from exc
        return JSONResponse({"armed_speaker": armed})

    @app.delete("/api/translator/sessions/{session_id}")
    async def close_session(session_id: str) -> JSONResponse:
        return JSONResponse({"closed": service.sessions.close(session_id)})

    @app.websocket("/api/translator/stream")
    async def stream(websocket: WebSocket) -> None:
        await _Connection(service, websocket).run()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        page = CLIENT_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="client not installed")
        body = page.read_text(encoding="utf-8")
        # Stamp the asset with its own content hash. A phone can serve a
        # cached page for a long time, and without an identity in the page
        # itself every diagnosis is a guess about which code is running --
        # a fix that is deployed and a fix that is reaching the device look
        # exactly the same from here.
        build = hashlib.sha256(body.encode("utf-8")).hexdigest()[:10]
        body = body.replace("__CLIENT_BUILD__", build)
        return HTMLResponse(
            body,
            headers={
                # The page carries its own identity, so it must not be served
                # from a cache that predates it.
                "Cache-Control": "no-cache, must-revalidate",
            },
        )

    @app.get("/manifest.webmanifest")
    async def manifest() -> JSONResponse:
        # start_url and scope are RELATIVE ("./"), which resolves against the
        # manifest's own URL. Served at the site root that is "/"; served
        # behind a reverse-proxy prefix it becomes that prefix, so the PWA's
        # installed scope follows the mount point without the server having to
        # be told where it lives.
        return JSONResponse(
            {
                "name": "htsglang live translator",
                "short_name": "translate",
                "start_url": "./",
                "scope": "./",
                "display": "standalone",
                "background_color": "#101014",
                "theme_color": "#101014",
                "icons": [],
            }
        )

    return app


def encode_audio_event(audio: AudioChunk) -> Dict[str, object]:
    """Re-exported for the tests, which assert on the JSON audio form."""
    return encode_event_audio(audio)
