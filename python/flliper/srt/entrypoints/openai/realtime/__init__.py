"""Realtime transcription WebSocket package. Exposes the FastAPI WS entry."""

from flliper.srt.entrypoints.openai.realtime.handler import (
    handle_realtime_transcription as handle_realtime_transcription,
)
