# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``/v1/files`` for fine-tuning uploads (#341-M1).

The files endpoint is not a general object store here. It exists because
``POST /v1/fine_tuning/jobs`` takes a ``training_file`` id and every client in
the ecosystem obtains that id by calling ``client.files.create(...,
purpose="fine-tune")`` first. So this surface accepts exactly the purposes it
can act on and rejects the others by name, rather than storing a batch file it
will never process and reporting success.

The adapter's job is parse-and-shape. Every decision -- what a valid training
file is, whether a file may be deleted -- belongs to
:class:`~sglang.srt.training.service.TrainingService`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi.responses import ORJSONResponse, Response

from sglang.srt.entrypoints.openai.errors import (
    error_type_for_status,
    openai_error_response,
)
from sglang.srt.training.store import StoreError, page

logger = logging.getLogger(__name__)


def store_error_response(exc: StoreError, **extension: Any) -> ORJSONResponse:
    """One store exception, as the envelope every SDK parses."""
    return openai_error_response(
        str(exc),
        status_code=exc.status_code,
        err_type=error_type_for_status(exc.status_code),
        param=exc.param,
        code=exc.code,
        extension=extension or None,
    )


class OpenAIServingFiles:
    """Uploads and lists training data. Backed by the training service."""

    def __init__(self, service) -> None:
        self._service = service

    async def create(
        self, *, filename: str, content: bytes, purpose: str
    ) -> ORJSONResponse:
        try:
            stored = self._service.create_file(
                filename=filename, content=content, purpose=purpose
            )
        except StoreError as exc:
            return store_error_response(exc)
        return ORJSONResponse(stored.to_json())

    async def list(self, *, purpose: Optional[str] = None) -> ORJSONResponse:
        return ORJSONResponse(page(self._service.files.list(purpose=purpose)))

    async def retrieve(self, file_id: str) -> ORJSONResponse:
        try:
            return ORJSONResponse(self._service.files.get(file_id).to_json())
        except StoreError as exc:
            return store_error_response(exc)

    async def delete(self, file_id: str) -> ORJSONResponse:
        try:
            self._service.delete_file(file_id)
        except StoreError as exc:
            return store_error_response(exc)
        return ORJSONResponse({"id": file_id, "object": "file", "deleted": True})

    async def content(self, file_id: str):
        try:
            raw = self._service.files.content(file_id)
        except StoreError as exc:
            return store_error_response(exc)
        # The SDK reads this through ``.content`` / ``.text`` and does not
        # parse it, so the honest media type is the one it was uploaded as.
        return Response(content=raw, media_type="application/jsonl")
