# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The phone's telemetry lands on disk, and says where.

User order (2026-08-04): "die website braucht quasi ganz oben ein button, wo
ich alle lokalen handy logdaten an den server uebertraegt ... so stochern wir
die ganze zeit doch nur blind herum."

The server half is small on purpose -- the value is entirely in the client's
ring buffer -- but every one of its refusals matters, because this endpoint is
pressed exactly when something is already wrong. A store that silently drops
the package, or accepts a hostile filename, converts a missing diagnosis into
a false belief that one exists.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

from sglang.srt.translator import server as server_module  # noqa: E402
from sglang.srt.translator.server import build_app  # noqa: E402
from test_audio_and_http import build_service  # noqa: E402


class TestClientLogEndpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved_dir = server_module.CLIENT_LOG_DIR
        server_module.CLIENT_LOG_DIR = self.tmp
        self.client = TestClient(build_app(build_service()))

    def tearDown(self):
        server_module.CLIENT_LOG_DIR = self._saved_dir

    def test_a_package_is_stored_and_the_path_comes_back(self):
        payload = {
            "session_id": "abc123",
            "client_build": "test-build",
            "entries": [{"t": 1, "k": "boot", "d": None},
                        {"t": 2, "k": "ws.open", "d": {"attempt": 0}}],
        }
        response = self.client.post("/api/translator/client-logs", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()
        self.assertTrue(answer["stored"])
        self.assertEqual(answer["entries"], 2)
        path = Path(answer["path"])
        self.assertTrue(path.exists(), f"nothing was written at {path}")
        # The path must be usable by whoever reads the response.
        stored = json.loads(path.read_text())
        self.assertEqual(stored["session_id"], "abc123")
        self.assertEqual(stored["client_build"], "test-build")
        self.assertIn("received_at", stored, "the server did not stamp arrival")
        self.assertEqual(stored["received_bytes"], len(response.request.content))

    def test_a_package_without_a_session_is_still_stored(self):
        """THE CASE THAT MATTERS MOST, and the reason this is not a
        /sessions/{id} route: a client whose session never opened has no id to
        post under, and that failure is exactly the one nobody can diagnose.
        """
        response = self.client.post(
            "/api/translator/client-logs", json={"entries": []}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("nosession", response.json()["path"])

    def test_a_hostile_session_id_cannot_escape_the_directory(self):
        response = self.client.post(
            "/api/translator/client-logs",
            json={"session_id": "../../etc/passwd", "entries": []},
        )
        self.assertEqual(response.status_code, 200, response.text)
        path = Path(response.json()["path"])
        self.assertEqual(path.parent, Path(self.tmp), "the write escaped the store")
        self.assertNotIn("..", path.name)

    def test_a_package_over_the_limit_is_refused_with_the_size_named(self):
        saved = server_module.CLIENT_LOG_MAX_BYTES
        server_module.CLIENT_LOG_MAX_BYTES = 512
        try:
            response = self.client.post(
                "/api/translator/client-logs",
                json={"session_id": "s", "entries": ["x" * 2000]},
            )
        finally:
            server_module.CLIENT_LOG_MAX_BYTES = saved
        self.assertEqual(response.status_code, 413)
        self.assertIn("512", response.text)

    def test_a_non_object_body_is_refused(self):
        response = self.client.post("/api/translator/client-logs", json=[1, 2, 3])
        self.assertEqual(response.status_code, 400)

    def test_malformed_json_is_refused_rather_than_stored(self):
        response = self.client.post(
            "/api/translator/client-logs",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
