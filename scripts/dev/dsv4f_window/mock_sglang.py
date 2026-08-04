#!/usr/bin/env python3
"""A mock sglang /generate endpoint, for validating probes without a GPU boot.

Why this exists: the probe suite's server-facing path was desk-written and
never executed against a real stream. Two ~6-minute DSV4F boots were spent
discovering that `stream_bounded` returned one chunk and zero tokens, which
made every decode floor `None` and would have silently voided the ms/round
numbers that #470 and #462 both depend on. A boot is far too expensive an
instrument-test harness.

The SSE shape reproduced here was captured verbatim from the live 1a arm:

    data: {"text":" Paris","output_ids":[11111],"meta_info":{...,
           "completion_tokens":1,"finish_reason":null,...}}
    <blank line>
    data: ...
    data: [DONE]

Note the blank line between events and the cumulative `text` / `completion_tokens`.
`spec_verify_ct` is emitted ONLY on the final chunk when --spec is given,
matching tokenizer_manager.py:2145-2153, so the probe's handling of the
speculative case is exercised too.

Run: python3 mock_sglang.py --port 31999 [--spec] [--delay-ms 20]
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ARGS = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: A002 - silence the default access log
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path in ("/health", "/health_generate"):
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json"})
            return

        if self.path == "/v1/chat/completions":
            self._json(
                200,
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": "Paris"}}
                    ]
                },
            )
            return

        if self.path != "/generate":
            self._json(404, {"error": "not found"})
            return

        sp = body.get("sampling_params") or {}
        want = int(sp.get("max_new_tokens", 16))
        ctx = ARGS.context_length
        # The real server rejects a request whose max_new_tokens exceeds the
        # context. This is the exact failure that produced the 1-chunk,
        # empty-meta record on the live arm, so the mock reproduces it.
        if want > ctx:
            self._json(
                400,
                {"object": "error", "message": f"max_new_tokens {want} > context {ctx}"},
            )
            return

        n_tok = min(want, ARGS.max_tokens)
        if not body.get("stream"):
            self._json(
                200,
                {
                    "text": " token" * n_tok,
                    "meta_info": self._meta(n_tok, final=True),
                },
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for i in range(1, n_tok + 1):
                time.sleep(ARGS.delay_ms / 1000.0)
                chunk = {
                    "text": " token" * i,
                    "output_ids": list(range(i)),
                    "meta_info": self._meta(i, final=(i == n_tok)),
                }
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the probe cut the stream off; that is a valid probe action

    def _meta(self, completion_tokens: int, final: bool) -> dict:
        meta = {
            "id": "mock",
            "prompt_tokens": 5,
            "completion_tokens": completion_tokens,
            "cached_tokens": 0,
            "finish_reason": {"type": "length"} if final else None,
        }
        # Speculative counters exist ONLY on the final chunk -- the property
        # verified at tokenizer_manager.py:2145-2153.
        if final and ARGS.spec:
            meta["spec_verify_ct"] = max(1, completion_tokens // 3)
            meta["spec_num_correct_drafts"] = completion_tokens // 2
            # The real server derives this at tokenizer_manager.py:2421.
            meta["spec_accept_length"] = completion_tokens / max(1, completion_tokens // 3)
        return meta


def main() -> int:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31999)
    ap.add_argument("--spec", action="store_true")
    ap.add_argument("--delay-ms", type=float, default=20.0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--context-length", type=int, default=8192)
    ARGS = ap.parse_args()
    srv = HTTPServer(("127.0.0.1", ARGS.port), Handler)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
