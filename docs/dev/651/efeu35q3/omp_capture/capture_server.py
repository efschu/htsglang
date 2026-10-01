#!/usr/bin/env python3
"""efeu-TP14: OpenAI-compatible capture endpoint for omp's requests.

Records every POST body to <dir>/req_<n>.json and answers with a minimal
completion ("ok"), streaming or not. Lets us see omp's exact system prompt
WITHOUT touching the user's model service.

    python3 capture_server.py <port> <dir>
"""
import json
import pathlib
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT, DIR = int(sys.argv[1]), pathlib.Path(sys.argv[2])
DIR.mkdir(parents=True, exist_ok=True)
N = [0]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        b = json.dumps({"object": "list", "data": [{"id": "qwen38-35b-a3b", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        N[0] += 1
        (DIR / f"req_{N[0]}.json").write_bytes(body)
        req = json.loads(body or b"{}")
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"id": "c", "object": "chat.completion.chunk", "created": int(time.time()), "model": req.get("model"),
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
                {"id": "c", "object": "chat.completion.chunk", "created": int(time.time()), "model": req.get("model"),
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            ):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        b = json.dumps({"id": "c", "object": "chat.completion", "created": int(time.time()), "model": req.get("model"),
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


HTTPServer(("127.0.0.1", PORT), H).serve_forever()
