"""Minimal fake completions server, so the harness can be smoke-tested
end-to-end with no GPU and no model.

It answers /health and /v1/completions with a configurable per-request delay
and a deterministic body, which is enough to exercise every path in
prefill_perf.py and content_gate.py: span scoring, warmup, concurrency,
usage accounting and the byte-comparison oracle.

  python mock_server.py --port 30099 --delay 0.05 [--vary 0.01] [--drift-token]
"""

import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CFG = {"delay": 0.05, "vary": 0.0, "drift_token": False}
_lock = threading.Lock()
_seen = {"n": 0}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b"")
        else:
            self._send(404, b"")

    def do_POST(self):
        if self.path != "/v1/completions":
            self._send(404, b"")
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n).decode())
        prompt = req.get("prompt", "")
        # Delay proportional to prompt length so longer points really are
        # slower, the way a real prefill is.
        approx_tokens = max(1, len(prompt.split()))
        d = CFG["delay"] * approx_tokens
        if CFG["vary"]:
            d *= 1.0 + random.uniform(-CFG["vary"], CFG["vary"])
        time.sleep(d)
        with _lock:
            _seen["n"] += 1
            idx = _seen["n"]
        # drift_token flips one character after a while, so the content gate's
        # divergence path gets exercised too.
        text = " Paris is the capital of France."
        if CFG["drift_token"] and idx > 4:
            text = " Paris is the capitol of France."
        body = json.dumps(
            {
                "choices": [
                    {
                        "text": text,
                        "finish_reason": "length",
                        "logprobs": {
                            "tokens": ["a", "b"],
                            "token_logprobs": [-0.5, -0.25],
                        },
                    }
                ],
                "usage": {"prompt_tokens": approx_tokens, "cached_tokens": 0},
            }
        ).encode()
        self._send(200, body)

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--delay", type=float, default=0.002)
    ap.add_argument("--vary", type=float, default=0.0)
    ap.add_argument("--drift-token", action="store_true")
    a = ap.parse_args()
    CFG["delay"], CFG["vary"], CFG["drift_token"] = a.delay, a.vary, a.drift_token
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
