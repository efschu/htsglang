#!/usr/bin/env python3
"""Capture what a client actually sends, without touching the GPU.

The coding agent's prompt size is the number that decides whether it fits under
this machine's wedge envelope, and measuring it by running against the real
model costs ~4 minutes and risks a GPU reset per attempt. This stands in for
the model: it records the request body and returns a 400, so the client gives
up immediately. Iterating on --tools / --system-prompt against this is free.
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

OUTDIR = "/root/651-p2/results"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        path = f"{OUTDIR}/omp_capture_{time.strftime('%H%M%S')}.json"
        with open(path, "wb") as fh:
            fh.write(raw)
        print(f"captured {len(raw)} bytes -> {path}", flush=True)
        body = json.dumps(
            {"error": {"message": "capture server: not a model", "type": "BadRequestError"}}
        ).encode()
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        body = json.dumps({"data": [{"id": "qwen36-35b-a3b", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 31999
    print(f"capture server on 127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
