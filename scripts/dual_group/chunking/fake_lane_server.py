#!/usr/bin/env python3
"""Smoke vehicle for ``chunking/probe_arms.py`` -- a lane server with no GPU.

Speaks the two endpoints ``lane_run`` / ``lane_snapshot`` use and models the
§13.10 chunked-prefill result rows: a job with ``prefill_chunk > 0`` gets
``prefill_chunks`` / ``prefill_chunk_ms`` that tile its prompt and sum to its
``prefill_ms``; a chunk-0 job gets neither. Exists so the driver is EXECUTED
before it reaches the cards (desk-written-never-executed), and carries its
own can-fail arms so a green smoke also proves the red branches fire:

    --dirty tail      chunked arms diverge from the reference trajectory at a
                      position past the band (COHERENCE must go red)
    --dirty chunks    chunked arms report one chunk too few (STRUCTURE red)
    --dirty sum       chunk timings sum to half of prefill_ms (STRUCTURE red)
    --dirty band      the reference draws disagree from position 0
                      (COHERENCE must answer VOID -- not green, not red)

Usage: fake_lane_server.py PORT [--dirty tail|chunks|sum|band|none]
"""

from __future__ import annotations

import json
import math
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List

DIRTY = "none"

STATE: Dict[str, Any] = {"total": 0, "results": []}

_REF_IDS = list(range(2000, 2064))
#: A divergence at position 18 -- far past any reasonable band.
_TAIL_IDS = _REF_IDS[:18] + [7777] + _REF_IDS[19:]
#: Mode B for --dirty band: differs from _REF_IDS at position 0.
_MODE_B = [9999] + _REF_IDS[1:]


def set_dirty(value: str) -> None:
    global DIRTY
    DIRTY = value


def _result_row(job: Dict[str, Any]) -> Dict[str, Any]:
    n = len(job.get("input_ids") or [])
    chunk = int(job.get("prefill_chunk") or 0)
    spec = bool(job.get("spec"))
    draw = STATE["total"]
    if DIRTY == "band":
        ids = list(_MODE_B if draw % 2 else _REF_IDS)
    else:
        ids = list(_REF_IDS)
    row: Dict[str, Any] = {
        "input_len": n,
        "spec_mode": spec,
        "output_ids": ids,
        "prefill_ms": 100.0,
        "decode_ms_mean": 9.0,
        "decode_steps": len(ids) - 1,
    }
    if chunk > 0:
        if DIRTY == "tail":
            row["output_ids"] = list(_TAIL_IDS)
        n_chunks = math.ceil(n / chunk)
        if DIRTY == "chunks":
            n_chunks = max(1, n_chunks - 1)
        per = round(row["prefill_ms"] / n_chunks, 3)
        chunk_ms = [per] * n_chunks
        if DIRTY == "sum":
            chunk_ms = [x / 2 for x in chunk_ms]
        row["prefill_chunks"] = n_chunks
        row["prefill_chunk_ms"] = chunk_ms
    return row


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _reply(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/get_server_info"):
            self._reply(
                {
                    "internal_states": [
                        {
                            "dual_group_lanes": [
                                {
                                    "results_total": STATE["total"],
                                    "results": STATE["results"][-8:],
                                }
                            ]
                        }
                    ]
                }
            )
        else:
            self._reply({})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        job = (payload.get("server_args") or {}).get("dual_group_lane_prefill")
        if job:
            STATE["results"].append(_result_row(job))
            STATE["total"] += 1
        self._reply({"ok": True})


def serve(port: int) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), _Handler)
    return server


def main(argv: List[str]) -> int:
    port = int(argv[0])
    if len(argv) >= 3 and argv[1] == "--dirty":
        set_dirty(argv[2])
    server = serve(port)
    print(f"fake lane server on {port} (dirty={DIRTY})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
