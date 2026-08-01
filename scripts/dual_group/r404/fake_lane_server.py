#!/usr/bin/env python3
"""Smoke vehicle for ``bracket_arms.py`` -- a lane server with no GPU.

Speaks the two endpoints ``lane_run`` / ``lane_snapshot`` use
(``POST /set_internal_state``, ``GET /get_server_info``) and produces result
rows carrying every field the extended harness reads: the rung histogram, the
verify-graph counter, and the #404 per-round pool checksums.

It exists so the harness is EXECUTED before it reaches the cards -- the
desk-written-never-executed rule -- and it carries its own can-fail arms, so a
green smoke also proves the red branches fire:

    --dirty tv        the ``tv_max_accept_0`` arm returns a divergent
                      trajectory (the trajectory gate must go red)
    --dirty eager     the K=3 arms report ``verify_graph_rounds 0`` (the
                      CAPTURE expectation must go red -- this is exactly the
                      gap the previous window left, replayed as a test)
    --dirty mixed     the mixed-rung arms report a single-rung histogram (the
                      MIXED expectation must go red)
    --dirty checksum  the speculative side's pool checksums break the
                      append-only invariant at a named round (the checksum
                      verdict must go red AND name that round)

Usage:  fake_lane_server.py PORT [--dirty tv|eager|mixed|checksum|none]
"""

from __future__ import annotations

import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List

DIRTY = "none"
if "--dirty" in sys.argv:
    DIRTY = sys.argv[sys.argv.index("--dirty") + 1]

STATE: Dict[str, Any] = {"total": 0, "results": []}

# A correct continuation in this tokenizer would need the tokenizer; the smoke
# only needs the SHAPES to travel, so the ids are synthetic. What matters is
# that ref and arm ids agree (or, in the dirty arm, do not).
_REF_IDS = list(range(1000, 1064))
_DIRTY_IDS = _REF_IDS[:18] + [7777] + _REF_IDS[19:]

#: The verify shapes a boot with ``--dual-group-lane-spec-rungs 0,1,3`` records.
_CAPTURED = {2, 4}

#: Where ``--dirty checksum`` plants its break.
_PLANT_ROUND = 5


def _digest(*parts) -> str:
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(str(part).encode())
    return h.hexdigest()


def _checksums(ids: List[int], accepts: List[int], tag, prompt_len: int):
    """One record per committed round, digests derived from the committed ids.

    The property the real probe has and this mock must reproduce: two jobs
    standing at the same ``committed_len`` with the same committed content hash
    the same, however they got there. ``map`` deliberately does NOT have it --
    it carries the tag, so two jobs never agree on it, exactly as two jobs
    never draw the same physical slots.
    """
    records = []
    committed = prompt_len
    prev = 0
    prev_map = prev_kv = None
    for round_index, kept in enumerate([0] + accepts):
        committed += kept
        prefix = ids[: committed - prompt_len]
        rec = {
            "tag": tag,
            "lane": 0,
            "rank": 0,
            "round": round_index,
            "path": "prefill" if round_index == 0 else "spec",
            "n_accept": None if round_index == 0 else kept - 1,
            "rung": 0 if round_index == 0 else max(0, kept - 1),
            "committed_len": committed,
            "prev_committed_len": prev,
            "kv_len": committed,
            "emitted": len(prefix),
            "region": "committed_prefix",
            "map": _digest("map", tag, committed),
            "map_stable": prev_map,
            "kv": _digest("kv", prefix),
            "kv_stable": prev_kv,
            "conv": _digest("conv", prefix),
            "ssm": _digest("ssm", prefix),
            "probe_ms": 1.0,
        }
        if (
            DIRTY == "checksum"
            and tag
            and tag.endswith(":spec")
            and round_index == _PLANT_ROUND
        ):
            # A row the lane had already committed moved: the stable digest no
            # longer matches the previous round's full one, and the full digest
            # walks away from the reference from here on.
            rec["kv_stable"] = _digest("planted", round_index)
            rec["kv"] = _digest("planted-kv", round_index)
        prev = committed
        prev_map, prev_kv = rec["map"], rec["kv"]
        records.append(rec)
    return records


def _accepts(rung, n_tokens: int, cap) -> List[int]:
    """Tokens committed per round, and therefore the rung schedule's effect."""
    accepts: List[int] = []
    emitted = 0
    index = 0
    while emitted < n_tokens:
        k = _rung_of(rung, index)
        if k == 0:
            kept = 1
        elif cap == 0:
            kept = 1
        else:
            kept = min(k + 1, 2)
        accepts.append(kept)
        emitted += kept
        index += 1
    return accepts


def _rung_of(rung, index: int) -> int:
    """The rung of round ``index`` for a scalar pin, a schedule, or no pin.

    ``--dirty mixed`` collapses both mixed shapes to a single non-zero rung --
    a job that never takes the K=0 branch, which is exactly the shape the
    mixed-rung expectation exists to refuse.
    """
    if isinstance(rung, (list, tuple)):
        if DIRTY == "mixed":
            return max(int(x) for x in rung)
        return int(rung[index % len(rung)])
    if rung is None:
        return 1 if DIRTY == "mixed" else [3, 1, 0][index % 3]
    return int(rung)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        out = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path.startswith("/get_server_info"):
            self._send(
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
            return
        if self.path.startswith("/health"):
            self._send({"ok": True})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        job = (body.get("server_args") or {}).get("dual_group_lane_prefill") or {}
        spec = bool(job.get("spec"))
        cap = job.get("tv_max_accept")
        rung = job.get("spec_steps")
        tag = job.get("probe_tag")
        prompt_len = len(job.get("input_ids") or [])
        ids = list(_REF_IDS)
        if spec and DIRTY == "tv" and cap == 0:
            ids = list(_DIRTY_IDS)

        if not spec:
            accepts = [1] * len(ids)
            row = {
                "input_len": prompt_len,
                "spec_mode": False,
                "verify_mode": None,
                "spec_rounds": None,
                "accept_len_mean": None,
                "output_ids": ids,
                "prefill_ms": 12.0,
                "decode_ms_mean": 16.0,
                "decode_steps": len(ids),
                "margins": [2.0] * len(ids),
            }
        else:
            accepts = _accepts(rung, len(ids), cap)
            ids = ids[: sum(accepts)]
            rungs: Dict[int, int] = {}
            spec_rounds = 0
            graph_rounds = 0
            for i in range(len(accepts)):
                k = _rung_of(rung, i)
                rungs[k] = rungs.get(k, 0) + 1
                if k > 0:
                    spec_rounds += 1
                    if DIRTY == "eager" and k >= 3:
                        continue
                    if (k + 1) in _CAPTURED:
                        graph_rounds += 1
            row = {
                "input_len": prompt_len,
                "spec_mode": True,
                "verify_mode": "target_verify",
                "spec_rounds": spec_rounds or None,
                "accept_len_mean": (
                    round(sum(accepts) / len(accepts), 3) if accepts else None
                ),
                "output_ids": ids,
                "prefill_ms": 12.0,
                "decode_ms_mean": 24.0,
                "decode_steps": len(accepts),
                "verify_ms_mean": 21.0,
                "propose_ms_mean": 3.0,
                "verify_graph_rounds": graph_rounds,
                "head_forwards": spec_rounds,
                "head_graph_forwards": spec_rounds,
                "rungs": dict(sorted(rungs.items())),
                "rung_mean": round(
                    sum(_rung_of(rung, i) for i in range(len(accepts)))
                    / max(1, len(accepts)),
                    3,
                ),
                "margins": [2.0] * len(ids),
            }
        records = _checksums(ids, accepts, tag, prompt_len)
        row["pool_checksums"] = records
        row["pool_checksum_rounds"] = len(records)
        STATE["results"].append(row)
        STATE["total"] += 1
        self._send({})


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
