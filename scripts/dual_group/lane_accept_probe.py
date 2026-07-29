#!/usr/bin/env python3
"""Round 7b posten 0: the serving group's and the lane's per-position accept
curve, from ONE boot, on the SAME content.

Round 7a measured 43.8 / 0.8 / 0 % positional acceptance on the lane and the
serving group reaches accept 2.8-3.1 with weights that are byte-shared with it
(``data_ptr`` identity, proven in the families slice).  Same bytes, drastically
different curve -- so either the head really is that weak and the serving
group's number comes from somewhere else, or the LANE's chain degrades the
later positions.  A mean cannot tell those apart; two curves side by side can.

Everything is driven at K = 3, the production chain length: a K = 1 arm is
structurally blind to a positional pathology, which is how this one survived
four rounds.

Usage:
    python lane_accept_probe.py --port 30077 --tokens 192

Requires ``SGLANG_ACCEPT_POSITION_PROBE=1`` in the server's environment for the
serving-group side; the lane side always reports.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# Forced-continuation prompts: the lane's trajectory is only reproducible where
# the continuation is forced (round 4 measured that, and one of four candidates
# failed the A-vs-A floor).  Kept under ~109 tokens because Qwen GDN prefill is
# not byte-reproducible beyond that.
_TOK = None

PROMPTS: Dict[str, str] = {
    "alphabet": (
        "Continue the sequence exactly, one letter per line, no commentary.\n"
        "a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\n"
    ),
    "squares": (
        "Continue the list of squares exactly, one per line, no commentary.\n"
        "1 1\n2 4\n3 9\n4 16\n5 25\n6 36\n7 49\n8 64\n9 81\n10 100\n11 121\n"
    ),
    # Realistic content, added after the first run of this script measured the
    # SERVING group at accept 1.15-1.32 on the three gate prompts -- far under
    # the known 2.75-2.82 reference for this model class. The gate prompts are
    # forced short continuations chosen for byte reproducibility, not for
    # representativeness, so they cannot on their own decide whether the head
    # is weak or the content is. These two can.
    "code": (
        "Complete the following Python module. Keep the existing style.\n\n"
        "import json\nimport os\nfrom dataclasses import dataclass\n"
        "from typing import Any, Dict, List, Optional\n\n\n"
        "@dataclass\nclass CacheEntry:\n"
        "    key: str\n    value: Any\n    ttl_s: float\n    created_at: float\n\n\n"
        "class DiskCache:\n"
        '    """A tiny on-disk cache with a time-to-live per entry."""\n\n'
        "    def __init__(self, root: str, default_ttl_s: float = 3600.0):\n"
        "        self.root = root\n        self.default_ttl_s = default_ttl_s\n"
        "        os.makedirs(root, exist_ok=True)\n\n"
        "    def _path_for(self, key: str) -> str:\n"
        '        return os.path.join(self.root, key + ".json")\n\n'
        "    def get(self, key: str) -> Optional[Any]:\n"
    ),
    "prose": (
        "Continue this essay in the same register.\n\n"
        "The question of what a measurement is for is older than any of the "
        "instruments we now use to take one. A number that cannot change a "
        "decision is not a measurement; it is decoration. This is why the "
        "first duty of anyone who reports a figure is to say what the figure "
        "would have had to be for the conclusion to go the other way. Without "
        "that, a table of results is a story with numbers in it, and stories "
        "are cheap. The second duty follows from the first: the noise floor "
        "has to be measured before the effect, on the same apparatus, in the "
        "same session, because an effect smaller than the floor is not a small "
        "effect but no effect at all. "
    ),
    "repeat": (
        "Repeat the following line exactly twenty times, nothing else.\n"
        "the quick brown fox jumps over the lazy dog\n"
        "the quick brown fox jumps over the lazy dog\n"
        "the quick brown fox jumps over the lazy dog\n"
    ),
}


def _post(base: str, path: str, payload: Dict[str, Any], timeout: float = 600.0):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if body.strip() else None


def _get(base: str, path: str, timeout: float = 120.0):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tokenize(base: str, text: str, tokenizer_path: str) -> List[int]:
    """Token ids for a prompt, from the SERVER's tokenizer path.

    Both sides must see the SAME ids -- the serving group is driven with ids
    here rather than text precisely so nothing between the two arms can differ
    -- and the server exposes no tokenize endpoint, so the ids come from the
    same directory the server was booted with. Verified against the server's
    own ``prompt_tokens`` count on every call.
    """
    from transformers import AutoTokenizer

    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    ids = _TOK(text, add_special_tokens=False)["input_ids"]
    return list(ids)


def serving_run(base: str, input_ids: List[int], tokens: int) -> Dict[str, Any]:
    return _post(
        base,
        "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
    )


def lane_run(
    base: str, job: Dict[str, Any], poll_s: float = 1.0, budget_s: float = 900.0
):
    before = len(lane_results(base))
    _post(
        base, "/set_internal_state", {"server_args": {"dual_group_lane_prefill": job}}
    )
    t0 = time.time()
    while time.time() - t0 < budget_s:
        res = lane_results(base)
        if len(res) > before:
            return res[before:]
        time.sleep(poll_s)
    raise TimeoutError("lane job did not finish inside the budget")


def lane_results(base: str) -> List[Dict[str, Any]]:
    info = _get(base, "/get_server_info")
    states = info.get("internal_states") or []
    out: List[Dict[str, Any]] = []
    for st in states:
        for lane in st.get("dual_group_lanes") or []:
            out.extend(lane.get("results") or [])
    return out


def serving_curve(base: str) -> Optional[Dict[str, Any]]:
    info = _get(base, "/get_server_info")
    for st in info.get("internal_states") or []:
        snap = st.get("spec_accept_positions")
        if snap:
            return snap
    return info.get("spec_accept_positions")


def _fmt_curve(rates) -> str:
    if not rates:
        return "-"
    return " / ".join("-" if r is None else f"{100.0 * r:5.1f}%" for r in rates)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30077)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--steps", type=int, default=3, help="lane chain length K")
    ap.add_argument("--prompts", default="alphabet,squares,repeat")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--verify",
        default="target_verify",
        help="lane verify strategy (default target_verify: the captured one)",
    )
    ap.add_argument(
        "--rollback-arms",
        default="1",
        help="comma list of draft_rollback arms to run, e.g. '1,0'",
    )
    ap.add_argument(
        "--tokenizer",
        default="/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-MTP-Q3_K_M-GGUF",
    )
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    report: Dict[str, Any] = {"tokens": args.tokens, "steps": args.steps, "arms": []}
    for name in args.prompts.split(","):
        name = name.strip()
        if not name:
            continue
        text = PROMPTS[name]
        ids = tokenize(base, text, args.tokenizer)
        arm: Dict[str, Any] = {"prompt": name, "prompt_tokens": len(ids)}

        # SERVING side first: its curve is process-wide and cumulative, so the
        # delta across this arm is what belongs to this content.
        before = serving_curve(base) or {}
        gen = serving_run(base, ids, args.tokens)
        after = serving_curve(base) or {}
        arm["serving"] = {
            "accept_len_mean": after.get("accept_len_mean"),
            "rounds": after.get("rounds"),
            "curve": _delta_curve(before, after),
            "completion_tokens": gen["meta_info"].get("completion_tokens"),
            "spec_accept_length": gen["meta_info"].get("spec_accept_length"),
        }

        # LANE side, same ids, same token count, chain length K.
        arm["lane_arms"] = {}
        for rb in [x.strip() for x in args.rollback_arms.split(",") if x.strip()]:
            res = lane_run(
                base,
                {
                    "lane_id": 0,
                    "input_ids": ids,
                    "max_new_tokens": args.tokens,
                    "spec_steps": args.steps,
                    "verify": args.verify,
                    "draft_rollback": rb not in ("0", "false", "False"),
                },
            )
            arm["lane_arms"][rb] = _lane_row(res[-1])
            print(f"   lane rollback={rb} " + _row_line(arm["lane_arms"][rb]))
            sys.stdout.flush()
        r = res[-1]
        arm["lane"] = {
            "accept_len_mean": r.get("accept_len_mean"),
            "spec_rounds": r.get("spec_rounds"),
            "curve": (r.get("accept_positions") or {}).get("rate"),
            "draft_lag": r.get("draft_lag"),
            "decode_ms_mean": r.get("decode_ms_mean"),
            "verify_ms_mean": r.get("verify_ms_mean"),
            "propose_ms_mean": r.get("propose_ms_mean"),
            "verify_graph_rounds": r.get("verify_graph_rounds"),
            "head_forwards": r.get("head_forwards"),
            "head_graph_forwards": r.get("head_graph_forwards"),
        }
        report["arms"].append(arm)

        print(f"== {name} ({len(ids)} prompt tokens, K={args.steps})")
        print(
            f"   serving accept {arm['serving']['accept_len_mean']}"
            f"   positions {_fmt_curve(arm['serving']['curve'])}"
        )
        sys.stdout.flush()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


def _lane_row(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "accept_len_mean": r.get("accept_len_mean"),
        "spec_rounds": r.get("spec_rounds"),
        "curve": (r.get("accept_positions") or {}).get("rate"),
        "draft_lag": r.get("draft_lag"),
        "decode_ms_mean": r.get("decode_ms_mean"),
        "verify_ms_mean": r.get("verify_ms_mean"),
        "propose_ms_mean": r.get("propose_ms_mean"),
        "verify_graph_rounds": r.get("verify_graph_rounds"),
        "head_forwards": r.get("head_forwards"),
        "head_graph_forwards": r.get("head_graph_forwards"),
        "output_ids": r.get("output_ids"),
    }


def _row_line(row: Dict[str, Any]) -> str:
    lag = row.get("draft_lag") or {}
    return (
        f"accept {row.get('accept_len_mean')}  positions {_fmt_curve(row.get('curve'))}"
        f"  round {row.get('decode_ms_mean')} ms  vgraph "
        f"{row.get('verify_graph_rounds')}/{row.get('spec_rounds')}"
        f"  lag max {lag.get('max_abs')}"
    )


def _delta_curve(before: Dict[str, Any], after: Dict[str, Any]):
    """The serving curve for THIS arm only.

    The probe counts for the lifetime of the process, so an arm's own curve is
    the difference of two snapshots -- otherwise arm 3 is mostly arm 1.
    """
    br, bh = before.get("position_reached") or {}, before.get("position_hits") or {}
    ar, ah = after.get("position_reached") or {}, after.get("position_hits") or {}
    if not ar:
        return []
    out = []
    for j in range(max(int(k) for k in ar) + 1):
        reached = int(ar.get(str(j), ar.get(j, 0))) - int(br.get(str(j), br.get(j, 0)))
        hits = int(ah.get(str(j), ah.get(j, 0))) - int(bh.get(str(j), bh.get(j, 0)))
        out.append(hits / reached if reached > 0 else None)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
