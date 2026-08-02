#!/usr/bin/env python3
"""Drive the LIVE session handover byte gate (#261 second half).

Both servers run SIMULTANEOUSLY -- that is the point. The gate: a session
seeded on the source (A), handed over live (export -> manifest-scoped
umsharder -> verify_import -> commit) and continued on the destination (B)
must produce a continuation byte-identical to the reference continuation
that never migrated. A-vs-A floor first: the reference continuation, run
twice on the same server, must be identical before any cross-server claim
is made. Keep continuations short: the known GDN-prefill non-determinism
onset makes long-window byte claims dishonest.

Subcommands (a shell driver chains them; every step is separately
re-runnable and every artifact is a plain json file):

  seed      -- send a determined prompt as raw token ids on --port, record
               prompt+output ids into the state file.
  continue  -- replay the recorded session ids plus a fixed follow-up on
               --port, record the continuation ids under --label. With
               --expect-cached, fail unless meta_info.cached_tokens > 0
               (proves the prefix came from the store, not a re-prefill).
  export    -- POST /session_handover action=export on the SOURCE with the
               recorded session ids; write the manifest json next to the
               state file. The prefix is now PARKED on the source.
  parked    -- prove the park: the same session ids on the source must be
               REFUSED while the handover is active.
  liveness  -- prove "live": an UNRELATED prompt on --port must still be
               served (run against both servers while the handover is
               active).
  verify    -- POST action=verify_import on the DESTINATION with the
               manifest (request-less import gate).
  commit    -- POST action=commit on the source.
  abort     -- POST action=abort on the source (rollback arm).
  compare   -- byte-compare two recorded continuations (--label vs
               --other-label); exit nonzero on any divergence, printing the
               first differing position.

Token ids, not text, everywhere: the handover is judged on a TOKEN prefix,
and re-tokenizing generated text does not reliably reproduce it.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

FOLLOW_UP = (
    "\n\nQuestion: name the machine that carries two smaller cards, and "
    "nothing else.\nAnswer:"
)

SEED_PROMPT = (
    "Field report, rig inventory.\n\n"
    "The workshop runs three machines. Machine ONE is called Kestrel and "
    "carries a single large card. Machine TWO is called Marlin and carries "
    "two smaller cards. Machine THREE is called Osprey and is the spare.\n\n"
    "Write a short continuation of this report describing what each "
    "machine is used for.\n\nContinuation:"
)

UNRELATED_PROMPT = "List three prime numbers below twenty.\nAnswer:"

# Deterministic filler for --seed-prompt-tokens. The destination only
# prefetches a prefix from storage once it is at least
# UnifiedRadixCache.prefetch_threshold (256) tokens long, and on a hybrid
# GDN model the handover length is further clamped to the deepest GDN
# checkpoint. A short seed therefore can never produce a store hit on the
# destination, which would make --expect-cached unpassable for reasons
# that have nothing to do with the handover. Padding is content-stable
# (numbered lines, no randomness) so the prompt stays a determined answer.
FILLER_LINE = (
    "Shelf {i:03d} holds spare cable {i:03d}, rated for bay {j:02d}, "
    "checked on day {k:02d}.\n"
)


def build_seed_prompt(tok, target_tokens: int) -> str:
    """SEED_PROMPT padded with deterministic inventory lines until the
    tokenized prompt reaches ``target_tokens`` (0 = no padding)."""
    if not target_tokens:
        return SEED_PROMPT
    head, tail = SEED_PROMPT.split("\n\nWrite a short continuation", 1)
    tail = "\n\nWrite a short continuation" + tail
    filler = ""
    i = 0
    while True:
        candidate = head + "\n\n" + filler + tail
        if len(tok.encode(candidate, add_special_tokens=False)) >= target_tokens:
            return candidate
        i += 1
        filler += FILLER_LINE.format(i=i, j=i % 12, k=i % 28)



def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def generate(base: str, ids: list, max_new: int, timeout: float = 600.0) -> dict:
    return post(
        f"{base}/generate",
        {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        },
        timeout=timeout,
    )


def output_ids(meta: dict) -> list:
    lp = meta.get("output_token_logprobs") or []
    return [int(e[1]) for e in lp]


def load_state(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_state(path: str, state: dict) -> None:
    with open(path, "w") as f:
        json.dump(state, f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=[
            "seed",
            "continue",
            "export",
            "parked",
            "liveness",
            "verify",
            "commit",
            "abort",
            "compare",
        ],
    )
    ap.add_argument("--port", type=int)
    ap.add_argument("--tokenizer")
    ap.add_argument("--state", required=True)
    ap.add_argument("--label", default="ref")
    ap.add_argument("--other-label")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--seed-new-tokens", type=int, default=64)
    ap.add_argument(
        "--seed-prompt-tokens",
        type=int,
        default=0,
        help="pad the seed prompt with deterministic filler until it is at "
        "least N tokens. Needed because the destination only prefetches a "
        "prefix of at least 256 tokens from storage",
    )
    ap.add_argument("--expect-cached", action="store_true")
    ap.add_argument(
        "--export-len",
        type=int,
        help="hand over only the first N session tokens (default: all)",
    )
    ap.add_argument(
        "--no-gdn-align",
        action="store_true",
        help="do not re-export at the GDN checkpoint boundary the server "
        "names; report the refusal instead (negative arms want this)",
    )
    ap.add_argument("--deadline-s", type=float, default=60.0)
    args = ap.parse_args()

    state = load_state(args.state)
    base = f"http://127.0.0.1:{args.port}" if args.port else None

    if args.cmd == "seed":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        prompt = build_seed_prompt(tok, args.seed_prompt_tokens)
        ids = tok.encode(prompt, add_special_tokens=False)
        t0 = time.time()
        ret = generate(base, ids, args.seed_new_tokens)
        out = output_ids(ret["meta_info"])
        state["session_ids"] = list(ids) + out
        state["follow_up_ids"] = tok.encode(FOLLOW_UP, add_special_tokens=False)
        save_state(args.state, state)
        print(
            f"seed: {len(ids)} prompt + {len(out)} output tokens in "
            f"{time.time() - t0:.1f}s"
        )
        return 0

    if args.cmd == "continue":
        ids = state["session_ids"] + state["follow_up_ids"]
        ret = generate(base, ids, args.max_new_tokens)
        meta = ret["meta_info"]
        out = output_ids(meta)
        cached = int(meta.get("cached_tokens") or 0)
        state.setdefault("continuations", {})[args.label] = out
        save_state(args.state, state)
        print(
            f"continue[{args.label}]: {len(out)} tokens, cached_tokens={cached}"
        )
        if args.expect_cached and cached <= 0:
            print(
                "VERDICT: no prefix was imported (cached_tokens == 0) -- the "
                "continuation above is a full re-prefill, not a handover"
            )
            return 1
        return 0

    if args.cmd == "export":
        ids = state["session_ids"]
        if args.export_len:
            ids = ids[: args.export_len]

        def _try_export(tok_ids):
            """Returns (response, None) or (None, refusal-body)."""
            try:
                return (
                    post(
                        f"{base}/session_handover",
                        {
                            "action": "export",
                            "token_ids": tok_ids,
                            "deadline_s": args.deadline_s,
                        },
                    ),
                    None,
                )
            except urllib.error.HTTPError as e:
                # A refused export answers HTTP 400 with the NAMED reason in
                # the body (the GDN gate among them). Surfacing that message
                # is the whole point of the negative arms, so it must not be
                # lost behind a traceback.
                return None, e.read().decode()

        ret, refusal = _try_export(ids)
        if refusal is not None:
            # On a hybrid-GDN model the resumable prefix ends at the deepest
            # GDN checkpoint, not at the last generated token, so an export of
            # the whole session is refused for almost every session length.
            # That boundary is a property of the running server, so the probe
            # asks once, then hands over exactly at the boundary the server
            # named. The remaining few tokens are re-prefilled on top of the
            # imported recurrent state -- which is precisely the property the
            # byte gate is there to check.
            m = re.search(r"only (\d+) of (\d+) tokens are resident", refusal)
            if not m or args.no_gdn_align:
                print(f"export FAILED: HTTP 400: {refusal[:600]}")
                return 1
            resident = int(m.group(1))
            print(
                f"export: session is {m.group(2)} tokens but only {resident} "
                "are resident (GDN checkpoint boundary); re-exporting at that "
                "boundary"
            )
            ids = ids[:resident]
            ret, refusal = _try_export(ids)
            if refusal is not None:
                print(f"export FAILED: HTTP 400: {refusal[:600]}")
                return 1
        manifest = ret.get("manifest")
        if not ret.get("success") or not manifest:
            print(f"export FAILED: {ret.get('message')}")
            return 1
        state["manifest"] = manifest
        save_state(args.state, state)
        print(
            f"export: handover {manifest['handover_id']}, "
            f"{len(manifest['kv_keys'])} KV pages, mamba={manifest['mamba_key']}"
        )
        return 0

    if args.cmd == "parked":
        ids = state["session_ids"] + state["follow_up_ids"]
        try:
            ret = generate(base, ids, 4, timeout=60.0)
        except urllib.error.HTTPError as e:
            print(f"parked: refused with HTTP {e.code} (ok)")
            return 0
        meta = ret.get("meta_info", {})
        finish = meta.get("finish_reason") or {}
        message = str(finish.get("message", "")) + str(ret.get("text", ""))
        if "handover" in message:
            print("parked: refused with the handover message (ok)")
            return 0
        print(f"parked FAILED: the parked prefix was served: {ret}")
        return 1

    if args.cmd == "liveness":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        ids = tok.encode(UNRELATED_PROMPT, add_special_tokens=False)
        t0 = time.time()
        ret = generate(base, ids, 16, timeout=120.0)
        n = len(output_ids(ret["meta_info"]))
        print(f"liveness: {n} tokens from port {args.port} in {time.time() - t0:.1f}s")
        return 0 if n > 0 else 1

    if args.cmd in ("verify", "commit", "abort"):
        payload = {"action": {"verify": "verify_import"}.get(args.cmd, args.cmd)}
        if args.cmd == "verify":
            payload["manifest_json"] = json.dumps(state["manifest"])
        else:
            payload["handover_id"] = state["manifest"]["handover_id"]
        try:
            ret = post(f"{base}/session_handover", payload)
        except urllib.error.HTTPError as e:
            print(f"{args.cmd} FAILED: HTTP {e.code}: {e.read().decode()[:400]}")
            return 1
        print(f"{args.cmd}: {ret.get('message')}")
        return 0 if ret.get("success") else 1

    if args.cmd == "compare":
        a = state["continuations"][args.label]
        b = state["continuations"][args.other_label]
        if a == b:
            print(
                f"BYTE GATE PASSED: continuations {args.label!r} and "
                f"{args.other_label!r} are identical ({len(a)} tokens)"
            )
            return 0
        n = min(len(a), len(b))
        pos = next((i for i in range(n) if a[i] != b[i]), n)
        print(
            f"BYTE GATE FAILED: {args.label!r} vs {args.other_label!r} "
            f"diverge at token {pos} "
            f"({a[pos] if pos < len(a) else '<end>'} vs "
            f"{b[pos] if pos < len(b) else '<end>'}); lengths {len(a)}/{len(b)}"
        )
        return 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
