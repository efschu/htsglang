#!/usr/bin/env python3
"""#307 arm B: drive a fitted-ceiling server and record the admission float.

Runs ON THE HOST (the server binds 127.0.0.1). Two phases against one boot:

  1. pressure -- more concurrent requests than the start value, long enough
     prompts to push pool occupancy over --admission-throttle-high. The float
     must throttle, and it must throttle BEFORE anything is retracted.
  2. release -- one light request keeps the controller sampling at low usage
     (the release path lives in observe(), which only runs while a decode
     batch exists, so an idle window proves nothing). The float must climb
     back ABOVE its start value and stop at the fitted ceiling.

Prints one JSON object; every number in it is read from /get_server_info.
"""

import json
import sys
import threading
import time
import urllib.request

from s307_probe_sizing import (
    admitted_from_info,
    context_tokens_from_info,
    default_concurrency,
    default_prompt_repeat,
    pool_from_info,
    token_pool_from_info,
)

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30047
BASE = f"http://127.0.0.1:{PORT}"
T0 = time.time()


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as resp:
        return json.loads(resp.read())


def state():
    info = get("/get_server_info")
    st = (info.get("internal_states") or [{}])[0]
    return st.get("admission_limiter") or info.get("admission_limiter") or {}


def _live_info():
    """This server's /get_server_info, or None when the query fails (e.g. it
    is not up yet). Fetched ONCE: every sizing input below has to describe the
    same server state, and three separate queries could straddle a change."""
    try:
        return get("/get_server_info")
    except Exception:
        return None


_INFO = _live_info()

# BOTH pressure dimensions are sized from what the server reports, never from
# a predicted value -- see s307_probe_sizing for the two card runs that each
# scored a quiet throttle_count 0 by getting one of them wrong.
#
#   CONCURRENCY    only has to keep the ADMITTED slots full; extra clients
#                  queue and hold no tokens.
#   PROMPT_REPEAT  is the one that actually moves occupancy, because the
#                  controller samples HELD TOKENS over max_total_num_tokens.
CONCURRENCY = (
    int(sys.argv[2]) if len(sys.argv) > 2 else default_concurrency(pool_from_info(_INFO))
)
if len(sys.argv) > 3:
    PROMPT_REPEAT, PROMPT_REPEAT_NOTE = int(sys.argv[3]), "pinned on the command line"
else:
    PROMPT_REPEAT, PROMPT_REPEAT_NOTE = default_prompt_repeat(
        token_pool_from_info(_INFO),
        admitted_from_info(_INFO),
        context_tokens_from_info(_INFO),
    )
NEW_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 160


samples = []
stop = threading.Event()


def sampler():
    while not stop.is_set():
        try:
            samples.append({"t": round(time.time() - T0, 2), "lim": state()})
        except Exception as exc:  # a sample is evidence, not a control path
            samples.append({"t": round(time.time() - T0, 2), "err": str(exc)[:80]})
        stop.wait(0.5)


prompt = (
    "The quick brown fox jumps over the lazy dog near the riverbank. " * PROMPT_REPEAT
)
failed = 0
lock = threading.Lock()


def worker():
    global failed
    try:
        post(
            "/generate",
            {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": NEW_TOKENS,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
        )
    except Exception:
        with lock:
            failed += 1


threading.Thread(target=sampler, daemon=True).start()
start_state = state()

threads = [threading.Thread(target=worker) for _ in range(CONCURRENCY)]
for t in threads:
    t.start()
for t in threads:
    t.join()
pressure_end = round(time.time() - T0, 2)

# Release phase: keep a light batch alive so observe() keeps sampling.
deadline = time.time() + 240
while time.time() < deadline:
    try:
        post(
            "/generate",
            {
                "text": "Count slowly to twenty.",
                "sampling_params": {
                    "max_new_tokens": 64,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
    except Exception:
        break
    cur = state()
    if cur.get("current") and cur.get("ceiling") and cur["current"] >= cur["ceiling"]:
        break

stop.set()
time.sleep(1.0)
end_state = state()

limits = [s["lim"].get("current") for s in samples if s.get("lim", {}).get("current")]
print(
    json.dumps(
        {
            "start_state": start_state,
            "end_state": end_state,
            # How the pressure phase was sized, and why. Without this a run
            # that could never reach the throttle mark looks exactly like a
            # run whose mechanism is broken -- both report throttle_count 0.
            "sizing": {
                "concurrency": CONCURRENCY,
                "prompt_repeat": PROMPT_REPEAT,
                "prompt_repeat_note": PROMPT_REPEAT_NOTE,
                "new_tokens": NEW_TOKENS,
                "token_pool": token_pool_from_info(_INFO),
                "admitted": admitted_from_info(_INFO),
                "mamba_pool": pool_from_info(_INFO),
            },
            "pressure_end_s": pressure_end,
            "peak_limit": max(limits) if limits else None,
            "min_limit": min(limits) if limits else None,
            # Retraction is not a limiter counter -- it is counted from the
            # server log's "Retract requests" lines by the verdict script.
            "throttle_count": end_state.get("throttle_count"),
            "release_count": end_state.get("release_count"),
            "failed": failed,
            "samples": samples[-400:],
        },
        indent=1,
    )
)
