#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631: does a tp_to_pp cutover LOSE tokens, and on which side?

WHY THIS EXISTS, and what it separates that the carry probe cannot.
The carry probe checks the DECODED TEXT of a counting task and reports a
break. A break has two structurally different causes and the same face:

  * the id path lost the token -- it was never appended to ``output_ids``,
    so no consumer could ever have seen it;
  * the send/detokenize path lost it -- ``output_ids`` HAS it and the
    text does not, i.e. the token was generated, committed, and then
    dropped by a cursor.

Those need opposite fixes, and the carry probe cannot tell them apart
because it only ever reads the text. This one reads BOTH and reports the
comparison, so one run decides it.

HOW THE COMPARISON IS MADE. The counting task decodes to a stream of
single-digit tokens separated by a space token, so the ids can be turned
back into a digit string WITHOUT the tokenizer: only the digit ids and
the space id are needed, and both are read off the stream itself rather
than assumed (``_digit_map`` derives them from the leading "2 3 4 5 6..."
run that every reply starts with). If the id-derived number sequence is
unbroken while the text's is not, the loss is downstream of the ids.

WHY THE CUTOVER LANDS WHERE IT DOES. Measured on this rig: with three
concurrent counters and the flip armed ~4 s in, the cutover falls at
``seen`` ~388, which is around number 115 of 120 -- three numbers before
the end, which is exactly where the observed breaks are. The delay is a
knob so the landing point can be moved deliberately.

Stdlib only; runs against a live server.
"""

import argparse
import json
import re
import sys
import threading
import time
import urllib.request

PORT = 30030
COUNT_TO = 120
PROMPT = (
    "Write the integers from 1 to %d in order, separated by single "
    "spaces, and write nothing else at all.\nAnswer: 1" % COUNT_TO
)


def post(path, payload, timeout=180):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def flip(direction):
    try:
        return post("/phase_flip", {"direction": direction}, timeout=60)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def run_one(idx, out):
    t0 = time.time()
    try:
        r = post(
            "/generate",
            {
                "text": PROMPT,
                "sampling_params": {"max_new_tokens": 600, "temperature": 0},
                "return_text_in_logprobs": False,
            },
        )
        out[idx] = {
            "text": r.get("text", ""),
            "ids": r.get("output_ids", []),
            "meta": r.get("meta_info", {}),
            "s": round(time.time() - t0, 1),
        }
    except Exception as exc:  # noqa: BLE001
        out[idx] = {"error": str(exc), "s": round(time.time() - t0, 1)}


def numbers(text):
    """The numbers of the TASK, and only those.

    ``max_new_tokens`` is deliberately well above what the count needs, so
    every reply carries whatever the model does after finishing -- restart
    the count, answer a different question, ramble. Judging that as a
    break reports the model's habits as a flip defect, which this probe
    did on its first run. Truncate at the first number that reaches the
    target, exactly as the carry probe's checker does.
    """
    nums = [int(x) for x in re.findall(r"\d+", text or "")]
    for i, v in enumerate(nums):
        if v >= COUNT_TO:
            return nums[: i + 1]
    return nums


def first_break(nums):
    """Index of the first number that is not previous+1, or None."""
    if not nums:
        return 0
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1] + 1:
            return i
    return None


def digit_map(ids):
    """Derive {id: character} for the digits and the space, from the reply
    itself. The reply opens with ' 2 3 4 5 6 7 8 9 10' -- a run whose
    token sequence pins every single-digit id and the separator without
    any tokenizer knowledge.

    Returns (mapping, space_id) or (None, None) when the opening run does
    not have the expected shape, which is itself worth reporting rather
    than papering over.
    """
    if len(ids) < 18:
        return None, None
    # ' 2' ' 3' ... each as (space, digit); the separator is whatever id
    # occupies every even position of that opening run.
    space = ids[0]
    mapping = {}
    val = 2
    i = 0
    while i + 1 < len(ids) and val <= 9:
        if ids[i] != space:
            return None, None
        mapping[ids[i + 1]] = str(val)
        val += 1
        i += 2
    if len(mapping) != 8:
        return None, None
    # ' 10' pins '1' and '0'
    if i + 2 < len(ids) and ids[i] == space:
        mapping.setdefault(ids[i + 1], "1")
        mapping.setdefault(ids[i + 2], "0")
    return mapping, space


def ids_to_text(ids, mapping, space):
    """Render the id stream as a digit string, marking anything that is
    neither a digit nor the separator, so a foreign token is visible
    rather than silently skipped."""
    parts = []
    for t in ids:
        if t == space:
            parts.append(" ")
        elif t in mapping:
            parts.append(mapping[t])
        else:
            parts.append(f"<{t}>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="tp_to_pp")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--delay", type=float, default=4.0)
    ap.add_argument("--no-flip", action="store_true")
    args = ap.parse_args()

    out = {}
    threads = [
        threading.Thread(target=run_one, args=(i, out))
        for i in range(args.concurrency)
    ]
    for t in threads:
        t.start()
    time.sleep(args.delay)
    if args.no_flip:
        print("CONTROL: no flip", flush=True)
    else:
        print(f"arming {args.direction}: "
              f"{json.dumps(flip(args.direction))[:160]}", flush=True)
    for t in threads:
        t.join()

    verdict_bad = 0
    for i in sorted(out):
        r = out[i]
        if "error" in r:
            print(f"req{i}: ERROR {r['error']}")
            verdict_bad += 1
            continue
        t_nums = numbers(r["text"])
        t_break = first_break(t_nums)
        mapping, space = digit_map(r["ids"])
        if mapping is None:
            print(f"req{i}: id stream does not open with the counting run; "
                  f"ids[:20]={r['ids'][:20]}")
            continue
        rendered = ids_to_text(r["ids"], mapping, space)
        i_nums = numbers(rendered)
        i_break = first_break(i_nums)
        same = t_nums == i_nums
        print(
            f"req{i}: text_break={t_break} ids_break={i_break} "
            f"streams_equal={same} tok={r['meta'].get('completion_tokens')} "
            f"({r['s']}s)"
        )
        if t_break is not None or i_break is not None or not same:
            verdict_bad += 1
            pos = t_break if t_break is not None else (i_break or 0)
            lo = max(0, pos - 6)
            print(f"   text nums {t_nums[lo:pos + 4]}")
            print(f"   ids  nums {i_nums[lo:pos + 4]}")
            # The raw id window around the break, because WHICH token is
            # missing is the whole question.
            hit = 0
            for m in re.finditer(r"\d+", rendered):
                hit += 1
                if hit >= pos:
                    a = max(0, m.start() - 24)
                    print(f"   ids  raw ...{rendered[a:m.end() + 24]!r}...")
                    break
            hit = 0
            for m in re.finditer(r"\d+", r["text"] or ""):
                hit += 1
                if hit >= pos:
                    a = max(0, m.start() - 24)
                    print(f"   text raw ...{r['text'][a:m.end() + 24]!r}...")
                    break

    print("\nVERDICT:", "CLEAN" if verdict_bad == 0 else f"{verdict_bad} BAD")
    return 0 if verdict_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
