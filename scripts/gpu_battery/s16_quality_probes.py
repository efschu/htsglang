#!/usr/bin/env python3
"""s16 -- graded QUALITY probes against a live server, task #360.

Why this exists next to the #151 benchmark suite. ``planner/bench_suite.py``
is the club-3090 verify-full / verify-stress catalog: it answers "does this
deployment WORK" with a tri-state per scenario, and for six of its sixteen
tests any HTTP 200 is a pass. That is the right shape for a deployment gate
and the wrong shape for a FORMAT gate. Deciding whether an INT8-W8A8
checkpoint may replace an FP8 one as the standard model needs a graded score
-- a number that can move -- on axes where weight/activation quantization is
known to hurt: arithmetic, factual recall, literal instruction compliance,
executable code, and long-context retrieval.

So this file runs the suite AND a graded probe set, from one entry point, and
scores them with the same discipline the rig applies to throughput:

* EVERY PROBE IS VERIFIABLE WITHOUT A JUDGE. Exact numeric match, keyword
  containment, a mechanically checkable format constraint, unit tests that
  execute the emitted function, or the needle secret. No model-as-judge, no
  human read-out, no partial credit that a rerun could re-litigate.
* THE PROBES ARE DETERMINISTIC BY REQUEST. ``temperature: 0`` everywhere and
  a fixed RNG seed for the needle secrets, so two arms see byte-identical
  requests. What is left of the variance is the ENGINE's, which is exactly
  the quantity the A-vs-A arm measures.
* THE NOISE FLOOR COMES FIRST AND IT IS TWO NUMBERS. ``--mode compare``
  reports the score delta between the two same-format arms (the band a
  cross-format delta has to clear) AND the fraction of probes whose answers
  were byte-identical between them. The second one frames the first: on this
  rig GDN prefill is not reproducible past ~109 tokens (upstream, all
  backends), so identical scores with non-identical text is the expected
  shape, and a cross-arm text difference is only evidence when the identity
  rate of the A-vs-A pair was high to begin with.
* ACCEPT LENGTH IS READ WHERE IT EXISTS. ``/v1/chat/completions`` carries no
  spec block at all; the number lives in ``meta_info.spec_accept_length`` on
  the native ``/generate`` answer. The Prometheus ``spec_ema_accept_len``
  gauge that the #151 suite reads is an EMA over the server's whole life, not
  this request's accept length -- it is recorded here for provenance and is
  not the reported figure.

Modes::

    --mode run     one arm: bench suite + graded probes -> <out>/…_<arm>.json*
    --mode compare all arms: noise band, decision table -> stdout + files
    --mode rescore re-grade the answers on disk with the current scorers
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

REPO_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python",
)
if REPO_PYTHON not in sys.path:
    sys.path.insert(0, REPO_PYTHON)

from sglang.srt.planner import bench_suite as bs  # noqa: E402

#: Seed for every stochastic choice a probe or the suite makes, so that two
#: arms are asked literally the same questions.
SEED = 360360


# ===========================================================================
# Graded probe catalog.
#
# Each probe: (id, category, body-builder, scorer). The scorer returns
# (score in [0,1], note) and NEVER raises -- a broken answer scores 0 with the
# reason in the note, because a crash here would look like a missing arm.
# ===========================================================================
def _chat(content: str, max_tokens: int) -> dict:
    return {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


_ANSWER_RE = re.compile(r"ANSWER:\s*(-?[\d.,]+)", re.IGNORECASE)


def _num_scorer(expected: float) -> Callable[[str], Tuple[float, str]]:
    def score(answer: str) -> Tuple[float, str]:
        if not answer:
            return 0.0, "empty answer"
        hits = _ANSWER_RE.findall(answer)
        if not hits:
            # Fall back to the last number in the text: a right answer in the
            # wrong wrapper is a formatting miss, not an arithmetic miss, and
            # the instruction-following category is where format is graded.
            nums = re.findall(r"-?\d[\d,]*\.?\d*", answer)
            if not nums:
                return 0.0, "no number in answer"
            got = nums[-1]
            note = "no ANSWER: line, used last number"
        else:
            got, note = hits[-1], ""
        try:
            val = float(got.replace(",", ""))
        except ValueError:
            return 0.0, f"unparsable number {got!r}"
        ok = abs(val - expected) < 1e-6
        if ok:
            return 1.0, note
        return 0.0, (f"got {val}, want {expected}"
                     + (f" ({note})" if note else ""))

    return score


def _keyword_scorer(*needles: str) -> Callable[[str], Tuple[float, str]]:
    def score(answer: str) -> Tuple[float, str]:
        low = (answer or "").lower()
        hit = [n for n in needles if n.lower() in low]
        return (1.0 if hit else 0.0), ("" if hit else f"none of {needles!r}")

    return score


MATH_PROBES: List[Tuple[str, str, float]] = [
    ("math_mul", "What is 17 * 23? Think briefly, then end with a final "
                 "line exactly: ANSWER: <number>", 391),
    ("math_avgspeed", "A train travels 240 km in 3 hours, then 180 km in 2 "
                      "hours. What is its average speed over the whole trip "
                      "in km/h? End with a final line exactly: "
                      "ANSWER: <number>", 84),
    ("math_pow", "What is 2^10 + 3^5? End with a final line exactly: "
                 "ANSWER: <number>", 1267),
    ("math_pct", "What is 15% of 840? End with a final line exactly: "
                 "ANSWER: <number>", 126),
    ("math_sum100", "What is the sum of all integers from 1 to 100? End "
                    "with a final line exactly: ANSWER: <number>", 5050),
    ("math_linear", "Solve for x: x/4 + 7 = 19. End with a final line "
                    "exactly: ANSWER: <number>", 48),
    ("math_gcd", "What is the greatest common divisor of 462 and 1071? End "
                 "with a final line exactly: ANSWER: <number>", 21),
    ("math_rect", "A rectangle has perimeter 34 and length 10. What is its "
                  "area? End with a final line exactly: ANSWER: <number>", 70),
    ("math_deriv", "Let f(x) = 3x^3 - 5x. What is f'(2)? End with a final "
                   "line exactly: ANSWER: <number>", 31),
    ("math_perm", "How many distinct arrangements are there of the letters "
                  "in the word BANANA? End with a final line exactly: "
                  "ANSWER: <number>", 60),
    ("math_solve2", "Solve for x: 3x - 7 = 2x + 5. End with a final line "
                    "exactly: ANSWER: <number>", 12),
    ("math_bin", "Convert the binary number 10110110 to decimal. End with a "
                 "final line exactly: ANSWER: <number>", 182),
]

FACT_PROBES: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("fact_capital", "What is the capital city of Australia? Answer in one "
                     "short sentence.", ("canberra",)),
    ("fact_author", "Who wrote the novel 'One Hundred Years of Solitude'? "
                    "Answer in one short sentence.", ("marquez", "márquez")),
    ("fact_tungsten", "What is the chemical symbol for tungsten? Answer in "
                      "one short sentence.", (" w ", "'w'", '"w"', "symbol w",
                                              "is w.", "is w,")),
    ("fact_planet", "Which is the largest planet in our solar system? "
                    "Answer in one short sentence.", ("jupiter",)),
    ("fact_wall", "In which year did the Berlin Wall fall? Answer in one "
                  "short sentence.", ("1989",)),
    ("fact_organelle", "Which organelle is known as the powerhouse of the "
                       "cell? Answer in one short sentence.",
     ("mitochondri",)),
    ("fact_fermat", "Who proved Fermat's Last Theorem? Answer in one short "
                    "sentence.", ("wiles",)),
    ("fact_river", "What is the longest river in South America? Answer in "
                   "one short sentence.", ("amazon",)),
    ("fact_http404", "What does the HTTP status code 404 mean? Answer in "
                     "one short sentence.",
     ("not found", "could not find", "could not be found", "cannot find",
      "cannot be found", "couldn't find", "wasn't found", "was not found",
      "does not exist", "doesn't exist", "no such", "unable to find",
      "missing")),
    ("fact_gold", "Which chemical element has atomic number 79? Answer in "
                  "one short sentence.", ("gold", "aurum")),
]


def _score_exact_ok(answer: str) -> Tuple[float, str]:
    got = (answer or "").strip().strip(".")
    return (1.0 if got == "OK" else 0.0), f"got {got[:40]!r}"


def _score_json_obj(answer: str) -> Tuple[float, str]:
    text = (answer or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        d = json.loads(text)
    except Exception as e:
        return 0.0, f"not JSON: {e}"
    if not isinstance(d, dict):
        return 0.0, "JSON is not an object"
    if set(d) != {"name", "age"}:
        return 0.0, f"keys {sorted(d)}"
    if not isinstance(d.get("name"), str) or not isinstance(d.get("age"), int):
        return 0.0, "wrong value types"
    return 1.0, ""


def _score_five_bullets(answer: str) -> Tuple[float, str]:
    lines = [ln.strip() for ln in (answer or "").split("\n") if ln.strip()]
    bullets = [ln for ln in lines if ln.startswith("- ")]
    if len(lines) != len(bullets):
        return 0.0, f"{len(lines) - len(bullets)} non-bullet lines"
    return (1.0 if len(bullets) == 5 else 0.0), f"{len(bullets)} bullets"


def _score_three_words(answer: str) -> Tuple[float, str]:
    words = re.findall(r"[A-Za-z']+", answer or "")
    return (1.0 if len(words) == 3 else 0.0), f"{len(words)} words"


def _score_banana(answer: str) -> Tuple[float, str]:
    got = (answer or "").strip()
    want = "BANANA BANANA BANANA BANANA"
    return (1.0 if got == want else 0.0), f"got {got[:40]!r}"


def _score_no_e(answer: str) -> Tuple[float, str]:
    text = (answer or "").strip()
    if len(text) < 10:
        return 0.0, "too short to be a sentence"
    n = text.lower().count("e")
    return (1.0 if n == 0 else 0.0), f"{n} 'e' characters"


def _score_primes(answer: str) -> Tuple[float, str]:
    got = (answer or "").strip().strip(".")
    return (1.0 if got == "2,3,5,7,11,13" else 0.0), f"got {got[:40]!r}"


def _score_upper(answer: str) -> Tuple[float, str]:
    text = (answer or "").strip()
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 10:
        return 0.0, "too short"
    lower = [c for c in letters if c.islower()]
    return (1.0 if not lower else 0.0), f"{len(lower)} lowercase letters"


INSTR_PROBES: List[Tuple[str, str, Callable[[str], Tuple[float, str]]]] = [
    ("instr_ok", "Reply with exactly the two characters OK and nothing "
                 "else. No punctuation, no explanation.", _score_exact_ok),
    ("instr_json", "Output a single JSON object with exactly the keys "
                   "\"name\" (a string) and \"age\" (an integer). Output "
                   "only the JSON, no prose and no code fence.",
     _score_json_obj),
    ("instr_bullets", "List exactly 5 bullet points about rain. Every line "
                      "must start with '- '. Output nothing else: no "
                      "heading, no closing sentence.", _score_five_bullets),
    ("instr_threewords", "Answer in exactly three words and nothing else: "
                         "what colour is the sky on a clear day?",
     _score_three_words),
    ("instr_repeat", "Reply with the word BANANA repeated 4 times, "
                     "separated by single spaces, and nothing else.",
     _score_banana),
    ("instr_noe", "Write one sentence about cats that contains no letter "
                  "'e' at all. Output only that sentence.", _score_no_e),
    ("instr_primes", "Return only a comma-separated list of the first 6 "
                     "prime numbers. No spaces, no prose, no trailing "
                     "period.", _score_primes),
    ("instr_upper", "Reply in ALL UPPERCASE with a single sentence about "
                    "the ocean. Output only that sentence.", _score_upper),
]

#: (id, prompt, harness). The harness is appended to the extracted code and
#: must print nothing and raise on failure.
CODE_PROBES: List[Tuple[str, str, str]] = [
    ("code_subarray",
     "Write a Python solution: `class Solution:` with method "
     "`def maxSubArrayLen(self, nums: list[int], k: int) -> int` returning "
     "the length of the longest subarray summing to k, or 0 if none exists. "
     "Output the code in a single ```python fenced block.",
     "s = Solution()\n"
     "assert s.maxSubArrayLen([1, -1, 5, -2, 3], 3) == 4\n"
     "assert s.maxSubArrayLen([-2, -1, 2, 1], 1) == 2\n"
     "assert s.maxSubArrayLen([1, 2, 3], 100) == 0\n"
     "assert s.maxSubArrayLen([0, 0, 0], 0) == 3\n"),
    ("code_palindrome",
     "Write a Python function `def is_palindrome(s: str) -> bool` that "
     "ignores case and every non-alphanumeric character. Output the code in "
     "a single ```python fenced block, no tests.",
     "assert is_palindrome('A man, a plan, a canal: Panama') is True\n"
     "assert is_palindrome('race a car') is False\n"
     "assert is_palindrome('') is True\n"
     "assert is_palindrome('0P') is False\n"),
    ("code_intervals",
     "Write a Python function `def merge_intervals(intervals)` that merges "
     "overlapping intervals given as a list of [start, end] pairs and "
     "returns the merged list sorted by start, each item a list of two "
     "ints. Output the code in a single ```python fenced block, no tests.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == "
     "[[1,6],[8,10],[15,18]]\n"
     "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
     "assert merge_intervals([]) == []\n"),
    ("code_fib",
     "Write a Python function `def fib(n)` returning the nth Fibonacci "
     "number with fib(0) == 0 and fib(1) == 1, computed iteratively. Output "
     "the code in a single ```python fenced block, no tests.",
     "assert fib(0) == 0 and fib(1) == 1 and fib(10) == 55\n"
     "assert fib(30) == 832040\n"),
    ("code_wordfreq",
     "Write a Python function `def word_freq(text)` returning a dict "
     "mapping each lowercased word to its count. A word is a maximal run of "
     "the characters a-z, A-Z and the apostrophe. Output the code in a "
     "single ```python fenced block, no tests.",
     "assert word_freq(\"The cat, the CAT; a dog\") == "
     "{'the': 2, 'cat': 2, 'a': 1, 'dog': 1}\n"
     "assert word_freq('') == {}\n"
     "assert word_freq(\"don't don't\") == {\"don't\": 2}\n"),
    ("code_rotate",
     "Write a Python function `def rotate(matrix)` that rotates an n x n "
     "matrix 90 degrees clockwise IN PLACE and returns None. Output the "
     "code in a single ```python fenced block, no tests.",
     "m = [[1,2,3],[4,5,6],[7,8,9]]\n"
     "assert rotate(m) is None\n"
     "assert m == [[7,4,1],[8,5,2],[9,6,3]]\n"
     "m2 = [[1,2],[3,4]]\n"
     "rotate(m2)\n"
     "assert m2 == [[3,1],[4,2]]\n"),
]

#: Long-context recall: (filler target tokens, how many distinct secrets).
NEEDLE_TARGETS: Tuple[Tuple[int, int], ...] = ((4000, 2), (16000, 2),
                                               (30000, 2))


def extract_python(answer: str) -> str:
    """The largest ```python block, else the whole answer minus prose lines
    that obviously are not code."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer or "",
                        flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len)
    return answer or ""


def run_code_probe(answer: str, harness: str,
                   timeout: float = 20.0) -> Tuple[float, str]:
    """Execute the extracted code plus its assertion harness in a separate
    interpreter. Any non-zero exit is a 0; the first line of stderr is the
    note, so a wrong answer is distinguishable from a syntax error."""
    code = extract_python(answer)
    if not code.strip():
        return 0.0, "no code in answer"
    src = code + "\n\n" + harness + "\nprint('PROBE_OK')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.unlink(path)
        return 0.0, f"timeout after {timeout:.0f}s"
    os.unlink(path)
    if p.returncode == 0 and "PROBE_OK" in p.stdout:
        return 1.0, ""
    err = (p.stderr or "").strip().split("\n")
    return 0.0, (err[-1][:160] if err else f"exit {p.returncode}")


# ===========================================================================
# Running one arm.
# ===========================================================================
def _post(base: str, path: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"status": r.getcode() or 200,
                    "body": r.read().decode("utf-8", errors="replace"),
                    "wall_ms": (time.time() - t0) * 1000.0}
    except Exception as e:
        return {"status": 0, "body": "", "error": str(e),
                "wall_ms": (time.time() - t0) * 1000.0}


def _chat_answer(base: str, model: str, body: dict,
                 timeout: float) -> Tuple[str, dict]:
    body = dict(body, model=model)
    res = _post(base, "/v1/chat/completions", body, timeout)
    meta = {"http_code": res["status"], "wall_ms": round(res["wall_ms"], 1)}
    if res["status"] != 200:
        meta["error"] = res.get("error") or res["body"][:300]
        return "", meta
    try:
        d = json.loads(res["body"])
        choice = (d.get("choices") or [{}])[0]
        usage = d.get("usage") or {}
        meta["finish"] = choice.get("finish_reason")
        meta["prompt_tokens"] = usage.get("prompt_tokens")
        meta["completion_tokens"] = usage.get("completion_tokens")
        return (choice.get("message") or {}).get("content") or "", meta
    except Exception as e:
        meta["error"] = f"unparsable answer: {e}"
        return "", meta


def probe_records(model: str) -> List[dict]:
    """The full graded catalog as (id, category, body, scorer) records. Built
    once per arm from the same seed, so the arms are asked the same things."""
    rng = random.Random(SEED)
    out: List[dict] = []
    for pid, prompt, expected in MATH_PROBES:
        out.append({"id": pid, "category": "math", "timeout": 180.0,
                    "body": _chat(prompt, 1024),
                    "scorer": _num_scorer(expected)})
    for pid, prompt, needles in FACT_PROBES:
        out.append({"id": pid, "category": "factual", "timeout": 120.0,
                    "body": _chat(prompt, 128),
                    "scorer": _keyword_scorer(*needles)})
    for pid, prompt, scorer in INSTR_PROBES:
        out.append({"id": pid, "category": "instruction", "timeout": 180.0,
                    "body": _chat(prompt, 400), "scorer": scorer})
    for pid, prompt, harness in CODE_PROBES:
        out.append({"id": pid, "category": "code", "timeout": 420.0,
                    "body": _chat(prompt, 1200),
                    "scorer": (lambda h: lambda a: run_code_probe(a, h))(
                        harness)})
    for target, count in NEEDLE_TARGETS:
        for i in range(count):
            secret = bs.make_needle_secret(rng)
            scale = max(100, target // bs.TOKENS_PER_FILLER_SCALE)
            body = bs.request_needle("", scale, secret)
            body.pop("model", None)
            body["stream"] = False
            body.pop("stream_options", None)
            out.append({
                "id": f"needle_{target}_{i + 1}", "category": "longctx",
                "timeout": 600.0, "body": body,
                "scorer": (lambda s: lambda a: (
                    (1.0, "") if bs.needle_recall_ok(a, s)
                    else (0.0, f"want {s!r}, got {(a or '')[:60]!r}")))(secret),
                "secret": secret, "target_tokens": target,
            })
    return out


def accept_length_probe(base: str, timeout: float = 240.0) -> dict:
    """The accept length of ONE decode, read where the number actually is.

    ``/v1/chat/completions`` returns no spec block; the native ``/generate``
    answer carries ``meta_info.spec_accept_length`` for that request. The
    Prometheus EMA gauge is captured next to it for provenance only -- it
    averages over the server's whole life and is not this request's figure.
    """
    body = {
        "text": "Count from 1 to 200, one number per line, nothing else.\n",
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 600},
    }
    res = _post(base, "/generate", body, timeout)
    rec: dict = {"http_code": res["status"], "wall_ms": round(res["wall_ms"], 1)}
    if res["status"] == 200:
        try:
            d = json.loads(res["body"])
            mi = d.get("meta_info") or {}
            rec["spec_accept_length"] = mi.get("spec_accept_length")
            rec["spec_verify_ct"] = mi.get("spec_verify_ct")
            rec["completion_tokens"] = mi.get("completion_tokens")
            rec["e2e_latency"] = mi.get("e2e_latency")
        except Exception as e:
            rec["error"] = str(e)
    else:
        rec["error"] = res.get("error") or res["body"][:300]
    m = bs._default_http("GET", base + "/metrics", timeout=15)
    if m.status == 200:
        flat = bs.parse_prometheus_metrics(m.body)
        rec["prom_spec_ema_accept_len"] = flat.get(bs.SPEC_EMA_ACCEPT_LEN_METRIC)
        rec["prom_note"] = ("server-lifetime EMA, provenance only -- not this "
                            "request's accept length")
    return rec


def run_arm(base: str, model: str, arm: str, out_dir: str,
            suite_tests: List[int], vram_cb: Optional[Callable[[], Optional[int]]]
            ) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # --- the #151 suite ---------------------------------------------------
    sink: list = []
    suite_rows: List[dict] = []
    t0 = time.time()
    for res in bs.run_suite(base, model, selected=suite_tests,
                            rng=random.Random(SEED), vram_free_mib=vram_cb,
                            transcript_sink=sink, timeout=300.0):
        row = dict(res)
        suite_rows.append(row)
        print(f"  [{arm}] suite {row.get('test_id')} {row.get('label')}: "
              f"{row.get('status')} {(row.get('metric') or {}).get('value')}",
              flush=True)
    suite_secs = time.time() - t0
    with open(os.path.join(out_dir, f"suite_{arm}.jsonl"), "w") as fh:
        for row in suite_rows:
            fh.write(json.dumps(row, default=str) + "\n")
    transcript = sink[0].transcript if sink else []
    with open(os.path.join(out_dir, f"transcript_{arm}.json"), "w") as fh:
        json.dump(transcript, fh, default=str, indent=1)

    # --- the graded probes ------------------------------------------------
    rows: List[dict] = []
    t1 = time.time()
    for rec in probe_records(model):
        answer, meta = _chat_answer(base, model, rec["body"], rec["timeout"])
        try:
            score, note = rec["scorer"](answer)
        except Exception as e:  # a scorer must never take the arm down
            score, note = 0.0, f"scorer raised: {e}"
        # A miss on a truncated answer is a token-budget artefact, not a wrong
        # answer, and it must be visible as such in the raw rows.
        if score < 1.0 and meta.get("finish") == "length":
            note = (note + " [TRUNCATED at max_tokens]").strip()
        row = {"id": rec["id"], "category": rec["category"], "score": score,
               "note": note, "answer": answer, **meta}
        if "secret" in rec:
            row["secret"] = rec["secret"]
            row["target_tokens"] = rec["target_tokens"]
        rows.append(row)
        print(f"  [{arm}] probe {rec['id']}: {score:.0f} {note[:60]}",
              flush=True)
    probe_secs = time.time() - t1
    with open(os.path.join(out_dir, f"probes_{arm}.jsonl"), "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")

    accept = accept_length_probe(base)
    print(f"  [{arm}] accept_length: {accept.get('spec_accept_length')} "
          f"(prom EMA {accept.get('prom_spec_ema_accept_len')})", flush=True)
    with open(os.path.join(out_dir, f"accept_{arm}.json"), "w") as fh:
        json.dump(accept, fh, indent=1)

    summary = {"arm": arm, "model": model, "endpoint": base,
               "suite_seconds": round(suite_secs, 1),
               "probe_seconds": round(probe_secs, 1),
               "suite_counts": _tally([r.get("status") for r in suite_rows]),
               "probe_scores": category_scores(rows),
               "accept": accept}
    with open(os.path.join(out_dir, f"summary_{arm}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary["probe_scores"], indent=1), flush=True)


def _tally(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def category_scores(rows: List[dict]) -> Dict[str, dict]:
    cats: Dict[str, List[float]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(float(r["score"]))
    out: Dict[str, dict] = {}
    total: List[float] = []
    for cat, vals in cats.items():
        out[cat] = {"n": len(vals), "hits": int(sum(vals)),
                    "score": round(sum(vals) / len(vals), 4)}
        total += vals
    if total:
        out["ALL"] = {"n": len(total), "hits": int(sum(total)),
                      "score": round(sum(total) / len(total), 4)}
    return out


# ===========================================================================
# Comparing arms.
# ===========================================================================
def _read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def rescore(out_dir: str, arms: List[str]) -> None:
    """Re-apply the CURRENT scorers to the answers already on disk.

    A scorer that misses a correct answer is a defect in the scorer, and the
    honest fix is to widen it and re-grade every arm with the identical rule
    -- not to re-run the arm that happened to trip it, which would give one
    format a second draw. Nothing here touches a GPU or a model: the answers
    are the evidence, they were written once, and they do not change. Each
    rewritten row keeps ``score_before`` so the correction is auditable.
    """
    scorers = {r["id"]: r["scorer"] for r in probe_records("")}
    for arm in arms:
        path = os.path.join(out_dir, f"probes_{arm}.jsonl")
        rows = _read_jsonl(path)
        if not rows:
            continue
        changed = 0
        for r in rows:
            fn = scorers.get(r["id"])
            if fn is None:
                continue
            try:
                score, note = fn(r.get("answer") or "")
            except Exception as e:
                score, note = 0.0, f"scorer raised: {e}"
            if score != r["score"]:
                r["score_before"] = r["score"]
                r["note_before"] = r.get("note")
                changed += 1
            r["score"], r["note"] = score, note
            if score < 1.0 and r.get("finish") == "length":
                r["note"] = (r["note"] + " [TRUNCATED at max_tokens]").strip()
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=str) + "\n")
        spath = os.path.join(out_dir, f"summary_{arm}.json")
        if os.path.exists(spath):
            with open(spath) as fh:
                summary = json.load(fh)
            summary["probe_scores"] = category_scores(rows)
            summary["rescored"] = True
            with open(spath, "w") as fh:
                json.dump(summary, fh, indent=1)
        print(f"{arm}: {changed} row(s) re-graded, "
              f"now {category_scores(rows)['ALL']}")


#: Continuous readings pulled out of the suite rows and the accept probe.
#: A tri-state suite plus a graded set that a 27B model saturates would answer
#: "no gross regression" and nothing finer. These are the axes that still MOVE
#: at the ceiling, so a difference too small to flip a verdict is still
#: visible -- and each one is read against the same A-vs-A band.
def continuous_metrics(out_dir: str, arm: str) -> Dict[str, Optional[float]]:
    rows = {r.get("test_id"): r
            for r in _read_jsonl(os.path.join(out_dir, f"suite_{arm}.jsonl"))}
    probes = _read_jsonl(os.path.join(out_dir, f"probes_{arm}.jsonl"))

    def detail(tid: int, key: str) -> Optional[float]:
        d = (rows.get(tid) or {}).get("detail") or {}
        v = d.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    out: Dict[str, Optional[float]] = {
        # test 6: the degeneracy scan's own continuous output.
        "lexical_variety_t6": detail(6, "lexical_variety"),
        "essay_chars_t6": detail(6, "chars"),
        "max_line_repeat_t6": detail(6, "max_line_repeat"),
        # test 8: how deep the needle was still recalled.
        "deepest_recall_tokens_t8": None,
        # test 13: how much of the 8K reasoning budget was actually used.
        "completion_tokens_t13": detail(13, "completion_tokens"),
    }
    m8 = (rows.get(8) or {}).get("metric") or {}
    if isinstance(m8.get("numeric"), (int, float)):
        out["deepest_recall_tokens_t8"] = float(m8["numeric"])

    acc_path = os.path.join(out_dir, f"accept_{arm}.json")
    if os.path.exists(acc_path):
        with open(acc_path) as fh:
            acc = json.load(fh)
        v = acc.get("spec_accept_length")
        out["spec_accept_length"] = float(v) if isinstance(v, (int, float)) \
            else None

    # Mean answer length over the graded probes: a coarse but sensitive
    # summary of "is this model still saying the same amount of thing".
    lens = [len(r.get("answer") or "") for r in probes]
    out["mean_answer_chars"] = round(sum(lens) / len(lens), 1) if lens else None
    return out


def compare(out_dir: str, arms: List[str], noise_pair: Tuple[str, str]) -> dict:
    per_arm = {a: _read_jsonl(os.path.join(out_dir, f"probes_{a}.jsonl"))
               for a in arms}
    suites = {a: _read_jsonl(os.path.join(out_dir, f"suite_{a}.jsonl"))
              for a in arms}
    scores = {a: category_scores(rows) for a, rows in per_arm.items() if rows}

    def identity(a: str, b: str) -> dict:
        ra = {r["id"]: (r.get("answer") or "") for r in per_arm.get(a, [])}
        rb = {r["id"]: (r.get("answer") or "") for r in per_arm.get(b, [])}
        common = sorted(set(ra) & set(rb))
        same = [i for i in common if ra[i] == rb[i]]
        return {"n": len(common), "identical": len(same),
                "rate": round(len(same) / len(common), 4) if common else None,
                "differing": [i for i in common if ra[i] != rb[i]]}

    n_a, n_b = noise_pair
    cats = sorted({c for s in scores.values() for c in s})
    band = {c: round(abs(scores.get(n_a, {}).get(c, {}).get("score", 0.0)
                         - scores.get(n_b, {}).get(c, {}).get("score", 0.0)), 4)
            for c in cats}
    out = {
        "arms": arms,
        "noise_pair": list(noise_pair),
        "noise_band_by_category": band,
        "identity_rates": {f"{a}|{b}": identity(a, b)
                           for a, b in zip(arms, arms[1:])},
        "scores": scores,
        "suite_status": {a: _tally([r.get("status") for r in rows])
                         for a, rows in suites.items() if rows},
        "suite_by_test": {
            a: {f"{r.get('test_id')}:{r.get('label')}": r.get("status")
                for r in rows}
            for a, rows in suites.items() if rows},
    }
    cont = {a: continuous_metrics(out_dir, a) for a in arms}
    out["continuous"] = cont
    out["continuous_noise_band"] = {
        k: (None if cont.get(n_a, {}).get(k) is None
            or cont.get(n_b, {}).get(k) is None
            else round(abs(cont[n_a][k] - cont[n_b][k]), 4))
        for k in sorted({k for m in cont.values() for k in m})
    }
    for a, b in [(x, y) for x in arms for y in arms if x != y]:
        out["identity_rates"].setdefault(f"{a}|{b}", identity(a, b))
    with open(os.path.join(out_dir, "compare.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def print_table(cmp_out: dict) -> None:
    scores, arms = cmp_out["scores"], cmp_out["arms"]
    cats = sorted({c for s in scores.values() for c in s if c != "ALL"}) + ["ALL"]
    head = "category".ljust(14) + "".join(a.ljust(16) for a in arms) + "band"
    print(head)
    print("-" * len(head))
    for c in cats:
        line = c.ljust(14)
        for a in arms:
            s = scores.get(a, {}).get(c)
            line += (f"{s['score']:.3f} ({s['hits']}/{s['n']})".ljust(16)
                     if s else "-".ljust(16))
        line += f"{cmp_out['noise_band_by_category'].get(c, 0.0):.3f}"
        print(line)

    cont = cmp_out.get("continuous") or {}
    if cont:
        print("\ncontinuous metrics (band = the same A-vs-A pair):")
        head2 = "metric".ljust(26) + "".join(a.ljust(16) for a in arms) + "band"
        print(head2)
        print("-" * len(head2))
        for k in sorted({k for m in cont.values() for k in m}):
            line = k.ljust(26)
            for a in arms:
                v = cont.get(a, {}).get(k)
                line += ("-" if v is None else f"{v:g}").ljust(16)
            b = (cmp_out.get("continuous_noise_band") or {}).get(k)
            line += "-" if b is None else f"{b:g}"
            print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("run", "compare", "rescore"),
                    required=True)
    ap.add_argument("--port", type=int, default=30360)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="")
    ap.add_argument("--arm", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--suite-tests", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14")
    ap.add_argument("--arms", default="")
    ap.add_argument("--noise-pair", default="")
    args = ap.parse_args()

    if args.mode == "run":
        base = f"http://{args.host}:{args.port}"
        tests = [int(x) for x in args.suite_tests.split(",") if x.strip()]
        model = args.model
        if not model:
            r = bs._default_http("GET", base + "/get_model_info", timeout=15)
            model = ((r.json() or {}).get("model_path") or "").split("/")[-1]
        run_arm(base, model, args.arm, args.out_dir, tests, _nvml_min_free)
        return 0

    arms = [a for a in args.arms.split(",") if a.strip()]
    if args.mode == "rescore":
        rescore(args.out_dir, arms)
        return 0
    pair = tuple(args.noise_pair.split(",")) if args.noise_pair else \
        (arms[0], arms[1])
    out = compare(args.out_dir, arms, (pair[0], pair[1]))
    print_table(out)
    print("\nsuite status per arm:")
    for a, t in out["suite_status"].items():
        print(f"  {a}: {t}")
    print("\nanswer identity between arms:")
    for k, v in out["identity_rates"].items():
        if v["n"]:
            print(f"  {k}: {v['identical']}/{v['n']} identical "
                  f"({v['rate']:.2f})")
    return 0


def _nvml_min_free() -> Optional[int]:
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        vals = [int(x) for x in p.stdout.split() if x.strip().isdigit()]
        return min(vals) if vals else None
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
