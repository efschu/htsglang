#!/usr/bin/env python3
"""#1243 step 1 -- quality probe runner for the fp8-body / bf16-tail question.

ONE question, answered with numbers, before anything is built: does the fp8_e4m3
KV cache cost measurable quality against a bf16 KV cache, and does keeping the
youngest N tokens in 16 bit buy it back?  If the gain is inside the A-vs-A noise
floor, the feature is NOT built.

WHAT THIS MEASURES, AND WITH WHICH INSTRUMENT
=============================================
Two instruments, deliberately kept apart because they have different blind
spots and different denominators:

  (1) TEACHER-FORCED DISTRIBUTION DISTANCE (the primary instrument).
      A fixed token sequence is fed to the server and the per-position
      distribution over that same sequence is read back.  Sampling is removed
      entirely -- both arms are scored at byte-identical prefixes, position by
      position.  Reported per arm vs the bf16 reference:
        * mean/median truncated KLD (nats) over the reference's top-k support,
          renormalised on both sides.  This is a LOWER bound on the true KLD:
          mass the arms disagree about outside the reference's top-k is
          invisible to it.  Denominator: scored positions, printed.
        * top-1 and top-5 agreement (the argmax a greedy decoder would emit).
          Denominator: scored positions, printed.
        * mean |delta logprob| on the ACTUAL next token.  This one is EXACT
          and full-vocab -- no truncation anywhere -- and is the number to
          quote when the truncated KLD is challenged.
      Estimator inherited verbatim from the #855 capture,
      ``scripts/int8_368/kld_capture_855.py`` at commit 477fb3dc7b.

  (2) DETERMINED-ANSWER CORRECTNESS (the secondary instrument).
      Generation at temperature 0 against prompts with a checkable answer.
      Denominator: probes attempted, printed.  A correctness delta of 1/45 is
      not a signal; this instrument exists to catch a GROSS regression the
      distribution distance would under-read, not to resolve small effects.

LOGPROBS CAPABILITY -- VERIFIED FROM THIS TREE'S CODE, NOT ASSUMED
==================================================================
The OpenAI-compatible endpoints DO NOT return prompt logprobs on this server:

    entrypoints/openai/serving_completions.py:71-75
        # NOTE: with openai API, the prompt's logprobs are always not computed
        if request.echo and request.logprobs:
            logger.warning("Echo is not compatible with logprobs. "
                "To compute logprobs of input prompt, please use the native
                 /generate API.")

They return top-k logprobs for GENERATED tokens only
(``protocol.py:772-773`` ``logprobs: bool`` / ``top_logprobs: Optional[int]``
-> ``serving_chat.py:647`` ``top_logprobs_num=request.top_logprobs or 0``;
``protocol.py:407`` ``logprobs: Optional[int]`` ->
``serving_completions.py:114``).

The NATIVE ``/generate`` endpoint does return them, per input position:
``managers/io_struct.py:199-204`` (``return_logprob`` / ``logprob_start_len`` /
``top_logprobs_num``) and ``managers/tokenizer_manager.py:2278``
(``meta_info["input_top_logprobs"]``).

So instrument (1) uses the NATIVE ``/generate`` endpoint and instrument (2)
uses the OpenAI ``/v1/chat/completions`` endpoint (it needs the chat template).
Only top-k is ever available, never the full 150k-vocab distribution, so the
KLD is TRUNCATED and this file says so on every line that prints it.

CORPUS -- REAL TEXT ONLY, EVERY SOURCE CITED
============================================
No lorem.  ``corpus`` writes a manifest naming every source file with its
sha256 and byte size, and every built passage records the measured token count
the server itself reported.

  * SHORT set: the club-3090 quality-suite scenarios, MIT-licensed, from
    ``benchlocal_cli`` packs on this box -- ``reasonmath-15`` and
    ``dataextract-15`` (both carry determined answers: ``canonical_answer`` /
    ``accepted_answers`` and ``expected`` respectively) and
    ``instructfollow-15``.  Same suite the BOOT records under
    /spinning/gpu-arb/weg2/ score with (``scripts/quality-test.sh``).
  * LONG set: prefixes of real technical documents on this box at ~8k / ~32k /
    ~100k tokens.  Long-context recall probes are needle-in-a-haystack over
    that REAL text: one inserted marker sentence near the start, everything
    else is the document.  The insertion offset and the marker are recorded.

USAGE
=====
    probe_kv_tail.py corpus   --out corpus.json [--model-path DIR]
    probe_kv_tail.py capture  --arm A0_bf16 --corpus corpus.json --out A0.json \
                              [--url http://127.0.0.1:30032] [--topk 20]
    probe_kv_tail.py compare  --ref A0.json --ref2 A0b.json --arm N0.json ...
    probe_kv_tail.py self-test          # hermetic, no server, no GPU
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Corpus sources.  Every entry is a real file on this box; the manifest records
# sha256 + size so a corpus can never silently drift between arms.
# --------------------------------------------------------------------------

PACK_DIR = "/root/.venvs/benchlocal/lib/python3.12/site-packages/benchlocal_cli/packs"

#: club-3090 quality-suite packs.  MIT ("license": "MIT" in each pack's
#: __meta__ row).  reasonmath/dataextract carry determined answers.
SHORT_PACKS = ("reasonmath-15.jsonl", "dataextract-15.jsonl", "instructfollow-15.jsonl")

#: Real long documents on this box.  English + German technical prose, tables,
#: code blocks and log excerpts -- the registers this server actually serves.
LONG_SOURCES = (
    "/spinning/htsglang/docs/rig-runbook.md",
    "/spinning/htsglang/docs/dev/FEATURE_CATALOG.md",
    "/spinning/gpu-arb/weg2/WEG2_BUILD_DECISIONS_0906.md",
)

#: Target prompt sizes for the long set, in TOKENS.  Measured per passage and
#: reported; these are targets, never the quoted number.
LONG_TARGET_TOKENS = (8192, 32768, 102400)

#: Positions scored per LONG passage: the last SCORE_WINDOW of them.  Short
#: passages are always scored whole.
SCORE_WINDOW = 4096

#: The needle marker.  Fixed, not random per run, so two arms are byte-identical.
NEEDLE_TEMPLATE = (
    "\n\nMerkzeile 7 (Pruefmarke dieses Dokuments): "
    "the arbitration token for this document is {token}. "
    "Diese Zeile gehoert zur Pruefung und steht genau einmal im Text.\n\n"
)
NEEDLE_TOKENS = ("QF7-2K9", "MB4-7X1", "ZR8-3C6")
NEEDLE_QUESTION = (
    "\n\nFrage zum obigen Dokument: Welcher arbitration token wird in "
    "'Merkzeile 7' genannt? Antworte mit genau einer Zeile der Form "
    "TOKEN: <wert> und sonst nichts."
)
#: Fraction of the document the needle sits after.  Deliberately EARLY: the
#: whole point is an answer that depends on content the tail window no longer
#: covers.
NEEDLE_OFFSET_FRACTION = 0.02


# --------------------------------------------------------------------------
# The math.  Pure, importable, hermetically tested by
# test/registered/unit/mem_cache/test_kv_tail_probe_1243.py.
# --------------------------------------------------------------------------


def _as_pairs(entry) -> List[Tuple[float, int]]:
    """Normalise one position's top-k list to ``[(logprob, token_id), ...]``.

    The native endpoint returns ``[logprob, token_id, token_text]`` triples.
    """
    out: List[Tuple[float, int]] = []
    for item in entry or ():
        if item is None:
            continue
        lp, tok = item[0], item[1]
        if lp is None or tok is None:
            continue
        out.append((float(lp), int(tok)))
    return out


def truncated_kld(ref_top, new_top) -> Optional[float]:
    """KL(ref || new) over the REFERENCE's top-k support, renormalised.

    Returns None when fewer than two reference tokens are present in the new
    arm's top-k -- a one-point "distribution" carries no divergence and
    including it as 0.0 would dilute the mean with a fake agreement.

    Both sides are renormalised over the shared support, so this is a proper
    KLD between two probability vectors and is >= 0 up to float error.  It is a
    LOWER bound on the true full-vocab KLD.
    """
    ref = _as_pairs(ref_top)
    new = _as_pairs(new_top)
    if not ref or not new:
        return None
    nmap = {tok: lp for lp, tok in new}
    pairs = [(lp, nmap[tok]) for lp, tok in ref if tok in nmap]
    if len(pairs) < 2:
        return None
    zr = math.log(sum(math.exp(a) for a, _ in pairs))
    zn = math.log(sum(math.exp(b) for _, b in pairs))
    return sum(math.exp(a - zr) * ((a - zr) - (b - zn)) for a, b in pairs)


def topk_agreement(ref_top, new_top, k: int) -> Optional[bool]:
    """Is the reference's argmax inside the new arm's top-k?

    k=1 is plain argmax agreement (what a greedy decoder would emit).  Returns
    None when either side has no usable entry, so the denominator stays honest.
    """
    ref = _as_pairs(ref_top)
    new = _as_pairs(new_top)
    if not ref or not new:
        return None
    ref_sorted = sorted(ref, key=lambda p: -p[0])
    new_sorted = sorted(new, key=lambda p: -p[0])
    if len(new_sorted) < k:
        return None
    return ref_sorted[0][1] in {tok for _, tok in new_sorted[:k]}


def aggregate(values: Sequence[float]) -> Dict[str, object]:
    """n / mean / median / p95 / max.  ``n`` IS the denominator; always printed."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(vals)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p95": ordered[idx],
        "max": ordered[-1],
    }


def score_pair(ref_passage: dict, new_passage: dict) -> Dict[str, List]:
    """All per-position quantities for one passage of one arm vs the reference."""
    klds: List[float] = []
    top1: List[bool] = []
    top5: List[bool] = []
    dlp: List[float] = []

    ref_top = ref_passage.get("input_top_logprobs") or []
    new_top = new_passage.get("input_top_logprobs") or []
    for rt, nt in zip(ref_top, new_top):
        k = truncated_kld(rt, nt)
        if k is not None:
            klds.append(k)
        a1 = topk_agreement(rt, nt, 1)
        if a1 is not None:
            top1.append(a1)
        a5 = topk_agreement(rt, nt, 5)
        if a5 is not None:
            top5.append(a5)

    ref_lp = ref_passage.get("input_token_logprobs") or []
    new_lp = new_passage.get("input_token_logprobs") or []
    for lr, ln in zip(ref_lp, new_lp):
        if not lr or not ln or lr[0] is None or ln[0] is None:
            continue
        dlp.append(abs(float(lr[0]) - float(ln[0])))

    return {"kld": klds, "top1": top1, "top5": top5, "dlogprob": dlp}


# --------------------------------------------------------------------------
# Server I/O
# --------------------------------------------------------------------------


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def harness_provenance() -> dict:
    """Which tree produced this capture.

    An A/A floor is routinely banked in a DIFFERENT gpuq slot from the arm it
    gates (see KVTAIL_PROBE_PLAN_0908.md "run between bases"), so "same tree,
    same corpus" cannot be a doc instruction -- it has to be recorded at
    capture time and enforced at compare time, or a floor taken against one
    harness silently gates an arm taken against another.
    """
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    out = {"commit": None, "dirty": None}
    try:
        out["commit"] = subprocess.run(
            ["git", "-C", here, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
        out["dirty"] = bool(
            subprocess.run(
                ["git", "-C", here, "status", "--porcelain"],
                capture_output=True, text=True, timeout=20, check=True,
            ).stdout.strip()
        )
    except Exception as exc:  # noqa: BLE001 - unknown provenance is recorded, never guessed
        out["error"] = repr(exc)
    return out


def provenance_mismatch(ref: dict, other: dict) -> Optional[str]:
    """Why ``other`` may not be compared against ``ref`` -- or None if it may.

    Deliberately strict and deliberately NOT overridable by a flag: the whole
    point is that a mismatched pair is unusable, and a flag to force it would
    be used exactly once, at 02:00, on the run that mattered.
    """
    for field, label in (
        ("corpus_sha256", "corpus"),
        ("harness_commit", "harness commit"),
    ):
        a, b = ref.get(field), other.get(field)
        if a is None or b is None:
            return (
                f"{label} not recorded in one of the two captures "
                f"(ref={a!r}, other={b!r}); a capture from before provenance "
                "was recorded cannot be paired with one from after"
            )
        if a != b:
            return f"{label} differs: ref={a} vs {b}"
    if ref.get("harness_dirty") or other.get("harness_dirty"):
        return (
            "one of the captures was taken against a DIRTY worktree, so its "
            "commit does not identify the code that ran"
        )
    if ref.get("kv_tail_env") != other.get("kv_tail_env"):
        # only meaningful for the A/A pair; arms are supposed to differ here
        return None
    return None


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def check_logprob_capability(url: str, timeout: float = 120.0) -> dict:
    """Prove, against the RUNNING server, that input top-k logprobs come back.

    A capture that silently produced empty ``input_top_logprobs`` would compare
    two arms on zero positions and print a perfect agreement.  This runs first
    and refuses the whole capture if the capability is not there.
    """
    meta = _post(
        f"{url}/generate",
        {
            "text": "The capital of France is Paris.",
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 5,
        },
        timeout,
    )["meta_info"]
    itl = meta.get("input_token_logprobs") or []
    itp = meta.get("input_top_logprobs") or []
    usable = [p for p in itp if p]
    ok = len(itl) > 1 and len(usable) > 0
    width = max((len(p) for p in usable), default=0)
    return {
        "endpoint": "native /generate",
        "input_positions": len(itl),
        "positions_with_top_logprobs": len(usable),
        "top_k_width_returned": width,
        "requested_top_k": 5,
        "ok": ok,
        "full_vocab_available": False,
        "note": (
            "Only top-k is returned; the KLD is TRUNCATED to the reference's "
            "top-k support and renormalised. The exact full-vocab quantity "
            "reported beside it is mean |delta logprob| on the actual token."
        ),
    }


def capture_teacher_forced(
    url: str, passages: List[dict], topk: int, timeout: float
) -> List[dict]:
    out = []
    for i, p in enumerate(passages):
        t0 = time.time()
        # SCORING WINDOW. Short passages are scored whole (start_len 0). A long
        # passage is scored over its LAST ``SCORE_WINDOW`` positions only --
        # deliberately, on two grounds: (a) that is where the body/tail split
        # actually bites, because only there does the model have ~100k tokens of
        # fp8 body behind it, while position 300 of a 100k prompt has almost no
        # body at all and would dilute the mean with positions the feature
        # cannot affect; (b) top-20 logprobs for 100k positions is ~60 MB of
        # JSON per passage per arm. The window is recorded per record, so the
        # denominator is never in doubt.
        start_len = p.get("logprob_start_len", 0)
        meta = _post(
            f"{url}/generate",
            {
                "text": p["text"],
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "logprob_start_len": start_len,
                "top_logprobs_num": topk,
            },
            timeout,
        )["meta_info"]
        rec = {
            "id": p["id"],
            "group": p["group"],
            "source": p.get("source"),
            "logprob_start_len": start_len,
            "measured_prompt_tokens": meta.get("prompt_tokens"),
            "scored_positions": len(meta.get("input_token_logprobs") or []),
            "cached_tokens": meta.get("cached_tokens"),
            "input_token_logprobs": meta.get("input_token_logprobs"),
            "input_top_logprobs": meta.get("input_top_logprobs"),
            "wall_s": round(time.time() - t0, 3),
        }
        out.append(rec)
        print(
            f"  [{i+1}/{len(passages)}] {p['id']:<28} "
            f"{rec['measured_prompt_tokens']} tok, "
            f"{rec['scored_positions']} scored  "
            f"cached={rec['cached_tokens']}  {rec['wall_s']:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return out


def run_determined(
    url: str, model: str, probes: List[dict], timeout: float
) -> List[dict]:
    out = []
    for i, pr in enumerate(probes):
        t0 = time.time()
        try:
            body = _post(
                f"{url}/v1/chat/completions",
                {
                    "model": model,
                    "messages": pr["messages"],
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": pr.get("max_tokens", 1024),
                    # Thinking OFF for the determined-answer instrument. With
                    # thinking on, a reasonmath probe spends ~2k tokens per
                    # answer and the arm becomes a generation-length probe
                    # rather than a quality probe. Uniform across arms.
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout,
            )
            text = body["choices"][0]["message"].get("content") or ""
            usage = body.get("usage", {})
        except Exception as exc:  # noqa: BLE001 - a failed probe is data
            text, usage = "", {"error": repr(exc)}
        rec = {
            "id": pr["id"],
            "kind": pr["kind"],
            "answer_text": text,
            "usage": usage,
            "wall_s": round(time.time() - t0, 3),
            "correct": verify_probe(pr, text),
        }
        out.append(rec)
        print(
            f"  [{i+1}/{len(probes)}] {pr['id']:<28} correct={rec['correct']} "
            f"{rec['wall_s']:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    return out


def verify_probe(probe: dict, text: str) -> Optional[bool]:
    """Determined-answer verification.  None = not checkable, never silently False."""
    kind = probe["kind"]
    if kind == "needle":
        want = probe["expect_token"]
        m = re.search(r"TOKEN:\s*([A-Za-z0-9\-]+)", text)
        if m:
            return m.group(1).strip() == want
        return want in text
    if kind == "reason_math":
        accepted = probe.get("accepted") or []
        norm = " ".join(text.split())
        return any(" ".join(a.split()) in norm for a in accepted)
    if kind == "data_extract":
        expected = probe.get("expected") or {}
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return False
        try:
            got = json.loads(m.group(0))
        except json.JSONDecodeError:
            return False
        return all(str(got.get(k)) == str(v) for k, v in expected.items())
    return None


# --------------------------------------------------------------------------
# Corpus construction
# --------------------------------------------------------------------------


def _sha256(path: str) -> str:
    return file_sha256(path)


def _load_pack(name: str) -> Tuple[dict, List[dict]]:
    rows = []
    meta = {}
    with open(os.path.join(PACK_DIR, name)) as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("__meta__"):
                meta = row
            else:
                rows.append(row)
    return meta, rows


def _render_messages(messages: List[dict], tokenizer=None) -> str:
    if tokenizer is not None:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001 - fall back and record it
            pass
    return "\n\n".join(f"{m['role']}:\n{m['content']}" for m in messages)


def build_corpus(model_path: Optional[str]) -> dict:
    tokenizer = None
    tokenizer_note = "none (character calibration)"
    if model_path:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            tokenizer_note = f"transformers AutoTokenizer from {model_path}"
        except Exception as exc:  # noqa: BLE001
            tokenizer_note = f"unavailable ({exc!r}); character calibration"

    manifest: List[dict] = []
    passages: List[dict] = []
    probes: List[dict] = []

    # ---- SHORT set: club-3090 quality-suite packs -------------------------
    for pack in SHORT_PACKS:
        path = os.path.join(PACK_DIR, pack)
        meta, rows = _load_pack(pack)
        manifest.append(
            {
                "kind": "quality-pack",
                "path": path,
                "sha256": _sha256(path),
                "bytes": os.path.getsize(path),
                "pack_id": meta.get("pack_id"),
                "license": meta.get("license"),
                "upstream_repo": meta.get("upstream_repo"),
                "upstream_commit": meta.get("upstream_commit"),
                "scenarios": len(rows),
            }
        )
        for row in rows:
            pid = f"{meta.get('pack_id')}/{row['id']}"
            passages.append(
                {
                    "id": pid,
                    "group": "short",
                    "source": path,
                    "text": _render_messages(row["messages"], tokenizer),
                }
            )
            asserts = (row.get("verifier") or {}).get("asserts") or []
            if meta.get("pack_id") == "reasonmath-15" and asserts:
                a = asserts[0]
                accepted = [a.get("canonical_answer")] + list(
                    a.get("accepted_answers") or []
                )
                probes.append(
                    {
                        "id": pid,
                        "kind": "reason_math",
                        "messages": row["messages"],
                        "accepted": [x for x in accepted if x],
                        "max_tokens": 2048,
                    }
                )
            elif meta.get("pack_id") == "dataextract-15" and row.get("expected"):
                probes.append(
                    {
                        "id": pid,
                        "kind": "data_extract",
                        "messages": row["messages"],
                        "expected": row["expected"],
                        "max_tokens": 1024,
                    }
                )

    # ---- LONG set: real document prefixes --------------------------------
    def n_tokens(s: str) -> int:
        if tokenizer is not None:
            return len(tokenizer(s, add_special_tokens=False)["input_ids"])
        return max(1, len(s) // 4)

    def cut_to_tokens(doc: str, target: int) -> str:
        if tokenizer is None:
            return doc[: target * 4]
        ids = tokenizer(doc, add_special_tokens=False)["input_ids"][:target]
        return tokenizer.decode(ids)

    for src_i, src in enumerate(LONG_SOURCES):
        if not os.path.exists(src):
            continue
        raw = open(src, encoding="utf-8", errors="replace").read()
        manifest.append(
            {
                "kind": "long-document",
                "path": src,
                "sha256": _sha256(src),
                "bytes": os.path.getsize(src),
                "first_line": raw.split("\n", 1)[0][:200],
            }
        )
        for target in LONG_TARGET_TOKENS:
            body = cut_to_tokens(raw, target)
            pid = f"long/{os.path.basename(src)}/{target}"
            passages.append(
                {
                    "id": pid,
                    "group": f"long-{target}",
                    "source": src,
                    "text": body,
                    "target_tokens": target,
                }
            )
            # Needle recall probe over the SAME real text.
            needle = NEEDLE_TEMPLATE.format(token=NEEDLE_TOKENS[src_i % len(NEEDLE_TOKENS)])
            off = int(len(body) * NEEDLE_OFFSET_FRACTION)
            haystack = body[:off] + needle + body[off:]
            probes.append(
                {
                    "id": f"needle/{os.path.basename(src)}/{target}",
                    "kind": "needle",
                    "messages": [
                        {"role": "user", "content": haystack + NEEDLE_QUESTION}
                    ],
                    "expect_token": NEEDLE_TOKENS[src_i % len(NEEDLE_TOKENS)],
                    "needle_char_offset": off,
                    "needle_offset_fraction": NEEDLE_OFFSET_FRACTION,
                    "max_tokens": 256,
                }
            )

    for p in passages:
        p["approx_tokens"] = n_tokens(p["text"])
        # Short passages whole; long passages over their last SCORE_WINDOW
        # positions (see capture_teacher_forced for why).
        p["logprob_start_len"] = (
            0
            if p["group"] == "short"
            else max(0, p["approx_tokens"] - SCORE_WINDOW)
        )

    return {
        "schema": 1,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": tokenizer_note,
        "manifest": manifest,
        "passages": passages,
        "probes": probes,
        "notes": [
            "No synthetic filler. Every long passage is a prefix of a real "
            "document on this box; every source is named in `manifest` with "
            "its sha256 and byte size.",
            "The needle probes are needle-in-a-haystack over that real text: "
            "exactly one inserted marker sentence, recorded with its offset.",
        ],
    }


# --------------------------------------------------------------------------
# Compare / report
# --------------------------------------------------------------------------


def _by_id(records: Iterable[dict]) -> Dict[str, dict]:
    return {r["id"]: r for r in records}


def compare_arms(ref: dict, arms: List[dict], ref2: Optional[dict]) -> dict:
    """Every arm against the reference, plus the A/A floor when ref2 is given."""
    report: Dict[str, object] = {
        "reference_arm": ref.get("arm"),
        "logprob_capability": ref.get("logprob_capability"),
        "arms": {},
    }

    def one(new: dict) -> dict:
        refmap = _by_id(ref["teacher_forced"])
        groups: Dict[str, Dict[str, List]] = {}
        for rec in new["teacher_forced"]:
            r = refmap.get(rec["id"])
            if r is None:
                continue
            if (
                r.get("scored_positions") != rec.get("scored_positions")
                or r.get("logprob_start_len") != rec.get("logprob_start_len")
                or r.get("measured_prompt_tokens") != rec.get("measured_prompt_tokens")
            ):
                # Different prompt length or a different scoring window means
                # the two records are not comparable. Never fold a mismatched
                # pair into a mean -- that is how a denominator lies.
                continue
            scored = score_pair(r, rec)
            g = groups.setdefault(
                rec["group"], {"kld": [], "top1": [], "top5": [], "dlogprob": []}
            )
            for k, v in scored.items():
                g[k].extend(v)
        out: Dict[str, object] = {"by_group": {}}
        pooled: Dict[str, List] = {"kld": [], "top1": [], "top5": [], "dlogprob": []}
        for gname, g in sorted(groups.items()):
            pooled["kld"].extend(g["kld"])
            pooled["top1"].extend(g["top1"])
            pooled["top5"].extend(g["top5"])
            pooled["dlogprob"].extend(g["dlogprob"])
            out["by_group"][gname] = _summarise(g)
        out["overall"] = _summarise(pooled)
        # determined answers
        rmap = _by_id(ref.get("determined") or [])
        checked = [
            r for r in (new.get("determined") or []) if r.get("correct") is not None
        ]
        out["determined"] = {
            "n_checked": len(checked),
            "n_correct": sum(1 for r in checked if r["correct"]),
            "ref_n_correct": sum(
                1
                for r in checked
                if (rmap.get(r["id"]) or {}).get("correct") is True
            ),
            "disagreements": [
                r["id"]
                for r in checked
                if (rmap.get(r["id"]) or {}).get("correct") != r["correct"]
            ],
        }
        out["probe_stats"] = new.get("probe_stats")
        return out

    report["provenance"] = {
        "harness_commit": ref.get("harness_commit"),
        "harness_dirty": ref.get("harness_dirty"),
        "corpus_sha256": ref.get("corpus_sha256"),
    }
    for arm in arms:
        name = arm.get("arm", "?")
        why = provenance_mismatch(ref, arm)
        if why is not None:
            report["arms"][name] = {"REFUSED": why, "by_group": {}, "overall": {}}
        else:
            report["arms"][name] = one(arm)

    if ref2 is None:
        report["A_vs_A_noise_floor"] = None
        report["decision_rule"] = (
            "NO A/A FLOOR CAPTURED. Every delta below is uninterpretable until "
            "the bf16 arm is captured twice. Do not read a verdict off this."
        )
        return report

    why = provenance_mismatch(ref, ref2)
    if why is not None:
        report["A_vs_A_noise_floor"] = None
        report["decision_rule"] = (
            "A/A FLOOR REFUSED -- " + why + ". The floor and the reference must "
            "come from the same harness and the same corpus, whether or not "
            "they were captured in the same gpuq slot. Re-capture, do not "
            "reinterpret."
        )
        return report

    report["A_vs_A_noise_floor"] = one(ref2)
    report["floor_geometry"] = {
        "same_boot": ref.get("boot_log") is not None
        and ref.get("boot_log") == ref2.get("boot_log"),
        "groups_covered": sorted(report["A_vs_A_noise_floor"]["by_group"]),
        "note": (
            "A floor only gates the GROUPS it covers. A short-set floor cannot "
            "be used to judge a long-context delta."
        ),
    }
    report["decision_rule"] = (
        "A gain is real only if it EXCEEDS the A/A floor on the same "
        "instrument and the same group. gain <= floor => do not build."
    )
    return report


def _summarise(g: Dict[str, List]) -> dict:
    def rate(bools: List[bool]) -> Dict[str, object]:
        if not bools:
            return {"n": 0, "rate": None}
        return {"n": len(bools), "hits": sum(bools), "rate": sum(bools) / len(bools)}

    return {
        "kld_truncated_nats": aggregate(g["kld"]),
        "delta_logprob_actual_token_nats_EXACT": aggregate(g["dlogprob"]),
        "top1_agreement": rate(g["top1"]),
        "top5_agreement": rate(g["top5"]),
    }


def print_report(rep: dict) -> None:
    cap = rep.get("logprob_capability") or {}
    print("=" * 78)
    print("#1243 step 1 -- fp8-body / bf16-tail quality probe")
    print("=" * 78)
    print(f"reference arm      : {rep.get('reference_arm')}")
    print(
        f"logprob instrument : {cap.get('endpoint')}, top-k width returned "
        f"{cap.get('top_k_width_returned')} of {cap.get('requested_top_k')} "
        f"requested; full vocab available: {cap.get('full_vocab_available')}"
    )
    print(f"  -> {cap.get('note')}")
    print()
    prov = rep.get("provenance") or {}
    print(
        f"harness             : {prov.get('harness_commit')} "
        f"dirty={prov.get('harness_dirty')}"
    )
    print(f"corpus sha256       : {prov.get('corpus_sha256')}")
    geo = rep.get("floor_geometry")
    if geo:
        print(
            f"floor geometry      : same_boot={geo['same_boot']} "
            f"groups={geo['groups_covered']}"
        )
        print(f"  -> {geo['note']}")
    print()
    floor = rep.get("A_vs_A_noise_floor")
    if floor is not None:
        print("--- A/A NOISE FLOOR (bf16 vs bf16, same flags, same prompts) ---")
        _print_arm("A/A floor", floor)
        print()
    else:
        print("!!! NO A/A FLOOR -- deltas below are uninterpretable !!!\n")
    for name, arm in rep["arms"].items():
        print(f"--- ARM {name} vs reference ---")
        if "REFUSED" in arm:
            print(f"  REFUSED: {arm['REFUSED']}")
        else:
            _print_arm(name, arm)
        print()
    print(f"DECISION RULE: {rep['decision_rule']}")


def _print_arm(name: str, arm: dict) -> None:
    for gname, g in arm["by_group"].items():
        k = g["kld_truncated_nats"]
        d = g["delta_logprob_actual_token_nats_EXACT"]
        t1 = g["top1_agreement"]
        t5 = g["top5_agreement"]
        print(f"  [{gname}]")
        print(
            f"    truncated KLD (nats)   mean {_f(k['mean'])} median "
            f"{_f(k['median'])} p95 {_f(k['p95'])}   over {k['n']} positions"
        )
        print(
            f"    |dlogprob| EXACT       mean {_f(d['mean'])} median "
            f"{_f(d['median'])}                     over {d['n']} positions"
        )
        print(
            f"    top-1 agreement        {_pct(t1)}   top-5 {_pct(t5)}"
        )
    det = arm.get("determined") or {}
    print(
        f"  determined answers     {det.get('n_correct')}/{det.get('n_checked')} "
        f"correct (reference {det.get('ref_n_correct')}/{det.get('n_checked')}); "
        f"disagreements: {det.get('disagreements')}"
    )
    ps = arm.get("probe_stats")
    if ps:
        print(
            f"  emulation self-report  tail_n={ps.get('tail_n')} "
            f"writes={ps.get('writes')} rounded_rows={ps.get('rounded_rows_total')} "
            f"residual_rows_max={ps.get('residual_rows_max')} "
            f"(residual MUST be 0; see kv_tail_probe.py)"
        )


def _f(x) -> str:
    return "     n/a" if x is None else f"{x:8.6f}"


def _pct(r: dict) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    return f"{r['hits']}/{r['n']} = {100.0*r['rate']:.3f} %"


# --------------------------------------------------------------------------
# self-test (hermetic: no server, no GPU, no model)
# --------------------------------------------------------------------------


def self_test() -> int:
    checks = 0
    fails = []

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(msg)

    # identical distributions -> KLD 0, top-1 and top-5 agree
    a = [[-0.1, 5, "a"], [-1.0, 7, "b"], [-3.0, 9, "c"]]
    ok(abs(truncated_kld(a, a)) < 1e-12, "KLD of a distribution with itself is 0")
    ok(topk_agreement(a, a, 1) is True, "top-1 agrees with itself")
    ok(topk_agreement(a, a, 5) is None, "top-5 needs 5 entries; returns None at 3")

    # a genuinely different distribution -> KLD > 0
    b = [[-3.0, 5, "a"], [-1.0, 7, "b"], [-0.1, 9, "c"]]
    k = truncated_kld(a, b)
    ok(k is not None and k > 0.5, f"KLD of a reordered distribution is large ({k})")
    ok(topk_agreement(a, b, 1) is False, "top-1 disagrees when the argmax moves")
    ok(topk_agreement(a, b, 3) is True, "top-3 still contains the ref argmax")

    # closed form: ref = [ln .5, ln .5], new = [ln .25, ln .75] over shared support
    ref = [[math.log(0.5), 1, "x"], [math.log(0.5), 2, "y"]]
    new = [[math.log(0.25), 1, "x"], [math.log(0.75), 2, "y"]]
    want = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    got = truncated_kld(ref, new)
    ok(abs(got - want) < 1e-12, f"closed-form KLD {got} != {want}")

    # renormalisation: scaling both sides by a constant must not move the KLD
    ref_s = [[lp - 2.0, t, s] for lp, t, s in ref]
    ok(
        abs(truncated_kld(ref_s, new) - want) < 1e-12,
        "KLD is invariant to an unnormalised reference",
    )

    # support of size < 2 -> None, not 0.0 (a fake agreement would dilute means)
    ok(truncated_kld(a, [[-0.1, 5, "a"]]) is None, "one shared token -> None")
    ok(truncated_kld([], a) is None, "empty reference -> None")
    ok(truncated_kld(a, []) is None, "empty new arm -> None")
    ok(topk_agreement([], a, 1) is None, "empty reference -> no agreement datum")

    # aggregation
    agg = aggregate([1.0, 2.0, 3.0, 4.0])
    ok(agg["n"] == 4, "aggregate counts its denominator")
    ok(agg["mean"] == 2.5 and agg["median"] == 2.5, "mean/median")
    ok(agg["max"] == 4.0, "max")
    ok(aggregate([])["n"] == 0 and aggregate([])["mean"] is None, "empty aggregate")
    ok(aggregate([1.0, None, 3.0])["n"] == 2, "None values are dropped, not counted")

    # score_pair skips positions the server left empty
    rp = {"input_top_logprobs": [None, a], "input_token_logprobs": [[None, 1, None], [-0.5, 2, None]]}
    np_ = {"input_top_logprobs": [None, b], "input_token_logprobs": [[None, 1, None], [-0.7, 2, None]]}
    sc = score_pair(rp, np_)
    ok(len(sc["kld"]) == 1, "score_pair scores only the usable position")
    ok(len(sc["dlogprob"]) == 1, "score_pair drops the None logprob position")
    ok(abs(sc["dlogprob"][0] - 0.2) < 1e-12, "|dlogprob| is exact")

    # verifiers
    ok(
        verify_probe({"kind": "needle", "expect_token": "QF7-2K9"}, "TOKEN: QF7-2K9")
        is True,
        "needle verifier accepts the exact token",
    )
    ok(
        verify_probe({"kind": "needle", "expect_token": "QF7-2K9"}, "TOKEN: NOPE")
        is False,
        "needle verifier rejects a wrong token",
    )
    ok(
        verify_probe(
            {"kind": "reason_math", "accepted": ["ANSWER: $35.98"]},
            "steps...\nANSWER: $35.98",
        )
        is True,
        "reason_math verifier accepts an accepted answer",
    )
    ok(
        verify_probe(
            {"kind": "data_extract", "expected": {"a": 1}}, 'noise {"a": 1} noise'
        )
        is True,
        "data_extract verifier accepts the expected object",
    )
    ok(
        verify_probe({"kind": "data_extract", "expected": {"a": 1}}, "not json")
        is False,
        "data_extract verifier rejects non-JSON",
    )
    ok(verify_probe({"kind": "unknown"}, "x") is None, "unknown kind is not checkable")

    for f in fails:
        print(f"FAIL: {f}")
    print(f"self-test: {checks - len(fails)}/{checks} checks passed")
    return 1 if fails else 0


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus")
    c.add_argument("--out", required=True)
    c.add_argument(
        "--model-path",
        default="/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed",
    )

    p = sub.add_parser("capture")
    p.add_argument("--arm", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default="http://127.0.0.1:30032")
    p.add_argument("--model", default="Qwen3.8-27B")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--groups", default="", help="comma list; empty = all")
    p.add_argument("--skip-determined", action="store_true")

    m = sub.add_parser("compare")
    m.add_argument("--ref", required=True)
    m.add_argument("--ref2", default=None, help="second bf16 run => A/A floor")
    m.add_argument("--arm", action="append", default=[])
    m.add_argument("--json-out", default=None)

    sub.add_parser("self-test")

    a = ap.parse_args()

    if a.cmd == "self-test":
        return self_test()

    if a.cmd == "corpus":
        corpus = build_corpus(a.model_path)
        with open(a.out, "w") as fh:
            json.dump(corpus, fh)
        print(
            f"wrote {a.out}: {len(corpus['passages'])} passages, "
            f"{len(corpus['probes'])} determined-answer probes, "
            f"{len(corpus['manifest'])} sources"
        )
        for src in corpus["manifest"]:
            print(f"  {src['kind']:<14} {src['path']}  sha256={src['sha256'][:16]}")
        return 0

    if a.cmd == "capture":
        corpus = json.load(open(a.corpus))
        prov = harness_provenance()
        print(
            f"harness commit {prov.get('commit')} dirty={prov.get('dirty')}; "
            f"corpus sha256 {file_sha256(a.corpus)}",
            file=sys.stderr,
        )
        if prov.get("dirty"):
            print(
                "WARNING: the worktree is DIRTY. This capture cannot be paired "
                "with any other capture (compare will refuse it) because its "
                "commit does not identify the code that ran.",
                file=sys.stderr,
            )
        cap = check_logprob_capability(a.url, a.timeout)
        print(json.dumps(cap, indent=1), file=sys.stderr)
        if not cap["ok"]:
            print(
                "REFUSED: the server did not return input top-k logprobs. A "
                "capture without them compares two arms on zero positions and "
                "prints a perfect agreement. Fix the endpoint, not the harness.",
                file=sys.stderr,
            )
            return 2
        want = [g for g in a.groups.split(",") if g] or None
        passages = [
            p for p in corpus["passages"] if want is None or p["group"] in want
        ]
        print(f"teacher-forced capture: {len(passages)} passages", file=sys.stderr)
        tf = capture_teacher_forced(a.url, passages, a.topk, a.timeout)
        det = []
        if not a.skip_determined:
            print(
                f"determined-answer probes: {len(corpus['probes'])}", file=sys.stderr
            )
            det = run_determined(a.url, a.model, corpus["probes"], a.timeout)
        stats = None
        try:
            with urllib.request.urlopen(
                f"{a.url}/kvtail_probe_stats", timeout=10
            ) as r:
                stats = json.loads(r.read())
        except Exception:  # noqa: BLE001 - endpoint is optional
            stats = None
        with open(a.out, "w") as fh:
            json.dump(
                {
                    "arm": a.arm,
                    "url": a.url,
                    "topk": a.topk,
                    "captured_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "corpus_built_at": corpus.get("built_at"),
                    "corpus_sha256": file_sha256(a.corpus),
                    "harness_commit": prov.get("commit"),
                    "harness_dirty": prov.get("dirty"),
                    "harness_provenance": prov,
                    "logprob_capability": cap,
                    "teacher_forced": tf,
                    "determined": det,
                    "probe_stats": stats,
                    "env": {
                        k: v
                        for k, v in os.environ.items()
                        if k.startswith("SGLANG_KV_TAIL_PROBE")
                    },
                },
                fh,
            )
        print(f"wrote {a.out}")
        return 0

    if a.cmd == "compare":
        ref = json.load(open(a.ref))
        ref2 = json.load(open(a.ref2)) if a.ref2 else None
        arms = [json.load(open(p)) for p in a.arm]
        rep = compare_arms(ref, arms, ref2)
        print_report(rep)
        if a.json_out:
            with open(a.json_out, "w") as fh:
                json.dump(rep, fh, indent=1)
            print(f"\nwrote {a.json_out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
