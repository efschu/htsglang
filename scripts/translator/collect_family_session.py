# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Read-only harvest of a live family session, and the analysis it feeds.

WHY A SCRIPT AND NOT A LOOK: the numbers the restart bundle is waiting on are
distributions, and a distribution read by eye from a log is an anecdote. This
takes every live session's decision log, transcript and metrics through ONE
read-only pass and prints the three things the next decisions need:

  1. WITHIN-SPEAKER COSINE. The match threshold is 0.637 and it was calibrated
     on the PRESET POOL, not on phone audio of real people. Every decision
     record carries the full candidate ranking, so the distribution of
     same-speaker scores can be re-derived from the log alone -- which is what
     the threshold has to be set from.
  2. THE CHILD, SEPARATELY. A four-year-old speaking clear German is reported
     as Spanish by the LID. His turns are split out with language and
     confidence, because the question they answer is whether a confidence gate
     plus the pin prior is enough or whether a session language lock is
     required. A high-confidence misdetection cannot be reached by a gate, and
     only the data says which one this is.
  3. WHAT THE NEW PATH ACTUALLY DID. Sticky pin, overlap discards and stage
     deadlines are all live for the first time; the counts say whether they
     fired at all, which is the difference between a mechanism and a mechanism
     with reach (#493).

STRICTLY READ-ONLY. It opens no session, sends no control frame, and touches
no process -- it is safe to run while the family is testing.

    /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/collect_family_session.py
    ... --json-out /tmp/family.json     # keep the raw harvest
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from typing import Dict, List, Optional

DEFAULT_BASE = "http://192.168.0.101:30800"


def get(url: str, timeout: float = 8.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_text(url: str, timeout: float = 8.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def harvest(base: str) -> Dict[str, object]:
    health = get(f"{base}/api/translator/health")
    out: Dict[str, object] = {"health": health, "sessions": {}}
    for detail in health.get("session_detail", []):
        sid = detail["session_id"]
        entry: Dict[str, object] = {"detail": detail}
        for name, path in (
            ("decisions", f"/api/translator/sessions/{sid}/speaker-decisions"),
            ("transcript", f"/api/translator/sessions/{sid}/transcript"),
        ):
            try:
                entry[name] = get(f"{base}{path}")
            except Exception as exc:  # noqa: BLE001 - a gone session is normal
                entry[name] = {"error": str(exc)}
        out["sessions"][sid] = entry
    try:
        out["metrics"] = {
            line.split(" ")[0]: float(line.split(" ")[1])
            for line in get_text(f"{base}/metrics").splitlines()
            if line.startswith("translator_") and " " in line
        }
    except Exception as exc:  # noqa: BLE001
        out["metrics"] = {"error": str(exc)}
    return out


def within_speaker_scores(decisions: List[dict]) -> Dict[str, List[float]]:
    """Cosines a segment scored against the speaker it was ASSIGNED to.

    Read off `candidates`, not off `score`: `score` is 1.0 for a mint and for
    a manual attribution, and counting those would report a distribution of
    the bookkeeping rather than of the embedder.
    """
    per: Dict[str, List[float]] = {}
    for record in decisions:
        sid = record.get("speaker_id")
        for candidate in record.get("candidates") or []:
            if len(candidate) == 2 and candidate[0] == sid:
                per.setdefault(sid, []).append(float(candidate[1]))
    return per


def summarise(values: List[float]) -> str:
    if not values:
        return "n=0"
    ordered = sorted(values)
    p05 = ordered[max(0, int(len(ordered) * 0.05) - 1)]
    return (
        f"n={len(values)} min={ordered[0]:.3f} p05={p05:.3f} "
        f"median={statistics.median(ordered):.3f} max={ordered[-1]:.3f}"
    )


def report(data: Dict[str, object], child_speaker: Optional[str]) -> None:
    health = data["health"]
    print(f"[collect] tenant up {health.get('uptime_s', 0):.0f}s, "
          f"{health.get('sessions')} session(s)")

    all_scores: List[float] = []
    lid_rows: List[tuple] = []
    interjections = 0
    pinned = 0
    total = 0
    for sid, entry in data["sessions"].items():  # type: ignore[union-attr]
        decisions = (entry.get("decisions") or {}).get("decisions") or []
        lines = (entry.get("transcript") or {}).get("lines") or []
        if not decisions and not lines:
            continue
        print(f"\n[collect] session {sid}: {len(decisions)} decisions, "
              f"{len(lines)} lines")
        per = within_speaker_scores(decisions)
        for speaker, values in sorted(per.items()):
            print(f"           {speaker}: {summarise(values)}")
            all_scores.extend(values)
        for record in decisions:
            total += 1
            if record.get("pin"):
                pinned += 1
            if record.get("overlapped"):
                interjections += 1
            lid_rows.append((
                record.get("speaker_id"),
                record.get("language") or "?",
                float(record.get("language_confidence") or 0.0),
                float(record.get("segment_s") or 0.0),
            ))
        for line in lines:
            if not line.get("translations"):
                print(f"           UNFINISHED line {line['line_id']}: "
                      f"{line.get('source_text', '')[:60]!r}")

    print("\n[collect] WITHIN-SPEAKER COSINE, pooled")
    print(f"           {summarise(all_scores)}")
    if all_scores:
        below = [v for v in all_scores if v < 0.637]
        print(f"           below the 0.637 bar: {len(below)} of "
              f"{len(all_scores)} -- each one is a phantom speaker")
        ordered = sorted(all_scores)
        # A threshold PROPOSAL, stated as what it is: the 5th percentile of
        # observed same-speaker scores, which keeps 95% of real continuations
        # together. Not a recommendation until n is large enough to mean it.
        p05 = ordered[max(0, int(len(ordered) * 0.05) - 1)]
        verdict = "PROPOSAL" if len(all_scores) >= 15 else "TOO FEW SAMPLES"
        print(f"           {verdict}: match_threshold ~ {p05:.3f} "
              f"(5th percentile, n={len(all_scores)}, need n>=15)")

    print("\n[collect] LID, per decision (the child question)")
    for speaker, language, confidence, seconds in lid_rows:
        mark = ""
        if child_speaker and speaker == child_speaker:
            mark = "  <-- CHILD"
        flag = "  LOW" if confidence < 0.5 else ""
        print(f"           {speaker:<20} {language:<4} conf={confidence:.3f} "
              f"{seconds:5.2f}s{flag}{mark}")
    if lid_rows:
        low = [row for row in lid_rows if row[2] < 0.5]
        print(f"           low-confidence: {len(low)} of {len(lid_rows)} -- "
              "a confidence gate can only reach these; a HIGH-confidence "
              "misdetection needs the pin prior or a session lock")

    print("\n[collect] NEW PATH, did it bind at all")
    print(f"           decisions with a pin held : {pinned} of {total}")
    print(f"           segments marked overlapped: {interjections} of {total}")
    metrics = data.get("metrics") or {}
    if isinstance(metrics, dict) and "error" not in metrics:
        for key in sorted(metrics):
            if any(k in key for k in ("turns", "first_audio", "tts", "queue")):
                print(f"           {key} = {metrics[key]}")
    else:
        print(f"           metrics unavailable: {metrics}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--child-speaker", default=None,
                        help="speaker id of the child, to split his turns out")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    data = harvest(args.base)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        print(f"[collect] raw harvest -> {args.json_out}")
    report(data, args.child_speaker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
