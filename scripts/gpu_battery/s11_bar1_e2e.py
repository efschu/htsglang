#!/usr/bin/env python3
"""s11 -- compose bar1_e2e.json from what the host run left behind.

The step script talks to the host; this reads the files it brought back and
turns them into ONE artifact with a decidable content. Nothing here touches a
card, a socket or ssh, which is what makes it testable against fixtures.

Three extractions carry the step:

  * the graph gate. bar1_graph_check.py prints one PASSED/FAILED line per
    case with a [Gate]/[Info] marker and exits 0 only when every gate case
    passed. Both are recorded: the exit code alone would hide WHICH case fell.
  * per-group attainment. parallel_state logs "HTCCL enabled for group '<x>':
    requested=<a>, ACHIEVED=<e>" on success and the same pair as a WARNING on
    fallback. The requested name is worthless -- it says bar1 either way. Only
    ACHIEVED counts, and it counts PER GROUP: with SGLANG_UNEVEN_DCP=1 there
    are two (tp:0, dcp:0), and a run where one of them fell back to gloo is a
    mixed measurement, not a bar1 measurement.
  * the coverage bolt. htccl._select raises rather than falling back to the
    host-staged gloo level during a graph capture. It is extracted as its own
    field, with op and size, because it is the expected failure of the current
    integration and needs to be distinguishable from any other crash.
  * the smoke, over /generate. It used to go through /v1/chat/completions,
    where two things went wrong at once on 2026-07-30: the chat template
    answered a temperature-0 request with a thinking preamble and spent the
    whole token budget on it, and `meta_info` is opt-in there
    (`return_meta_info`, default False), so spec_accept_length could not be
    read at all. A continuation prompt has no template and /generate always
    carries meta_info. The full reasoning sits at the request in
    s11_bar1_e2e.sh; what matters here is that the parser reads the shape the
    step really asks for.

WHICH FILE THE EVIDENCE IS READ FROM, and why that is not a detail. This used
to read `htccl_lines.txt` alone -- the grep result the step script harvests
AFTER the server has answered. On every early exit (server never came up,
capture aborted) the shell writes only `server.log` and jumps to compose, so
that one file did not exist. `read_lines` returns [] for a missing file, so
the artifact reported "no group line, no bolt, no fatal" for a run whose log
carried all three, and the check then failed on the CONSEQUENCE (no ERREICHT
line) instead of the CAUSE (the capture aborted). Exactly the s01 pattern:
a loader that reads a shape the producer does not write, and is silently
empty rather than loud.

Both files are read now, and which ones existed is recorded (`log_quellen`).
"nobody harvested a log" and "the log holds nothing" are then two different
answers rather than one empty list. The two files overlap -- the grep result
carries grep's "<lineno>:" prefix, the tail does not -- so lines are
deduplicated on their content, without which `aufbau_lines` would count the
same setup line twice.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

KIND = "bar1_e2e"
#: 2: `log_quellen` / `log_zeilen` were added, and the evidence is no longer
#: read from `htccl_lines.txt` alone. An older artifact must not slip through
#: here -- it does not carry the fields the check uses to tell "nobody
#: looked" from "nothing was found".
#: 3: the smoke goes over /generate instead of /v1/chat/completions, and
#: `smoke` therefore carries `endpunkt`, `finish_reason`, `zahlen_erwartet`,
#: `spec_verify_ct` and `unterprovisioniert`. Same reason: without
#: `endpunkt` a check cannot read WHAT the coherence number refers to.
#: 4: the criterion no longer measures obedience but whether the language
#: model is intact -- `anker_zahlen`, `muell_befunde`, `lm_intakt` and the
#: metrics behind them. A schema-3 artifact does not carry those fields, and
#: its `kohaerent` meant something other than the field of that name here.
#: 5: the per-group entries under `gruppen` renamed their keys from German to
#: English -- `gruppe`/`angefordert`/`erreicht` became `group`/`requested`/
#: `achieved`, in step with htccl.py's report_state(). A schema-4 artifact
#: still spells them in German, so every group would read back as empty and
#: the transport check would pass on nothing. Rejecting it by version is the
#: point: re-run the step rather than read a stale artifact through the new
#: names.
SCHEMA_VERSION = 5

#: The continuation prompt s11_bar1_e2e.sh sends to /generate. It lives here
#: AND there; test_gpu_battery_checks_bar1.py pins the two together against
#: the source of the step script -- otherwise the parser would count a
#: sequence the request never asked for.
SMOKE_PROMPT = "1 2 3 4"
#: The first number the continuation has to produce -- immediately after the
#: prompt. What the prompt itself says is no evidence about the answer.
ZAHLEN_VON = 5
ZAHLEN_BIS = 20

#: How many numbers must follow IMMEDIATELY and WITHOUT A GAP.
#:
#: This used to be 15, and it measured the wrong thing. Attempt 4 counted
#: " 5 6 7 8 9 10" on correctly and then drifted into a coherent Russian
#: forum post -- the raw continuation characteristic of a base path with no
#: instruction, not damage to the model. The 15 demanded obedience over 16
#: numbers; what this step wants to know is something else: does a determined
#: prefix produce the RIGHT tokens, and is the rest well-formed text? Real
#: corruption looks different -- it does not deliver six correct numbers and
#: then non-sequitur garbage.
#:
#: 4 and not 6, even though 6 were measured: a single observation does not
#: justify a threshold at its own value. The distance that matters is the one
#: between 0 and 4 -- a broken collective does not deliver four correct
#: numbers and then come off the rails.
ANKER_MIN = 4

#: Garbage thresholds. EVERY one is calibrated against the real artifact of
#: attempt 4 (smoke.json, 1055 characters), not guessed -- the measured values
#: are noted with each. A test drives exactly that artifact and requires it to
#: pass.
#: measured 1.0000
MUELL_DRUCKBAR_MIN = 0.98
#: measured 0.435 (the forum post repeats a block; that is web text, not a
#: defect). A token loop sits at ~0.01 -- the distance is large, so the
#: threshold is deliberately far below the measured value.
MUELL_VIELFALT_MIN = 0.15
#: Below that few words the diversity is noise and is not checked.
MUELL_VIELFALT_AB_WORTEN = 30
#: measured 3 ("###"). A degeneration repeats a short unit dozens of times.
MUELL_WDH_MAX = 10
MUELL_EINHEIT_MAX = 32
#: Shorter than this is not an answer anything can be said about.
MUELL_MIN_ZEICHEN = 20

#: Where the evidence is read from, in this order. `htccl_lines.txt` is the
#: complete grep result and therefore comes first; `server.log` is the bounded
#: excerpt the step script writes on EVERY path -- including the ones that
#: branch off before the grep.
LOG_QUELLEN = ("htccl_lines.txt", "server.log")

RE_GROUP = re.compile(
    r"group '(?P<group>[^']+)': requested=(?P<requested>[^,\s]+),\s*"
    r"ACHIEVED=(?P<achieved>[A-Za-z0-9_\-]+)"
)
#: #315: these three used to spell "Aufbau"/"Kasse"/"waehrend einer
#: CUDA-Graph-Aufzeichnung" -- the German wording #295 moved htccl_bar1.py and
#: htccl.py away from. Dead on every real run since, hidden by fixtures that
#: were never re-captured either; see test_bar1_marker_coupling.py, which
#: checks these regexes against the actual emitter source so the next rename
#: fails loudly instead of quietly.
RE_KASSE = re.compile(r"BAR1 ledger of this card after group '(?P<group>[^']+)'")
RE_AUFBAU = re.compile(r"HTCCL-BAR1: setup in\s+(?P<ms>[0-9.]+)\s*ms")
RE_RIEGEL = re.compile(
    r"HTCCL: '(?P<op>[A-Za-z0-9_]+)' with (?P<bytes>\d+) bytes during a "
    r"CUDA graph capture"
)
RE_GATE_CASE = re.compile(
    r"^\s*(?P<marke>PASSED|FAILED)\s*\[(?P<art>Gate|Info)\]\s*(?P<name>\S+)"
)

FATAL_MARKERS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "NCCL error",
    "Watchdog caught collective operation timeout",
    # A boot killed by a plain traceback -- no OOM, no NCCL error -- is just as
    # dead. The shell already harvests this marker (s11_bar1_e2e.sh) and
    # check_common.FATAL_MARKERS carries it for every other step.
    "Traceback (most recent call last)",
)

# Lines the server QUOTED from a helper subprocess it ran and recovered from
# (the stage-0 hardware probe). They carry the emitter's marker, and a fatal
# marker inside one is evidence about the helper, not about this boot -- see
# check_common.QUOTED_SUBLOG_PREFIX, keep the literals in step.
QUOTED_SUBLOG_PREFIX = "[probe-subprocess] "


def read_lines(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, errors="replace") as f:
        return [line.rstrip("\n") for line in f]


def parse_graph_check(step_dir: str) -> dict:
    lines = read_lines(os.path.join(step_dir, "graph_check.txt"))
    rc_raw = read_lines(os.path.join(step_dir, "graph_check_rc.txt"))
    try:
        rc = int(rc_raw[0].strip()) if rc_raw else None
    except ValueError:
        rc = None
    cases = []
    for line in lines:
        m = RE_GATE_CASE.match(line)
        if m:
            cases.append(
                {
                    "name": m.group("name"),
                    "gate": m.group("art") == "Gate",
                    "ok": m.group("marke") == "PASSED",
                }
            )
    gates = [c for c in cases if c["gate"]]
    failed = [c["name"] for c in gates if not c["ok"]]
    return {
        "rc": rc,
        "cases": len(cases),
        "gate_cases": len(gates),
        "gefallen": failed,
        "alle_bestanden": bool(gates) and not failed and rc == 0,
        # #315: bar1_graph_check.py's header was "Zusammenfassung", moved to
        # "Summary" together with PASSED/FAILED in commit 896a443222.
        "zusammenfassung_vorhanden": any("Summary" in line for line in lines),
    }


def _without_grep_prefix(line: str) -> str:
    """The content of a line, regardless of who harvested it.

    ``grep -n`` puts "<lineno>:" in front, ``tail`` does not. The same log
    line therefore looks different in the two sources even though it is the
    same one -- without this normalisation `aufbau_lines` counted it twice.
    """
    return " ".join(re.sub(r"^\d+:", "", line).split())


def collect_log_lines(step_dir: str) -> tuple:
    """All harvested log lines, deduplicated, plus the sources that existed.

    The source list is the actual result: with it, an empty line list means
    "nothing found"; without it, "nobody looked". Those are two findings, and
    only one of them is a measurement.
    """
    sources = []
    lines = []
    seen = set()
    for name in LOG_QUELLEN:
        path = os.path.join(step_dir, name)
        if not os.path.exists(path):
            continue
        sources.append(name)
        for line in read_lines(path):
            key = _without_grep_prefix(line)
            if not key or key in seen:
                continue
            seen.add(key)
            lines.append(line)
    return lines, sources


def parse_log_evidence(step_dir: str) -> dict:
    lines, sources = collect_log_lines(step_dir)
    groups: dict = {}
    ledger_groups = []
    setup_ms = []
    bolt = None
    fatal = None
    for line in lines:
        m = RE_GROUP.search(line)
        if m:
            groups[m.group("group")] = {
                "group": m.group("group"),
                "requested": m.group("requested"),
                "achieved": m.group("achieved"),
            }
        m = RE_KASSE.search(line)
        if m and m.group("group") not in ledger_groups:
            ledger_groups.append(m.group("group"))
        m = RE_AUFBAU.search(line)
        if m:
            setup_ms.append(float(m.group("ms")))
        m = RE_RIEGEL.search(line)
        if m and bolt is None:
            bolt = {
                "op": m.group("op"),
                "bytes": int(m.group("bytes")),
                "zeile": " ".join(line.split())[:300],
            }
        if fatal is None and QUOTED_SUBLOG_PREFIX not in line:
            for marker in FATAL_MARKERS:
                if marker in line:
                    fatal = " ".join(line.split())[:300]
                    break
    return {
        "gruppen": sorted(groups.values(), key=lambda g: g["group"]),
        "aufbau_gruppen": ledger_groups,
        "aufbau_lines": len(setup_ms),
        "aufbau_ms": setup_ms,
        "riegel": bolt,
        "fatal": fatal,
        "log_quellen": sources,
        "log_zeilen": len(lines),
    }


def count_in_order(text: str, start: int, end: int) -> int:
    """How many of the numbers ``start..end`` appear SOMEWHERE in that order.

    The old counter, kept only for the chat shape (which ends in a STOP
    anyway). Why it does not work as a criterion: it says nothing about
    WHERE the hits come from. On 2026-07-30 it reported 3 for an answer that
    never counted at all -- the hits were the bullet points "1.", "2.", "3."
    of a thinking preamble. And in attempt 4 it reported 10 for an answer
    whose first six numbers were right and whose remainder is Russian forum
    text: the other four hits are digits out of "220В", "1450" and dates. A
    counter that scatters over the whole text always finds something in
    prose.

    The criterion is therefore :func:`anchor_run` -- immediate and without
    gaps.
    """
    pos = 0
    hits = 0
    for n in range(start, end + 1):
        at = text.find(str(n), pos)
        if at < 0:
            continue
        hits += 1
        pos = at + len(str(n))
    return hits


def anchor_run(text: str, start: int) -> tuple:
    """``(count, remainder)`` -- the numbers IMMEDIATELY at the start.

    Counts how many numbers from ``start`` on stand right at the front, in
    order and without a gap, separated only by whitespace. The first break
    ends it; whatever follows is the remainder and is not searched for
    numbers any more.

    **This is the difference between "obeyed" and "computes correctly".** A
    base path with no instruction continues a number sequence for a while and
    then drifts into whatever the prefix reminds it of -- that is the
    characteristic of a raw continuation, not a defect. What the anchor
    checks is the one question this step can answer: does a determined prefix
    produce the right tokens? A shot-up collective does not deliver them.

    A scattering search would be wrong here: "0/10" right behind the 10 (that
    is how the artifact of attempt 4 reads) must not pass as an 11 just
    because an 11 shows up somewhere later.
    """
    n = start
    rest = text
    hits = 0
    while True:
        m = re.match(r"\s*(\d+)", rest)
        if m is None or int(m.group(1)) != n:
            break
        hits += 1
        n += 1
        rest = rest[m.end():]
    return hits, rest


def _max_repetition(text: str) -> int:
    """How often a short unit repeats IMMEDIATELY back to back.

    The garbage test that separates degeneration from web text. A token loop
    repeats the same few characters dozens of times in a row; a forum post
    that carries a quoted block a second time does not do so immediately and
    not in a short unit. Measured against the artifact of attempt 4: 3
    ("###").
    """
    text = text[:4000]
    best = 0
    for length in range(1, MUELL_EINHEIT_MAX + 1):
        i = 0
        while i + length <= len(text):
            unit = text[i:i + length]
            n = 1
            while text[i + n * length:i + (n + 1) * length] == unit:
                n += 1
                if n > 64:            # enough to call it degenerate
                    return n
            if n > best:
                best = n
            i += 1
    return best


def garbage_check(text: str) -> tuple:
    """``(findings, metrics)`` -- is the text well-formed, or garbage?

    Three questions, deliberately blunt and with no opinion about WHAT the
    text talks about. What is NOT checked here is sense: a Russian forum post
    about three-phase motors is a perfectly intact language-model result,
    even if nobody ordered it.

    Every threshold is calibrated against the real artifact of attempt 4; the
    measured values are noted at the constants.

    The finding strings themselves stay German: out-of-scope unit tests match
    on "druckbare", "Wortvielfalt" and "Tokenschleife".
    """
    findings = []
    metrics = {}
    trimmed = text.strip()
    if len(trimmed) < MUELL_MIN_ZEICHEN:
        findings.append(f"nur {len(trimmed)} Zeichen Text")
        return findings, metrics

    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    share = printable / len(text)
    metrics["druckbar_anteil"] = round(share, 4)
    if share < MUELL_DRUCKBAR_MIN:
        findings.append(
            f"nur {share:.3f} druckbare Zeichen (< {MUELL_DRUCKBAR_MIN})"
        )

    words = text.split()
    if len(words) >= MUELL_VIELFALT_AB_WORTEN:
        diversity = len(set(words)) / len(words)
        metrics["wort_vielfalt"] = round(diversity, 4)
        if diversity < MUELL_VIELFALT_MIN:
            findings.append(
                f"Wortvielfalt {diversity:.3f} (< {MUELL_VIELFALT_MIN}) -- "
                f"{len(set(words))} verschiedene von {len(words)} Worten"
            )

    repeats = _max_repetition(text)
    metrics["max_wiederholung"] = repeats
    if repeats >= MUELL_WDH_MAX:
        findings.append(
            f"eine kurze Einheit wiederholt sich {repeats}x unmittelbar "
            f"(>= {MUELL_WDH_MAX}) -- Tokenschleife"
        )
    return findings, metrics


def parse_smoke(step_dir: str) -> dict:
    """Coherence, mechanically -- out of the answer the step really fetches.

    What is read is the /generate shape (``{"text": ..., "meta_info":
    {...}}``). The chat shape is still understood, so that an artifact from
    an older run does not pass as "unreadable" but as what it is -- but it is
    no longer what the step asks for. The reasoning sits at the request in
    s11_bar1_e2e.sh.

    ``spec_accept_length`` comes from ``meta_info`` and EXPLICITLY not from
    ``spec_ema_accept_len``: that is a smoothed curve, not this request's
    acceptance length, and confusing the two is a known measurement trap.
    """
    path = os.path.join(step_dir, "smoke.json")
    out: dict = {
        "vorhanden": os.path.exists(path),
        "endpunkt": None,
        "content_prefix": None,
        "spec_accept_length": None,
        "spec_verify_ct": None,
        "finish_reason": None,
        "zahlen_in_folge": 0,
        "zahlen_erwartet": ZAHLEN_BIS - ZAHLEN_VON + 1,
        "anker_zahlen": 0,
        "anker_min": ANKER_MIN,
        "drift_zeichen": 0,
        "muell_befunde": [],
        "lm_intakt": False,
        "kohaerent": False,
        "unterprovisioniert": False,
        "error": None,
    }
    if not out["vorhanden"]:
        return out
    try:
        with open(path, errors="replace") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        out["error"] = f"smoke.json not readable: {exc}"
        return out
    if not isinstance(payload, dict):
        out["error"] = "smoke.json is not a JSON object"
        return out
    if payload.get("error") or payload.get("object") == "error":
        out["error"] = " ".join(str(payload.get("error") or payload).split())[:200]
        return out

    meta: dict = {}
    if isinstance(payload.get("text"), str):
        # /generate: text and meta_info sit at the top, no ifs and buts.
        out["endpunkt"] = "generate"
        text = payload["text"]
        meta = payload.get("meta_info") or {}
        start, end = ZAHLEN_VON, ZAHLEN_BIS
    else:
        # Chat shape. It counts from 1, because no prompt is continued there
        # -- an older artifact should yield the same number it did back then.
        out["endpunkt"] = "chat"
        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"] or ""
            meta = choice.get("meta_info") or {}
        except (KeyError, IndexError, TypeError) as exc:
            out["error"] = (
                f"answer is neither /generate (text) nor chat "
                f"(choices[0].message.content): {exc}"
            )
            return out
        start, end = 1, 20
        out["zahlen_erwartet"] = end - start + 1

    out["content_prefix"] = text[:300]
    out["spec_accept_length"] = meta.get("spec_accept_length")
    out["spec_verify_ct"] = meta.get("spec_verify_ct")
    finish = meta.get("finish_reason")
    if isinstance(finish, dict):
        finish = finish.get("type")
    if finish is None and out["endpunkt"] == "chat":
        try:
            finish = payload["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            finish = None
    out["finish_reason"] = finish

    # The scattering count stays in the artifact as a METRIC -- it is the
    # number both wrong conclusions hang on (preamble bullets, digits out of
    # prose), and whoever sees it in the log later should be able to find it
    # again. It is no longer the criterion.
    out["zahlen_in_folge"] = count_in_order(text, start, end)

    # (a) The anchor: does a determined prefix produce the right tokens?
    hits, rest = anchor_run(text, start)
    out["anker_zahlen"] = hits
    out["anker_min"] = ANKER_MIN
    out["drift_zeichen"] = len(rest.strip())
    anchor_ok = hits >= ANKER_MIN

    # (b) And is the rest well-formed text? The WHOLE section is checked, not
    #     just the drift: an answer that carries exactly the numbers and then
    #     stops has no drift and is fine all the same.
    findings, metrics = garbage_check(text)
    out["muell_befunde"] = findings
    out.update(metrics)

    out["lm_intakt"] = bool(anchor_ok and not findings)
    # `kohaerent` keeps its name in the artifact because the check and the
    # analysis carry it -- but it now means "LM intact" and no longer "it
    # obeyed". That is the change this is about.
    out["kohaerent"] = out["lm_intakt"]

    # The named state: coherent text, the token budget spent to the stop, and
    # the numbers still never came. That is not a transport fault and not
    # corruption but a smoke whose budget went elsewhere -- the 2026-07-30
    # case (thinking preamble). It now requires that the ANCHOR was missing:
    # if the answer only drifts AFTER correctly continued numbers, it is not
    # an under-provisioned smoke but a passing one.
    out["unterprovisioniert"] = bool(
        not anchor_ok
        and not findings
        and finish == "length"
        and len(text.strip()) >= 200
    )
    return out


def compose(step_dir: str, port: int, host_log: str) -> dict:
    unreachable = os.path.exists(os.path.join(step_dir, "host_unreachable.txt"))
    missing_integration = os.path.exists(
        os.path.join(step_dir, "integration_missing.txt")
    )
    blocked_lines = read_lines(os.path.join(step_dir, "blocked.txt"))
    payload: dict = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "host": os.environ.get("BAR1_HOST", ""),
        "reachable": not unreachable,
        "integration_present": not missing_integration,
        # A step that never got the cards must not be diagnosed through the
        # empty artifacts it left behind.
        "blocked": " ".join(" ".join(blocked_lines).split())[:200] or None,
        "port": port,
        "server_log_remote": host_log,
        "transport_angefordert": "bar1",
        "graph_freigabe": True,
        "uneven_dcp": True,
        "graph_check": parse_graph_check(step_dir),
        "smoke": parse_smoke(step_dir),
    }
    payload.update(parse_log_evidence(step_dir))
    payload["gruppen_bar1"] = sorted(
        g["group"] for g in payload["gruppen"] if g["achieved"] == "bar1"
    )
    payload["gruppen_ausgewichen"] = sorted(
        g["group"] for g in payload["gruppen"] if g["achieved"] != g["requested"]
    )
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--host-log", default="")
    args = ap.parse_args()

    payload = compose(args.step_dir, args.port, args.host_log)
    out = os.path.join(args.step_dir, "bar1_e2e.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"bar1_e2e.json written: groups on bar1={payload['gruppen_bar1']}, "
        f"fell back={payload['gruppen_ausgewichen']}, "
        f"setup lines={payload['aufbau_lines']}, "
        f"bolt={'yes' if payload['riegel'] else 'no'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
