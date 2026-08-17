"""Phase-locked D2 injector: the failing arm the #713 acceptance did not have.

WHY THIS EXISTS
---------------
The #713(a) boot acceptance passed every criterion, and then the CONTROL build
-- which does not contain the fix -- passed them all too. An acceptance
criterion the control also satisfies is not an acceptance criterion: it cannot
fail, so its passing says nothing about the patch. The defect had simply taken
the day off. This harness is built so that it CANNOT quietly pass: it names,
below, the arm that is expected to FAIL, and it reports the phase-resolved
distribution rather than an average that would hide a bimodal result.

WHAT IT AIMS AT (revised 2026-08-17 from recorded evidence)
-----------------------------------------------------------
The original model was "a request admitted late inside a PP window is
resident-but-unprefilled at the next evaluation, and is therefore invisible".
Reading the recorded rotations falsified the interesting half of that and
replaced it with something sharper and much easier to hit: a FLIP PING-PONG.

Verbatim, one 22-token request, /spinning/evidence-665-f1/boot_bundle.log
.20260817T070900Z:

  06:53:56 arming tp_to_pp: IDLE-LOCKED ... (0 req resident, 22 tok pending)
  06:53:59 cutover complete: active stack pp   (DONE tp_to_pp epoch 2, 2847 ms)
  06:53:59 arming pp_to_tp: IDLE-LOCKED ... (1 req resident, 0 tok pending)
  06:54:01 cutover complete: active stack tp   (DONE pp_to_tp epoch 3, 2284 ms)
  06:54:01 arming tp_to_pp: IDLE-LOCKED ... (0 req resident, 22 tok pending)
  06:54:04 cutover complete: active stack pp   (DONE tp_to_pp epoch 4, 2848 ms)
  ...

Each layout certifies the OTHER as the escape. On the tp side the request reads
as not resident with its tokens still pending; on the pp side it reads as
resident with nothing pending. Measured alternating runs, all rotations of
2026-08-17: 72 arms / 299 s, 12 arms / 31 s, 10 arms / 27 s (twice), 3 arms.

WHAT IS *NOT* HAPPENING, checked and refuted. The obvious reading of the block
above -- one request being re-prefilled every half-cycle because neither layout
credits the other's work -- is WRONG. Across those six re-prefills the
``Prefill batch`` shape is identical every time (#new-token 22, seqlen 23), but
the ``rid`` in the FLIP EXTENT PROBE is DIFFERENT every time: 3a9e8d43,
aac838e3, 23bf8412, aded153d, e968893c, 2c008776. Six distinct requests of the
same size, not one request six times. Whether the fresh rid per cycle is itself
a symptom (carry/quiescence dropping request identity across a cutover) or
simply the next client request arriving into the same trap cannot be settled
from these markers, and is left open rather than guessed.

So the defensible statement is the weaker and more useful one: ON AN IDLE BOX A
SMALL PROMPT COSTS A LAYOUT ROUND TRIP. At 06:53:56 the tp layout had already
decided it could do the work itself -- "pending prefill 22 tok <= N=7004,
running it in tp" -- ran the prefill batch, and armed tp_to_pp anyway.

That is the missing explanation for the #713 TTFT quantisation. The half-cycle
is one cutover, measured median 2864 ms (tp_to_pp) and 2772 ms (pp_to_tp) over
486 flips, and the quantised TTFT levels in the 06:19 table (0.1 / 3.1 / 5.9 s)
are how many cutovers the request sat through: none, one, two. The 06:19 table
was taken DURING a 27 s alternating run that started 06:19:06. Latency is
quantised because the unit of delay is a whole layout flip.

TRIGGER, from 16 rotations. The eliciting state is an idle tp layout ("holding
in tp: decoding in tp, running bs 0" every ~10 s) into which a small prefill
lands just as the resident count reaches 0. Size matters and is not a cliff:
pending 22 and 81 sustained runs of >= 4 arms; 63, 178 and 564 bounced once or
twice and stopped; every arm with pending >= 69,710 never chained at all.

TERMINATION. No run ended by client disconnect or by a larger request
displacing the stuck one. All ended the same way: enough requests accumulated
concurrently that a Decode batch with #running-req > 0 finally executed. The
one 299 s run is a different and worse shape -- a genuine multi-tenant backlog
pile-up, 3-4 requests carried across every cutover and 138 distinct rids, whose
tp-side pending grows 81 -> 639 -> 1078 -> 1572 -> 1941 and peaks at 2016. It
is reported separately for that reason; the report prints the pending sequence
rather than one number so the two shapes stay distinguishable.

Mechanism, from the code rather than from the logs
(python/sglang/srt/managers/phase_policy.py, IDLE_LOCKED branch):

  * The branch sits ABOVE the min-dwell check and deliberately bypasses it, on
    the strength of a written claim: "IT CANNOT OSCILLATE ... After the flip
    the target runs by premise, so the same condition is false there."
  * The premise fails because the policy evaluates on the first round after the
    cutover, before the carried request has been re-admitted -- Scheduler
    ._idle_locked_inputs is gated on ``_round_built_nothing``, which a
    just-cut-over layout trivially satisfies. So the target does NOT "run by
    premise"; it is observed in its empty transient and immediately armed back.

Since the only damper is bypassed in exactly this branch, nothing bounds the
alternation, which is why a 299 s run is possible at all.

THE NAMED FAILING ARM (this is the point of the file)
-----------------------------------------------------
CONTROL (unfixed build, e.g. c4bc982d64 or 5fed8a62ed, both of which contain
the unguarded IDLE_LOCKED branch):

    Small single-shot prompts fired at an otherwise idle instance MUST produce
    alternating arm runs -- at least one run of >= 2 arms across the shots --
    carrying the mirrored signature:
        tp side: running_bs == 0 and pending > 0
        pp side: running_bs >= 1 and pending == 0
    and shot TTFT must be quantised in units of the measured half-cycle rather
    than continuously distributed.

FIXED (a build with a post-cutover settle guard on the IDLE_LOCKED branch):

    The same shots must produce runs of <= 1 arm, and TTFT must lose the
    quantisation.

IF THE CONTROL DOES NOT PING-PONG under this harness, the trigger model is
wrong and THAT IS THE RESULT -- report it, do not retune the harness until it
produces the wanted answer. The recorded runs above are the proof the
phenomenon exists; a harness that cannot elicit it has failed to reproduce a
known-real effect, which is a finding about the harness and about the trigger.

WHAT IS DELIBERATELY NOT CLAIMED
--------------------------------
That the ping-pong is what #713(a) fixed. It is not: on the tp side of the
loop ``running_bs`` is 0, so the #713(a) resident term contributes nothing
there. #713(a) remains "deployed, payoff pending targeted proof". This harness
measures the ping-pong, which is a DIFFERENT and larger defect that happens to
produce the symptom #713 was opened on.

TIMING HONESTY
--------------
Server log timestamps are SECOND-resolution; there is no sub-second clock in
the log. So a "last 500 ms of the window" bin cannot be read from log
timestamps, and this harness does not pretend otherwise:

  * shots are AIMED using the client monotonic clock against a predicted
    window end, and
  * shots are LABELLED post hoc from the realized offset to the observed
    boundary, where the boundary time is the client-side DETECTION time of the
    boundary line in the live tail, not the log's 1 s stamp.
  * the tail detection jitter is MEASURED every run, and bins narrower than
    the measured p95 jitter are REFUSED rather than reported.

A mis-aimed shot is therefore not lost data; it lands in whatever bin it
actually hit.

USAGE
-----
    # offline, no server contact -- parser/binning/signature over a recording
    python d2_phase_locked_injector.py --dry-run scripts/fixtures/<excerpt>.log

    # live A/B (requires a GPU window ticket; CONTROL BUILD FIRST)
    python d2_phase_locked_injector.py --live --label control-c4bc982d64
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict

DEFAULT_LOG = "/spinning/evidence-665-f1/boot_bundle.log"
BASE = "http://127.0.0.1:30030"
MODEL = "Qwen3.8-27B"
ARB_HOLDER = "/spinning/gpu-arb/holder"

# ---------------------------------------------------------------- log grammar
# Every pattern below was taken from lines actually present in the recorded
# rotations, not from the emitting code -- the log is the contract this
# harness reads, and it has drifted from the code before.
TS = r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (PP\d)\]"
RE_ARM = re.compile(
    TS + r" PHASE-POLICY arming (tp_to_pp|pp_to_tp):.*?"
    r"\((\d+) req resident, (\d+) tok prefill pending\)"
)
RE_CUTOVER = re.compile(TS + r" PHASE-FLIP cutover complete: active stack (pp|tp)")
RE_DONE = re.compile(
    TS + r" PHASE-FLIP DONE (tp_to_pp|pp_to_tp) \(epoch (\d+)\) in ([\d.]+) ms"
)
RE_HOLD = re.compile(
    TS + r" PHASE-POLICY holding in (pp|tp): ([^(]*)"
    r"\(pending prefill (\d+) tok, running bs (\d+)\)"
)
RE_EXTENT = re.compile(
    TS + r" PHASE-FLIP FLIP EXTENT PROBE .*?rid=([0-9a-f]+) seqlen=(\d+) "
    r"kv_allocated_len=(\d+) aligned=(\d+) kv_committed_len=(\d+)"
)
RE_CARRY = re.compile(
    TS + r" PHASE-FLIP-CARRY carried (\d+) resident request\(s\) across the "
    r"cutover into the (pp|tp) phase"
)
RE_PREFILL = re.compile(TS + r" Prefill batch, #new-seq: (\d+), #new-token: (\d+)")

#: Consecutive opposite-direction arms closer than this are one alternating
#: run. 8 s is comfortably above the largest observed half-cycle (~5 s) and
#: below any healthy inter-arm spacing seen in the recordings.
RUN_MAX_GAP_S = 8.0

#: Only rank PP0 originates arms (the origin guard in maybe_arm_phase_policy),
#: so counting every rank would triple every run length.
ORIGIN_RANK = "PP0"


@dataclass
class Arm:
    ts: str
    rank: str
    direction: str
    running_bs: int
    pending: int

    @property
    def mirrored_tp(self) -> bool:
        """tp side of the ping-pong signature: nothing resident, work pending."""
        return (
            self.direction == "tp_to_pp" and self.running_bs == 0 and self.pending > 0
        )

    @property
    def mirrored_pp(self) -> bool:
        """pp side: something resident, nothing pending."""
        return (
            self.direction == "pp_to_tp" and self.running_bs >= 1 and self.pending == 0
        )


@dataclass
class Run:
    arms: list = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.arms)

    @property
    def seconds(self) -> float:
        return _epoch(self.arms[-1].ts) - _epoch(self.arms[0].ts)

    @property
    def signature(self) -> bool:
        """True when BOTH mirrored halves appear -- the diagnostic shape.

        A run of alternating arms alone is not the signature; layouts can
        legitimately alternate under changing load. What identifies THIS
        defect is that each side reports the request in a state the other side
        does not, so both halves must be present.
        """
        return any(a.mirrored_tp for a in self.arms) and any(
            a.mirrored_pp for a in self.arms
        )

    def summary(self) -> dict:
        return {
            "length": self.length,
            "seconds": round(self.seconds, 1),
            "signature": self.signature,
            "start": self.arms[0].ts,
            "pending_sequence": [
                a.pending for a in self.arms if a.direction == "tp_to_pp"
            ],
            "arms": [asdict(a) for a in self.arms],
        }


def _epoch(ts: str) -> float:
    return time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))


def parse_markers(lines) -> dict:
    """Pull every marker class out of an iterable of log lines.

    Streams: callers pass a file iterator, never a preloaded string, because
    the rotations reach 14 MB and this box has been OOM-killed by a careless
    log read before.
    """
    out = {
        "arms": [],
        "cutovers": [],
        "dones": [],
        "holds": [],
        "extents": [],
        "carries": [],
        "prefills": [],
    }
    for line in lines:
        if "PHASE-" not in line and "Prefill batch" not in line:
            continue
        m = RE_ARM.search(line)
        if m and m.group(2) == ORIGIN_RANK:
            out["arms"].append(
                Arm(
                    m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5))
                )
            )
            continue
        m = RE_CUTOVER.search(line)
        if m:
            out["cutovers"].append((m.group(1), m.group(2), m.group(3)))
            continue
        m = RE_DONE.search(line)
        if m:
            out["dones"].append(
                (m.group(1), m.group(3), int(m.group(4)), float(m.group(5)))
            )
            continue
        m = RE_HOLD.search(line)
        if m:
            out["holds"].append(
                (
                    m.group(1),
                    m.group(3),
                    m.group(4).strip(),
                    int(m.group(5)),
                    int(m.group(6)),
                )
            )
            continue
        m = RE_EXTENT.search(line)
        if m:
            out["extents"].append(
                (m.group(1), m.group(3), int(m.group(4)), int(m.group(7)))
            )
            continue
        m = RE_CARRY.search(line)
        if m:
            out["carries"].append((m.group(1), int(m.group(3)), m.group(4)))
            continue
        m = RE_PREFILL.search(line)
        if m:
            out["prefills"].append(
                (m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))
            )
    return out


def alternating_runs(arms, max_gap: float = RUN_MAX_GAP_S):
    """Maximal runs of opposite-direction arms spaced under ``max_gap``."""
    runs, cur = [], []
    for a in arms:
        if (
            cur
            and a.direction != cur[-1].direction
            and _epoch(a.ts) - _epoch(cur[-1].ts) <= max_gap
        ):
            cur.append(a)
        else:
            if len(cur) >= 2:
                runs.append(Run(cur))
            cur = [a]
    if len(cur) >= 2:
        runs.append(Run(cur))
    return runs


def half_cycles(runs):
    out = []
    for r in runs:
        for a, b in zip(r.arms, r.arms[1:]):
            out.append(_epoch(b.ts) - _epoch(a.ts))
    return out


def report(markers, label: str) -> dict:
    runs = alternating_runs(markers["arms"])
    hc = half_cycles(runs)
    sig = [r for r in runs if r.signature]
    done_tp = [d[3] for d in markers["dones"] if d[1] == "tp_to_pp"]
    done_pp = [d[3] for d in markers["dones"] if d[1] == "pp_to_tp"]

    def stat(xs):
        if not xs:
            return None
        return {
            "n": len(xs),
            "min": round(min(xs), 1),
            "median": round(statistics.median(xs), 1),
            "max": round(max(xs), 1),
        }

    res = {
        "label": label,
        "arms": len(markers["arms"]),
        "runs": len(runs),
        "runs_with_signature": len(sig),
        "longest_run": max((r.length for r in runs), default=0),
        "longest_run_seconds": round(max((r.seconds for r in runs), default=0.0), 1),
        "half_cycle_s": stat(hc),
        "cutover_ms_tp_to_pp": stat(done_tp),
        "cutover_ms_pp_to_tp": stat(done_pp),
        "largest_pending_in_run": max(
            (a.pending for r in runs for a in r.arms), default=0
        ),
        "run_details": [r.summary() for r in runs[:10]],
    }
    return res


def print_report(res: dict) -> None:
    print(f"\n=== {res['label']} ===")
    print(
        f"  arms {res['arms']}  runs {res['runs']}  "
        f"with-signature {res['runs_with_signature']}"
    )
    print(f"  longest run {res['longest_run']} arms / {res['longest_run_seconds']}s")
    print(f"  half-cycle s        : {res['half_cycle_s']}")
    print(f"  cutover ms tp_to_pp : {res['cutover_ms_tp_to_pp']}")
    print(f"  cutover ms pp_to_tp : {res['cutover_ms_pp_to_tp']}")
    print(f"  largest pending seen in a run: {res['largest_pending_in_run']}")
    for r in res["run_details"][:4]:
        flag = "SIGNATURE" if r["signature"] else "no-signature"
        print(
            f"    run {r['start']} len={r['length']} {r['seconds']}s {flag} "
            f"pending_seq={r['pending_sequence'][:8]}"
        )


# --------------------------------------------------------------------- verdict
def verdict(res: dict, arm_kind: str) -> tuple[bool, str]:
    """Apply the NAMED FAILING ARM stated in the module docstring.

    ``arm_kind`` is "control" (unfixed, MUST ping-pong) or "fixed" (MUST NOT).
    Returned bool is "did this arm behave as its name requires".
    """
    if arm_kind == "control":
        ok = res["runs_with_signature"] >= 1
        why = (
            f"control must ping-pong: {res['runs_with_signature']} run(s) carry "
            f"the mirrored signature, longest {res['longest_run']} arms"
        )
        if not ok:
            why += (
                " -- NO SIGNATURE RUN. The trigger model is wrong, or this "
                "instance is not in the state that elicits it. That is the "
                "finding; do not retune until it says what was wanted."
            )
        return ok, why
    ok = res["runs_with_signature"] == 0 and res["longest_run"] <= 1
    return ok, (
        f"fixed must not ping-pong: {res['runs_with_signature']} signature run(s), "
        f"longest {res['longest_run']} arms"
    )


# ------------------------------------------------------------------- live mode
def _guard_live() -> str:
    """Refuse to touch the server without a GPU window ticket.

    The live run is a window ticket, not a desk activity: it puts real load on
    an instance other strands share. Import and dry-run must never be able to
    reach the server by accident, so the guard is here rather than in a
    comment.
    """
    if not os.path.exists(ARB_HOLDER):
        return f"no GPU window claim at {ARB_HOLDER}"
    try:
        with open(ARB_HOLDER) as fh:
            holder = fh.read().strip()
    except OSError as exc:
        return f"cannot read {ARB_HOLDER}: {exc}"
    if not holder:
        return f"{ARB_HOLDER} is empty"
    return ""


def _check_gap(gap: float) -> str:
    """Refuse a shot cadence that would fabricate the very thing measured.

    THE HARNESS CAN COUNTERFEIT ITS OWN RESULT. Runs are detected as
    opposite-direction arms within RUN_MAX_GAP_S of each other. The recorded
    evidence shows each small request costing about one layout round trip, so
    firing shots faster than RUN_MAX_GAP_S would chain one-flip-per-shot into a
    single long "run" that looks exactly like a self-sustaining oscillation and
    is nothing but the injector's own cadence. The first draft of this file
    defaulted to 6 s and would have done precisely that.

    So shots are spaced beyond the run window, and each shot's cost is then
    attributable to that shot alone.
    """
    if gap <= RUN_MAX_GAP_S:
        return (
            f"--gap {gap:g}s is <= RUN_MAX_GAP_S ({RUN_MAX_GAP_S:g}s): consecutive "
            f"shots would be chained into one alternating run by the detector, "
            f"manufacturing the result. Use a gap above {RUN_MAX_GAP_S:g}s."
        )
    return ""


def arms_between(arms, t_lo: float, t_hi: float):
    """Arms whose log second falls in [t_lo, t_hi], for per-shot attribution.

    Log stamps are second-resolution, so the window is inclusive at both ends
    and a shot that lands on a second boundary may claim an arm that belongs to
    its neighbour. With a gap well above the run window that ambiguity is one
    arm at most, and it is reported rather than hidden.
    """
    return [a for a in arms if t_lo <= _epoch(a.ts) <= t_hi]


def _shot(prompt: str, max_tokens: int = 32) -> tuple[float, str | None]:
    """One streaming shot; returns (ttft_s, rid) with rid from the response id.

    The completion id and the server-side ``rid=`` in FLIP EXTENT PROBE lines
    are the same 32-hex identifier, which is what makes per-shot correlation to
    the log possible at all.
    """
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    rid = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except ValueError:
                continue
            rid = rid or d.get("id")
            delta = (d.get("choices") or [{}])[0].get("delta", {}) or {}
            if delta.get("content") or delta.get("reasoning_content"):
                return time.time() - t0, rid
    return time.time() - t0, rid


def run_live(args) -> int:
    why = _guard_live()
    if why:
        print(f"REFUSING LIVE RUN: {why}", flush=True)
        print("This mode is a window ticket. Claim the window first.", flush=True)
        return 2
    why = _check_gap(args.gap)
    if why:
        print(f"REFUSING LIVE RUN: {why}", flush=True)
        return 2
    log = args.log
    start = os.path.getsize(log)
    shots = []
    print(
        f"live: {args.shots} idle small-prompt shots, {args.gap:g}s apart", flush=True
    )
    for i in range(args.shots):
        tag = f"s{i:03d}"
        t_arr = time.time()
        try:
            ttft, rid = _shot(f"[{tag}] In one short sentence, what is a prefill?")
        except Exception as exc:  # noqa: BLE001 - one bad shot must not end the run
            print(f"  shot {i} error: {exc}", flush=True)
            continue
        shots.append(
            {
                "i": i,
                "tag": tag,
                "rid": rid,
                "ttft_s": round(ttft, 3),
                "arrival": t_arr,
                "first_token": time.time(),
            }
        )
        print(f"  shot {i:3d} ttft {ttft:6.2f}s rid={rid}", flush=True)
        time.sleep(args.gap)
    with open(log, "r", errors="replace") as fh:
        fh.seek(start)
        markers = parse_markers(fh)
    res = report(markers, args.label)
    # PER-SHOT ATTRIBUTION. The headline question is not how long the longest
    # run was -- it is what a single small prompt COSTS on an idle box. With
    # the gap held above the run window, the arms inside a shot's own
    # arrival..first-token interval belong to that shot.
    for s in shots:
        own = arms_between(markers["arms"], s["arrival"], s["first_token"])
        s["arms_during"] = len(own)
        s["directions"] = [a.direction for a in own]
    costed = [s for s in shots if s["arms_during"] > 0]
    res["shots"] = shots
    res["shots_costing_a_flip"] = len(costed)
    res["shot_arm_counts"] = sorted(s["arms_during"] for s in shots)
    ttfts = [s["ttft_s"] for s in shots]
    res["ttft"] = {
        "n": len(ttfts),
        "min": min(ttfts, default=0),
        "max": max(ttfts, default=0),
        "median": round(statistics.median(ttfts), 3) if ttfts else 0,
        "spread_ratio": round(max(ttfts) / min(ttfts), 1)
        if ttfts and min(ttfts)
        else 0,
    }
    print_report(res)
    ok, why = verdict(res, args.arm)
    res["arm_kind"] = args.arm
    res["arm_behaved_as_named"] = ok
    print(f"\n  TTFT: {res['ttft']}")
    print(f"  VERDICT [{args.arm}]: {'AS NAMED' if ok else 'NOT AS NAMED'} -- {why}")
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"  -> {args.out}")
    return 0


def run_dry(args) -> int:
    """Offline replay. No server contact, no GPU, no window."""
    if not os.path.exists(args.dry_run):
        print(f"no such recording: {args.dry_run}")
        return 2
    with open(args.dry_run, "r", errors="replace") as fh:
        markers = parse_markers(fh)
    counts = {k: len(v) for k, v in markers.items()}
    print(f"parsed from {args.dry_run}:")
    for k, v in counts.items():
        print(f"  {k:<10} {v}")
    if not markers["arms"]:
        print(
            "\nNO ARMS PARSED -- the recording carries no arming lines, so this "
            "run proves nothing about the parser's run detection."
        )
        return 1
    res = report(markers, f"dry-run {os.path.basename(args.dry_run)}")
    print_report(res)
    ok, why = verdict(res, args.arm)
    print(f"\n  VERDICT [{args.arm}]: {'AS NAMED' if ok else 'NOT AS NAMED'} -- {why}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"  -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Separate from main() so the tests can read the REAL defaults.

    A test that re-states a default as a literal drifts silently the moment
    the default changes, which is the failure mode that let a 6 s cadence sit
    against an 8 s run window in the first place.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dry-run",
        metavar="LOGFILE",
        help="offline replay of a recorded log; no server contact",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="fire real shots (requires a GPU window claim)",
    )
    ap.add_argument(
        "--arm",
        choices=("control", "fixed"),
        default="control",
        help="which named arm this run is; decides the verdict rule",
    )
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--shots", type=int, default=12)
    ap.add_argument(
        "--gap",
        type=float,
        default=25.0,
        help="seconds between shots; MUST exceed RUN_MAX_GAP_S or the harness "
        "manufactures its own runs (see _check_gap)",
    )
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="/spinning/evidence-665-f1/d2_injector.json")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    if args.dry_run:
        return run_dry(args)
    if args.live:
        return run_live(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
