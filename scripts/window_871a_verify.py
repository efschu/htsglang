#!/usr/bin/env python3
"""#871a WINDOW TICKET -- one call, PASS/FAIL with numbers, for a boot log.

Decides the claims that could not be settled at a desk. It reads a boot log and
the cgroup; it never boots anything, never touches a card, and never restores
serving. Run it AFTER the operator's boot, against that boot's log.

    scripts/window_871a_verify.py --log /root/current_boot.log

EXIT CODES -- and the reason they are spelled out here rather than left to
convention: an exit code that is not enumerated gets read as "not 0, so it
failed" or, worse, "not 1, so it passed". Both have happened.

    0  every claim PASSED
    1  at least one claim FAILED   (a real, decided negative)
    2  the run could not be DECIDED (log missing/truncated, no fences ran,
       markers absent) -- THIS IS NOT A PASS. It means the evidence was not
       there, which is a different fact from the claim being false, and it must
       send the reader back to the boot rather than into the next ticket.

`--self-test` runs every claim against synthetic logs that are known-good and
known-bad, and exits non-zero unless BOTH directions come out right. A window
script whose first execution is on metal is not a ticket, it is a hope -- so
this is wired to run in the desk gate as well.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# --------------------------------------------------------------------------
# The markers. Each is the string the PRODUCT emits; if a marker changes, this
# script must fail LOUD (undecided) rather than quietly stop matching.
# --------------------------------------------------------------------------

M_LEDGER = "HOST-LEDGER POST #847/#871"
M_POOLS_FULL = "pools=['kv', 'mamba']"
M_MAMBA_MISSING = "MAMBA half not built"
M_REBIND_REFUSED = "#871 PHASE-FLIP REBIND"
M_PIN_REFUSED = "#847 PHASE-FLIP REBIND: could not allocate"
M_BUDGET_REFUSAL = "does not fit"
M_STORE_NEVER = "#871a STORE NEVER DELIVERED"
M_FENCE = "[#703 flip-writeback]"

RE_ACKED = re.compile(r"acked=(\d+)")
RE_ELIGIBLE = re.compile(r"eligible=(\d+)")


class Claim:
    def __init__(self, key: str, title: str):
        self.key, self.title = key, title
        self.verdict = "UNDECIDED"
        self.detail = "not evaluated"

    def decide(self, ok: bool, detail: str) -> None:
        self.verdict = "PASS" if ok else "FAIL"
        self.detail = detail

    def undecided(self, detail: str) -> None:
        self.verdict = "UNDECIDED"
        self.detail = detail


def read_cgroup_facts() -> dict:
    """anon / file / shmem / current / peak, in bytes. Absent keys -> None.

    `file` is here because #534 measured that CUDA PINNED host memory is
    accounted in the cgroup's `file` bucket, not `anon` -- "the offload ledger
    reported 49.66 GiB of pinned pool while `anon` sat steady at 14.6 GiB".
    `honest_host_memory_bytes` charges anon + kernel + unreclaimable shmem and
    deliberately never charges `file`, because page cache is reclaimable. Pinned
    bytes in that bucket are NOT, so this pair is what decides whether the
    pinned-host guard can see its own posts at all. Desk reading on this box
    before any boot: anon 2.12 GiB, file 19.06 GiB, current 21.39 GiB.
    """
    # SCOPE, RECORDED RATHER THAN ASSUMED. `/sys/fs/cgroup` from inside this
    # container is the container's ROOT view, which aggregates every process in
    # it -- serving, lanes and this script alike. That is the right scope for a
    # host-RAM question, but it is NOT the scope #721's "system.slice peak
    # 111.3 GiB" was taken at, and confusing the two is easy: measured here,
    # root / system.slice / system.slice+claude.service report three different
    # `memory.current` values and all three carry `memory.max = max`. The path
    # is printed with the numbers so a later reader can tell which one they are
    # looking at instead of inferring it.
    root = "/sys/fs/cgroup"
    out: dict = {k: None for k in ("anon", "file", "shmem", "current", "peak")}
    out["scope"] = root
    for name in ("memory.current", "memory.peak"):
        try:
            with open(f"{root}/{name}", encoding="utf-8") as fh:
                out[name.split(".", 1)[1]] = int(fh.read().strip())
        except (OSError, ValueError):
            pass
    try:
        with open(f"{root}/memory.stat", encoding="utf-8") as fh:
            for line in fh:
                k, _, v = line.partition(" ")
                if k in ("anon", "file", "shmem"):
                    out[k] = int(v)
    except (OSError, ValueError):
        pass
    return out


def claim_tier_armed(text: str, c: Claim) -> None:
    """#871: the phase host tier is built with the FULL pool set."""
    ledgers = text.count(M_LEDGER)
    if ledgers == 0:
        c.undecided(
            f"no {M_LEDGER!r} line in the log: the pin builder never reached "
            "its ledger, so nothing about the tier can be read from this boot"
        )
        return
    full = text.count(M_POOLS_FULL)
    missing = text.count(M_MAMBA_MISSING)
    refusals = text.count(M_REBIND_REFUSED)
    ok = full == ledgers and missing == 0 and refusals == 0
    c.decide(
        ok,
        f"ledger lines={ledgers}, pools=['kv','mamba']={full}, "
        f"'MAMBA half not built'={missing}, coverage refusals={refusals} "
        f"(want full==ledgers, 0 missing, 0 refusals)",
    )


def claim_pin_admitted(text: str, c: Claim) -> None:
    """#871a: the pin went through the pinned-host budget, and said so.

    PASS has two shapes and both are correct outcomes: ADMITTED (the ledger
    line printed and no refusal) or REFUSED BY NAME (the budget raised and the
    existing #847 path caught it). The FAILURE this looks for is neither
    happening -- a pin that allocated with no admission recorded at all, which
    is the state before this ticket.
    """
    admitted = text.count(M_LEDGER)
    refused = text.count(M_PIN_REFUSED)
    if admitted == 0 and refused == 0:
        c.undecided(
            "neither an admission ledger line nor a named refusal is present; "
            "the pin builder did not run in this boot"
        )
        return
    if refused:
        c.decide(
            True,
            f"REFUSED BY NAME x{refused} -- the budget declined the pin and the "
            "#847 path caught it, which is the guard working (the rebind will "
            "refuse at the first cutover with a legible reason, not OOM)",
        )
        return
    c.decide(
        True,
        f"ADMITTED x{admitted} with no named refusal -- the pin was priced "
        "against the pinned-host budget and fitted",
    )


def claim_store_delivered(text: str, c: Claim) -> None:
    """#871a: has a byte EVER reached the geometry-neutral store?

    The whole point of the lifetime counter: a fence reporting `acked=0` is
    indistinguishable from a healthy store on an idle instance, so this asks
    the question over the WHOLE boot instead of per cutover.
    """
    # SCOPED TO THE FENCE'S OWN LINES, and this is not defensive tidiness --
    # the first version of this function scanned the WHOLE log for `acked=` and
    # returned PASS on the W40 boot, which is a boot where every one of the 21
    # fences reported `acked=0`. Seven lines in that log carry `acked=` from a
    # different subsystem, three of them `acked=24`, so the script summed 72
    # acknowledgements that no fence ever made. A window script that reports a
    # false PASS is worse than no script: it closes a claim that is open.
    fence_lines = [ln for ln in text.splitlines() if M_FENCE in ln]
    fences = len(fence_lines)
    if fences == 0:
        c.undecided(
            "no writeback fence ran in this boot, so store delivery cannot be "
            "decided -- this is the 'no evidence' case, not a pass"
        )
        return
    scoped = "\n".join(fence_lines)
    acked_total = sum(int(m) for m in RE_ACKED.findall(scoped))
    eligible_total = sum(int(m) for m in RE_ELIGIBLE.findall(scoped))
    alarm = text.count(M_STORE_NEVER)
    if acked_total > 0:
        c.decide(
            True,
            f"{acked_total} storage acknowledgement(s) across {fences} fence(s) "
            f"(eligible total {eligible_total}); the store HAS taken bytes and "
            f"the #871a alarm correctly stayed silent (fired {alarm}x)",
        )
        return
    if eligible_total == 0:
        c.undecided(
            f"{fences} fence(s) ran but every one had eligible=0 -- the tree "
            "was empty at every cutover, so this boot carried no work to "
            "persist and says nothing about the store. Re-run under load."
        )
        return
    # Work was present and nothing was acknowledged: a decided negative, and
    # the alarm is expected to have fired.
    c.decide(
        False,
        f"{fences} fence(s), {eligible_total} eligible node(s), and acked=0 "
        f"throughout: NOTHING ever reached the store. #871a alarm fired "
        f"{alarm}x (expected >=1 once {fences} fences had run)",
    )


CLAIMS = (
    (
        "tier_armed",
        "#871 phase host tier built with the full pool set",
        claim_tier_armed,
    ),
    (
        "pin_admitted",
        "#871a staging pin admitted through the pinned-host budget",
        claim_pin_admitted,
    ),
    (
        "store_delivered",
        "#871a a byte has reached the store (lifetime, not per-fence)",
        claim_store_delivered,
    ),
)


def run(text: str, cg_before: dict, cg_after: dict) -> int:
    claims = []
    for key, title, fn in CLAIMS:
        c = Claim(key, title)
        try:
            fn(text, c)
        except Exception as exc:  # noqa: BLE001 - an undecidable claim is not a pass
            c.undecided(f"evaluator raised: {exc}")
        claims.append(c)

    print("=" * 78)
    print("#871a WINDOW VERIFY")
    print("=" * 78)
    for c in claims:
        print(f"[{c.verdict:9}] {c.title}")
        print(f"            {c.detail}")

    print("-" * 78)
    print(
        f"HOST LEDGER (#721) -- cgroup bytes, before vs after "
        f"[scope: {cg_after.get('scope') or cg_before.get('scope') or 'unknown'}]"
    )

    def g(v):
        return "unknown" if v is None else f"{v / 2**30:.2f} GiB"

    for k in ("current", "peak", "anon", "file", "shmem"):
        print(f"  {k:8} {g(cg_before.get(k)):>12}  ->  {g(cg_after.get(k)):>12}")
    a, f = cg_after.get("anon"), cg_after.get("file")
    if a is not None and f is not None:
        print(
            f"  #534 BUCKET QUESTION: file={g(f)} vs anon={g(a)}. "
            "`honest_host_memory_bytes` charges anon and NEVER charges file, so "
            "any pinned bytes in file are invisible to the pinned-host guard -- "
            "which would also make the credit-back at "
            "pinned_host_budget.py:253 ('their bytes are therefore already "
            "missing from it') unsound for those posts. A large file figure "
            "that grows with the pin is the positive result."
        )

    failed = [c for c in claims if c.verdict == "FAIL"]
    undecided = [c for c in claims if c.verdict == "UNDECIDED"]
    print("-" * 78)
    if failed:
        print(f"RESULT: FAIL ({len(failed)} claim(s) decided negative) -> exit 1")
        return 1
    if undecided:
        print(
            f"RESULT: UNDECIDED ({len(undecided)} claim(s) lacked evidence) -> "
            "exit 2. NOT a pass: the boot did not produce what these claims "
            "need. Re-boot or re-run under load."
        )
        return 2
    print("RESULT: PASS (all claims decided positive) -> exit 0")
    return 0


# --------------------------------------------------------------------------
# Self-test: both directions, before metal.
# --------------------------------------------------------------------------

_GOOD = (
    f"{M_LEDGER} ... pools=['kv', 'mamba'] ... x\n" * 3
    + f"{M_FENCE} eligible=2 staged=2 already_staged=0 acked=2 outstanding=0\n"
)
_BAD_STORE = (
    f"{M_LEDGER} ... pools=['kv', 'mamba'] ... x\n" * 3
    + f"{M_FENCE} eligible=2 staged=2 already_staged=0 acked=0 outstanding=2\n" * 4
    + f"PHASE-FLIP {M_STORE_NEVER}: 4 fences ...\n"
)
_BAD_TIER = f"{M_LEDGER} ... pools=['kv'] ...\n" * 3 + f"{M_MAMBA_MISSING}\n"
_EMPTY_IDLE = (
    f"{M_LEDGER} ... pools=['kv', 'mamba'] ...\n" * 3
    + f"{M_FENCE} eligible=0 staged=0 already_staged=0 acked=0 outstanding=0\n" * 5
)
#: The regression for the false PASS this script shipped with for one commit:
#: a foreign subsystem's `acked=` on a boot whose fences all acked NOTHING.
#: Taken from the real W40 log, which carries three `acked=24` lines that no
#: fence emitted.
_FOREIGN_ACKED = (
    f"{M_LEDGER} ... pools=['kv', 'mamba'] ...\n" * 3
    + "[some-other-subsystem] batches acked=24 outstanding=0\n" * 3
    + f"{M_FENCE} eligible=1 staged=0 already_staged=1 acked=0 outstanding=0\n" * 4
    + f"PHASE-FLIP {M_STORE_NEVER}: 4 fences ...\n"
)


def self_test() -> int:
    cg = read_cgroup_facts()
    cases = [
        ("known-good boot", _GOOD, 0),
        ("store never delivered", _BAD_STORE, 1),
        ("tier not full pool set", _BAD_TIER, 1),
        ("idle boot, no work", _EMPTY_IDLE, 2),
        ("foreign acked= must not count as delivery", _FOREIGN_ACKED, 1),
        ("empty log", "", 2),
    ]
    bad = 0
    for name, text, want in cases:
        got = run(text, cg, cg)
        mark = "ok" if got == want else "WRONG"
        if got != want:
            bad += 1
        print(f"\n### self-test {name!r}: want exit {want}, got {got} [{mark}]\n")
    print("=" * 78)
    if bad:
        print(f"SELF-TEST FAILED: {bad} case(s) wrong")
        return 1
    print("SELF-TEST PASSED: every case decided in the intended direction")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", help="boot log to read")
    ap.add_argument(
        "--cgroup-before",
        help="optional file written by a pre-boot run of --snapshot",
    )
    ap.add_argument(
        "--snapshot", action="store_true", help="write cgroup facts and exit 0"
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    now = read_cgroup_facts()
    if args.snapshot:
        for k, v in now.items():
            print(f"{k}={'' if v is None else v}")
        return 0

    before = {}
    if args.cgroup_before and os.path.exists(args.cgroup_before):
        for line in open(args.cgroup_before, encoding="utf-8"):
            k, _, v = line.strip().partition("=")
            before[k] = int(v) if v else None

    if not args.log:
        print("RESULT: UNDECIDED -- no --log given -> exit 2")
        return 2
    if not os.path.exists(args.log):
        print(f"RESULT: UNDECIDED -- log {args.log!r} does not exist -> exit 2")
        return 2
    with open(args.log, encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    if not text.strip():
        print(f"RESULT: UNDECIDED -- log {args.log!r} is empty -> exit 2")
        return 2
    return run(text, before or now, now)


if __name__ == "__main__":
    sys.exit(main())
