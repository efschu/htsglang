#!/usr/bin/env python3
"""Merge-train pass 3, mechanically: the executor for LEDGER_merge_train_0818.

This script IS section (c) of docs/dev/LEDGER_merge_train_0818.md -- same
order, same conflict predictions, same resolutions. It exists because the
55-commit lineage near-miss proved a written map must precede the train, and
a map an operator re-derives by hand under time pressure is a map that
drifts.

CONTRACT:

* GATE: refuses to run unless the comp4-green marker exists
  (--marker, default /spinning/evidence-665-f1/COMP4_ACCEPTED) -- pass 3 is
  boot-proof-gated by the ledger's own rule.
* STANDING RULE first: re-derives the ledger's (a) table against the ACTUAL
  tip before touching anything; aborts if the lineage facts moved.
* Per step: cherry-pick/merge exactly as mapped. A conflict that is NOT in
  the step's predicted set aborts loudly with state saved -- an unpredicted
  conflict means the map is stale, and a stale map is re-derived by a
  person, never resolved ad hoc by a script.
* Predicted conflicts resolve to the MAPPED resolution only:
  - "ours"    drop the incoming hunk for that path (the #754 fixture hunk;
              #756's class attribute covers it),
  - "theirs"  take the incoming file, then VERIFY the load-bearing marker
              strings survived (the #735 doc numbers),
  - "manual"  stop with resumable state and exact instructions (the #727
              gate hunks -- the ledger says re-application by hand).
* Targeted suites after each risky step; stop on first failure.
* State file (JSON) makes every stop resumable: --resume continues after
  the operator finished a manual resolution (git cherry-pick --continue or
  a clean tree) or fixed a suite failure.
* --dry-run executes the SAME plan in a scratch shared clone (suites
  skipped, nothing pushed, the real repo untouched) -- the desk smoke.

Never run against the real repo without the marker. Nothing here pushes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

REVIEW_TIP = "c3fd6f6ab8"
COMPOSITE = "c546eed923"
#: Second wave: F4-r5's harvest composite (contains comp4, #757/#748'/#759/
#: #755-reorder/#758-emitters). The train base; feat/753 lands on it BY ITS
#: OWNER before this executor runs (ledger REFRESH (c')).
HARVEST = "59ce2d8a30"
DEFAULT_TIP = HARVEST
DEFAULT_MARKER = "/spinning/evidence-665-f1/COMP4_ACCEPTED"

PYTEST = [sys.executable, "-m", "pytest", "-q"]
HERMETIC_ENV = {"CUDA_VISIBLE_DEVICES": "99"}


@dataclasses.dataclass
class Step:
    name: str
    kind: str  # "merge" | "cherry"
    ref: str
    #: path -> "ours" | "theirs" | "manual"
    predicted: Dict[str, str] = dataclasses.field(default_factory=dict)
    #: marker strings that must survive a "theirs" resolution, per path
    markers: Dict[str, List[str]] = dataclasses.field(default_factory=dict)
    suites: List[str] = dataclasses.field(default_factory=list)
    note: str = ""


#: Section (c), verbatim as data. Order is load-bearing.
PLAN: List[Step] = [
    Step(
        name="749-order-dependence",
        kind="merge",
        ref="2c936ff82b",
        suites=["test/registered/unit/test_global_leak_guard_749.py"],
        note="review-based branch merge (brings ae739399d7 #733); "
        "de-flakes every suite after it",
    ),
    Step(
        name="751-preflight-boundary",
        kind="merge",
        ref="2cf6a6e4b8",
        suites=["test/registered/unit/turnkey/test_preflight_and_config.py"],
        note="review-based, test-only",
    ),
    Step(
        name="745-anchor-reachability",
        kind="cherry",
        ref="eab1926ea8",
        suites=[
            "test/registered/unit/mem_cache/test_anchor_write_reachability_745.py",
        ],
        note="test-only reachability suite; #754 step RETIRED here -- "
        "superseded by feat/753's fold at the same seam (ledger REFRESH)",
    ),
    Step(
        name="735-arithmetic-docs",
        kind="cherry",
        ref="3facc4b80c",
        predicted={"docs/dev/DESIGN_pp_layer_set.md": "theirs"},
        markers={
            # The load-bearing corrections that must survive the union
            # (ledger (c)5): the NVML-total slot ceiling and its inputs.
            "docs/dev/DESIGN_pp_layer_set.md": ["32607", "4577"],
        },
        note="docs pair; DESIGN union resolved as theirs + marker check, "
        "operator reviews the editorial union afterwards",
    ),
    Step(
        name="727-requant-method",
        kind="cherry",
        ref="3f8996a1a6",
        predicted={
            "python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py": "manual",
            "python/sglang/srt/models/qwen3_5.py": "manual",
        },
        note="trio 1/3; gate hunks are re-applied by hand per the ledger",
    ),
    Step(
        name="727-lmhead-artifact",
        kind="cherry",
        ref="5745a545d6",
        note="trio 2/3; ticket-only",
    ),
    Step(
        name="727-ab-runner",
        kind="cherry",
        ref="d7984ab810",
        predicted={
            "python/sglang/srt/layers/quantization/compressed_tensors/ct_embedding.py": "manual",
        },
        suites=[
            "test/registered/unit/quantization/test_ct_embedding_int8_727.py",
            "test/registered/unit/tools/test_ab_runner_727.py",
        ],
        note="trio 3/3; suites cover the whole trio",
    ),
    Step(
        name="727-head-chain",
        kind="cherry",
        ref="3682331d33",
        suites=[
            "test/registered/unit/quantization/test_ct_lmhead_chain_727.py",
        ],
        note="trio 4/4 (second wave): the lm_head chain pins + accept-len",
    ),
    Step(
        name="738-pageout-verdict",
        kind="cherry",
        ref="f199828d11",
        note="probe tool + verdict note; docs+tool only",
    ),
    Step(
        name="535-unblock-verdict",
        kind="cherry",
        ref="c92e65c547",
        note="ticket updates only",
    ),
    Step(
        name="755-determination-docs",
        kind="cherry",
        ref="d60b11a258",
        note="two new doc files",
    ),
    Step(
        name="740-scaffold-note",
        kind="cherry",
        ref="0480f4be27",
        note="the ORIGINAL note commit -- required first: d11b29d2dd is only "
        "the s5a diff and needs this file as context (found by the dry-run "
        "smoke; the ledger's first draft said 'lands as a new file')",
    ),
    Step(
        name="740-scaffold-residual",
        kind="cherry",
        ref="d11b29d2dd",
        note="the s5a cross-agent measurement on top of the note",
    ),
]


class Abort(RuntimeError):
    pass


def run(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, check=check
    )


def log(msg: str) -> None:
    print(f"[pass3] {msg}", flush=True)


# ------------------------------------------------------------------ gate


def check_gate(marker: str) -> None:
    if not os.path.exists(marker):
        raise Abort(
            f"GATE: comp4-green marker {marker!r} does not exist. Pass 3 is "
            "boot-proof-gated (LEDGER_merge_train_0818): F4-r5 writes the "
            "marker when comp4 proves. Refusing."
        )
    log(f"gate OK: {marker} present")


# ---------------------------------------------------- standing-rule table


def check_lineage(repo: str, tip: str) -> None:
    """The ledger's standing rule: re-derive the (a) facts against the
    ACTUAL tip; abort if the world moved."""
    for must_contain, why in (
        (REVIEW_TIP, "the review lineage"),
        (COMPOSITE, "the composite (absorbed desk fixes)"),
        (HARVEST, "the harvest composite (second wave: #757/#748'/#759/"
         "#755-reorder/#758 emitters)"),
    ):
        r = run(repo, "merge-base", "--is-ancestor", must_contain, tip, check=False)
        if r.returncode != 0:
            raise Abort(
                f"lineage check: tip {tip} does not contain {must_contain} "
                f"({why}). The map is stale -- re-derive the ledger's (a) "
                "table before running pass 3."
            )
    for step in PLAN:
        r = run(repo, "cat-file", "-e", f"{step.ref}^{{commit}}", check=False)
        if r.returncode != 0:
            raise Abort(
                f"lineage check: step {step.name} ref {step.ref} is not a "
                "commit in this repo (fetch missing?)."
            )
        r = run(repo, "merge-base", "--is-ancestor", step.ref, tip, check=False)
        if r.returncode == 0:
            log(f"note: {step.name} ({step.ref}) already contained in tip; "
                "it will be skipped")
            step.kind = "skip"
    log(f"lineage OK: tip {tip} contains review tip and composite")


# ------------------------------------------------------------- state file


def load_state(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"done": [], "current": None, "status": "fresh"}


def save_state(path: str, state: Dict) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------- steps


def conflicted_paths(repo: str) -> List[str]:
    r = run(repo, "diff", "--name-only", "--diff-filter=U")
    return [line for line in r.stdout.splitlines() if line.strip()]


def apply_resolutions(repo: str, step: Step, conflicts: List[str]) -> List[str]:
    """Resolve predicted conflicts to their mapped resolutions. Returns the
    list of paths needing MANUAL work (which stops the run, resumably)."""
    manual: List[str] = []
    for path in conflicts:
        resolution = step.predicted[path]
        if resolution == "ours":
            run(repo, "checkout", "--ours", "--", path)
            run(repo, "add", "--", path)
            log(f"  resolved {path} -> ours (incoming hunk dropped, as mapped)")
        elif resolution == "theirs":
            run(repo, "checkout", "--theirs", "--", path)
            run(repo, "add", "--", path)
            with open(os.path.join(repo, path)) as f:
                text = f.read()
            missing = [m for m in step.markers.get(path, []) if m not in text]
            if missing:
                raise Abort(
                    f"step {step.name}: 'theirs' resolution of {path} LOST "
                    f"the load-bearing markers {missing} -- the mapped "
                    "resolution does not hold; re-derive the map."
                )
            log(f"  resolved {path} -> theirs (markers verified)")
        else:  # manual
            manual.append(path)
    return manual


def execute_step(repo: str, step: Step, dry_run: bool) -> Optional[str]:
    """Run one step. Returns None on success or a 'manual' status string."""
    if step.kind == "skip":
        log(f"step {step.name}: already in tip, skipping")
        return None
    log(f"step {step.name}: {step.kind} {step.ref}  ({step.note})")
    if step.kind == "merge":
        r = run(repo, "merge", "--no-ff", "--no-edit", step.ref, check=False)
    else:
        r = run(repo, "cherry-pick", "-x", step.ref, check=False)
    if r.returncode != 0:
        conflicts = conflicted_paths(repo)
        if not conflicts:
            raise Abort(
                f"step {step.name}: git failed without conflicts:\n{r.stderr[-1500:]}"
            )
        unpredicted = [p for p in conflicts if p not in step.predicted]
        if unpredicted:
            raise Abort(
                f"step {step.name}: UNPREDICTED conflict in {unpredicted} "
                f"(predicted set: {sorted(step.predicted)}). The map is "
                "stale. Aborting WITHOUT resolving -- run "
                f"`git -C {repo} {'merge' if step.kind == 'merge' else 'cherry-pick'} --abort`, "
                "re-derive LEDGER_merge_train_0818, then resume."
            )
        manual = apply_resolutions(repo, step, conflicts)
        if manual:
            return (
                f"manual resolution required in {manual} (as the ledger "
                f"predicted). Re-apply the hunks by hand, `git -C {repo} add` "
                "them, run `git -C {repo} cherry-pick --continue`, then "
                "re-run with --resume."
            )
        cont = ["merge", "--continue"] if step.kind == "merge" else [
            "cherry-pick", "--continue"
        ]
        env_over = dict(os.environ)
        env_over["GIT_EDITOR"] = "true"
        rc = subprocess.run(
            ["git", "-C", repo, *cont], capture_output=True, text=True,
            env=env_over,
        )
        if rc.returncode != 0:
            staged = run(repo, "diff", "--cached", "--quiet", check=False)
            if step.kind == "cherry" and staged.returncode == 0:
                # Every predicted hunk resolved to 'ours': the pick is now
                # EMPTY, which --continue refuses. Skipping is the mapped
                # outcome -- the tip already carries the content.
                run(repo, "cherry-pick", "--skip")
                log("  pick became empty after mapped resolutions; skipped")
            else:
                raise Abort(
                    f"step {step.name}: --continue failed:\n{rc.stderr[-1200:]}"
                )
    if step.suites and not dry_run:
        env = dict(os.environ)
        env.update(HERMETIC_ENV)
        env["PYTHONPATH"] = os.path.join(repo, "python")
        r = subprocess.run(
            PYTEST + step.suites, cwd=repo, env=env, capture_output=True, text=True
        )
        if r.returncode != 0:
            raise Abort(
                f"step {step.name}: suite FAILED after the pick -- stopping "
                f"(stop-on-first-failure).\n{r.stdout[-2000:]}"
            )
        log(f"  suites green: {', '.join(step.suites)}")
    elif step.suites:
        log(f"  dry-run: suites skipped ({', '.join(step.suites)})")
    return None


# ----------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, help="the train worktree")
    p.add_argument("--tip", default=DEFAULT_TIP,
                   help="expected base tip (default comp4)")
    p.add_argument("--marker", default=DEFAULT_MARKER)
    p.add_argument("--state", default=None,
                   help="state file (default <repo>/.merge_train_pass3.json)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="execute the plan in a scratch shared clone of "
                   "--repo; suites skipped; the real repo is untouched")
    args = p.parse_args()

    try:
        check_gate(args.marker)
        repo = args.repo
        if args.dry_run:
            import tempfile

            scratch = tempfile.mkdtemp(prefix="pass3-dryrun-")
            log(f"dry-run: shared scratch clone at {scratch}")
            subprocess.run(
                ["git", "clone", "--shared", "--no-checkout", args.repo, scratch],
                capture_output=True, text=True, check=True,
            )
            run(scratch, "checkout", "--detach", args.tip)
            repo = scratch

        state_path = args.state or os.path.join(repo, ".merge_train_pass3.json")
        state = load_state(state_path)
        if state["status"] == "manual-stop" and not args.resume:
            raise Abort(
                f"state file {state_path} records a manual stop at step "
                f"{state['current']}; finish the resolution and re-run "
                "with --resume."
            )

        head = run(repo, "rev-parse", "HEAD").stdout.strip()
        if not args.resume:
            r = run(repo, "merge-base", "--is-ancestor", args.tip, head, check=False)
            if r.returncode != 0:
                raise Abort(
                    f"HEAD {head[:12]} does not contain the expected tip "
                    f"{args.tip}; check out the train branch first."
                )
        check_lineage(repo, head)

        for step in PLAN:
            if step.name in state["done"]:
                log(f"step {step.name}: done earlier, skipping")
                continue
            state["current"] = step.name
            state["status"] = "running"
            save_state(state_path, state)
            manual_msg = execute_step(repo, step, args.dry_run)
            if manual_msg:
                state["status"] = "manual-stop"
                save_state(state_path, state)
                log(f"STOP (resumable): {manual_msg}")
                return 3
            state["done"].append(step.name)
            state["current"] = None
            state["status"] = "between-steps"
            save_state(state_path, state)

        state["status"] = "complete"
        save_state(state_path, state)
        log("pass 3 plan complete. Nothing was pushed -- review, run the "
            "full regression, then push by hand.")
        return 0
    except Abort as e:
        print(f"[pass3] ABORT: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
