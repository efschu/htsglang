#!/usr/bin/env python3
"""#727 three-arm vocab-int8 A/B: the turnkey runner for the window.

WHAT DECIDES. The lm_head half of #727 carries a HYPOTHESIS (per-channel
int8 error lands on logit differences, near-ties can flip); the desk does not
get to spend that hypothesis as a veto, the measurement does
(TICKET_727_vocab_int8.md). This runner executes the ticket's protocol
exactly: arms A (baseline), B (embed-only), C (both), with arm A run TWICE
first because the model is not deterministic across boots -- every B/C
verdict is read against A's own boot-to-boot floor.

WHY C MATTERS BEYOND QUALITY (#735): the full-plan slot arithmetic
(DESIGN_pp_layer_set.md) prices GDN slots 21-24 on the 5090 AFTER the #727
int8 vocab saving, and the full plan places lm_head ON the 5090 -- so arm C
is the arm that funds those slots; on BF16 lm_head the same card supports ~7.
If C fails GATE B, the 21-24 slot claim loses its funder and that consequence
is part of the verdict, not a footnote.

SEQUENCE, stop at the first failure:

    verify-artifacts   shard dtypes/scales/ignore/hardlinks, BEFORE any boot
                       spends a window on a wrong checkpoint
    A1, A2             baseline twice -> the A-vs-A floor
    B                  embed-only
    C                  both

Per boot: GATE 0 (int8 path ENGAGED -- counts the ct_embedding load lines:
0/1/2 for A/B/C; a silent dense fallback proves nothing and shows a WORSE
VRAM number), GATE A (per-stage VRAM delta ~1212 MiB where the arm promises
it), GATE B (suite score + determined-answer accuracy within the A floor),
GATE C (TTFT/decode within the A floor; C is where an lm_head dequant-GEMM
regression would appear).

DESK-WRITTEN-NEVER-EXECUTED rule: this file boots nothing at the desk. The
boot/suite/perf legs are COMMAND TEMPLATES the window operator fills
(--boot-cmd/--suite-cmd/--perf-cmd), and ``--mock DIR`` replaces all three
with fixture JSONs so the orchestration, gates, floor arithmetic, decision
rule and stop logic are smoke-run hermetically (test_ab_runner_727.py drives
every gate in both directions).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MIB = 1 << 20
#: Per-tensor saving the requant promises (TICKET_727 table).
EXPECTED_SAVING_MIB = 1212.0
ENGAGED_RE = re.compile(r"INT8-VOCAB ENGAGED")

ARMS = ("A1", "A2", "B", "C")
ARM_MODEL = {
    "A1": "Qwen3.8-27B-INT8-yarn1.5",
    "A2": "Qwen3.8-27B-INT8-yarn1.5",
    "B": "Qwen3.8-27B-INT8-vocabint8-embed",
    "C": "Qwen3.8-27B-INT8-vocabint8-both",
}
#: GATE 0: number of ct_embedding engagement lines each arm's boot must show.
ARM_ENGAGED_LINES = {"A1": 0, "A2": 0, "B": 1, "C": 2}
#: GATE A: (PP0 delta expected, PP2 delta expected) in MiB vs arm A1.
ARM_VRAM_DELTA = {"B": (-EXPECTED_SAVING_MIB, 0.0), "C": (-EXPECTED_SAVING_MIB, -EXPECTED_SAVING_MIB)}


class GateFailure(RuntimeError):
    pass


# ---------------------------------------------------------------- artifacts


def _shard_header(path: str) -> Dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def verify_artifacts(models_root: str) -> List[str]:
    """Shard-level proof the two requant artifacts are what the arms claim,
    run BEFORE any boot -- a window spent on a wrong checkpoint is the most
    expensive possible way to discover a build defect.

    Checks (corrected shard table, TICKET_727: embed -> 00003,
    lm_head -> 00018): dtype I8 [248320, 5120] beside BF16 [248320, 1]
    scales exactly where each arm promises them, the matching ignore entries
    dropped, and the hardlink economy (B and C share every shard except C's
    rewritten 00018)."""
    lines = []
    emb_dir = os.path.join(models_root, ARM_MODEL["B"])
    both_dir = os.path.join(models_root, ARM_MODEL["C"])

    def _expect(hdr, key, dtype, shape, where):
        t = hdr.get(key)
        if t is None:
            raise GateFailure(f"artifact check: {key} missing in {where}")
        if t["dtype"] != dtype or t["shape"] != shape:
            raise GateFailure(
                f"artifact check: {key} in {where} is {t['dtype']} "
                f"{t['shape']}, expected {dtype} {shape}"
            )

    shape_w = [248320, 5120]
    shape_s = [248320, 1]
    for d, want_lmh_i8 in ((emb_dir, False), (both_dir, True)):
        h3 = _shard_header(os.path.join(d, "model-00003-of-00018.safetensors"))
        h18 = _shard_header(os.path.join(d, "model-00018-of-00018.safetensors"))
        _expect(h3, "model.language_model.embed_tokens.weight", "I8", shape_w, d)
        _expect(
            h3, "model.language_model.embed_tokens.weight_scale", "BF16", shape_s, d
        )
        if want_lmh_i8:
            _expect(h18, "lm_head.weight", "I8", shape_w, d)
            _expect(h18, "lm_head.weight_scale", "BF16", shape_s, d)
        else:
            _expect(h18, "lm_head.weight", "BF16", shape_w, d)
        ignore = (
            json.load(open(os.path.join(d, "config.json")))
            .get("quantization_config", {})
            .get("ignore", [])
        )
        vocab_left = [e for e in ignore if "embed" in e or "lm_head" in e]
        want_left = [] if want_lmh_i8 else ["lm_head"]
        if sorted(vocab_left) != sorted(want_left):
            raise GateFailure(
                f"artifact check: {d} ignore carries {vocab_left}, "
                f"expected {want_left}"
            )
        lines.append(f"artifact OK: {os.path.basename(d)}")

    shared = sum(
        1
        for i in range(1, 19)
        if os.stat(
            os.path.join(emb_dir, f"model-{i:05d}-of-00018.safetensors")
        ).st_ino
        == os.stat(
            os.path.join(both_dir, f"model-{i:05d}-of-00018.safetensors")
        ).st_ino
    )
    if shared != 17:
        raise GateFailure(
            f"artifact check: hardlink economy broken -- B and C share "
            f"{shared}/18 shard inodes, expected 17 (only C's 00018 "
            "rewritten); the arms would not differ in exactly one tensor"
        )
    lines.append("artifact OK: B/C hardlink economy (17 shared, 1 rewritten)")
    return lines


# ---------------------------------------------------------------- gates


def gate0_engaged(boot_log_text: str, arm: str) -> None:
    got = len(ENGAGED_RE.findall(boot_log_text))
    want = ARM_ENGAGED_LINES[arm]
    if got != want:
        raise GateFailure(
            f"GATE 0 [{arm}]: {got} INT8-VOCAB ENGAGED line(s) in the boot "
            f"log, expected {want}. A mismatch means the int8 path did not "
            "engage (or engaged where it must not); the arm proves nothing."
        )


def gateA_vram(vram: Dict[str, float], baseline: Dict[str, float], arm: str, tol_mib: float) -> None:
    if arm not in ARM_VRAM_DELTA:
        return
    for stage, want_delta in zip(("pp0", "pp2"), ARM_VRAM_DELTA[arm]):
        delta = vram[stage] - baseline[stage]
        if abs(delta - want_delta) > tol_mib:
            raise GateFailure(
                f"GATE A [{arm}]: {stage} moved {delta:+.0f} MiB vs baseline, "
                f"expected {want_delta:+.0f} +- {tol_mib:.0f}. A saving "
                "materially below the promise means the method did not "
                "engage on that stage."
            )


@dataclass
class Floor:
    """The A-vs-A floor: the baseline's own boot-to-boot variance, plus a
    configurable absolute epsilon so a floor of exactly zero (two lucky
    identical boots) does not turn GATE B into a bit-identity test the
    baseline itself would fail on the third boot."""

    score: float
    determined: float
    ttft_ms: float
    decode_tps: float
    eps_score: float = 0.0
    eps_perf_frac: float = 0.0

    @classmethod
    def from_arms(cls, a1: Dict, a2: Dict, eps_score: float, eps_perf_frac: float):
        return cls(
            score=abs(a1["score"] - a2["score"]),
            determined=abs(a1["determined"] - a2["determined"]),
            ttft_ms=abs(a1["ttft_ms"] - a2["ttft_ms"]),
            decode_tps=abs(a1["decode_tps"] - a2["decode_tps"]),
            eps_score=eps_score,
            eps_perf_frac=eps_perf_frac,
        )


def gateB_quality(metrics: Dict, a1: Dict, floor: Floor, arm: str) -> None:
    for key, bound in (("score", floor.score), ("determined", floor.determined)):
        delta = a1[key] - metrics[key]  # positive = worse than baseline
        if delta > bound + floor.eps_score:
            raise GateFailure(
                f"GATE B [{arm}]: {key} fell {delta:.4f} below arm A, outside "
                f"the A-vs-A floor {bound:.4f} (+eps {floor.eps_score}). "
                "The quality delta is real at this suite's resolution."
            )


def gateC_perf(perf: Dict, a1: Dict, floor: Floor, arm: str) -> None:
    worse_ttft = perf["ttft_ms"] - a1["ttft_ms"]
    bound_ttft = floor.ttft_ms + floor.eps_perf_frac * a1["ttft_ms"]
    if worse_ttft > bound_ttft:
        raise GateFailure(
            f"GATE C [{arm}]: TTFT +{worse_ttft:.1f} ms vs arm A, outside the "
            f"floor {bound_ttft:.1f} ms."
        )
    worse_tps = a1["decode_tps"] - perf["decode_tps"]
    bound_tps = floor.decode_tps + floor.eps_perf_frac * a1["decode_tps"]
    if worse_tps > bound_tps:
        raise GateFailure(
            f"GATE C [{arm}]: decode -{worse_tps:.2f} tok/s vs arm A, outside "
            f"the floor {bound_tps:.2f} tok/s."
        )


def decision(b_passed: bool, c_passed: bool) -> str:
    """The ticket's rule, verbatim shape, fixed before the run. Includes the
    #735 consequence so the verdict prices what it decides."""
    if not b_passed:
        return (
            "STOP: arm B degraded -- the per-row scheme is not benign even "
            "for a gather; the lm_head question is retired with it, and "
            "#735's 21-24 GDN slots stay unfunded (~7 on BF16 vocab)."
        )
    if c_passed:
        return (
            "SHIP BOTH: the logit-difference hypothesis is REFUTED by "
            "measurement; record it as such. Arm C funds #735's GDN slots "
            "21-24 on the 5090 (lm_head lives there)."
        )
    return (
        "SHIP EMBED-ONLY: the logit-difference hypothesis is CONFIRMED; "
        "lm_head stays BF16 and the priced cost is 1212 MiB on PP2 AND "
        "#735's 21-24 GDN slot plan loses its funder (~7 slots on BF16 "
        "lm_head) -- the full-plan ladder must be re-derived."
    )


# ---------------------------------------------------------------- driving


@dataclass
class ArmResult:
    arm: str
    model: str
    gates: Dict[str, str] = field(default_factory=dict)
    metrics: Optional[Dict] = None
    vram: Optional[Dict] = None


def _run_template(template: str, model_path: str, arm: str) -> str:
    cmd = template.format(model_path=model_path, arm=arm)
    out = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=7200
    )
    if out.returncode != 0:
        raise GateFailure(f"[{arm}] command failed rc={out.returncode}: {cmd}\n{out.stderr[-2000:]}")
    return out.stdout


def _load_arm_fixture(mock_dir: str, arm: str) -> Dict:
    with open(os.path.join(mock_dir, f"{arm}.json")) as f:
        return json.load(f)


def run(args) -> int:
    results = {"arms": [], "verdict": None, "floor": None}
    print("== verify-artifacts")
    for line in verify_artifacts(args.models_root):
        print("  ", line)

    arm_data: Dict[str, Dict] = {}
    passed: Dict[str, bool] = {}
    for arm in ARMS:
        model_path = os.path.join(args.models_root, ARM_MODEL[arm])
        print(f"== arm {arm}: {ARM_MODEL[arm]}")
        if args.mock:
            data = _load_arm_fixture(args.mock, arm)
        else:
            boot_out = _run_template(args.boot_cmd, model_path, arm)
            suite_out = _run_template(args.suite_cmd, model_path, arm)
            perf_out = _run_template(args.perf_cmd, model_path, arm)
            data = {
                "boot_log": boot_out,
                "vram": json.loads(suite_out).get("vram"),
                "metrics": json.loads(suite_out),
                "perf": json.loads(perf_out),
            }
        arm_data[arm] = data
        res = ArmResult(arm=arm, model=ARM_MODEL[arm])
        try:
            gate0_engaged(data["boot_log"], arm)
            res.gates["gate0"] = "pass"
            if arm in ARM_VRAM_DELTA:
                gateA_vram(data["vram"], arm_data["A1"]["vram"], arm, args.vram_tol_mib)
            res.gates["gateA"] = "pass"
            if arm in ("B", "C"):
                floor = Floor.from_arms(
                    arm_data["A1"]["metrics"],
                    arm_data["A2"]["metrics"],
                    args.eps_score,
                    args.eps_perf_frac,
                )
                results["floor"] = vars(floor)
                gateB_quality(data["metrics"], arm_data["A1"]["metrics"], floor, arm)
                res.gates["gateB"] = "pass"
                gateC_perf(data["perf"], arm_data["A1"]["perf"], floor, arm)
                res.gates["gateC"] = "pass"
            passed[arm] = True
        except GateFailure as e:
            print("  FAIL:", e)
            res.gates["failure"] = str(e)
            passed[arm] = False
            results["arms"].append(vars(res))
            if arm in ("A1", "A2"):
                # No floor without two clean baseline boots: nothing below
                # is readable. Stop the whole run.
                results["verdict"] = f"ABORT: baseline arm {arm} failed its gates"
                break
            continue
        results["arms"].append(vars(res))

    if results["verdict"] is None and "B" in passed:
        results["verdict"] = decision(passed.get("B", False), passed.get("C", False))
    print("== verdict:", results["verdict"])
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)
    print("results written:", args.results)
    return 0 if results["verdict"] and not results["verdict"].startswith("ABORT") else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--models-root",
        default="/spinning/llm_stuff/club-3090/models-cache",
    )
    p.add_argument(
        "--boot-cmd",
        default=None,
        help="Template booting one arm and printing the boot log to stdout; "
        "{model_path} and {arm} are substituted. Filled by the window "
        "operator; required unless --mock.",
    )
    p.add_argument(
        "--suite-cmd",
        default=None,
        help="Template running the club-3090 suite + determined probes "
        "against the running arm; must print JSON with keys score, "
        "determined, vram{pp0,pp1,pp2}.",
    )
    p.add_argument(
        "--perf-cmd",
        default=None,
        help="Template printing JSON with keys ttft_ms, decode_tps.",
    )
    p.add_argument("--results", default="ab_vocab_int8_results.json")
    p.add_argument("--vram-tol-mib", type=float, default=200.0)
    p.add_argument("--eps-score", type=float, default=0.0)
    p.add_argument("--eps-perf-frac", type=float, default=0.05)
    p.add_argument(
        "--mock",
        default=None,
        help="Fixture dir with A1/A2/B/C.json (boot_log, vram, metrics, "
        "perf); replaces every command template -- the desk smoke.",
    )
    args = p.parse_args()
    if not args.mock and not (args.boot_cmd and args.suite_cmd and args.perf_cmd):
        p.error("without --mock, --boot-cmd, --suite-cmd and --perf-cmd are required")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
