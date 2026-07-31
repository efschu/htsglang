#!/usr/bin/env python3
"""Verdicts for the #307 card proof (see s307_ceiling_fit.sh).

Reads only harvested artifacts -- a marker grep and a /get_server_info dump --
never a whole server log. Every check prints PASS/FAIL with the number it
judged on; the exit code is the arm's verdict.
"""

import json
import re
import sys

M_FIT = "[auto-mamba] the concurrency target does not fit"
M_SCHED = "does not fit the memory budget"
M_ADMIT = "Dynamic admission limit:"
M_POOL = "[auto-mamba] demand-driven mamba pool"
M_LEDGER = "leaves no GPU memory for the KV cache"
M_RETRACT = "Retract requests"

_FITTED_RE = re.compile(r"were fitted to (\d+)")
_REQUESTED_RE = re.compile(r"requested ceiling (\d+)")
_SLOTS_RE = re.compile(r"max_mamba_cache_size=(\d+) slots")

_results = []


def check(name, ok, detail=""):
    _results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        return f"<<unreadable: {exc}>>"


def _json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _limiter(info):
    """The admission snapshot, wherever /get_server_info puts it."""
    if isinstance(info.get("admission_limiter"), dict):
        return info["admission_limiter"]
    for st in info.get("internal_states") or []:
        if isinstance(st, dict) and isinstance(st.get("admission_limiter"), dict):
            return st["admission_limiter"]
    return {}


def arm_a(markers_path, info_path, requested, *_extra):
    """The boot that died must now be up, fitted, and rank-uniform."""
    markers = _read(markers_path)
    info = _json(info_path)
    lim = _limiter(info)

    check("server answered /get_server_info", bool(info), f"keys={len(info)}")
    check(
        "no budget-exhausted ledger in the log",
        M_LEDGER not in markers,
        "the fit is supposed to prevent exactly this message",
    )
    check(
        "the fit fired on at least one rank",
        markers.count(M_FIT) >= 1,
        f"{markers.count(M_FIT)} rank(s)",
    )
    check(
        "the scheduler reported the gap",
        M_SCHED in markers,
        "requested vs effective ceiling must be said out loud",
    )

    fitted = lim.get("ceiling")
    check(
        f"the limiter ceiling is below the requested {requested}",
        isinstance(fitted, int) and 0 < fitted < requested,
        f"ceiling={fitted}",
    )
    check(
        "the float starts at --max-running-requests, not at the ceiling",
        lim.get("current") == 8 or lim.get("start") == 8,
        json.dumps({k: lim.get(k) for k in ("current", "start", "ceiling", "floor")}),
    )

    # Rank uniformity: every rank that logged a pool size logged the SAME one,
    # and the scheduler's fitted ceiling is that size // mamba_ratio.
    sizes = {int(m) for m in _SLOTS_RE.findall(markers)}
    reported = {int(m) for m in _FITTED_RE.findall(markers)}
    check(
        "all ranks ended on one pool size",
        len(sizes) == 1,
        f"sizes={sorted(sizes)}",
    )
    check(
        "one effective ceiling for the group",
        len(reported) <= 1,
        f"reported={sorted(reported)}",
    )
    check(
        "the scheduler line names the ceiling that was asked for",
        {int(m) for m in _REQUESTED_RE.findall(markers)} in ({requested}, set()),
        f"requested={sorted({int(m) for m in _REQUESTED_RE.findall(markers)})}",
    )
    return 0 if all(_results) else 1


def arm_b(raise_path, info_path, requested, markers_path=None):
    """The float must actually rise above its start, and stop at the fit."""
    probe = _json(raise_path)
    lim = _limiter(_json(info_path))
    end = probe.get("end_state") or {}
    fitted = end.get("ceiling") or lim.get("ceiling") or 0
    retracts = _read(markers_path).count(M_RETRACT) if markers_path else 0

    peak = probe.get("peak_limit")
    check("the raise probe produced a trajectory", bool(probe.get("samples")))
    check(
        "the float rose above its start of 8",
        isinstance(peak, int) and peak > 8,
        f"peak={peak}",
    )
    check(
        "the float never exceeded the fitted ceiling",
        isinstance(peak, int) and 0 < peak <= max(fitted, 1),
        f"peak={peak} fitted={fitted}",
    )
    check(
        "the pressure phase throttled",
        (probe.get("throttle_count") or 0) > 0,
        f"throttle_count={probe.get('throttle_count')}",
    )
    check(
        "throttling came before retraction, i.e. nothing was retracted",
        retracts == 0,
        f"'{M_RETRACT}' lines={retracts}",
    )
    check(
        "every request was served",
        probe.get("failed", 1) == 0,
        f"failed={probe.get('failed')}",
    )
    return 0 if all(_results) else 1


def arm_c(markers_path, info_path, requested, *_extra):
    """The carrying run must be untouched."""
    markers = _read(markers_path)
    lim = _limiter(_json(info_path))
    sizes = {int(m) for m in _SLOTS_RE.findall(markers)}

    check("no fit line anywhere", M_FIT not in markers)
    check("no scheduler gap line", M_SCHED not in markers)
    check(
        "the pool is the 100 slots of the 2026-07-30 run",
        sizes == {100},
        f"sizes={sorted(sizes)}",
    )
    # 100 slots // mamba_ratio 5 = 20 admissible requests, above the
    # requested 16 -- so the ceiling stays the requested one, unfitted.
    check(
        "the limiter ceiling is the requested one, not a fitted one",
        lim.get("ceiling") == requested,
        f"ceiling={lim.get('ceiling')} requested={requested}",
    )
    check("the admission line is present", M_ADMIT in markers)
    return 0 if all(_results) else 1


def main(argv):
    arms = {"arm_a": arm_a, "arm_b": arm_b, "arm_c": arm_c}
    if len(argv) < 5 or argv[1] not in arms:
        print(
            f"usage: {argv[0]} arm_a|arm_b|arm_c <artifact> <server_info> "
            f"<ceiling> [markers]"
        )
        return 2
    print(f"--- {argv[1]} ---")
    return arms[argv[1]](argv[2], argv[3], int(argv[4]), *argv[5:])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
