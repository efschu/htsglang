# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``report_effective`` -- the resolved config of a booted arm (#349).

The generalisation of ``scripts/dual_group/dcp_report.sh`` to every matrix
axis. Its one load-bearing rule is that script's rule:

    Read the resolved geometry FROM THE SERVER'S OWN LOG, never re-derive it
    from the launch flags.

That is not a stylistic preference. #340 published "the deviation is
uneven-TP-specific" because it inferred each arm's DCP size from the flag it
passed, while the shared harness environment silently set ``SGLANG_UNEVEN_DCP``
-- so the ratio arm ran at ``dcp_size=2`` and the control at ``dcp_size=1``,
and the flag was carrying a second, undeclared change. The scheduler prints the
size it actually resolved; taking it from there is the difference between a bug
net and a second copy of the same wrong inference.

Everything here is a pure function of log text: no server, no card, no
endpoint. It works on a truncated log from a boot that never came up, which is
exactly when its answer matters most (a hung boot's log still says what it
thought it was configured for).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EffectiveConfig:
    """What an arm actually resolved to, read back from its server log.

    Every field is Optional because a truncated or foreign log may not carry
    the line that sets it; ``None`` means "the log did not say", which the
    check treats differently from a value that disagrees with what the arm
    declared.
    """

    tp_size: Optional[int] = None
    dcp_size: Optional[int] = None
    #: True when the weighted token-sharded KV path actually engaged, read from
    #: the scheduler's own "Uneven DCP: auto-set dcp_size" / token-sizing line
    #: -- NOT inferred from SGLANG_UNEVEN_DCP being set in the environment,
    #: which is precisely the inference #340 got wrong.
    dcp_engaged: Optional[bool] = None
    rank_tp_ratio: Optional[str] = None
    token_vector: Optional[str] = None
    spec_algorithm: Optional[str] = None  # resolved alias (NEXTN -> EAGLE)
    eagle_topk: Optional[int] = None
    cross_algorithm: Optional[bool] = None
    draft_kv_layout: Optional[str] = None
    offload: Optional[bool] = None
    dual_group_lane: Optional[bool] = None
    #: The barlink transport actually in force ("device"|"bar1"|...|"nccl").
    #: "nccl" means barlink was off and stock NCCL carried the collectives.
    barlink: Optional[str] = None
    #: True when full CUDA graphs were captured (not eager).
    graphs: Optional[bool] = None
    #: Whether the boot reached the ready marker at all. Not a config fact, but
    #: it is read from the same log in the same pass, and the check needs it.
    ready: bool = False
    #: Free-form resolved facts a reader wants in the per-arm report line but
    #: the matrix does not gate on (e.g. per-rank budgets, ownership vector).
    extra: Dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        """The one line printed per arm, always -- the dcp_report contract,
        widened to every axis."""
        parts = [
            f"tp={self.tp_size}",
            f"dcp={self.dcp_size}",
            f"dcp_engaged={self.dcp_engaged}",
            f"ratio={self.rank_tp_ratio}",
            f"vector={self.token_vector}",
            f"spec={self.spec_algorithm}",
            f"topk={self.eagle_topk}",
            f"cross_algo={self.cross_algorithm}",
            f"draft_kv={self.draft_kv_layout}",
            f"offload={self.offload}",
            f"dual_lane={self.dual_group_lane}",
            f"barlink={self.barlink}",
            f"graphs={self.graphs}",
            f"ready={self.ready}",
        ]
        return " ".join(parts)

    def to_json(self) -> dict:
        return {
            "tp_size": self.tp_size,
            "dcp_size": self.dcp_size,
            "dcp_engaged": self.dcp_engaged,
            "rank_tp_ratio": self.rank_tp_ratio,
            "token_vector": self.token_vector,
            "spec_algorithm": self.spec_algorithm,
            "eagle_topk": self.eagle_topk,
            "cross_algorithm": self.cross_algorithm,
            "draft_kv_layout": self.draft_kv_layout,
            "offload": self.offload,
            "dual_group_lane": self.dual_group_lane,
            "barlink": self.barlink,
            "graphs": self.graphs,
            "ready": self.ready,
            "extra": dict(self.extra),
        }


# The server prints its parsed ServerArgs as one long ``server_args=ServerArgs(
# ... )`` line. We read the RESOLVED fields from there and from the scheduler's
# own geometry / capture lines -- never from the flags the sweep passed.
READY_MARKER = "The server is fired up and ready to roll!"

_INT_FIELDS = {
    "tp_size": r"tp_size=(\d+)",
    "dcp_size": r"dcp_size=(\d+)",
    "eagle_topk": r"speculative_eagle_topk=(\d+)",
}


def _search_int(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _search_bool_field(text: str, field_name: str) -> Optional[bool]:
    """Read ``field_name=True|False|None`` from the ServerArgs dump."""
    m = re.search(rf"\b{field_name}=(True|False|None)\b", text)
    if not m:
        return None
    return {"True": True, "False": False, "None": None}[m.group(1)]


def _resolved_spec_algorithm(text: str) -> Optional[str]:
    """The RESOLVED algorithm, not the flag. NEXTN is rewritten to EAGLE by
    handle_speculative_decoding before this line is printed, so reading the
    dump gives the resolved value for free -- the whole point of not parsing
    the launch flag."""
    m = re.search(r"speculative_algorithm='?([A-Za-z0-9_]+)'?", text)
    if not m:
        return None
    val = m.group(1)
    return None if val == "None" else val


def report_effective(log_text: str) -> EffectiveConfig:
    """Parse a server log into its resolved configuration.

    Pure function of the text. Reads the scheduler's own resolved lines, so an
    arm that declared one thing and resolved to another is visible here even
    when both come from the same launch.
    """
    ready = READY_MARKER in log_text

    tp_size = _search_int(log_text, _INT_FIELDS["tp_size"])
    dcp_size = _search_int(log_text, _INT_FIELDS["dcp_size"])
    eagle_topk = _search_int(log_text, _INT_FIELDS["eagle_topk"])

    # dcp_engaged: the SCHEDULER's own statement that token-sharding turned on,
    # not the environment variable. Both lines below are printed only when the
    # weighted owner rule actually installed a plan.
    dcp_engaged = bool(
        re.search(r"Uneven DCP: auto-set dcp_size", log_text)
        or re.search(r"Uneven-DCP token sizing", log_text)
    )
    # If neither appears but dcp_size>1 is in the dump, the path did NOT engage
    # (the #345 silent-no-op shape) -- report False, not None, so the check can
    # catch an arm that thought it was token-sharding and was not.
    if not dcp_engaged and dcp_size is not None:
        dcp_engaged = False

    ratio_m = re.search(r"rank_tp_ratio=(None|\[[0-9, ]*\]|'[^']*')", log_text)
    rank_tp_ratio = ratio_m.group(1) if ratio_m else None

    vec_m = re.search(r"active vector (\[[0-9, ]*\])|vector (\[[0-9, ]*\])", log_text)
    token_vector = None
    if vec_m:
        token_vector = vec_m.group(1) or vec_m.group(2)

    spec_algorithm = _resolved_spec_algorithm(log_text)
    cross_algorithm = _search_bool_field(log_text, "speculative_cross_algorithm")
    offload = _search_bool_field(log_text, "enable_kv_session_offload")
    dual_group_lane = _search_bool_field(log_text, "dual_group_lane")

    draft_m = re.search(r"draft_kv_layout='?([A-Za-z_]+)'?", log_text)
    draft_kv_layout = draft_m.group(1) if draft_m else None

    # barlink: the transport the groups ACHIEVED, which is not the same
    # question as whether a transport came up.
    #
    # This used to match r"barlink[^\n]*transport[=: ]+([a-z0-9]+)" and read
    # the word after "transport" -- on a real log that is the line
    # "barlink device transport up", so the axis resolved to "up" and arm
    # E_barlink failed against its own correct declaration of "device". The
    # same log also carries "barlink shm transport up", so whichever came
    # first would have won.
    #
    # The authoritative line is per group and names both sides:
    #   barlink enabled for group 'tp:0': requested=device, ACHIEVED=device
    # A group that fell back prints ACHIEVED=gloo there instead. Every group
    # is read, not the first: if they DISAGREE the run is mixed, and a mixed
    # run must not be flattened into one reassuring name -- that is the #366
    # gate lesson. It is reported as "mixed:<a>+<b>" so the expect comparison
    # fails loudly instead of silently matching one of them.
    barlink: Optional[str] = None
    achieved = {m.lower() for m in re.findall(r"ACHIEVED=([A-Za-z0-9_]+)", log_text)}
    if len(achieved) == 1:
        barlink = achieved.pop()
    elif len(achieved) > 1:
        barlink = "mixed:" + "+".join(sorted(achieved))
    elif re.search(r"using nccl==", log_text):
        barlink = "nccl"

    # graphs: the capture-begin line is printed for every captured phase. Eager
    # runs never print it; a disabled-graph run prints the disable notice.
    #
    # ONLY THE TARGET WORKER'S LINE COUNTS. model_runner.py:3907 builds the
    # role as `"draft" if self.is_draft_worker else "target"`, so a spec boot
    # prints BOTH roles. The previous pattern covered `(draft )?` and neither
    # the `target` role nor the `verify` phase, which meant every boot arm --
    # all of which run with speculation -- confirmed `graphs=True` from the
    # DRAFT model alone, and the matrix never once observed whether the TARGET
    # model captured. A target capture that silently fell back to eager while
    # the draft captured went green. `BASE_EXPECT["graphs"]` is a claim about
    # the SERVED model, so draft evidence is not evidence for it.
    #
    # A roleless line is a pre-role-prefix log; those were target captures, so
    # historical artifacts stay readable.
    graphs: Optional[bool] = None
    if re.search(
        r"Capture (?:target )?(?:decode|extend|prefill|verify) CUDA graph", log_text
    ):
        graphs = True
    elif re.search(r"Disable(d)? cuda graph|disable_cuda_graph=True", log_text):
        graphs = False

    return EffectiveConfig(
        tp_size=tp_size,
        dcp_size=dcp_size,
        dcp_engaged=dcp_engaged if dcp_size is not None else None,
        rank_tp_ratio=rank_tp_ratio,
        token_vector=token_vector,
        spec_algorithm=spec_algorithm,
        eagle_topk=eagle_topk,
        cross_algorithm=cross_algorithm,
        draft_kv_layout=draft_kv_layout,
        offload=offload,
        dual_group_lane=dual_group_lane,
        barlink=barlink,
        graphs=graphs,
        ready=ready,
    )


def first_refusal(log_text: str, markers: List[str]) -> Optional[str]:
    """The ONE refusal message that carries ALL the given markers, or None.

    A reject arm must be refused by the guard its crossing names, and this
    function is what decides whether that happened. It used to decide it far
    too loosely: it asked whether every marker appeared ANYWHERE in the whole
    log, then returned the first line carrying ANY ONE of them. Two of sweep
    1's six reject arms passed on that -- ``reject_dcp_offload`` died on the
    ``KVSO_ALLOW_SPEC`` bring-up gate and ``reject_dcp_crossalgo`` on a missing
    ``--speculative-cross-algorithm-force``, neither of which is the crossing
    the arm exists to prove, and both refusals happened to contain one marker.
    A reject arm reporting PASS while never reaching its guard is worse than
    no arm: it is a green light nobody earned.

    So: all markers must land in ONE refusal message. A message may wrap over
    several lines in a traceback, so the candidate is the contiguous block
    beginning at a raised-error line and continuing through its indented
    continuation lines -- the same shape argparse and ValueError produce.
    """
    if not markers:
        return None
    for block, head in error_blocks(log_text):
        if all(m in block for m in markers):
            return head[:300]
    return None


def error_blocks(log_text: str) -> List[Tuple[str, str]]:
    """(joined message, first line) for each raised error in the log.

    A block starts on a line naming an exception class and runs while the
    following lines are continuations -- indented, or not starting a new
    stamped/exception line. That keeps a wrapped message together without
    swallowing the next unrelated one, which is exactly the difference between
    asserting a guard and asserting the log.
    """
    starter = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*Error|SystemExit)\b\s*:")
    lines = log_text.splitlines()
    blocks: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        if not starter.search(lines[i]):
            i += 1
            continue
        head = lines[i].strip()
        parts = [head]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if not nxt.strip():
                break
            if starter.search(nxt) or _SERVER_STAMP_RE.match(nxt.strip()):
                break
            if nxt.startswith((" ", "\t")) or not nxt.startswith("["):
                parts.append(nxt.strip())
                j += 1
                continue
            break
        blocks.append((" ".join(parts), head))
        i = j
    return blocks


_SERVER_STAMP_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
