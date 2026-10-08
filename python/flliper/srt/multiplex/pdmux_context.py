from dataclasses import dataclass, field
from typing import List

import torch
import yaml

STREAM_GROUPS = []
SM_COUNTS = []
SM_GROUP_NUM = 8  # Default number of SM groups
CURRENT_STREAM_IDX = 0
CURRENT_STREAM_GROUP = None


@dataclass
class PDMuxConfig:
    sm_group_num: int = 8
    manual_divisions: List[List[int]] = field(
        default_factory=list
    )  # [prefill_sm, decode_sm, decode_bs_threshold]
    split_forward_token_budget: int = 65536
    decode_bs_divisor: int = 36


def load_pdmux_config(config_path: str) -> PDMuxConfig:
    """Load pdmux configuration from YAML file into a dataclass."""
    if not config_path:
        return PDMuxConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if "sm_group_num" not in raw:
        raise ValueError("Missing required field: sm_group_num")

    if raw["sm_group_num"] < 3:
        raise ValueError("sm_group_num must be >= 3")

    manual_divisions = raw.get("manual_divisions", [])

    expected = raw["sm_group_num"] - 2
    if manual_divisions and len(manual_divisions) != expected:
        raise ValueError(
            f"manual_divisions must have {expected} entries, "
            f"but got {len(manual_divisions)}"
        )

    return PDMuxConfig(
        sm_group_num=raw["sm_group_num"],
        manual_divisions=manual_divisions,
        split_forward_token_budget=raw.get("split_forward_token_budget", 65536),
        decode_bs_divisor=raw.get("decode_bs_divisor", 36),
    )


#: Green-context SM partition granularity per architecture major version:
#: (min_per_part, multiple). Everything from Hopper on shares the Hopper
#: granularity; the table below is therefore a lookup for the OLDER majors
#: only, not the list of supported architectures.
_GREEN_CTX_CONSTRAINTS = {
    6: (1, 1),
    7: (2, 2),
    8: (4, 2),
}
#: Applied to every major >= 9 (Hopper, Blackwell SM100/SM103, SM120, and
#: whatever comes next).
_GREEN_CTX_CONSTRAINTS_HOPPER_ONWARDS = (8, 8)
#: Oldest major this function can answer for at all.
_GREEN_CTX_MIN_MAJOR = min(_GREEN_CTX_CONSTRAINTS)


def get_arch_constraints(compute_capability):
    """Green-context SM partition constraints ``(min_per_part, multiple)``.

    Upstream sgl-project/sglang#32933: the previous body was a closed table
    ending at ``major == 9`` with a ``raise`` for everything else, so
    ``--enable-pdmux`` aborted the scheduler on SM100/SM103 (B200/B300) and
    SM120 (RTX PRO 6000, consumer Blackwell) with "Unsupported compute
    capability: 12.0" -- architectures that are strictly newer than the last
    entry and inherit its granularity.

    An arch table is the wrong shape for a question that is really "what is
    the SM split granularity here" (same family as #417). Newer majors are
    therefore not enumerated: they take the Hopper-onwards granularity, which
    is the direction CUDA has held since green contexts were introduced. If a
    future architecture ever disagrees, the driver rejects the split with its
    own error at ``cuDevSmResourceSplitByCount`` -- a named failure at the
    call that actually knows -- instead of this function refusing to answer.

    Older-than-Pascal majors still raise: there the granularity is not a
    forward-extrapolation, green contexts do not exist at all.
    """
    major, minor = compute_capability
    if major < _GREEN_CTX_MIN_MAJOR:
        raise ValueError(
            f"Unsupported compute capability for PD multiplexing: "
            f"{major}.{minor}. Green contexts require compute capability "
            f"{_GREEN_CTX_MIN_MAJOR}.0 or newer."
        )
    if major in _GREEN_CTX_CONSTRAINTS:
        return _GREEN_CTX_CONSTRAINTS[major]
    return _GREEN_CTX_CONSTRAINTS_HOPPER_ONWARDS


def divide_sm(total_sms, compute_capability, groups):
    """
    :param total_sms: total sm count on a single GPU
    :param compute_capability: (major, minor)
    :return: SM partition group(prefill sm, decode sm)
    """
    min_per_part, multiple = get_arch_constraints(compute_capability)
    possible_values = [
        x
        for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
        if x >= total_sms - x and total_sms - x >= 16
    ]
    if not possible_values:
        raise ValueError(
            f"No valid partitions found for total SMs {total_sms} "
            f"with constraints (min per part: {min_per_part}, multiple: {multiple})"
        )

    if len(possible_values) >= groups:
        step = max(1, len(possible_values) // groups)
        selected_values = possible_values[::step][:groups]
    else:
        selected_values = possible_values

    divisions = []
    for part1 in selected_values:
        part2 = total_sms - part1
        divisions.append((part1, part2))

    divisions.reverse()  # Reverse to have larger prefill SM first

    return divisions


def initialize_stream_groups(gpu_id: int, config: PDMuxConfig):
    from sgl_kernel import spatial

    global STREAM_GROUPS, SM_COUNTS, SM_GROUP_NUM, CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    # for pd_multiplexing, Init stream_groups
    # The card whose SMs are being divided is ``gpu_id`` -- the same one
    # ``get_sm_available`` is asked about. Reading torch's current device here
    # instead was a latent mismatch whenever the two differ (#343).
    device = gpu_id
    total_sm_count = spatial.get_sm_available(gpu_id)
    # (prefill_sm_count, decode_sm_count)
    if config.manual_divisions:
        divisions = [
            (prefill_sm, decode_sm)
            for prefill_sm, decode_sm, _ in config.manual_divisions
        ]
    else:
        divisions = divide_sm(
            total_sm_count,
            torch.cuda.get_device_capability(device),
            config.sm_group_num - 2,
        )

    SM_COUNTS = []
    SM_COUNTS.append((total_sm_count, 0))  # Normal stream for prefill
    SM_COUNTS.extend(divisions)  # Add the divided SM counts
    SM_COUNTS.append((0, total_sm_count))  # Normal stream for decode
    STREAM_GROUPS = []
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for prefill
    for prefill_sm, decode_sm in divisions:
        STREAM_GROUPS.append(
            (spatial.create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id))
        )
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for decode

    CURRENT_STREAM_IDX = 0
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def set_current_stream_idx(idx: int):
    global CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    if idx < 0 or idx >= len(STREAM_GROUPS):
        raise ValueError(f"Invalid stream index: {idx}")
    CURRENT_STREAM_IDX = idx
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def get_stream_groups() -> list[tuple[torch.cuda.Stream, torch.cuda.Stream]]:
    """Get the stream groups."""
    return STREAM_GROUPS


def get_sm_counts() -> list[tuple[int, int]]:
    """Get the SM counts."""
    return SM_COUNTS


def get_current_stream_idx() -> int:
    """Get the current stream index."""
    return CURRENT_STREAM_IDX
