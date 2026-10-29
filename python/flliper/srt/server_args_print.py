"""``--print-config``: the effective ServerArgs as a copy-paste command line.

``/server_info`` already lists every effective value; this turns the same
data back into the ``flliper serve`` flags that reproduce it. A guessed
command line is worse than none, so every flag is VERIFIED: its tokens are
parsed again by a fresh parser and the parsed value must equal the effective
value, otherwise the flag is listed as a comment instead of a command. That
covers the arguments with custom ``type=`` parsers (``--rank-*`` ratios,
CUDA-graph config, ...) without a per-parser renderer. Credentials are never
written; they appear as ``<redacted>`` in a comment.
"""

import argparse
import shlex
from typing import Any, Callable, Dict, List, Optional, Tuple

REDACTED = "<redacted>"
_COMMAND = "flliper serve"


def _is_deprecated(action: argparse.Action) -> bool:
    return type(action).__name__.startswith("Deprecated")


def _value_tokens(action: argparse.Action, value: Any) -> Optional[List[str]]:
    """The argv tokens after the flag for ``value``, or None when it has none."""
    if isinstance(action, argparse.BooleanOptionalAction):
        return []
    if action.nargs == 0:
        return []
    if isinstance(value, (list, tuple)) and action.nargs in ("+", "*"):
        return [str(item) for item in value]
    if isinstance(value, (dict, list, tuple)):
        import json

        return [json.dumps(value)]
    return [str(value)]


def _flag_tokens(action: argparse.Action, value: Any) -> Optional[List[str]]:
    """``[flag, *values]`` that should reproduce ``value``, or None to skip."""
    long_flags = [o for o in action.option_strings if o.startswith("--")]
    if not long_flags:
        return None
    flag = long_flags[0]
    if isinstance(action, argparse.BooleanOptionalAction):
        if not isinstance(value, bool):
            return None
        negative = [o for o in long_flags if o.startswith("--no-")]
        return [flag] if value else ([negative[0]] if negative else None)
    if action.nargs == 0:
        # store_true / store_const style: present means the const
        return [flag] if value == action.const else None
    tokens = _value_tokens(action, value)
    return None if tokens is None else [flag, *tokens]


def _reparsed(
    verifier: argparse.ArgumentParser, tokens: List[str], dest: str
) -> Tuple[bool, Any]:
    """Parse ``tokens`` with the verifier parser; (parsed ok, value of ``dest``)."""
    try:
        namespace, unknown = verifier.parse_known_args(tokens)
    except (SystemExit, argparse.ArgumentError, ValueError, TypeError):
        return False, None
    return not unknown, getattr(namespace, dest, None)


def render_effective_cli(
    make_parser: Callable[[], argparse.ArgumentParser],
    effective: Dict[str, Any],
) -> List[str]:
    """Lines of a shell command (plus ``#`` comment lines) for ``effective``.

    ``effective`` maps argparse dest to the effective value (e.g.
    ``ServerArgs.redacted_dict()``). Only values that differ from the parser
    default are emitted.
    """
    parser = make_parser()
    verifier = make_parser()
    for action in verifier._actions:
        action.required = False
    flags: List[str] = []
    notes: List[str] = []
    for action in parser._actions:
        if not action.option_strings or _is_deprecated(action):
            continue
        if isinstance(action, argparse._HelpAction) or action.dest in ("config",):
            continue
        if action.dest not in effective:
            continue
        value = effective[action.dest]
        if value is None or (
            value == action.default and type(value) is type(action.default)
        ):
            continue
        label = (action.option_strings[0] if action.option_strings else action.dest)
        if value == REDACTED:
            notes.append(f"# {label} <redacted> (credential, set it yourself)")
            continue
        tokens = _flag_tokens(action, value)
        if tokens is None:
            notes.append(f"# {label}: no command-line form for {value!r}")
            continue
        parsed_ok, parsed = _reparsed(verifier, tokens, action.dest)
        if not parsed_ok or parsed != value:
            notes.append(
                f"# {label}: effective value {value!r} does not round-trip "
                f"through the parser (tokens {tokens!r} give {parsed!r})"
            )
            continue
        flags.append(" ".join(shlex.quote(token) for token in tokens))
    command = [_COMMAND] + [f"  {flag}" for flag in flags]
    return [" \\\n".join(command)] + notes
