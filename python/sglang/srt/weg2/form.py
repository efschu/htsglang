"""WEG2-FORM: the boot's FORM AXES -- one resolver, one line, one env, one accessor.

User order 2026-09-24 (verbatim): "da muessen schalter rein, in den code, dense,
moe, vollstaendig im vram, mit offload - oder sowas aehnliches oder mehr oder
weniger (ergruende/begruende). sonst knallts doch an jeder stelle".

WHY THIS MODULE EXISTS.  The Qwen3.8-27B form never booted on the Next-Flash
line, and every probe of 24.09. died on an NF assumption in SHARED code that
keyed on an INCIDENTAL fact instead of a declared form: a group-env pop hanging
on the else of the expert-map branch (xsn412), P's draft flags read from the
NEXTN constants (xsn414/415), a #66 instrument only the NEXTN producer prints
(xsn417) -- and, worse because silent, the NF MEASUREMENTS (the #114 prefill
transient, the #1444 D residue record, the P logs the cut and the depth are
calibrated from) read by the 27B because "newest file" was the only key
(FORM_AXES_INVENTORY.md, 24.09.).

THE AXES (each derived from the flags that actually build the argv, never
typed twice):

``arch``     dense | moe            -- the checkpoint (config ``num_experts``)
``experts``  none | resident | offload
                                    -- dense -> none; moe -> offload as soon as
                                       an expert store or a resident fraction
                                       < 1 is stated, else resident.  Not named
                                       ``weights``: ``--flip-weights resident``
                                       already means "weights never flip".
``draft``    dflash | mtp | none    -- ``--spec-form`` (DFLASH | NEXTN)
``p_draft``  compute | cold | none  -- ``--draft-kv-on-p`` x ``--dflash-produce-on-p``
                                       x ``--weg2-disable-hicache``
``kv``       paged_dcp | qsa_forma  -- Form A (a ``worker`` rank role on D)
``flip``     family | resident      -- ``--flip-weights``
``vision``   off | resident | transient -- ``--weg2-vision``

A PROFILE (``--profile``) is a named bundle of EXPECTATIONS; a ``--form-*``
flag overrides the expectation; a derived value outside the expectation is
REFUSED by name (W140 Weg2FormContradiction).  ``--form-draft`` and
``--form-p-draft`` additionally DRIVE the legacy flags they map onto when those
were not given explicitly (one switch, one writer: the resolver writes ``ns``
before :func:`launcher.apply_spec_form` reads it); an explicit legacy flag
that disagrees is refused, never silently overridden.

THE CONTRACT TO THE RANKS: ONE environment variable, :data:`FORM_ENV`,
published by the launcher into its own ``os.environ`` (so ``build_env`` and
the front inherit the same value -- the B4k pattern), and ONE accessor,
:func:`current_form`.  ``None`` means "not a weg2 boot" and every form
declaration then applies, so non-weg2 paths never change behaviour.

Torch-free and launcher-free on purpose: ranks import this module.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

FORM_ENV = "SGLANG_WEG2_FORM"
FORM_LINE_TAG = "WEG2-FORM"
W_CONTRADICTION = "W140 Weg2FormContradiction"

AXES: Tuple[str, ...] = ("arch", "experts", "draft", "p_draft", "kv", "flip", "vision")
AXIS_VALUES: Dict[str, Tuple[str, ...]] = {
    "arch": ("dense", "moe"),
    "experts": ("none", "resident", "offload"),
    "draft": ("dflash", "mtp", "none"),
    "p_draft": ("compute", "cold", "none"),
    "kv": ("paged_dcp", "qsa_forma"),
    "flip": ("family", "resident"),
    "vision": ("off", "resident", "transient"),
}
#: The axes an operator can STATE (``--form-<axis>``). ``flip`` and ``vision``
#: already are single flags of their own; a second spelling would be the
#: #1358 class ("one predicate, two call sites").
STATED_AXES: Tuple[str, ...] = ("arch", "experts", "draft", "p_draft", "kv")

PROFILE_QWEN27B = "qwen27b"
PROFILE_NEXTFLASH = "nextflash"

#: Profile = named bundle of EXPECTED values per axis (a set: the 27B runs
#: DFLASH today and ran NEXTN for weeks, both are its forms). Only the stated
#: axes carry expectations; flip/vision are printed, never expected.
PROFILE_EXPECT: Dict[str, Dict[str, Tuple[str, ...]]] = {
    PROFILE_QWEN27B: {
        "arch": ("dense",),
        "experts": ("none",),
        "draft": ("dflash", "mtp"),
        "p_draft": ("compute", "cold", "none"),
        "kv": ("paged_dcp",),
    },
    PROFILE_NEXTFLASH: {
        "arch": ("moe",),
        "experts": ("resident", "offload"),
        # Memory DRAFT-ZUORDNUNG: DFlash2 is the 27B's draft only, NF = MTP.
        "draft": ("mtp",),
        "p_draft": ("compute", "none"),
        "kv": ("qsa_forma",),
    },
}

#: Checkpoint config keys that name routed experts (top level or text_config).
_EXPERT_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
#: Launcher/extra flags whose presence states an expert layout (MoE only).
_EXPERT_FLAGS = (
    "--rank-moe-ratio",
    "--rank-moe-resident-fraction",
    "--pp-cut-expert-device-fraction",
    "--pp-cut-expert-lru-rows",
)
_EXPERT_FRACTION_FLAGS = ("--rank-moe-resident-fraction", "--pp-cut-expert-device-fraction")
_EXPERT_STORE_ENV = "SGLANG_MOE_EXPERT_STORE_DIR"
_EXPERT_FRACTION_ENV = "SGLANG_MOE_RESIDENT_EXPERT_FRACTION"


class Weg2FormContradiction(RuntimeError):
    """W140: the boot's stated/expected form contradicts what its flags build."""


# --------------------------------------------------------------------------
# the value object
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Weg2Form:
    arch: str
    experts: str
    draft: str
    p_draft: str
    kv: str
    flip: str
    vision: str
    profile: str = ""
    model: str = ""
    #: axis -> where the value came from (printed only, never published)
    sources: Tuple[Tuple[str, str], ...] = field(default=(), compare=False)

    def axes(self) -> Dict[str, str]:
        return {a: getattr(self, a) for a in AXES}

    def describe(self) -> str:
        """``arch=dense experts=none ...`` -- the form in one span."""
        return " ".join(f"{a}={getattr(self, a)}" for a in AXES)

    def env_value(self) -> str:
        """The :data:`FORM_ENV` payload: ``,``-separated, no spaces -- NOT
        ``;``, because the launcher's WEG2-GROUP-ENV line and ``--env-p`` /
        ``--env-d`` already use ``;`` between variables."""
        parts = [f"{a}={getattr(self, a)}" for a in AXES]
        parts.append(f"profile={self.profile}")
        parts.append(f"model={self.model}")
        return ",".join(parts)

    def line(self) -> str:
        src = "; ".join(f"{a} <- {s}" for a, s in self.sources)
        return (
            f"{FORM_LINE_TAG} {self.describe()} profile={self.profile} "
            f"model={self.model}" + (f" (sources: {src})" if src else "")
        )

    def matches(self, **want: Iterable[str]) -> bool:
        for axis, allowed in want.items():
            if getattr(self, axis) not in tuple(allowed):
                return False
        return True


def parse_form(value: str) -> Optional[Weg2Form]:
    """Inverse of :meth:`Weg2Form.env_value`; ``None`` for an unusable value."""
    if not value:
        return None
    kv: Dict[str, str] = {}
    for item in str(value).split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            kv[k.strip()] = v.strip()
    if any(a not in kv for a in AXES):
        return None
    if any(kv[a] not in AXIS_VALUES[a] for a in AXES):
        return None
    return Weg2Form(
        **{a: kv[a] for a in AXES},
        profile=kv.get("profile", ""),
        model=kv.get("model", ""),
    )


def current_form(environ: Optional[Mapping[str, str]] = None) -> Optional[Weg2Form]:
    """THE rank-side accessor. ``None`` = no weg2 form published (not a weg2
    boot, or a desk test): callers then keep their pre-form behaviour."""
    env = os.environ if environ is None else environ
    return parse_form(env.get(FORM_ENV, ""))


# --------------------------------------------------------------------------
# gate declarations
# --------------------------------------------------------------------------

#: Every gate/solve that only means something for some forms, and WHICH forms.
#: A gate outside its forms logs ONE line ``<name> SKIPPED (form ...)`` and
#: does not evaluate -- instead of evaluating incidental facts (a vector
#: present or not) and printing "ENTFAELLT" for a form it was never built for.
#: A gate inside its forms prints nothing new, so NF logs stay as they are.
FORM_GATES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "#106 WEG2-STORE-GEOMETRY": {"arch": ("moe",)},
    "#107 WEG2-EXPERT-MAP": {"arch": ("moe",)},
    "#134 WEG2-EXPERT-BAND": {"arch": ("moe",)},
    "#140 PP-CUT FRACTION-SOLVE": {"arch": ("moe",)},
    "#145 D-RANK FRACTION-SOLVE": {"arch": ("moe",)},
    "H14 WAKE-CREDIT": {"arch": ("moe",)},
    "#114 P-PREFILL-TRANSIENT": {},  # keyed on the calibration model, see launcher
}


#: DEFAULT-ON switches of SHARED (form-independent) code that arrived with the
#: Next-Flash line (git diff eb5d04453f 1e49837c84 -- environ.py) and were
#: proven on the metal only by MoE boots. Not switched per form -- there is no
#: evidence against them on the 27B, only no evidence FOR them -- but a boot of
#: another form names them, so a first-flip failure has its candidates and
#: their off-switches in the boot's own log (FORM_AXES_INVENTORY H5).
#: env -> (default, where it acts, the value that switches it off)
NF_LINE_DEFAULT_ON: Dict[str, Tuple[str, str, str]] = {
    "SGLANG_WEG2_SLEEP_RELEASE_LMEM": ("1", "weight_updater sleep: LMEM park", "0"),
    "SGLANG_WEG2_TAG_PLAN_PREWARM": ("1", "weight_updater: tag-plan prewarm", "0"),
    "SGLANG_WEG2_WAKE_LANE_TURNS": ("1", "weight_updater wake: lane turns", "0"),
    "SGLANG_WEG2_WAKE_COLLECT_SPARE": ("1", "weight_updater wake: spare collect worker", "0"),
    "SGLANG_WEG2_FLIP_ORDER_CREDIT": ("1", "front: credit-ordered flip", "0"),
    "SGLANG_WEG2_CREDIT_LIVE_STAGING": ("1", "memory saver: live credit staging", "0"),
    "SGLANG_OPT_HICACHE_DEVICE_INDEX_WRITE": ("1", "hicache write path: device index write", "0"),
    "SGLANG_WEG2_MAMBA_ARENA_RID_ANCHORS": ("-1", "unified radix: mamba arena rid anchors (auto)", "0"),
}
#: The forms those switches were proven on.
NF_LINE_PROVEN = {"arch": ("moe",)}


def first_use_line(form: Optional[Weg2Form], environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The one line naming the NF-line default-on switches THIS form runs
    without a metal proof (None for a form they were proven on)."""
    if form is None or form.matches(**NF_LINE_PROVEN):
        return None
    env = os.environ if environ is None else environ
    parts = []
    for k, (default, where, off) in NF_LINE_DEFAULT_ON.items():
        v = env.get(k)
        if v is not None and str(v).strip() == off:
            continue
        parts.append(f"{k}={v if v is not None else default}{'' if v is not None else '(default)'} [{where}; off: {off}]")
    if not parts:
        return None
    return (f"{FORM_LINE_TAG} FIRST-USE (form {form.describe()}): {len(parts)} default-on switch(es) "
            f"of the Next-Flash line run here without a metal proof on this form -- "
            + "; ".join(parts))


def gate_applies(name: str, form: Optional[Weg2Form]) -> bool:
    if form is None:
        return True
    want = FORM_GATES.get(name)
    if not want:
        return True
    return form.matches(**want)


def gate_skip_line(name: str, form: Optional[Weg2Form]) -> Optional[str]:
    """``None`` if the gate applies to ``form``; else the one SKIPPED line."""
    if gate_applies(name, form):
        return None
    want = " ".join(f"{a}={'|'.join(v)}" for a, v in FORM_GATES[name].items())
    return f"{name} SKIPPED (form {form.describe()}) -- declared for {want} only"


# --------------------------------------------------------------------------
# argv helpers (no argparse: the extra strings are shell-split words)
# --------------------------------------------------------------------------


def flag_values(words: Sequence[str], flag: str) -> List[str]:
    """Every value of ``flag`` in ``words`` (``--f v`` and ``--f=v``), in order."""
    out: List[str] = []
    words = list(words)
    for i, w in enumerate(words):
        if w == flag and i + 1 < len(words):
            out.append(words[i + 1])
        elif w.startswith(flag + "="):
            out.append(w.split("=", 1)[1])
    return out


def flag_given(words: Sequence[str], flag: str) -> bool:
    return any(w == flag or w.startswith(flag + "=") for w in words)


def _fractions(value: str) -> List[float]:
    out: List[float] = []
    for tok in str(value).replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------
# derivations
# --------------------------------------------------------------------------


def model_key(model: str) -> str:
    """The calibration identity of a checkpoint: its directory name."""
    return os.path.basename(str(model or "").rstrip("/").strip("'\""))


def checkpoint_arch(model: str) -> Tuple[Optional[str], str]:
    """``("moe"|"dense", source)`` from the checkpoint config; ``(None, why)``
    when it cannot be read (the profile's expectation then stands, and the
    source says so)."""
    path = os.path.join(str(model or ""), "config.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        return None, f"checkpoint config unreadable ({path}: {type(e).__name__})"
    if not isinstance(cfg, dict):
        return None, f"checkpoint config is not an object ({path})"
    for scope, d in (("", cfg), ("text_config.", cfg.get("text_config"))):
        if not isinstance(d, dict):
            continue
        for k in _EXPERT_CONFIG_KEYS:
            v = d.get(k)
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                return "moe", f"checkpoint {scope}{k}={v}"
    return "dense", "checkpoint config names no routed experts"


def expert_evidence(
    launcher_words: Sequence[str],
    extra_p: Sequence[str],
    extra_d: Sequence[str],
    env_p: Mapping[str, str],
    env_d: Mapping[str, str],
) -> Tuple[List[str], bool]:
    """``(stated expert facts, offload?)`` from the arm's OWN statements (the
    launcher argv, the extra strings, the per-group env) -- never from the
    inherited shell, which a dense boot does not own."""
    facts: List[str] = []
    offload = False
    for words, where in (
        (launcher_words, "launcher"), (extra_p, "extra_p"), (extra_d, "extra_d")
    ):
        for flag in _EXPERT_FLAGS:
            vals = flag_values(words, flag)
            if not vals:
                continue
            facts.append(f"{flag} ({where})")
            if flag in _EXPERT_FRACTION_FLAGS and any(
                x < 1.0 for v in vals for x in _fractions(v)
            ):
                offload = True
    for env, where in ((env_p, "env_p"), (env_d, "env_d")):
        if env.get(_EXPERT_STORE_ENV):
            facts.append(f"{_EXPERT_STORE_ENV} ({where})")
            offload = True
        if env.get(_EXPERT_FRACTION_ENV):
            facts.append(f"{_EXPERT_FRACTION_ENV} ({where})")
            if any(x < 1.0 for x in _fractions(env[_EXPERT_FRACTION_ENV])):
                offload = True
    return facts, offload


def derive_kv(extra_d: Sequence[str]) -> Tuple[str, str]:
    roles = flag_values(extra_d, "--rank-role")
    if roles and "worker" in [r.strip() for r in roles[-1].split(",")]:
        return "qsa_forma", f"--rank-role {roles[-1]} (extra_d): Form A"
    return "paged_dcp", "no worker rank role on D"


# --------------------------------------------------------------------------
# the resolver
# --------------------------------------------------------------------------

_DRAFT_TO_SPEC = {"dflash": "DFLASH", "mtp": "NEXTN"}
_SPEC_TO_DRAFT = {v: k for k, v in _DRAFT_TO_SPEC.items()}


def _refuse(msg: str) -> None:
    raise Weg2FormContradiction(f"{W_CONTRADICTION}: {msg}")


def _apply_stated_draft(ns, words: Sequence[str], sources: Dict[str, str]) -> None:
    stated = getattr(ns, "form_draft", None)
    if not stated:
        return
    if stated == "none":
        _refuse(
            "--form-draft none is not a form this launcher builds: argv_d always "
            "carries a drafter (spec_flags) and P's producer/cold forms are keyed "
            "on it. State --form-draft dflash or mtp."
        )
    want = _DRAFT_TO_SPEC[stated]
    if flag_given(words, "--spec-form"):
        have = str(getattr(ns, "spec_form", "") or "").upper()
        if have != want:
            _refuse(
                f"--form-draft {stated} contradicts the EXPLICIT --spec-form "
                f"{have} (the two name one switch; drop one of them)"
            )
    else:
        ns.spec_form = want
        sources["draft"] = f"--form-draft {stated} (drives --spec-form {want})"


def _apply_stated_p_draft(ns, words: Sequence[str], sources: Dict[str, str]) -> None:
    stated = getattr(ns, "form_p_draft", None)
    if not stated:
        return
    spec = str(getattr(ns, "spec_form", "") or "NEXTN").upper()
    if stated == "cold" and spec != "DFLASH":
        _refuse(
            f"--form-p-draft cold needs --spec-form DFLASH (this boot resolves "
            f"{spec}): only the DFlash producer has a cold form "
            "(--dflash-produce-on-p off); the NEXTN producer computes or is absent"
        )
    want_on_p = "off" if stated == "none" else "on"
    if flag_given(words, "--draft-kv-on-p"):
        if str(getattr(ns, "draft_kv_on_p", "")) != want_on_p:
            _refuse(
                f"--form-p-draft {stated} contradicts the EXPLICIT --draft-kv-on-p "
                f"{getattr(ns, 'draft_kv_on_p', '')}"
            )
    else:
        ns.draft_kv_on_p = want_on_p
    if spec == "DFLASH" and stated in ("compute", "cold"):
        want_produce = "on" if stated == "compute" else "off"
        if flag_given(words, "--dflash-produce-on-p"):
            if str(getattr(ns, "dflash_produce_on_p", "")) != want_produce:
                _refuse(
                    f"--form-p-draft {stated} contradicts the EXPLICIT "
                    f"--dflash-produce-on-p {getattr(ns, 'dflash_produce_on_p', '')}"
                )
        else:
            ns.dflash_produce_on_p = want_produce
    sources["p_draft"] = f"--form-p-draft {stated}"


def resolve_form(
    ns,
    argv_words: Sequence[str],
    *,
    parse_group_env: Callable[[str], Dict[str, str]],
    shlex_split: Callable[[str], List[str]],
    arch_probe: Callable[[str], Tuple[Optional[str], str]] = checkpoint_arch,
) -> Weg2Form:
    """THE resolver. Called ONCE by ``launcher.main`` right after parsing and
    BEFORE ``apply_spec_form`` (``--form-draft``/``--form-p-draft`` may write
    ``ns.spec_form``/``ns.draft_kv_on_p``/``ns.dflash_produce_on_p``).

    Raises :class:`Weg2FormContradiction` (W140) by name; never guesses.
    """
    words = list(argv_words)
    profile = str(getattr(ns, "profile", PROFILE_QWEN27B) or PROFILE_QWEN27B)
    expect = {a: tuple(v) for a, v in PROFILE_EXPECT.get(profile, {}).items()}
    expect_src = {a: f"--profile {profile}" for a in expect}
    for axis in STATED_AXES:
        stated = getattr(ns, f"form_{axis}", None)
        if stated:
            if stated not in AXIS_VALUES[axis]:
                _refuse(f"--form-{axis.replace('_', '-')} {stated!r} is not one of {AXIS_VALUES[axis]}")
            expect[axis] = (stated,)
            expect_src[axis] = f"--form-{axis.replace('_', '-')} {stated}"

    sources: Dict[str, str] = {}
    _apply_stated_draft(ns, words, sources)
    _apply_stated_p_draft(ns, words, sources)

    model = str(getattr(ns, "model", "") or "")
    extra_p = shlex_split(str(getattr(ns, "extra_p", "") or ""))
    extra_d = shlex_split(str(getattr(ns, "extra_d", "") or ""))
    env_p = parse_group_env(str(getattr(ns, "env_p", "") or ""))
    env_d = parse_group_env(str(getattr(ns, "env_d", "") or ""))

    # arch: the checkpoint is the truth; unreadable -> the expectation stands.
    arch, arch_src = arch_probe(model)
    if arch is None:
        fallback = expect.get("arch", ("dense",))[0]
        arch, arch_src = fallback, f"{expect_src.get('arch', 'default')} ({arch_src})"
    sources["arch"] = arch_src

    facts, offload = expert_evidence(words, extra_p, extra_d, env_p, env_d)
    if arch == "dense":
        if facts:
            _refuse(
                f"the checkpoint is DENSE ({arch_src}) but this arm states an "
                f"expert layout: {', '.join(facts)}. Those flags drive the MoE "
                "solves (#106/#107/#140/#145) and the expert store; on a dense "
                "model they are a wrong-profile arm, not a no-op"
            )
        experts, experts_src = "none", "dense checkpoint"
    elif offload:
        experts, experts_src = "offload", ", ".join(facts)
    else:
        experts, experts_src = "resident", (
            "moe checkpoint, no expert store and no resident fraction < 1 stated"
        )
    sources["experts"] = experts_src

    spec = str(getattr(ns, "spec_form", "NEXTN") or "NEXTN").upper()
    draft = _SPEC_TO_DRAFT.get(spec)
    if draft is None:
        _refuse(f"--spec-form {spec!r} maps to no draft axis value")
    sources.setdefault("draft", f"--spec-form {spec}")

    hicache_disabled = bool(getattr(ns, "weg2_disable_hicache", False))
    on_p = str(getattr(ns, "draft_kv_on_p", "on")) == "on" and not hicache_disabled
    if not on_p:
        p_draft = "none"
        why = "--weg2-disable-hicache" if hicache_disabled else "--draft-kv-on-p off"
    elif draft == "dflash":
        produce = str(getattr(ns, "dflash_produce_on_p", "off")).lower() == "on"
        p_draft = "compute" if produce else "cold"
        why = f"--dflash-produce-on-p {'on' if produce else 'off'}"
    else:
        p_draft, why = "compute", "NEXTN producer on P (--draft-kv-on-p on)"
    sources.setdefault("p_draft", why)

    kv, kv_src = derive_kv(extra_d)
    sources["kv"] = kv_src

    flip = str(getattr(ns, "flip_weights", "family") or "family")
    vision = str(getattr(ns, "weg2_vision", "off") or "off")
    sources["flip"] = f"--flip-weights {flip}"
    sources["vision"] = f"--weg2-vision {vision}"

    derived = {"arch": arch, "experts": experts, "draft": draft, "p_draft": p_draft, "kv": kv}
    for axis in STATED_AXES:
        allowed = expect.get(axis)
        if allowed and derived[axis] not in allowed:
            _refuse(
                f"axis {axis}: this boot BUILDS {axis}={derived[axis]} "
                f"({sources.get(axis, '?')}) but {expect_src[axis]} expects "
                f"{'|'.join(allowed)}. Fix the arm, or state the form you mean "
                f"with --form-{axis.replace('_', '-')} (a stated form overrides "
                "the profile's expectation; it never rewrites what the flags build)"
            )

    return Weg2Form(
        arch=arch, experts=experts, draft=draft, p_draft=p_draft, kv=kv,
        flip=flip, vision=vision, profile=profile, model=model_key(model),
        sources=tuple((a, sources[a]) for a in AXES if a in sources),
    )


def add_form_arguments(ap) -> None:
    """The ``--form-*`` switches (all optional; unset = the profile's expectation)."""
    for axis in STATED_AXES:
        ap.add_argument(
            f"--form-{axis.replace('_', '-')}",
            dest=f"form_{axis}",
            choices=list(AXIS_VALUES[axis]),
            default=None,
            help=(
                f"WEG2-FORM axis '{axis}' ({'|'.join(AXIS_VALUES[axis])}). Unset: "
                "the --profile's expectation. Set: overrides it; the boot is "
                f"REFUSED ({W_CONTRADICTION}) when its flags build a different "
                "value. --form-draft/--form-p-draft also DRIVE --spec-form / "
                "--draft-kv-on-p / --dflash-produce-on-p when those are not given."
            ),
        )


# --------------------------------------------------------------------------
# calibration identity: which boot measured which model
# --------------------------------------------------------------------------

_FORM_LINE_MODEL_RE = re.compile(re.escape(FORM_LINE_TAG) + r" .*?\bmodel=(\S+)")
_FORM_LINE_AXES_RE = re.compile(
    re.escape(FORM_LINE_TAG) + r" (arch=\S+ experts=\S+ draft=\S+ p_draft=\S+ kv=\S+ "
    r"flip=\S+ vision=\S+) profile=(\S*) model=(\S+)")
_MODEL_PATH_RE = re.compile(r"--model-path[= ]'?([^\s']+)")
_SPEC_ALGO_RE = re.compile(r"--speculative-algorithm[= ]'?([A-Z_]+)")
_SERVER_ARGS_MODEL_RE = re.compile(r"\bmodel_path='([^']+)'")
#: A front log states its groups' argv within its first few hundred lines; a
#: P log its ServerArgs within the first few thousand. Bounded either way.
_SCAN_MAX_BYTES = 16 << 20


@dataclass(frozen=True)
class BootIdentity:
    """What a boot log says it ran: the checkpoint, and -- when it can tell --
    its form (a WEG2-FORM line) or at least its draft (the groups' argv)."""
    model: Optional[str]
    form: Optional[Weg2Form] = None
    draft: Optional[str] = None


def log_identity(path: str) -> BootIdentity:
    """Scan a boot log's head for its checkpoint and draft (and form line)."""
    model = draft = None
    try:
        with open(path, "rb") as f:
            seen = 0
            for raw in f:
                seen += len(raw)
                if seen > _SCAN_MAX_BYTES:
                    break
                line = raw.decode("utf-8", "replace")
                if FORM_LINE_TAG in line:
                    m = _FORM_LINE_AXES_RE.search(line)
                    if m:
                        kv = dict(x.split("=", 1) for x in m.group(1).split())
                        form = Weg2Form(**kv, profile=m.group(2), model=m.group(3))
                        return BootIdentity(model=form.model, form=form, draft=form.draft)
                    m = _FORM_LINE_MODEL_RE.search(line)
                    if m and model is None:
                        model = m.group(1)
                if "argv" in line or "model_path=" in line:
                    if model is None:
                        m = _MODEL_PATH_RE.search(line) or _SERVER_ARGS_MODEL_RE.search(line)
                        if m:
                            model = model_key(m.group(1))
                    if draft is None and model is not None:
                        m = _SPEC_ALGO_RE.search(line)
                        if m:
                            draft = _SPEC_TO_DRAFT.get(m.group(1))
                if model is not None and draft is not None:
                    break
    except OSError:
        return BootIdentity(model=None)
    return BootIdentity(model=model, draft=draft)


def log_model(path: str) -> Optional[str]:
    """The checkpoint (``model_key``) a boot log says it ran, or ``None``."""
    return log_identity(path).model


@lru_cache(maxsize=4096)
def _group_log_model_cached(path: str, mtime: float) -> Optional[str]:
    front = None
    for suffix in (".P.log", ".D.log"):
        if path.endswith(suffix):
            front = path[: -len(suffix)] + ".front.log"
    if front and os.path.exists(front):
        got = log_model(front)
        if got:
            return got
    return log_model(path)


def group_log_model(path: str) -> Optional[str]:
    """Model of a ``*.P.log``/``*.D.log``: its sibling front log first (it
    carries the argv), the group log's own ServerArgs line second."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    return _group_log_model_cached(path, mtime)


_FRONT_LOG_RE = re.compile(
    r"^boot_weg2_(?P<tag>.+)_(?P<tip>[0-9a-f]{7,40})_(?P<day>\d{4})_(?P<time>\d{6})\.front\.log$"
)


@lru_cache(maxsize=8)
def _front_log_index(evidence_dir: str, _stamp: int) -> Dict[str, Tuple[str, ...]]:
    """tag -> its front logs, newest first. ONE directory listing per
    evidence-dir state instead of one glob per record sample."""
    by_tag: Dict[str, List[Tuple[str, str]]] = {}
    try:
        names = os.listdir(evidence_dir)
    except OSError:
        return {}
    for n in names:
        m = _FRONT_LOG_RE.match(n)
        if m:
            by_tag.setdefault(m.group("tag"), []).append(
                (m.group("day") + m.group("time"), os.path.join(evidence_dir, n)))
    return {t: tuple(p for _, p in sorted(v, reverse=True)) for t, v in by_tag.items()}


@lru_cache(maxsize=4096)
def _front_log_identity_cached(path: str, _mtime: float) -> BootIdentity:
    return log_identity(path)


def boot_tag_identity(tag: str, evidence_dir: str) -> BootIdentity:
    """What the boot ``tag`` (a measured-record ``boot_tag``) ran, read from
    that boot's own front log; ``model=None`` when no log names one."""
    if not tag:
        return BootIdentity(model=None)
    try:
        stamp = int(os.path.getmtime(evidence_dir))
    except OSError:
        return BootIdentity(model=None)
    for path in _front_log_index(evidence_dir, stamp).get(str(tag), ()):
        try:
            got = _front_log_identity_cached(path, os.path.getmtime(path))
        except OSError:
            continue
        if got.model:
            return got
    return BootIdentity(model=None)


def boot_tag_model(tag: str, evidence_dir: str) -> Optional[str]:
    return boot_tag_identity(tag, evidence_dir).model


#: The axes that change what a group leaves on the device and in the host
#: image (the D residue, the dormant image, the run peak): a sample measured
#: under another value of any of them is not this form's measurement.
RESIDUE_AXES: Tuple[str, ...] = ("arch", "experts", "draft", "kv", "flip")


def same_model_sample(
    model: str, evidence_dir: str, form: Optional[Weg2Form] = None
) -> Callable[[dict], bool]:
    """Predicate for measured-record samples: measured on ``model`` (and, with
    ``form``, under the same residue axes)?

    A sample whose boot names no model is NOT accepted -- an unproven
    provenance is exactly the mix-up this predicate exists to stop. A boot
    with a WEG2-FORM line is compared on :data:`RESIDUE_AXES`; an older boot
    (no form line) on its draft when its argv names one, else on its model.
    """
    want = model_key(model)

    def accept(sample: dict) -> bool:
        ident = boot_tag_identity(str(sample.get("boot_tag", "")), evidence_dir)
        if ident.model != want:
            return False
        if form is None:
            return True
        if ident.form is not None:
            return all(getattr(ident.form, a) == getattr(form, a) for a in RESIDUE_AXES)
        return ident.draft is None or ident.draft == form.draft

    return accept


def same_model_log(model: str) -> Callable[[str], bool]:
    """Predicate for ``*.P.log`` calibration sources: this boot's checkpoint?"""
    want = model_key(model)

    def accept(path: str) -> bool:
        return group_log_model(path) == want

    return accept
