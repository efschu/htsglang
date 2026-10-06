"""AP-C ``propose()``: stage A of the profile planner (plan PLAN-PROFIL-PLANER-1006 section 2): hardware + model + operating form +
goals -> a candidate launch (argv + environment) with the ORIGIN of every value.

::

    propose(hardware, modell, form, ziele, *, basis=None, draft=None, rates=None, library=None) -> Vorschlag (dict)

* ``hardware``  a ``flliper.hardware/1`` document (``rigmon.hardware_profile.build``), or a list of card dicts: NVML replay
  rows (``index, uuid, name, total_bytes, cc_major, ...``) or ``rigdash.kartenplan_catalog`` rows (``usable_mib, nvml_name``).
* ``modell``    a ``flliper.model/1`` profile (``weg2/model_profile.estimate``); ``draft`` the optional separate draft profile
  (``model_profile.estimate_draft``).
* ``form``      ``"flip"`` (P = PP<N>, D = TP<N>, the standard form) or ``"tp"`` (``--d-only``: pure tensor parallel).  Dual
  (AP-E) and a single card (AP-F) are other packages: they raise :class:`ProposeError`, they are not guessed.
* ``ziele``     the goals (all optional): ``seats`` (the "Sitze gleichzeitig" regulator = decode-bs target), ``kv_tokens``
  (K1 obligation, default 262144), ``kv_dtype``, ``p_cut`` (``auto`` | ``pin`` | ``seed``), ``d_objective``,
  ``draft_kv_on_p``, ``force_rules`` (derive everything by rule even where the profile's own inventory is the live one).
* ``basis``     the release profile of the model line as its launcher input (``propose_oracle.LaunchInput`` or a dict with
  ``argv``, ``env``, ``vars``): the SCALARS of the form (census, store, host riegel, chunk policy ...) that no hardware decides.
  Without it only the rule-derived flags are produced.

What it decides, by the criteria K1-K5 (rules in ``propose_rules.py``):

* **ONE principle for every positional per-card value of the profile.**  The profile's vectors are measurements or operator
  pins taken on ITS inventory (``PROFILE_INVENTORY``).  When the live inventory is that inventory, they are CARRIED (origin:
  "Profil, gleiches Inventar").  Otherwise each is re-derived: per-card measured values by class maximum with the arch twin as
  BORROWED (``class_rekey``), the P cut by the GEMM rates, the resident expert fractions from what a stage has left, the D side by
  the Form A solve (MoE) or VRAM-proportional TP shares (dense).  A value no rule and no record can give is DROPPED and named
  in ``unbelegt`` -- the launcher then says what it needs (HW-UNCALIBRATED), the planner invents nothing.
* **EVERY positional flag/token of ``launcher.POSITIONAL_VECTOR_FLAGS/TOKENS`` the proposal carries has exactly N entries**
  (:func:`vector_lengths`, checked into ``vector_ok``).
* **The scalars of the profile are kept** (``--d-tp-objective decode-bs1`` of the 27B profile stays) unless a goal names them;
  the rule defaults (``maxkv``, ``--max-kv-per-request`` = KV obligation) fill only what the profile does not say.

A P cut seed is a PIN for the launcher; it goes into the argv only where the profile pins a cut itself (NF) or ``ziele["p_cut"]
== "pin"``; otherwise the launcher's own cut solver stays in charge (27B) and the seed is shown as ``in_argv: false``.

The result is HOCHRECHNUNG, not MESSUNG: stage B (``propose_oracle``) runs the launcher dry run on it and says what the
launcher makes of it.  STDLIB ONLY at module level (see ``propose_rules``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.weg2 import propose_rules as R

SCHEMA = "flliper.propose-a/1"
FORMS = ("flip", "tp")
#: the forms of the plan that are other packages (AP-E dual, AP-F single card)
OTHER_FORMS = {"dual": "AP-E (Dual)", "single": "AP-F (Einzelkarte)", "einzel": "AP-F (Einzelkarte)"}

KV_TOKENS_DEFAULT = 262144
KV_DTYPE_DEFAULT = "fp8_e4m3"


class ProposeError(ValueError):
    """The request cannot be answered by stage A (wrong form, no cards, no usable model profile)."""


# ---------------------------------------------------------------------------
# hardware -> normalised cards
# ---------------------------------------------------------------------------

def _nv(node: Any) -> Any:
    return node.get("v") if isinstance(node, Mapping) else node


def _catalog_clock(bw_gbs: Optional[float], bus_bits: Optional[int]) -> Optional[int]:
    if not bw_gbs or not bus_bits:
        return None
    return int(round(float(bw_gbs) * 1000.0 / (int(bus_bits) / 8.0 * 2.0)))


def cards_from_hardware(hardware: Any) -> List[Dict[str, Any]]:
    """Plain card dicts (``card_identity.props_of`` fields + ``tflops``/``tflops_src``/``h2d_gbs``) of a hardware profile or a
    card list.  Nothing is invented: a field the source does not state stays absent."""
    out: List[Dict[str, Any]] = []
    if isinstance(hardware, Mapping):
        if str(hardware.get("schema", "")) != "flliper.hardware/1":
            raise ProposeError("hardware is not a flliper.hardware/1 document (schema=%r)" % (hardware.get("schema"),))
        for c in hardware.get("cards") or []:
            pcie = c.get("pcie") or {}
            bf16 = (c.get("compute") or {}).get("bf16") or {}
            d = {"nvml_index": int(c["nvml_index"]), "uuid": str(c["uuid"]), "name": str(c["name"]),
                 "total_mib": int(_nv(c["vram_total_mib"])), "cc": list(c["cc"]) if c.get("cc") else None,
                 "bar1_total_mib": _nv(c.get("bar1_total_mib")), "pcie_max_gen": _nv(pcie.get("max_gen")),
                 "pcie_max_width": _nv(pcie.get("max_width"))}
            if bf16.get("v") is not None:
                d["tflops"], d["tflops_src"] = float(bf16["v"]), str(bf16.get("src", ""))
            h2d = _nv((c.get("h2d") or {}).get("gbs"))
            if h2d:
                d["h2d_gbs"] = float(h2d)
            out.append(d)
    else:
        for i, e in enumerate(hardware or []):
            if "usable_mib" in e:                                   # a kartenplan_catalog row (a datasheet card)
                pn = e.get("pcie_native") or {}
                out.append({"nvml_index": i, "uuid": "GPU-synth-%d-%s" % (i, e.get("id", i)), "name": str(e["nvml_name"]),
                            "total_mib": int(e["usable_mib"]), "cc": list(e["cc"]), "pcie_max_gen": pn.get("gen"),
                            "pcie_max_width": pn.get("lanes"), "mem_bus_width_bits": e.get("bus_bits"),
                            "mem_clock_max_mhz": _catalog_clock(e.get("mem_bw_gbs"), e.get("bus_bits"))})
            else:                                                   # an NVML replay row / a launcher Card as dict
                d = dict(e)
                if d.get("total_mib") is None and d.get("total_bytes") is not None:
                    d["total_mib"] = int(d["total_bytes"]) >> 20
                if d.get("nvml_index") is None:
                    d["nvml_index"] = d.get("index", i)
                out.append(d)
    if not out:
        raise ProposeError("hardware holds no card")
    return out


# ---------------------------------------------------------------------------
# the launch argv as a structure (copy on write: untouched tokens stay byte-equal)
# ---------------------------------------------------------------------------

class LaunchArgv:
    """The launcher argv ``t`` and the process environment ``env`` of a launch, with exact-token editing.

    Three places carry a value: a top-level flag (``--flag value`` / ``--flag=value``), a word inside a group string
    (``--extra-p``/``--extra-d``: shell text) and an assignment inside ``--env-p``/``--env-d`` (``K=V;K=V``).  An edit
    replaces ONE value in place; nothing else of the token is re-serialised, so a value the proposal leaves alone is
    byte-identical to the profile (that is what makes the reference plan diff 0)."""

    def __init__(self, argv: Sequence[str], env: Mapping[str, str]):
        self.t: List[str] = [str(x) for x in argv]
        self.env: Dict[str, str] = dict(env)

    # --- top-level flags -------------------------------------------------
    def _find(self, name: str) -> Tuple[int, str]:
        for i in range(len(self.t) - 1, -1, -1):
            if self.t[i] == name:
                return i, "sep"
            if self.t[i].startswith(name + "="):
                return i, "eq"
        return -1, ""

    def get_flag(self, name: str) -> Optional[str]:
        i, mode = self._find(name)
        if i < 0:
            return None
        if mode == "eq":
            return self.t[i].split("=", 1)[1]
        return self.t[i + 1] if i + 1 < len(self.t) else ""

    def set_flag(self, name: str, value: str) -> None:
        i, mode = self._find(name)
        if i < 0:
            self.t += [name, str(value)]
        elif mode == "eq":
            self.t[i] = "%s=%s" % (name, value)
        else:
            self.t[i + 1] = str(value)

    def del_flag(self, name: str) -> bool:
        i, mode = self._find(name)
        if i < 0:
            return False
        del self.t[i:i + (1 if mode == "eq" else 2)]
        return True

    # --- group strings ---------------------------------------------------
    def gtext(self, kind: str, g: str) -> Optional[str]:
        return self.get_flag("--%s-%s" % (kind, g))

    def set_gtext(self, kind: str, g: str, text: str) -> None:
        self.set_flag("--%s-%s" % (kind, g), text)

    # --- --env-p / --env-d assignments -----------------------------------
    def env_get(self, g: str, key: str) -> Optional[str]:
        text = self.gtext("env", g)
        for item in (text or "").split(";"):
            if item.startswith(key + "="):
                return item[len(key) + 1:]
        return None

    def env_set(self, g: str, key: str, value: str) -> None:
        text = self.gtext("env", g)
        items = (text or "").split(";") if text else []
        for k, item in enumerate(items):
            if item.startswith(key + "="):
                items[k] = "%s=%s" % (key, value)
                break
        else:
            items.append("%s=%s" % (key, value))
        self.set_gtext("env", g, ";".join(items))

    def env_del(self, g: str, key: str) -> bool:
        text = self.gtext("env", g)
        items = (text or "").split(";") if text else []
        keep = [x for x in items if not x.startswith(key + "=")]
        if len(keep) == len(items):
            return False
        self.set_gtext("env", g, ";".join(keep))
        return True

    # --- words inside --extra-p / --extra-d -------------------------------
    @staticmethod
    def _rx(flag: str) -> "re.Pattern[str]":
        return re.compile(r"(?<![\w-])" + re.escape(flag) + r"(?:\s+|=)(\S+)")

    def extra_get(self, g: str, flag: str) -> Optional[str]:
        m = self._rx(flag).search(self.gtext("extra", g) or "")
        return m.group(1) if m else None

    def extra_set(self, g: str, flag: str, value: str) -> None:
        text = self.gtext("extra", g)
        m = self._rx(flag).search(text or "")
        if m:
            text = text[:m.start(1)] + str(value) + text[m.end(1):]
        else:
            text = ((text + " ") if text else "") + "%s %s" % (flag, value)
        self.set_gtext("extra", g, text)

    def extra_del(self, g: str, flag: str) -> bool:
        text = self.gtext("extra", g)
        m = self._rx(flag).search(text or "")
        if not m:
            return False
        a = m.start()
        b = m.end()
        self.set_gtext("extra", g, (text[:a].rstrip() + " " + text[b:].lstrip()).strip())
        return True


# ---------------------------------------------------------------------------
# the slots: every place a positional per-card value lives, and how it is decided
# ---------------------------------------------------------------------------

# kind: "flag" | "extra" | "env" | "proc"; group: "-" | "p" | "d"; name; policy
SLOTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("flag", "-", "--d-foreign-context-mib", "class"),
    ("flag", "-", "--d-nontorch-mib", "class"),
    ("flag", "-", "--user-reserve-mib", "class"),
    ("flag", "-", "--d-reserve-mib", "class"),
    ("flag", "-", "--pp-cut-reserve-mib", "class"),
    ("extra", "p", "--rank-user-reserve-mib", "class"),
    ("extra", "d", "--rank-user-reserve-mib", "class"),
    ("env", "p", "SGLANG_WEG2_EXTEND_TRIM_MIB", "class"),
    ("env", "d", "SGLANG_WEG2_EXTEND_TRIM_MIB", "class"),
    ("env", "p", "SGLANG_WEG2_L15_MIB", "class"),
    ("env", "d", "SGLANG_WEG2_L15_MIB", "class"),
    ("proc", "-", "SGLANG_WEG2_EXTEND_TRIM_MIB", "class"),
    ("proc", "-", "SGLANG_WEG2_L15_MIB", "class"),
    ("flag", "-", "--pp-stage-ratio", "cut"),
    ("flag", "-", "--pp-attn-stage-ratio", "cut_attn"),
    ("extra", "p", "--pp-stage-ratio", "cut"),
    ("extra", "p", "--pp-attn-stage-ratio", "cut_attn"),
    ("flag", "-", "--pp-cut-expert-device-fraction", "fr_p"),
    ("extra", "p", "--rank-moe-resident-fraction", "fr_p"),
    ("env", "p", "SGLANG_MOE_RESIDENT_EXPERT_FRACTION", "fr_p"),
    ("flag", "-", "--pp-cut-expert-lru-rows", "lru"),
    ("extra", "d", "--rank-role", "role"),
    ("extra", "d", "--rank-tp-ratio", "tp_ratio"),
    ("extra", "d", "--rank-moe-ratio", "moe_ratio"),
    ("extra", "d", "--rank-moe-resident-fraction", "fr_d"),
    ("env", "d", "SGLANG_MOE_RESIDENT_EXPERT_FRACTION", "fr_d"),
    ("env", "p", "SGLANG_MOE_SCRATCH_SLOTS", "scratch"),
    ("env", "d", "SGLANG_MOE_SCRATCH_SLOTS", "scratch"),
    ("flag", "-", "--d-reshard-presets", "advisory"),
    ("flag", "-", "--p-barlink-bar1-window-mib", "advisory"),
    ("extra", "p", "--rank-gpu-memory-mib", "advisory"),
    ("extra", "d", "--rank-gpu-memory-mib", "advisory"),
)
#: origin text per slot policy (what the rule is, in words; the policy key itself is an internal name and never shown)
RULE_ORIGIN = {
    "cut": "Regel: Schicht-Schnitt der P-Stufen nach GEMM-Rate und Speicherkapazitaet",
    "cut_attn": "Regel: Voll-Attention-Schichten je P-Stufe aus dem Schichtschnitt",
    "fr_p": "Regel: residenter Expertenanteil je P-Stufe aus dem Rest der Stufe",
    "fr_d": "Regel Form A: residenter Expertenanteil je D-Rang",
    "moe_ratio": "Regel Form A: Experten-Eigentum je D-Rang",
    "role": "Regel Form A: Rang 0 Attention-Host, alle anderen Experten-Worker",
    "tp_ratio": "Regel Form A: nur der Host traegt die dichten Gewichte",
}
#: launcher flags whose value is a measurement (boot logs) of the profile's own inventory
INVENTORY_BOUND_FLAGS = ("--wake-credit-reference-logs", "--p-card-reference-logs", "--d-card-reference-logs",
                         "--d-residency-reference-logs")
#: the Form A tokens a MoE D layout needs when the profile names none (no basis)
_FORM_A_TOKENS = ("role", "tp_ratio", "moe_ratio", "fr_d")


def slot_label(kind: str, group: str, name: str) -> str:
    return {"flag": "%s", "extra": "--extra-" + group + " %s", "env": "--env-" + group + " %s", "proc": "env %s"}[kind] % name


def _get(la: LaunchArgv, kind: str, group: str, name: str) -> Optional[str]:
    if kind == "flag":
        return la.get_flag(name)
    if kind == "extra":
        return la.extra_get(group, name)
    if kind == "env":
        return la.env_get(group, name)
    return la.env.get(name)


def _set(la: LaunchArgv, kind: str, group: str, name: str, value: str) -> None:
    if kind == "flag":
        la.set_flag(name, value)
    elif kind == "extra":
        la.extra_set(group, name, value)
    elif kind == "env":
        la.env_set(group, name, value)
    else:
        la.env[name] = value


def _del(la: LaunchArgv, kind: str, group: str, name: str) -> None:
    if kind == "flag":
        la.del_flag(name)
    elif kind == "extra":
        la.extra_del(group, name)
    elif kind == "env":
        la.env_del(group, name)
    else:
        la.env.pop(name, None)


def vector_lengths(argv: Sequence[str], env: Optional[Mapping[str, str]] = None) -> Dict[str, int]:
    """Every positional per-card vector of a launch -> its entry count (the stdlib twin of ``launcher.positional_vector_lengths``:
    the flags of :data:`propose_rules.POSITIONAL_VECTOR_FLAGS` at top level, the tokens of :data:`POSITIONAL_VECTOR_TOKENS`
    inside the group strings, and the two process variables ``SGLANG_WEG2_L15_MIB`` / ``SGLANG_WEG2_EXTEND_TRIM_MIB``).  The
    key says WHERE: ``"<name>"`` or ``"--extra-d <name>"``; a scalar value is no vector and does not appear."""
    la = LaunchArgv(argv, env or {})
    out: Dict[str, int] = {}

    def add(key: str, value: Optional[str]) -> None:
        v = R.parse_csv(value) if value is not None else None
        if v is not None:
            out[key] = len(v)

    for dest in R.POSITIONAL_VECTOR_FLAGS:
        flag = "--" + dest.replace("_", "-")
        add(flag, la.get_flag(flag))
    for g in ("p", "d"):
        for tok in R.POSITIONAL_VECTOR_TOKENS:
            name = tok.rstrip("=")
            if tok.endswith("="):
                add("--env-%s %s" % (g, name), la.env_get(g, name))
            else:
                add("--extra-%s %s" % (g, name), la.extra_get(g, name))
    for name in ("SGLANG_WEG2_L15_MIB", "SGLANG_WEG2_EXTEND_TRIM_MIB"):
        add("env " + name, la.env.get(name))
    return out


# ---------------------------------------------------------------------------
# the proposal
# ---------------------------------------------------------------------------

def _basis_of(basis: Any) -> Dict[str, Any]:
    if basis is None:
        return {"argv": [], "env": {}, "vars": {}, "name": ""}
    if isinstance(basis, Mapping):
        d = dict(basis)
    else:
        d = {"argv": basis.argv, "env": basis.env, "vars": basis.vars, "name": getattr(basis, "source", "")}
    d.setdefault("vars", {})
    d.setdefault("name", "")
    d["name"] = str(d["name"]).replace("\\", "/").rsplit("/", 1)[-1]
    return d


def _inventory_of(vars_: Mapping[str, str]) -> Optional[Tuple[str, ...]]:
    from sglang.srt.weg2 import card_identity as ci

    return ci.parse_inventory(str(vars_.get("PROFILE_INVENTORY", "") or "") or None)


def _fit_argv(argv: Sequence[str]) -> List[str]:
    """``--extra-p X`` -> ``--extra-p=X`` (the form ``hw_fit._scoped_tokens`` reads)."""
    out: List[str] = []
    t = list(argv)
    i = 0
    while i < len(t):
        if t[i] in ("--extra-p", "--extra-d") and i + 1 < len(t):
            out.append("%s=%s" % (t[i], t[i + 1]))
            i += 2
            continue
        out.append(t[i])
        i += 1
    return out


class _Rec:
    """The per-value records of a proposal (what the dashboard shows beside each value)."""

    def __init__(self) -> None:
        self.values: List[Dict[str, Any]] = []
        self.unbelegt: List[str] = []
        self.hinweise: List[str] = []
        self.blocker: List[str] = []

    def add(self, label: str, *, group: str, old: Optional[str], new: Optional[str], state: str, herkunft: str, grund: str,
            in_argv: bool = True, policy: str = "") -> None:
        n = len(R.parse_csv(new) or []) if new else 0
        self.values.append({"key": label, "group": group, "policy": policy, "alt": old, "wert": new,
                            "eintraege": n if n else (1 if new not in (None, "") else 0), "zustand": state,
                            "herkunft": herkunft, "grund": grund, "in_argv": in_argv,
                            "geaendert": (old != new)})
        if state == R.UNBELEGT:
            self.unbelegt.append("%s: %s" % (label, herkunft))


def propose(hardware: Any, modell: Mapping[str, Any], form: str = "flip", ziele: Optional[Mapping[str, Any]] = None, *,
            basis: Any = None, draft: Optional[Mapping[str, Any]] = None, rates: Optional[Mapping[str, float]] = None,
            library: Any = None) -> Dict[str, Any]:
    """The stage A proposal (see the module docstring).  Pure: reads nothing from disk or the box; every input is an argument."""
    from sglang.srt.weg2 import hw_fit

    form = str(form or "flip").lower()
    if form in OTHER_FORMS:
        raise ProposeError("form %r is %s, not stage A of AP-C (flip | tp)" % (form, OTHER_FORMS[form]))
    if form not in FORMS:
        raise ProposeError("unknown form %r (flip | tp)" % form)
    z = dict(ziele or {})
    cards = R.order_cards(cards_from_hardware(hardware))
    n = len(cards)
    if n < 2:
        raise ProposeError("%d card: the weg2 launcher forms need N >= 2 (single card = AP-F)" % n)
    rec = _Rec()
    b = _basis_of(basis)
    la = LaunchArgv(b["argv"], b["env"])
    bname = b["name"] or "(kein Profil)"
    live_cls = tuple(c["class"] for c in cards)
    binv = _inventory_of(b["vars"])
    same_inv = binv is not None and tuple(binv) == live_cls and not z.get("force_rules")

    # --- K1: the model terms (hw_fit) -----------------------------------------------------------------------------------
    fp = R.fit_profile_from_model(modell, draft)
    is_moe = bool(fp.is_moe)
    kv_tokens = int(z.get("kv_tokens") or KV_TOKENS_DEFAULT)
    kv_dtype = str(z.get("kv_dtype") or KV_DTYPE_DEFAULT)
    seats_ref = None
    for src in (la.get_flag("--p-bs"), la.extra_get("p", "--max-running-requests"), la.extra_get("d", "--max-running-requests")):
        if src and str(src).isdigit():
            seats_ref = int(src)
            break
    # The regulator "Sitze gleichzeitig" is the DECODE bs goal, so its reference is what group D runs today: an explicit --d-bs of the
    # profile, else the D group's --max-running-requests, else the LAUNCHER's own default (``weg2.DEFAULT_D_BS``, argparse default of
    # --d-bs, also ``DEFAULT_D_BS_NEXTFLASH`` for the nextflash profile) -- never the P side's --p-bs (27B: --p-bs 1, D runs 6).
    from sglang.srt.weg2 import DEFAULT_D_BS
    d_seats_base, d_seats_src = DEFAULT_D_BS, "Launcher-Default DEFAULT_D_BS"
    for src, what in ((la.get_flag("--d-bs"), "--d-bs des Profils"), (la.extra_get("d", "--max-running-requests"), "--max-running-requests D des Profils")):
        if src and str(src).isdigit():
            d_seats_base, d_seats_src = int(src), what
            break
    seats = int(z.get("seats") or d_seats_base)
    # group P's seat-bound values (--p-bs, P --max-running-requests, --max-mamba-cache-size, P scratch) follow the goal only when the
    # profile couples them to D (same seat count, NF: 6 and 6); a profile that runs P at another count (27B: --p-bs 1) keeps its P side.
    p_coupled = seats_ref is None or seats_ref == d_seats_base
    seats_p = seats if p_coupled else seats_ref       # the seat count P's stages are priced for
    prof_name = la.get_flag("--profile") or ("nextflash" if is_moe else "qwen27b")
    asm = hw_fit.Assumptions(kv_tokens=kv_tokens, kv_dtype=kv_dtype, p_mamba_slots=R.mamba_slots_p(seats_p),
                             d_mamba_slots=R.mamba_slots_d(seats))
    fit_cards = [hw_fit.FitCard(total_mib=c["total_mib"], arch=c["arch"], cls=c["class"] if c["calibrated"] else "",
                                label="%s/%d" % (c["class"], c["total_mib"])) for c in cards]
    verdict = hw_fit.Verdict()
    sb = R.stage_budgets(fp, fit_cards, asm, _fit_argv(la.t), records_profile=prof_name, verdict=verdict)

    # --- K2: rates and the P cut seed ----------------------------------------------------------------------------------
    rate, rate_src, rate_basis = R.rate_table(cards, measured=rates, library=library)
    layers, cut_notes = R.split_layers(fp.n_layers, rate, sb["caps"])
    attn = R.attn_counts(fp.layer_families, layers)
    rec.hinweise += ["P-Schnitt: " + x for x in cut_notes]
    if rate_basis.startswith(("Datenblatt", "unbelegt")):
        rec.unbelegt.append("GEMM-Raten: " + rate_basis)
    pin_in_basis = la.get_flag("--pp-stage-ratio") is not None or la.extra_get("p", "--pp-stage-ratio") is not None
    pin_mode = str(z.get("p_cut") or "auto")
    apply_cut = (pin_mode == "pin" or (pin_mode == "auto" and pin_in_basis))
    frp, frp_notes = R.fr_p(fp, layers, sb["avail"], sb["cost"])
    rec.hinweise += ["Residenter Expertenanteil der P-Stufen: " + x for x in frp_notes]

    carry_h = "Profil %s, gleiches Inventar [%s]" % (bname, ",".join(live_cls))
    carry_g = "Wert des Profils (gemessen bzw. vom Betreiber gesetzt) unveraendert: das Profil gilt fuer genau diese Karten"
    # A value of the profile that is handed to ANOTHER inventory is never "vorgeschlagen / gleiches Inventar": the profile's
    # measurement or pin was taken on its own cards (review round 3, finding 1).  ``carried()`` binds state and origin to ``same_inv``.
    inv_txt = ",".join(binv) if binv else "unbekannt (das Profil nennt kein PROFILE_INVENTORY)"
    carry_x = "aus Profil %s fuer [%s] uebernommen, Inventar hier [%s]" % (bname, inv_txt, ",".join(live_cls))
    carry_gx = ("Wert des Profils unveraendert uebertragen; er wurde auf den Karten des Profils gemessen bzw. gesetzt und ist fuer "
                "diese Karten nicht belegt")

    def carried() -> Tuple[str, str, str]:
        """``(state, origin, reason)`` of a profile value kept as it is."""
        return (R.VORGESCHLAGEN, carry_h, carry_g) if same_inv else (R.UNBELEGT, carry_x, carry_gx)

    def rule_origin(pol: str) -> str:
        return "%s fuer die Karten [%s]" % (RULE_ORIGIN.get(pol, "Regel des Planers"), ",".join(live_cls))

    # --- first the per-class measured values (the D context is an input of the D layout) --------------------------------
    results: Dict[str, Any] = {}

    def per_class(kind: str, group: str, name: str) -> None:
        old = _get(la, kind, group, name)
        if old is None:
            return
        lab = slot_label(kind, group, name)
        vec = R.parse_csv(old)
        if vec is None:
            return
        if same_inv or (len(vec) == n and binv is None):
            st, hk, gr = carried()
            rec.add(lab, group=group, old=old, new=old, state=st, herkunft=hk, grund=gr, policy="class")
            return
        if binv is not None and len(vec) == len(binv):
            new, srcs = R.class_rekey(vec, binv, cards)
            newt = R.csv(new)
            _set(la, kind, group, name, newt)
            borrowed = [s for s in srcs if s.startswith(("geborgt", "unbelegt"))]
            rec.add(lab, group=group, old=old, new=newt, state=R.UNBELEGT if borrowed else R.VORGESCHLAGEN,
                    herkunft="Klassenmaximum der Profil-Messung %s [%s] -> [%s]: %s" % (bname, ",".join(binv), ",".join(live_cls),
                                                                                       "; ".join(srcs)),
                    grund="je Karte das Maximum der gemessenen Eintraege der EIGENEN Klasse (konservativ); eine Karte ohne "
                          "Zwilling in der Messung borgt den Wert ihrer Arch-Klasse und ist unbelegt", policy="class")
            return
        _del(la, kind, group, name)
        rec.add(lab, group=group, old=old, new=None, state=R.UNBELEGT, herkunft="nicht ableitbar: Profil-Messung %s hat %d "
                "Eintraege fuer [%s], die Karten sind [%s]" % (bname, len(vec), ",".join(binv or ()), ",".join(live_cls)),
                grund="entfernt (kein Wert erfunden); der Launcher nennt, was er statt dessen braucht", in_argv=False,
                policy="class")

    for kind, group, name, pol in SLOTS:
        if pol == "class":
            per_class(kind, group, name)

    # --- the D side ------------------------------------------------------------------------------------------------------
    foreign = R.parse_csv(la.get_flag("--d-foreign-context-mib"))
    nontorch = R.parse_csv(la.get_flag("--d-nontorch-mib"))
    reserve = R.parse_csv(la.get_flag("--user-reserve-mib")) or R.parse_csv(la.extra_get("d", "--rank-user-reserve-mib"))
    if foreign and nontorch and len(foreign) == n == len(nontorch):
        ctx = [float(foreign[i]) + float(nontorch[i]) for i in range(n)]
        ctx_src = "D-Kontext (--d-foreign-context-mib + --d-nontorch-mib) der Karten"
    else:
        ctx = [float(c["total_mib"] - sb["avail"][i]) for i, c in enumerate(cards)]       # total - (total - residue - posts)
        ctx_src = "P-Residuum + Stage-Posten (Fallback, unbelegt)"
        rec.unbelegt.append("D-Kontext je Karte: kein --d-foreign-context-mib/--d-nontorch-mib im Profil fuer dieses Inventar")
    d_budget = [int(c["total_mib"] - ctx[i]) for i, c in enumerate(cards)]
    kv_cell = fp.kv_bytes_per_token_per_attn_layer[kv_dtype]
    kv_mib = fp.attn_layers * kv_cell * kv_tokens / R.MIB
    mamba_slot = hw_fit.mamba_slot_mib(fp, sb["posts"])
    d_mamba_mib = fp.linear_layers * mamba_slot * asm.d_mamba_slots
    host_fixed = fp.resident_dense_mib()
    dp = R.draft_placement(fp, d_budget[0], host_fixed, kv_mib, d_mamba_mib, fp.draft_mib)
    results["draft"] = dp
    fa = None
    res_mib = [int(float(x)) for x in reserve] if reserve and len(reserve) == n else [0] * n
    if is_moe:
        fa = R.form_a_d(fp, cards, d_budget, res_mib, kv_tokens=kv_tokens, kv_cell_bytes=kv_cell, draft_mib=fp.draft_mib,
                        d_mamba_slots=asm.d_mamba_slots, mamba_slot_mib=mamba_slot)
        results["form_a"] = fa
        rec.unbelegt += fa["unbelegt"]
        if not fa["ok"]:
            rec.blocker.append("Form A passt nicht: " + fa["error"])
    else:
        results["dense_d_shares"] = R.dense_d_shares(d_budget)

    d_obj_old = la.get_flag("--d-tp-objective")

    # --- K4 MoE: the "Sitze gleichzeitig" regulator derives FR_P / FR_D ---------------------------------------------------------
    # The profile's FR values are valid for the seat count the profile ran (``seats_base``).  A different seat goal changes the
    # Mamba slots every P stage / D rank holds (``R.mamba_slots_p`` / ``R.mamba_slots_d``), and that VRAM is paid from the
    # resident experts (the rest rule of K4: KV obligation fixed, the rest buys experts).  The derivation is the DIFFERENCE of the
    # hw_fit arithmetic at the goal against the one at the base seats, added to the profile's value: the profile's measured /
    # operator-set level stays, only the seat-bound part moves (a rule value alone would replace a measurement by a model).
    seats_base = d_seats_base
    seats_changed = bool(z.get("seats")) and int(z["seats"]) != seats_base
    fit_argv0 = _fit_argv(la.t)
    seat_cache: Dict[int, Dict[str, Any]] = {}

    def seat_seed(s_: int) -> Dict[str, Any]:
        """The rule terms at ``s_`` seats: P stage budgets (``sb``) and the Form A D layout (``fa``)."""
        if s_ not in seat_cache:
            asm_s = hw_fit.Assumptions(kv_tokens=kv_tokens, kv_dtype=kv_dtype, p_mamba_slots=R.mamba_slots_p(s_ if p_coupled else seats_p),
                                       d_mamba_slots=R.mamba_slots_d(s_))
            sb_s = R.stage_budgets(fp, fit_cards, asm_s, fit_argv0, records_profile=prof_name, verdict=hw_fit.Verdict())
            if foreign and nontorch and len(foreign) == n == len(nontorch):
                budget_s = d_budget
            else:
                budget_s = [int(c["total_mib"] - (c["total_mib"] - sb_s["avail"][i])) for i, c in enumerate(cards)]
            fa_s = None
            if is_moe:
                fa_s = R.form_a_d(fp, cards, budget_s, res_mib, kv_tokens=kv_tokens, kv_cell_bytes=kv_cell,
                                  draft_mib=fp.draft_mib, d_mamba_slots=asm_s.d_mamba_slots,
                                  mamba_slot_mib=hw_fit.mamba_slot_mib(fp, sb_s["posts"]))
            seat_cache[s_] = {"sb": sb_s, "fa": fa_s}
        return seat_cache[s_]

    def effective_cut() -> List[int]:
        """The P layer cut in effect: the argv's (when it is a valid cut of this model over N stages), else the rule seed."""
        for text in (la.get_flag("--pp-stage-ratio"), la.extra_get("p", "--pp-stage-ratio")):
            v = R.parse_csv(text)
            if v and len(v) == n and all(x.isdigit() for x in v) and sum(int(x) for x in v) == fp.n_layers:
                return [int(x) for x in v]
        return list(layers)

    def seat_fr(pol: str, vec: Sequence[str]) -> Optional[Tuple[str, str]]:
        """``(new vector, reason)`` of an FR vector of the profile moved to the seat goal; None when it cannot be derived."""
        if len(vec) != n:
            return None
        try:
            old_v = [float(x) for x in vec]
        except ValueError:
            return None
        a, b_ = seat_seed(seats), seat_seed(seats_base)
        if pol == "fr_p":
            cut = effective_cut()
            fx, _ = R.fr_p(fp, cut, a["sb"]["avail"], a["sb"]["cost"])
            f0, _ = R.fr_p(fp, cut, b_["sb"]["avail"], b_["sb"]["cost"])
            if not fx or not f0:
                return None
            delta = [x - y for x, y in zip(fx, f0)]
            what = "P-Stufen (hw_fit: Mamba-Slots %d statt %d je Stufe)" % (R.mamba_slots_p(seats), R.mamba_slots_p(seats_base))
        else:
            if not (a["fa"] and b_["fa"] and a["fa"]["ok"] and b_["fa"]["ok"]):
                return None
            delta = [x - y for x, y in zip(a["fa"]["fr"], b_["fa"]["fr"])]
            what = "D-Raenge (Form A: Mamba-Slots %d statt %d)" % (R.mamba_slots_d(seats), R.mamba_slots_d(seats_base))
        new = [max(0.0, min(1.0, o + d)) for o, d in zip(old_v, delta)]
        return ",".join("%.3f" % x for x in new), (
            "Sitze gleichzeitig %d statt %d: die zusaetzlichen Mamba-Slots der %s werden aus den residenten Experten bezahlt "
            "(KV-Pflicht %d Token bleibt); Profilwert + Differenz der hw_fit-Rechnung" % (seats, seats_base, what, kv_tokens))

    def decide(kind: str, group: str, name: str, pol: str) -> None:
        old = _get(la, kind, group, name)
        lab = slot_label(kind, group, name)
        if old is None:
            return
        vec = R.parse_csv(old)
        if pol == "scratch":
            if vec is None and not str(old).replace(".", "").isdigit():
                return
            ref_s = seats_base if (group != "p" or p_coupled) else (seats_ref or seats_base)
            tgt_s = seats if (group != "p" or p_coupled) else ref_s
            if vec is None:
                nt = R.scale_by_seats([old], tgt_s, ref_s)[0]
            elif same_inv and tgt_s == ref_s:
                nt = old
            else:
                base = vec if len(vec) == n else R.role_rekey(vec, n)
                nt = R.csv(R.scale_by_seats(base, tgt_s, ref_s))
            if nt != old:
                _set(la, kind, group, name, nt)
            rekeyed = vec is not None and len(vec) != n
            by_seats = tgt_s != ref_s
            if nt == old:
                st, hk, gr = carried()
            else:
                # a value that was moved is a transfer: by the seat goal (an extrapolation from ONE measured point) and / or by the
                # rank role (first / middle / last entry) onto another card count -- either way not measured on these cards: unbelegt
                parts = []
                if rekeyed:
                    parts.append("auf %d Karten nach Rang-Rolle umgeschluesselt (erster/mittlerer/letzter Eintrag des Profils mit %d "
                                 "Eintraegen)" % (n, len(vec)))
                if by_seats:
                    parts.append("linear in den Sitzen skaliert (%d -> %d), Hochrechnung aus einem Messpunkt" % (ref_s, tgt_s))
                st, hk = R.UNBELEGT, "Profil %s: %s" % (bname, "; ".join(parts))
                if not same_inv:
                    hk += "; Inventar des Profils [%s], hier [%s]" % (inv_txt, ",".join(live_cls))
                gr = ("Scratch-Zeilen gehoeren zu Rang-Rolle und Sitzzahl; im Profil steht nur EIN Messpunkt, die Umrechnung ist "
                      "nicht belegt")
            rec.add(lab, group=group, old=old, new=nt, state=st, herkunft=hk, grund=gr, policy=pol)
            return
        if pol == "advisory":
            if vec is None or same_inv or len(vec) == n:
                st, hk, gr = carried()
                rec.add(lab, group=group, old=old, new=old, state=st, herkunft=hk, grund=gr, policy=pol)
            else:
                _del(la, kind, group, name)
                rec.add(lab, group=group, old=old, new=None, state=R.UNBELEGT,
                        herkunft="nicht ableitbar (%d Eintraege, %d Karten)" % (len(vec), n),
                        grund="entfernt (kein Wert erfunden)", in_argv=False, policy=pol)
            return
        if vec is None:
            return
        if pol == "lru":
            if same_inv:
                rec.add(lab, group=group, old=old, new=old, state=R.VORGESCHLAGEN, herkunft=carry_h, grund=carry_g, policy=pol)
                return
            new = R.csv(R.role_rekey(vec, n))
            _set(la, kind, group, name, new)
            rec.add(lab, group=group, old=old, new=new, state=R.UNBELEGT,
                    herkunft="Profil %s: nach Rang-Rolle (erster/mittlerer/letzter Eintrag) auf %d Karten umgeschluesselt; Inventar "
                             "des Profils [%s], hier [%s]" % (bname, n, inv_txt, ",".join(live_cls)),
                    grund="LRU-Zeilen je Stufe sind eine Pool-Groesse, keine Kartenmessung; fuer diese Karten nicht belegt", policy=pol)
            return
        # cut / cut_attn / fr_p / fr_d / moe_ratio / role / tp_ratio
        if same_inv and seats_changed and pol in ("fr_p", "fr_d") and is_moe and (pol == "fr_d" or p_coupled):
            sf = seat_fr(pol, vec)
            if sf is not None:
                _set(la, kind, group, name, sf[0])
                rec.add(lab, group=group, old=old, new=sf[0], state=R.UNBELEGT, herkunft="Profil %s + Regel fuer den Regler Sitze gleichzeitig (%d)" % (
                    bname, seats), grund=sf[1] + "; am Metall unbelegt (Planer-Rechnung)", policy=pol)
                return
            rec.hinweise.append("%s: nicht auf %d Sitze umrechenbar (Vektorlaenge/Form A); Profilwert bleibt" % (lab, seats))
        if same_inv:
            rec.add(lab, group=group, old=old, new=old, state=R.VORGESCHLAGEN, herkunft=carry_h, grund=carry_g, policy=pol)
            return
        new: Optional[str] = None
        why = ""
        stt = R.VORGESCHLAGEN
        if pol == "cut":
            if apply_cut:
                new, why = R.csv(layers), ("Schicht-Schnitt proportional zur GEMM-Rate (%s), je Stufe auf die Speicher-Kapazitaet "
                                           "begrenzt (hw_fit: %s Schichten)" % (rate_basis, R.csv(sb["caps"])))
                stt = R.UNBELEGT if rate_basis.startswith(("Datenblatt", "unbelegt")) else R.VORGESCHLAGEN
        elif pol == "cut_attn":
            if apply_cut:
                new, why = R.csv(attn), "die Voll-Attention-Schichten jeder Stufe des Schichtschnitts, exakt aus den Layer-Familien"
        elif pol == "fr_p":
            if frp:
                new, why = ",".join("%.3f" % f for f in frp), ("Rest der Stufe nach Gewichten, LRU-Zeilen, KV (%d Token) und Mamba-"
                                                              "Slots kauft residente Experten (hw_fit-Terme)" % kv_tokens)
        elif fa is not None and fa["ok"]:
            if pol == "fr_d":
                new, why = ",".join("%.3f" % f for f in fa["fr"]), ("Form-A-Loesung: residenter Anteil der eigenen Experten je "
                                                                   "Rang" + R.fr_d_cap_note(fa))
            elif pol == "moe_ratio":
                new, why = R.csv(fa["moe_ratio"]), "Form-A-Loesung: Experten-Eigentum je Rang (Kapazitaet + Spill nach PCIe-Rate)"
            elif pol == "role":
                new, why = ",".join(fa["role"]), "Form A: Rang 0 = Attention-Host, alle anderen Experten-Worker"
            elif pol == "tp_ratio":
                new, why = R.csv(fa["tp_ratio"]), "Form A: nur der Host traegt die dichten Gewichte"
        if new is None:
            if pol in ("cut", "cut_attn") and not apply_cut:
                return
            _del(la, kind, group, name)
            rec.add(lab, group=group, old=old, new=None, state=R.UNBELEGT,
                    herkunft="nicht ableitbar fuer die Karten [%s]%s" % (",".join(live_cls), " (Form A passt nicht)" if pol in (
                        "fr_d", "moe_ratio", "role", "tp_ratio") and fa is not None and not fa["ok"] else ""),
                    grund="entfernt (kein Wert erfunden)", in_argv=False, policy=pol)
            return
        _set(la, kind, group, name, new)
        if pol in ("fr_p", "fr_d", "moe_ratio", "role", "tp_ratio"):
            stt = R.UNBELEGT if pol != "role" and pol != "tp_ratio" else R.VORGESCHLAGEN
        rec.add(lab, group=group, old=old, new=new, state=stt, herkunft=rule_origin(pol),
                grund=why + ("; am Metall unbelegt (Planer-Rechnung)" if stt == R.UNBELEGT else ""), policy=pol)

    for kind, group, name, pol in SLOTS:
        if pol != "class":
            decide(kind, group, name, pol)

    # measured REFERENCE logs of a boot on the profile's inventory: they describe THAT inventory's stages and ranks; the launcher
    # refuses a foreign stage count (W167 Weg2PCutRecutRefused, not forceable) rather than reading them for another one
    for rflag in INVENTORY_BOUND_FLAGS:
        old = la.get_flag(rflag)
        if old is not None and not same_inv:
            la.del_flag(rflag)
            rec.add(rflag, group="-", old=old, new=None, state=R.UNBELEGT,
                    herkunft="Messung des Profils %s auf [%s]" % (bname, ",".join(binv or ())),
                    grund="entfernt: die Referenz-Logs gelten nur fuer diese Karten und Stufenzahl; fuer [%s] muss ein Lauf "
                          "DIESER Form sie messen" % ",".join(live_cls), in_argv=False, policy="reference")

    # the P cut seed, shown when it is not applied (the launcher plans group P in BOTH forms: ``--d-only`` keeps "P's resting
    # residue planned, exactly the D form of the flip boot", ``launcher.py:27160``)
    rec.add("--pp-stage-ratio (Seed)", group="-", old=None, new=R.csv(layers), state=R.UNBELEGT if rate_basis.startswith((
        "Datenblatt", "unbelegt")) else R.VORGESCHLAGEN, herkunft="Regel: P-Schnitt nach GEMM-Rate; Rate: " + rate_basis,
            grund="Schicht-Schnitt proportional zur GEMM-Rate, je Stufe auf die Speicher-Kapazitaet begrenzt; Attention-"
                  "Schichten je Stufe %s" % R.csv(attn) + ("" if apply_cut else "; NICHT gesetzt: der Launcher loest den "
                                                            "Schnitt selbst (p_cut=pin setzt ihn)"),
            in_argv=apply_cut, policy="cut_seed")

    # a MoE flip needs the resident fraction and the LRU rows of every P stage (the launcher refuses without them, W40: "Pass
    # the resident fraction per stage ... and --pp-cut-expert-lru-rows"): added when the profile names none
    if is_moe and frp and la.get_flag("--pp-cut-expert-device-fraction") is None:
        for flag, val, why in (("--pp-cut-expert-device-fraction", ",".join("%.3f" % f for f in frp),
                                "Rest der Stufe nach Gewichten, LRU-Zeilen, KV und Mamba kauft residente Experten"),
                               ("--pp-cut-expert-lru-rows", R.csv([asm.lru_rows] * n),
                                "LRU-Zeilen je Stufe: der hw_fit-Standard (%d), eine Pool-Groesse" % asm.lru_rows)):
            if la.get_flag(flag) is None:
                la.set_flag(flag, val)
                rec.add(flag, group="p", old=None, new=val, state=R.UNBELEGT if "fraction" in flag else R.VORGESCHLAGEN,
                        herkunft="Regel: Rest der P-Stufe kauft residente Experten (MoE)", grund=why + ("; am Metall unbelegt (Planer-Rechnung)" if "fraction" in flag else ""),
                        policy="fr_p" if "fraction" in flag else "lru")

    # --- Form A skeleton when no profile names it (no basis) ---------------------------------------------------------------
    if is_moe and fa is not None and fa["ok"] and not b["argv"] and form in FORMS:
        for pol, flag, val, why in (("role", "--rank-role", ",".join(fa["role"]), "Form A: Rang 0 = Attention-Host"),
                                    ("tp_ratio", "--rank-tp-ratio", R.csv(fa["tp_ratio"]), "Form A: nur der Host traegt dichte Gewichte"),
                                    ("moe_ratio", "--rank-moe-ratio", R.csv(fa["moe_ratio"]), "Form-A-Loesung: Experten-Eigentum"),
                                    ("fr_d", "--rank-moe-resident-fraction", ",".join("%.3f" % f for f in fa["fr"]),
                                     "Form-A-Loesung: residenter Anteil" + R.fr_d_cap_note(fa))):
            la.extra_set("d", flag, val)
            rec.add("--extra-d " + flag, group="d", old=None, new=val, state=R.UNBELEGT if pol in ("moe_ratio", "fr_d") else R.VORGESCHLAGEN,
                    herkunft="Regel Form A: Aufteilung der D-Raenge (MoE)", grund=why, policy=pol)

    # --- K3: the draft ------------------------------------------------------------------------------------------------------
    plc_old = la.extra_get("d", "--speculative-draft-placement") or la.get_flag("--speculative-draft-placement")
    if is_moe and fp.draft_mib > 0:
        want = dp["placement"]
        if plc_old is None and not b["argv"]:
            la.extra_set("d", "--speculative-draft-placement", want)
            rec.add("--extra-d --speculative-draft-placement", group="d", old=None, new=want, state=R.VORGESCHLAGEN,
                    herkunft="Regel: Draft solo auf Rang 0, wenn er mit den dichten Gewichten und der KV-Pflicht passt", grund=dp["why"], policy="draft")
        elif plc_old is not None:
            rec.add("--extra-d --speculative-draft-placement", group="d", old=plc_old, new=plc_old, state=R.VORGESCHLAGEN,
                    herkunft="Profil %s (unveraendert)" % bname, grund="die Draft-Regel sagt fuer diese Karten %s: %s" % (want, dp["why"]), policy="draft")
            if want != plc_old and plc_old == "solo":
                rec.hinweise.append("Draft: das Profil setzt solo, aber fuer diese Karten passt es nicht (%s). Form A verweigert die Aufteilung auf alle Raenge." % dp["why"])
        if dp["placement"] == "split":
            rec.hinweise.append("Draft solo passt nicht auf Rang 0 (%s). Ohne ihn auf Rang 0 bliebe nur die Aufteilung auf alle Raenge, die Form A verweigert." % dp["why"])
            rec.blocker.append("Draft solo auf Rang 0 passt nicht (Form A verlangt solo)")
    else:
        rec.hinweise.append("Draft: %s (die Draft-Regel gilt fuer die Form-A-MoE-Linie; bei einem dichten Modell laeuft der Draft nach Profil)" % dp["why"])
    d_kv_old = la.get_flag("--draft-kv-on-p")
    if z.get("draft_kv_on_p") in ("on", "off"):
        la.set_flag("--draft-kv-on-p", z["draft_kv_on_p"])
        rec.add("--draft-kv-on-p", group="-", old=d_kv_old, new=z["draft_kv_on_p"], state=R.VORGESCHLAGEN, herkunft="Ziel draft_kv_on_p",
                grund="Draft-KV auf P (Nutzer-Order 07.09.: Draft-KV ueber den Flip)", policy="draft")
    elif d_kv_old is not None:
        rec.add("--draft-kv-on-p", group="-", old=d_kv_old, new=d_kv_old, state=R.VORGESCHLAGEN, herkunft="Profil %s (unveraendert)" % bname,
                grund="vom Profil gesetzt; ohne Angabe gilt der Launcher-Default on", policy="draft")

    # --- the scalar knobs: D objective, seats, KV obligation ----------------------------------------------------------------
    obj = z.get("d_objective")
    if obj:
        la.set_flag("--d-tp-objective", str(obj))
        rec.add("--d-tp-objective", group="-", old=d_obj_old, new=str(obj), state=R.VORGESCHLAGEN, herkunft="Ziel d_objective",
                grund="vom Anwender gewaehlt", policy="knob")
    elif d_obj_old is not None:
        rec.add("--d-tp-objective", group="-", old=d_obj_old, new=d_obj_old, state=R.VORGESCHLAGEN, herkunft="Profil %s (unveraendert)" % bname,
                grund="der Zielwert des Profils bleibt (ein Skalar der Form, keine Kartenmessung)", policy="knob")
    else:
        la.set_flag("--d-tp-objective", "maxkv")
        rec.add("--d-tp-objective", group="-", old=None, new="maxkv", state=R.VORGESCHLAGEN, herkunft="Regel: Standardziel der Form (kein Wert im Profil)",
                grund=("Dense: TP-symmetrisch, VRAM-proportional" if not is_moe else "MoE: Rest an KV/Experten nach VRAM")
                + " (maxkv = die groesste KV-Platzzahl; der Launcher loest die Raenge selbst)", policy="knob")
    if form == "tp":
        if "--d-only" not in la.t:
            la.t.append("--d-only")
        rec.add("--d-only", group="-", old=None, new="", state=R.VORGESCHLAGEN, herkunft="Form nur-TP",
                grund="nur Gruppe D, TP ueber alle %d Karten, kein Flip" % n, policy="form")
    if seats_changed and seats_ref is not None and not p_coupled:
        rec.hinweise.append("Sitze gleichzeitig %d: das Profil faehrt P mit %d Sitzen (--p-bs / --max-running-requests) und D mit %d (%s); "
                            "das Ziel ist das Decode-bs-Ziel, die P-Seite des Profils bleibt" % (seats, seats_ref, d_seats_base, d_seats_src))
    dbs_added = False
    if seats_changed:
        # the regulator reaches D: an absent --d-bs leaves the launcher default (``apply_profile_d_bs_default``: 6 seats for
        # the nextflash profile), so a goal other than the default must be SAID.  An explicit --d-bs is the hard bound (bs_source).
        if la.get_flag("--d-bs") is None:
            dbs_added = True
            la.set_flag("--d-bs", str(seats))
            rec.add("--d-bs", group="-", old=None, new=str(seats), state=R.VORGESCHLAGEN, herkunft="Ziel Sitze gleichzeitig = %d" % seats,
                    grund="Decode-bs-Ziel des Anwenders; ohne die Angabe bliebe D bei %d Sitzen (%s)" % (seats_base, d_seats_src),
                    policy="seats")
        # the decode CUDA graph ladder follows the seats when it is the contiguous 1..k ladder of the profile
        dtext = la.gtext("extra", "d") or ""
        m = re.search(r"(?<![\w-])--cuda-graph-bs-decode((?:\s+\d+)+)", dtext)
        if m:
            ladder = [int(x) for x in m.group(1).split()]
            if ladder == list(range(1, len(ladder) + 1)):
                if ladder[-1] != seats:
                    newl = " ".join(str(i) for i in range(1, seats + 1))
                    la.set_gtext("extra", "d", dtext[:m.start(1)] + " " + newl + dtext[m.end(1):])
                    rec.add("--extra-d --cuda-graph-bs-decode", group="d", old=" ".join(map(str, ladder)), new=newl,
                            state=R.UNBELEGT, herkunft="Ziel Sitze gleichzeitig = %d" % seats,
                            grund="die Decode-Graph-Leiter 1..%d des Profils folgt dem bs-Ziel; Graph-Speicher je bs am Metall "
                                  "unbelegt" % ladder[-1], policy="seats")
            else:
                rec.hinweise.append("--cuda-graph-bs-decode ist keine 1..k-Leiter (%s): bleibt, bs ueber der Leiter laeuft ohne Graph"
                                    % " ".join(map(str, ladder)))
    if seats_changed:
        for kind, group, name, setter in (("flag", "-", "--p-bs", lambda v: la.set_flag("--p-bs", v)),
                                          ("flag", "-", "--d-bs", lambda v: la.set_flag("--d-bs", v)),
                                          ("extra", "p", "--max-running-requests", lambda v: la.extra_set("p", "--max-running-requests", v)),
                                          ("extra", "d", "--max-running-requests", lambda v: la.extra_set("d", "--max-running-requests", v)),
                                          ("extra", "p", "--max-mamba-cache-size", lambda v: la.extra_set("p", "--max-mamba-cache-size", v))):
            old = _get(la, kind, group, name)
            if old is None or (name == "--d-bs" and dbs_added):
                continue
            if group == "p" or name == "--p-bs":
                if not p_coupled:
                    continue            # the profile runs P at another seat count than D: the decode goal does not move P
            new = str(R.mamba_slots_p(seats)) if name == "--max-mamba-cache-size" else str(seats)
            setter(new)
            rec.add(slot_label(kind, group, name), group=group, old=old, new=new, state=R.VORGESCHLAGEN, herkunft="Ziel Sitze gleichzeitig = %d" % seats,
                    grund=("Mamba-Slots = %d je Sitz + %d Retention (Launcher-Zeile PP-CUT budget posts)" % (R.P_MAMBA_SLOTS_PER_SEAT,
                           R.P_MAMBA_RETENTION)) if name == "--max-mamba-cache-size" else "Decode-bs-Ziel des Anwenders", policy="seats")
    if z.get("kv_tokens") and int(z["kv_tokens"]) != KV_TOKENS_DEFAULT or la.get_flag("--max-kv-per-request") is None:
        old = la.get_flag("--max-kv-per-request")
        la.set_flag("--max-kv-per-request", str(kv_tokens))
        rec.add("--max-kv-per-request", group="-", old=old, new=str(kv_tokens), state=R.VORGESCHLAGEN,
                herkunft="Ziel kv_tokens" if z.get("kv_tokens") else "Regel: KV-Pflicht 262144 Token (Nutzer 05.10.)",
                grund="ein Request traegt diesen Kontext (P muss 262144 Token Prefill-Kontext tragen)", policy="knob")

    # --- K4 regulators that are scalars of the profile: the pool floor and the X ceiling ---------------------------------------
    # ``--pp-solve-pool-floor`` is a hard lower bound, in WORLD KV tokens, on the priced pool the P cut solve ranks over
    # (launcher help of the flag; plan section 4c: Dual asks the KV obligation 262144 as the floor).  When the goal ``kv_tokens`` moves
    # and the profile names a positive floor, the floor follows the obligation.  A profile without the flag keeps the launcher's own
    # default (the floor of its ordered 3-stage cut); the planner does not type a number there.
    pf_old = la.get_flag("--pp-solve-pool-floor")
    pf_expl = ("untere Grenze der bepreisten KV-Pool-Groesse (in Welt-KV-Token), ueber der der Schnitt-Loeser ranked; 0 = aus; "
               "ohne Flag gilt der Launcher-Standard (der Pool des geordneten 3-Stufen-Schnitts, bei anderer Stufenzahl keine Grenze)")
    if pf_old is not None:
        if z.get("kv_tokens") and str(pf_old).lstrip("-").isdigit() and int(pf_old) > 0 and int(pf_old) != kv_tokens:
            la.set_flag("--pp-solve-pool-floor", str(kv_tokens))
            rec.add("--pp-solve-pool-floor", group="-", old=pf_old, new=str(kv_tokens), state=R.VORGESCHLAGEN,
                    herkunft="Ziel kv_tokens: der Pool-Floor folgt der KV-Pflicht (Profil %s hatte %s)" % (bname, pf_old),
                    grund=pf_expl + "; ob dieser Schnitt auf der Frontier liegt, entscheidet der Loeser beim Boot (W67 sonst)", policy="knob")
        else:
            rec.add("--pp-solve-pool-floor", group="-", old=pf_old, new=pf_old, state=R.VORGESCHLAGEN,
                    herkunft="Profil %s (unveraendert)" % bname, grund=pf_expl, policy="knob")
    else:
        rec.add("--pp-solve-pool-floor", group="-", old=None, new=None, state=R.VORGESCHLAGEN,
                herkunft="nicht im Profil %s: Launcher-Standard" % bname, grund=pf_expl, in_argv=False, policy="knob")
    xc_old = la.get_flag("--x-ceiling-tokens")
    xc_expl = ("Obergrenze, bis zu der die Front das X (Token, ab denen ein Request zu D statt P geht) zur Laufzeit anheben darf, und "
               "zugleich der Riegel --tp-prefill-max-tokens von D; 0 = aus (D und Front behalten das Start-X)")
    if xc_old is not None:
        rec.add("--x-ceiling-tokens", group="-", old=xc_old, new=xc_old, state=R.VORGESCHLAGEN,
                herkunft="vom Profil %s (unveraendert; ein Skalar der Form, keine Kartenmessung)" % bname, grund=xc_expl, policy="knob")
    else:
        rec.add("--x-ceiling-tokens", group="-", old=None, new=None, state=R.VORGESCHLAGEN,
                herkunft="nicht im Profil %s: Launcher-Standard 0 (aus)" % bname, grund=xc_expl, in_argv=False, policy="knob")

    # --- the checks and the result -------------------------------------------------------------------------------------------
    lens = vector_lengths(la.t, la.env)
    bad = {k: v for k, v in lens.items() if v != n}
    fv = hw_fit.evaluate(fp, fit_cards, argv=_fit_argv(la.t), asm=asm, records_profile=prof_name)
    return {
        "schema": SCHEMA, "form": form, "n": n,
        "cards": [{"ordinal": c["ordinal"], "nvml_index": c.get("nvml_index"), "uuid": c.get("uuid"), "name": c.get("name"),
                   "class": c["class"], "arch": c["arch"], "total_mib": c["total_mib"], "tflops": rate[i], "tflops_src": rate_src[i]}
                  for i, c in enumerate(cards)],
        "inventory": {"live": list(live_cls), "profil": list(binv) if binv else None, "gleich_wie_profil": bool(same_inv)},
        "argv": list(la.t), "env": dict(la.env), "basis": b["name"],
        "werte": rec.values,
        "seeds": {"p_cut": {"layers": layers, "attn": attn, "basis": rate_basis, "gesetzt": bool(apply_cut), "caps": sb["caps"]},
                  "fr_p": frp, "draft": dp, "form_a": fa, "dense_d_shares": results.get("dense_d_shares")},
        "fit": {"level": fv.level, "first": fv.first, "margin_mib": fv.margin_mib, "lines": list(fv.lines),
                "marks": sorted(set(fv.marks + verdict.marks))},
        "ziele": {"seats": seats, "kv_tokens": kv_tokens, "kv_dtype": kv_dtype, "p_cut": pin_mode},
        "unbelegt": rec.unbelegt, "hinweise": rec.hinweise, "blocker": rec.blocker,
        "vektorlaengen": lens, "vektoren_ok": not bad, "vektoren_falsch": bad,
    }
