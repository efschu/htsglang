"""PROFIL-EDITOR S3 (Auftrag 960): Modellprofil am Desk schätzen -- POST Modellpfad, Antwort ``flliper.model/1`` als JSON.

Der Schätzer ``flliper/srt/pdflip/model_profile.py`` ist reine Standardbibliothek ("PURE: stdlib only") und wird, wie das Gate des
Kartenplaners (``kartenplan_gate``), per Dateipfad aus dem ausgelieferten Planer-Baum geladen -- nicht über ``import flliper``, das
torch zieht.  Gelesen werden nur ``config.json`` und die Kopfzeilen der Safetensors-/GGUF-Dateien; kein Gewicht, keine GPU, kein
Launcher.  Der Pfad muss unter einer der Modellwurzeln liegen (``--model-root`` bzw. ``RIGDASH_MODEL_ROOTS``, Standard der
Modell-Cache des Rigs); Symlinks aus der Wurzel hinaus werden abgewiesen.

Die Oberfläche baut Auftrag 930; hier liegen nur die Route und das JSON (``static/modellprofil.js`` ist das kleine Modul dazu).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

from . import kartenplan

DEFAULT_ROOTS = ("/spinning/llm_stuff/club-3090/models-cache",)
#: Planer-Bäume, aus denen der Schätzer gelesen wird (``MODELLPROFIL_TREE`` bzw. ``KARTENPLAN_TREE`` = <baum>/python überschreibt)
TREE_CANDIDATES = tuple(kartenplan.TREE_CANDIDATES) + (
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "python"),
)
MODULE_REL = os.path.join("flliper", "srt", "pdflip", "model_profile.py")
MAX_PATH = 1024
KV_DTYPES = (None, "auto", "fp8_e4m3")
SSM_DTYPES = (None, "float32", "bfloat16")
_ID_FIRST = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
_ID_REST = _ID_FIRST | frozenset("._-")


class _IdCheck:
    """Registry identifier ``[a-z0-9][a-z0-9._-]{0,63}`` (no regex: the rigdash log-parser freeze counts every regex literal)."""

    @staticmethod
    def match(text):
        t = str(text or "")
        return bool(0 < len(t) <= 64 and t[0] in _ID_FIRST and all(c in _ID_REST for c in t))


ID_RE = _IdCheck


def _slug(text):
    out, dash = [], False
    for c in str(text):
        if c in _ID_REST:
            out.append(c)
            dash = False
        elif not dash:
            out.append("-")
            dash = True
    return "".join(out)
CACHE_SIZE = 8


class ModellprofilUnavailable(RuntimeError):
    """Der Planer-Baum liefert ``model_profile.py`` nicht (zu alte Linie): benannt, nie geraten."""


def find_tree(explicit: Optional[str] = None) -> Optional[str]:
    for t in (explicit, os.environ.get("MODELLPROFIL_TREE"), os.environ.get("KARTENPLAN_TREE"), *TREE_CANDIDATES):
        if t and os.path.isfile(os.path.join(t, MODULE_REL)):
            return t
    return None


def _load(path: str):
    spec = importlib.util.spec_from_file_location("rigdash_model_profile", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rigdash_model_profile"] = mod       # @dataclass-frei, aber der Eintrag macht Fehlerzeilen lesbar
    spec.loader.exec_module(mod)
    return mod


def roots_from_env(explicit: Optional[List[str]] = None) -> Tuple[str, ...]:
    if explicit:
        return tuple(explicit)
    env = os.environ.get("RIGDASH_MODEL_ROOTS")
    if env:
        return tuple(p for p in env.split(os.pathsep) if p)
    return DEFAULT_ROOTS


class ModelEstimator:
    def __init__(self, *, tree: Optional[str] = None, roots: Optional[List[str]] = None):
        self.tree = find_tree(tree)
        self.roots = tuple(os.path.realpath(r) for r in roots_from_env(roots))
        self._mod = None
        self._lock = threading.Lock()
        self._cache: Dict[tuple, dict] = {}
        self._order: List[tuple] = []

    # ------------------------------------------------------------------ Baum und Modul
    def module(self):
        with self._lock:
            if self._mod is None:
                if not self.tree:
                    raise ModellprofilUnavailable(
                        "no planner tree with flliper/srt/pdflip/model_profile.py found (MODELLPROFIL_TREE or KARTENPLAN_TREE)")
                self._mod = _load(os.path.join(self.tree, MODULE_REL))
            return self._mod

    # ------------------------------------------------------------------ Pfade
    def check_path(self, raw, what: str = "path", must_exist: bool = True) -> str:
        """Absoluter, wirklicher Pfad unter einer Wurzel -- sonst ``ValueError`` mit dem Grund.  ``must_exist=False`` lässt einen
        fehlenden Pfad zu (:meth:`status` meldet ihn dann als Zustand ``not_mounted``)."""
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("%s is missing (model directory or .gguf file)" % what)
        if len(raw) > MAX_PATH or "\x00" in raw:
            raise ValueError("%s is not allowed" % what)
        if not os.path.isabs(raw):
            raise ValueError("%s must be an absolute path" % what)
        real = os.path.realpath(raw)
        if not any(real == r or real.startswith(r + os.sep) for r in self.roots):
            raise ValueError("%s is not under a model root (%s)" % (what, ", ".join(self.roots)))
        if must_exist and not os.path.exists(real):
            raise ValueError("%s does not exist" % what)
        return real

    def models(self) -> dict:
        """Die Verzeichnisse unter den Wurzeln, die wie ein Modell aussehen (nur ``stat``, nichts wird gelesen)."""
        out = []
        for root in self.roots:
            try:
                names = sorted(os.listdir(root))
            except OSError:
                continue
            for n in names:
                p = os.path.join(root, n)
                if n.startswith(".") or not os.path.isdir(p):
                    continue
                try:
                    files = os.listdir(p)
                except OSError:
                    continue
                shards = [f for f in files if f.endswith((".safetensors", ".gguf"))]
                if "config.json" not in files:
                    continue
                out.append({"name": n, "path": p, "shards": len(shards), "gguf": [f for f in files if f.endswith(".gguf")],
                            "index": "model.safetensors.index.json" in files})
        return {"ok": True, "roots": list(self.roots), "models": out, "planner_tree": self.tree}

    # ------------------------------------------------------------------ Zustand (AP-B 06.10.2026)
    def status(self, req: dict) -> dict:
        """Der Zustand eines Modellpfads als Data (``model_profile.probe``): ``not_mounted`` | ``empty`` | ``no_model_files`` |
        ``no_config`` | ``config_only`` | ``index_only`` | ``complete`` | ``gguf_incomplete`` | ``ambiguous`` | ``unreadable`` mit ``estimable``
        und Grund.  Nur Dateinamen werden gelesen.  Der Pfad muss unter einer Modellwurzel liegen, darf aber FEHLEN -- das ist der
        Fall "nicht gemountet", den der Planer als ``unverified`` liest, nicht als Fehler."""
        if not isinstance(req, dict):
            raise ValueError("Request must be a JSON object: {path, gguf_file?}")
        path = self.check_path(req.get("path"), "path", must_exist=False)
        gguf = req.get("gguf_file") or None
        if gguf is not None and (not isinstance(gguf, str) or os.sep in gguf or gguf.startswith(".") or not gguf.endswith(".gguf")):
            raise ValueError("gguf_file is a file name in the model directory")
        mp = self.module()
        if not hasattr(mp, "probe"):
            raise ModellprofilUnavailable("model_profile.py of this planner tree does not know probe() (line too old)")
        return dict(mp.probe(path, gguf_file=gguf), ok=True, planner_module=os.path.join(self.tree, MODULE_REL))

    # ------------------------------------------------------------------ Schätzung
    def _fingerprint(self, path: str) -> tuple:
        items = []
        try:
            names = [path] if os.path.isfile(path) else [os.path.join(path, f) for f in sorted(os.listdir(path))
                                                       if f.endswith((".safetensors", ".gguf", ".json")) and f != "generation_config.json"]
            for f in names:
                st = os.stat(f)
                items.append((os.path.basename(f), st.st_size, st.st_mtime_ns))
        except OSError:
            pass
        return tuple(items)

    def estimate(self, req: dict) -> dict:
        if not isinstance(req, dict):
            raise ValueError("Request must be a JSON object: {path, draft_path?, kv_dtype?, mamba_ssm_dtype?, gguf_file?, registry?}")
        path = self.check_path(req.get("path"), "path")
        draft = self.check_path(req["draft_path"], "draft_path") if req.get("draft_path") else None
        kv = req.get("kv_dtype") or None
        ssm = req.get("mamba_ssm_dtype") or None
        if kv not in KV_DTYPES:
            raise ValueError("kv_dtype must be auto or fp8_e4m3")
        if ssm not in SSM_DTYPES:
            raise ValueError("mamba_ssm_dtype must be float32 or bfloat16")
        gguf = req.get("gguf_file") or None
        if gguf is not None and (not isinstance(gguf, str) or os.sep in gguf or gguf.startswith(".") or not gguf.endswith(".gguf")):
            raise ValueError("gguf_file is a file name in the model directory")
        reg = req.get("registry")
        reg_id = None
        if reg:
            reg_id = reg if isinstance(reg, str) else os.path.basename(path.rstrip(os.sep)).lower()
            if reg is True or not isinstance(reg, str):
                reg_id = _slug(reg_id).strip("-")[:64] or "modell"
            if not ID_RE.match(reg_id):
                raise ValueError("registry: identifier must be [a-z0-9._-]{1,64}")
        mp = self.module()
        key = (path, draft, kv, ssm, gguf, reg_id, self._fingerprint(path), self._fingerprint(draft) if draft else None)
        hit = self._cache.get(key)
        if hit is not None:
            return dict(hit, cached=True)
        t0 = time.time()
        try:
            prof = mp.estimate(path, draft_path=draft, kv_dtype=kv, mamba_ssm_dtype=ssm, gguf_file=gguf)
        except mp.ModelProfileError as exc:
            raise ValueError(str(exc))
        out = {"ok": True, "profile": prof, "elapsed_s": round(time.time() - t0, 3), "cached": False,
               "planner_module": os.path.join(self.tree, MODULE_REL), "schema": mp.SCHEMA}
        if reg_id:
            out["registry_fields"] = mp.derive_registry_fields(prof, row_id=reg_id)
        with self._lock:
            self._cache[key] = out
            self._order.append(key)
            while len(self._order) > CACHE_SIZE:
                self._cache.pop(self._order.pop(0), None)
        return out
