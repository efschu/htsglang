"""AP-D (Plan Profil-Planer 06.10., Stufe B/C): der Dienst hinter dem Orakel -- Kindprozess + Cache.

Der Dashboard-Prozess rechnet nichts selbst (kein sglang-Import, MemoryMax=1G).  ``OracleService`` haelt EINEN langlebigen Kindprozess
(``kartenplan_build/oracle_worker.py``) mit dem Python der sglang-Umgebung, der den Launcher-Trockenlauf (``weg2/propose_oracle``) und
``propose()`` fuer das Dashboard fragt.  Er benutzt Start, Neustart nach Tod/Zeitueberschreitung und die Fehlerform des
``CouplingsService`` (``{"ok": False, "error": ...}``, nie ein Absturz des Dashboards), ist aber ein EIGENER Prozess: ein Trockenlauf
dauert Sekunden (gemessen 06.10.: 16-18 s, wenn der Launcher bis zum Ende plant; 0,3-3 s bei fruehem Abbruch), importiert den Launcher und
fasst Prozesszustand an -- das darf die Balken des Editors nicht blockieren.

CACHE je (Inventar, Form, Argv-Hash): der Schluessel ist der sha256 ueber die Teile der Anfrage (Inventar, Form, Ziele, Hash des Profils bzw. des
gerenderten Profiltextes, Modellpfade) UND ueber den Stand der Quellen des Orakels (Groesse + mtime von launcher.py, refusals.py, hw_fit.py,
topology.py, propose*.py des Planer-Baums): aendert sich der Launcher oder driftet das Live-Profil (Plan 4c), ist es ein anderer Schluessel.
Nur ``ok``-Antworten werden gemerkt; eine Anfrage, die schon laeuft, haelt die naechste gleiche an (sie trifft dann den Cache).

Dieses Modul kennt weder HTTP noch Dateien des Hosts ausser den Quellen des Planer-Baums fuer den Cache-Schluessel."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional

from .profil_recompute import CouplingsService

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(os.path.dirname(HERE), "kartenplan_build", "oracle_worker.py")
#: ein Trockenlauf bis zum Ende dauerte 16-18 s; zwei Laeufe (ohne Force, dann mit Force) und ein langsamer Host brauchen Luft
TIMEOUT_S = 300.0
START_TIMEOUT_S = 180.0
CACHE_SIZE = 48
#: die Quellen, von denen ein Verdikt abhaengt (relativ zu ``<Baum>/sglang/srt/weg2/``)
SOURCES = ("launcher.py", "refusals.py", "hw_fit.py", "topology.py", "propose.py", "propose_rules.py", "propose_oracle.py",
           "propose_verdict.py", "card_identity.py", "model_profile.py", "profile_json.py")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()


def sha256_file(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


class OracleService(CouplingsService):
    """Orakel-Worker + Cache (siehe Modulkopf).  ``ask(kind, req, parts)``: ``kind`` ``verdikt`` | ``propose``."""

    def __init__(self, tree_python: Optional[str], python: Optional[str] = None, worker: str = WORKER, timeout_s: float = TIMEOUT_S,
                 start_timeout_s: float = START_TIMEOUT_S, cache_size: int = CACHE_SIZE, request_extra: Optional[Mapping[str, Any]] = None):
        super().__init__(tree_python, python=python, worker=worker, timeout_s=timeout_s, start_timeout_s=start_timeout_s)
        #: Felder, die jede Anfrage zusaetzlich traegt (z. B. ``snapshots``: Kopf-Snapshots fuer leere Modell-Mountpunkte einer Entwicklungs-Box);
        #: sie gehoeren zum Cache-Schluessel
        self.request_extra: Dict[str, Any] = dict(request_extra or {})
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._flight = threading.Lock()
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ Schluessel
    def stamp(self) -> List[List[Any]]:
        """Groesse und mtime der Quellen des Orakels im Planer-Baum (``None`` = Datei fehlt): ein anderer Launcher ist ein anderer Schluessel."""
        out: List[List[Any]] = []
        base = os.path.join(self.tree_python or "", "sglang", "srt", "weg2")
        for name in SOURCES:
            try:
                st = os.stat(os.path.join(base, name))
                out.append([name, st.st_size, st.st_mtime_ns])
            except OSError:
                out.append([name, None, None])
        return out

    def key(self, kind: str, parts: Mapping[str, Any]) -> str:
        return sha256_text(canonical({"kind": kind, "parts": parts, "extra": self.request_extra, "quellen": self.stamp()}))

    # ------------------------------------------------------------------ Cache
    def _get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                self.hits += 1
                return json.loads(json.dumps(hit))
        return None

    def _put(self, key: str, res: Dict[str, Any]) -> None:
        with self._cache_lock:
            self._cache[key] = json.loads(json.dumps(res, default=str))
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def cache_info(self) -> Dict[str, int]:
        with self._cache_lock:
            return {"eintraege": len(self._cache), "treffer": self.hits, "fehlgriffe": self.misses, "groesse": self.cache_size}

    # ------------------------------------------------------------------ Anfrage
    def ask(self, kind: str, req: Mapping[str, Any], parts: Mapping[str, Any]) -> Dict[str, Any]:
        """Antwort des Workers (aus dem Cache, wenn Schluessel und Quellen gleich sind); ``cached`` und ``cache_key`` (16 Zeichen) stehen dabei.
        Ein Fehler des Workers kommt als ``{"ok": False, "error": ...}`` und wird nicht gemerkt."""
        if kind not in ("verdikt", "propose"):
            return {"ok": False, "error": "unbekannte Orakel-Anfrage %r" % (kind,)}
        key = self.key(kind, parts)
        hit = self._get(key)
        if hit is not None:
            return dict(hit, cached=True, cache_key=key[:16])
        with self._flight:
            hit = self._get(key)                    # die Anfrage vor uns hat sie vielleicht gerade gerechnet
            if hit is not None:
                return dict(hit, cached=True, cache_key=key[:16])
            with self._cache_lock:
                self.misses += 1
            body = dict(self.request_extra)
            body.update(req)
            body["what"] = kind
            res = self.request(body)
            if res.get("ok"):
                self._put(key, res)
            return dict(res, cached=False, cache_key=key[:16])
