"""Auftrag 910 / Punkt 1: die Kollisions-Ausnahme je Linie AUTOMATISCH waehlen (statt von Hand per COLLISION_OK_FILE).

usage: collision_auto.py <worktree root> <out.json>

Liest den unveraenderten Basisbaum (Verzeichnis, oder `ref:<git-ref>` ohne Checkout; REPO=<git-Repo>) und schreibt {pfad: [wort,...]} fuer genau eine Klasse: das Ticket-Label
`PDFLIP-A..X` (bzw. PdFlip/pdflip) neben den alten `WEG2-*`-Markern in derselben Datei -- das Wort, in dem
PDFLIP und WEG2 zusammenfallen. Eine Datei bekommt die Ausnahme NUR, wenn
  (a) jede Kollision der Datei diese Klasse ist (neues Wort in {PDFLIP,PdFlip,pdflip}, alte Woerter in dem
      Paar {PDFLIP,WEG2} / {PdFlip,Weg2} / {pdflip,weg2}), und
  (b) die Vollwort-Pruefung (wie collision_fulltoken.py) keinen echten Treffer hat: kein abgebildetes Token
      (WEG2->PDFLIP ...) ist gleich einem ANDEREN bereits vorhandenen Token derselben Datei.
Alles andere (flliper-Kollisionen, andere Woerter, echte Vollwort-Treffer, Ident-Map-Kollisionen) bleibt
unberuehrt: das Kit bricht dort wie bisher in Schritt 1 ab. Pfade, die schon in COLLISION_OK stehen (fest
verdrahtet oder per COLLISION_OK_FILE), gelten als ausgenommen und werden nicht doppelt geschrieben.
Ausgabe (stdout, letzte Zeile): AUTO-EXEMPT files=<n> refused=<m> (refused = Dateien mit Kollision ausserhalb der Klasse)."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rename_to_flliper as R


def iter_tree(root):
    """A worktree directory, or 'ref:<git-ref>' = the committed tree of that ref in $REPO (default /spinning/htsglang), no checkout (F0-A)."""
    if root.startswith("ref:"):
        return R.iter_ref(os.environ.get("REPO", "/spinning/htsglang"), root[4:])
    return R.iter_root(root)

PAIR = {"PDFLIP": {"PDFLIP", "WEG2"}, "PdFlip": {"PdFlip", "Weg2"}, "pdflip": {"pdflip", "weg2"}}
TOK = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*[A-Za-z0-9_]|[A-Za-z0-9_]")


def mp(t):
    return t.replace("WEG2", "PDFLIP").replace("Weg2", "PdFlip").replace("weg2", "pdflip")


def fulltoken_clash(text):
    old = set(TOK.findall(text))
    return sorted((t, mp(t)) for t in old if mp(t) != t and mp(t) in old)


def main(root, out):
    exempt, refused = {}, {}
    for path, mode, data in iter_tree(root):
        if mode == "120000":
            continue
        text = R.as_text(data)
        ext = os.path.splitext(path)[1]
        if text is None or not R.in_scope(path) or ext in R.CXX_EXT:
            continue
        py = path.endswith(".py")
        clash = R.file_collisions(text, R.rewrite_all(text, py, True, {}, path)[0])
        clash = {k: v for k, v in clash.items() if k not in R.COLLISION_OK.get(path, frozenset())}
        if not clash:
            continue
        in_class = all(k in PAIR and set(v) <= PAIR[k] for k, v in clash.items())
        if in_class and not fulltoken_clash(text):
            exempt[path] = sorted(clash)
        else:
            refused[path] = clash
    # eine von Hand gesetzte COLLISION_OK_FILE bleibt wirksam (Vereinigung), das Kit reicht nur EINE Datei weiter
    if os.environ.get("COLLISION_OK_FILE"):
        for p, ws in json.load(open(os.environ["COLLISION_OK_FILE"])).items():
            exempt[p] = sorted(set(exempt.get(p, [])) | set(ws))
    with open(out, "w") as f:
        json.dump(exempt, f, indent=1, sort_keys=True)
    for p, c in sorted(refused.items()):
        print("REFUSED", p, json.dumps(c, sort_keys=True))
    print("AUTO-EXEMPT files=%d refused=%d" % (len(exempt), len(refused)))


main(sys.argv[1], sys.argv[2])
