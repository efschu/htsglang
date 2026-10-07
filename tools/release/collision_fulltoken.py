"""Item 600: for the files with a PDFLIP/WEG2 word collision, check the FULL tokens: does a mapped token
(WEG2->PDFLIP, weg2->pdflip, Weg2->PdFlip) equal a DIFFERENT token that already exists in the same file?
usage: collision_fulltoken.py <root> <file>...   (read only)"""
import re, sys, os
root = sys.argv[1]
tok = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*[A-Za-z0-9_]|[A-Za-z0-9_]")
def mp(t):
    return t.replace("WEG2", "PDFLIP").replace("Weg2", "PdFlip").replace("weg2", "pdflip")
bad_total = 0
for f in sys.argv[2:]:
    text = open(os.path.join(root, f), errors="ignore").read()
    old = set(tok.findall(text))
    hits = sorted((t, mp(t)) for t in old if mp(t) != t and mp(t) in old)
    print(("CLASH " if hits else "ok    ") + f + (("  " + str(hits[:6])) if hits else ""))
    bad_total += bool(hits)
print("files with a real full-token clash:", bad_total)
